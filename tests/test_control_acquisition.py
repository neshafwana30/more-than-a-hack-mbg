"""Synthetic A1 tests; no competition labels or manually selected match windows."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from src.team_control import estimate_team_control
from src.control_acquisition import acquisition_support
from src.observation_quality import analyze_observations
from src.evidence_fusion import fuse_evidence
from src.live_dead import estimate_live_dead
from src.temporal_decoder import decode_states
from src.pipeline import infer_from_features
from src.artifacts import read_table,read_json


def observations(bundle):
    f=bundle.frames.copy();n=len(f)
    f['ball_ok']=True;f['ball_x']=.3+np.arange(n)*8/(25*105.3);f['ball_y']=.5
    f['ball_speed2d_mps']=8.;f['ball_displacement_m']=8/25
    f['ball_missing_run_frames']=0;f['valid_h_fraction']=1.;f['valid_a_fraction']=1.
    f['tag_h_support']=0.;f['tag_a_support']=0.
    set_distances(f,np.full(n,.6),np.full(n,6.))
    return f


def set_distances(f,h,a):
    f['nearest_h_m']=h;f['nearest_a_m']=a
    for side in ('h','a'):
        f[f'nearest_{side}_x']=f.ball_x-f[f'nearest_{side}_m']/105.3
        f[f'nearest_{side}_y']=f.ball_y
    f['nearest_team_proxy']=np.where(f.nearest_h_m<f.nearest_a_m,'H',np.where(f.nearest_a_m<f.nearest_h_m,'A','?'))


def calculate(f,enabled=True,**kwargs):
    q=analyze_observations(f)
    return estimate_team_control(f,q.frames,{'acquisition':{'enabled':enabled,**kwargs}}),q


def decode(f,control,q):
    from types import SimpleNamespace
    ev=SimpleNamespace(frames=pd.DataFrame({'stoppage_anchor':np.zeros(len(f)), 'restart_anchor':np.zeros(len(f)),
                                           'out_of_pitch_hint':np.zeros(len(f))}))
    live=estimate_live_dead(f,q.frames,ev.frames)
    fusion=fuse_evidence(f,q,control,live,ev,{'control_normalization':'max'})
    return decode_states(fusion.state_log_scores,f.half.to_numpy(),fusion.transition_hint,fusion.stop_hint,fusion.restart_hint).labels


def test_disabled_exact_legacy_formula_and_hints(fake_bundle):
    f=observations(fake_bundle)
    a,q=calculate(f,False);b=estimate_team_control(f,q.frames)
    np.testing.assert_array_equal(a.scores,b.scores)
    np.testing.assert_array_equal(a.transition_hint,b.transition_hint)
    # Independently known steady distance contributions after centered persistence settles.
    expected_h=np.exp(-.6/2.5)+.5+.25*np.tanh((6-.6)/5)+.04
    expected_a=np.exp(-6/2.5)-.25*np.tanh((6-.6)/5)+.04
    np.testing.assert_allclose(a.scores[20],[expected_h,expected_a],rtol=0,atol=1e-15)
    assert (a.components.winner_gate==1).all()
    for side in ('h','a'):
        reconstructed=a.components[[f'distance_{side}',f'tag_{side}',f'persistence_{side}',f'signed_margin_{side}','symmetric_prior']].sum(axis=1)
        np.testing.assert_allclose(reconstructed,a.components[f'legacy_{side}_score'],rtol=0,atol=1e-15)


def test_stable_comotion_preserves_control_and_decoder(fake_bundle):
    f=observations(fake_bundle);r,q=calculate(f)
    assert (r.components.acquisition_gate_h.iloc[5:]==1).all()
    assert (r.components.acquisition_gate_a==0).all()
    np.testing.assert_allclose(r.components.relative_speed_h_mps.iloc[1:],0,atol=1e-12)
    assert (decode(f,r,q)=='H').all()


def test_close_fast_flyby_is_not_acquisition(fake_bundle):
    f=observations(fake_bundle)
    set_distances(f,np.full(len(f),6.),np.full(len(f),.6))
    # Every frame's nearest A can be a DIFFERENT stationary player: its offset
    # changes 0.16 m/frame. Nearest distance remains <2.5 m, ball speed is reliable.
    f['nearest_a_x']=f.ball_x-(.6+.16*(np.arange(len(f))%8))/105.3
    r,_=calculate(f)
    assert not r.components.acquisition_cue_a.any()
    assert (r.components.acquisition_gate_a==0).all()
    np.testing.assert_allclose(r.scores[:,0],r.scores[:,1],atol=1e-15)


def test_contested_ball_can_support_both_without_inventing_winner(fake_bundle):
    f=observations(fake_bundle);set_distances(f,np.ones(len(f)),np.ones(len(f)))
    r,_=calculate(f)
    assert (r.components[['acquisition_gate_h','acquisition_gate_a']].iloc[5:]==1).all().all()
    np.testing.assert_array_equal(r.scores[:,0],r.scores[:,1])


@pytest.mark.parametrize('case',['missing','bad_speed','low_coverage','invalid_position'])
def test_invalid_observations_release_support_immediately(fake_bundle,case):
    f=observations(fake_bundle)
    if case=='missing':f.loc[60,'ball_ok']=False
    if case=='bad_speed':f.loc[60,'ball_speed2d_mps']=60.
    if case=='low_coverage':f.loc[60,['valid_h_fraction','valid_a_fraction']]=.2
    if case=='invalid_position':f.loc[60,'nearest_h_x']=np.nan
    r,_=calculate(f)
    assert r.components.acquisition_gate_h.iloc[59]==1
    assert (r.components.acquisition_gate_h.iloc[60:62]==0).all()


def test_half_and_frame_gap_reset_history(fake_bundle):
    f=observations(fake_bundle);f.loc[80:,'half']=2
    r,_=calculate(f);assert (r.components.acquisition_gate_h.iloc[80:84]==0).all()
    f=observations(fake_bundle);f.loc[80:,'frame']+=10
    r,_=calculate(f);assert (r.components.acquisition_gate_h.iloc[80:84]==0).all()


def test_acquisition_is_causal_and_uses_fps_pitch_units(fake_bundle):
    f=observations(fake_bundle);q=analyze_observations(f).frames
    a,_,_=acquisition_support(f,q,{'enabled':True})
    b,_,_=acquisition_support(f.iloc[:71],q.iloc[:71],{'enabled':True})
    np.testing.assert_array_equal(a[:71],b)
    f['nearest_h_x']=f.ball_x-(.6+.16*(np.arange(len(f))%8))/105.3
    normal,_,_=acquisition_support(f,q,{'enabled':True})
    slower,_,_=acquisition_support(f,q,{'enabled':True},fps=12.5)
    smaller,_,_=acquisition_support(f,q,{'enabled':True},pitch=(52.65,68.))
    assert normal[:,0].sum()==0 and slower[:,0].sum()>0 and smaller[:,0].sum()>0


def test_brief_proximity_does_not_delete_genuine_short_turnover(fake_bundle):
    f=observations(fake_bundle);n=len(f)
    h=np.full(n,.6);a=np.full(n,6.)
    # One-frame challenger, followed later by a genuine 0.8 s control spell.
    h[40]=6.;a[40]=.6;h[80:100]=6.;a[80:100]=.6
    set_distances(f,h,a)
    r,q=calculate(f);labels=decode(f,r,q)
    assert labels[40]=='H'
    assert (labels[86:98]=='A').all()
    assert labels[70]=='H' and labels[110]=='H'


def test_symmetry_gap_contraction_and_legacy_hint_invariance(fake_bundle):
    f=observations(fake_bundle);f.loc[60:70,'nearest_h_x']+=.001*np.arange(11)
    new,q=calculate(f);old,_=calculate(f,False)
    np.testing.assert_array_equal(new.transition_hint,old.transition_hint)
    np.testing.assert_allclose(new.scores.mean(axis=1),old.scores.mean(axis=1),atol=1e-15)
    np.testing.assert_allclose(new.scores[:,0]-new.scores[:,1],(old.scores[:,0]-old.scores[:,1])*new.components.winner_gate,atol=1e-15)
    swap=f.copy()
    for suffix in ('m','x','y'):
        swap[f'nearest_h_{suffix}']=f[f'nearest_a_{suffix}'];swap[f'nearest_a_{suffix}']=f[f'nearest_h_{suffix}']
    swap['tag_h_support']=f.tag_a_support;swap['tag_a_support']=f.tag_h_support
    swap['nearest_team_proxy']=f.nearest_team_proxy.map({'H':'A','A':'H','?':'?'})
    other,_=calculate(swap)
    np.testing.assert_allclose(new.scores,other.scores[:,::-1],atol=1e-15)


@pytest.mark.parametrize('setting',[{'window_frames':1},{'window_frames':True},{'minimum_support_fraction':0},
    {'maximum_relative_speed_mps':-1},{'maximum_distance_m':0},{'minimum_player_quality':1.1},{'enabled':'yes'},{'extra':1}])
def test_invalid_config_rejected(fake_bundle,setting):
    f=observations(fake_bundle)
    with pytest.raises(ValueError):calculate(f,**setting)


def test_artifacts_and_config_isolation(fake_bundle,tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'config/a1.yaml').read_text())
    stripped=deepcopy(cfg);assert stripped['control'].pop('acquisition')['enabled']
    assert stripped==yaml.safe_load((root/'config/f1.yaml').read_text())
    f=observations(fake_bundle);fake_bundle.frames=f
    storage=tmp_path/'storage.yaml';storage.write_text('root_dir: runs\nsubmission_filename: synthetic.csv\n')
    result=infer_from_features(fake_bundle,cfg,project_dir=tmp_path,storage_config_path=storage)
    folder=Path(result.artifact_dir)
    c=read_table(folder,'core/initial/control/components.csv')
    assert c.acquisition_gate_h.iloc[-1]==1
    assert read_json(folder,'core/initial/control/metadata.json')['fps']==cfg['fps']
    assert read_json(folder,'core/initial/control/metadata.json')['pitch_m']==cfg['pitch']
    assert read_json(folder,'config/model.json')['control']['acquisition']['enabled']
