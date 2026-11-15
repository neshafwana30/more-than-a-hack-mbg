"""Synthetic tests for D1 joint cues and exact score-channel instrumentation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from src.live_dead import estimate_live_dead
from src.observation_quality import analyze_observations
from src.evidence_fusion import fuse_evidence
from src.temporal_decoder import decode_states
from src.pipeline import infer_from_features
from src.artifacts import read_json,read_table


def setup_observations(bundle):
    f=bundle.frames.copy();n=len(f)
    f['ball_ok']=True;f['ball_x']=.4+np.arange(n)*.2/(25*105.3);f['ball_y']=.5
    f['ball_speed2d_mps']=.2;f['ball_displacement_m']=.2/25;f['ball_missing_run_frames']=0
    f['player_motion_robust_h']=1.;f['player_motion_robust_a']=1.
    f['centroid_valid_h']=True;f['centroid_valid_a']=True
    f['valid_h_fraction']=1.;f['valid_a_fraction']=1.
    f['set_piece_tag']=True
    ev=pd.DataFrame({'frame':f.frame,'stoppage_anchor':np.zeros(n),'restart_anchor':np.zeros(n),
                     'out_of_pitch_hint':np.zeros(n),'is_set_piece_context':np.ones(n,bool)})
    return f,ev


def calculate(f,ev,enabled=True,extra=None):
    cfg={'context_stoppage':{'enabled':enabled,**(extra or {})}}
    q=analyze_observations(f)
    return estimate_live_dead(f,q.frames,ev,cfg),q


def test_legacy_scores_and_channel_sums_are_preserved(fake_bundle):
    f,ev=setup_observations(fake_bundle)
    f.loc[0:3,'ball_speed2d_mps']=[0.,5.,np.nan,.2]
    f.loc[0:3,['player_motion_robust_h','player_motion_robust_a']]=np.array([[0,0],[3,3],[2,2],[1,1]])
    f.loc[2,'ball_ok']=False
    ev.loc[1,['out_of_pitch_hint','stoppage_anchor','restart_anchor']]=[1.,1.,.5]
    ev.loc[2,['out_of_pitch_hint','stoppage_anchor']]=[1.,.5]
    ev.loc[3,'restart_anchor']=1.
    q=analyze_observations(f)
    default=estimate_live_dead(f,q.frames,ev)
    disabled,_=calculate(f,ev,False)
    np.testing.assert_allclose(default.live_score[:4],[.55,.89,.63,.5988],rtol=0,atol=1e-15)
    np.testing.assert_allclose(default.dead_score[:4],[.25,.99,.41,.51],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(default.live_score,disabled.live_score)
    np.testing.assert_array_equal(default.dead_score,disabled.dead_score)
    c=default.components
    np.testing.assert_allclose(c[['live_prior','ball_motion_live','player_motion_live']].sum(axis=1),default.live_score)
    np.testing.assert_allclose(c[['dead_prior','out_of_pitch_dead','stoppage_anchor_dead','restart_anchor_dead',
                                 'stationary_near_anchor_dead','context_stoppage_dead']].sum(axis=1),default.dead_score)
    np.testing.assert_array_equal(c.dead_without_context,default.dead_score)
    assert not c.context_active.any()


def test_joint_preparation_has_delayed_activation_and_bounded_bonus(fake_bundle):
    f,ev=setup_observations(fake_bundle)
    result,_=calculate(f,ev)
    assert not result.components.context_active.iloc[:9].any()
    assert result.components.context_active.iloc[9:].all()
    assert (result.components.context_stoppage_dead.iloc[9:]==.55).all()
    np.testing.assert_allclose(result.dead_score[9:],.8)
    np.testing.assert_allclose(result.dead_score-result.components.dead_without_context,result.components.context_stoppage_dead,rtol=0,atol=1e-15)
    assert (result.dead_score[9:]>result.live_score[9:]).all()


@pytest.mark.parametrize('case',['no_context','missing_ball','unreliable_ball','no_player_motion','one_team_missing','low_player_quality','players_moving'])
def test_absence_or_single_weak_cue_cannot_activate_d1(fake_bundle,case):
    f,ev=setup_observations(fake_bundle)
    if case=='no_context':ev['is_set_piece_context']=False
    elif case=='missing_ball':f['ball_ok']=False;f['ball_speed2d_mps']=np.nan
    elif case=='unreliable_ball':f['ball_speed2d_mps']=60.
    elif case=='no_player_motion':f[['centroid_valid_h','centroid_valid_a']]=False
    elif case=='one_team_missing':f['centroid_valid_a']=False
    elif case=='low_player_quality':f[['valid_h_fraction','valid_a_fraction']]=.3
    elif case=='players_moving':f[['player_motion_robust_h','player_motion_robust_a']]=3.
    result,_=calculate(f,ev)
    assert not result.components.context_active.any()
    assert (result.components.context_stoppage_dead==0).all()


def test_brief_stationary_cue_is_not_a_full_tag_dead_label(fake_bundle):
    f,ev=setup_observations(fake_bundle)
    f['ball_speed2d_mps']=8.;f.loc[30:36,'ball_speed2d_mps']=.2
    result,_=calculate(f,ev)
    assert not result.components.context_active.any()


def test_missing_frame_releases_contribution_and_history_does_not_cross_half(fake_bundle):
    f,ev=setup_observations(fake_bundle)
    f.loc[50:79,'ball_ok']=False;f.loc[50:80,'ball_speed2d_mps']=np.nan
    result,_=calculate(f,ev)
    assert result.components.context_active.iloc[49]
    assert (result.components.context_stoppage_dead.iloc[50:81]==0).all()
    assert not result.components.context_active.iloc[81:88].any()
    f,ev=setup_observations(fake_bundle)
    f.loc[len(f)-6:,'half']=2
    result,_=calculate(f,ev)
    assert result.components.context_active.iloc[-7]
    assert not result.components.context_active.iloc[-6:].any()


def test_restart_motion_releases_even_when_tag_and_outside_remain(fake_bundle):
    f,ev=setup_observations(fake_bundle);n=len(f)
    f.loc[100:,'ball_speed2d_mps']=8.;ev['out_of_pitch_hint']=1.
    result,q=calculate(f,ev)
    assert result.components.context_active.iloc[99]
    assert not result.components.context_active.iloc[100:].any()
    assert (result.components.context_stoppage_dead.iloc[100:]==0).all()
    assert result.live_score[110]>result.dead_score[110]
    control=SimpleNamespace(scores=np.tile([5.,1.],(n,1)),transition_hint=np.zeros(n))
    events=SimpleNamespace(frames=ev)
    fusion=fuse_evidence(f,q,control,result,events,{'control_normalization':'max'})
    labels=decode_states(fusion.state_log_scores,f.half.to_numpy(),fusion.transition_hint,fusion.stop_hint,fusion.restart_hint).labels
    assert (labels[30:90]=='D').all()
    assert (labels[120:]=='H').all()


@pytest.mark.parametrize('setting',[{'window_frames':0},{'window_frames':1.5},{'minimum_support_fraction':0.},
    {'minimum_support_fraction':1.1},{'minimum_player_quality':-1.},{'weight':-1.},{'enabled':'false'}])
def test_context_settings_rejected(fake_bundle,setting):
    f,ev=setup_observations(fake_bundle)
    with pytest.raises(ValueError):calculate(f,ev,extra=setting)


def test_pipeline_serializes_d1_channels_and_configuration(fake_bundle,tmp_path):
    f,ev=setup_observations(fake_bundle)
    f.loc[120:,'ball_speed2d_mps']=8.
    fake_bundle.frames=f
    result=infer_from_features(fake_bundle,{'fusion':{'control_normalization':'max'},
        'live_dead':{'context_stoppage':{'enabled':True}}},storage_config={'root_dir':str(tmp_path)})
    root=Path(result.artifact_dir)
    c=read_table(root,'core/initial/live_dead/components.csv')
    assert c.context_active.any()
    assert (c.context_stoppage_dead.iloc[120:]==0).all()
    assert read_json(root,'core/initial/live_dead/metadata.json')['context_stoppage']['window_frames']==10
    assert read_json(root,'config/model.json')['live_dead']['context_stoppage']['enabled'] is True
    assert 'D' in result.labels[20:100]
    assert 'D' not in result.labels[140:]


def test_d1_config_only_adds_context_evidence_to_f1():
    root=Path(__file__).resolve().parents[1]
    a=yaml.safe_load((root/'config/f1.yaml').read_text(encoding='utf-8-sig'))
    b=yaml.safe_load((root/'config/d1.yaml').read_text(encoding='utf-8-sig'))
    assert b['live_dead'].pop('context_stoppage')['enabled'] is True
    assert b==a
