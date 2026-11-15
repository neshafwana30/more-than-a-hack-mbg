"""A2a tests: exact unavailable fallback, preserved A1 elsewhere, synthetic risks."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from src.team_control import estimate_team_control
from src.observation_quality import analyze_observations
from src.acquisition_scenarios import acquisition_scenarios
from src.pipeline import infer_from_features
from src.artifacts import read_json,read_table
from test_control_acquisition import observations


def compare(f):
    q=analyze_observations(f).frames
    a1=estimate_team_control(f,q,{'acquisition':{'enabled':True}})
    a2=estimate_team_control(f,q,{'acquisition':{'enabled':True,'fallback_unavailable':True}})
    legacy=estimate_team_control(f,q)
    return a1,a2,legacy


@pytest.mark.parametrize('case',['missing','bad_speed','low_quality','bad_position','previous_missing','start'])
def test_unavailable_restores_exact_quality_weighted_legacy(fake_bundle,case):
    f=observations(fake_bundle);f['tag_h_support']=1.
    if case=='missing':f.loc[60:79,'ball_ok']=False;f.loc[60:79,['ball_x','ball_y','nearest_h_m','nearest_a_m']]=np.nan
    if case=='bad_speed':f.loc[65,'ball_speed2d_mps']=60.
    if case=='low_quality':f.loc[65,['valid_h_fraction','valid_a_fraction']]=.1
    if case=='bad_position':f.loc[65,'nearest_h_x']=np.nan
    if case=='previous_missing':f.loc[64,'ball_ok']=False
    a1,a2,old=compare(f);c=a2.components;fallback=c.fallback_applied.to_numpy(bool)
    assert fallback.any()
    expected=c.winner_evidence_status.eq('unavailable').to_numpy()
    np.testing.assert_array_equal(fallback,expected)
    np.testing.assert_array_equal(a2.scores[fallback],old.scores[fallback])
    np.testing.assert_array_equal(a2.scores[~fallback],a1.scores[~fallback])
    np.testing.assert_array_equal(a2.transition_hint,a1.transition_hint)
    np.testing.assert_array_equal(c[['acquisition_gate_h','acquisition_gate_a','winner_gate']],a1.components[['acquisition_gate_h','acquisition_gate_a','winner_gate']])
    assert c.loc[fallback,'effective_gate'].eq(1).all()
    assert c.loc[fallback,'winner_gate'].eq(0).all()
    if case=='missing':
        # Ball-geometry terms are already zero; only the existing tag gap remains.
        np.testing.assert_allclose(a2.scores[65],[.34,.04],rtol=0,atol=1e-15)


@pytest.mark.parametrize('case',['supported','far','relative_speed','short_support','outside'])
def test_other_statuses_remain_exact_a1(fake_bundle,case):
    f=observations(fake_bundle)
    if case=='far':f.loc[65,'nearest_h_m']=3.
    if case=='relative_speed':f.loc[65,'nearest_h_x']+=.01
    if case=='short_support':f.loc[62,'nearest_h_x']+=.01;f.loc[63,'nearest_h_x']-=.01
    if case=='outside':f.loc[65:70,'half']=0
    a1,a2,_=compare(f);c=a2.components
    assert not c.loc[65,'fallback_applied']
    np.testing.assert_array_equal(a1.scores[65],a2.scores[65])


def test_explicit_disable_and_acquisition_disable_preserve_behavior(fake_bundle):
    f=observations(fake_bundle);q=analyze_observations(f).frames
    implicit=estimate_team_control(f,q,{'acquisition':{'enabled':True}})
    explicit=estimate_team_control(f,q,{'acquisition':{'enabled':True,'fallback_unavailable':False}})
    np.testing.assert_array_equal(implicit.scores,explicit.scores)
    off=estimate_team_control(f,q,{'acquisition':{'enabled':False,'fallback_unavailable':True}})
    legacy=estimate_team_control(f,q)
    np.testing.assert_array_equal(off.scores,legacy.scores)
    assert not off.components.fallback_applied.any()
    with pytest.raises(ValueError):estimate_team_control(f,q,{'acquisition':{'fallback_unavailable':'yes'}})


def test_ties_and_reason_precedence_are_explicit(fake_bundle):
    f=observations(fake_bundle);f['nearest_h_m']=f.nearest_a_m
    f['nearest_team_proxy']='?';f['tag_h_support']=0.;f['tag_a_support']=0.
    f.loc[65,'ball_ok']=False
    _,a2,old=compare(f)
    assert a2.components.loc[65,'winner_tie'] and a2.components.loc[65,'fallback_applied']
    np.testing.assert_array_equal(a2.scores[65],old.scores[65])
    f=observations(fake_bundle);f.loc[65,['valid_h_fraction','valid_a_fraction']]=.1;f.loc[65,'nearest_h_m']=3.
    _,a2,_=compare(f)
    assert a2.components.loc[65,'winner_primary_reason']=='low_player_quality'
    assert a2.components.loc[65,'fallback_applied']


def test_fallback_scenarios_expose_stale_tag_limitation():
    r=acquisition_scenarios({'acquisition':{'enabled':True,'fallback_unavailable':True}})
    assert len(r.summary)==11 and r.summary.passed.all()
    table=r.summary.set_index('scenario')
    for name in ['occlusion_same_team_tag','occlusion_turnover_tag','low_quality_tag']:
        assert table.loc[name,'challenge_fallback_fraction']==1
        assert table.loc[name,'checked_label_matches']==table.loc[name,'checked_label_frames']
    stale=table.loc['occlusion_turnover_stale_tag']
    assert stale.limitation_detected
    assert stale.checked_label_matches<stale.checked_label_frames
    assert stale.expectation_kind=='known_limitation_stale_tag'
    gap=r.frames.loc[r.frames.scenario.eq('occlusion_turnover_tag')&r.frames.frame.between(87,93)]
    assert gap.decoded_label.eq('A').all()


def test_config_isolation_and_fallback_persistence(fake_bundle,tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'config/a2a.yaml').read_text())
    check=deepcopy(cfg);assert check['control']['acquisition'].pop('fallback_unavailable')
    assert check==yaml.safe_load((root/'config/a1.yaml').read_text())
    f=observations(fake_bundle);f['tag_h_support']=1.
    f.loc[60:79,'ball_ok']=False;f.loc[60:79,['ball_x','ball_y','nearest_h_m','nearest_a_m']]=np.nan
    fake_bundle.frames=f
    out=infer_from_features(fake_bundle,cfg,project_dir=tmp_path,storage_config={'root_dir':'runs','submission_filename':'synthetic.csv'})
    c=read_table(out.artifact_dir,'core/initial/control/components.csv')
    assert c.loc[65,'fallback_applied'] and c.loc[65,'effective_gate']==1
    summary=read_json(out.artifact_dir,'diagnostics/final/acquisition/summary.json')
    assert summary['fallback_frames']>0 and summary['fallback_restored_gap_sum']>0
    assert summary['effective_zero_gate_frames']<summary['zero_gate_frames']
    assert read_table(out.artifact_dir,'diagnostics/acquisition_synthetic/summary.csv').passed.all()
