"""Read-only audit tests: reasons, scope, identity and synthetic behavior."""
import numpy as np
import pytest
from src.control_acquisition import acquisition_support
from src.acquisition_diagnostics import _support_distances
from src.acquisition_scenarios import acquisition_scenarios
from src.artifacts import read_table,read_json
from src.pipeline import infer_from_features
from src.observation_quality import analyze_observations
from test_control_acquisition import observations,calculate


@pytest.mark.parametrize('case,expected',[
    ('missing','missing_ball'),('speed','unreliable_ball_motion'),('coverage','low_player_quality'),
    ('position','invalid_position'),('far','not_near'),('previous_missing','previous_observation_unavailable'),
    ('previous_far','previous_not_near'),('relative_speed','relative_speed_high'),
    ('support','insufficient_support')])
def test_zero_reason_taxonomy(fake_bundle,case,expected):
    f=observations(fake_bundle)
    if case=='missing':f.loc[65,'ball_ok']=False
    if case=='speed':f.loc[65,'ball_speed2d_mps']=60.
    if case=='coverage':f.loc[65,['valid_h_fraction','valid_a_fraction']]=.1
    if case=='position':f.loc[65,'nearest_h_x']=np.nan
    if case=='far':f.loc[65,'nearest_h_m']=3.
    if case=='previous_missing':f.loc[64,'ball_ok']=False
    if case=='previous_far':f.loc[64,'nearest_h_m']=3.
    if case=='relative_speed':f.loc[65,'nearest_h_x']+=.01
    if case=='support':f.loc[62,'nearest_h_x']+=.01;f.loc[63,'nearest_h_x']-=.01
    r,_=calculate(f);c=r.components
    assert c.loc[65,'acquisition_gate_h']==0
    assert c.loc[65,'primary_reason_h']==expected
    assert c.loc[65,f'fail_{expected}_h']
    unavailable=case in ['missing','speed','coverage','position','previous_missing']
    assert c.loc[65,'evidence_status_h']==('unavailable' if unavailable else 'observed_not_supporting_proxy')


def test_primary_reason_does_not_hide_overlapping_flags_and_history(fake_bundle):
    f=observations(fake_bundle);f.loc[65,'ball_ok']=False
    r,_=calculate(f);c=r.components
    assert c.loc[65,'fail_missing_ball_h'] and c.loc[65,'fail_unreliable_ball_motion_h']
    assert c.loc[65,'primary_reason_h']=='missing_ball'
    assert c.loc[1,'primary_reason_h']=='insufficient_history'
    assert c.loc[0,'primary_reason_h']=='block_start'
    assert c.loc[20,'primary_reason_h']=='supported'
    disabled,_=calculate(f,False)
    assert disabled.components.primary_reason_h.eq('disabled').all()


def test_support_distance_never_crosses_half():
    ids=np.arange(10);half=np.r_[np.ones(5,int),np.full(5,2)]
    supported=np.zeros(10,bool);supported[[2,7]]=True
    previous,following=_support_distances(ids,half,supported)
    assert np.isnan(previous[0]) and following[0]==2
    assert previous[2]==following[2]==0
    assert previous[4]==2 and np.isnan(following[4])
    assert np.isnan(previous[5]) and following[5]==2


def test_synthetic_suite_captures_continuation_without_acquisition():
    r=acquisition_scenarios({'acquisition':{'enabled':True}})
    assert len(r.summary)==7 and r.summary.passed.all()
    table=r.summary.set_index('scenario')
    assert table.loc['same_team_pass','challenge_zero_gate_fraction']>.5
    assert table.loc['nearest_player_switch','challenge_zero_gate_fraction']>.8
    assert table.loc['contested_ball','challenge_both_supported_fraction']>.8
    assert table.loc['occlusion','challenge_unavailable_fraction']==1
    assert table.loc['short_interception','checked_label_matches']==table.loc['short_interception','checked_label_frames']


def test_pipeline_saves_audit_without_changing_scores(fake_bundle,tmp_path):
    f=observations(fake_bundle);f[['nearest_h_m','nearest_a_m']]=6.
    f['nearest_team_proxy']='?';fake_bundle.frames=f
    result=infer_from_features(fake_bundle,{'control':{'acquisition':{'enabled':True}},'fusion':{'control_normalization':'max'}},
        project_dir=tmp_path,storage_config={'root_dir':'runs','submission_filename':'synthetic.csv'})
    root=result.artifact_dir
    summary=read_json(root,'diagnostics/final/acquisition/summary.json')
    assert summary['zero_gate_frames']==len(f)
    assert summary['zero_gate_already_zero_gap_frames']==len(f)
    assert summary['zero_gate_removed_nonzero_gap_frames']==0
    c=read_table(root,'core/initial/control/components.csv')
    np.testing.assert_array_equal(c.legacy_h_score,c.adjusted_h_score)
    np.testing.assert_array_equal(c.legacy_a_score,c.adjusted_a_score)
    audit=read_table(root,'diagnostics/final/acquisition/frames.csv')
    assert len(audit)==len(f) and audit.frame.is_unique
    assert read_table(root,'diagnostics/acquisition_synthetic/summary.csv').passed.all()
