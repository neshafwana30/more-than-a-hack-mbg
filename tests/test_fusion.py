"""Synthetic, label-free-of-match tests for the isolated F1 fusion change."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from src.evidence_fusion import fuse_evidence
from src.temporal_decoder import decode_states
from src.pipeline import infer_from_features
from src.artifacts import read_json, read_table


def fused(control, live, dead, cfg=None):
    control=np.asarray(control,dtype=float);n=len(control)
    live=np.broadcast_to(np.asarray(live,dtype=float),(n,)).copy()
    dead=np.broadcast_to(np.asarray(dead,dtype=float),(n,)).copy()
    return fuse_evidence(pd.DataFrame({'frame':np.arange(n)}),
        SimpleNamespace(frames=pd.DataFrame(index=np.arange(n))),
        SimpleNamespace(scores=control,transition_hint=np.zeros(n)),
        SimpleNamespace(live_score=live,dead_score=dead),
        SimpleNamespace(frames=pd.DataFrame({'stoppage_anchor':np.zeros(n),'restart_anchor':np.zeros(n)})),cfg)


def decoded(result):
    n=len(result.state_log_scores)
    return decode_states(result.state_log_scores,np.ones(n,int),result.transition_hint,
                         result.stop_hint,result.restart_hint).labels


def test_default_fusion_preserves_legacy_and_does_not_mutate_control():
    control=np.array([[1.,.2],[.04,.04],[-.2,.8]])
    original=control.copy()
    result=fused(control,[.55,.7,.6],[.25,.8,.2])
    expected=np.array([[1.55,.75,.25],[.74,.74,.8],[.4,1.4,.2]])
    np.testing.assert_allclose(result.state_log_scores,expected,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(result.state_log_scores,fused(control,[.55,.7,.6],[.25,.8,.2],{'control_normalization':'none'}).state_log_scores)
    np.testing.assert_array_equal(control,original)
    assert result.metadata['control_normalization']=='none'
    assert (result.control_components.common_offset==0).all()


@pytest.mark.parametrize('weight',[0.,.3,2.])
def test_max_centering_preserves_gap_live_reference_and_dead(weight):
    control=np.array([[2.,.3],[.04,.04],[-.1,.9]])
    live=np.array([.6,.55,.8]);dead=np.array([.3,.25,.9])
    cfg={'control_normalization':'max','team_weight':weight,'live_weight':.8,'dead_weight':1.7,'dead_bias':.2}
    result=fused(control,live,dead,cfg);scores=result.state_log_scores
    np.testing.assert_allclose(scores[:,:2].max(axis=1),.8*live,rtol=0,atol=1e-14)
    np.testing.assert_allclose(scores[:,0]-scores[:,1],weight*(control[:,0]-control[:,1]),rtol=0,atol=1e-14)
    np.testing.assert_allclose(scores[:,2],1.7*dead+.2)
    np.testing.assert_allclose(result.control_components[['relative_control_H','relative_control_A']].max(axis=1),0)


def test_common_control_offset_invariance():
    control=np.array([[2.,.3],[.04,.04],[-.1,.9]])
    offset=np.array([-10.,3.,20.])[:,None]
    cfg={'control_normalization':'max','team_weight':1.5}
    a=fused(control,.6,.25,cfg);b=fused(control+offset,.6,.25,cfg)
    np.testing.assert_allclose(a.state_log_scores,b.state_log_scores,rtol=0,atol=1e-13)
    np.testing.assert_allclose(b.control_components.common_offset-a.control_components.common_offset,1.5*offset[:,0])


def test_h_a_decoding_unchanged_when_dead_is_unavailable():
    n=300;control=np.tile([2.,.1],(n,1));control[150:]=[.1,2.]
    control[75]=[.1,2.]
    old=fused(control,.8,-100.)
    f1=fused(control,.8,-100.,{'control_normalization':'max'})
    np.testing.assert_array_equal(decoded(old),decoded(f1))
    assert decoded(f1)[75]=='H'
    assert (decoded(f1)[160:]=='A').all()


def test_genuine_brief_turnover_survives():
    control=np.tile([3.,.1],(180,1));control[80:88]=[.1,3.]
    labels=decoded(fused(control,.8,-100.,{'control_normalization':'max'}))
    assert (labels[80:88]=='A').all()
    assert (labels[:80]=='H').all() and (labels[88:]=='H').all()


def test_neutral_control_or_team_ambiguity_does_not_imply_dead():
    for level in (.04,10.):
        result=fused(np.full((80,2),level),.55,.25,{'control_normalization':'max'})
        assert (decoded(result)!='D').all()
        np.testing.assert_allclose(result.state_log_scores[:,:2],.55)


def test_stoppage_and_restart_with_nearby_player():
    n=300;control=np.tile([5.,1.],(n,1))
    live=np.full(n,.8);dead=np.full(n,.15)
    live[100:175]=.1;dead[100:175]=1.
    old=decoded(fused(control,live,dead))
    f1=decoded(fused(control,live,dead,{'control_normalization':'max'}))
    assert (old=='H').all()
    assert (f1[100:175]=='D').all()
    assert (f1[:100]=='H').all() and (f1[175:]=='H').all()


def test_isolated_dead_pulse_is_not_forced_into_a_stoppage():
    control=np.tile([5.,1.],(100,1));dead=np.full(100,.2);dead[50]=.9
    labels=decoded(fused(control,.5,dead,{'control_normalization':'max'}))
    assert (labels=='H').all()


@pytest.mark.parametrize('mode',['none','max'])
def test_forced_interval_preserved(mode):
    n=113750
    result=fused(np.tile([.8,.2],(n,1)),.7,.25,{'control_normalization':mode})
    np.testing.assert_array_equal(result.state_log_scores[110808:113742],np.tile([-1e6,-1e6,1e6],(2934,1)))
    assert result.control_components.forced_d.sum()==2934
    assert not result.control_components.forced_d.iloc[113742]


def test_unknown_normalization_rejected():
    with pytest.raises(ValueError,match='control_normalization'):
        fused([[1.,.2]],.6,.25,{'control_normalization':'unsupported'})


def test_f1_pipeline_snapshots_config_channels_and_missing_ball_behavior(fake_bundle,tmp_path):
    result=infer_from_features(fake_bundle,{'fusion':{'control_normalization':'max'}},
                               storage_config={'root_dir':str(tmp_path)})
    root=Path(result.artifact_dir)
    assert read_json(root,'config/model.json')['fusion']['control_normalization']=='max'
    assert read_json(root,'core/initial/fusion/metadata.json')['control_normalization']=='max'
    components=read_table(root,'core/initial/fusion/control_components.csv')
    scores=read_table(root,'core/initial/fusion/state_log_scores.csv')
    np.testing.assert_allclose(scores[['H','A']].max(axis=1),components.live_utility)
    assert (result.labels[50:80]!='D').all()


def test_f1_config_is_exactly_v2_plus_normalization():
    root=Path(__file__).resolve().parents[1]
    base=yaml.safe_load((root/'config/v2.yaml').read_text(encoding='utf-8-sig'))
    candidate=yaml.safe_load((root/'config/f1.yaml').read_text(encoding='utf-8-sig'))
    assert candidate['fusion'].pop('control_normalization')=='max'
    assert candidate==base
