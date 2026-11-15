"""A2b positive support: preservation, bounded differential, contracts and risks."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from src.team_control import estimate_team_control
from src.observation_quality import analyze_observations
from src.acquisition_scenarios import acquisition_scenarios
from src.control_contribution_audit import audit_control_contributions,run_contribution_audit
from src.pipeline import infer_from_features
from src.artifacts import read_table,read_json
from test_control_acquisition import observations,set_distances


def calculate(f,weight=.5):
    q=analyze_observations(f).frames
    cfg={'acquisition':{'enabled':True,'application':'bonus','bonus_weight':weight}}
    return estimate_team_control(f,q,cfg),estimate_team_control(f,q),q


@pytest.mark.parametrize('case',['missing','bad_speed','low_quality','not_near','position','history'])
def test_unsupported_preserves_legacy_exactly(fake_bundle,case):
    f=observations(fake_bundle);f['tag_h_support']=1.
    if case=='missing':f.loc[65,'ball_ok']=False
    if case=='bad_speed':f.loc[65,'ball_speed2d_mps']=60.
    if case=='low_quality':f.loc[65,['valid_h_fraction','valid_a_fraction']]=.2
    if case=='not_near':f.loc[65,'nearest_h_m']=3.
    if case=='position':f.loc[65,'nearest_h_x']=np.nan
    index=1 if case=='history' else 65
    r,legacy,_=calculate(f)
    assert r.components.loc[index,['bonus_h','bonus_a']].eq(0).all()
    np.testing.assert_array_equal(r.scores[index],legacy.scores[index])
    assert not r.components.fallback_applied.any()
    assert r.components.effective_gate.eq(1).all()


def test_bounded_positive_bonus_and_zero_weight_ablation(fake_bundle):
    f=observations(fake_bundle);r,legacy,_=calculate(f)
    c=r.components
    np.testing.assert_allclose(c.bonus_h.iloc[5:],.5,atol=0,rtol=0)
    assert c.bonus_a.eq(0).all()
    np.testing.assert_allclose(r.scores-legacy.scores,c[['bonus_h','bonus_a']],rtol=0,atol=1e-15)
    assert c[['bonus_h','bonus_a']].ge(0).all().all() and c[['bonus_h','bonus_a']].le(.5).all().all()
    np.testing.assert_array_equal(r.transition_hint,legacy.transition_hint)
    zero,_,_=calculate(f,0.)
    np.testing.assert_array_equal(zero.scores,legacy.scores)
    np.testing.assert_array_equal(zero.components.acquisition_gate_h,c.acquisition_gate_h)
    assert c.fallback_restored_gap.eq(0).all()


def test_equal_support_does_not_add_relative_preference(fake_bundle):
    f=observations(fake_bundle);set_distances(f,np.ones(len(f)),np.ones(len(f)))
    f['tag_h_support']=.5
    r,legacy,_=calculate(f)
    assert r.components.bonus_h.iloc[5:].eq(.5).all()
    assert r.components.bonus_a.iloc[5:].eq(.5).all()
    np.testing.assert_allclose(r.scores[:,0]-r.scores[:,1],legacy.scores[:,0]-legacy.scores[:,1],atol=1e-15)


def test_disabled_and_team_symmetry(fake_bundle):
    f=observations(fake_bundle);r,legacy,q=calculate(f)
    off=estimate_team_control(f,q,{'acquisition':{'enabled':False,'application':'bonus'}})
    np.testing.assert_array_equal(off.scores,legacy.scores)
    swap=f.copy()
    for suffix in ('m','x','y'):
        swap[f'nearest_h_{suffix}']=f[f'nearest_a_{suffix}'];swap[f'nearest_a_{suffix}']=f[f'nearest_h_{suffix}']
    swap['tag_h_support']=f.tag_a_support;swap['tag_a_support']=f.tag_h_support
    swap['nearest_team_proxy']=f.nearest_team_proxy.map({'H':'A','A':'H','?':'?'})
    other,_,_=calculate(swap)
    np.testing.assert_allclose(r.scores,other.scores[:,::-1],atol=1e-15)


@pytest.mark.parametrize('extra',[{'application':'invalid'},{'bonus_weight':-1},{'bonus_weight':float('nan')},
    {'bonus_weight':float('inf')},{'bonus_weight':True},{'fallback_unavailable':True}])
def test_invalid_bonus_config_rejected(fake_bundle,extra):
    f=observations(fake_bundle);q=analyze_observations(f).frames
    with pytest.raises(ValueError):estimate_team_control(f,q,{'acquisition':{'enabled':True,'application':'bonus',**extra}})


def test_bonus_scenarios_keep_turnover_and_expose_counterexample():
    r=acquisition_scenarios({'acquisition':{'enabled':True,'application':'bonus','bonus_weight':.5}})
    assert len(r.summary)==13
    assert r.summary.passed.all(),r.summary.loc[~r.summary.passed].to_string()
    table=r.summary.set_index('scenario')
    assert table.loc['same_team_pass','checked_label_matches']==table.loc['same_team_pass','checked_label_frames']
    assert table.loc['short_interception','checked_label_matches']==table.loc['short_interception','checked_label_frames']
    bad=table.loc['misleading_comotion_tag']
    assert bad.limitation_detected and bad.checked_label_matches<bad.f1_checked_label_matches
    assert bad.expectation_kind=='known_limitation_misleading_comotion'
    challenger=r.frames.loc[r.frames.scenario.eq('temporary_challenger_comotion')]
    assert challenger.bonus_h.gt(0).any() and challenger.bonus_a.eq(0).all()


def test_contribution_audit_is_signed_and_checks_frame_keys(fake_bundle):
    f=observations(fake_bundle);r,_,_=calculate(f)
    audit=audit_control_contributions(f,r.components)
    np.testing.assert_allclose(audit.frames[['distance_gap','signed_margin_gap','persistence_gap','tag_gap']].sum(axis=1),audit.frames.legacy_gap,atol=1e-12)
    with pytest.raises(ValueError):audit_control_contributions(f,r.components.iloc[::-1])


def test_config_and_persistence(fake_bundle,tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'config/a2b.yaml').read_text())
    check=deepcopy(cfg);acq=check['control'].pop('acquisition')
    assert check==yaml.safe_load((root/'config/f1.yaml').read_text())
    assert acq['application']=='bonus' and acq['bonus_weight']==cfg['control']['persistence_weight']
    fake_bundle.frames=observations(fake_bundle)
    out=infer_from_features(fake_bundle,cfg,project_dir=tmp_path,storage_config={'root_dir':'runs','submission_filename':'synthetic.csv'})
    c=read_table(out.artifact_dir,'core/initial/control/components.csv')
    assert c.bonus_h.gt(0).any() and not c.fallback_applied.any()
    summary=read_json(out.artifact_dir,'diagnostics/final/acquisition/summary.json')
    assert summary['score_application']=='bonus' and summary['effective_zero_gate_frames']==0
    assert read_table(out.artifact_dir,'diagnostics/acquisition_synthetic/summary.csv').passed.all()
    assert len(read_table(out.artifact_dir,'diagnostics/control_contributions/component_summary.csv'))==6
    storage=tmp_path/'storage.yaml';storage.write_text('root_dir: runs\nsubmission_filename: audit.csv\n')
    audit_dir=run_contribution_audit(out.artifact_dir,project_dir=tmp_path,storage_config_path=storage)
    assert read_json(audit_dir)['entrypoint']=='control_contribution_audit'
    assert read_json(audit_dir,'inputs/reference.json')['run_id']==out.run_id
