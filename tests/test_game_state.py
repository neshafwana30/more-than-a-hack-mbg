"""Synthetic D2, hierarchical isolation, carry and replay contracts."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from src.game_state_scenarios import scenario_inputs,game_state_scenarios
from src.game_state_evidence import DEFAULTS,estimate_signed_game_state
from src.observation_quality import analyze_observations
from src.live_dead import estimate_live_dead
from src.temporal_decoder import decode_states
from src.boundary_refinement import refine_boundaries
from src.pipeline import infer_from_features,run_pipeline
from src.artifacts import read_table,read_json
from src.feature_replay import load_feature_replay


def score(name,**extra):
    f,e=scenario_inputs(name);q=analyze_observations(f)
    return f,e,q,estimate_signed_game_state(f,q.frames,e,extra)


def test_signed_contracts_both_decoders():
    r=game_state_scenarios(DEFAULTS)
    assert len(r.summary)==9 and r.summary.passed.all(),r.summary.to_string()
    carry=game_state_scenarios({**DEFAULTS,'carry_enabled':True})
    assert carry.summary.passed.all(),carry.summary.to_string()


def test_components_reconstruct_and_missing_not_zero_motion():
    f,e,q,r=score('placed_stoppage');c=r.components
    np.testing.assert_allclose(c[r.metadata['signed_additive_columns']].sum(axis=1),r.dead_score-r.live_score,atol=1e-12)
    np.testing.assert_allclose(r.dead_score+r.live_score,0,atol=0)
    _,_,_,missing=score('missing_all')
    assert missing.components.pace_dead.eq(0).all() and missing.components.still_dead.eq(0).all()
    assert missing.components.smoothed_player_motion.isna().all()
    assert missing.components.dead_minus_live.eq(DEFAULTS['prior_delta']).all()


def test_carry_bounded_reset_and_not_missing_alone():
    _,_,_,base=score('outside_missing');f,e,q,carry=score('outside_missing',carry_enabled=True)
    assert base.components.carry_dead.eq(0).all()
    assert carry.components.outside_carry_support.iloc[100]>0
    assert carry.components.outside_carry_support.iloc[174]>0
    assert carry.components.outside_carry_support.iloc[175]==0
    assert carry.components.outside_carry_support.iloc[200]==0
    assert np.all(np.diff(carry.components.outside_carry_support.iloc[100:175])<0)
    _,_,_,missing=score('missing_live',carry_enabled=True)
    assert missing.components.carry_dead.eq(0).all()
    _,_,_,outlier=score('outside_outlier',carry_enabled=True)
    assert not outlier.components.outside_confirmed.any() and outlier.components.carry_dead.eq(0).all()
    f.loc[100:,'half']=2
    changed=estimate_signed_game_state(f,q.frames,e,{'carry_enabled':True})
    assert changed.components.carry_dead.iloc[100:].eq(0).all()


def test_smoothing_and_carry_do_not_cross_frame_gap():
    f,e,q,_=score('outside_missing');f.loc[100:,'frame']+=10;e['frame']=f.frame;q.frames['frame']=f.frame
    r=estimate_signed_game_state(f,q.frames,e,{'carry_enabled':True})
    assert r.components.carry_dead.iloc[100:].eq(0).all()


def test_hierarchy_ld_independent_of_team_contrast_and_resets():
    n=100;half=np.ones(n,int);scores=np.column_stack([np.ones(n),np.ones(n),np.full(n,-3.)])
    scores[30:50,2]=3.;scores[50:,0]=-4.
    a=decode_states(scores,half,config={'method':'hierarchical','short_spell_soft_penalty':0})
    other=scores.copy();other[:,:2]=other[:,[1,0]]
    b=decode_states(other,half,config={'method':'hierarchical','short_spell_soft_penalty':0})
    np.testing.assert_array_equal(a.game_state.dead_decoded,b.game_state.dead_decoded)
    assert (a.labels[:30]=='H').all() and (a.labels[30:50]=='D').all() and (a.labels[50:]=='A').all()
    refined=refine_boundaries(a.labels,scores,half,config={'search_radius_frames':15},fixed_dead_mask=a.game_state.dead_decoded)
    np.testing.assert_array_equal(refined.labels=='D',a.labels=='D')
    with pytest.raises(ValueError):refine_boundaries(a.labels,scores,half,fixed_dead_mask=np.zeros(n,bool))


def test_legacy_and_joint_defaults_unchanged(fake_bundle):
    f,e=scenario_inputs('running_live');q=analyze_observations(f)
    a=estimate_live_dead(f,q.frames,e);b=estimate_live_dead(f,q.frames,e,{'model':'legacy'})
    np.testing.assert_array_equal(a.live_score,b.live_score);np.testing.assert_array_equal(a.dead_score,b.dead_score)
    scores=np.column_stack([np.ones(len(f)),np.zeros(len(f)),np.full(len(f),-1.)]);half=f.half.to_numpy()
    np.testing.assert_array_equal(decode_states(scores,half).labels,decode_states(scores,half,config={'method':'joint'}).labels)


@pytest.mark.parametrize('extra',[{'player_window_frames':0},{'carry_enabled':'yes'},{'minimum_window_fraction':0},
    {'outside_margin_m':-1},{'prior_delta':float('nan')},{'fast_speed_mps':.1},{'unknown':1}])
def test_invalid_signed_config(extra):
    with pytest.raises(ValueError):score('running_live',**extra)


def test_cached_pipeline_recomputes_same_predictions(fake_bundle,tmp_path,monkeypatch):
    cfg={'fusion':{'control_normalization':'max'},'video':{'enabled':False}}
    storage={'root_dir':'runs','submission_filename':'synthetic.csv'}
    original=infer_from_features(fake_bundle,cfg,project_dir=tmp_path,storage_config=storage)
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline.games,'source_frames',lambda *a:len(fake_bundle.frames))
    replay=run_pipeline(cfg,project_dir=tmp_path,storage_config=storage,feature_run_dir=original.artifact_dir)
    np.testing.assert_array_equal(original.labels,replay.labels)
    for name in ['control/scores.csv','fusion/state_log_scores.csv','decode/labels.csv']:
        pd.testing.assert_frame_equal(read_table(original.artifact_dir,'core/initial/'+name),read_table(replay.artifact_dir,'core/initial/'+name),check_exact=True)
    assert read_json(replay.artifact_dir,'inputs/feature_replay.json')['run_id']==original.run_id
    eff=read_json(original.artifact_dir,'config/model.json');eff['fps']=50
    with pytest.raises(ValueError):load_feature_replay(original.artifact_dir,eff)
    eff=read_json(original.artifact_dir,'config/model.json');eff['video']['enabled']=True
    with pytest.raises(ValueError):load_feature_replay(original.artifact_dir,eff)


def test_signed_hierarchical_pipeline_persists_mask_and_config(fake_bundle,tmp_path):
    f,e=scenario_inputs('restart',n=len(fake_bundle.frames))
    for col in f:fake_bundle.frames[col]=f[col]
    root=Path(__file__).resolve().parents[1];cfg=yaml.safe_load((root/'config/d2_hierarchical.yaml').read_text())
    out=infer_from_features(fake_bundle,cfg,project_dir=tmp_path,storage_config={'root_dir':'runs','submission_filename':'s.csv'})
    mask=read_table(out.artifact_dir,'core/initial/decode/game_state.csv')
    np.testing.assert_array_equal(out.labels=='D',mask.dead_decoded)
    assert read_table(out.artifact_dir,'diagnostics/game_state_synthetic/summary.csv').passed.all()
    assert read_json(out.artifact_dir,'core/initial/live_dead/metadata.json')['model']=='signed'
