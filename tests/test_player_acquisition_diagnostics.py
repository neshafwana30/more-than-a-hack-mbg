"""Association/candidate tests use toy observations only, never match labels."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.player_acquisition_diagnostics import player_acquisition_diagnostics,diagnostic_config,run_player_acquisition_diagnostics
from src.player_acquisition_scenarios import scenario_observations,player_acquisition_scenarios
from src.pipeline import infer_from_features
from src.features import FeatureResult
from src.artifacts import read_json,read_table


CFG={'enabled':True}


def test_candidate_scenario_contracts():
    r=player_acquisition_scenarios(CFG)
    assert len(r.summary)==11
    assert r.summary.passed.all(),r.summary.loc[~r.summary.passed].to_string()
    rows=r.summary.set_index('scenario')
    assert rows.loc['same_team_pass','candidate_H_frames']>0
    assert rows.loc['interception','candidate_A_frames']>0
    assert rows.loc['short_control','candidate_A_frames']>0
    assert rows.loc['contested_reception','both_ambiguous_frames']>0
    assert rows.loc['nearest_switch','naive_high_associated_low_H_frames']>0
    for name in ['temporary_touch','comoving_opponent','ambiguous_duplicate','occlusion','player_dropout']:
        assert rows.loc[name,'candidate_H_frames']==rows.loc[name,'candidate_A_frames']==0


def test_slot_permutation_is_not_identity_switch():
    f,p,_=scenario_observations('same_team_pass');a=player_acquisition_diagnostics(f,p,CFG)
    g,q,_=scenario_observations('slot_permutation');b=player_acquisition_diagnostics(g,q,CFG)
    np.testing.assert_array_equal(a.frames[['candidate_h','candidate_a']],b.frames[['candidate_h','candidate_a']])
    assert b.frames.observation_column_changed_h.sum()>a.frames.observation_column_changed_h.sum()
    assert b.frames.nearest_candidate_continues_h.sum()==a.frames.nearest_candidate_continues_h.sum()
    assert b.frames.nearest_candidate_switch_h.sum()==a.frames.nearest_candidate_switch_h.sum()


def test_duplicate_points_are_rejected_not_forced():
    f,p,_=scenario_observations('ambiguous_duplicate');r=player_acquisition_diagnostics(f,p,CFG)
    assert r.frames.association_status_h.eq('ambiguous').any()
    assert not r.frames.candidate_h.any()
    ambiguous=r.frames.association_status_h.eq('ambiguous')
    assert r.frames.loc[ambiguous,'predecessor_index_h'].eq(-1).all()


def test_naive_speed_can_mix_different_candidates():
    f,p,_=scenario_observations('nearest_switch');r=player_acquisition_diagnostics(f,p,CFG).frames
    case=r.loc[r.naive_high_associated_low_h]
    assert len(case)>0 and case.nearest_candidate_switch_h.all()
    assert r.naive_high_associated_low_near_h.any()
    assert case.naive_point_speed_h_mps.gt(12).all()
    assert case.associated_point_speed_h_mps.le(12).all()
    assert not r.candidate_h.any()


@pytest.mark.parametrize('case',['half','frame_gap','missing_player'])
def test_links_never_bridge_gaps(case):
    f,p,_=scenario_observations('same_team_pass')
    if case=='half':f.loc[58:,'half']=2
    if case=='frame_gap':f.loc[58:,'frame']+=20
    if case=='missing_player':p['valid'][58,1]=False
    r=player_acquisition_diagnostics(f,p,CFG).frames
    at=59 if case=='missing_player' else 58
    assert r.loc[at,'predecessor_index_h']==-1
    assert r.loc[at,'local_history_h_frames']==1
    assert not r.loc[55:63,'candidate_h'].any()


def test_lookahead_is_explicit_and_does_not_cross_half():
    f,p,_=scenario_observations('same_team_pass')
    full=player_acquisition_diagnostics(f,p,CFG)
    end=57
    truncated_positions={k:(v[:end] if k!='sides' else v) for k,v in p.items()}
    short=player_acquisition_diagnostics(f.iloc[:end],truncated_positions,CFG)
    assert short.metadata['lookahead_frames']==5
    np.testing.assert_array_equal(full.frames.candidate_h.iloc[:end-5],short.frames.candidate_h.iloc[:end-5])
    assert not short.frames.candidate_h.iloc[-5:].any()


@pytest.mark.parametrize('extra',[{'association_max_speed_mps':0},{'association_ambiguity_margin_m':0},
    {'pre_frames':0},{'post_frames':True},{'minimum_turn_degrees':190},{'minimum_post_support_fraction':0},
    {'minimum_player_quality':2},{'unknown':1},{'history_cap_frames':3}])
def test_invalid_settings(extra):
    with pytest.raises(ValueError):diagnostic_config({**CFG,**extra})


def _pipeline_bundle():
    from conftest import synthetic_bundle
    bundle=synthetic_bundle(120);f,p,_=scenario_observations('same_team_pass')
    # Retain unrelated feature/tag columns from the synthetic pipeline fixture.
    for name in f:bundle.frames[name]=f[name]
    for side,code in [('H','h'),('A','a')]:
        mask=p['sides']==side
        dx=(p['x'][:,mask]-f.ball_x.to_numpy()[:,None])*105.3
        dy=(p['y'][:,mask]-f.ball_y.to_numpy()[:,None])*68.
        d=np.hypot(dx,dy);arg=np.argmin(d,axis=1)
        bundle.frames[f'nearest_{code}_m']=d[np.arange(len(f)),arg]
        bundle.frames[f'nearest_{code}_x']=p['x'][:,mask][np.arange(len(f)),arg]
        bundle.frames[f'nearest_{code}_y']=p['y'][:,mask][np.arange(len(f)),arg]
    bundle.frames['nearest_team_proxy']=np.where(bundle.frames.nearest_h_m<bundle.frames.nearest_a_m,'H','A')
    bundle.player_positions=p
    bundle.slots=pd.DataFrame({'slot':[str(i) for i in range(len(p['sides']))],'side':p['sides']})
    return bundle


def test_pipeline_diagnostics_do_not_change_scoring_or_labels(tmp_path):
    base={'fusion':{'control_normalization':'max'}};enabled=deepcopy(base)
    enabled['diagnostics']={'player_acquisition':CFG}
    storage={'root_dir':'runs','submission_filename':'synthetic.csv'}
    old=infer_from_features(_pipeline_bundle(),base,project_dir=tmp_path,storage_config=storage)
    new=infer_from_features(_pipeline_bundle(),enabled,project_dir=tmp_path,storage_config=storage)
    np.testing.assert_array_equal(old.labels,new.labels)
    for path in ['control/scores.csv','control/transition_hint.csv','live_dead/frames.csv','fusion/state_log_scores.csv','decode/labels.csv']:
        pd.testing.assert_frame_equal(read_table(old.artifact_dir,f'core/initial/{path}'),read_table(new.artifact_dir,f'core/initial/{path}'),check_exact=True)
    assert read_table(new.artifact_dir,'diagnostics/player_acquisition/frames.csv').candidate_h.any()
    coverage=read_table(new.artifact_dir,'diagnostics/player_acquisition/association_coverage.csv')
    assert coverage.linked_fraction_of_observed.dropna().between(0,1).all()
    assert len(read_table(new.artifact_dir,'diagnostics/player_acquisition/relative_speed_summary.csv'))==4
    assert read_table(new.artifact_dir,'diagnostics/player_acquisition_synthetic/summary.csv').passed.all()
    assert len(read_table(new.artifact_dir,'diagnostics/player_acquisition_reference/event_context.csv'))>0
    settings=tmp_path/'storage.yaml';settings.write_text('root_dir: runs\nsubmission_filename: audit.csv\n')
    m=read_json(new.artifact_dir)
    label_path='inference/result/labels.csv'
    assert label_path in m['files']
    folder=run_player_acquisition_diagnostics(new.artifact_dir,CFG,project_dir=tmp_path,storage_config_path=settings)
    assert read_json(folder)['entrypoint']=='player_acquisition_diagnostics'
    assert read_json(folder,'inputs/reference.json')['files'][label_path]==m['files'][label_path]['sha256']
    assert read_table(folder,'diagnostics/player_acquisition_synthetic/summary.csv').passed.all()
    assert read_json(new.artifact_dir)['files'][label_path]['sha256']==m['files'][label_path]['sha256']
