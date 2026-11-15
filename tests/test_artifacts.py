"""Behavior tests for persistence, failure visibility and repeated inference passes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import src.pipeline as pipeline
from src.artifacts import ArtifactRun, read_table, read_json, list_runs
from src.features import FeatureResult
from src.visual_analyzer import VisualResult
from src.boundary_refinement import refine_boundaries


def storage(tmp_path):
    return {'root_dir': str(tmp_path/'runs'), 'submission_filename': 'possession.csv'}


def test_unique_runs_and_lossless_csv_json(tmp_path):
    table = pd.DataFrame({'text':['N/A','',None], 'value':[.12345678901234568, np.nan, 2.0],
                          'elapsed':pd.to_timedelta([1,2,3],unit='s')})
    paths=[]
    for _ in range(2):
        with ArtifactRun({'fps':25}, storage_config=storage(tmp_path)) as run:
            paths.append(run.path)
            run.csv('table.csv', table)
            run.json('nullable.json', {'value':np.nan,'count':np.int64(3)})
    assert paths[0] != paths[1]
    pd.testing.assert_frame_equal(read_table(paths[0],'table.csv'),table,check_exact=True)
    assert read_json(paths[0],'nullable.json') == {'value':None,'count':3}
    assert read_json(paths[0])['status']=='completed'
    assert len(list_runs(tmp_path/'runs'))==2


def test_inference_serializes_all_core_fields_and_preserves_predictions(fake_bundle, tmp_path):
    cfg={'fps':25,'video':{'enabled':False}}
    q,e,c,l,f,d=pipeline._run_core(fake_bundle.frames,fake_bundle.events,cfg)
    expected=refine_boundaries(d.labels,f.state_log_scores,fake_bundle.frames.half,
                               q.observation_weight,f.stop_hint,f.restart_hint).labels
    result=pipeline.infer_from_features(fake_bundle,cfg,storage_config=storage(tmp_path))
    run=Path(result.artifact_dir);m=read_json(run)
    np.testing.assert_array_equal(result.labels,expected)
    np.testing.assert_allclose(read_table(run,'core/initial/fusion/state_log_scores.csv')[['H','A','D']], f.state_log_scores)
    np.testing.assert_allclose(read_table(run,'core/initial/control/scores.csv')[['H','A']], c.scores)
    np.testing.assert_array_equal(read_table(run,'boundary/labels.csv').label,result.labels)
    expected_files=['quality/frames.csv','quality/observation_weight.csv','events/frames.csv',
                    'events/candidates.csv','control/frames.csv','control/scores.csv','control/transition_hint.csv',
                    'live_dead/frames.csv','live_dead/live_score.csv','live_dead/dead_score.csv',
                    'fusion/state_log_scores.csv','fusion/transition_hint.csv','fusion/stop_hint.csv',
                    'fusion/restart_hint.csv','fusion/metadata.json','decode/labels.csv',
                    'decode/transitions.csv','decode/diagnostics.json']
    assert all('core/initial/'+p in m['files'] for p in expected_files)
    assert m['stages']['video']['status']=='skipped'
    assert m['stages']['core/after_video']['status']=='skipped'
    assert m['final_core_pass']=='initial'
    assert read_json(run,'config/model.json')['pitch']==[105.3,68.0]
    assert cfg=={'fps':25,'video':{'enabled':False}}


def test_video_refresh_saves_both_passes_and_player_arrays(fake_bundle,tmp_path,monkeypatch):
    n=len(fake_bundle.frames)
    positions={'x':np.full((n,2),.4),'y':np.full((n,2),.5),
               'valid':np.ones((n,2),bool),'sides':np.array(['H','A'])}
    fake_bundle.player_positions=positions
    obs=pd.DataFrame({'frame':[60],'x':[.4],'y':[.5],'quality':[.9]})
    monkeypatch.setattr(pipeline,'analyze_video_windows',lambda *a,**k:VisualResult(obs,{'status':'completed','accepted':1}))
    result=pipeline.infer_from_features(fake_bundle,{'video':{'enabled':True}},
                                       {'video':'fake.mp4','homography':'fake.csv'},storage_config=storage(tmp_path))
    run=Path(result.artifact_dir);m=read_json(run)
    assert m['final_core_pass']=='after_video'
    assert not read_table(run,'features/initial/frames.csv').ball_ok.iloc[60]
    assert read_table(run,'features/refreshed/result.csv').ball_ok.iloc[60]
    assert 'core/initial/decode/labels.csv' in m['files']
    assert 'core/after_video/decode/labels.csv' in m['files']
    assert read_table(run,'video/observations.csv').frame.tolist()==[60]
    stored=read_table(run,'features/initial/player_positions.csv')
    np.testing.assert_array_equal(stored.slot_1_valid,positions['valid'][:,0])
    np.testing.assert_allclose(stored.slot_2_y,positions['y'][:,1])
    assert read_json(run,'features/initial/player_positions_metadata.json')['sides']==['H','A']


def test_failed_stage_keeps_prior_results(fake_bundle,tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('synthetic control failure')
    monkeypatch.setattr(pipeline,'estimate_team_control',fail)
    with pytest.raises(RuntimeError,match='synthetic control failure'):
        pipeline.infer_from_features(fake_bundle,{},storage_config=storage(tmp_path))
    runs=list_runs(tmp_path/'runs',status=None);assert len(runs)==1
    m=read_json(runs[0]['path'])
    assert m['status']=='failed'
    assert m['stages']['core/initial/control']['status']=='failed'
    assert 'core/initial/quality/frames.csv' in m['files']
    assert 'boundary/labels.csv' not in m['files']


def test_full_entrypoint_uses_yaml_and_effective_override(fake_bundle,tmp_path,monkeypatch):
    config_path=tmp_path/'storage.yaml'
    config_path.write_text('root_dir: saved_runs\nsubmission_filename: labels.csv\n')
    monkeypatch.setattr(pipeline,'build_features',lambda **kwargs:fake_bundle)
    monkeypatch.setattr(pipeline.games,'source_frames',lambda *a,**k:len(fake_bundle.frames))
    monkeypatch.setattr(pipeline,'official_validator_if_available',lambda *a:{'status':'passed'})
    result=pipeline.run_pipeline({'video':{'enabled':True}},project_dir=tmp_path,
                                 storage_config_path=config_path,enable_video=False)
    run=Path(result.artifact_dir)
    assert run.parent==tmp_path/'saved_runs'
    assert Path(result.output['path'])==run/'submission/labels.csv'
    assert read_json(run,'config/model.json')['video']['enabled'] is False
    assert read_json(run)['execution']['storage_config_source']==str(config_path)
    assert read_json(run,'submission/diagnostics.json')['validator']['status']=='passed'
    np.testing.assert_array_equal(read_table(run,'submission/labels.csv').label,result.labels)


def test_storage_rejects_escape_and_bad_settings(tmp_path):
    with pytest.raises(ValueError,match='without directories'):
        ArtifactRun({},storage_config={**storage(tmp_path),'submission_filename':'../bad.csv'})
    with ArtifactRun({},storage_config=storage(tmp_path)) as run:
        with pytest.raises(ValueError,match='inside'):
            run.json('../outside.json',{})


def test_transient_windows_file_lock_is_retried(tmp_path,monkeypatch):
    original=Path.replace
    attempts=[]
    with ArtifactRun({},storage_config=storage(tmp_path)) as run:
        def locked_once(path,target):
            if Path(target).name=='manifest.json' and len(attempts)<2:
                attempts.append(str(target))
                raise PermissionError('synthetic Windows file lock')
            return original(path,target)
        monkeypatch.setattr(Path,'replace',locked_once)
        run.json('value.json',{'value':1})
    assert len(attempts)==2
    assert read_json(run.path)['status']=='completed'
    assert read_json(run.path,'value.json')=={'value':1}
