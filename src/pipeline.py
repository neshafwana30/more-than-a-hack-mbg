"""Deterministic offline inference with CSV/JSON artifacts for every run."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import games
from .artifacts import ArtifactRun, load_storage_config, PROJECT_DIR
from .features import FeatureResult, build_features, refresh_visual_ball
from .observation_quality import analyze_observations
from .event_evidence import detect_events
from .team_control import estimate_team_control
from .live_dead import estimate_live_dead
from .evidence_fusion import fuse_evidence
from .temporal_decoder import decode_states
from .boundary_refinement import refine_boundaries
from .ambiguity_detector import select_ambiguities
from .visual_analyzer import VisualResult, analyze_video_windows
from .diagnostics import feature_diagnostics, core_diagnostics, prediction_diagnostics
from .submission import (save_submission, aggregate_diagnostics,
                         official_validator_if_available, validate_labels)


@dataclass
class PipelineResult:
    labels: np.ndarray
    diagnostics: dict
    output: dict | None
    transitions: pd.DataFrame
    artifact_dir: str | None = None
    run_id: str | None = None


def _effective_config(config, enable_video=None):
    cfg = deepcopy(config)
    cfg.setdefault('game', games.GAME)
    cfg.setdefault('fps', games.GAMES[cfg['game']]['fps'])
    cfg.setdefault('pitch', games.GAMES[cfg['game']]['pitch'])
    cfg.setdefault('source', 'tactical')
    cfg.setdefault('video', {})
    cfg['video'].setdefault('enabled', False)
    if enable_video is not None:
        cfg['video']['enabled'] = bool(enable_video)
    return cfg


def _run_core(frame, events, cfg, *, artifacts=None, pass_name='initial'):
    """Capture every returned object; initial and after_video have separate paths."""
    def call(name, function, *args):
        if artifacts is None:
            return function(*args)
        return artifacts.capture(f'core/{pass_name}/{name}', function, *args,
                                 frame_ids=frame.frame.to_numpy())
    quality = call('quality', analyze_observations, frame, cfg.get('quality'))
    event = call('events', detect_events, frame, events, cfg.get('events'))
    control = call('control', estimate_team_control, frame, quality.frames, cfg.get('control'), cfg.get('fps',25), tuple(cfg.get('pitch',(105.3,68.0))))
    live = call('live_dead', estimate_live_dead, frame, quality.frames, event.frames, cfg.get('live_dead'))
    fusion = call('fusion', fuse_evidence, frame, quality, control, live, event, cfg.get('fusion'))
    decode = call('decode', decode_states, fusion.state_log_scores, frame.half.to_numpy(int),
                  fusion.transition_hint, fusion.stop_hint, fusion.restart_hint, cfg.get('temporal'))
    if artifacts is not None:
        artifacts.capture(f'diagnostics/{pass_name}/core', core_diagnostics, frame, fusion, cfg)
        artifacts.capture(f'diagnostics/{pass_name}/prediction', prediction_diagnostics,
                          decode.labels, frame.half.to_numpy(int), cfg.get('fps', 25))
    return quality, event, control, live, fusion, decode


def _infer_from_features(bundle, cfg, video_paths, run, *, features_saved=False):
    frame = bundle.frames.copy()
    if not np.array_equal(frame.frame.to_numpy(), np.arange(len(frame))):
        raise ValueError('Features must be indexed by sorted canonical 0-based frame')
    if not features_saved:
        run.capture('features/initial', lambda: bundle)
    run.capture('diagnostics/features_initial', feature_diagnostics, bundle, cfg.get('fps', 25))
    quality, event, control, live, fusion, decode = _run_core(frame, bundle.events, cfg, artifacts=run)
    ambiguity = run.capture('ambiguity', select_ambiguities, frame, quality.frames, control.frames,
                            fusion, decode.labels, cfg.get('ambiguity'))
    video_diag = {'status': 'disabled', 'accepted': 0}
    observations = pd.DataFrame(columns=['frame', 'x', 'y', 'quality'])
    if cfg.get('video', {}).get('enabled', False):
        if not video_paths:
            video_diag = {'status': 'skipped', 'reason': 'video_paths_missing', 'accepted': 0}
            run.skipped('video', 'video_paths_missing', VisualResult(observations, video_diag))
        else:
            video_cfg = dict(cfg.get('video', {}))
            video_cfg['fps'] = cfg.get('fps', 25)
            run.json('config/video_effective.json', video_cfg)
            visual = run.capture('video', analyze_video_windows, ambiguity.windows, frame,
                                 video_paths['video'], video_paths['homography'], video_cfg)
            video_diag = visual.diagnostics
            observations = visual.observations
            if video_diag.get('status') == 'skipped':
                run.skipped('video', video_diag.get('reason', 'unavailable'))
    else:
        run.skipped('video', 'disabled_by_config', VisualResult(observations, video_diag))
    if len(observations):
        if bundle.player_positions is None:
            warnings.warn('No player coordinates: video recovery skipped', RuntimeWarning)
            run.skipped('video/application', 'player_positions_missing')
        else:
            frame = run.capture('features/refreshed', refresh_visual_ball,
                                frame, bundle.player_positions, observations,
                                pitch=tuple(cfg.get('pitch', (105.3, 68.0))), fps=cfg.get('fps', 25))
            run.json('video/application.json', {'status': 'applied',
                     'recovered_frames': int(frame.video_recovered_ball.sum())})
            refreshed = FeatureResult(frame, bundle.events, bundle.slots, bundle.player_positions)
            run.capture('diagnostics/features_refreshed', feature_diagnostics, refreshed, cfg.get('fps', 25))
            quality, event, control, live, fusion, decode = _run_core(
                frame, bundle.events, cfg, artifacts=run, pass_name='after_video')
    else:
        run.skipped('video/application', 'no_observations')
    if 'core/after_video/decode' not in run.manifest['stages']:
        run.skipped('core/after_video', 'video_did_not_refresh_features')
    run.manifest['final_core_pass'] = 'after_video' if 'core/after_video/decode' in run.manifest['stages'] else 'initial'
    boundary = run.capture('boundary', refine_boundaries, decode.labels, fusion.state_log_scores,
                           frame.half.to_numpy(int), quality.observation_weight, fusion.stop_hint,
                           fusion.restart_hint, cfg.get('boundary'), frame_ids=frame.frame.to_numpy())
    labels = boundary.labels
    if cfg.get('control',{}).get('acquisition',{}).get('enabled',False):
        from .acquisition_diagnostics import acquisition_diagnostics
        from .control_contribution_audit import audit_control_contributions
        run.capture('diagnostics/control_contributions',audit_control_contributions,frame,control.components,
                    fusion.control_components.forced_d)
        from .acquisition_scenarios import acquisition_scenarios
        run.capture('diagnostics/final/acquisition', acquisition_diagnostics,
                    frame, control, fusion, decode, labels, cfg.get('fps',25))
        run.capture('diagnostics/acquisition_synthetic', acquisition_scenarios,
                    cfg.get('control',{}), cfg.get('fps',25), tuple(cfg.get('pitch',(105.3,68.0))))
    run.capture('inference/validation', validate_labels, labels, frame.half.to_numpy(int), len(frame))
    prediction = run.capture('diagnostics/final/prediction', prediction_diagnostics,
                             labels, frame.half.to_numpy(int), cfg.get('fps', 25))
    extra = {'preliminary_transitions': decode.diagnostics['transition_count'],
             'soft_short_spells_merged': decode.diagnostics['short_runs_merged'],
             'boundaries_examined': len(boundary.report),
             'boundaries_shifted': int((boundary.report.shift_frames != 0).sum()) if len(boundary.report) else 0,
             'video_windows_selected': len(ambiguity.windows), 'video': video_diag,
             'ball_observed_share_graded': float(frame.loc[frame.graded, 'ball_ok'].mean())}
    diag = run.capture('inference/diagnostics', aggregate_diagnostics, labels, frame.half.to_numpy(int),
                       extra, fps=int(cfg.get('fps', 25)), segments=prediction.segments)
    result = PipelineResult(labels, diag, None, decode.transitions, str(run.path), run.run_id)
    run.save('inference/result', result, frame.frame.to_numpy())
    return result


def infer_from_features(bundle: FeatureResult, cfg: dict, video_paths: dict | None = None, *,
                        storage_config=None, storage_config_path=None, project_dir=None) -> PipelineResult:
    """Each direct inference call owns a run, including synthetic notebook/test inputs."""
    cfg = _effective_config(cfg)
    with ArtifactRun(cfg, project_dir=project_dir, storage_config=storage_config,
                     storage_config_path=storage_config_path, entrypoint='infer_from_features') as run:
        return _infer_from_features(bundle, cfg, video_paths, run)


def run_pipeline(config: dict, output_path=None, project_dir=None, data_dir=None, enable_video=None, *,
                 storage_config=None, storage_config_path=None, execution_metadata=None) -> PipelineResult:
    """One run directory contains features, all inference stages and validated submission.

    output_path is retained only as a filename override; output directories belong to storage YAML.
    """
    project_dir = Path(project_dir or PROJECT_DIR).resolve()
    cfg = _effective_config(config, enable_video)
    # Resolve a YAML config once. ArtifactRun validates the resulting input settings again.
    storage = load_storage_config(project_dir, storage_config, storage_config_path)
    settings = {k: storage[k] for k in ('root_dir', 'submission_filename')}
    if output_path is not None:
        settings['submission_filename'] = str(output_path)
    execution = {'data_dir': str(Path(data_dir).resolve()) if data_dir else games.DATA_DIR,
                 'enable_video_override': enable_video, 'storage_config_source': storage['source_file'],
                 **(execution_metadata or {})}
    with ArtifactRun(cfg, project_dir=project_dir, storage_config=settings,
                     entrypoint='run_pipeline', execution=execution) as run:
        if cfg['source'] != 'tactical':
            raise ValueError('Broadcast is out of scope')
        bundle = run.capture('features/initial', build_features, game=cfg['game'], data_dir=data_dir, source='tactical')
        if len(bundle.frames) != games.source_frames('tactical', cfg['game']):
            raise ValueError('Wrong tactical timeline')
        paths = games.asset_paths(game=cfg['game'], data_dir=data_dir)
        run.json('inputs/assets.json', paths)
        result = _infer_from_features(bundle, cfg, paths, run, features_saved=True)
        submission_path = run.path / 'submission' / run.storage['submission_filename']
        result.output = run.capture('submission/write', save_submission, result.labels, submission_path,
                                   expected_count=games.source_frames('tactical', cfg['game']),
                                   half=bundle.frames.half.to_numpy(int))
        run.register(submission_path, 'csv', rows=len(result.labels), columns=['frame', 'label'],
                     dtypes={'frame': 'int64', 'label': 'object'}, null_token='__NULL__')
        result.diagnostics['submission_sha256'] = result.output['sha256']
        result.diagnostics['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
        official = run.capture('submission/validator', official_validator_if_available, project_dir, submission_path)
        result.diagnostics['validator'] = official
        run.json('submission/diagnostics.json', result.diagnostics)
        run.save('result', result, bundle.frames.frame.to_numpy())
        return result
