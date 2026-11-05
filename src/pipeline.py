"""Single-command deterministic, competition-compliant offline inference pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import games
from .features import FeatureResult, build_features, refresh_visual_ball
from .observation_quality import analyze_observations
from .event_evidence import detect_events
from .team_control import estimate_team_control
from .live_dead import estimate_live_dead
from .evidence_fusion import fuse_evidence
from .temporal_decoder import decode_states
from .boundary_refinement import refine_boundaries
from .ambiguity_detector import select_ambiguities
from .visual_analyzer import analyze_video_windows
from .submission import (save_submission, aggregate_diagnostics, save_diagnostics,
                         official_validator_if_available, validate_labels)


@dataclass
class PipelineResult:
    labels: np.ndarray
    diagnostics: dict
    output: dict | None
    transitions: pd.DataFrame


def _run_core(frame, events, cfg):
    quality=analyze_observations(frame,cfg.get('quality'))
    event=detect_events(frame,events,cfg.get('events'))
    control=estimate_team_control(frame,quality.frames,cfg.get('control'))
    live=estimate_live_dead(frame,quality.frames,event.frames,cfg.get('live_dead'))
    fusion=fuse_evidence(frame,quality,control,live,event,cfg.get('fusion'))
    decode=decode_states(fusion.state_log_scores,frame.half.to_numpy(int),
                         fusion.transition_hint,fusion.stop_hint,fusion.restart_hint,cfg.get('temporal'))
    return quality,event,control,live,fusion,decode


def infer_from_features(bundle:FeatureResult, cfg:dict, video_paths:dict|None=None) -> PipelineResult:
    """Pure inference integration surface; synthetic FeatureResult works without source assets."""
    frame=bundle.frames.copy()
    if not np.array_equal(frame.frame.to_numpy(),np.arange(len(frame))):
        raise ValueError('Features must be indexed by sorted canonical 0-based frame')
    quality,event,control,live,fusion,decode=_run_core(frame,bundle.events,cfg)
    ambiguity=select_ambiguities(frame,quality.frames,control.frames,fusion,decode.labels,cfg.get('ambiguity'))
    video_diag={'status':'disabled','accepted':0}
    if cfg.get('video',{}).get('enabled',False):
        if not video_paths:
            video_diag={'status':'skipped','reason':'video_paths_missing','accepted':0}
        else:
            video_cfg=dict(cfg.get('video',{}))
            video_cfg['fps']=cfg.get('fps',25)
            result=analyze_video_windows(ambiguity.windows,frame,video_paths['video'],
                                         video_paths['homography'],video_cfg)
            video_diag=result.diagnostics
            if len(result.observations):
                if bundle.player_positions is None:
                    warnings.warn('No player coordinates: video recovery skipped',RuntimeWarning)
                else:
                    frame=refresh_visual_ball(frame,bundle.player_positions,result.observations,
                                              pitch=tuple(cfg.get('pitch',(105.3,68.0))),fps=cfg.get('fps',25))
                    quality,event,control,live,fusion,decode=_run_core(frame,bundle.events,cfg)
    boundary=refine_boundaries(decode.labels,fusion.state_log_scores,frame.half.to_numpy(int),
                               quality.observation_weight,fusion.stop_hint,fusion.restart_hint,cfg.get('boundary'))
    labels=boundary.labels
    validate_labels(labels,frame.half.to_numpy(int),len(frame))
    extra={'preliminary_transitions':decode.diagnostics['transition_count'],
           'soft_short_spells_merged':decode.diagnostics['short_runs_merged'],
           'boundaries_examined':len(boundary.report),
           'boundaries_shifted':int((boundary.report.shift_frames!=0).sum()) if len(boundary.report) else 0,
           'video_windows_selected':len(ambiguity.windows),'video':video_diag,
           'ball_observed_share_graded':float(frame.loc[frame.graded,'ball_ok'].mean())}
    diag=aggregate_diagnostics(labels,frame.half.to_numpy(int),extra,fps=int(cfg.get('fps',25)))
    return PipelineResult(labels,diag,None,decode.transitions)


def run_pipeline(config:dict,output_path,project_dir, data_dir=None, enable_video=None) -> PipelineResult:
    """Load private match data locally, infer, write CSV, validate and save aggregates."""
    cfg=dict(config)
    cfg.setdefault('game',games.GAME)
    cfg.setdefault('fps',games.GAMES[cfg['game']]['fps'])
    cfg.setdefault('pitch',games.GAMES[cfg['game']]['pitch'])
    if cfg.get('source','tactical')!='tactical':raise ValueError('Broadcast is out of scope')
    if enable_video is not None:
        cfg['video']=dict(cfg.get('video',{}),enabled=bool(enable_video))
    bundle=build_features(game=cfg['game'],data_dir=data_dir,source='tactical')
    if len(bundle.frames)!=games.source_frames('tactical',cfg['game']):
        raise ValueError('Wrong tactical timeline')
    paths=games.asset_paths(game=cfg['game'],data_dir=data_dir)
    result=infer_from_features(bundle,cfg,video_paths=paths)
    result.output=save_submission(result.labels,output_path,
                                  expected_count=games.source_frames('tactical',cfg['game']),
                                  half=bundle.frames.half.to_numpy(int))
    result.diagnostics['submission_sha256']=result.output['sha256']
    result.diagnostics['config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True,default=str).encode()).hexdigest()
    official=official_validator_if_available(project_dir,output_path)
    result.diagnostics['validator']=official
    diag_path=Path(output_path).with_suffix('.diagnostics.json')
    save_diagnostics(result.diagnostics,diag_path)
    return result
