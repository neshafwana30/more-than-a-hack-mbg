"""Data reliability masks, not possession predictions or calibrated confidence."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class QualityResult:
    frames: pd.DataFrame
    observation_weight: np.ndarray


def analyze_observations(frames: pd.DataFrame, config: dict | None = None) -> QualityResult:
    cfg = config or {}
    n = len(frames)
    ball = frames.ball_ok.to_numpy(bool)
    v = frames.ball_speed2d_mps.to_numpy(float)
    max_speed = float(cfg.get('max_ball_speed_mps', 45.0))
    speed_bad = ball & np.isfinite(v) & (v > max_speed)
    movement_ok = ball & np.isfinite(v) & ~speed_bad
    ph = np.clip(frames.valid_h_fraction.to_numpy(float), 0, 1)
    pa = np.clip(frames.valid_a_fraction.to_numpy(float), 0, 1)
    players = np.sqrt(ph * pa)
    player_motion_ok = frames.centroid_valid_h.to_numpy(bool) & frames.centroid_valid_a.to_numpy(bool)
    missing_length = frames.ball_missing_run_frames.to_numpy(int)
    bin_gap = np.where(ball,'observed',np.where(missing_length<=5,'very_short',np.where(missing_length<=125,'short','long')))
    # These are bounded quality WEIGHTS, not probabilities.
    bweight=np.where(ball, 1.0, 0.0)
    bweight[speed_bad] = 0.25
    # First observed frame after any gap lacks a reliable instantaneous speed.
    observation_weight=np.clip(0.65*bweight+0.35*players,0,1)
    flags = np.zeros(n,dtype=np.uint8)
    flags[~ball] |= 1
    flags[speed_bad] |= 2
    flags[players<0.65] |= 4
    flags[~player_motion_ok] |= 8
    if 'video_recovered_ball' in frames:
        flags[frames.video_recovered_ball.to_numpy(bool)] |= 16
    result=pd.DataFrame({
        'frame':frames.frame.to_numpy(int),
        'ball_observed':ball,
        'suspected_ball_jump':speed_bad,
        'ball_motion_reliable':movement_ok,
        'ball_weight':bweight,
        'player_quality_weight':players,
        'motion_reliable':player_motion_ok,
        'gap_class':bin_gap,
        'observability_flags':flags,
        'observation_weight':observation_weight,
    })
    return QualityResult(result,observation_weight)
