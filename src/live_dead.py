"""Independent live/dead heuristic evidence, not a complete referee-event classifier."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class LiveDeadResult:
    frames: pd.DataFrame
    live_score: np.ndarray
    dead_score: np.ndarray


def estimate_live_dead(frames: pd.DataFrame, quality: pd.DataFrame, event_frames: pd.DataFrame, config: dict | None = None) -> LiveDeadResult:
    cfg=config or {}
    n=len(frames)
    ball=quality.ball_observed.to_numpy(bool)
    good_speed=quality.ball_motion_reliable.to_numpy(bool)
    speed=np.nan_to_num(frames.ball_speed2d_mps.to_numpy(float),nan=0,posinf=0)
    player_motion=np.column_stack([frames.player_motion_robust_h.to_numpy(float),frames.player_motion_robust_a.to_numpy(float)])
    reliable=np.column_stack([frames.centroid_valid_h.to_numpy(bool), frames.centroid_valid_a.to_numpy(bool)])
    valid_motion=reliable & np.isfinite(player_motion)
    motion_sum=np.where(valid_motion,player_motion,0).sum(axis=1)
    motion_n=valid_motion.sum(axis=1)
    motion=np.divide(motion_sum,motion_n,out=np.zeros(n,float),where=motion_n>0)
    # Motion alone cannot prove live (or dead): evidence deliberately weak.
    live=np.full(n,float(cfg.get('live_prior_score',0.55)))
    dead=np.full(n,float(cfg.get('dead_prior_score',0.25)))
    live+=float(cfg.get('ball_moving_live_weight',0.22))*good_speed*np.clip(speed/5.0,0,1)
    live+=float(cfg.get('player_motion_live_weight',0.12))*(motion_n>0)*np.clip(motion/3.0,0,1)
    dead+=float(cfg.get('out_of_pitch_weight',0.35))*event_frames.out_of_pitch_hint.to_numpy(float)*quality.ball_weight.to_numpy(float)
    # Concentrate stoppage evidence at event boundaries, never throughout a long tag span.
    dead+=float(cfg.get('stoppage_anchor_weight',0.32))*event_frames.stoppage_anchor.to_numpy(float)
    dead+=float(cfg.get('restart_anchor_weight',0.14))*event_frames.restart_anchor.to_numpy(float)
    # Stationary ball near an event boundary supports preparation but never proves D.
    near_anchor=(event_frames.stoppage_anchor.to_numpy(float)>0)|(event_frames.restart_anchor.to_numpy(float)>0)
    dead+=float(cfg.get('stationary_near_anchor_weight',0.12))*near_anchor*good_speed*(speed<0.6)
    # Do not interpret missing ball as D.
    df=pd.DataFrame({'frame':frames.frame.to_numpy(int), 'live_score':live,'dead_score':dead,
                     'player_motion_evidence':motion,'has_player_motion':motion_n>0})
    return LiveDeadResult(df,live,dead)
