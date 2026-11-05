"""Fuse bounded uncalibrated H/A/D evidence; never report as calibrated probabilities."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


STATE_ORDER=('H','A','D')


@dataclass
class EvidenceBundle:
    state_log_scores: np.ndarray
    transition_hint: np.ndarray
    stop_hint: np.ndarray
    restart_hint: np.ndarray
    metadata: dict


def fuse_evidence(frames: pd.DataFrame, quality, control, live_dead, events, config: dict | None = None) -> EvidenceBundle:
    cfg=config or {}; n=len(frames)
    if len(quality.frames)!=n or len(control.scores)!=n or len(live_dead.live_score)!=n:
        raise ValueError('Evidence source lengths must match canonical timeline')
    # Scores are additive *utilities* for Viterbi, not true normalized log-likelihoods.
    w_control=float(cfg.get('team_weight',1.0));w_live=float(cfg.get('live_weight',1.0))
    w_dead=float(cfg.get('dead_weight',1.0))
    h=w_control*control.scores[:,0]+w_live*live_dead.live_score
    a=w_control*control.scores[:,1]+w_live*live_dead.live_score
    d=w_dead*live_dead.dead_score+float(cfg.get('dead_bias',0))
    scores=np.column_stack((h,a,d))
    # Hard-mask only legal publicly published match metadata, not inferred private events.
    for half, start, end in __import__('games').half_video_blocks('BAR-ATH'):
        if half==2:
            kickoff=int(cfg.get('second_half_kickoff_frame',113742))
            if not (start<=kickoff<=end): raise ValueError('Invalid second-half kickoff frame')
            scores[start:kickoff,:2]=-1e6
            scores[start:kickoff,2]=1e6
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite fused scores: check NaN guards in evidence models')
    return EvidenceBundle(scores, control.transition_hint.copy(),
                          events.frames.stoppage_anchor.to_numpy(float).copy(),
                          events.frames.restart_anchor.to_numpy(float).copy(),
                          {'score_semantics':'uncalibrated additive utilities','order':STATE_ORDER})
