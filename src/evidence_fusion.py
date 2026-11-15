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
    control_components: pd.DataFrame | None = None


def fuse_evidence(frames: pd.DataFrame, quality, control, live_dead, events, config: dict | None = None) -> EvidenceBundle:
    cfg=config or {}; n=len(frames)
    if len(quality.frames)!=n or len(control.scores)!=n or len(live_dead.live_score)!=n:
        raise ValueError('Evidence source lengths must match canonical timeline')
    # Scores are additive *utilities* for Viterbi, not true normalized log-likelihoods.
    w_control=float(cfg.get('team_weight',1.0));w_live=float(cfg.get('live_weight',1.0))
    w_dead=float(cfg.get('dead_weight',1.0))
    normalization=cfg.get('control_normalization','none')
    if normalization not in ('none','max'):
        raise ValueError("control_normalization must be 'none' or 'max'")
    weighted_control=np.asarray(control.scores,dtype=float)*w_control
    if weighted_control.shape!=(n,2):
        raise ValueError('Team-control scores must have shape (N,2) in H,A order')
    if not np.isfinite(weighted_control).all():
        raise ValueError('Nonfinite team-control utilities')
    # F1 removes only the per-frame common level, preserving the weighted H-A gap.
    # Default 'none' reproduces the original additive fusion.
    offset=weighted_control.max(axis=1) if normalization=='max' else np.zeros(n,float)
    relative_control=weighted_control-offset[:,None]
    live_utility=w_live*live_dead.live_score
    h=relative_control[:,0]+live_utility
    a=relative_control[:,1]+live_utility
    d=w_dead*live_dead.dead_score+float(cfg.get('dead_bias',0))
    scores=np.column_stack((h,a,d))
    components=pd.DataFrame({'frame':frames.frame.to_numpy(int),
        'weighted_control_H':weighted_control[:,0], 'weighted_control_A':weighted_control[:,1],
        'common_offset':offset, 'relative_control_H':relative_control[:,0],
        'relative_control_A':relative_control[:,1], 'live_utility':live_utility,
        'dead_utility':d, 'forced_d':np.zeros(n,dtype=bool)})
    # Hard-mask only legal publicly published match metadata, not inferred private events.
    for half, start, end in __import__('games').half_video_blocks('BAR-ATH'):
        if half==2:
            kickoff=int(cfg.get('second_half_kickoff_frame',113742))
            if not (start<=kickoff<=end): raise ValueError('Invalid second-half kickoff frame')
            scores[start:kickoff,:2]=-1e6
            scores[start:kickoff,2]=1e6
            components.loc[start:kickoff-1,'forced_d']=True
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite fused scores: check NaN guards in evidence models')
    return EvidenceBundle(scores, control.transition_hint.copy(),
                          events.frames.stoppage_anchor.to_numpy(float).copy(),
                          events.frames.restart_anchor.to_numpy(float).copy(),
                          {'score_semantics':'uncalibrated additive utilities','order':STATE_ORDER,
                           'control_normalization':normalization, 'control_order':('H','A'),
                           'control_components_semantics':'weighted channels before forced-D override'},
                          components)
