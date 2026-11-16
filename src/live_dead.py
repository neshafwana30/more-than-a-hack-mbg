"""Live/dead evidence, auditable score channels and optional D1 contextual preparation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class LiveDeadResult:
    frames: pd.DataFrame
    live_score: np.ndarray
    dead_score: np.ndarray
    components: pd.DataFrame | None = None
    metadata: dict | None = None


def _context_stoppage(frames, quality, event_frames, motion, valid_motion, config):
    """Causal, half-local evidence support; never a latched dead-ball label."""
    supplied=config.get('context_stoppage',{})
    if not isinstance(supplied,dict):
        raise ValueError('context_stoppage must be a mapping')
    cfg={'enabled':False,'window_frames':10,'minimum_support_fraction':0.8,
         'stationary_ball_speed_mps':0.6,'max_player_motion_mps':1.5,
         'minimum_player_quality':0.65,'weight':0.55}
    if set(supplied)-set(cfg):
        raise ValueError(f'Unknown context_stoppage settings: {sorted(set(supplied)-set(cfg))}')
    cfg.update(supplied)
    if not isinstance(cfg['enabled'],bool):
        raise ValueError('context_stoppage.enabled must be boolean')
    width=cfg['window_frames']
    if isinstance(width,bool) or not isinstance(width,int) or width<1:
        raise ValueError('context_stoppage.window_frames must be an integer >=1')
    for name in ('minimum_support_fraction','stationary_ball_speed_mps','max_player_motion_mps','minimum_player_quality','weight'):
        if not np.isfinite(float(cfg[name])):
            raise ValueError(f'context_stoppage.{name} must be finite')
    fraction=float(cfg['minimum_support_fraction'])
    if not 0<fraction<=1:
        raise ValueError('minimum_support_fraction must be in (0,1]')
    if not 0<=float(cfg['minimum_player_quality'])<=1:
        raise ValueError('minimum_player_quality must be in [0,1]')
    if any(float(cfg[k])<0 for k in ('stationary_ball_speed_mps','max_player_motion_mps','weight')):
        raise ValueError('context_stoppage speeds and weight must be >=0')
    n=len(frames);cue=np.zeros(n,bool);support=np.zeros(n,float);active=np.zeros(n,bool);bonus=np.zeros(n,float)
    if cfg['enabled']:
        half=frames.half.to_numpy(int)
        raw_speed=frames.ball_speed2d_mps.to_numpy(float)
        player_quality=quality.player_quality_weight.to_numpy(float)
        ball_weight=quality.ball_weight.to_numpy(float)
        both_motion=valid_motion.all(axis=1)
        cue=((half!=0) & event_frames.is_set_piece_context.to_numpy(bool)
             & quality.ball_observed.to_numpy(bool) & quality.ball_motion_reliable.to_numpy(bool)
             & np.isfinite(raw_speed) & (raw_speed>=0) & (raw_speed<=float(cfg['stationary_ball_speed_mps']))
             & both_motion & np.isfinite(motion) & (motion<=float(cfg['max_player_motion_mps']))
             & np.isfinite(player_quality) & (player_quality>=float(cfg['minimum_player_quality']))
             & np.isfinite(ball_weight) & (ball_weight>0))
        for h in (1,2):
            ids=np.flatnonzero(half==h)
            if not len(ids):continue
            if np.any(np.diff(ids)!=1):
                raise ValueError('Context evidence requires contiguous frame blocks per half')
            right=np.arange(1,len(ids)+1)
            left=np.maximum(0,right-width)
            cumulative=np.r_[0.,np.cumsum(cue[ids].astype(float))]
            support[ids]=(cumulative[right]-cumulative[left])/width
            # Full history inside this half, plus a valid CURRENT cue. Missing/motion
            # invalidity releases this contribution immediately; no state is carried.
            active[ids]=(right>=width) & cue[ids] & (support[ids]>=fraction)
        bonus=float(cfg['weight'])*support*np.clip(ball_weight,0,1)*active
    return cue,support,active,bonus,cfg


def estimate_live_dead(frames: pd.DataFrame, quality: pd.DataFrame, event_frames: pd.DataFrame, config: dict | None = None, fps=25, pitch=(105.3,68.0)) -> LiveDeadResult:
    cfg=config or {};n=len(frames)
    model=cfg.get('model','legacy')
    if model not in ('legacy','signed'):raise ValueError('live_dead.model must be legacy or signed')
    if model=='signed':
        if cfg.get('context_stoppage',{}).get('enabled',False):raise ValueError('Signed mode cannot combine with D1')
        from .game_state_evidence import estimate_signed_game_state
        return estimate_signed_game_state(frames,quality,event_frames,cfg.get('signed',{}),fps,pitch)
    good_speed=quality.ball_motion_reliable.to_numpy(bool)
    speed=np.nan_to_num(frames.ball_speed2d_mps.to_numpy(float),nan=0,posinf=0)
    player_motion=np.column_stack([frames.player_motion_robust_h.to_numpy(float),frames.player_motion_robust_a.to_numpy(float)])
    reliable=np.column_stack([frames.centroid_valid_h.to_numpy(bool),frames.centroid_valid_a.to_numpy(bool)])
    valid_motion=reliable & np.isfinite(player_motion)
    motion_sum=np.where(valid_motion,player_motion,0).sum(axis=1)
    motion_n=valid_motion.sum(axis=1)
    motion=np.divide(motion_sum,motion_n,out=np.zeros(n,float),where=motion_n>0)
    # Keep legacy operation order: disabling D1 preserves original numeric scores.
    live_prior=np.full(n,float(cfg.get('live_prior_score',0.55)))
    dead_prior=np.full(n,float(cfg.get('dead_prior_score',0.25)))
    ball_live=float(cfg.get('ball_moving_live_weight',0.22))*good_speed*np.clip(speed/5.0,0,1)
    player_live=float(cfg.get('player_motion_live_weight',0.12))*(motion_n>0)*np.clip(motion/3.0,0,1)
    outside_dead=float(cfg.get('out_of_pitch_weight',0.35))*event_frames.out_of_pitch_hint.to_numpy(float)*quality.ball_weight.to_numpy(float)
    stop_dead=float(cfg.get('stoppage_anchor_weight',0.32))*event_frames.stoppage_anchor.to_numpy(float)
    restart_dead=float(cfg.get('restart_anchor_weight',0.14))*event_frames.restart_anchor.to_numpy(float)
    near_anchor=(event_frames.stoppage_anchor.to_numpy(float)>0)|(event_frames.restart_anchor.to_numpy(float)>0)
    stationary_dead=float(cfg.get('stationary_near_anchor_weight',0.12))*near_anchor*good_speed*(speed<0.6)
    live=live_prior.copy();live+=ball_live;live+=player_live
    dead=dead_prior.copy();dead+=outside_dead;dead+=stop_dead;dead+=restart_dead;dead+=stationary_dead
    dead_without_context=dead.copy()
    cue,support,active,context_dead,context_cfg=_context_stoppage(frames,quality,event_frames,motion,valid_motion,cfg)
    if context_cfg['enabled']:
        dead+=context_dead
    df=pd.DataFrame({'frame':frames.frame.to_numpy(int),'live_score':live,'dead_score':dead,
                     'player_motion_evidence':motion,'has_player_motion':motion_n>0})
    components=pd.DataFrame({'frame':frames.frame.to_numpy(int),'live_prior':live_prior,
        'ball_motion_live':ball_live,'player_motion_live':player_live,'dead_prior':dead_prior,
        'out_of_pitch_dead':outside_dead,'stoppage_anchor_dead':stop_dead,
        'restart_anchor_dead':restart_dead,'stationary_near_anchor_dead':stationary_dead,
        'context_stoppage_dead':context_dead,'dead_without_context':dead_without_context,'preparation_cue':cue,'context_support_fraction':support,
        'context_active':active,'both_team_motion_valid':valid_motion.all(axis=1)})
    return LiveDeadResult(df,live,dead,components,{
        'score_semantics':'uncalibrated additive utilities; components are not probabilities',
        'live_additive_columns':['live_prior','ball_motion_live','player_motion_live'],
        'dead_additive_columns':['dead_prior','out_of_pitch_dead','stoppage_anchor_dead','restart_anchor_dead',
                                 'stationary_near_anchor_dead','context_stoppage_dead'],
        'nonadditive_reference':'dead_without_context is the exact pre-D1 score, not an additional term',
        'context_stoppage':context_cfg,'context_window':'trailing, half-local; full history required',
        'release':'current cue must remain valid; no label latch or hold-until-tag-end',
        'limitations':'broad automatic event context can include stationary live play; D1 remains an experimental proxy'})
