"""Uncalibrated H/A team-control support. Distance != controlled possession."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ControlResult:
    frames: pd.DataFrame
    scores: np.ndarray    # (N,2), column order H,A
    transition_hint: np.ndarray


def _rolling_proxy(proxy: np.ndarray, half: np.ndarray, window: int) -> tuple[np.ndarray,np.ndarray]:
    n=len(proxy); h=np.zeros(n,float);a=np.zeros(n,float)
    width=max(1,int(window))|1  # odd, so centered window is well-defined
    kernel=np.ones(width,float)
    for hv in (1,2):
        ix=np.flatnonzero(half==hv)
        if not len(ix):continue
        x=proxy[ix]
        oh=(x=='H').astype(float);oa=(x=='A').astype(float)
        pad=width//2
        denom=np.convolve(np.pad(oh+oa,(pad,pad)),kernel,'valid')
        hs=np.convolve(np.pad(oh,(pad,pad)),kernel,'valid')
        ass=np.convolve(np.pad(oa,(pad,pad)),kernel,'valid')
        denom=np.maximum(denom,1)
        h[ix]=hs/denom; a[ix]=ass/denom
    return h,a


def estimate_team_control(frames: pd.DataFrame, quality: pd.DataFrame, config: dict | None = None) -> ControlResult:
    cfg=config or {};n=len(frames)
    scale=max(0.1,float(cfg.get('distance_scale_m',2.5)))
    d_h=frames.nearest_h_m.to_numpy(float);d_a=frames.nearest_a_m.to_numpy(float)
    ball_quality=quality.ball_weight.to_numpy(float)
    proxy=frames.nearest_team_proxy.to_numpy(str)
    half=frames.half.to_numpy(int)
    close_h=np.exp(-np.maximum(d_h,0)/scale)
    close_a=np.exp(-np.maximum(d_a,0)/scale)
    close_h=np.nan_to_num(close_h,nan=0,posinf=0,neginf=0)
    close_a=np.nan_to_num(close_a,nan=0,posinf=0,neginf=0)
    margin=np.nan_to_num(d_a-d_h,nan=0,posinf=0,neginf=0)
    # Saturating geometric margin; not an independent certainty claim.
    advantage=np.tanh(margin/5.0)
    observed=(np.isfinite(d_h)&np.isfinite(d_a)).astype(float)
    tag_h=np.clip(frames.tag_h_support.to_numpy(float),0,1)
    tag_a=np.clip(frames.tag_a_support.to_numpy(float),0,1)
    ph,pa=_rolling_proxy(proxy,half,int(cfg.get('proximity_window_frames',11)))
    ball_weight=float(cfg.get('distance_weight',1.0))
    tag_weight=float(cfg.get('tag_weight',0.3))
    persist_weight=float(cfg.get('persistence_weight',0.5))
    margin_weight=float(cfg.get('margin_weight',0.25))
    # When ball is absent geometry is neutral; tag context can still inform a weak prior.
    hs=ball_weight*ball_quality*close_h + tag_weight*tag_h + persist_weight*ph*ball_quality + margin_weight*observed*ball_quality*advantage
    aa=ball_weight*ball_quality*close_a + tag_weight*tag_a + persist_weight*pa*ball_quality - margin_weight*observed*ball_quality*advantage
    # Broad symmetric prior: no arbitrary bias to home/away when both have no observations.
    hs+=0.04;aa+=0.04
    scores=np.column_stack([hs,aa])
    gap=np.abs(hs-aa)
    # Change cue from sustained evidence crossover; only an aid to decoding, never a forced label.
    winner=np.where(hs>aa,1,np.where(aa>hs,-1,0))
    win_window=max(1,int(cfg.get('transition_window_frames',9)))
    hint=np.zeros(n)
    for hh in (1,2):
        ix=np.flatnonzero(half==hh)
        if len(ix)<2:continue
        width=win_window|1
        sm=np.convolve(np.pad(hs[ix]-aa[ix],(width//2,width//2),mode='edge'),np.ones(width)/width,'valid')
        left=np.r_[sm[0],sm[:-1]]
        changed=(sm*left<0)&(np.abs(sm-left)>0.01)
        hint[ix]=np.where(changed, np.minimum(1.0, np.abs(sm-left)),0.0)
    df=pd.DataFrame({'frame':frames.frame.to_numpy(int),'h_control_score':hs,'a_control_score':aa,
                     'control_margin':gap,'proxy_persistence_h':ph,'proxy_persistence_a':pa,
                     'controlled_candidate':np.maximum(close_h,close_a)*ball_quality})
    return ControlResult(df,scores,hint)
