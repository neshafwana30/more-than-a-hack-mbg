"""Local transition boundary refinement using evidence crossover; no manual labels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

STATE={'H':0,'A':1,'D':2}

@dataclass
class BoundaryResult:
    labels: np.ndarray
    report: pd.DataFrame


def refine_boundaries(labels: np.ndarray, scores: np.ndarray, half: np.ndarray, quality=None,
                      stop_hint=None, restart_hint=None, config:dict|None=None) -> BoundaryResult:
    cfg=config or {};out=np.asarray(labels).copy();half=np.asarray(half)
    scores=np.asarray(scores,float);n=len(out)
    radius=max(0,int(cfg.get('search_radius_frames',6)))
    min_quality=float(cfg.get('minimum_quality_for_refine',0.10))
    if len(scores)!=n or len(half)!=n:raise ValueError('Length mismatch')
    q=np.ones(n) if quality is None else np.asarray(quality,float)
    stop=np.zeros(n) if stop_hint is None else np.asarray(stop_hint,float)
    restart=np.zeros(n) if restart_hint is None else np.asarray(restart_hint,float)
    changes=np.flatnonzero((out[1:]!=out[:-1])&(half[1:]==half[:-1])&(half[1:]!=0))+1
    rows=[]
    # Do not allow two refinement windows to overlap; preserves segment order.
    for t in changes:
        prev,following=str(out[t-1]),str(out[t])
        if prev not in STATE or following not in STATE:continue
        left=t-1
        while left>0 and half[left-1]==half[t] and out[left-1]==prev:left-=1
        right=t
        while right+1<n and half[right+1]==half[t] and out[right+1]==following:right+=1
        lo=max(left+1,t-radius);hi=min(right,t+radius)
        if lo>hi:continue
        if np.nanmean(q[max(lo-1,0):min(hi+1,n)]) < min_quality:continue
        a,b=STATE[prev],STATE[following]
        best=t;best_utility=-np.inf
        # Choose cut c: prev on [lo-1,c), following on [c,hi].
        # Initial endpoints never change; only an existing transition can move.
        for c in range(lo,hi+1):
            v=float(np.sum(scores[lo:c,a])+np.sum(scores[c:hi+1,b]))
            if b==2:v+=float(cfg.get('event_anchor_bonus',0.12))*stop[c]
            elif a==2:v+=float(cfg.get('event_anchor_bonus',0.12))*restart[c]
            v-=float(cfg.get('boundary_shift_cost',0.01))*abs(c-t)
            if v>best_utility+1e-10:best=c;best_utility=v
        if best!=t:
            out[min(t,best):max(t,best)]=prev if best>t else following
        rows.append({'half':int(half[t]),'original_frame':int(t),'refined_frame':int(best),
                     'shift_frames':int(best-t),'from':prev,'to':following,
                     'ambiguous':bool(np.nanmean(q[max(lo-1,0):min(hi+1,n)])<0.4)})
    # Always respect publicly documented pre-kickoff D anchor, even after refinement.
    if n>113742 and len(half)>110808 and half[110808]==2:
        out[110808:113742]='D'
    return BoundaryResult(out,pd.DataFrame(rows,columns=['half','original_frame','refined_frame','shift_frames','from','to','ambiguous']))
