"""Offline Viterbi with state-dependent, event-conditioned transitions.

Viterbi is global over each half and therefore uses observations before and after
an ambiguous interval. An optional soft short-spell prior removes only weak flicker,
not all short spells. Never decodes across half boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

STATE_ORDER=np.array(['H','A','D'])

@dataclass
class DecodeResult:
    labels: np.ndarray
    transitions: pd.DataFrame
    diagnostics: dict


def state_cost_matrix(cfg: dict) -> np.ndarray:
    # Symmetric defaults are hypotheses, not trained on public GT.
    ha=float(cfg.get('ha_cost',2.4))
    hd=float(cfg.get('hd_cost',1.65))
    dh=float(cfg.get('dh_cost',1.9))
    mat=np.array([[0,ha,hd],[ha,0,hd],[dh,dh,0]],dtype=float)
    if not np.isfinite(mat).all() or (mat<0).any():raise ValueError('Transition costs must be finite >=0')
    return mat


def transition_costs(base: np.ndarray, turnover: float, stop: float, restart: float, cfg: dict) -> np.ndarray:
    mat=base.copy()
    mat[0,1]=mat[1,0]=max(float(cfg.get('minimum_transition_cost',0.05)),mat[0,1]-float(cfg.get('turnover_discount',0.8))*turnover)
    mat[0,2]=mat[1,2]=max(float(cfg.get('minimum_transition_cost',0.05)),mat[0,2]-float(cfg.get('stoppage_discount',0.7))*stop)
    mat[2,0]=mat[2,1]=max(float(cfg.get('minimum_transition_cost',0.05)),mat[2,0]-float(cfg.get('restart_discount',0.7))*restart)
    return mat


def _segments(states: np.ndarray) -> list[tuple[int,int,int]]:
    if len(states)==0:return []
    boundaries=np.r_[0,np.flatnonzero(states[1:]!=states[:-1])+1,len(states)]
    return [(int(l),int(r),int(states[l])) for l,r in zip(boundaries[:-1],boundaries[1:])]


def _soft_short_run_refine(states: np.ndarray, scores: np.ndarray, cfg: dict) -> tuple[np.ndarray,int]:
    """Merge a short sandwiched run only if its cumulative evidence is insufficient.

    The combined duration prior and two boundary costs are compared with the
    emission advantage of keeping the run. A genuinely strong brief control survives.
    This is a limited post-Viterbi prior, not an exact HSMM implementation.
    """
    scale=int(cfg.get('short_spell_scale_frames',12))
    soft=float(cfg.get('short_spell_soft_penalty',0.45))
    if scale<2 or soft<=0:return states,0
    states=states.copy()
    removed=0
    for left,right,s in _segments(states):
        if left==0 or right==len(states) or right-left>=scale:continue
        other=int(states[left-1])
        if other!=int(states[right]):continue
        # No hard minimum; require weak evidence for the middle run.
        advantage=float(np.sum(scores[left:right,s]-scores[left:right,other]))
        duration_penalty=soft*(1-(right-left)/scale)
        if advantage < duration_penalty:
            states[left:right]=other
            removed+=1
    return states,removed


def decode_states(scores: np.ndarray, half: np.ndarray, transition_hint=None, stop_hint=None, restart_hint=None,
                  config: dict | None = None) -> DecodeResult:
    cfg=config or {};score=np.asarray(scores,dtype=float);half=np.asarray(half,dtype=int)
    if score.shape!=(len(half),3) or not np.isfinite(score).all():raise ValueError('scores must be finite (N,3)')
    n=len(half)
    zero=np.zeros(n,float)
    turnover=zero if transition_hint is None else np.asarray(transition_hint,dtype=float)
    stop=zero if stop_hint is None else np.asarray(stop_hint,dtype=float)
    restart=zero if restart_hint is None else np.asarray(restart_hint,dtype=float)
    if any(x.shape!=(n,) or not np.isfinite(x).all() for x in (turnover,stop,restart)):
        raise ValueError('Invalid transition hint array')
    labels=np.full(n,'X',dtype='<U1')
    base=state_cost_matrix(cfg); removed=0
    for block in (1,2):
        ids=np.flatnonzero(half==block)
        if len(ids)==0:continue
        if np.any(np.diff(ids)!=1):raise ValueError('Each half must be a contiguous block')
        l=int(ids[0]);r=int(ids[-1])+1
        seq=score[l:r];m=len(seq)
        # Backpointer only 3 bytes per frame; local DP only 3 floats.
        back=np.empty((m,3),dtype=np.uint8)
        back[0]=np.arange(3,dtype=np.uint8)
        dp=seq[0].copy()
        for t in range(1,m):
            f=l+t
            costs=transition_costs(base,float(turnover[f]),float(stop[f]),float(restart[f]),cfg)
            candidates=dp[:,None]-costs
            indices=np.argmax(candidates,axis=0)
            back[t]=indices
            dp=seq[t]+candidates[indices,np.arange(3)]
        states=np.empty(m,dtype=np.uint8)
        states[-1]=np.argmax(dp)
        for t in range(m-1,0,-1):
            states[t-1]=back[t,states[t]]
        # Do not apply duration merging across forced published kickoff boundary.
        if block==2:
            k=int(cfg.get('second_half_kickoff_frame',113742))-l
            if 0<k<m:
                first, r1=_soft_short_run_refine(states[:k],seq[:k],cfg)
                second,r2=_soft_short_run_refine(states[k:],seq[k:],cfg)
                states=np.r_[first,second];removed+=r1+r2
            else:
                states,r1=_soft_short_run_refine(states,seq,cfg);removed+=r1
        else:
            states,r1=_soft_short_run_refine(states,seq,cfg);removed+=r1
        labels[l:r]=STATE_ORDER[states]
    transitions=[]
    for block in (1,2):
        ids=np.flatnonzero(half==block)
        if len(ids)<2:continue
        lo=int(ids[0]); hi=int(ids[-1])+1
        for t in np.flatnonzero(labels[lo+1:hi]!=labels[lo:hi-1])+lo+1:
            transitions.append({'half':block,'frame':int(t),'from':str(labels[t-1]),'to':str(labels[t])})
    df=pd.DataFrame(transitions,columns=['half','frame','from','to'])
    return DecodeResult(labels,df,{'short_runs_merged':removed,'transition_count':len(df)})
