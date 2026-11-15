"""Uncalibrated H/A team-control support. Distance != controlled possession."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .control_acquisition import acquisition_support


@dataclass
class ControlResult:
    frames: pd.DataFrame
    scores: np.ndarray    # (N,2), column order H,A
    transition_hint: np.ndarray
    components: pd.DataFrame | None = None
    metadata: dict | None = None


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


def estimate_team_control(frames: pd.DataFrame, quality: pd.DataFrame, config: dict | None = None, fps: float = 25, pitch=(105.3,68.0)) -> ControlResult:
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
    legacy_scores=scores.copy()
    gates,components,metadata=acquisition_support(frames,quality,cfg.get('acquisition',{}),fps,pitch)
    winner_gate=np.where(hs>aa,gates[:,0],np.where(aa>hs,gates[:,1],np.maximum(gates[:,0],gates[:,1])))
    # Select status for the same legacy winning team used by the A1 gate.
    choose_h=np.where(legacy_scores[:,0]>legacy_scores[:,1],True,
                      np.where(legacy_scores[:,1]>legacy_scores[:,0],False,gates[:,0]>=gates[:,1]))
    components['legacy_winner']=np.where(legacy_scores[:,0]==legacy_scores[:,1],'tie',np.where(choose_h,'H','A'))
    for stem in ['primary_reason','evidence_status']+[f'fail_{k}' for k in metadata['reason_precedence'] if k!='supported']:
        components[f'winner_{stem}']=np.where(choose_h,components[f'{stem}_h'],components[f'{stem}_a'])
    components['winner_tie']=legacy_scores[:,0]==legacy_scores[:,1]
    if metadata['acquisition']['enabled']:
        # Contract the H/A gap only. No incumbent state, hysteresis, forced winner,
        # new transition discount, or extra Live evidence is introduced.
        center=legacy_scores.mean(axis=1)
        scores=center[:,None]+(legacy_scores-center[:,None])*winner_gate[:,None]
        hs=scores[:,0];aa=scores[:,1];gap=np.abs(hs-aa)
    a1_scores=scores.copy()
    bonus=np.zeros_like(legacy_scores)
    bonus_mode=metadata['acquisition']['enabled'] and metadata['acquisition']['application']=='bonus'
    if bonus_mode:
        scores=legacy_scores.copy()
        bonus=float(metadata['acquisition']['bonus_weight'])*gates
        for side in (0,1):
            active=bonus[:,side]>0
            scores[active,side]+=bonus[active,side]
        hs=scores[:,0];aa=scores[:,1];gap=np.abs(hs-aa)
    fallback=(metadata['acquisition']['enabled'] and metadata['acquisition']['fallback_unavailable']) & components.winner_evidence_status.eq('unavailable').to_numpy()
    effective_gate=np.ones(n,float) if bonus_mode else np.where(fallback,1.0,winner_gate)
    if np.any(fallback):
        # Restore the exact existing quality-weighted scores. Do not fabricate
        # acquisition support, retain an incumbent state or add a new prior.
        scores[fallback]=legacy_scores[fallback]
        hs=scores[:,0];aa=scores[:,1];gap=np.abs(hs-aa)
    components['a1_h_score']=a1_scores[:,0];components['a1_a_score']=a1_scores[:,1]
    components['fallback_applied']=fallback;components['effective_gate']=effective_gate
    components['fallback_restored_gap']=np.where(fallback,np.maximum(0,np.abs(scores[:,0]-scores[:,1])-np.abs(a1_scores[:,0]-a1_scores[:,1])),0.0)
    components['bonus_h']=bonus[:,0];components['bonus_a']=bonus[:,1]
    components['bonus_gap']=bonus[:,0]-bonus[:,1]
    components['legacy_h_score']=legacy_scores[:,0];components['legacy_a_score']=legacy_scores[:,1]
    components['distance_h']=ball_weight*ball_quality*close_h
    components['distance_a']=ball_weight*ball_quality*close_a
    components['tag_h']=tag_weight*tag_h;components['tag_a']=tag_weight*tag_a
    components['persistence_h']=persist_weight*ph*ball_quality
    components['persistence_a']=persist_weight*pa*ball_quality
    components['signed_margin_h']=margin_weight*observed*ball_quality*advantage
    components['signed_margin_a']=-components['signed_margin_h']
    components['symmetric_prior']=0.04;components['winner_gate']=winner_gate
    components['adjusted_h_score']=scores[:,0];components['adjusted_a_score']=scores[:,1]
    components['legacy_transition_hint']=hint
    metadata['application']='contract legacy H/A gap by support of the legacy winning team; preserve midpoint and framewise order; legacy transition hint unchanged'
    if bonus_mode:metadata['application']='F1 legacy scores plus bounded qualified co-motion support per team; no contraction or fallback; legacy transition hint unchanged'
    metadata['bonus_semantics']='bonus_i = bonus_weight * qualified acquisition_gate_i; zero without current valid cue; equal H/A bonus cancels relative preference after F1 max-centering; effective_gate=1 in bonus mode is a legacy coefficient, not support'
    metadata['fallback_semantics']='optional exact legacy score restoration only when selected legacy-winner status is unavailable; raw support and winner_gate unchanged; effective_gate is the applied coefficient, not confidence; disabled/outside_half/observed_not_supporting_proxy never trigger fallback'
    metadata['disabled']='exact legacy scores and hints; acquisition gates are 1 (bypass), not measured support'
    df=pd.DataFrame({'frame':frames.frame.to_numpy(int),'h_control_score':hs,'a_control_score':aa,
                     'control_margin':gap,'proxy_persistence_h':ph,'proxy_persistence_a':pa,
                     'controlled_candidate':np.maximum(close_h,close_a)*ball_quality})
    return ControlResult(df,scores,hint,components,metadata)
