"""Budgeted event-centered candidate windows, not whole missing-ball intervals."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AmbiguityResult:
    windows: pd.DataFrame
    flags: pd.DataFrame


def coalesce_windows(windows: list[tuple[int,int]], max_span: int | None = None) -> list[tuple[int,int]]:
    out=[]
    for l,r in sorted(windows):
        if l>r:continue
        if out and l<=out[-1][1]+1 and (max_span is None or max(out[-1][1],r)-out[-1][0]+1<=max_span):
            out[-1]=(out[-1][0],max(out[-1][1],r))
        else:out.append((int(l),int(r)))
    return out


def select_ambiguities(frames: pd.DataFrame, quality: pd.DataFrame, control: pd.DataFrame,
                       evidence, preliminary_labels: np.ndarray, config: dict | None = None) -> AmbiguityResult:
    cfg=config or {};n=len(frames)
    half=frames.half.to_numpy(int)
    margin=control.control_margin.to_numpy(float)
    missing=~quality.ball_observed.to_numpy(bool)
    jump=quality.suspected_ball_jump.to_numpy(bool)
    q=quality.observation_weight.to_numpy(float)
    transitions=np.r_[False, (preliminary_labels[1:]!=preliminary_labels[:-1])&(half[1:]==half[:-1])&(half[1:]!=0)]
    lowmargin=(margin < float(cfg.get('margin_threshold',0.18)))
    potential=transitions | (jump & (half!=0))
    radius=int(cfg.get('window_radius_frames',35))
    # Only select high-impact intervals: changes, anomalies, near-tag candidate boundary.
    frames_list=[]
    for i in np.flatnonzero(potential):
        h=half[i]
        if not h:continue
        s=max(i-radius,int(np.flatnonzero(half==h)[0]))
        e=min(i+radius,int(np.flatnonzero(half==h)[-1]))
        risk=2.0*float(transitions[i])+float(jump[i])+float(missing[s:e+1].mean())+float(lowmargin[s:e+1].mean())
        risk+=float(np.mean(1-q[s:e+1]))
        frames_list.append((risk,s,e,int(i),'state_change' if transitions[i] else 'tracking_jump'))
    frames_list.sort(key=lambda x:(-x[0],x[3]))
    max_windows=max(0,int(cfg.get('max_windows',40)))
    max_frames=max(0,int(cfg.get('max_total_video_frames',3000)))
    selected=[];spent=0
    for risk,s,e,center,reason in frames_list:
        if len(selected)>=max_windows:break
        # Budget conservative: reject if already mostly covered by a selected window.
        overlaps=any(s<=e2 and e>=s2 for s2,e2, *_ in selected)
        if overlaps:continue
        size=e-s+1
        if spent+size>max_frames:continue
        selected.append((s,e,center,reason,float(risk)))
        spent+=size
    selected.sort()
    out=pd.DataFrame([{'start_frame':s,'end_frame':e,'center_frame':center,'reason':reason,'priority':risk}
                      for s,e,center,reason,risk in selected],
                     columns=['start_frame','end_frame','center_frame','reason','priority'])
    flags=pd.DataFrame({'frame':frames.frame.to_numpy(int),'suspected_transition':transitions,
                        'weak_control_margin':lowmargin,'ball_missing':missing,
                        'tracking_anomaly':jump})
    return AmbiguityResult(out,flags)
