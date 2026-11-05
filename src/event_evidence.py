"""Automated Smart Tagging anchors and weak geometric event cues.

Set-piece intervals are *context*, never a guaranteed dead/live ground truth.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class EventResult:
    frames: pd.DataFrame
    candidates: pd.DataFrame


def _localized_pulse(n: int, events: list[int], width: int, half: np.ndarray) -> np.ndarray:
    out=np.zeros(n,float)
    width=max(1,int(width))
    for f in events:
        if f<0 or f>=n or half[f]==0: continue
        lo=max(0,f-width);hi=min(n,f+width+1)
        idx=np.arange(lo,hi)
        vals=np.maximum(0.0,1.0-np.abs(idx-f)/width)
        sel=half[idx]==half[f]
        out[idx[sel]]=np.maximum(out[idx[sel]],vals[sel])
    return out


def detect_events(frames: pd.DataFrame, events: pd.DataFrame, config: dict | None = None) -> EventResult:
    cfg=config or {}
    n=len(frames); half=frames.half.to_numpy(int)
    start_positions=[];end_positions=[]; candidate=[]
    if len(events):
        for ev in events.itertuples(index=False):
            if str(ev.code_clean).upper() != 'SET PIECES': continue
            s=int(ev.start_frame); e=int(ev.end_frame)
            if 0<=s<n and half[s]!=0:
                start_positions.append(s)
                candidate.append({'frame':s,'event':'set_piece_start','side':str(ev.side),'strength':0.35,'source':'smart_tagging'})
            if 0<=e<n and half[e]!=0:
                end_positions.append(e)
                candidate.append({'frame':e,'event':'set_piece_end','side':str(ev.side),'strength':0.35,'source':'smart_tagging'})
    width=int(cfg.get('anchor_window_frames',25))
    start=_localized_pulse(n,start_positions,width,half)
    end=_localized_pulse(n,end_positions,width,half)
    bx=frames.ball_x.to_numpy(float);by=frames.ball_y.to_numpy(float)
    visible=frames.ball_ok.to_numpy(bool)
    # Ball coordinates outside normalized pitch may indicate out-of-play OR tracking noise.
    outside=visible & ( (bx < -0.015) | (bx > 1.015) | (by < -0.015) | (by > 1.015) )
    # Downweight isolated out-of-pitch observations by local 3-frame persistence.
    outside_support=np.convolve(outside.astype(float),np.ones(3)/3,mode='same')
    if n:
        outside_support[0]=float(outside[0]);outside_support[-1]=float(outside[-1])
    out=pd.DataFrame({
        'frame':frames.frame.to_numpy(int),
        'stoppage_anchor':start,
        'restart_anchor':end,
        'out_of_pitch_hint':outside_support,
        'is_set_piece_context':frames.set_piece_tag.to_numpy(bool),
    })
    candidates=pd.DataFrame(candidate,columns=['frame','event','side','strength','source'])
    return EventResult(out,candidates)
