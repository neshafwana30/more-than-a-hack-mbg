import numpy as np
import pandas as pd
from src.event_evidence import detect_events
from src.observation_quality import analyze_observations
from src.live_dead import estimate_live_dead


def test_tag_anchor_is_local_not_entire_interval(fake_bundle):
    f=fake_bundle.frames.copy()
    ev=pd.DataFrame([{'start_frame':20,'end_frame':100,'code_clean':'SET PIECES','side':'?'}])
    r=detect_events(f,ev,{'anchor_window_frames':3})
    assert r.frames.loc[20,'stoppage_anchor']==1
    assert r.frames.loc[100,'restart_anchor']==1
    assert r.frames.loc[60,'stoppage_anchor']==0
    assert r.frames.loc[60,'restart_anchor']==0
    q=analyze_observations(f)
    ld=estimate_live_dead(f,q.frames,r.frames)
    assert ld.dead_score[20]>ld.dead_score[60]
    assert ld.live_score[60]>0


def test_missing_ball_does_not_equal_dead(fake_bundle):
    f=fake_bundle.frames.copy()
    q=analyze_observations(f)
    ev=detect_events(f,pd.DataFrame(columns=['start_frame','end_frame','code_clean','side']))
    ld=estimate_live_dead(f,q.frames,ev.frames)
    assert ld.dead_score[55]==ld.dead_score[48]
    assert ld.live_score[55]>ld.dead_score[55]
