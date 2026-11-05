import numpy as np
import pandas as pd
from src.visual_analyzer import analyze_video_windows,image_to_pitch,project_pitch_to_image
from src.ambiguity_detector import coalesce_windows,select_ambiguities
from src.observation_quality import analyze_observations
from src.team_control import estimate_team_control


def test_homography_roundtrip_and_invalid():
    h=np.array([[100,0,50],[0,80,20],[0,0,1.]])
    p=project_pitch_to_image(h,.3,.4)
    assert p is not None
    xy=image_to_pitch(h,*p)
    assert np.allclose(xy,(.3,.4))
    assert image_to_pitch(np.zeros((3,3)),50,50) is None


def test_video_gracefully_skips_no_weights(fake_bundle):
    wins=pd.DataFrame([{'start_frame':5,'end_frame':10}])
    r=analyze_video_windows(wins,fake_bundle.frames,'missing.mp4','missing.csv')
    assert r.diagnostics['status']=='skipped'
    assert len(r.observations)==0


def test_event_centered_budget(fake_bundle):
    f=fake_bundle.frames;q=analyze_observations(f).frames
    control=estimate_team_control(f,q).frames
    pred=np.where(f.frame.to_numpy()<80,'H','A')
    a=select_ambiguities(f,q,control,None,pred,{'window_radius_frames':10,'max_windows':1,'max_total_video_frames':30})
    assert len(a.windows)<=1
    if len(a.windows):assert int(a.windows.iloc[0].end_frame-a.windows.iloc[0].start_frame+1)<=30
    assert coalesce_windows([(1,4),(3,5),(10,11)])==[(1,5),(10,11)]
