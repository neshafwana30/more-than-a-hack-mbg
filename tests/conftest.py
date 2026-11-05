"""Small fully SYNTHETIC football observations: no user match data or labels."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.features import FeatureResult


def synthetic_bundle(n=160):
    frame=np.arange(n)
    half=np.ones(n,dtype=int)
    ball_ok=np.ones(n,dtype=bool);ball_ok[50:80]=False
    bx=np.linspace(0.2,0.8,n);by=np.full(n,.5)
    bx[~ball_ok]=np.nan;by[~ball_ok]=np.nan
    # A clear geometry switch from H to A around halfway through.
    hdist=np.where(frame<n//2,0.6,6.0).astype(float)
    adist=np.where(frame<n//2,6.0,0.6).astype(float)
    hdist[~ball_ok]=np.nan;adist[~ball_ok]=np.nan
    proxy=np.where(frame<n//2,'H','A').astype('<U1');proxy[~ball_ok]='?'
    run=np.zeros(n,dtype=int);run[50:80]=30
    since=np.zeros(n,float);until=np.zeros(n,float)
    data=pd.DataFrame({
        'frame':frame,'half':half,'graded':np.ones(n,dtype=bool),
        'ball_x':bx,'ball_y':by,'ball_ok':ball_ok,
        'ball_missing_run_frames':run,'frames_since_ball_seen':since,'frames_until_ball_seen':until,
        'ball_visibility_age_s':since/25,'ball_next_visibility_s':until/25,
        'home_count':np.full(n,11),'away_count':np.full(n,11),
        'valid_h_fraction':np.ones(n),'valid_a_fraction':np.ones(n),
        'nearest_h_m':hdist,'nearest_a_m':adist,
        'nearest_h_x':np.full(n,.4),'nearest_h_y':np.full(n,.5),
        'nearest_a_x':np.full(n,.6),'nearest_a_y':np.full(n,.5),
        'distance_margin_a_minus_h_m':adist-hdist,
        'abs_distance_margin_m':np.abs(adist-hdist),'nearest_team_proxy':proxy,
        'ball_speed2d_mps':np.full(n,3.0),'ball_displacement_m':np.full(n,.12),
        'ball_speed_quality':np.ones(n,dtype=bool),
        'centroid_speed_H_mps':np.full(n,2.0),'centroid_speed_A_mps':np.full(n,2.0),
        'player_motion_robust_h':np.full(n,2.0),'player_motion_robust_a':np.full(n,2.0),
        'centroid_valid_h':np.ones(n,dtype=bool),'centroid_valid_a':np.ones(n,dtype=bool),
        'tag_h_support':np.where(frame<n//2,1.,0.),
        'tag_a_support':np.where(frame>=n//2,1.,0.),
        'active_h_tags':np.where(frame<n//2,1,0),'active_a_tags':np.where(frame>=n//2,1,0),
        'active_tag_count':np.ones(n,dtype=int),
        'set_piece_tag':np.zeros(n,dtype=bool),'set_piece_start':np.zeros(n,dtype=bool),
        'set_piece_end':np.zeros(n,dtype=bool),'tag_set_piece_type':np.full(n,'unknown',dtype=object),
        'tag_set_piece_team':np.full(n,'?'), 'tag_boundary_distance_frames':np.full(n,np.nan),
        'video_recovered_ball':np.zeros(n,dtype=bool),
    })
    events=pd.DataFrame(columns=['start_frame','end_frame','code_clean','side'])
    return FeatureResult(data,events,pd.DataFrame({'slot':['1','2'],'side':['H','A']}))


@pytest.fixture
def fake_bundle():
    return synthetic_bundle()
