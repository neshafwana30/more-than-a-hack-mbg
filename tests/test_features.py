from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.features import (_interval_counts,_robust_player_motion,_team_for_slots,
                          _ball_geometry,refresh_visual_ball,_tag_support)


def test_distance_normalized_to_metres_and_nan():
    bx=np.array([.5,np.nan]);by=np.array([.5,np.nan])
    px=np.array([[.5,.51],[.5,.51]]);py=np.array([[.5,.5],[.5,.5]])
    valid=np.ones((2,2),bool);side=np.array(['H','A'])
    d,_,_,_,_= _ball_geometry(bx,by,px,py,valid,side,np.ones(2,int),(105.3,68),25)
    assert d['nearest_h_m'][0]==0
    assert d['nearest_a_m'][0]==pytest.approx(1.053)
    assert np.isnan(d['nearest_h_m'][1])


def test_robust_motion_drops_identity_switch():
    x=np.array([[.1,.2,.3,.4],[.101,.201,.301,.99],[.102,.202,.302,.99]])
    y=np.full_like(x,.5);valid=np.ones_like(x,bool)
    motion,ok=_robust_player_motion(x,y,valid,np.ones(4,bool),np.ones(3,int),(105.3,68),25,max_speed=12,min_tracks=3)
    assert ok[1] and np.isfinite(motion[1]) and motion[1]<5
    assert not ok[0]


def test_metadata_mapping_disallows_guessing():
    teams=[SimpleNamespace(name='Barcelona',players=[SimpleNamespace(player_id='Track 1')]),
           SimpleNamespace(name='Atletic Club',players=[SimpleNamespace(player_id='Track 2')])]
    data=SimpleNamespace(metadata=SimpleNamespace(teams=teams))
    assert _team_for_slots(data,['1','2'])=={'1':'H','2':'A'}
    with pytest.raises(ValueError):_team_for_slots(data,['1','2','3'])


def test_tag_intervals_half_open_overlaps():
    events=pd.DataFrame({'start_frame':[1,2],'end_frame':[4,5]})
    assert _interval_counts(events,6).tolist()==[0,1,2,2,1,0]


def test_capped_tag_families_prevent_double_counting():
    events=pd.DataFrame({'start_frame':[0,0,0],'end_frame':[5,5,5],
                         'code_clean':['BUILD UP','PROGRESSION','BALL IN FINAL THIRD'],'side':['H']*3})
    scores=_tag_support(events,5)
    assert (scores['tag_h_support']==1.0).all()  # 2 families; progress counted once
    assert (scores['tag_a_support']==0.0).all()


def test_visual_recovery_recomputes_distances(fake_bundle):
    df=fake_bundle.frames.copy()
    p={'x':np.full((len(df),2),.5),'y':np.full((len(df),2),.5),
       'valid':np.ones((len(df),2),bool),'sides':np.array(['H','A'])}
    obs=pd.DataFrame([{'frame':55,'x':.5,'y':.5,'quality':.9}])
    recovered=refresh_visual_ball(df,p,obs)
    assert recovered.at[55,'ball_ok'] and recovered.at[55,'video_recovered_ball']
    assert recovered.at[55,'nearest_h_m']==pytest.approx(0)
    assert recovered.at[54,'ball_missing_run_frames']>0


def test_build_features_with_mock_kloppy(monkeypatch):
    """Exercise actual file-to-feature glue with *entirely synthetic* kloppy input."""
    import sys
    from types import SimpleNamespace
    import games
    from src.features import build_features
    n=30
    raw=pd.DataFrame({'frame_id':np.arange(1,n+1),
                      'ball_x':np.linspace(.3,.6,n), 'ball_y':np.full(n,.5),
                      '1_x':np.linspace(.31,.61,n),'1_y':np.full(n,.5), '1_s':np.ones(n),
                      '2_x':np.full(n,.8),'2_y':np.full(n,.5),'2_s':np.ones(n)})
    teams=[SimpleNamespace(name='Barcelona',players=[SimpleNamespace(player_id='Track 1')]),
           SimpleNamespace(name='Athletic Club',players=[SimpleNamespace(player_id='Track 2')])]
    dataset=SimpleNamespace(metadata=SimpleNamespace(teams=teams),to_df=lambda:raw)
    tag=pd.DataFrame({'timestamp':[pd.Timedelta(seconds=.2)],'end_timestamp':[pd.Timedelta(seconds=.6)],
                      'code':['SET PIECES'],'Team':['Atletic Club'],'Type':['Throw-in']})
    source=SimpleNamespace(metrica=SimpleNamespace(load_tracking_epts=lambda **kwargs:dataset),
                           sportscode=SimpleNamespace(load=lambda _:SimpleNamespace(to_df=lambda:tag)))
    monkeypatch.setitem(sys.modules,'kloppy',source)
    monkeypatch.setattr(games,'require',lambda *a,**kw:[])
    monkeypatch.setattr(games,'asset_paths',lambda *a,**kw:{'atd_meta':'mock.xml','atd_raw':'mock.txt','smart_tagging':'mock_sports.xml'})
    monkeypatch.setattr(games,'source_frames',lambda **kw:n)
    monkeypatch.setattr(games,'half_video_blocks',lambda *a,**kw:[(1,0,14),(2,15,29)])
    bundle=build_features()
    f=bundle.frames
    assert len(f)==n and f.frame.iloc[-1]==29
    assert f.ball_ok.all()
    assert f.set_piece_tag.iloc[5]
    assert f.tag_set_piece_type.iloc[5]=='Throw-in'
    assert np.isfinite(f.nearest_h_m).all()
    assert not f.centroid_valid_h.any()  # only one synthetic track; correctly masked
