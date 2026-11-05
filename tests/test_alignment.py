import numpy as np
import games
from src.features import _run_features


def test_half_boundaries_and_raw_ids():
    blocks=games.half_video_blocks('BAR-ATH')
    assert blocks==[(1,11185,90651),(2,110808,190197)]
    assert games.video_to_id('BAR-ATH',11185)==1713242
    assert games.id_to_video('BAR-ATH',1812866)==110808
    assert games.video_to_id('BAR-ATH',90652) is None
    assert games.half_of_video_frame('BAR-ATH',199318) is None


def test_missing_runs_stop_at_half():
    half=np.array([0,1,1,1,0,2,2,2,0])
    present=np.array([False,True,False,False,False,False,False,True,False])
    r,s,u=_run_features(present,half)
    assert r.tolist()==[0,0,2,2,0,2,2,0,0]
    assert np.isnan(u[3]) and np.isnan(s[5])
    assert u[5]==2


def test_all_missing_half():
    r,s,u=_run_features(np.zeros(5,dtype=bool),np.ones(5,dtype=int))
    assert r.tolist()==[5]*5
    assert np.isnan(s).all() and np.isnan(u).all()
