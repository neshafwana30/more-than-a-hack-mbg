import numpy as np
from src.temporal_decoder import decode_states
from src.boundary_refinement import refine_boundaries


def test_weak_single_frame_flicker_suppressed():
    n=50;half=np.ones(n,int)
    scores=np.zeros((n,3));scores[:,0]=2.;scores[:,1]=0.
    scores[25,1]=2.3;scores[25,0]=2.
    r=decode_states(scores,half,config={'ha_cost':1,'hd_cost':1,'dh_cost':1,'short_spell_scale_frames':10})
    assert (r.labels=='H').all()


def test_strong_short_real_takeover_survives_soft_prior():
    n=50;half=np.ones(n,int)
    scores=np.zeros((n,3));scores[:,0]=2.;scores[:,1]=0.
    scores[20:23,0]=0.;scores[20:23,1]=10.
    r=decode_states(scores,half,config={'ha_cost':1,'hd_cost':1,'dh_cost':1,'short_spell_scale_frames':10})
    assert (r.labels[20:23]=='A').all()
    assert (r.labels[24:]=='H').all()


def test_dead_breaks_same_team_spells_and_restart_can_be_opponent():
    n=80;half=np.ones(n,int)
    scores=np.zeros((n,3))
    scores[:20,0]=10.;scores[20:40,2]=10.;scores[40:60,1]=10.;scores[60:,2]=10.
    r=decode_states(scores,half,config={'ha_cost':1,'hd_cost':1,'dh_cost':1})
    assert r.labels[19]=='H' and r.labels[25]=='D' and r.labels[45]=='A'


def test_halves_independent_and_outside_x():
    half=np.array([0,1,1,1,0,2,2,2,0])
    sc=np.zeros((len(half),3));sc[:,0]=1;sc[5:8,1]=10
    r=decode_states(sc,half,config={'ha_cost':0.3,'hd_cost':0.3,'dh_cost':0.3})
    assert r.labels[0]=='X' and r.labels[4]=='X' and r.labels[8]=='X'
    assert (r.labels[1:4]=='H').all() and (r.labels[5:8]=='A').all()


def test_boundary_refinement_uses_future_evidence():
    n=40;half=np.ones(n,int)
    labs=np.array(['H']*20+['A']*20)
    sc=np.zeros((n,3));sc[:17,0]=3;sc[17:,1]=3
    r=refine_boundaries(labs,sc,half,config={'search_radius_frames':6,'minimum_quality_for_refine':0})
    assert r.labels[17]=='A' and r.labels[16]=='H'


def test_all_finite_decoding_required():
    from pytest import raises
    with raises(ValueError):decode_states(np.array([[np.nan,0,1]]),np.array([1]))
