import numpy as np
from src.observation_quality import analyze_observations
from src.team_control import estimate_team_control


def test_control_geometry_not_hard_radius(fake_bundle):
    f=fake_bundle.frames; quality=analyze_observations(f)
    r=estimate_team_control(f,quality.frames)
    assert r.scores.shape==(len(f),2)
    assert r.scores[0,0]>r.scores[0,1]
    assert r.scores[-1,1]>r.scores[-1,0]
    assert np.isfinite(r.scores).all()
    # During missing gap the two team scores remain finite; no fake ball position.
    assert np.isfinite(r.scores[60]).all()


def test_motion_bad_jump_is_quality_gated(fake_bundle):
    f=fake_bundle.frames.copy()
    f.loc[15,'ball_speed2d_mps']=110.
    q=analyze_observations(f)
    assert q.frames.loc[15,'suspected_ball_jump']
    assert q.frames.loc[15,'ball_weight']<q.frames.loc[14,'ball_weight']
    assert q.frames.loc[60,'gap_class']=='short'
