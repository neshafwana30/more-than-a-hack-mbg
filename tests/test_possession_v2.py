"""Synthetic tests for the controlled-possession pipeline. No competition data or labels are used."""
from pathlib import Path

import numpy as np
import yaml

from main import ROOT, run
from src.control import Episode, episodes_from_owner, mark_controlled
from src.live_dead import absorb_short_runs, two_state_viterbi
from src.shape_model import _fit_logistic, nan_percentile_rows


# --------------------------------------------------------------------------- unit tests

def test_two_state_viterbi_ignores_a_blip_but_follows_a_sustained_change():
    lo = np.full(200, -1.0)
    lo[50:53] = 2.0          # 3 frames of weak contrary evidence: below the penalty
    lo[120:] = 1.0           # sustained change
    state = two_state_viterbi(lo, penalty=8.0)
    assert not state[:120].any()
    assert state[121:].all()


def test_absorb_short_runs_removes_flicker_but_keeps_long_runs():
    s = np.zeros(300, dtype=bool)
    s[100:110] = True        # 10-frame "stoppage": too short
    s[200:280] = True        # 80-frame stoppage: real
    out = absorb_short_runs(s, min_true=50, min_false=50)
    assert not out[100:110].any()
    assert out[200:280].all()


def test_single_touch_is_not_control_but_a_dribble_and_a_pass_are():
    fps = 25
    cfg = {"control_min_s": 0.5, "pass_min_distance_m": 5.0, "pass_max_gap_s": 4.0}
    ball = np.zeros((400, 2))
    ball[:, 0] = np.linspace(0.1, 0.9, 400)
    dead = np.zeros(400, dtype=bool)
    eps = [
        Episode(10, 13, team=1),     # 3-frame deflection by A, nothing follows
        Episode(50, 70, team=0),     # 0.8 s dribble by H -> kept
        Episode(100, 104, team=0),   # quick touch by H ...
        Episode(140, 143, team=0),   # ... and another H touch far away -> completed pass
    ]
    out = mark_controlled(eps, ball, (105.3, 68.0), dead, cfg, fps)
    assert [e.controlled for e in out] == [False, True, True, True]
    assert out[1].reason == "kept" and out[2].reason == "pass"


def test_pass_across_a_stoppage_does_not_count():
    ball = np.zeros((200, 2))
    ball[:, 0] = np.linspace(0.1, 0.9, 200)
    dead = np.zeros(200, dtype=bool)
    dead[60:90] = True
    eps = [Episode(40, 43, team=0), Episode(100, 103, team=0)]
    cfg = {"control_min_s": 0.5, "pass_min_distance_m": 5.0, "pass_max_gap_s": 4.0}
    out = mark_controlled(eps, ball, (105.3, 68.0), dead, cfg, 25)
    assert not any(e.controlled for e in out)


def test_episodes_bridge_small_gaps_only():
    owner = np.array([-1, 0, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 1, 1], dtype=np.int8)
    eps = episodes_from_owner(owner, max_gap=2)
    assert [(e.start, e.end, e.team) for e in eps] == [(1, 6, 0), (12, 13, 0), (13, 15, 1)]


def test_nan_percentile_rows_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 23))
    a[rng.random(a.shape) < 0.3] = np.nan
    a[7] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for q in (10, 50, 90):
            np.testing.assert_allclose(nan_percentile_rows(a, q),
                                       np.nanpercentile(a, q, axis=1), equal_nan=True)


def test_logistic_recovers_the_sign_of_a_clear_feature():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 2))
    y = (x[:, 0] + 0.3 * rng.normal(size=2000) > 0).astype(float)
    w = _fit_logistic(x, y, l2=1.0)
    assert w[0] > 2.0 and abs(w[1]) < 0.5


# --------------------------------------------------------------------------- integrated

def _synthetic_match(tmp_path: Path, n: int = 2000, seed: int = 0):
    """Two 6-player teams jogging; H dribbles, then A wins the ball, then the ball goes out."""
    rng = np.random.default_rng(seed)
    folder = tmp_path / "tactical"
    folder.mkdir()
    players = "".join(f'<Player id="{i}" teamId="{"h" if i <= 6 else "a"}"/>' for i in range(1, 13))
    (folder / "BAR-ATH_tactical_atd.xml").write_text(
        '<main><Teams><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team>'
        f"</Teams><Players>{players}</Players></main>", encoding="utf-8")
    home = np.column_stack([np.full(6, 0.35), np.linspace(0.2, 0.8, 6)])
    away = np.column_stack([np.full(6, 0.65), np.linspace(0.2, 0.8, 6)])
    lines = []
    for f in range(n):
        t = f / 25.0
        wobble = 0.02 * np.sin(t * 2.0 + np.arange(6))[:, None]   # ~3 m/s jogging
        h = home + wobble + rng.normal(scale=0.001, size=home.shape)
        a = away - wobble + rng.normal(scale=0.001, size=away.shape)
        if f < 900:                     # H player 0 dribbles the ball
            ball = h[0] + [0.005, 0.0]
        elif f < 1700:                  # A player 0 has it
            ball = a[0] + [0.005, 0.0]
        else:                           # ball out over the touchline, everyone walks
            ball = np.array([0.5, 1.05])
            h, a = home.copy(), away.copy()
        triples = [f"{x:.5f},{y:.5f},0" for x, y in np.vstack([h, a])]
        lines.append(f"{f + 1}:" + ";".join(triples) + f":{ball[0]:.5f},{ball[1]:.5f},1")
    (folder / "BAR-ATH_tactical_atd.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "BAR-ATH_tactical_smart_tagging.xml").write_text(
        "<file><ALL_INSTANCES></ALL_INSTANCES></file>", encoding="utf-8")


def test_controlled_possession_end_to_end(tmp_path):
    n = 2000
    _synthetic_match(tmp_path, n)
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["match"]["frames"] = n
    config["match"]["halves"] = [[1, 100, 1999]]
    config["possession_v2"]["shape"]["train_stride_frames"] = 1
    labels, _ = run(config, tmp_path)
    assert (labels[:100] == "X").all()            # outside the half: never graded
    assert (labels[200:850] == "H").mean() > 0.95
    assert (labels[950:1650] == "A").mean() > 0.95
    assert (labels[1760:] == "D").mean() > 0.95    # ball out, players walking
    # one possession each, no flicker
    live = labels[100:1700]
    assert np.count_nonzero(live[1:] != live[:-1]) <= 2
