"""Frame-level motion features shared by the dead-ball and possession models.

Everything here is derived from the automatic tactical ATD only. No label of any kind is read.

Conventions
-----------
* Positions are the ATD's normalised [0, 1] pitch coordinates; distances and speeds are converted
  to metres and m/s with the configured pitch size.
* Track slots are tracklets, not people, so every player feature is collective (per team).
* ``attack_sign`` is +1 on frames where the home team attacks towards x = 1, -1 where it attacks
  towards x = 0. It is measured from the data (which half of the pitch each team occupies on
  average in each half), not hard-coded.
"""
from dataclasses import dataclass
import warnings

import numpy as np

from .data_loader import Tracking
from .ball_recovery import BallEstimate


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred, NaN-aware moving average of a 1-D series. NaN where the window has no data."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("rolling_mean expects a 1-D series")
    if window <= 1:
        return x.copy()
    valid = np.isfinite(x)
    kernel = np.ones(int(window), dtype=np.float64)
    total = np.convolve(np.where(valid, x, 0.0), kernel, mode="same")
    count = np.convolve(valid.astype(np.float64), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count < 0.5] = np.nan
    return out


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Length of the True-run each frame belongs to (0 on False frames)."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape[0], dtype=np.int64)
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:end] = end - start
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of every True-run."""
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


@dataclass
class Motion:
    player_speed: np.ndarray      # (n, slots) m/s, NaN when unknown
    team_speed: np.ndarray        # (n,) rolling mean speed of all tracked players, m/s
    ball_speed: np.ndarray        # (n,) rolling ball speed, m/s, NaN when ball unknown
    ball_known: np.ndarray        # (n,) ball position available (observed or short interpolation)
    ball_out: np.ndarray          # (n,) ball known and clearly outside the pitch
    missing_run: np.ndarray       # (n,) length of the ball-missing run the frame belongs to
    dist_to_ball: np.ndarray      # (n, slots) metres, inf when unknown
    nearest_h: np.ndarray         # (n,) metres to nearest home player, inf when unknown
    nearest_a: np.ndarray         # (n,) metres to nearest away player, inf when unknown
    nearest_slot: np.ndarray      # (n,) slot of the nearest player overall, -1 when unknown
    attack_sign: np.ndarray       # (n,) +1 / -1, see module docstring


def attack_sign_per_frame(tracking: Tracking, halves: list[tuple[int, int, int]]) -> np.ndarray:
    """+1 where home attacks towards x=1. Measured per half from mean team positions."""
    n = tracking.n_frames
    sign = np.ones(n, dtype=np.int8)
    home = tracking.teams == 0
    away = tracking.teams == 1
    bounds = [(first, last + 1) for _, first, last in halves]
    for first, last in bounds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hx = np.nanmean(tracking.players_xy[first:last][:, home, 0])
            ax = np.nanmean(tracking.players_xy[first:last][:, away, 0])
        sign[first:last] = 1 if hx < ax else -1
    # Frames outside the halves inherit the nearest half's orientation (they are never graded).
    if bounds:
        sign[: bounds[0][0]] = sign[bounds[0][0]]
        for (_, end), (nxt, _) in zip(bounds[:-1], bounds[1:]):
            sign[end:nxt] = sign[end - 1]
        sign[bounds[-1][1]:] = sign[bounds[-1][1] - 1]
    return sign


def compute_motion(tracking: Tracking, ball: BallEstimate, halves: list[tuple[int, int, int]],
                   cfg: dict, chunk_size: int = 8192) -> Motion:
    n = tracking.n_frames
    fps = tracking.fps
    pitch = np.asarray(tracking.pitch, dtype=np.float32)
    max_speed = float(cfg.get("max_player_speed_m_s", 12.0))

    pos_m = tracking.players_xy * pitch
    step = np.linalg.norm(np.diff(pos_m, axis=0), axis=2) * fps
    player_speed = np.vstack([step[:1], step]).astype(np.float32)
    player_speed[~np.isfinite(player_speed) | (player_speed > max_speed)] = np.nan

    known = np.isfinite(player_speed)
    with np.errstate(invalid="ignore", divide="ignore"):
        frame_mean = (np.where(known, player_speed, 0.0).sum(axis=1)
                      / known.sum(axis=1))
    team_speed = rolling_mean(frame_mean, int(cfg.get("team_speed_window_frames", 2 * fps)))

    ball_known = ball.confidence > 0
    ball_m = np.where(ball_known[:, None], ball.xy, np.nan) * pitch
    lag = int(cfg.get("ball_speed_lag_frames", 5))
    raw_ball_speed = np.full(n, np.nan)
    raw_ball_speed[lag:] = np.linalg.norm(ball_m[lag:] - ball_m[:-lag], axis=1) * fps / lag
    ball_speed = rolling_mean(raw_ball_speed, int(cfg.get("ball_speed_window_frames", fps)))
    ball_speed[~ball_known] = np.nan

    margin = float(cfg.get("out_of_bounds_margin", 0.005))
    with np.errstate(invalid="ignore"):
        ball_out = ball_known & np.any((ball.xy < -margin) | (ball.xy > 1 + margin), axis=1)

    dist = np.full(tracking.players_xy.shape[:2], np.inf, dtype=np.float32)
    for first in range(0, n, chunk_size):
        last = min(n, first + chunk_size)
        with np.errstate(invalid="ignore"):
            delta = (tracking.players_xy[first:last] - ball.xy[first:last, None, :]) * pitch
            d = np.sqrt(np.sum(delta * delta, axis=2))
        d[~np.isfinite(d)] = np.inf
        d[~ball_known[first:last]] = np.inf
        dist[first:last] = d
    home = tracking.teams == 0
    nearest_h = dist[:, home].min(axis=1) if home.any() else np.full(n, np.inf)
    nearest_a = dist[:, ~home].min(axis=1) if (~home).any() else np.full(n, np.inf)
    nearest_slot = np.argmin(dist, axis=1).astype(np.int64)
    nearest_slot[~np.isfinite(dist.min(axis=1))] = -1

    return Motion(
        player_speed=player_speed,
        team_speed=team_speed,
        ball_speed=ball_speed,
        ball_known=ball_known,
        ball_out=ball_out,
        missing_run=run_lengths(~ball_known),
        dist_to_ball=dist,
        nearest_h=nearest_h,
        nearest_a=nearest_a,
        nearest_slot=nearest_slot,
        attack_sign=attack_sign_per_frame(tracking, halves),
    )
