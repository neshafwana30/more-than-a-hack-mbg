"""Possession from team shape when the ball is not visible.

The ball is missing on ~41% of graded frames, mostly in long stretches (median ~11 s), so a
possession model that needs the ball is blind for a large part of the match. Players, however,
are almost always tracked, and team shape says a lot about who has the ball: the team in
possession pushes up, stretches wide and long, and moves forward; the other team drops and
compacts.

This is a SELF-SUPERVISED model, which RULES.md allows explicitly ("pseudo-labels produced by an
algorithm, provided no person decided any of them"):

  1. On frames where the ball IS visible, the geometric control detector (src/control.py) names
     the team in controlled possession. Those algorithmic labels are the training targets.
  2. A logistic regression learns P(home | team-shape features) from them. Features use players
     only, so the model applies unchanged on frames without the ball.
  3. Its log-odds become weak evidence for the possession decoder everywhere, and the main
     evidence where the ball is missing.

No ground truth, public or private, and no human label is read anywhere in this module.
"""
from dataclasses import dataclass
import warnings

import numpy as np

from .data_loader import Tracking
from .motion import Motion, rolling_mean

FEATURE_NAMES = [
    "centroid_forward_m",   # how far up its attacking direction the team is
    "front_player_m",       # most advanced player
    "back_line_m",          # deepest outfield-ish player (10th percentile)
    "length_m",             # depth of the team
    "width_m",              # spread across the pitch
    "speed_m_s",            # mean running speed
    "forward_velocity_m_s", # mean speed towards the opponent goal
]


@dataclass
class ShapeModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray  # last entry is the intercept
    train_frames: int
    cross_half_agreement: list[float]


def nan_percentile_rows(values: np.ndarray, q: float) -> np.ndarray:
    """Row-wise np.nanpercentile (linear interpolation), vectorised; NaN for empty rows."""
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    ordered = np.sort(np.where(valid, values, np.inf), axis=1)
    pos = (q / 100.0) * np.maximum(count - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
    rows = np.arange(values.shape[0])
    low, high = ordered[rows, lo], ordered[rows, hi]
    with np.errstate(invalid="ignore"):
        out = low + (pos - lo) * np.where(hi > lo, high - low, 0.0)
    out[count == 0] = np.nan
    return out


def _team_features(tracking: Tracking, motion: Motion, team: int, window: int) -> np.ndarray:
    pitch = np.asarray(tracking.pitch, dtype=np.float64)
    members = tracking.teams == team
    direction = motion.attack_sign.astype(np.float64) * (1.0 if team == 0 else -1.0)
    x = (tracking.players_xy[:, members, 0].astype(np.float64) - 0.5) * pitch[0]
    y = tracking.players_xy[:, members, 1].astype(np.float64) * pitch[1]
    fwd = x * direction[:, None]
    speed = motion.player_speed[:, members].astype(np.float64)
    vx = np.vstack([np.zeros((1, fwd.shape[1])), np.diff(fwd, axis=0)]) * tracking.fps
    vx[~np.isfinite(speed)] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames are expected
        cols = [
            np.nanmean(fwd, axis=1),
            np.nanmax(fwd, axis=1),
            nan_percentile_rows(fwd, 10),
            nan_percentile_rows(fwd, 90) - nan_percentile_rows(fwd, 10),
            nan_percentile_rows(y, 90) - nan_percentile_rows(y, 10),
            np.nanmean(speed, axis=1),
            np.nanmean(vx, axis=1),
        ]
    return np.column_stack([rolling_mean(c, window) for c in cols])


def shape_features(tracking: Tracking, motion: Motion, cfg: dict) -> np.ndarray:
    """Home-minus-away differences of each team-shape feature, shape (n, len(FEATURE_NAMES))."""
    window = int(cfg.get("feature_window_frames", tracking.fps))
    home = _team_features(tracking, motion, 0, window)
    away = _team_features(tracking, motion, 1, window)
    return home - away


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, iterations: int = 50) -> np.ndarray:
    """Newton-Raphson logistic regression with an L2 penalty (intercept unpenalised)."""
    design = np.column_stack([x, np.ones(len(x))])
    w = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], l2)
    penalty[-1] = 0.0
    for _ in range(iterations):
        z = np.clip(design @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = design.T @ (p - y) + penalty * w
        hess = (design * (p * (1 - p))[:, None]).T @ design + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def fit_shape_model(features: np.ndarray, targets: np.ndarray, halves_mask: list[np.ndarray],
                    cfg: dict) -> ShapeModel:
    """targets: 1 = home, 0 = away, -1 = no pseudo-label on that frame."""
    stride = int(cfg.get("train_stride_frames", 5))
    l2 = float(cfg.get("l2", 1.0))
    usable = (targets >= 0) & np.all(np.isfinite(features), axis=1)
    idx = np.flatnonzero(usable)[::stride]
    if idx.size < 100:
        raise RuntimeError("Too few pseudo-labelled frames to fit the team-shape model")
    mean = features[idx].mean(axis=0)
    std = features[idx].std(axis=0) + 1e-6
    z = (features - mean) / std

    agreement = []
    for held_out in halves_mask:
        train = idx[~held_out[idx]]
        test = idx[held_out[idx]]
        if train.size < 100 or test.size < 100:
            continue
        w = _fit_logistic(z[train], targets[train].astype(np.float64), l2)
        pred = (np.column_stack([z[test], np.ones(test.size)]) @ w) > 0
        agreement.append(float(np.mean(pred == (targets[test] == 1))))

    weights = _fit_logistic(z[idx], targets[idx].astype(np.float64), l2)
    return ShapeModel(mean, std, weights, int(idx.size), agreement)


def shape_log_odds(model: ShapeModel, features: np.ndarray, clip: float) -> np.ndarray:
    z = (features - model.mean) / model.std
    finite = np.all(np.isfinite(z), axis=1)
    out = np.zeros(len(z), dtype=np.float64)
    out[finite] = np.column_stack([z[finite], np.ones(finite.sum())]) @ model.weights
    return np.clip(out, -clip, clip)
