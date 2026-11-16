"""Vectorized ball-to-team geometry with confidence-weighted evidence."""
from dataclasses import dataclass

import numpy as np
from .data_loader import Tracking
from .ball_recovery import BallEstimate


@dataclass
class GeometryEvidence:
    log_odds_home: np.ndarray
    reliability: np.ndarray
    nearest_h_m: np.ndarray
    nearest_a_m: np.ndarray
    nearby_h: np.ndarray
    nearby_a: np.ndarray


def geometry_evidence(tracking: Tracking, ball: BallEstimate, cfg: dict,
                      chunk_size: int = 4096) -> GeometryEvidence:
    n = tracking.n_frames
    h_dist = np.full(n, np.inf, dtype=np.float32)
    a_dist = np.full(n, np.inf, dtype=np.float32)
    nearby_h = np.zeros(n, dtype=np.int16)
    nearby_a = np.zeros(n, dtype=np.int16)
    pitch = np.asarray(tracking.pitch, dtype=np.float32)
    radius = float(cfg["nearby_radius_m"])
    for first in range(0, n, chunk_size):
        last = min(n, first + chunk_size)
        p = tracking.players_xy[first:last]
        b = ball.xy[first:last]
        with np.errstate(invalid="ignore"):
            delta = (p - b[:, None, :]) * pitch
            distance = np.sqrt(np.sum(delta * delta, axis=2))
        distance[~np.isfinite(p).all(axis=2)] = np.inf
        distance[~np.isfinite(b).all(axis=1)] = np.inf
        if np.any(tracking.teams == 0):
            ds = distance[:, tracking.teams == 0]
            h_dist[first:last] = ds.min(axis=1)
            nearby_h[first:last] = (ds <= radius).sum(axis=1)
        if np.any(tracking.teams == 1):
            ds = distance[:, tracking.teams == 1]
            a_dist[first:last] = ds.min(axis=1)
            nearby_a[first:last] = (ds <= radius).sum(axis=1)

    # Score is positive for H, negative for A. A large nearest-player distance
    # weakens evidence: an airborne ball isn't owned by the closest defender.
    good = (np.isfinite(h_dist) & np.isfinite(a_dist) & (ball.confidence > 0))
    log_odds = np.zeros(n, dtype=np.float32)
    reliability = np.zeros(n, dtype=np.float32)
    separation = (a_dist[good] - h_dist[good]) / max(float(cfg["contrast_scale_m"]), 0.1)
    log_odds[good] = np.clip(separation, -4.0, 4.0)
    closest = np.minimum(h_dist[good], a_dist[good])
    scale = max(float(cfg["max_reliable_distance_m"]), 0.1)
    reliability[good] = (np.exp(-0.5 * (closest / scale) ** 2)
                         * ball.confidence[good])
    return GeometryEvidence(log_odds, reliability, h_dist, a_dist, nearby_h, nearby_a)
