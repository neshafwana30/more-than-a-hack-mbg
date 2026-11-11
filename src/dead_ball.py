"""Conservative dead-ball probability from objective features, not missing-ball alone."""
import numpy as np

from .data_loader import Tracking
from .ball_recovery import BallEstimate
from .tactical_features import TacticalEvidence


def estimate_dead_probability(tracking: Tracking, ball: BallEstimate,
                              tactical: TacticalEvidence, cfg: dict,
                              base_probability: float = 0.04,
                              max_probability: float = 0.97) -> np.ndarray:
    n = tracking.n_frames
    if ball.observed.shape != (n,) or tactical.set_piece.shape != (n,):
        raise ValueError("Tracking, ball, and Smart Tagging evidence must align by frame")
    p = np.full(n, base_probability, dtype=np.float32)
    margin = float(cfg["out_of_bounds_margin"])
    observed = ball.observed
    # Coordinates outside [0,1] by a small tolerance are geometric evidence
    # that play has stopped, but not an absolute rule.
    outside = observed & np.any((ball.xy < -margin) | (ball.xy > 1 + margin), axis=1)
    p[outside] += float(cfg["out_of_bounds_boost"])
    p += (tactical.set_piece * (~observed)
          * float(cfg["set_piece_without_ball_boost"]))
    p += (tactical.set_piece * observed
          * float(cfg["set_piece_with_ball_boost"]))
    scarce = tracking.player_count < int(cfg["low_player_count_threshold"])
    p[scarce] += float(cfg["low_player_count_boost"])
    return np.clip(p, 1e-4, max_probability)
