"""Conservative short-gap ball interpolation; never hallucinate long gaps."""
from dataclasses import dataclass

import numpy as np


@dataclass
class BallEstimate:
    xy: np.ndarray       # normalized pitch coordinates
    confidence: np.ndarray  # 1 observed, <1 interpolated, 0 unavailable
    observed: np.ndarray


def recover_ball(ball_xy: np.ndarray, fps: float, pitch: tuple[float, float],
                 max_gap_frames: int = 12, max_average_speed_m_s: float = 32.0,
                 interpolated_confidence: float = 0.48) -> BallEstimate:
    xy = np.asarray(ball_xy, dtype=np.float32).copy()
    observed = np.isfinite(xy).all(axis=1)
    confidence = observed.astype(np.float32)
    valid_idx = np.flatnonzero(observed)
    for left, right in zip(valid_idx[:-1], valid_idx[1:]):
        gap = right - left - 1
        if not (0 < gap <= max_gap_frames):
            continue
        distance = np.linalg.norm((xy[right] - xy[left]) * np.asarray(pitch))
        average_speed = distance * fps / (right - left)
        if average_speed > max_average_speed_m_s:
            continue
        frac = np.arange(1, gap + 1, dtype=np.float32) / float(gap + 1)
        xy[left + 1:right] = xy[left] + frac[:, None] * (xy[right] - xy[left])
        confidence[left + 1:right] = interpolated_confidence
    return BallEstimate(xy=xy, confidence=confidence, observed=observed)
