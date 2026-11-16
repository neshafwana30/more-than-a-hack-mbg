"""Offline reliability-aware Viterbi decoder for H/A/D possession states.

Does not read labels or ground truth. Reliability must come from upstream
tracking/observation metadata, NOT from prediction labels.
"""
import numpy as np


def reliability_viterbi(
    probabilities: np.ndarray,
    cfg: dict,
    observation_reliability: np.ndarray,
) -> np.ndarray:
    """Decode 3-state emissions using observation-dependent transition costs.

    observation_reliability: 1D float array in [0, 1], aligned frame-for-frame
    with probabilities. 1 = high-confidence observation, 0 = unavailable.

    When all reliability values equal 1, this is equivalent to the original
    Viterbi decoder (up to floating point tie-breaking).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    r = np.asarray(observation_reliability, dtype=np.float64)

    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_frames, 3) for H/A/D")
    n = len(p)
    if r.shape != (n,):
        raise ValueError(f"reliability must have shape ({n},), got {r.shape}")
    if not np.all(np.isfinite(r)) or np.any((r < 0) | (r > 1)):
        raise ValueError("reliability must be finite and in [0, 1]")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if n == 0:
        return np.empty(0, dtype=np.int8)

    base_team = float(cfg["home_away_switch_penalty"])
    base_dead = float(cfg["play_dead_switch_penalty"])
    team_boost = float(cfg.get("unreliable_team_penalty_boost", 0.5))
    dead_boost = float(cfg.get("unreliable_dead_penalty_boost", 0.0))
    if min(base_team, base_dead, team_boost, dead_boost) < 0:
        raise ValueError("transition penalties and boosts must be nonnegative")

    log_p = np.log(np.clip(p, 1e-8, 1.0))
    backptr = np.zeros((n, 3), dtype=np.int8)
    previous = log_p[0].copy()
    state_ids = np.arange(3)

    for t in range(1, n):
        uncertainty = 1.0 - r[t]
        team_cost = base_team * (1.0 + team_boost * uncertainty)
        dead_cost = base_dead * (1.0 + dead_boost * uncertainty)
        transition = np.array(
            [[0.0, -team_cost, -dead_cost],
             [-team_cost, 0.0, -dead_cost],
             [-dead_cost, -dead_cost, 0.0]],
            dtype=np.float64,
        )
        scores = previous[:, None] + transition
        backptr[t] = np.argmax(scores, axis=0)
        previous = scores[backptr[t], state_ids] + log_p[t]

    result = np.empty(n, dtype=np.int8)
    result[-1] = int(np.argmax(previous))
    for t in range(n - 1, 0, -1):
        result[t - 1] = backptr[t, result[t]]
    return result
