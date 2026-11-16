"""Ball-in-play vs dead-ball segmentation.

A dead ball is a stoppage, not a possession, so getting these stretches right does two jobs at
once: it fills the `D` class, and it cuts possessions where the ground truth cuts them
(`H H D D H H` is two possessions, not one).

Evidence per frame is a log-odds score, positive = dead. Every weight is an engineered prior with
a football reason next to it; none of them was fitted to labels of the provided game.

    * players walking, not running        stoppages are when 22 people slow down together
    * ball clearly outside the pitch       and it stays dead while the tracker has lost it there
    * ball stationary                      a placed ball (free kick, goal kick, corner)
    * ball moving fast inside the pitch    play is on
    * ball lost for a long stretch         weak: the tracker also loses it during live play
    * Smart Tagging SET PIECES start       the tag starts while the restart is being set up

The per-frame scores are decoded with a two-state Viterbi (switching costs a fixed penalty), then
runs shorter than a minimum duration are absorbed by their neighbours: a real stoppage lasts
seconds, and a real stretch of play between two stoppages lasts more than a moment.
"""
import numpy as np

from .motion import Motion, runs


def _extend_through_missing(start_mask: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Carry a True frame forward through the ball-missing frames that immediately follow it."""
    out = start_mask.copy()
    carrying = False
    for t in range(len(out)):
        if start_mask[t]:
            carrying = True
        elif carrying and missing[t]:
            out[t] = True
        else:
            carrying = False
    return out


def dead_log_odds(motion: Motion, events: list[dict], cfg: dict, fps: int) -> np.ndarray:
    n = len(motion.team_speed)
    score = np.full(n, float(cfg["prior_log_odds"]), dtype=np.float64)

    # Collective pace. Walking pace is ~1.3 m/s; during play the mean is clearly above it.
    team_speed = np.nan_to_num(motion.team_speed, nan=float(cfg["walking_speed_m_s"]))
    pace = (float(cfg["walking_speed_m_s"]) - team_speed) / float(cfg["speed_scale_m_s"])
    score += float(cfg["slow_players_weight"]) * np.clip(pace, -2.0, 2.0)

    missing = ~motion.ball_known
    out = _extend_through_missing(motion.ball_out, missing)
    score[out] += float(cfg["ball_out_weight"])

    known = motion.ball_known & np.isfinite(motion.ball_speed)
    still = known & ~motion.ball_out & (motion.ball_speed < float(cfg["still_ball_speed_m_s"]))
    score[still] += float(cfg["still_ball_weight"])
    fast = known & ~motion.ball_out & (motion.ball_speed > float(cfg["fast_ball_speed_m_s"]))
    score[fast] -= float(cfg["fast_ball_weight"])

    long_missing = motion.missing_run >= int(cfg["long_missing_frames"])
    score[long_missing] += float(cfg["long_missing_weight"])

    before = int(round(float(cfg["set_piece_before_s"]) * fps))
    after = int(round(float(cfg["set_piece_after_s"]) * fps))
    for event in events:
        if str(event["code"]).strip().upper() != "SET PIECES":
            continue
        start = int(event["start"])
        score[max(0, start - before):min(n, start + after)] += float(cfg["set_piece_weight"])
    return score


def two_state_viterbi(log_odds: np.ndarray, penalty: float) -> np.ndarray:
    """MAP 0/1 sequence for per-frame log-odds of state 1, with a cost per switch."""
    n = len(log_odds)
    if n == 0:
        return np.zeros(0, dtype=bool)
    # Emission scores up to a constant: state 1 gets +lo/2, state 0 gets -lo/2.
    half = 0.5 * np.asarray(log_odds, dtype=np.float64)
    back = np.zeros((n, 2), dtype=np.int8)
    s0, s1 = -half[0], half[0]
    for t in range(1, n):
        stay0, move0 = s0, s1 - penalty
        stay1, move1 = s1, s0 - penalty
        if stay0 >= move0:
            new0 = stay0
        else:
            new0, back[t, 0] = move0, 1
        if stay1 >= move1:
            new1, back[t, 1] = stay1, 1
        else:
            new1 = move1
        s0, s1 = new0 - half[t], new1 + half[t]
    state = np.empty(n, dtype=np.int8)
    state[-1] = 1 if s1 > s0 else 0
    for t in range(n - 1, 0, -1):
        state[t - 1] = back[t, state[t]]
    return state.astype(bool)


def absorb_short_runs(state: np.ndarray, min_true: int, min_false: int) -> np.ndarray:
    """Flip runs that are too short to be real, shortest first, until none remain."""
    out = np.asarray(state, dtype=bool).copy()
    for _ in range(len(out)):
        short = []
        for start, end in runs(out) + runs(~out):
            limit = min_true if out[start] else min_false
            # Never touch the first/last run: we cannot see its true extent.
            if start > 0 and end < len(out) and end - start < limit:
                short.append((end - start, start, end))
        if not short:
            break
        _, start, end = min(short)
        out[start:end] = not out[start]
    return out


def detect_dead_ball(motion: Motion, events: list[dict], blocks: list[tuple[int, int]],
                     cfg: dict, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (dead mask, per-frame log-odds). Each block is decoded independently."""
    score = dead_log_odds(motion, events, cfg, fps)
    dead = np.zeros(len(score), dtype=bool)
    min_dead = int(round(float(cfg["min_dead_s"]) * fps))
    min_live = int(round(float(cfg["min_live_s"]) * fps))
    for first, last in blocks:
        state = two_state_viterbi(score[first:last], float(cfg["switch_penalty"]))
        dead[first:last] = absorb_short_runs(state, min_dead, min_live)
    return dead, score
