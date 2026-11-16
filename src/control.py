"""Who is controlling the ball, from geometry alone, and when it counts as controlled.

The challenge defines a change of possession as the opponent gaining *controlled* possession, and
lists what does not count: a touch without control, a deflection that does not intercept a pass,
an aerial duel won without the team then keeping the ball. It also suggests two proxies:

    * a player keeps the ball for a moment            -> a long enough touch episode
    * the team completes a pass                       -> two consecutive episodes of the same
                                                         team with the ball travelling between them

This module turns those proxies into per-frame evidence for the possession decoder:

    * a frame inside a CONTROLLED episode carries strong evidence for that team
    * a frame inside an uncontrolled touch carries only weak evidence

so one deflection cannot flip a possession, but a dribble or a completed pass can, and the switch
lands where the controlling episode began (the decoder is offline and sees it).
"""
from dataclasses import dataclass

import numpy as np

from .motion import Motion


@dataclass
class Episode:
    start: int
    end: int          # exclusive
    team: int         # 0 = H, 1 = A
    controlled: bool = False
    reason: str = ""


def touch_owner(motion: Motion, cfg: dict) -> np.ndarray:
    """Per-frame 0 (H), 1 (A) or -1 (nobody clearly on the ball)."""
    radius = float(cfg["own_radius_m"])
    margin = float(cfg["own_margin_m"])
    flight = float(cfg["in_flight_speed_m_s"])
    h, a = motion.nearest_h, motion.nearest_a
    owner = np.full(len(h), -1, dtype=np.int8)
    owner[(h <= radius) & (h + margin <= a)] = 0
    owner[(a <= radius) & (a + margin <= h)] = 1
    with np.errstate(invalid="ignore"):
        flying = np.nan_to_num(motion.ball_speed, nan=0.0) > flight
    owner[flying | motion.ball_out | ~motion.ball_known] = -1
    return owner


def episodes_from_owner(owner: np.ndarray, max_gap: int) -> list[Episode]:
    result: list[Episode] = []
    idx = np.flatnonzero(owner >= 0)
    if idx.size == 0:
        return result
    start = prev = int(idx[0])
    team = int(owner[start])
    for t in idx[1:]:
        t = int(t)
        if int(owner[t]) == team and t - prev - 1 <= max_gap:
            prev = t
            continue
        result.append(Episode(start, prev + 1, team))
        start = prev = t
        team = int(owner[t])
    result.append(Episode(start, prev + 1, team))
    return result


def mark_controlled(episodes: list[Episode], ball_xy: np.ndarray, pitch: tuple[float, float],
                    dead: np.ndarray, cfg: dict, fps: int) -> list[Episode]:
    keep = int(round(float(cfg["control_min_s"]) * fps))
    pass_min = float(cfg["pass_min_distance_m"])
    pass_gap = int(round(float(cfg["pass_max_gap_s"]) * fps))
    scale = np.asarray(pitch, dtype=np.float64)
    for ep in episodes:
        if ep.end - ep.start >= keep:
            ep.controlled, ep.reason = True, "kept"
    # Completed pass: the same team touches it again somewhere else, nobody else in between,
    # and the ball did not go dead on the way.
    for first, second in zip(episodes[:-1], episodes[1:]):
        if first.team != second.team or second.start - first.end > pass_gap:
            continue
        if dead[first.end:second.start].any():
            continue
        travel = np.linalg.norm((ball_xy[second.start] - ball_xy[first.end - 1]) * scale)
        if np.isfinite(travel) and travel >= pass_min:
            for ep in (first, second):
                if not ep.controlled:
                    ep.controlled, ep.reason = True, "pass"
    return episodes


def control_evidence(motion: Motion, ball_xy: np.ndarray, pitch: tuple[float, float],
                     dead: np.ndarray, cfg: dict, fps: int) -> tuple[np.ndarray, list[Episode]]:
    """Per-frame home log-odds from touches (positive = H)."""
    owner = touch_owner(motion, cfg)
    owner[dead] = -1
    eps = episodes_from_owner(owner, int(cfg["episode_max_gap_frames"]))
    eps = mark_controlled(eps, ball_xy, pitch, dead, cfg, fps)
    evidence = np.zeros(len(owner), dtype=np.float64)
    strong = float(cfg["controlled_weight"])
    weak = float(cfg["touch_weight"])
    for ep in eps:
        sign = 1.0 if ep.team == 0 else -1.0
        evidence[ep.start:ep.end] += sign * (strong if ep.controlled else weak)
    return evidence, eps
