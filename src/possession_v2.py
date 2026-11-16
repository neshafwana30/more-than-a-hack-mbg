"""Controlled-possession pipeline ("decoder.method: controlled_possession").

    tracking ─► ball recovery ─► motion features
                                   │
                                   ├─► live / dead segmentation ............ src/live_dead.py
                                   │       (D where the ball is out of play)
                                   │
                                   ├─► touches ─► controlled episodes ...... src/control.py
                                   │       (strong evidence where the ball is visible)
                                   │
                                   ├─► team-shape model, self-supervised ... src/shape_model.py
                                   │       (evidence where the ball is missing)
                                   │
    Smart Tagging ─────────────────┴─► team phases + set-piece restarts ... src/tactical_features.py
                                   │
                                   ▼
            per live stretch: two-state (H/A) Viterbi with a switch penalty
                                   │
                                   ▼
            H / A / D inside the two halves, X outside them (never graded)

Every live stretch between two stoppages is decoded on its own, so a dead ball always ends a
possession and the team that restarts is decided by what happens after the restart. Inside a
stretch, the switch penalty is what makes a possession a possession: the evidence for the other
team has to add up to more than the penalty before the label changes, which one deflection
cannot do and a controlled dribble or a completed pass can.
"""
from dataclasses import dataclass, field

import numpy as np

from .ball_recovery import BallEstimate
from .control import Episode, control_evidence
from .data_loader import Tracking
from .live_dead import detect_dead_ball, two_state_viterbi
from .motion import Motion, compute_motion, runs
from .shape_model import ShapeModel, fit_shape_model, shape_features, shape_log_odds
from .tactical_features import TacticalEvidence


@dataclass
class Diagnostics:
    dead_fraction_graded: float = 0.0
    n_stoppages: int = 0
    n_episodes: int = 0
    n_controlled: int = 0
    shape: ShapeModel | None = None
    n_live_stretches: int = 0
    notes: list[str] = field(default_factory=list)


def _restart_prior(events: list[dict], live: list[tuple[int, int]], team_map: dict[str, str],
                   n: int, cfg: dict, fps: int) -> np.ndarray:
    """Set pieces name the restarting team: nudge the first seconds after the matching restart."""
    prior = np.zeros(n, dtype=np.float64)
    weight = float(cfg["restart_team_weight"])
    span = int(round(float(cfg["restart_team_window_s"]) * fps))
    lookback = int(round(float(cfg["restart_match_before_s"]) * fps))
    lookahead = int(round(float(cfg["restart_match_after_s"]) * fps))
    normalized = {" ".join(k.casefold().split()): v for k, v in team_map.items()}
    starts = np.asarray([s for s, _ in live], dtype=np.int64)
    if starts.size == 0:
        return prior
    for event in events:
        if str(event["code"]).strip().upper() != "SET PIECES":
            continue
        side = normalized.get(" ".join(str(event["tags"].get("team", "")).casefold().split()))
        if side not in ("H", "A"):
            continue
        tag = int(event["start"])
        candidates = np.flatnonzero((starts >= tag - lookback) & (starts <= tag + lookahead))
        if candidates.size == 0:
            continue
        start = int(starts[candidates[0]])
        prior[start:min(n, start + span)] += weight if side == "H" else -weight
    return prior


def run_controlled_possession(tracking: Tracking, ball: BallEstimate, events: list[dict],
                              tactical: TacticalEvidence, config: dict,
                              halves: list[tuple[int, int, int]]
                              ) -> tuple[np.ndarray, Diagnostics]:
    n = tracking.n_frames
    fps = tracking.fps
    cfg = config["possession_v2"]
    diag = Diagnostics()

    blocks = [(first, last + 1) for _, first, last in halves]
    graded = np.zeros(n, dtype=bool)
    for first, last in blocks:
        graded[first:last] = True

    motion: Motion = compute_motion(tracking, ball, halves, cfg["motion"])

    # 1. Live / dead.
    dead, _ = detect_dead_ball(motion, events, blocks, cfg["dead_ball"], fps)
    live_mask = graded & ~dead
    live = runs(live_mask)
    diag.dead_fraction_graded = float(dead[graded].mean())
    diag.n_stoppages = len(runs(graded & dead))
    diag.n_live_stretches = len(live)

    # 2. Touches and controlled episodes (ball visible only).
    touch, episodes = control_evidence(motion, ball.xy, tracking.pitch, dead | ~graded,
                                       cfg["control"], fps)
    diag.n_episodes = len(episodes)
    diag.n_controlled = sum(ep.controlled for ep in episodes)

    # 3. Self-supervised team-shape model, trained on the controlled episodes.
    features = shape_features(tracking, motion, cfg["shape"])
    targets = np.full(n, -1, dtype=np.int8)
    for ep in episodes:
        if ep.controlled:
            targets[ep.start:ep.end] = 1 if ep.team == 0 else 0
    targets[~live_mask] = -1
    half_masks = []
    for first, last in blocks:
        m = np.zeros(n, dtype=bool)
        m[first:last] = True
        half_masks.append(m)
    model = fit_shape_model(features, targets, half_masks, cfg["shape"])
    diag.shape = model
    shape = shape_log_odds(model, features, float(cfg["shape"]["clip"]))

    # 4. Combine evidence (positive = home).
    weights = cfg["evidence"]
    shape_weight = np.where(motion.ball_known, float(weights["shape_when_ball_visible"]),
                            float(weights["shape_when_ball_missing"]))
    tactical_weight = np.where(motion.ball_known, float(weights["tactical_when_ball_visible"]),
                               float(weights["tactical_when_ball_missing"]))
    evidence = (touch
                + shape_weight * shape
                + tactical_weight * tactical.log_odds_home
                + _restart_prior(events, live, config["tactical"].get("team_map", {}), n,
                                 cfg["restart"], fps))

    # 5. Decode each live stretch separately; a stoppage always ends a possession.
    labels = np.full(n, "X", dtype="<U1")
    labels[graded & dead] = "D"
    penalty = float(cfg["decoder"]["team_switch_penalty"])
    for first, last in live:
        home = two_state_viterbi(evidence[first:last], penalty)
        labels[first:last] = np.where(home, "H", "A")
    return labels, diag


def describe(diag: Diagnostics, episodes: list[Episode] | None = None) -> list[str]:
    lines = [
        f"dead ball on {diag.dead_fraction_graded:.1%} of graded frames "
        f"({diag.n_stoppages} stoppages, {diag.n_live_stretches} live stretches)",
        f"touch episodes: {diag.n_episodes:,}, controlled: {diag.n_controlled:,}",
    ]
    if diag.shape is not None:
        agree = ", ".join(f"{a:.1%}" for a in diag.shape.cross_half_agreement) or "n/a"
        lines.append(f"team-shape model: {diag.shape.train_frames:,} pseudo-labelled frames; "
                     f"agreement with the detector on the held-out half: {agree}")
    return lines
