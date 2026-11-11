"""Audit predicted possession switches without reading competition ground truth.

From possession-model root:
  python -m experiments.transition_evidence --prediction outputs/viterbi.csv \
      --config config_viterbi.yaml --data-dir data \
      --output outputs/viterbi_turnovers.csv

Use --predictions-only to skip tracking and evidence loading.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loader import input_paths, load_tracking
from src.ball_recovery import recover_ball
from src.geometry import geometry_evidence
from src.tactical_features import read_events, tactical_evidence
from src.dead_ball import estimate_dead_probability
from src.fusion import fuse_evidence


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["frame", "label"])
    if df.empty or df.frame.isna().any() or df.frame.duplicated().any():
        raise ValueError("Prediction must contain unique, non-null frame IDs")
    if not pd.api.types.is_integer_dtype(df.frame):
        raise ValueError("frame column must contain integer IDs")
    df = df.sort_values("frame").reset_index(drop=True)
    frames = df.frame.to_numpy()
    if not np.array_equal(frames, np.arange(len(df))):
        raise ValueError("Prediction frames must be contiguous and 0-based")
    if not df.label.isin(["H", "A", "D", "X"]).all():
        raise ValueError("Unexpected label in prediction")
    return df


def build_evidence(config: dict, data_dir: Path, n: int):
    meta, raw, smart = input_paths(data_dir)
    fps = int(config["match"]["fps"])
    pitch = (float(config["match"]["pitch_length_m"]),
             float(config["match"]["pitch_width_m"]))
    tracking = load_tracking(meta, raw, n, fps, pitch)
    ball = recover_ball(tracking.ball_xy, fps, pitch, **config["ball_recovery"])
    geom = geometry_evidence(tracking, ball, config["geometry"])
    events = read_events(smart, fps)
    tactical = tactical_evidence(events, n, config["tactical"])
    dead = estimate_dead_probability(
        tracking, ball, tactical, config["dead_ball"],
        config["fusion"]["base_dead_probability"],
        config["fusion"]["max_dead_probability"])
    probs = fuse_evidence(geom, tactical, dead, config["fusion"])
    return ball, geom, tactical, probs, events


def mean_finite(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return round(float(a.mean()), 4) if a.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prediction", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output", type=Path, default=Path("outputs/transition_evidence.csv"))
    ap.add_argument("--window-seconds", type=float, default=1.0)
    ap.add_argument("--event-window-seconds", type=float, default=2.0)
    ap.add_argument("--predictions-only", action="store_true",
                    help="Only calculate switch/run structure; do not load tracking or XML")
    args = ap.parse_args()
    if args.window_seconds <= 0 or args.event_window_seconds < 0:
        ap.error("Window sizes must be positive (event window may be zero)")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fps = int(config["match"]["fps"])
    df = read_predictions(args.prediction)
    labels = df.label.to_numpy()
    n = len(labels)
    if n != int(config["match"]["frames"]):
        raise ValueError(f"CSV has {n} frames, config expects {config['match']['frames']}")
    print(f"Loaded {n:,} predictions; computing transition evidence...")
    evidence = None if args.predictions_only else build_evidence(config, args.data_dir, n)
    if evidence is not None:
        ball, geom, tactical, probs, events = evidence
    window = max(1, round(args.window_seconds * fps))
    event_window = round(args.event_window_seconds * fps)

    # Contiguous runs. A turnover is a DIRECT H->A or A->H boundary.
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    run_labels = labels[starts]
    rows = []
    for k in range(1, len(starts)):
        old, new = str(run_labels[k - 1]), str(run_labels[k])
        if {old, new} != {"H", "A"}:
            continue
        t = int(starts[k])
        left, right = max(0, t - window), min(n, t + window)
        prev_dur = (ends[k - 1] - starts[k - 1]) / fps
        next_dur = (ends[k] - starts[k]) / fps
        # A-B-A is a *candidate* noisy interruption, never assumed false.
        sandwich = (k + 1 < len(starts) and run_labels[k + 1] == old)
        row = {
            "frame": t, "time_s": round(t / fps, 3),
            "from": old, "to": new,
            "previous_spell_s": round(float(prev_dur), 3),
            "new_spell_s": round(float(next_dur), 3),
            "returns_to_previous_team": bool(sandwich),
            "short_sandwich_under_2s": bool(sandwich and next_dur < 2),
            "new_spell_start": t, "new_spell_end_exclusive": int(ends[k]),
        }
        if evidence is not None:
            # Probabilities are engineered emission scores, NOT calibrated truths.
            new_idx = 0 if new == "H" else 1
            old_idx = 0 if old == "H" else 1
            before = slice(left, t)
            after = slice(t, right)
            logratio = np.log(np.maximum(probs[:, new_idx], 1e-7)) - np.log(
                np.maximum(probs[:, old_idx], 1e-7))
            lo = max(0, t - event_window)
            hi = min(n, t + event_window)
            nearby_events = [e for e in events if e["start"] < hi and e["end"] > lo]
            codes = sorted({e["code"] for e in nearby_events})
            row.update({
                "p_new_before": mean_finite(probs[before, new_idx]),
                "p_new_after": mean_finite(probs[after, new_idx]),
                "p_old_after": mean_finite(probs[after, old_idx]),
                "p_dead_after": mean_finite(probs[after, 2]),
                "log_evidence_new_minus_old_before": mean_finite(logratio[before]),
                "log_evidence_new_minus_old_after": mean_finite(logratio[after]),
                "ball_observed_before_pct": round(float(ball.observed[before].mean() * 100), 2),
                "ball_observed_after_pct": round(float(ball.observed[after].mean() * 100), 2),
                "ball_available_after_pct": round(float((ball.confidence[after] > 0).mean() * 100), 2),
                "geometry_reliability_after": mean_finite(geom.reliability[after]),
                "nearest_H_m_after": mean_finite(geom.nearest_h_m[after]),
                "nearest_A_m_after": mean_finite(geom.nearest_a_m[after]),
                "tactical_log_odds_H_after": mean_finite(tactical.log_odds_home[after]),
                "set_piece_nearby": bool(tactical.set_piece[lo:hi].any()),
                "nearby_event_count": len(nearby_events),
                "nearby_event_codes": " | ".join(codes),
            })
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        pd.DataFrame(columns=["frame", "time_s", "from", "to"]).to_csv(args.output, index=False)
    print(f"Direct H/A switches: {len(rows)}")
    print(f"Short A-B-A/H-A-H candidates (<2s): {sum(r['short_sandwich_under_2s'] for r in rows)}")
    print(f"Saved: {args.output}")
    print("Note: candidates are NOT verified false turnovers; no ground truth was read.")


if __name__ == "__main__":
    main()
