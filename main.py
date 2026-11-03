
"""Generate a reproducible full-match possession submission from licensed tactical ATD.

Examples:
    python main.py
    python main.py --config config_viterbi.yaml --output outputs/viterbi.csv
    python main.py --config config_hysteresis_backdated.yaml --output outputs/hysteresis_backdated.csv
    python main.py --config config_confidence.yaml --output outputs/confidence.csv
    python main.py --config config_reliability_viterbi.yaml --output outputs/reliability_viterbi.csv

No competition public ground truth is read anywhere.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data_loader import input_paths, load_tracking
from src.ball_recovery import recover_ball
from src.geometry import geometry_evidence
from src.tactical_features import read_events, tactical_evidence
from src.dead_ball import estimate_dead_probability
from src.fusion import fuse_evidence
from src.temporal_decoder import active_segments, decode_states
from src.submission import write_submission, validate_csv


ROOT = Path(__file__).resolve().parent

SUPPORTED_DECODERS = {
    "viterbi",
    "hysteresis",
    "confidence_hysteresis",
    "reliability_viterbi",
}


def run(
    config: dict,
    data_dir: Path
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Run the complete possession prediction pipeline.

    Steps:
        1. Load tactical tracking data.
        2. Recover missing ball positions.
        3. Calculate geometric evidence.
        4. Calculate tactical evidence.
        5. Estimate dead-ball probabilities.
        6. Fuse evidence into H/A/D probabilities.
        7. Detect active playing segments.
        8. Calculate observation reliability.
        9. Decode possession states.
       10. Return the final labels and segments.

    Returns:
        labels:
            Full-match array containing H, A, D, or X.

        segments:
            Active playing regions represented as
            (start_frame, end_frame_exclusive).
    """

    # --------------------------------------------------
    # 1. Read configuration
    # --------------------------------------------------

    match = config["match"]
    decoder_cfg = config["decoder"]

    n = int(match["frames"])
    fps = int(match["fps"])

    pitch = (
        float(match["pitch_length_m"]),
        float(match["pitch_width_m"]),
    )

    method = decoder_cfg["method"]

    if method not in SUPPORTED_DECODERS:
        raise ValueError(
            f"Unsupported decoder method: {method}. "
            f"Supported methods: {sorted(SUPPORTED_DECODERS)}"
        )

    # --------------------------------------------------
    # 2. Locate tactical tracking files
    # --------------------------------------------------

    meta, raw, smart = input_paths(data_dir)

    print(
        "Reading tactical tracking "
        "(this requires the full licensed ATD)...",
        flush=True,
    )

    # --------------------------------------------------
    # 3. Load tracking data
    # --------------------------------------------------

    tracking = load_tracking(
        meta,
        raw,
        n,
        fps,
        pitch,
    )

    print(
        f"Loaded {n:,} frames; "
        f"{len(tracking.teams)} track slots",
        flush=True,
    )

    # --------------------------------------------------
    # 4. Recover ball positions
    # --------------------------------------------------

    ball = recover_ball(
        tracking.ball_xy,
        fps,
        pitch,
        **config["ball_recovery"],
    )

    # --------------------------------------------------
    # 5. Calculate geometric evidence
    # --------------------------------------------------

    geom = geometry_evidence(
        tracking,
        ball,
        config["geometry"],
    )

    # --------------------------------------------------
    # 6. Read tactical events
    # --------------------------------------------------

    events = read_events(
        smart,
        fps,
    )

    tactical = tactical_evidence(
        events,
        n,
        config["tactical"],
    )

    # --------------------------------------------------
    # 7. Estimate dead-ball probabilities
    # --------------------------------------------------

    dead = estimate_dead_probability(
        tracking,
        ball,
        tactical,
        config["dead_ball"],
        config["fusion"]["base_dead_probability"],
        config["fusion"]["max_dead_probability"],
    )

    # --------------------------------------------------
    # 8. Fuse H/A/D evidence
    # --------------------------------------------------

    probabilities = fuse_evidence(
        geom,
        tactical,
        dead,
        config["fusion"],
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.shape != (n, 3):
        raise ValueError(
            "Invalid probability array shape. "
            f"Expected ({n}, 3), "
            f"received {probabilities.shape}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Probability array contains NaN "
            "or infinite values."
        )

    # --------------------------------------------------
    # 9. Detect active playing segments
    # --------------------------------------------------

    segments = active_segments(
        tracking.player_count,
        config["activity"],
    )

    if not segments:
        raise RuntimeError(
            "No active play segments detected. "
            "Inspect tracking completeness and "
            "activity configuration before "
            "producing a submission."
        )

    # --------------------------------------------------
    # 10. Calculate observation reliability
    # --------------------------------------------------
    #
    # First experiment:
    #
    #   1.0 = ball directly observed
    #   0.0 = ball not directly observed
    #
    # This is deliberately a simple binary signal.
    #
    # Later experiments can use ball.confidence
    # or a combination of observation features.
    # --------------------------------------------------

    observation_reliability = np.asarray(
        ball.observed,
        dtype=np.float64,
    )

    if observation_reliability.shape != (n,):
        raise ValueError(
            "Observation reliability must contain "
            f"exactly {n} values. "
            f"Received shape: "
            f"{observation_reliability.shape}"
        )

    if not np.all(
        np.isfinite(observation_reliability)
    ):
        raise ValueError(
            "Observation reliability contains "
            "NaN or infinite values."
        )

    if np.any(
        (observation_reliability < 0.0)
        | (observation_reliability > 1.0)
    ):
        raise ValueError(
            "Observation reliability must "
            "be between 0 and 1."
        )

    # --------------------------------------------------
    # 11. Decode possession states
    # --------------------------------------------------

    print(
        f"Decoder: {method}",
        flush=True,
    )

    labels = decode_states(
        probabilities,
        segments,
        decoder_cfg,
        observation_reliability=observation_reliability,
    )

    # --------------------------------------------------
    # 12. Validate decoded labels
    # --------------------------------------------------

    labels = np.asarray(labels)

    if labels.shape != (n,):
        raise ValueError(
            "Decoder returned an invalid label shape. "
            f"Expected ({n},), "
            f"received {labels.shape}."
        )

    valid_labels = np.array(
        ["H", "A", "D", "X"]
    )

    if not np.all(
        np.isin(labels, valid_labels)
    ):
        raise ValueError(
            "Decoder returned invalid labels. "
            "Allowed labels: H, A, D, X."
        )

    # --------------------------------------------------
    # 13. Print diagnostics
    # --------------------------------------------------

    observed_count = int(
        np.sum(ball.observed)
    )

    observed_percentage = float(
        np.mean(ball.observed)
    )

    print(
        f"Tactical phases: "
        f"{tactical.event_count}, "
        f"observed ball: "
        f"{observed_count:,} "
        f"({observed_percentage:.1%})",
        flush=True,
    )

    print(
        "Detected candidate playing regions "
        "(end exclusive):",
        segments,
        flush=True,
    )

    active_mask = np.zeros(
        n,
        dtype=bool,
    )

    for first, last in segments:
        active_mask[first:last] = True

    active_frames = int(
        np.sum(active_mask)
    )

    print(
        f"Active playing frames: "
        f"{active_frames:,} "
        f"({active_frames / n:.1%})",
        flush=True,
    )

    if active_frames > 0:
        active_observed = float(
            np.mean(
                observation_reliability[active_mask]
            )
        )

        print(
            f"Ball observed within active regions: "
            f"{active_observed:.1%}",
            flush=True,
        )

    return labels, segments


def main():
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help=(
            "Directory containing tactical "
            "tracking and smart-tagging files."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "submission.csv",
        help="Output CSV file path.",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    with args.config.open(
        "r",
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be a YAML mapping."
        )

    # --------------------------------------------------
    # Run prediction pipeline
    # --------------------------------------------------

    labels, segments = run(
        config,
        args.data_dir,
    )

    # --------------------------------------------------
    # Write submission
    # --------------------------------------------------

    output = write_submission(
        labels,
        args.output,
        int(config["match"]["frames"]),
    )

    print(
        "Wrote",
        output,
        flush=True,
    )

    # --------------------------------------------------
    # Validate submission
    # --------------------------------------------------

    validation = validate_csv(
        output,
        len(labels),
    )

    print(
        "Validated labels:",
        validation,
        flush=True,
    )

    print(
        "Run the organizer's "
        "evaluation/validate_submission.py "
        "on this exact file too.",
        flush=True,
    )


if __name__ == "__main__":
    main()
