"""Automatically detect active video sections and decode coherent H/A/D states."""
import numpy as np

LABELS = np.asarray(["H", "A", "D"], dtype="<U1")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def active_segments(player_count: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """Detect long play-like tracking blocks and merge brief tracking dropouts.

    This does not read kickoff labels/GT. It can under-detect some odd recordings;
    inspect its reported boundaries and validate on allowed external examples.
    """
    n = len(player_count)
    mask = player_count >= int(cfg["min_tracked_players"])
    runs = _runs(~mask)
    for start, end in runs:
        if start > 0 and end < n and end - start <= int(cfg["max_tracking_hole_frames"]):
            mask[start:end] = True
    minimum = int(cfg["min_active_block_frames"])
    padding = int(cfg.get("end_padding_frames", 0))
    regions = []
    for start, end in _runs(mask):
        if end - start >= minimum:
            regions.append((max(0, start - padding), min(n, end + padding)))
    merged = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def viterbi(probabilities: np.ndarray, cfg: dict) -> np.ndarray:
    """Deterministic maximum a-posteriori sequence with transition penalties."""
    n = len(probabilities)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    log_p = np.log(np.clip(probabilities, 1e-8, 1))
    transition = np.array(
        [[0, -cfg["home_away_switch_penalty"], -cfg["play_dead_switch_penalty"]],
         [-cfg["home_away_switch_penalty"], 0, -cfg["play_dead_switch_penalty"]],
         [-cfg["play_dead_switch_penalty"], -cfg["play_dead_switch_penalty"], 0]],
        dtype=np.float64,
    )
    backptr = np.zeros((n, 3), dtype=np.int8)
    previous = log_p[0].astype(np.float64)
    state_ids = np.arange(3)
    for t in range(1, n):
        transition_scores = previous[:, None] + transition
        backptr[t] = np.argmax(transition_scores, axis=0)
        previous = transition_scores[backptr[t], state_ids] + log_p[t]
    result = np.empty(n, dtype=np.int8)
    result[-1] = int(np.argmax(previous))
    for t in range(n - 1, 0, -1):
        result[t - 1] = backptr[t, result[t]]
    return result


def confidence_hysteresis(probabilities: np.ndarray, cfg: dict) -> np.ndarray:
    """Decode H/A/D states using accumulated log-score evidence."""
    n = len(probabilities)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0)
    states = np.empty(n, dtype=np.int8)
    current = int(np.argmax(p[0]))
    pending = -1
    evidence = 0.0
    team_threshold = float(cfg.get("team_evidence_threshold", 5.0))
    dead_threshold = float(cfg.get("dead_evidence_threshold", 3.3))
    max_evidence = float(cfg.get("max_accumulated_evidence", 20.0))
    for t in range(n):
        candidate = int(np.argmax(p[t]))
        if candidate == current:
            pending = -1
            evidence = 0.0
        else:
            if candidate != pending:
                pending = candidate
                evidence = 0.0
            increment = np.log(p[t, candidate]) - np.log(p[t, current])
            evidence = float(np.clip(evidence + increment, 0.0, max_evidence))
            threshold = (team_threshold if current != 2 and candidate != 2
                         else dead_threshold)
            if evidence >= threshold:
                current = candidate
                pending = -1
                evidence = 0.0
        states[t] = current
    return states


def hysteresis(probabilities: np.ndarray, confirmation: int,
               backdate_confirmed: bool = False) -> np.ndarray:
    """Require consecutive argmax choices before accepting a state change.

    With backdate_confirmed=False, reproduce the original online-style decoder.
    With backdate_confirmed=True, assign a confirmed new state retrospectively
    to the first frame of its consecutive candidate run. Unconfirmed candidate
    runs remain in the previous state. This requires offline look-ahead.
    """
    if confirmation < 1:
        raise ValueError("confirmation must be >= 1")
    if not len(probabilities):
        return np.empty(0, dtype=np.int8)
    choices = np.argmax(probabilities, axis=1).astype(np.int8)
    states = np.empty(len(choices), dtype=np.int8)
    current = int(choices[0])
    pending = current
    count = 0
    candidate_start = 0

    for i, choice in enumerate(choices):
        choice = int(choice)
        if choice == current:
            pending, count = current, 0
        elif choice != pending:
            pending, count = choice, 1
            candidate_start = i
        else:
            count += 1

        if count >= confirmation:
            current, pending, count = pending, pending, 0
            if backdate_confirmed:
                states[candidate_start:i] = current
        states[i] = current
    return states


def merge_short_runs(states: np.ndarray, minimum: int) -> np.ndarray:
    """Remove tiny A-B-A flickers, never merge unlike flanking states."""
    result = states.copy()
    if minimum <= 0 or len(result) < 3:
        return result
    for _ in range(3):
        starts = np.r_[0, np.flatnonzero(result[1:] != result[:-1]) + 1]
        ends = np.r_[starts[1:], len(result)]
        changed = False
        for j in range(1, len(starts) - 1):
            left, right = starts[j], ends[j]
            if right - left < minimum and result[starts[j - 1]] == result[starts[j + 1]]:
                result[left:right] = result[starts[j - 1]]
                changed = True
        if not changed:
            break
    return result


def decode_states(
    probabilities,
    segments,
    cfg,
    observation_reliability=None
):
    """Return X outside inferred active regions; decode each region separately."""
    labels = np.full(len(probabilities), "X", dtype="<U1")
    for first, last in segments:
        if first < 0 or last > len(probabilities) or first >= last:
            raise ValueError(f"Invalid activity region ({first}, {last})")
        if cfg["method"] == "viterbi":
            codes = viterbi(probabilities[first:last], cfg)
        elif cfg["method"] == "confidence_hysteresis":
            codes = confidence_hysteresis(probabilities[first:last], cfg)
        elif cfg["method"] == "hysteresis":
            codes = hysteresis(
                probabilities[first:last],
                int(cfg["hysteresis_confirm_frames"]),
                backdate_confirmed=bool(cfg.get("hysteresis_backdate_confirmed", False)),
            )
        elif cfg["method"] == "reliability_viterbi":
            from src.reliability_viterbi import reliability_viterbi

            if observation_reliability is None:
                raise ValueError(
                    "Reliability-aware Viterbi requires "
                    "per-frame observation reliability."
                )

            codes = reliability_viterbi(
                probabilities[first:last],
                cfg,
                observation_reliability[first:last]
            )
        else:
            raise ValueError(f"Unknown decoder method: {cfg['method']}")
        codes = merge_short_runs(codes, int(cfg["short_run_merge_frames"]))
        labels[first:last] = LABELS[codes]
    return labels
