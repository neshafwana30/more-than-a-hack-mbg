"""Score predictions against independently sourced labels from OTHER matches.

Do NOT use this script to tune against the publicly labelled 5 minutes of the
competition game. The reference CSV here must concern a DIFFERENT match.
The spellF1 implementation follows the organizers' greedy IoU matching rule.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

IOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def read_csv(path: Path) -> np.ndarray:
    values = []
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["frame", "label"]:
            raise ValueError(f"{path}: expected frame,label header")
        for index, row in enumerate(reader):
            if int(row["frame"]) != index:
                raise ValueError(f"{path}: invalid frame at row {index}")
            label = row["label"].strip().upper()
            if label not in ("H", "A", "D", "X"):
                raise ValueError(f"{path}: invalid label {label} at {index}")
            values.append(label)
    return np.asarray(values, dtype="<U1")


def valid_regions(reference: np.ndarray) -> list[tuple[int, int]]:
    good = reference != "X"
    edges = np.diff(np.r_[False, good, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_abstentions(predict: np.ndarray, regions: list[tuple[int,int]]) -> np.ndarray:
    out = predict.copy()
    for first, last in regions:
        part = out[first:last]
        choices = np.flatnonzero(part != "X")
        if len(choices) == 0:
            continue
        part[:choices[0]] = part[choices[0]]
        last_seen = part[choices[0]]
        for k in range(choices[0]+1, len(part)):
            if part[k] == "X":
                part[k] = last_seen
            else:
                last_seen = part[k]
    return out


def spells(arr: np.ndarray, regions: list[tuple[int,int]]) -> list[tuple[int,int,str]]:
    out = []
    for first, last in regions:
        if last <= first:
            continue
        starts = np.r_[first, np.flatnonzero(arr[first+1:last] != arr[first:last-1]) + first + 1]
        ends = np.r_[starts[1:], last]
        for lo, hi in zip(starts, ends):
            label = arr[lo]
            if label in ("H", "A"):
                out.append((int(lo), int(hi), str(label)))
    return out


def compute_metrics(reference: np.ndarray, predicted: np.ndarray,
                    fps: float = 25.0) -> dict:
    if len(reference) != len(predicted):
        raise ValueError("Reference and predictions must have same row count")
    regions = valid_regions(reference)
    mask = reference != "X"
    macro = {}
    for label in "HAD":
        true = reference[mask] == label
        pred = predicted[mask] == label
        tp = int((true & pred).sum())
        fp = int((~true & pred).sum())
        fn = int((true & ~pred).sum())
        macro[label] = 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.0

    filled_predicted = fill_abstentions(predicted, regions)
    gt_spells = spells(reference, regions)
    pred_spells = spells(filled_predicted, regions)
    pairs = []
    for i, (a, b, team) in enumerate(gt_spells):
        for j, (c, d, pred_team) in enumerate(pred_spells):
            if team != pred_team or b <= c or d <= a:
                continue
            overlap = min(b,d) - max(a,c)
            union = max(b,d) - min(a,c)
            pairs.append((overlap/union, i, j))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    by_threshold = {}
    for threshold in IOU_THRESHOLDS:
        taken_gt, taken_pred = set(), set()
        for iou, i, j in pairs:
            if iou < threshold:
                break
            if i not in taken_gt and j not in taken_pred:
                taken_gt.add(i)
                taken_pred.add(j)
        tp = len(taken_gt)
        f1 = 2*tp / (len(gt_spells) + len(pred_spells)) if (len(gt_spells) + len(pred_spells)) else 0.0
        by_threshold[f"{threshold:.1f}"] = f1
    lengths = [b-a for a,b,_ in pred_spells]
    return {
        "n_scored_frames": int(mask.sum()),
        "macroF1": float(np.mean(list(macro.values()))),
        "class_F1": macro,
        "spellF1": float(np.mean(list(by_threshold.values()))),
        "spellF1_by_IoU": by_threshold,
        "reference_spells": len(gt_spells),
        "predicted_spells": len(pred_spells),
        "predicted_mean_spell_s": float(np.mean(lengths)/fps) if lengths else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True,
                        help="Independent labels for a DIFFERENT match, not BAR-ATH")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()
    if "bar-ath" in args.reference.name.casefold() or "possession_gt_public" in args.reference.name.casefold():
        parser.error("Competition ground-truth file cannot be used for tuning. "
                     "Supply a separate labeled match.")
    print(json.dumps(compute_metrics(read_csv(args.reference), read_csv(args.prediction),
                                     fps=args.fps), indent=2))


if __name__ == "__main__":
    main()
