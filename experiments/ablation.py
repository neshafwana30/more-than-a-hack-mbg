"""Compare pre-generated variant CSVs against external match labels.

This is a diagnostic tool, NOT a competition-match hyperparameter sweep. The
predictions must all refer to the same independently labeled *other* match.
"""
import argparse
import csv
from pathlib import Path

try:
    from .external_evaluation import read_csv, compute_metrics
except ImportError:
    from external_evaluation import read_csv, compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True,
                        help="CSV of external game labels, NOT public challenge labels")
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="Directory containing variant predictions as CSV")
    parser.add_argument("--out", type=Path, default=Path("outputs/ablation_report.csv"))
    args = parser.parse_args()
    if "bar-ath" in args.reference.name.lower() or "possession_gt_public" in args.reference.name.lower():
        parser.error("Do not perform ablation against competition ground truth")
    truth = read_csv(args.reference)
    rows = []
    for path in sorted(args.predictions_dir.glob("*.csv")):
        if path.resolve() == args.reference.resolve():
            continue
        m = compute_metrics(truth, read_csv(path))
        rows.append({"variant":path.stem, "macroF1":m["macroF1"],
                     "spellF1":m["spellF1"], "predicted_spells":m["predicted_spells"],
                     "n_scored_frames":m["n_scored_frames"]})
    if not rows:
        parser.error("No variant CSVs found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for r in sorted(rows, key=lambda row: row["spellF1"], reverse=True):
        print(f'{r["variant"]:24} spellF1={r["spellF1"]:.4f} macroF1={r["macroF1"]:.4f}')
    print("Report:", args.out)


if __name__ == "__main__":
    main()
