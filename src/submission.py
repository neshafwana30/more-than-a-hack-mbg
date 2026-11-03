"""Official 0-based frame,label CSV contract and simple diagnostics."""
from collections import Counter
from pathlib import Path
import csv

import numpy as np

VALID = frozenset("HADX")


def write_submission(labels: np.ndarray, output: Path, expected_rows: int) -> Path:
    if len(labels) != expected_rows:
        raise ValueError(f"Refusing invalid submission: {len(labels)} rows != {expected_rows}")
    if not set(np.unique(labels)).issubset(VALID):
        raise ValueError("Unexpected labels; only H, A, D, X permitted")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows((i, str(label)) for i, label in enumerate(labels))
    return output


def validate_csv(path: Path, expected_rows: int) -> dict:
    counter = Counter()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["frame", "label"]:
            raise ValueError("Submission header must be exactly frame,label")
        for i, row in enumerate(reader):
            if len(row) != 2 or row[0] != str(i) or row[1] not in VALID:
                raise ValueError(f"Invalid row {i}: {row}")
            counter[row[1]] += 1
    if sum(counter.values()) != expected_rows:
        raise ValueError(f"Wrong submission length: {sum(counter.values())} != {expected_rows}")
    return dict(counter)
