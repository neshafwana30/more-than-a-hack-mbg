
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("csv")
parser.add_argument("--fps", type=float, default=25.0)
args = parser.parse_args()

df = pd.read_csv(args.csv)
labels = df["label"].to_numpy()

starts = np.r_[
    0,
    np.flatnonzero(labels[1:] != labels[:-1]) + 1
]
ends = np.r_[starts[1:], len(labels)]

runs = pd.DataFrame({
    "label": labels[starts],
    "start": starts,
    "end": ends,
    "duration_s": (ends - starts) / args.fps
})

possessions = runs[runs["label"].isin(["H", "A"])]

print("Total H/A spells:", len(possessions))
print("Mean duration:", possessions["duration_s"].mean())
print("Median duration:", possessions["duration_s"].median())

for seconds in [0.5, 1, 2, 5]:
    count = (possessions["duration_s"] < seconds).sum()
    print(f"Spells shorter than {seconds}s:", count)

print("\nDuration distribution:")
print(possessions["duration_s"].describe())
