
import pandas as pd
import numpy as np

files = {
    "Viterbi": "outputs/viterbi.csv",
    "Hysteresis": "outputs/hysteresis_backdated.csv",
    "Confidence": "outputs/confidence.csv",
    "Reliability Viterbi": "outputs/reliability_viterbi.csv"
}

data = {}

for name, path in files.items():
    df = pd.read_csv(path).sort_values("frame")
    labels = df["label"].to_numpy()
    data[name] = labels

    previous = labels[:-1]
    current = labels[1:]

    changed = previous != current

    # Direct H <-> A transitions
    team_turnovers = (
        ((previous == "H") & (current == "A")) |
        ((previous == "A") & (current == "H"))
    ).sum()

    # Changes involving dead-ball state
    dead_transitions = (
        changed &
        ((previous == "D") | (current == "D"))
    ).sum()

    # Changes involving abstention state
    x_transitions = (
        changed &
        ((previous == "X") | (current == "X"))
    ).sum()

    # Consecutive label runs
    starts = np.r_[
        0,
        np.flatnonzero(changed) + 1
    ]
    ends = np.r_[starts[1:], len(labels)]

    runs = pd.DataFrame({
        "label": labels[starts],
        "duration_frames": ends - starts
    })

    spells = runs[runs["label"].isin(["H", "A"])]

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")

    print("Total transitions:", changed.sum())
    print("Direct team turnovers:", team_turnovers)
    print("Dead-ball transitions:", dead_transitions)
    print("Abstention transitions:", x_transitions)

    print("H/A spells:", len(spells))
    print("Mean spell duration (s):",
          round(spells["duration_frames"].mean() / 25, 2))
    print("Median spell duration (s):",
          round(spells["duration_frames"].median() / 25, 2))

    for seconds in [1, 2, 5]:
        count = (
            spells["duration_frames"] < seconds * 25
        ).sum()

        print(f"Spells shorter than {seconds}s:", count)

print("\nDifferences from Viterbi:")

for name in ["Hysteresis", "Confidence"]:
    changed = np.sum(data[name] != data["Viterbi"])
    percentage = changed / len(data["Viterbi"]) * 100

    print(f"{name}: {changed} frames ({percentage:.2f}%)")
