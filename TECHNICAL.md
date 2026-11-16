# Technical document — controlled-possession pipeline

> Draft. Before submitting: rename to `<team_name>.md` (or export to `<team_name>.pdf`), add the
> team name and members, and re-read the numbers in §6 against the file you actually upload.

## 1. Approach in one paragraph

We label each frame in two steps. **First we separate ball-in-play from dead ball**. **Then, inside each stretch of play, we decide which team has *controlled* possession.** A stoppage always ends a possession, so the stretches between stoppages are decoded independently. Inside a stretch, the label changes only when the evidence for the other team adds up to more than a fixed switch penalty. A controlled dribble or a completed pass clears that penalty. A single touch or a deflection does not. Wherever the ball is visible, the evidence comes from touch geometry. Where it is missing, which is 41% of graded frames, usually in stretches of 10–40 s, it comes from a self-supervised team-shape model. Smart Tagging adds weak team evidence and names the restarting team at set pieces.

## 2. Pipeline

| step | module | what it does |
|---|---|---|
| load | `src/data_loader.py` | tactical ATD only, 199,319 frames, 46 track slots → positions per team |
| ball | `src/ball_recovery.py` | interpolates gaps ≤ 12 frames when the implied speed is physical; never invents long gaps |
| motion | `src/motion.py` | collective player pace (2 s), ball speed (1 s), ball out of bounds, distance from the ball to the nearest home and away player, attacking direction per half (measured from mean team positions) |
| live / dead | `src/live_dead.py` | per-frame dead-ball log-odds → two-state Viterbi → stoppages under 2 s and live stretches under 2 s absorbed |
| control | `src/control.py` | touches → episodes → *controlled* episodes → per-frame team evidence |
| team shape | `src/shape_model.py` | logistic regression on player-only features, trained on the algorithm's own controlled episodes |
| decode | `src/possession_v2.py` | H/A Viterbi per live stretch; `D` on stoppages; `X` outside the two halves |

All parameters live in `config.yaml` → `possession_v2`, each with its football reason next to it.

## 3. The possession rules we implemented

**Dead ball (`D`).** Per-frame evidence, positive meaning dead:

- **Players walking.** Two-second mean player speed below walking pace (1.3 m/s) counts as dead evidence, and above it as live evidence.
- **Ball out of the pitch.** The ball is more than about 0.5 m beyond a line. This counts as strong evidence and is carried forward while the tracker has lost the ball out there.
- **Ball stationary.** A placed ball for a free kick, goal kick or corner. Speed below 0.5 m/s counts as dead evidence.
- **Ball moving fast inside the pitch.** Speed above 5 m/s counts as live evidence.
- **Ball missing for 2 s or more.** Weak dead evidence, because the tracker also loses the ball during play.
- **Smart Tagging `SET PIECES`.** Dead evidence around the tag's start, when the restart is being set up.

The scores are decoded with a switching penalty. A stoppage must last at least 2 s, and so must a live stretch between two stoppages.

**Control.** A player is *on the ball* when they are within 1.5 m of it and at least 0.5 m closer than the nearest opponent. A ball moving faster than 15 m/s is travelling, not controlled. Touches by the same team separated by 4 frames or fewer are merged into one episode. An episode counts as **controlled** in either of two cases, which are the two proxies `CHALLENGE.md` suggests:

- **Kept.** The episode lasts at least 0.5 s.
- **Completed pass.** The same team touches the ball again within 4 s, at least 5 m away, with no opponent touch and no stoppage in between.

**What does not count.** Touches that are not controlled, which covers deflections, blocks and duels won without keeping the ball, carry only weak evidence (0.5 log-odds per frame against 3.0). On their own they cannot overcome the switch penalty of 15.

**Loose balls and challenges.** Possession stays with the last team until the other team's controlled evidence exceeds the penalty. Because decoding is offline, the switch is placed where the controlling episode **began**, not where it was confirmed.

**Restarts.** After each stoppage the possession is decided afresh from what happens after the restart. If Smart Tagging names a team for the set piece (every throw-in is `N/A`), that team gets a small prior for the first 2 s after the restart.

**`X`.** We write `X` only outside the two published halves, where nothing is graded. Inside the halves we always write a label. Under `spellF1` an `X` is replaced by the previous label anyway, so holding the label explicitly costs nothing on the ranking metric and is never worse on `macroF1`.

## 4. Assets used, and dropped

| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | **yes** | everything geometric |
| Smart Tagging | **yes** | team phases as weak evidence; `SET PIECES` for stoppages and restart team |
| published half boundaries (`DATA.md`, `games.py`) | **yes** | graded vs ungraded frames, attacking direction per half |
| tactical homography | no | ATD is already in pitch coordinates |
| broadcast ATD / video | no | not synchronised; left for future work |
| tactical video | no (only watched for understanding) | — |

## 5. Training data

**No human-labelled data of any kind.** The team-shape model is self-supervised. Its targets are the frames that our own control detector marks as controlled possession on this match's automatic tracking. Its features use players only:

- how far up the pitch each team is
- the position of its most advanced player
- the position of its deepest players
- the team's length and width
- the team's pace
- its forward velocity

All features are taken as home-minus-away differences, oriented by attacking direction. `RULES.md` allows this explicitly ("pseudo-labels produced by an algorithm, provided no person decided any of them").

When trained on one half, the model agrees with the detector on the other half about **85%** of the time (85.0% and 84.1%). This number is printed on every run.

## 6. Checks we ran — and what we did *not* do

- `python evaluation/validate_submission.py outputs/submission.csv` passes. The file has 344 possessions, mean 11.5 s.
- Without using any labels, about 38% of graded frames are dead ball, across 128 stoppages, and home has 63% of live frames. All three are plausible for a LaLiga match.
- **Public 5-minute window.** We scored against it **once**, to check that the pipeline is aligned and that the labels mean what we think. We did **not** use it to choose any parameter or any version. The result was spellF1 0.46 and macroF1 0.80, with 22 predicted possessions against 17. The published baseline gets 0.33 / 0.62 on the same window. Seventeen possessions cannot rank anything (see `EVALUATION.md`).
- **Determinism.** The same file is produced twice in a row, and with the pinned `numpy==2.3.5`.
- 18 unit and synthetic tests (`pytest`), none of which read competition labels.

## 7. Limitations and what we would do next

- **Stoppage boundaries lag by about 1–2 s**, because player pace is averaged over 2 s. Next step: detect the kick at the restart and the line crossing directly from ball speed and position, and snap the boundaries to them.
- **Possessions shorter than about 1 s** are mostly merged away by the switch penalty. This is the price of not fragmenting.
- **Aerial balls.** The ball is projected onto the ground, so a high ball can look "near" the wrong player. The in-flight speed gate catches fast balls but not slow, lofted ones.
- **Ball-missing stretches** rely on team shape, which is a weaker signal than touches. The 85% agreement above is measured where the ball is visible, so it is probably optimistic for the missing stretches.
- All parameters are hand-set priors. With labelled data from *other* matches we would fit them there.

## 8. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy 2.3.5, PyYAML 6.0.3 (Python ≥ 3.10)
# put the licensed bundle in ./data (see DATA.md) — it is gitignored
python main.py                             # -> outputs/submission.csv  (~5 s)
python evaluation/validate_submission.py outputs/submission.csv
python -m pytest -q tests                  # optional
```

`python main.py` with no arguments regenerates the submitted file exactly. The older decoders are still available with `--config config_<name>.yaml` for comparison.
