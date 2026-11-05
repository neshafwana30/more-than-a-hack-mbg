# Barça Possession Intelligence System — System Specification v2.0

**Status:** Architecture and implementation specification (not an implementation)  
**Date:** 2026-11-05  
**Reference implementation:** `src/features.py` + `notebooks/03_baseline_possession.ipynb` (v1)  
**Target:** An inspectable, reproducible offline inference pipeline predicting `H`, `A`, or `D` for every graded tactical-video frame, and `X` outside graded blocks.  
**Intended reader:** Implementation team / subsequent coding LLM.  
**Priority:** Correctness, timeline integrity, valid competition compliance, clear failure analysis, then improvements.

> **READ THIS FIRST.** This document distinguishes (1) official challenge constraints; (2) measurements from executed notebook 3; (3) proposed v2 design; and (4) unverified hypotheses. Do **not** present heuristics or uncalibrated scores as ground-truth facts. Do **not** train or tune on the five minutes of public ground truth. Do not hand-correct outputs based on watching this match. Do not upload restricted match assets to hosted/cloud third-party LLMs.

---

## 0. Executive decision

**Do not attempt a monolithic end-to-end deep learning model yet.** Build a modular, event-aware, two-stage inference system:

1. **Evidence generation:** tracking quality + team control + live/dead + tag context; optional, selective tactical-video evidence.
2. **Offline temporal inference:** resolve ambiguity using past and future evidence, with state-dependent transition constraints and an *optional soft duration prior*.
3. **Boundary refinement:** identify the most defensible start/end frame for turnovers and dead/live transitions.
4. **Submission and diagnostics:** write a deterministic full-length CSV and perform strict validation.

**V2 focus:** prevent proxy flicker from becoming possessions, establish credible `D` intervals (including restart boundaries), and handle long missing-ball stretches explicitly. **Video is an optional plug-in**, not a prerequisite for a working v2 baseline. Broadcast is deliberately out of scope.

**Execution goal, once implemented:** `python run_pipeline.py --config config/v2.yaml --output evaluation/results/team.csv`. The command is a **planned interface**, not currently available.

---

## 1. Sources of truth and competition contract

### 1.1 Official source files

- `CHALLENGE.md`: task and semantics.
- `DATA.md`: match assets and data conventions.
- `EVALUATION.md`: metric semantics.
- `RULES.md`: eligibility, reproducibility, no human labeling on this match, data-use restrictions.
- `SUBMISSION.md`: submission schema and deadline.
- `FAQ.md`: clarified edge cases.
- `games.py`: canonical local asset paths, reference video indices, half boundaries and pitch size.

Consult actual repository versions before changing the code. The rules, official evaluator, and dataset terms are authoritative over this proposal.

### 1.2 Output semantics

- `H`: Barcelona/home controls the ball **while live**.
- `A`: Athletic/away controls the ball **while live**.
- `D`: ball not in play. This is **not** ownership by the team who will take a restart.
- `X`: abstention; not a fourth ground-truth game state on graded frames.

One possession spell is a maximal contiguous run of live `H` or `A`. Two same-team runs separated by `D` are **two possessions**. A mere touch, deflection or uncompleted aerial duel is **not automatically a turnover**; controlled possession by the other team is required. While live and contested/loose, last owning team persists.

### 1.3 Timeline and output schema

- Tactical video: `N=199319` frames, `FPS=25`; exactly **one row for each frame**.
- CSV: header `frame,label`; `frame=0..199318` sequential; labels from `{H,A,D,X}`; no blank values or index column.
- Graded blocks, inclusive: half 1 `11185..90651`; half 2 `110808..190197`; total `158857` graded frames.
- Outside graded blocks use `X` by policy in v2. Note that the first 2,934 frames of the *graded* second-half block precede kickoff and are `D` according to FAQ; **never convert the entire pre-kickoff graded interval to `X`**.
- `kloppy` ATD DataFrame `frame_id` is sequential `1..199319`: `video_frame = frame_id - 1`. The separate raw `IdFrame` counter uses **half-specific** offsets from `games.py`. Do not apply both conversions to the same representation.
- Smart Tagging timestamps are seconds relative to tactical video start; frame conversion convention: `round(seconds*FPS)`. A tag boundary does **not** automatically define a possession/dead-ball boundary.
- Homography tactical rows are frame `0..199318`; the matrix maps normalized pitch-plane position to video pixel coordinates. Assess `valid` and `reprojection_error`.
- Broadcast timeline is not aligned to tactical. **V2 does not use broadcast.**

### 1.4 Official evaluation and compliance

- Ranking: **`spellF1`**. Spells matched one-to-one by temporal IoU; F1 averaged at IoU thresholds `0.3,0.4,0.5,0.6,0.7`.
- Secondary metric and first tie-breaker: **`macroF1`** over `H`, `A`, `D` per graded frame.
- On graded frames, `X` is counted as a miss by macroF1; for spellF1, official scoring fills it with last non-`X` state in the same half (with initial backfill).
- Official published simple baseline `baseline_smooth50` has full-match spellF1 `0.2433`; **do not claim v1 beats/loses to it** without an official scored result.
- Five-minute public ground truth is **only for pipeline correctness/label-semantic checks**, not training, hyperparameter selection, architecture search or leaderboard optimization. Private match truth is unavailable.
- Human hand labels, manual corrections or frame-specific constants on this supplied match are forbidden. Training/validation annotations of *other games* are allowable subject to availability and documentation.
- Data/video may not be redistributed, committed to public repository, or uploaded to restricted hosted/cloud LLMs. Code, synthetic tests, aggregate non-reconstructive summaries and public documentation can be shared as permitted by terms.
- Final deliverables: full possession CSV, public reproducible repository, and short technical report. One form, one response/team, deadline **2026-11-17**.

---

## 2. Baseline v1 — observed execution results

**Evidence source:** executed `/mnt/data/03_baseline_possession.ipynb` supplied by user. Results are descriptive outputs **not evaluation scores**.

### 2.1 Successful execution and integrity

| Item | Observed output |
|---|---:|
| Tactical feature rows | 199,319 |
| Feature columns | 25 |
| Team metadata slots | H: 24; A: 22 |
| Smart Tagging instances | 638 |
| Ball observed on graded frames | 59.31% (94,214 frames) |
| Ball missing on graded frames | 64,643 frames |
| Evidence matrix | `(199319, 3)` for `H,A,D` |
| Output file | `evaluation/results/baseline_v1.csv` on user's machine |
| File size | 1,882,093 bytes |
| Official submission-format validation | **Passed** |
| Public/private scoring | **Not run** |

The public ground truth was deliberately not loaded. A printed `RuntimeWarning: Mean of empty slice` appeared in the evidence calculation; the cell still executed, but the warning should be eliminated by explicit valid-observation handling.

### 2.2 Predictions and segmentation

| Output label | Graded frames | Share of graded frames | Predicted segments | Median segment duration |
|---|---:|---:|---:|---:|
| `H` | 84,601 | 53.26% | 592 | 2.52 s |
| `A` | 59,672 | 37.56% | 560 | 1.68 s |
| `D` | 14,584 | 9.18% | 154 | 2.04 s |
| Total graded | 158,857 | 100% | 1,306 state segments | — |

- **1,152 predicted possession segments** (`H` + `A`), **1,304 within-half state transitions** (1,306 segments minus initial segment of each half).
- Predicted possession mean **5.009 s**, median **2.04 s**; P10 **0.48 s**, P25 **0.88 s**, P75 **5.68 s**, P90 **11.912 s**; minimum **0.16 s**, maximum **208.96 s**.
- Segments shorter than 8 frames: H `4`, A `7`, D `9`. **Do not interpret this as absence of over-segmentation**: the vast majority of 1,152 possession spells are still only a few seconds long.
- Outside graded blocks `X=40,462` frames (as intended).

**Interpretation, not proof:** The very large number of short predicted spells is a **strong diagnostic signal** of possible over-segmentation. The entire match ground truth is private, so the true possession count and v1 spellF1 are not known. The five-minute public window has different scope and may not be used to tune to a desired spell count.

### 2.3 Missing-ball behavior

- `150` missing-ball runs within graded blocks.
- Endpoint nearest-team proxies: `A→A:49`, `H→H:37`, `H→A:35`, `A→H:25`, other/unknown combinations: `4`.
- In **51.3%** of missing runs, the **predicted state set** contained at least two distinct labels. This metric is **not** a ground-truth transition rate and does not reveal whether predicted transitions are correct.
- Need to distinguish (i) stale-control carry-forward; (ii) legitimate event-supported switches; (iii) tag-driven artificial switches; (iv) restart crossings during gaps.

### 2.4 Feature anomalies and weak assumptions

- Team centroid speed can reach **1,164 m/s (H)** and **1,365 m/s (A)** according to feature summaries. These are nonsensical as actual group motion and likely reflect changing set of visible tracks. They must be quality-gated, not simply treated as movement.
- Maximum observed ball 2D speed approximately **109.60 m/s**: anomaly candidate, not proof of erroneous tracking; inspect algorithmically and avoid blind trust.
- `ball_state`, `ball_owning_team_id`, `ball_z`, `ball_speed` from ATD parser were previously found entirely null; do not rely on them in v2.
- Current v1 uses near-player distance and overlapping tag counts, not true ball-control detections.
- Current `D` evidence is heuristic: `dead_base=0.012`, `dead_setpiece_bonus=0.06`, `dead_edge_bonus=0.20` for ±25 frames, stationary ball bonus `0.10`, low centroid-motion bonus `0.035`.
- `H/A` heuristic: `exp(-distance/2.5m)` + saturating weighted active team-tag count (`0.14`) + fallback (`0.02`). These are **handchosen evidence parameters**, not learned calibrated probabilities.
- Current Viterbi transition cost is constant `1.5` across **all** state changes. It does not encode `H↔A` vs `H/A↔D` differences or explicit spell duration.
- Baseline config declares a `continuity_bonus=0.015` but the shown evidence function does not use it; avoid relying on settings that have no effect. It also declares `short_run_frames=8` **for diagnostic only**.
- The notebook's current tag features (`active_h_tags` / `active_a_tags`) collapse heterogeneous, overlapping phases into counts; independent evidential weight is unsupported.
- The raw centroid average (`nanmean`) can have empty slices; explicitly handle these and ensure no unexpected NaNs enter log scores.

### 2.5 What the run DOES NOT establish

- No measured v1 `spellF1`, `macroF1`, precision/recall, or relative leaderboard rank.
- No verified possession thresholds, duration parameters, event-recognition accuracy or calibration.
- No confirmed tactical-video inference performance. A prior optional overlay image visually aligned tracking markers with the tactical frame, but **visual inference/ball detection is not implemented**.
- No proof that predicted 9.18% `D` is too low/high; verify only with legitimate external data or officially permitted semantic sanity checks.

---

## 3. Requirements and non-goals

### 3.1 Functional requirements (FR)

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-01 | Build features on canonical tactical timeline | Exactly 199,319 sorted rows with `frame=0..199318`, no duplicated indices |
| FR-02 | Preserve competition frame/block semantics | Half boundaries match `games.half_video_blocks`; outside blocks `X` |
| FR-03 | Validate team/slot mappings | Both sides identified from metadata, fail closed on unknown slots |
| FR-04 | Produce separate observation quality and missingness evidence | Mask, run length, age since/next observation, reason flags |
| FR-05 | Produce H/A control evidence when possible | Evidence per team plus missingness and supporting features |
| FR-06 | Produce Live/Dead evidence independently of H/A | Evidence plus candidate stoppage/restart events and boundary windows |
| FR-07 | Integrate Smart Tagging as *context* | No direct `SET PIECES = D`; tag overlaps handled explicitly |
| FR-08 | Determine state sequences using temporal model | No across-half leakage; explicit transition model; backtracking offline |
| FR-09 | Refine ambiguous transition boundaries | Event/window output + deterministic boundary strategy |
| FR-10 | Optional local visual escalation | Trigger + video QA; never manual per-frame correction |
| FR-11 | Generate/validate final CSV | Exact schema and official validator pass |
| FR-12 | Emit safe diagnostics | Counts, durations, warnings, config hash, no restricted tracking/video exports in public repo |

### 3.2 Non-functional requirements (NFR)

- Deterministic: same code, inputs, config and dependencies → identical CSV bytes.
- Modular: feature extractor, scorers, decoders and video subsystem independently testable.
- Explainable: each detected transition can report evidence categories and uncertainty **internally** without exporting restricted frame-level datasets.
- Robust: NaNs, missing players, long ball gaps, invalid transformations, and failed optional video analysis cannot crash base pipeline.
- Auditable: pinned dependencies, command, config, source revision, and runtime metadata documented.
- Reasonable performance: avoid decoding entire tactical video unless needed; use windowed reads and caching where appropriate.
- Output-state semantics: only `H/A/D` within graded blocks, except deliberate `X` under a documented fallback policy. Do not create extra ground-truth classes.

### 3.3 Out of scope for core v2

- Synchronizing or using broadcast video.
- Re-identification or stable identity of all 22 players.
- Complete multi-event football understanding (VAR, injury, fights, substitution, etc.). Such events are *optional evidence*, not separate required outputs.
- Training on provided match's public/private ground truth or hand annotating it.
- Assuming access to a pretrained ball detector without verifying local availability, license, and reproducibility.

---

## 4. Architecture

```text
 games.py + tactical ATD + Smart Tagging + homography/video (optional)
                               |
                     [Data / Frame Alignment]
                               |
                   [Feature & Quality Extractor]
                               |
               +---------------+---------------+
               |                               |
    [Team Control Evidence]         [Live/Dead & Event Evidence]
               |                               |
               +---------------+---------------+
                               |
                      [Ambiguity Detector]
                               |
          [Optional Tactical Video Analysis + Quality Gate]
                               |
                       [Evidence Fusion]
                               |
              [Offline Temporal Decoder (H/A/D)]
                               |
                [Boundary Refinement & Checks]
                               |
               [Submission CSV + Diagnostics]
```

**Important:** `Ambiguity Detector` selects video work or exposes uncertainty; it does **not** itself turn missing ball into `D`. The temporal decoder operates on complete evidence across each half; it does not simply label an uncertain run with the final observed team.

### 4.1 Proposed repository layout (implementation plan)

```text
project/
├── games.py                       # provided; canonical timeline and paths
├── config/
│   └── v2.yaml                   # initial hypotheses; not GT-tuned
├── src/
│   ├── __init__.py
│   ├── features.py               # existing v1; extend/test, do not rewrite blindly
│   ├── observation_quality.py
│   ├── team_control.py
│   ├── live_dead.py
│   ├── event_evidence.py
│   ├── ambiguity_detector.py
│   ├── visual_analyzer.py        # optional
│   ├── evidence_fusion.py
│   ├── temporal_decoder.py
│   ├── boundary_refinement.py
│   ├── pipeline.py
│   └── submission.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_exploration.ipynb
│   ├── 03_baseline_possession.ipynb
│   └── 04_error_analysis.ipynb   # planned; synthetic and safe diagnostics
├── tests/
│   ├── test_alignment.py
│   ├── test_features.py
│   ├── test_events_and_live_dead.py
│   ├── test_control.py
│   ├── test_decoder.py
│   └── test_submission.py
├── evaluation/                   # organizer evaluator, not to modify
├── data/                         # restricted; not committed
├── outputs/                      # generated; private/ignored
├── run_pipeline.py               # planned one-command CLI
├── requirements.txt              # pinned runtime deps
└── README.md
```

---

## 5. Detailed data contracts

### 5.1 Data loading

- `games.asset_paths(source='tactical')` resolves assets; `games.require(...)` checks existence.
- Do not hardcode absolute locations, private file names beyond canonical `games.py`, or local host paths.
- `kloppy.metrica.load_tracking_epts(meta_data=..., raw_data=...)` → `to_df()`.
- `kloppy.sportscode.load(...).to_df()` for Smart Tagging.
- Use `games.GAMES[game]['pitch']` for conversion to metres: `(105.3, 68.0)`.
- Align frames once, and assert: `len == N`, contiguous order, half masks, numeric array shape.
- Smart Tagging intervals represented as half-open `[start_frame, end_frame)` in features for consistency; note this is an **analysis convention**, not an official event-boundary guarantee.

### 5.2 Feature table (one row per tactical video frame)

**Existing from v1:**

| Field | Type / units | Origin | Notes |
|---|---|---|---|
| `frame` | int | games/video | Primary key 0-based |
| `half` | int 0/1/2 | games | `0` is ungraded |
| `graded` | bool | games | Allowed scoring window |
| `ball_x`, `ball_y` | float normalized | ATD | May be NaN |
| `ball_ok` | bool | ATD | Both coords finite |
| `ball_missing_run_frames` | int frames | derived | 0 where ball observed |
| `frames_since_ball_seen`, `frames_until_ball_seen` | float frames or NaN | derived offline | Restrict to half |
| `home_count`, `away_count` | int | ATD | Count valid team slots, can exceed 11 |
| `nearest_h_m`, `nearest_a_m` | float metres or NaN | ATD | Closest **track**, not possession |
| `distance_margin_a_minus_h_m` | float m or NaN | derived | Positive → H nearer |
| `abs_distance_margin_m` | float m or NaN | derived | Proximity ambiguity proxy |
| `nearest_team_proxy` | H/A/? | derived | **Never treat as ground truth** |
| `ball_speed2d_mps` | float m/s or NaN | derived | Finite consecutive graded observations only |
| `centroid_speed_H_mps`, `centroid_speed_A_mps` | float m/s or NaN | derived | Needs strong quality gating |
| `active_tag_count`, `active_h_tags`, `active_a_tags` | int | Smart Tagging | Overlaps/correlation; not independent evidence |
| `set_piece_tag`, `set_piece_start`, `set_piece_end` | bool | Smart Tagging | Context and candidate anchors only |

**Proposed additions in v2, subject to implementability:**

| Field | Type / units | Purpose | Missing policy |
|---|---|---|---|
| `ball_visibility_age_s` | float seconds | elapsed time since visible ball | `NaN` if no earlier observation in half |
| `ball_next_visibility_s` | float seconds | future visible ball for offline inference | `NaN` if none before half end |
| `ball_displacement_m` | float m | observable movement between consecutive frames | `NaN` with gaps |
| `ball_speed_quality` | bool/enum | reject suspicious jump from scoring | false on missing/anomalous data |
| `valid_h_fraction`, `valid_a_fraction` | float 0..1 | completeness relative to expected 11 each | 0 if none, clip at 1 for duplicate slots |
| `centroid_valid_h`, `centroid_valid_a` | bool | quality mask for motion | false with insufficient stable tracks |
| `player_motion_robust_h`, `_a` | float m/s or NaN | aggregate robust motion from plausible track displacements | `NaN` if too few valid matches |
| `tag_h_support`, `tag_a_support` | nonnegative score | context from *groups* of tag codes | neutral if no tag |
| `tag_set_piece_type` | categorical | throw-in, goal kick, etc., if tagged | unknown otherwise |
| `tag_boundary_distance_frames` | int/NaN | nearest set-piece interval edge | NaN if none |
| `observability_mask` | bit field or bool set | missing/inconsistent sources | deterministic flags |
| `candidate_transition_score` | uncalibrated evidence | suspected H/A switch | 0 or neutral if unsupported |
| `candidate_stoppage_score` | uncalibrated evidence | suspected live→dead | neutral if unsupported |
| `candidate_restart_score` | uncalibrated evidence | suspected dead→live | neutral if unsupported |

**Avoid leakage:** future observations (e.g. time until ball seen) are okay for *offline inference*, but no ground-truth labels or match-specific human correction may be features. Do not leak halftime state across halves.

### 5.3 Module interface contract (proposed pseudotypes)

```python
@dataclass
class FeatureBundle:
    frames: pd.DataFrame    # N rows, one per tactical frame
    events: pd.DataFrame    # auto-tagging intervals with source fields
    slots: pd.DataFrame     # track slot -> H/A metadata

@dataclass
class EvidenceBundle:
    # Full-length (N,3) or per-half; column order always ['H','A','D'].
    state_log_scores: np.ndarray
    quality: pd.DataFrame
    flags: pd.DataFrame
    event_candidates: pd.DataFrame
    metadata: dict

@dataclass
class PredictionBundle:
    frame: np.ndarray       # arange(N)
    labels: np.ndarray      # length N; H/A/D in graded, X outside
    segments: pd.DataFrame  # half, start, end inclusive, label, duration
    diagnostics: dict
```

Separate pure feature/scoring functions from file I/O. Define and assert field types and units. If scores are not calibrated, name them `score` or `log_score`, never `probability` or `confidence=90%` without calibration.

---

## 6. Evidence models

### 6.1 Observation quality (`observation_quality.py`)

Inputs: ball visibility/trajectory; player validity counts; track continuity; optional homography validity and reprojection error. Outputs: masks and quality weights, plus explicit reasons.

Must distinguish:

1. Ball observed continuously and plausible.
2. Ball observed but trajectory jump/anomaly suspected.
3. Short missing run in otherwise stable candidate phase.
4. Long missing run with no direct ball observation.
5. Missing run straddling suspected team switch.
6. Missing run straddling possible stoppage/restart.

**No fixed gap thresholds yet.** Exploratory classes may use illustrative bins; final thresholds require rule-based justification, synthetic stress tests or allowed external match evidence. A missing ball is **never itself proof** of `D`.

Centroid speed warning: computed centroid movement of a changing set of detected players can jump thousands of metres per second even when no player moves that fast. Replace with robust same-slot displacements **only over plausible consecutive track appearances**, or mask centroid movement unless enough high-quality matched observations exist. Slots still can identity-switch; reject implausible displacements.

### 6.2 Team-control evidence (`team_control.py`)

Inputs: team-wise nearest-ball distances, margin, ball motion quality, proximity persistence, team-tag support, prior context. Outputs: *independent uncalibrated evidence* for H and A and a controlled-ball candidate indicator.

Recommended initial form (illustrative only):

```
score_H = distance_evidence(H) + motion_consistency(H) + independent_tag_context(H)
score_A = distance_evidence(A) + motion_consistency(A) + independent_tag_context(A)
```

Constraints:

- Proximity **generates candidates**; it does not establish possession.
- Keep last controlled team when ball is live/loose/contested until new control is supported.
- A single distance threshold or nearest-team switch cannot force a turnover.
- A pass in flight may remain possession by the passing team though no player is close.
- Treat team counts/ball visibility as *quality context* not as H/A class labels.
- Deduplicate/cap correlated simultaneous tag families (e.g. phases describing both attacking/defending perspectives). Explicit code-to-team semantics must be documented and tested; `Team=N/A` provides no direct team assignment.
- When ball missing, do not fabricate a specific nearest-player location; rely on context and uncertainty.

Potential alternative in later iteration: supervised XGBoost/TCN on legitimately labeled **other games**, with identical interface. Not required for initial v2.

### 6.3 Live/Dead evidence (`live_dead.py`, `event_evidence.py`)

**Separate two questions:** (a) is ball live? (b) if live, who owns it? `D` is a state, not a catalogue of referee events.

Inputs: set-piece type and intervals, possible ball out-of-bounds location, ball motion/restart cues, team formation, player movement, optional video events. Outputs:

- `live_score(t)`, `dead_score(t)` (uncalibrated initially).
- Candidate `live→dead` and `dead→live` boundary windows.
- Optional `restart_candidate_team` (H/A/unknown), **not** automatically an active possession label.
- Evidence reason codes.

Detection hierarchy:

1. **Generic state evidence:** live interaction/motion vs stoppage pattern; avoid simplistic "ball stationary = D" (ball can be still while live).
2. **Event anchors:** set pieces (`Throw-in`, `Corner Kick`, `Goal Kick`, `Free Kick`, `Kick Off`); do not label the entire Smart Tagging span as `D`.
3. **Boundary refinement:** when did play actually stop or resume? A referee event need not be shown clearly in tactical video.
4. **Special cases:** goal, penalty setup, offside/foul, injuries, substitution, VAR, dropped ball are optional evidence or hard-case categories, not mandatory dedicated classifiers. No fabricated annotation exists for unspecified types.

Critical anchor: second graded half begins at 110808 while second kickoff is 113742. These 2934 graded frames are `D` per FAQ. This published boundary fact may be implemented as a **documented legal game rule/anchor**, not a manually derived possession label. Do not hardcode event-specific arbitrary labels elsewhere.

Dead restart policy: while waiting for restart → `D`, even if a team is poised over ball. On verified restart execution → `H` or `A` according to new controlled team. Discard previous-team ownership as the default *after* a dead interval; record it only as context.

### 6.4 Event hypotheses are not labels

Example: players clustering is weak evidence of a stoppage, but also can happen during live corner situations; substitution board in video signals a possible restart situation but does not determine precise dead/live frame. Tags `SET PIECES` may cover live play before/after a restart. Every event cue must have a reliability/quality flag.

---

## 7. Temporal inference and boundaries (`temporal_decoder.py`)

### 7.1 State space and continuity

Output states `S = {H,A,D}` for graded frames. Each half decoded independently. Internal modes may include `stable`, `candidate_switch`, `stoppage_candidate`, `restart_pending`, `uncertain`, but are **not output labels**.

Proposed initial algorithm: log-additive emissions + state-specific transition matrix + Viterbi. Do not assume current `1.5` constant penalty is adequate.

Example structure, **no fixed numerical values specified**:

```text
         to H   to A   to D
from H   stay   HA     HD
from A   AH     stay   AD
from D   DH     DA     stay
```

`HA` / `AH` should require evidence of controlled turnover; `HD` / `AD` depend on stoppage evidence; `DH` / `DA` depend on verified restart and owner evidence. Costs may depend on time-local event features. Review prior state memory and treat D reset explicitly.

### 7.2 Soft duration model vs hard minimum

**Do not impose "possession must last ≥10 frames" as a rule.** A valid brief control/turnover can be shorter. Instead use an optional *soft short-spell penalty* and allow strong evidence to override it. Analyze with synthetic edge cases and allowed external data. If a true semi-Markov/HSMM is implemented, explicitly model state duration and document complexity/limits.

- Avoid unconditionally merging a short `A` between `H` runs.
- Avoid extending `H` across true `D` simply to improve smoothness.
- Do not use public GT to choose duration priors.

### 7.3 Deferred decision / temporal buffering

Offline processing allows the decoder to examine **past and future** evidence. An uncertain frame is not frozen in the input pipeline: evidence is collected, then the entire interval is resolved jointly. Do not merely assign all uncertain frames to the eventual winner. Boundary refinement should choose the best-supported transition frame within a candidate interval.

### 7.4 Boundary refinement

For every proposed state change:

1. Identify a search window near score crossover, ball-contact indications, tagging anchors or restart execution.
2. Use high-quality observed evidence to localize a change point; explicitly mark low-confidence boundaries.
3. Respect half boundaries, continuity, and `D` semantics.
4. For a gap with no sufficient evidence, prefer model-consistent uncertain boundary treatment; do not invent a frame-accurate event based only on duration heuristics.
5. Retain one final deterministic label per frame; diagnostics may record boundary uncertainty separately.

### 7.5 Treatment of `X`

- Outside graded halves `X` always.
- Inside graded halves: default try to output H/A/D. Do not use X as a universal cure: macroF1 counts it as an error and spellF1 forward-fills previous state. Any graded `X` policy requires an explicitly documented reason.

---

## 8. Tactical-video integration (optional but specified)

### 8.1 Activation triggers

Only run local video analysis when *expected value of additional evidence is high*:

- ball missing near suspected H/A turnover or dead/live boundary;
- weak conflicting H/A evidence in the boundary window;
- suspicious ATD ball jump or player mapping anomaly;
- plausible restart not resolvable from tags/tracking;
- short, ambiguous candidate spell whose existence strongly changes segmentation.

A long missing gap alone **must not** trigger decoding its full duration (previous exploration selected 96/150 gaps totaling ~30.9 minutes). Instead schedule short **event-centred windows** around candidate boundaries and optionally expand after unsuccessful attempts. Deduplicate overlapping windows, enforce compute budget, and log trigger reasons.

### 8.2 Homography and quality gate

- Read frame `t` of tactical mp4; use tactical `H_t` matrix for normalized (x,y) on pitch plane, then convert homogeneous pixel coordinates.
- Check homography `valid`, finite matrix and acceptable reprojection quality. Numeric thresholds remain hypotheses.
- Crop around relevant player/ball area *with padding* because an airborne ball may not align with a pitch-plane projection.
- Run an approved, locally accessible ball/object detector or track recovery method; report position, quality, temporal persistence, and evidence type.
- Reject/downweight a single isolated detection, implausible speed jump, projection mismatch or poor visual coverage.
- Do not let video output force a label solely because a ball-like object appears close to a player.
- Fallback safely to tracking/tagging model if video subsystem fails or no authorized detector is installed.

### 8.3 Status

The optional overlay produced by prior notebook was visually plausible on tactical frame 11186 (players approximately coincide with markers), **not a measured detector accuracy test**. No automatic tactical-video inference is implemented in v1. No broadcast integration.

---

## 9. Evidence fusion and observability

Fusion must distinguish these concepts:

| Category | Definition | Example |
|---|---|---|
| Observation quality | Reliability/completeness of the available input | ball NaN, missing players, implausible jump |
| H/A support | Evidence for which team controls ball **conditional on live** | ball-player proximity/persistence |
| Live/Dead support | Evidence whether play is active | restart execution vs waiting |
| Transition support | Evidence state changed and where | multi-frame control handover |
| Video quality | Reliability of optional visual evidence | persistent ball detection, valid projection |

**Do not call these probabilities unless calibrated.** Use deterministic weighted scores/log-scores as a first pass, with bounded contribution per independent evidence family. Avoid double-counting correlated Smart Tagging categories. Missing/weak input → neutral evidence with explicit uncertainty, not arbitrary certainty.

Evidence fusion contract: `state_log_scores` finite shape `(N,3)` in strict order `H,A,D`; NaN and all-zero input must be handled; probabilities normalized **only when numerically meaningful**. Include instrumentation for dominance of a single source.

---

## 10. Testing, acceptance criteria and evaluation policy

### 10.1 Required unit tests (synthetic, no restricted match labels)

1. **Alignment:** frame 0, each half start/end, halftime, frame 199318; kloppy sequential vs raw IdFrame offsets; tag timing conversion.
2. **Missing runs:** single missing frame, long run, head/tail-of-half, whole-half missing, no cross-half leakage.
3. **Team mapping:** metadata all slots match exactly H/A; unknown/conflicting team assignment causes explicit failure.
4. **Distance:** coordinate-to-metre geometry; ball absent; no player from one side; ties.
5. **Motion quality:** player track disappears/reappears, duplicate objects, identity switch, simulated impossible centroid jump.
6. **Tag intervals:** overlaps, `N/A` team, empty/invalid intervals, half-open boundaries, start/end at frame 0/N.
7. **Turnover:** one-frame opposition touch that does not take control; genuine short controlled takeover; long stable possession; ball in flight.
8. **Dead ball:** H→D→H and H→D→A; separate preparation vs execution; second-half graded pre-kickoff D; D breaks same-team spells.
9. **Temporal decoder:** strongly supported brief transition survives; weak flicker suppressed; no half leakage; deterministic ties; all log scores finite.
10. **Video gate:** unavailable video returns neutral evidence; failed homography blocks projection; duplicated windows coalesce; video confidence is not confused with possession confidence.
11. **Submission:** exact 199319 rows; legal labels; no whitespace/NaN; official validator pass; identical bytes on rerun.

### 10.2 Useful automatic diagnostics (not accuracy measurements)

| Diagnostic | Why inspect | Misinterpretation to avoid |
|---|---|---|
| Predicted H/A/D frame proportions | Catch degenerate domination or missed D | Not ground truth proportions |
| Number of spells/mean/median duration | Detect implausible fragmentation | Do not tune to leaked or inferred true count |
| H↔A, H/A↔D transition counts | Locate subsystem issues | Counts are not themselves F1 |
| Distribution of D lengths | Detect pathological stoppage fragmentation | Long D can be legitimate |
| Missing gap length by predicted patterns | Assess reliance on tags and continuity | Endpoint proxy changes ≠ turnovers |
| Feature reliability vs time | Find low quality stretches | High quality ≠ correct possession |
| Video-triggered minutes and success rate | Monitor computational cost and evidence usefulness | A successful ball detection ≠ successful state prediction |
| Sensitivity to synthetic perturbations | Measure robustness | Stability alone does not imply accuracy |

### 10.3 Evaluation discipline

- **Allowed:** official validator for schema; public GT only to confirm frame alignment and label semantics; synthetic controls; algorithmic qualitative diagnostics without manually fixing labels; external-game labels under rules.
- **Not allowed:** compare several variants on public five-minute spellF1 to choose best radius, duration, tag weighting, model or threshold; manually edit match predictions at selected frames; use outside human event labels for this match.
- Do not optimize to a presumed total possession count. Do not claim improvement without an allowed objective benchmark.
- When measuring on legitimately labeled *other games*, report confidence intervals and preserve a held-out set.

### 10.4 Exit gates

**Gate A — Data correctness:** reproducible feature tables, no indexing mismatch, explicit masks, tests pass.

**Gate B — Baseline system:** integrated H/A/D predictions for both halves, finite evidence, valid CSV; diagnostics generated.

**Gate C — Robustness:** missing-ball/identity-switch/false-turnover/restart synthetic tests pass; anomalous centroid not treated as evidence; switch penalties and duration priors inspectable.

**Gate D — Optional video:** local detection works and respects quality gate; failure fallback; benefit evaluated via allowed test methodology.

**Gate E — Submission:** one-command deterministic regenerate + official validator + source/deps/report and data-use compliance.

---

## 11. Prioritized implementation roadmap (AFTER this specification is approved)

The user explicitly requested **analysis and documentation now**, with implementation/pipeline to follow separately. Therefore **no new v2 code** is part of the current deliverable.

| Phase | Priority | Deliverables | Why now |
|---|---|---|---|
| 0. Baseline analysis | Complete | This document and reproducible observations | Sets priorities |
| 1. Feature hardening | P0 | Extend `src/features.py`; quality masks; tests | Prevent contaminated evidence |
| 2. Live/Dead foundation | P0 | `live_dead.py`, `event_evidence.py`, synthetic restart fixtures | `D` currently weak and disrupts spells |
| 3. Team-control evidence | P0 | `team_control.py`; proxy confidence and persistence | Reduce nearest-player flicker |
| 4. Temporal decoder | P0 | `temporal_decoder.py`; state-specific costs; optional soft duration | Reduce fragmentation while preserving true short spells |
| 5. Fusion + run | P0 | `pipeline.py`, CLI, CSV validator, config | Reproducible end-to-end system |
| 6. Diagnostics and stress tests | P1 | `04_error_analysis.ipynb`, synthetic perturbations | Inspect failure modes safely |
| 7. Visual escalation | P2 | `visual_analyzer.py`, crop, gate, fail-safe | Target hard ambiguity, not entire video |
| 8. Advanced inference | P2 | HSMM/TCN/learned component only if justified | Complexity only when evidence supports it |

**Practical coding discipline:** create one module with pure functions and tests, integrate it, observe diagnostics, then move to next. Not an endless notebook chain; notebooks are laboratories. The final pipeline is a single deterministic command.

---

## 12. Open decisions and experiment registry

| ID | Question | Present position | Legitimate way to decide |
|---|---|---|---|
| O-01 | Ideal distance scale or control-radius model? | Prefer continuous evidence; not one hard radius | Domain priors, synthetic cases, permitted external match labels |
| O-02 | When does a brief run represent genuine control? | Soft penalty, not hard 10-frame cutoff | Synthetic edge cases and external matches |
| O-03 | Which tag families provide independent support? | Deduplicate/cap overlaps | Schema audit, co-occurrence analysis, external evidence |
| O-04 | Can player motion indicate stoppages reliably? | Current centroid speed is unsafe | Robust motion features + tests |
| O-05 | How to localize dead/live boundary? | Event anchor + boundary refinement | Synthetic restart cases, local visual evidence, permitted data |
| O-06 | How to resolve multi-second missing gaps? | Bidirectional inference + context, optional video | Missing-data stress tests, external match validation |
| O-07 | What transition costs and duration priors? | State-dependent; values not fixed | Model assumptions + allowed validation |
| O-08 | When should video run? | High-impact ambiguous boundary windows | Runtime / detection QA and permitted metrics |
| O-09 | What counts as high-quality video evidence? | Persistent, plausible, spatially consistent detection | Local QA/test footage, synthetic occlusions |
| O-10 | Should internal scores be calibrated? | Initially scores only | Calibration using independent allowed labels |
| O-11 | What is target spell count? | **Not a target for tuning** | Only official evaluator can measure real spellF1 |

Record all experiments with config, seed, code revision, objective, and whether they use synthetic or allowed external labels. Do not retroactively set hyperparameters from forbidden feedback.

---

## 13. Instructions for the next coding LLM

1. Read **this document** plus `games.py`, all six competition `.md` files, `src/features.py`, and notebook 3 code/output before editing.
2. Treat all values explicitly marked **observed** as descriptive and all other numeric design parameters as **hypotheses**. Baseline v1 code parameters are not official or tuned.
3. Do not rewrite `games.py` or official `evaluation/` scorer to improve a score. Use `games.asset_paths` and `games.half_video_blocks`.
4. Implement modules with deterministic, testable interfaces and per-frame/segment diagnostics. Do not silently hide missingness, fill NaNs as real observations or treat tag names as exact labels.
5. Honor `D` semantics; team before a dead stretch does not dictate team after restart. Respect first graded second-half pre-kickoff `D` interval.
6. Do not treat proximity proxy segments as actual possessions. No absolute minimum possession length hard-coded absent approved rationale.
7. All video inference must be algorithmic, local and optional. No manual fixes tied to particular frames; no redistribution of match data.
8. Include unit/synthetic tests before integration, then generate and validate submission as a separate step. Never claim spellF1 from schema validation.
9. Request clarification only for genuinely missing project-specific resources; otherwise implement the next agreed phase to completion with assumptions documented.
10. Preserve the user's explicit sequence: **analysis → v2 specification → separate implementation**. Do not launch a new pipeline merely because it appears in this planned architecture.

---

## 14. Short summary for stakeholders

Notebook 3 proved the project can load the tactical data, construct features, produce one legal label per frame and pass official submission-format validation. The predictions, however, are highly fragmented: 1,152 possession spells with a median of 2.04 seconds, alongside 154 short-median dead-ball segments. Long missing-ball periods, tracking anomalies, and overlapping Smart Tags complicate inference. These are *diagnostic findings*, not known official scoring failures.

Version 2 therefore formalizes a modular approach with separate team-control and live/dead evidence, explicit observation quality, offline temporal state inference, boundary refinement, and optional local tactical-video analysis. Its next implementation priority is to improve evidence quality, the `D` state and transition logic—not to select a bigger neural network or tune parameters against prohibited ground truth.
