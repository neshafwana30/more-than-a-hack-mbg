# Barça Possession Intelligence — System Specification v1.0

**Status:** Draft for implementation; architecture agreed in discussion, algorithmic parameters unvalidated.  
**Audience:** next coding LLM, software engineers, ML engineers, analysts.  
**Language:** Indonesian, with consistent English names for APIs and modules.  
**Source of official requirements:** `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `FAQ.md`, `RULES.md`, `SUBMISSION.md`.  
**Source of proposed design:** team discussions. All sections tagged **[DESIGN]** are recommendations, not organizer requirements.

## 0. Executive objective

Build a deterministic, fully reproducible *offline* pipeline that reads available match assets for **FC Barcelona (home=H) vs Athletic Club (away=A)** and produces exactly one label per tactical-video frame: `H`, `A`, `D`, or `X`. Optimize correct *possession intervals*, not independent frame accuracy. The official leaderboard ranks by `spellF1`; `macroF1` is the first tie-break. Build a useful baseline first, then evidence fusion and optional selective visual verification.

**Non-negotiable:** no manual labeling/correction of this match, no training or hyperparameter tuning on the public 5-minute ground truth, no third-party hosted/cloud LLM uploads of restricted competition data, and no redistribution of raw or reconstructive derived match data.

## 1. Official challenge contract [SOURCE]

### 1.1 Labels and semantics

- `H`: Barcelona has possession **while ball is live**.
- `A`: Athletic Club has possession **while ball is live**.
- `D`: ball **not in play**; team entitled to restart does **not** count as live owner.
- `X`: abstention / no answer. No `X` ground truth among graded frames, but `X` may occur in submission.
- Possession changes only when the opposing team gains **controlled** possession. A mere touch, deflection, or aerial duel without subsequent control is not sufficient. During loose/contested play, keep previous controlling team (when ball live).
- A *spell* is a maximal contiguous run of `H` or `A`. `H H D D H H` is **two** H possession spells. `D` is not a spell.
- Ground truth is mechanically derived from **corrected tracking**: `D` on dead frames, otherwise its owning team. No smoothing after derivation. Supplied ATD is **automatic, uncorrected**.

### 1.2 Frame and timeline contract

- Tactical video: 199,319 frames, 25 FPS, 1920×1080; indices 0–199318. This is the **only submission timeline**.
- Graded first half: frame **11185–90651** inclusive (79,467 frames).
- Ungraded halftime: **90652–110807**.
- Graded second half: **110808–190197** inclusive (79,390 frames). First 2,934 frames (110808–113741) of this range are dead ball prior to kickoff at 113742; this is a source-provided fact, **not an invitation for frame-specific hacks**.
- Ungraded before first half: **0–11184**; after final whistle: **190198–199318**. All submission rows still required. `X` in ungraded regions is fine.
- ATD `frame_id` is **1-based**: `submission_frame = frame_id - 1`.
- Smart Tagging timestamps are seconds since tactical video start: `frame = round(seconds * 25)`; inspect interval convention before rasterizing endpoints.
- Tactical homography `frame` already 0-based.
- Broadcast has 164,488 frames, **not synchronized** to tactical; out of scope for v1.

### 1.3 Sources and their known limitations

| Source | Shape / content | Proposed role | Known caveats |
|---|---|---|---|
| Tactical ATD | 199,319 frame records; ball x/y, 46 player track slots with x/y/s; team membership available via tracking metadata | Main geometric evidence | ball detected on 56.9% of all frames / 59.3% of graded frames; mean ~20.5 player objects on graded frames; NaNs, fragmented slots, identity swaps; slot ID is **not** shirt number/person |
| Tactical Smart Tagging | 638 intervals; timestamps, code, tags such as `Team`, `Half`, `Type` | Context and event hypotheses | automatically inferred, may be wrong; interval is phase, **not** possession; `Team=N/A` on 36 instances including all throw-ins |
| Tactical homography | 199,319 matrices 3x3; `valid`, `reprojection_error` | Project pitch positions to image for overlay/ROI/video analysis | `valid=1` across tactical recording; `reprojection_error` still varies; ground-plane transformation not necessarily accurate for airborne ball |
| Tactical video | full reference recording | Master timeline; optional machine visual evidence | video decoding and model inference cost; cannot be hand-labeled |
| Broadcast assets | second unsynchronized recording | Excluded from v1 | separate cuts/replays, homography usable only 65.8% |
| Public ground truth | 7,500 real-label frames (11185–18684), 17 possession spells; other rows X | **Pipeline correctness check only** | **not** training/tuning/selection/leaderboard |

Pitch reference dimensions: ATD metadata 105×68 m, `games.py` uses 105.3×68.0 m measured from corrected tracking. When converting normalized coordinate deltas to metres, explicitly choose/document one consistent source. Example: `sqrt((105.3*dx)^2 + (68*dy)^2)`.

### 1.4 Deliverable and scoring contract

Required CSV (`UTF-8`): exact header `frame,label`, exactly **199,319** data rows, `frame=0..199318` in order, labels only `H/A/D/X`, no blanks/no extra pandas index. Validate with:

```bash
python evaluation/validate_submission.py path/to/TEAM_NAME.csv
python evaluation/run_evaluation.py  # public only; correctness, NOT model selection
```

Additional deliverables: public code repository (pinned dependencies, instructions, deterministic regenerate command) and short `.pdf` or `.md` technical report; one form submission per team, last edited response before **17 November 2026** is scored. Team eligibility: **3–5 members** for Innovation Challenge/Final Pitch. **No match assets in public repository**. Baseline `baseline_smooth50`: full-match **spellF1 0.2433** (published organizer result). This is a reference, not directly measurable by participants on private truth.

Scoring:
- `macroF1`: unweighted F1 average across `H/A/D`, frame-exact; `X` on graded frames is a miss.
- `spellF1`: one-to-one same-label possession spell matching by temporal IoU, averaged at IoU thresholds **0.3, 0.4, 0.5, 0.6, 0.7**. On this metric, `X` is forward-filled from previous label **within each half**, back-filled at the start; it is **not** a free unknown state.
- Never choose thresholds based on the public 5-minute score. Evaluate algorithmic consistency via synthetic sequences, legal external game data, stress tests, and exploratory distributions that do not use ground truth as a tuning target.

## 2. Scope and guiding principles [DESIGN]

**In scope (v1):** tactical ATD, tactical Smart Tagging, optional tactical homography + local tactical video when justified, offline batch processing, confidence/evidence scoring, live/dead subproblem, team control subproblem, temporal decoding, CSV generator/validator, diagnostics. **Out of scope (v1):** broadcast alignment, player re-identification as prerequisite, exact taxonomy detection for every refereeing event, manually fixing any match frame.

Principles:
1. **Observation != event != state**. Observations (coordinates, visual measurements, tags) support hypotheses (throw-in/turnover/restart) used by the state estimator (`H/A/D`).
2. **Team possession != nearest player**. A pass in flight can remain with prior owning team; contact != control.
3. **Missing ball != dead ball**. Separate absence of detection from out-of-play state.
4. **Dead/alive and home/away are separate estimators**, combined by a joint temporal decision stage.
5. **Temporal evidence may revise earlier ambiguous frames**. Offline look-ahead allowed; do not literally stop reading frames.
6. **Short spells are suspicious, not impossible**; no absolute minimum-duration rule without legitimate evidence.
7. **Video evidence is optional and quality-gated**. An unusable video output must not override more reliable evidence.
8. Every module emits provenance/quality indicators and is independently testable.

## 3. Proposed module architecture [DESIGN]

```text
TACTICAL TIMELINE (frame 0..199318)
  ├─ ATD parser → ball/player frames + quality flags
  ├─ Smart Tagging parser → frame-aligned interval context
  ├─ Homography parser → matrices + error / validity
  └─ Tactical video reader (lazy, optional)
                    ↓
          Feature & candidate layer
  ├─ ball motion/visibility, player-ball distances
  ├─ home/away candidate scores, team geometry
  ├─ tagging/restart evidence
  └─ anomaly/missingness/observation quality
                    ↓
         Parallel evidence estimators
  ├─ Live/Dead evidence + event hypotheses
  └─ H/A team-control evidence when live
                    ↓
       Uncertainty/transition detector
           ↓ conditional escalation
      selective video + quality gate (optional)
                    ↓
    Joint temporal decoder + boundary refinement
                    ↓
        H/A/D state per frame (+ diagnostics)
                    ↓
         CSV generator + validator
```

### 3.1 Module contracts (logical, not fixed filenames)

| Module | Required input | Output | Acceptance |
|---|---|---|---|
| `load_align` | ATD meta/raw, tagging xml, homography csv | aligned arrays keyed by `frame_idx` | exactly 199319 rows; frame offsets verified; no broadcast indexing |
| `quality_features` | tracking arrays, homography metadata | observed/missing flags, gaps, anomaly scores | no NaN swallowed as numerical 0 |
| `candidate_detector` | ball, players, team affiliation | top-K nearby players for both teams; distance, relative motion | no assumption that track slot IDs are stable |
| `team_control_evidence` | candidates, ball motion, tagging context | raw evidence scores H/A + availability mask | do not equate nearest player with certainty |
| `event_evidence` | tagging, motion, optional visual | probabilities/scores for restart/stoppage/turnover event hypotheses | tag intervals are *evidence*, not label mapping |
| `live_dead_evidence` | event hypotheses, motion, tracking signals | raw evidence for live vs dead | no `ball_missing ⇒ D` shortcut |
| `ambiguity_detector` | evidence margins, quality, candidate transitions | frame intervals + reason codes | handles overlapping triggers, merges neighboring windows |
| `video_analyzer` | tactical video windows, homography, candidate pitch positions | visual ball/player observations, detector scores, quality flags | execute **locally under data-use terms**; discard low-quality measurements |
| `fusion` | all evidence, availability masks, provenance | frame-wise emissions / comparable state scores | absent evidence is neutral, not a vote against a state |
| `temporal_decoder` | emissions, transition/duration penalties, half boundaries | contiguous frame labels H/A/D and boundaries | resets at half boundary, handles D interruption |
| `export_validate` | frame labels | exact CSV and validation report | deterministic byte-identical output, exactly 199319 rows |

## 4. Input and intermediate schemas [DESIGN]

These are **proposed normalized interfaces**, not claimed raw columns. Inspect actual files before implementation.

### 4.1 `FrameObservation` (frame-level, sparse / nullable)

| Field | Type | Explanation / provenance |
|---|---|---|
| `frame_idx` | int | Tactical frame, 0-based |
| `time_sec` | float | `frame_idx / 25` |
| `half_id` | int / null | 1/2 for graded half window; otherwise null |
| `ball_x_norm`, `ball_y_norm` | float / null | ATD normalized coordinates |
| `ball_observed` | bool | both ball coordinates present and credible |
| `players` | collection | track slot index, team, normalized x/y, speed; optional missing fields |
| `tag_active` | list | overlapping Smart Tagging intervals with code / Team / Type / boundaries |
| `H_matrix` | 3×3 / null | tactical homography at frame |
| `homography_valid`, `homography_error` | bool, float/null | quality metadata |
| `video_frame_available` | bool | if optional video stage enabled |

### 4.2 `FrameFeatures` (derived)

| Feature | Role |
|---|---|
| `dist_home_min_m`, `dist_away_min_m` | nearest observed players by team, in metres |
| `topk_home`, `topk_away` | candidate player sets + ball distances |
| `relative_motion_home`, `relative_motion_away` | contextual ball/player trajectory evidence where calculable |
| `ball_speed_est`, `ball_acceleration_est` | derived from valid consecutive positions (guard against missing gaps and improbable jumps) |
| `ball_gap_length`, `tracking_anomaly_score` | reliability / escalate signal |
| `tag_team_context`, `tag_setpiece_type`, `tag_conflict` | contextual event evidence |
| `control_score_H`, `control_score_A` | uncalibrated team evidence until validated |
| `live_score`, `dead_score` | uncalibrated live/dead evidence |
| `transition_score`, `obs_quality` | confidence proxy and temporal changes |
| `video_trigger_reason`, `video_quality` | audit trail if video used |
| `final_state`, `boundary_score` | decoded label and boundary diagnostics |

**Schema safeguards:** distinguish `null/NaN` from zero; store source and timestamp for inferred observations; make team orientation/coordinates explicit; do not overwrite original ATD; don't export reconstructive match-level features publicly.

## 5. Core algorithms and decision logic [DESIGN]

### 5.1 Candidate identification is not possession identification

1. On ball-observed frame, calculate metre distances to valid player detections.
2. Build top-K candidate set for **each team**.
3. Optional *candidate radius* is a **search/gating heuristic**, not hard proof of control.
4. Use multi-frame behavior where feasible: persistent proximity, relative ball-player motion, ball trajectory change, spatial crowding.
5. When ball missing, mark missingness and use memory, team context, and before/after evidence. Do not force a candidate.

Radius must not be optimized with public GT. Start with documented geometry assumptions, investigate unlabeled distributions, or fit from other permitted games.

### 5.2 Dual state evidence

Generate independent **live/dead evidence** and **H/A conditional control evidence**. An event hypothesis supports, but does not unilaterally set, the final state.

- Event categories worth considering: `out_of_play`, `stoppage`, `restart_preparation`, `restart_executed`, `suspected_turnover`, `unknown`.
- Smart Tagging has set-piece categories, **not** guaranteed foul, injury, VAR, substitution, dropped-ball labels. Avoid building an exhaustive event classifier before a working dead/alive baseline.
- Distinguish *team entitled to restart* from *team currently possessing a live ball*.
- Dead interval boundaries need refinement: stop at true cessation of play; remain D through preparation; transition to H/A at live resumption.

### 5.3 Internal confidence representation

Use separate outputs:
- `observation_quality` (`Q_t`): reliability/completeness of sources, not possession probability.
- `live_dead_evidence` (later `P(live_t)` if calibrated): is play active?
- `team_control_evidence` (later `P(H|live)`, `P(A|live)` if calibrated): who controls while live?
- `transition_evidence` (`score_t`): evidence of H↔A or D↔live boundary.

**Do not call arbitrary weighted scores calibrated probabilities.** Explicitly handle unavailable feature signals with masks or neutral evidence.

### 5.4 Deferred decision and temporal decoding

Process offline in passes rather than permanently labeling one frame at a time:

```python
obs = load_and_align()                    # indexed to tactical frame
features = compute_frame_features(obs)
emissions = estimate_live_dead_and_team_control(features)
windows = find_ambiguous_windows(emissions, features)
if config.enable_video:
    video_evidence = analyze_selected_windows(windows, obs.homography)
    emissions = fuse_if_quality_passes(emissions, video_evidence)
states = temporal_decode(emissions, half_boundaries=...)  # H/A/D
states = refine_boundaries_if_supported(states, emissions)
write_validate_csv(states)
```

Recommended first implementation: **finite state machine with hysteresis, an explicit D state, buffered transition candidates**, followed by an HMM/Viterbi or duration-aware HSMM experiment. This is a choice to test, not a proven winner. Use temporal continuity penalty to suppress flicker but permit strongly supported short spells.

### 5.5 Duration-aware model rather than blanket minimum

Do **not** universally replace any state lasting <10 frames with neighbors. Short controlled possession can be real and dead-ball intervals matter. Compare:
1. no duration filter,
2. a simple short-run filter **as baseline/ablation** (never overwrite D blindly),
3. soft duration penalty with supporting evidence,
4. optional explicit duration model.

The cost of transitions should depend on control evidence, not a fixed prohibition. All parameters chosen without public GT tuning.

### 5.6 Restart handling and state memory

Maintain at minimum `last_live_team`, `current_game_state`, `restart_team_candidate`, `restart_type_candidate`, `transition_candidate_start`, `evidence_quality`.

- `H/A → D` when stoppage/out-of-play evidence clears temporal decision threshold.
- During `D`, do **not** label the team who will restart as H/A.
- `D → H/A` only after sufficiently supported resumption; determine restart possession anew from tagging/context/ball motion; do **not** simply reuse `last_live_team`.
- `H ↔ A` when opposing **controlled** possession supported; a deflection or transient closeness is insufficient.
- Reset temporal state across first/second half; respect source-provided evaluation blocks without frame-specific output hacks.

### 5.7 Video trigger and visual quality gate

**Trigger candidates** (no fixed thresholds until exploration):
- ball missing near a plausible turnover;
- H/A evidence difference small;
- `live↔dead` boundary ambiguous;
- tracking position jump / inconsistent motion;
- very short uncertain possession;
- expected restart without reliable corresponding evidence.

Video analysis uses **tactical** frames only, ROI projected by homography around candidates with padding; optionally full-frame search if ROI fails. Homography is ground-plane; airborne ball may project differently. Possible visual outputs: detected ball pixel and score, temporal visual track, candidate interaction, restart-motion evidence. Visual outputs have quality/consistency checks; low-quality outputs are ignored, never blindly overwrite ATD. No manual frame fixes. **Always check data license before processing in external infrastructure**.

## 6. Test and validation plan [DESIGN]

### 6.1 Data-level tests

- Exact frame count and alignment; assert `ATD frame_id=1` maps to tactical `0`.
- Check Smart Tagging timestamp-to-frame mapping and casing (`Team`, `SET PIECES`, etc.).
- Check homography existence, validity, reprojection error, invertibility if used.
- Quantify NaN, missing-run length distribution, player count per frame, coordinate ranges, duplicate track count, ball speed jumps.
- Confirm no raw match assets leave private local data area or enter public commits.

### 6.2 Behavioral unit tests using synthetic sequences

- `H H H A A`: one real H→A turnover.
- `H H D D H H`: two H spells.
- Brief A evidence within stable H should not always cause fake change.
- Strong short A control between H intervals must be **possible**.
- Missing-ball interval with H before/after does not automatically produce D.
- Missing-ball interval between H and A triggers boundary uncertainty.
- D before restart must not become A just because tagging names A.
- Restart after D may change previous possession team.
- Confidence scores remain separate from source availability.
- No cross-half state forward-filling in internal decoder.

### 6.3 Evaluation discipline

- Run official validator on final file.
- Run public evaluator **only to check pipeline correctness and label semantics**; do not select algorithms or tune hyperparameters using its score.
- Use legal external matches, simulation, or self-supervised consistency analyses for model selection, with assumptions clearly recorded.
- Track descriptive diagnostics: predicted spell counts, spell duration distribution, H/A/D proportions, switch rate, missing-ball coverage, video trigger frequency, quality gate acceptance, resource cost. Do not tune these to unpublished reference statistics.
- Keep an experiment manifest (`git_sha`, model/config hash, data source list, random seed, runtime, output file checksum) so reruns match exactly.

## 7. Suggested repository layout [DESIGN]

```text
repo/
  README.md
  requirements.txt                    # or uv.lock/poetry.lock, pinned
  config/default.yaml                 # parameters, documented provenance
  src/
    io/ (atd.py, tagging.py, homography.py, video.py, alignment.py)
    features/ (ball.py, players.py, quality.py)
    inference/ (candidate.py, events.py, deadball.py, control.py,
                fusion.py, temporal.py, boundaries.py)
    visual/ (triggers.py, detector.py, quality_gate.py)
    submission/ (export.py)
  notebooks/ (01_data_audit.ipynb, 02_feature_exploration.ipynb,
             03_synthetic_cases.ipynb)
  tests/ (test_alignment.py, test_state_machine.py, test_export.py)
  docs/ (technical_report.md, assumptions_log.md, experiment_log.md)
  scripts/run_pipeline.py
  data/                                # private; gitignored, DO NOT COMMIT
  outputs/                             # local generated match outputs; gitignored
```

This is a suggested layout, **not** an existing verified repository tree.

## 8. Iteration roadmap [DESIGN]

1. **Audit & alignment:** inspect permitted files, schema, missingness, tag overlap and frame indexing in notebook; no label-driven tuning.
2. **Baseline V1:** ATD nearest-team candidates + conservative temporal memory + explicit dead/alive heuristic + deterministic CSV.
3. **Evidence V2:** Smart Tagging as weak event/context evidence, robust missing-ball handling, uncertainty and transition buffering.
4. **Decoder V3:** compare state machine/hysteresis vs HMM/Viterbi, with soft duration penalties and boundary refinement on **permitted** test settings.
5. **Video V4 (conditional):** homography ROI + local ball/interaction detector only for high-value ambiguous windows; require measurable legitimate benefit and acceptable compute.
6. **Submission:** package reproducible repository, technical report, validated team-named CSV.

## 9. Open questions / hypotheses to resolve in notebooks

1. What actual ATD player columns and metadata provide affiliation? How is missing affiliation handled?
2. How long are missing-ball runs and how do they relate to tagging intervals without assuming truth?
3. How often do Smart Tagging events overlap, conflict, or cover very different lengths? Which `SET PIECES` type values actually appear?
4. How do player-ball proximity and motion features behave on unlabeled frames? Can they be defined without stable track IDs?
5. How do we infer a dead-ball boundary when no whistle/referee annotation is provided?
6. Which events, if any, offer reliable evidence of `D→live` start, not just restart preparation?
7. What are the smallest visual ROIs that remain robust to homography error and airborne ball?
8. How do we calibrate or at least compare evidence scores without forbidden labels?
9. What legitimate external annotated games, if any, are available for supervised calibration/training?
10. Which algorithm/hyperparameter choices remain robust under synthetic stress tests and distribution shifts?
11. Are all intended detection models accessible, redistributable/reproducible, and compliant with competition data-use restrictions?

## 10. Instructions to the next coding LLM

- **First read the six attached official `.md` files**, prioritizing `RULES.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`. This document is a plan; official files take precedence.
- **Do not invent raw column names or fabricate empirical results**. Use a real read of ATD files and metadata before implementing schemas.
- **Do not assume match data is attached**; request/use the participant's local authorized bundle only if available. Never upload restricted data to third-party cloud LLMs.
- Implement modular functions, typed/specified inputs, explicit NaN masks, half resets, structured logs, deterministic seeds, and tests.
- Do not tune on the public ground truth and do not use frame-specific hardcoded label corrections.
- Build a minimal end-to-end baseline before optional video/computationally expensive components.
- Validate `frame,label` count/sequence with the official validator and record reproducibility evidence.

## 11. Source map

- `CHALLENGE.md`: problem, labels, possession semantics, corrected tracking ground truth.
- `DATA.md`: asset structure, ATD/tagging/homography, counts, alignment, graded windows, public GT.
- `EVALUATION.md`: spellF1, macroF1, abstention behavior, baseline evidence, public-window warning.
- `FAQ.md`: edge cases, second-half kickoff, practical constraints, hosted LLM caution.
- `RULES.md`: algorithm-only labeling, external data restrictions, eligibility, inspectability, asset confidentiality.
- `SUBMISSION.md`: CSV formatting, three deliverables, validator, deadline.

**Version policy:** v1.0 freezes scope and contracts, **not** numerical thresholds. After actual authorized exploratory analysis, publish v1.1 with verified schemas, feature definitions, test evidence, and config rationale.
