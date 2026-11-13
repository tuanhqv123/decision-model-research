# Questions that test the R3 reasoning (2026-11-11)

Each question: what we know now (with source), or which plan task measures it, or that only the operator can decide.
Plan: `docs/superpowers/plans/2026-11-11-r3-elements-trajectories.md`.

## A. The benchmark

| # | Question | Answer now |
|---|---|---|
| A1 | How is the step budget set? | `int(10 × task.complexity)` of the **instance** (`suite_utils.py:491-502`). Our budget table matches for 116/116 classes; the runs use the instance value, which differs on 4 IR tasks (SportsTracker ×3, TasksDueNextWeek: 10/20/10/10 vs table 12/22/16/12) — the runs are right, the table is class-level (used only for post-hoc buckets of task-completion tasks, so no number moves). Complexity: 45 tasks at 1.0 (10 steps), 7 at 6.0 (60), max 12.0 (120). |
| A2 | Does the agent see complexity or the budget? | No — official agents see the goal and the screen only. R3 must not feed it (plan constraint, test `test_no_task_metadata_reaches_the_state`). |
| A3 | What is "done"? | The agent emits `status` (`goal_status = complete`); the episode ends at once and the task checker runs on the device. Success needs the checker > 0.5 **and** this declaration (`suite_utils.py:261`). Declaring too early ends the episode with the task unfinished: R1 s1 `ExpenseDeleteMultiple` (delete 3 expenses) deleted one (click, click, CONFIRM) and declared done at step 4; `AudioRecorderRecordAudioWithFileName` changed recording settings and declared done at step 8 without recording; `MarkorCreateFolder` toggled search options and declared done at step 6. |
| A4 | Are any of the 116 tasks infeasible (should end with `infeasible`)? | Not checked yet — Task 5 reads the registry. |
| A5 | Which tasks cannot be expressed by v3 at all? | Canvas/map targets (BrowserDraw, SimpleDrawPro, OsmAnd markers); text neither in goal nor on screen; IR answers that need arithmetic beyond count (Task 4 coverage report). |

## B. The data (does AndroidControl help or hurt?)

| # | Question | Answer now |
|---|---|---|
| B1 | Is AndroidControl the reason we are bad? | Partly, through its conventions, not its skills: every episode ends in `status` (13,603/13,603, no negatives), goals usually name the app, demonstrations never contain an error. And the model trained on 15k of it (B0′) is at the do-nothing floor (7.0% vs 6.6%). But MobileRL reaches 50.2 with AndroidControl (35.6k steps) **plus** 62.2k other steps, so it is usable as part of a mix. Measured answer: plan Task 9 (AC-only vs AW-only vs mix on an AndroidWorld-format validation set). |
| B2 | How much of AndroidControl is in AndroidWorld-like apps? | Keyword estimate on the 8,984 cur15k episodes: 21.9% touch an AndroidWorld app category; Settings toggles 52 episodes, Expense 8, SMS 7, Files 13, Chrome 1. The heavy AndroidWorld apps are almost absent. |
| B3 | Which other datasets have screenshot + element tree + trajectory? | AndroidControl (yes, a11y forests). AndroidLab's AndroidInstruct (XML mode; 10.5k traces / 94.3k steps reported, 726 traces / 6,208 steps used) — release contents unverified. AMEX (104k screenshots, element annotations, instruction chains) — whether the raw tree ships is unverified. GUI-Odyssey and AITW — screenshots only, no tree (unusable for element choice without a detector). OpenMobile (2.8k tasks / 34k steps in the AndroidWorld apps) — release unconfirmed. **Rule M1: download and parse before counting on any of them.** |
| B4 | Best-matching data source? | Our own rollouts in the AndroidWorld emulator: identical screens, element extractor and action space (plan Task 6). |
| B6 | Does our harness generate the official seed-30 instances? | Yes. Recomputed with upstream `create_suite._get_instance_seed` (`sha256(f'{seed}_{name}_{i}') % 2**32`, `suite_utils.py:149-152`): seed 30 gives exactly the goals our runs used (MarkorCreateNote 'wise_yacht.md / Ignorance is bliss.', ExpenseDeleteMultiple 'Ride-Sharing, Flight Tickets, Taxi Fare'). |
| B5 | Are AndroidWorld's tasks split into train/test? | No. The registry has 116 tasks (+ MiniWoB 92). "Training on AndroidWorld tasks" elsewhere means other random instances of the same 116 templates. That is operator decision D1. |

## C. The model input/output

| # | Question | Answer now |
|---|---|---|
| C1 | What does the model get per step? | Goal + current screenshot + previous trajectory as text lines (action, element label, changed/no change; last 12) + current element list (≤ 40) + app list. No previous screenshots in the first R3 model. **Operator 2026-11-11: add 2 past screenshots after the first model is trained** (plan Task 12; HiconAgent: +9 step-SR on AndroidControl with 2 past screenshots at 2.7x compute; for a single-pass model the extra cost is image encoding only). |
| C2 | Is text history enough to avoid loops? | Unknown. Measured by first-encounter repeat and no-effect rate in the pre-registration. |
| C3 | Does the 40-element cap drop the target on AndroidWorld screens? | Offline (AndroidControl, AW extractor): gold in list on 93.8% of rows. Live AndroidWorld: measured from rollouts (Task 6/9). |
| C4 | Does the image size matter? | Not measured (one 858 arm was within seed noise). v3 uses 858 = d1's own size for this screen (CONTRACT §2.2). |
| C5 | Is every output executable? | Yes by construction + CONTRACT §6 checks before any run. |

## D. Training and evaluation

| # | Question | Answer now |
|---|---|---|
| D1 | May we train on other instances of the 116 templates? | **Operator.** Many leaderboard entries do and say so; it inflates the score relative to unseen templates. |
| D2 | May an external model label or demonstrate at training time (never at inference)? | **No (operator, 2026-11-11).** AndroidWorld-app trajectories come only from our own model's checker-verified rollouts. |
| D3 | Which offline number predicts the online one? | Teacher-forced "episodes right at every step" was 11–12% on AndroidControl validation vs 12–15% online (plausibility only). Task 9 adds an AndroidWorld-format validation set. |
| D4 | How many seeds are needed? | 3, as for R1 (seed sd ≈ 2 points offline; online paired sign test over 116 tasks). |
| D5 | How long? | ~1 day per seed-epoch (INFERRED; Task 8 measures) + emulator hours for rollouts on two hosts. |
