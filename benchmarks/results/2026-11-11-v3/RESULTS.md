# v3 seed 0 — results of the final checkpoint (ckpt_1824)

Every number below is recomputed from the files in this folder. Run card: `docs/research/RUNCARD-v3_s0.md`;
pre-registration: `docs/research/PRE-REG-2026-11-11-v3.md`. Training finished 2026-11-11 11:36 UTC (`TRAIN exit 0`).

Checkpoint used: **ckpt_1824** (the last). It is also the pre-registered choice (highest `validation_v3` step accuracy,
ties -> later step) and it ties the highest "goals with every step right" (14). Other checkpoints' files are kept in
`validation/` for the record; they are not reported.

## 1. Validation (`validation_v3`: 137 official AndroidControl validation goals, every step, 827 rows)
Teacher-forced: every step gets the demonstrator's correct history. Scorer: `run_acv2.score_row` (point inside the gold
element; terminal `status` row included).

| | ckpt_1824 |
|---|---|
| **Goals with every step right** | **14/137** |
| Steps right | 524/827 = **63.4%** |
| Steps right without the 137 terminal `status` rows | 60.6% (690 rows) |
| Terminal `status` rows right | 77.4% (106/137) |
| Action type right | 81.3% |
| First error of a failed goal | start 47 · middle 71 · end 5 |
| Premature `status` (gold is not `status`) | 31 |

Per action type (right / rows): click 238/420 · scroll 55/86 · status 106/137 · type_text 38/53 · wait 27/56 ·
navigate_back 16/29 · open_app 44/45 · long_press 0/1.

**Caveat (found 2026-11-11):** on these rows the app list is the gold app + 24 distractors (`row_app_options`), so the
gold app is always offered on `open_app` steps; open_app accuracy here is optimistic. AndroidWorld (33 real apps) and
AC-High (one fixed list) are not affected.

Arithmetic check: at 63.4% per step and these goal lengths, independent steps would give ~16 goals all right; 14 observed.
Half of 6-step goals all right needs ~89% per step.

## 2. Held-out AndroidWorld templates (`awgen_heldout_v3`: 19 templates never trained, 463 first steps)
App chosen from the emulator's 33 launchable apps: **463/463**. Not yet split by whether the goal names the app (most do);
until it is, this shows copying a named app, not inferring one.

## 3. AndroidControl-High (InfiGUI test = 1,543 official test goals, 8,444 steps; InfiGUI's scorer verbatim)
Standard protocol: every action allowed (InfiGUI's own prompt offers `terminate`). App list: one fixed list of 444
(model's 297 + every app the test phones need). The model also reads the a11y element list (most published models read
pixels only).

| | ckpt_1824 |
|---|---|
| **SR (exact match)** | **67.10%** (5,666/8,444) |
| Type | 80.97% |
| Grounding (InfiGUI `click_match_acc`) | 67.70% |
| Goals with every step right | 374/1,543 (24.2%) |
| `status` predicted (always wrong: the benchmark has no end step) | 539 (6.4% of steps) |

Per gold action (type / exact): click 87.4 / 67.7 (5,074) · swipe 60.4 / 55.9 (1,211) · type 92.9 / 80.7 (632) ·
open 98.0 / 91.9 (608) · wait 53.3 / 53.3 (567) · system_button 53.6 / 53.6 (343) · long_press 0 / 0 (9).

**Adapted variants, NOT the standard number** (AC-High has no end step; report only as separately labelled adaptations):
* A — `status` excluded at decoding (best other action of the same pass, from the saved readouts;
  `benchmarks/train/actest_nostatus.py`): SR **70.63%** (5,964), type 85.29%.
* B — `status` removed from the action question itself (`eval_d1 --drop-actions terminate`, 8 options; the input
  changes): SR **70.63%** (5,964), type 85.21%, grounding 69.37%. Same exact-match count as A.

Published AC-High numbers, from primary sources (`outputs/2026-11-11-published-ac-aw-metrics.md`; none re-run here).
Only InfiGUI-R1-3B uses this exact test file (all 1,543 official test episodes, counted here) and this scorer; the others
use unstated or different test sets, and the same checkpoint has been reported up to ~15 points apart by different groups.

| Model | Size | High Type | High Grounding | High SR | Test set as stated |
|---|---|---|---|---|---|
| OmegaUse | 30B-A3B | 87.6 | n/r | 79.1 | not stated |
| UItron-7B (README, revised) | 7B | 91.4 | 77.2 | 79.0 | 1,000 episodes / 5,477 steps |
| GUI-SAGE-7B | 7B | 87.0 | n/r | 76.8 | not stated |
| UI-Venus-Navi-7B | 7B | 86.5 | n/r | 76.1 | not stated |
| GUI-SAGE-3B | 3B | 86.4 | n/r | 75.4 | not stated |
| GUI-Owl-7B | 7B | n/r | n/r | 72.8 | not stated |
| UI-TARS-7B | 7B | 83.7 | 80.5 | 72.5 | as OS-Atlas |
| InfiGUI-R1-3B | 3B | 82.7 | 74.4 | 71.1 | **same file as here** |
| AgentCPM-GUI-8B | 8B | 77.7 | n/r | 69.2 | not stated (code public) |
| MAI-UI-8B | 8B | n/r | n/r | 69.1 | not stated |
| UI-TARS-2B | 2B | 81.2 | 78.4 | 68.9 | as OS-Atlas |
| OS-Atlas-4B | 4B | 84.7 | 73.8 | 67.5 | 7,708 steps |
| MAI-UI-2B | 2B | n/r | n/r | 67.3 | not stated; metric not named |
| **d1-mobile (v3, this run)** ¹ | **2B** | **85.2** | **69.4** | **70.6** | full official test (8,444 steps) |
| Aguvis-7B | 7B | n/r | n/r | 61.5 | 500 random steps |
| GUI-R1-3B | 3B | 58.0 | 56.2 | 46.6 | not stated |

¹ Evaluated with the end-of-task action (`status`) removed from the action set, because AndroidControl's test steps never
end a task (`eval_d1 --drop-actions terminate`). With `status` allowed, as in InfiGUI's own prompt: SR 67.1, type 81.0,
grounding 67.7. d1-mobile also reads the accessibility element list and an app list; most models above read pixels only.

Interface ceiling (gold answers through the same decoding path, `eval_d1 --oracle`): **88.3%** (strict 297-app list 83.7%),
type 100%. Ceiling losses: click 82.5% (gold element not among the 40 kept, or its centre off the target), type 85.1%
(text not copyable from the input).

Raw decisions with every readout: `actest/eval-v3_s0_ckpt_1824_actest_v3-VA-val.jsonl.gz` (local only, 43 MB, not in git).
Files: `actest/perstep-*.jsonl` (one row per benchmark step: gold, mapped prediction, type/exact match),
`actest/eval-*-VA-val.jsonl` (raw v3 decisions), `actest/*oracle*`. ckpt_900 files (59.89% SR) are kept as an
intermediate record only.

## 4. AndroidWorld (116 tasks, official protocol, pass@1, one run)
Checkpoint `v3_s0_ckpt1824`, harness v5, `task_random_seed` 30, per-task step budgets, run as two disjoint halves of 58
on two emulators of the same host (configs `aw116_v3_s0_a/_b.yaml`). Audits: 18 + 22 preflight health lines, all
naming `v3_s0_ckpt1824`. Episodes and indexes: `androidworld/aw116_v3_s0*/`.

| | |
|---|---|
| **SR (official `official_sr`)** | **14.0%** (16/114; 2 tasks excluded by the official rule: checker unavailable) |
| Successes / 116 | 16 (13.8%, Wilson 95% 8.7–21.2) |
| Information-retrieval (answer-only) tasks | 0/25 (no `answer` action) |
| Failed episodes | 100: 49 declared done but not done; 51 never declared (step budget exhausted) |
| Steps that changed nothing | 663/1,476 (45%) |
| After a no-change step, the next action is identical | 534/609 (88%) |
| Failed episodes with one action repeated >= 5 times in a row | 46/100 (longest: 114 identical clicks) |

Successes: AudioRecorderRecordAudio, ClockStopWatchPausedVerify, ClockStopWatchRunning, ContactsAddContact,
ExpenseDeleteMultiple, ExpenseDeleteSingle, MarkorCreateFolder, OpenAppTaskEval, OsmAndFavorite,
RecipeDeleteDuplicateRecipes, RecipeDeleteSingleRecipe, RecipeDeleteSingleWithRecipeWithNoise, SystemBluetoothTurnOff,
SystemBluetoothTurnOn, SystemBluetoothTurnOnVerify, SystemWifiTurnOff.

Mechanism: in training only 2.0% of rows follow a step that changed nothing (539/27,454), so the model never learned
that `→ no change` means "do something else"; given an identical screen it repeats its decision.
Published 2B references (self-reported): UI-Venus-1.5-2B 55.6, MAI-UI-2B 49.1 (SFT only 45.1).
