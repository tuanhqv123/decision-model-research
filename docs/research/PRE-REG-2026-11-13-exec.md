# Pre-registration: execution model for the LLM + decision model harness (exec1, 2026-11-13)

Written **before training**. Reasons and evidence from the other groups: `docs/research/2026-11-13-LLM-PLUS-DECISION-HARNESS.md`.

## 1. What this model is for

In the harness, the two sides split the work as follows:
- **LLM:** plans, writes the text to be typed, answers questions, and decides when the **whole task** is done.
- **Decision model:** receives a **sub-goal** (one or a few steps) and carries it out by itself. When the sub-goal is done, the model reports `status`.

The current DMA-2B cannot do this role yet. Measured on validation, when given a sub-goal:
- The model does 76.2% of the steps correctly (526/690).
- But it almost never reports done: only 12/334 times (3.6%).
- Reason: at training time, the model's `status` meant only "the whole task is done".

## 2. Data

Script: `benchmarks/train/build_exec_data.py`, with tests in `test_build_exec_data.py`. Every number below is printed by the script.

**Source:** the whole AndroidControl training set. The script `build_ac_full_v2.py` rebuilds it from the raw records, with this result:
- 13,594 episodes, 88,308 rows. 9 faulty episodes dropped, 0 rows from the test set.
- Cross-check against the 4,197 episodes built on 11/11: 27,454/27,454 rows match, 0 differences.

**Not included:**
- AndroidWorld data;
- `answer` rows;
- "whole task done" rows;
- no-change augmentation (chain, avoid).

**How episodes are split.** Uses the seed `seed:ep`, so each episode always falls in the same group, on any machine and in any run:

| Group | Episodes | Rows | Content |
|---|---|---|---|
| `goal` (AC-High) | 4,624 | 25,340 | General goal only. Every action step is kept. **No `status` rows.** |
| `span` | 8,970 | 49,374 | Each row has an added `Step instruction: <sub-goal>`. |
| `span_done` | (same 8,970) | 23,933 | The screen right after the last step of the sub-goal, label `status`. |
| **Total** | 13,594 | **98,647** | `status` is 24.3% of rows. |

**How sub-goals are made:**
- Merge consecutive per-step instructions of AC. One instruction can last several steps, until `local_done`.
- Each sub-goal has 1–4 instructions, distributed as 1: 12,564, 2: 7,049, 3: 3,303, 4: 1,017.
- The instructions are joined with ", then".
- Rows in the middle of a sub-goal teach the model that it is **not yet done**.

**Input/output contract:** keep `v4`: the same question, the same state display, the same 720 px image. Each row decides for itself whether to show the instruction, through the field `show_instruction` that `ac_common.instruction_included` reads.

**No BLOCKED label:** AndroidControl has no failed sub-goal. The harness detects being blocked through three signals:
- low probability;
- the screen does not change;
- exceeding the allowed number of steps for one sub-goal.

## 3. Training recipe (fixed)

The settings are the same as v3/v4, only the data differs:
- initialise from `LiquidAI/d1-2B` (revision `cfe5760`);
- 1 epoch: 98,647 rows = **6,166 steps × 16**;
- lr 1e-5, warmup 30, cosine;
- tied embedding frozen; the vision tower is still trained;
- grad-ckpt, micro 1, seed 0.

Estimated time: about **13.4 hours** on an RTX 4090 (2.05 samples/second, measured in the previous training on the same machine). Every 1,000 steps one checkpoint is overwritten, so a run can resume if interrupted; the training machine has only about 9 GB of free disk. **Use only the final checkpoint.** No checkpoint selection and no early stopping on intermediate checkpoints (lesson M17).

**Deviation from the registration (recorded on 11/14, before any evaluation result was seen):**
- The actual training saved a checkpoint every 500 steps, each kept separately, and copied them to the Mac with md5. Reason: the operator asked to keep the checkpoints.
- Training stopped at about step 6,100/6,166 because of GPU out-of-memory (`torch.OutOfMemoryError`). One long sample pushed the training process itself over its own 0.70 × 24 GB cap. No other process intruded on the GPU (`YIELDED` was not written).
- The last checkpoint that exists is `ckpt_6000`. Learning rate per `train.log`: 1.79e-8 at step 6,000 and 2.83e-9 at step 6,100, against a peak of 1e-5.
- The 166 missing steps ran at lr below 2e-8. Each Adam step changes a parameter by about lr, so each parameter changes by less than 3e-6 in total. This level is considered negligible.
- **So `ckpt_6000` is used as the final checkpoint.** This is not checkpoint selection: it is the last checkpoint available, and this decision was made before any evaluation result.

## 4. Evaluation (fixed)

Every evaluation uses the final checkpoint, on the **AndroidControl test set** (1,543 episodes):

1. **Execution role**, using `compact_actest_exec.jsonl` with `--instruction-per-row` and scored with `score_exec.py`. This file has 1,530 `span` episodes; 13 episodes lacking an instruction are excluded. It has 8,364 action rows and 2,513 "done" rows; the last sub-goal of each episode has no screen after it, so it has no "done" row. Metrics:
   - rate of correct steps;
   - wrong done (when in the middle of a sub-goal, and when at its last step);
   - recall of "done";
   - AUROC of p(status).
2. **AC-Low:** scored with the InfiGUI scorer, 8,444 steps, with `--instruction`.
3. **AC-High:** the same scorer, 8,444 steps, without instruction.

In addition, run validation (`compact_validation_exec`, 137 episodes) as a side check. This result is not used to decide anything.

## 5. Reference: DMA-2B (v4 ckpt_2482) on exactly the files above

| Metric | DMA-2B |
|---|---|
| Validation, execution role: correct steps | 76.23% (690 rows) |
| Validation, execution role: "done" recall | 3.59% (12/334) |
| Validation, execution role: AUROC p(status) | 0.736 |
| **Test, execution role: correct steps** | **74.06%** (8,364 rows) |
| Test, execution role: wrong done in the middle of a sub-goal / at the last step | 12/5,851 / 1/2,513 |
| **Test, execution role: "done" recall** | **0.64%** (16/2,513); precision 55.2% (16/29) |
| Test, execution role: AUROC p(status) | 0.679 |
| AC-Low SR / AC-High SR | 89.3 / 61.7 |

**Correction 11/14 (scorer):** the row "wrong done in the middle of a sub-goal / at the last step" above uses the old scorer. The old scorer counted the last step of the last sub-goal of each episode as "in the middle of a sub-goal". According to the corrected `score_exec.py`, DMA-2B is **3/4,321 / 10/4,043**. Details in `PRE-REG-2026-11-14-exec-threshold.md`. The thresholds in section 6 do not change.

## 6. Conditions for using the model (one seed)

The new model replaces DMA-2B in the harness **only if it meets all three**:
1. **Execution role on test:** "done" recall ≥ 80%, wrong done in the middle of a sub-goal ≤ 5%, and AUROC p(status) ≥ 0.90.
2. **AC-Low SR ≥ 88.3**, that is DMA-2B − 1.0.
3. **Rate of correct steps in the execution role on test ≥ 73.06%**, that is the DMA-2B level (74.06%) − 1.0.

How to read the results:
- AC-High is only reported, not used as a condition. The new model does not learn "whole task done", so the strict AC-High number (with no terminating step) is expected to rise. But one seed is not enough to assert it.
- Any difference under 2 points is recorded as "indistinguishable from seed-to-seed noise" (the programme's seed spread is about sd 2).

## 7. Predictions (to be checked later)

1. "Done" recall rises strongly from 3.6%, since this is the only newly taught skill.
2. Wrong done on strict AC-High falls to near 0, since there are no "whole task done" rows any more. On 11/12, DMA-2B declared done wrongly 1,171 times over 8,444 steps.
3. Risk: the model declares done right at the last step of the sub-goal before doing that step. The metric "wrong done at the last step" will detect this error.

## 8. Order after training (not yet run)

Only if section 6 is met, continue:
1. Plug the model into the AndroidWorld harness: the LLM produces the sub-goal, the decision model executes.
2. Run the comparison pre-registered in section 5 of the harness document: the same LLM, three configurations.
