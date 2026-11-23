# Supplementary pre-registration: p(status) threshold for exec1 (2026-11-14)

Written **before seeing exec1's validation result**. The exec1 test result under the original registration (`PRE-REG-2026-11-13-exec.md`) is already available and **not met**:
- AC-Low: 84.05, below the 88.3 bar.
- Wrong done in the middle of a sub-goal: 5.54% (324/5,851), above 5%.

That result stays as the result of the original registration. This document is a **new** configuration, evaluated separately.

## The only change: the harness decision rule

The model, the checkpoint (`ckpt_6000`) and the data all stay the same. The change:
- The executor reports `status` only when `status` is the top choice **and** p(status) ≥ τ.
- Otherwise the harness takes the second-highest choice of the same run. The accompanying value (element, scroll direction, text, app) is also taken from that run.
- Code: `eval_d1.pick_action`, flag `--status-threshold`. Test: `test_pick_action.py`.

## How τ is chosen (fixed in advance, validation only)

- Data: `compact_validation_exec` (690 action rows, 334 "done" rows). Use the probabilities in the validation prediction file, no need to rerun the model.
- τ runs over the grid 0.50; 0.55; …; 0.95; 0.97; 0.99.
- Choose τ for the **smallest total errors**. Total errors = (number of "done" rows where the model does not report `status`) + (number of action rows where the model reports `status`).
- On a tie, take the smaller τ.
- Apply τ **once** to test, with no adjustment afterwards.

## Evaluation on test, the same files and the same scorer as the original registration

1. Execution role on test, using `--instruction-per-row --status-threshold τ`.
2. AC-Low, using `--instruction --status-threshold τ`.

The pass conditions are **unchanged** from section 6 of the original registration:
- "done" recall ≥ 80%, wrong done in the middle of a sub-goal ≤ 5%, AUROC ≥ 0.90;
- AC-Low ≥ 88.3;
- rate of correct steps in the execution role ≥ 73.06%.

AUROC does not change because the threshold does not change the probabilities.

## Additional report (not a condition)

AC-Low with `--exclude-actions terminate`: the model may not choose `terminate`, because the AC test set has no terminating step at all.
- This run started before this document was written, and before validation was seen.
- It measures action skill alone with the "done" decision removed entirely.
- For DMA-2B, this number equals exactly the AC-Low already reported (89.3), because DMA-2B never predicts `terminate` on AC-Low.

## Added after writing (11/14, before seeing the test result of the threshold configuration)

**Scorer fix (`score_exec.py`).** The old version determined the "last step of a sub-goal" by checking whether a "done" row follows that step. But the last sub-goal of each test episode has no "done" row (there is no screen after it). So the last step of those goals was wrongly counted as "in the middle of a sub-goal".
- The fix determines the last step by file order: a step is the last one if the row after it is not an action row of the same sub-goal.
- Test: `test_the_last_action_of_an_episode_without_a_done_row_is_a_last_action_not_inside`.
- The number of "in the middle of a sub-goal" rows on test changes from 5,851 to **4,321**, and "last step" changes from 2,513 to 4,043.
- Both models were re-scored on exactly the same old prediction files:
  - exec1: wrong done in the middle of a sub-goal **152/4,321 = 3.52%**, at the last step 298/4,043. Condition 1 of the original registration is **met** under the corrected scorer.
  - DMA-2B: wrong done in the middle of a sub-goal 3/4,321, at the last step 10/4,043.
- The other numbers do not change.
- So, under the corrected scorer, exec1 only **fails condition 2** (AC-Low 84.05).

**τ chosen on validation by exactly the rule above: τ = 0.50.** Total errors by τ:

| τ | 0.50 | 0.55 | 0.60 | 0.65 | … | 0.99 |
|---|---|---|---|---|---|---|
| Total errors | **39** | 44 | 43 | 43 | … | 271 |

**AC-Low diagnosis** (from the prediction file, model not rerun). There are 720 wrong `terminate` declarations:
- 530 at steps whose instruction is identical to the previous step: 530/1,547 = 34.3%, against 190/6,897 at the other steps.
- Split by correct action: click 270, wait 212, scroll 166, navigate_back 59.

A probability threshold can hardly fix this kind of error. The next training targets it: `PRE-REG-2026-11-14-exec2.md`.

**Result of the "additional report" part:** exec1's AC-Low with `terminate` forbidden is **90.41** (7,634/8,444; correct action type 96.97%).

## Result on test with τ = 0.50 (11/14)

| Metric | exec1, argmax | exec1, τ = 0.50 | Condition |
|---|---|---|---|
| Correct steps, execution role | 77.30% | **77.38%** | ≥ 73.06% — met |
| Wrong done in the middle of a sub-goal | 152/4,321 (3.52%) | **141/4,321 (3.26%)** | ≤ 5% — met |
| Wrong done at the last step | 298/4,043 | 293/4,043 | — |
| "Done" recall | 96.1% | **95.74%** | ≥ 80% — met |
| AUROC p(status) | 0.9906 | 0.9906 | ≥ 0.90 — met |
| **AC-Low** | 84.05 | **84.15** (7,106/8,444) | ≥ 88.3 — **not met** |

**Conclusion: the threshold configuration is also not met, only because of condition 2.** As the diagnosis recorded: most of the wrong done declarations in AC-Low have p(status) ≥ 0.5, so the threshold removes only 9 of them (7,106 against 7,097 correct steps).
