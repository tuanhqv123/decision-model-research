# Pre-registration: exec2, up-weighting the "continue the same instruction" steps (2026-11-14)

Written **before training**. Original: `PRE-REG-2026-11-13-exec.md` (exec1). Anything not said here stays as in the original.

## 1. Why this training run exists

exec1 (ckpt_6000) on the AndroidControl test set, scored with the corrected scorer (11/14):
- Condition 1 **met**: "done" recall 96.1% (2,415/2,513), wrong done in the middle of a sub-goal 3.52% (152/4,321), AUROC 0.9906.
- Condition 3 **met**: correct steps 77.30% (8,364 rows).
- Condition 2 **not met**: AC-Low 84.05 (needs ≥ 88.3).

The cause of condition 2, measured from the prediction file:
- 720 AC-Low steps were wrongly declared `terminate`.
- 530 of them fall on a step whose instruction is **identical to the previous step**, that is, an AC instruction that lasts several steps. The rate at these steps is 34.3% (530/1,547), against 2.8% (190/6,897) at the other steps.
- If `terminate` is forbidden (for reporting only, not a condition), exec1's AC-Low is **90.41** (7,634/8,444). With the same rule, DMA-2B is 89.3. So almost the whole shortfall of condition 2 lies in the wrong done declarations.

The exec1 data already has 8,746 rows of this kind: `span` rows whose previous step, in the same episode, has `local_done = False`. According to `rebuild_ac_v2.py`, `local_done = False` means the instruction of the next step is the same as this step's instruction. This number of rows is small against the 23,933 `status` rows.

**Note on meaning.** For an instruction like "Scroll down" repeated over 3 steps, after the first scroll the instruction is literally done. The AC label says "not done" because the annotator repeats the instruction for the next step. exec2 teaches the AC convention. The possible price is that "done" recall falls. Condition 1 still checks this, with a threshold of ≥ 80%.

## 2. The only change: data

- Keep `compact_exec_v1.jsonl` (98,647 rows).
- Add `compact_execcont.jsonl` with **17,492 rows**: each instruction-continuation row is copied 2 more times (`aug = "span_cont"`), so in total each such row appears 3 times.
  - Script: `benchmarks/train/upweight_exec.py --k 3 --extra-only`, with tests in `test_upweight_exec.py`.
- Train with `--train-set exec_v1,execcont`. Total **116,139 rows**.
- The number of `status` rows does not change, so the `status` share falls from 24.3% to 20.6%.

The run card is generated from the files before the run: `docs/research/RUNCARD-exec2_s0.md`.

## 3. Training recipe (fixed)

Same as exec1:
- initialise from `LiquidAI/d1-2B` (`cfe5760`);
- contract v4, track VA, 720 px images;
- lr 1e-5, warmup 30, cosine;
- embedding frozen, grad-ckpt, micro 1, accum 16, seed 0, `--no-memory`.

Different from exec1:
- **1 epoch = 7,259 steps** (116,139 / 16, rounded up).
- `--mem-frac 0.72` instead of 0.70. exec1 hit OOM at about step 6,100 by exceeding its own cap. This level still leaves ≥ 3 GB of GPU free, even with the 2.3 GB service that is running.
- Save a checkpoint every 500 steps, copied to the Mac with md5.

Estimated time: about 16 hours.

**Use only the final checkpoint.** If training stops before the last step:
- If it is the process's own OOM and the lr at the last remaining checkpoint is ≤ 1e-7, that checkpoint is the final checkpoint. This is the situation of exec1, and the rule is recorded here in advance.
- In every other case, including yielding the GPU (`YIELDED`): rerun later. Do not evaluate intermediate checkpoints.

## 4. Evaluation and conditions (fixed, same as exec1)

The same three test files, the same flags, the same scorer:
- `score_exec.py`, version corrected on 11/14;
- `score_actest.py` (InfiGUI).

The main configuration is **plain argmax** (no threshold), the same as exec1. The model replaces DMA-2B **only if it meets all three**:
1. Execution role on test: "done" recall ≥ 80%, wrong done in the middle of a sub-goal ≤ 5%, AUROC p(status) ≥ 0.90.
2. **AC-Low SR ≥ 88.3.**
3. Correct steps in the execution role ≥ 73.06%.

**Reported only, not used as conditions:**
- AC-High.
- AC-Low with `terminate` forbidden.
- Validation.
- The τ threshold configuration. τ is refit on exec2's validation with exactly the rule of `PRE-REG-2026-11-14-exec-threshold.md`.

The comparison between exec1 and exec2 is **one seed on each side**. A difference under 2 points is recorded as indistinguishable from seed-to-seed noise.

## 5. Predictions (to be checked later)

1. The number of wrong `terminate` declarations at the instruction-continuation steps of AC-Low falls by at least half: from 530 to ≤ 265.
2. AC-Low rises to about 87–90. To reach 88.3, about 360 of the 720 wrong done declarations must be removed without losing correct steps elsewhere.
3. "Done" recall falls but stays ≥ 80%. Correct steps in the execution role stay at about 77%.
4. The 190 wrong done declarations that are **not** at instruction-continuation steps will change little, because this training does not target them.

If prediction 1 is wrong, that is, the number of wrong declarations does not fall by half, then up-weighting is not the fix. In that case no further training will be done in this direction.

## Operations log (11/14)

- **Run 1 was stopped at about step 2,020/7,259**, at about 06:41 UTC. `train.out` records `TRAIN exit 143`, meaning the process received SIGTERM from outside.
- **This is not OOM, and the GPU-yielding script was not the cause:** there is no `YIELDED` file. Right after that, another user was working on the machine (a tmux session named `train` had just been attached).
- Under section 3, this case must be **rerun**. Under the GPU-yielding rule, do not restart on its own; rerun only when the GPU is given back.
- Checkpoints 500, 1,000, 1,500 and 2,000 were copied to the Mac, md5 matched. They were **not evaluated**, because they are intermediate checkpoints.

## Rerun and final checkpoint (11/14–15, recorded before seeing the evaluation result)

- The rerun started at 08:42 UTC on 11/14, when the GPU had been free continuously for 30 minutes.
- Training stopped at step 7,190/7,259 because of the process's own `torch.OutOfMemoryError` (`TRAIN exit 1`). No `YIELDED`.
- The last remaining checkpoint is `ckpt_7000`, lr 3.14e-8 ≤ 1e-7. Under section 3, **`ckpt_7000` is the final checkpoint**. All checkpoints 500–7,000 were copied to the Mac, md5 matched.

**Note on the denominator.** The script `exec_report.py` counts an "instruction continuation" step as a step with `step_instructions[s] == step_instructions[s-1]` (exact match). It gives back exactly exec1's numerator (530) but the denominator is **1,544/6,900**, not 1,547/6,897 as recorded in section 1. The threshold of prediction 1 (≤ 265) depends only on the numerator, so it does not change. The table below uses 1,544/6,900 for both models.

*Resolution (11/15, after seeing the result):* computed with the instruction-splitting rule of `build_exec_data.units` (compare the instruction with the trailing period removed, break at `local_done`), the result is exactly **1,547/6,897**, as section 1 recorded. Under this counting, exec1 has 530 and exec2 has **291** wrong done declarations at instruction-continuation steps, so the numerator is unchanged. The conclusion of prediction 1 stands. Deep analysis: `2026-11-15-EXEC2-DEEP-ANALYSIS.md`.

## Result (11/15), `ckpt_7000`, plain argmax

Files: `benchmarks/results/2026-11-15-exec2/`.

| Metric | exec1 | exec2 | Condition |
|---|---|---|---|
| "Done" recall (2,513 rows) | 96.1% | **93.67%** | ≥ 80% — met |
| Wrong done in the middle of a sub-goal | 152/4,321 (3.52%) | **68/4,321 (1.57%)** | ≤ 5% — met |
| AUROC p(status) | 0.9906 | **0.9913** | ≥ 0.90 — met |
| Correct steps, execution role (8,364) | 77.30% | **78.89%** | ≥ 73.06% — met |
| **AC-Low** (8,444) | 84.05 | **86.78** (7,328/8,444) | ≥ 88.3 — **not met** |
| Wrong done at the last step of a sub-goal | 298/4,043 | 191/4,043 | — |

**Conclusion: exec2 is not met, only because of condition 2** (short by 1.52 points, about 128 steps).

**Reported only:**
- AC-High: 74.21 (exec1 73.50). Difference < 2 points, indistinguishable from seed noise.
- AC-Low with `terminate` forbidden: 90.50 (exec1 90.41).
- Validation, execution role: correct steps 83.62% (690), recall 91.92% (334), in the middle of a sub-goal 1/356, AUROC 0.9907.
- τ refit on exec2's validation: **τ = 0.50** (48 errors; 0.55 → 50). The test was not rerun with τ, because there is an upper bound from the prediction file: of the 449 wrong done declarations on AC-Low only 25 have p(terminate) < 0.5, so AC-Low is at most (7,328 + 25)/8,444 = **87.08** < 88.3. The τ configuration cannot meet condition 2.

**Checking the predictions:**
1. **Wrong.** Wrong `terminate` at instruction-continuation steps fell from 530 to **291/1,544**, a fall of 45%, which does not reach the ≤ 265 bar. Under the recorded rule, **no further training in the up-weighting direction**.
2. Wrong, narrowly: AC-Low 86.78, below the predicted 87–90.
3. Right: recall fell 2.4 points but stays ≥ 80%; correct steps 78.89%.
4. Partly: 190 → 158 at the other steps (−17%), a larger fall than "changes little".

The total of wrong `terminate` declarations on AC-Low fell 720 → 449. The remaining shortfall of condition 2 still lies almost entirely in wrong done declarations: if `terminate` is forbidden, AC-Low is 90.50.
