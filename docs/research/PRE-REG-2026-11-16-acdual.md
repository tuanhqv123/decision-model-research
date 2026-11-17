# Pre-registration: AndroidControl-only training in both test formats (run `acdual`, 2026-11-16)

Written before any data of this run is trained on or evaluated. Results go in a separate file; this file is not edited
after the run starts, except to append a dated "launched" line.

## Question

Does training data in exactly the two AndroidControl test formats raise the step accuracy of the final recipe on
AndroidControl, compared with the released DMA-2B (run exec2, step 7,000)?

## What changes, and what does not

Only the data changes.

- **Data:** `build_exec_data.py --dual` on the full AndroidControl training split (`compact_ac_full_v2.jsonl`: 13,594
  episodes, 88,308 rows, test episodes excluded). Every action step appears twice:
  - `dual_high`: goal only, as in AndroidControl-High;
  - `dual_low`: goal plus that step's own raw instruction, as in AndroidControl-Low.
  - No `status` row: AndroidControl has no termination action.
  - A step without an instruction appears only in the goal form.
- **Unchanged from exec2:**
  - initialisation from `LiquidAI/d1-2B@cfe5760`;
  - contract v4, track VA, image width 720;
  - lr 1e-5, warmup 30, cosine decay, micro-batch 1, accumulation 16;
  - frozen tied embedding, gradient checkpointing, no memory, seed 0;
  - memory cap 0.72.
- **Steps:** one epoch, i.e. rows / 16, rounded down. The number is printed by the build and copied into the launched
  line.
- **Checkpoint used:** the last one saved. Intermediate checkpoints are only watched on validation and are never used
  to choose.

## Evaluation (fixed now)

- AndroidControl test split, 8,444 steps, InfiGUI scorer (`score_actest.py`).
- AndroidControl's own action set: `--exclude-actions terminate`.
- AC-Low uses `--instruction`; AC-High does not.
- Same command as the released model (`ev_seed.sh`, applied to this run's checkpoint).
- Reference (released DMA-2B, `benchmarks/results/2026-11-16-ac-noterm/paper_numbers.json`): AC-Low 90.50%
  (7,642/8,444), AC-High 74.21% (6,266/8,444).
- Comparison: paired McNemar test on the same 8,444 steps, two-sided.

## Predictions

- AC-High rises by 1 to 3 points over 74.21, because the goal-only format has about 3.5 times as many rows as in
  exec2.
- AC-Low changes by 0 to +1 point over 90.50. Its errors are mostly wrong elements, which the format change does not
  target.

## Decision rules

- **Pass:** AC-High ≥ 75.21 (+1.0 point) with McNemar p < 0.05, and AC-Low ≥ 90.00 (no loss beyond 0.5 point).
- **Fail:** anything else.
  - If AC-High does not rise, the conclusion is that data in the test format was not the bottleneck. The released
    model stays.
  - If AC-High rises but AC-Low drops by more than 0.5 point, report both; the released model stays.
- **Seed spread:** seeds 1–2 of exec2 (running) give the seed-to-seed spread of the old recipe. A gain smaller than
  2 standard deviations of that spread is reported as "not distinguishable from seed noise", even if p < 0.05.
- **Known cost:** this model never declares a task done, so it cannot replace exec2 in the LLM + decision-model
  harness. If it passes, it is reported as the AndroidControl model only.

## Resources

- One 24 GB GPU, about 18 GB with the 0.72 cap.
- It starts only after the exec2 seed runs have finished, and only if 21 GB of GPU memory is free, checked by the
  launcher.
- Expected time: about 10 s per step on the shared RTX 4090.
- Disk: before starting, the seed runs' intermediate checkpoints are deleted, keeping `ckpt_7000` and the final one.
  This run keeps one checkpoint every 2,000 steps plus the final one.

## Appended 2026-11-17, before training (the predictions and rules above are unchanged)

- **Data built.** `compact_dual_v1.jsonl`: 149,292 rows; md5 `8a834e82e4da9a812e6e347f45e31f36`.
  - 74,714 rows `dual_high` and 74,578 rows `dual_low`; 136 steps have no instruction.
  - 0 `status` rows.
  - Steps = 149,292 // 16 = 9,330.
- **Correction of a factual line in "Predictions".** The goal-only data grows by **2.95×** over exec2 (74,714 rows
  against 25,340), not "about 3.5 times". The 3.5 figure came from an estimate made before the data was built.
  - The predicted AC-High gain (+1 to +3 points) and the decision rules are kept exactly as registered.
  - The smaller ratio makes a gain at the low end of that range, or none, more likely.
- **Launched 2026-11-17 02:00 UTC** (`run_queue.sh` on the training host): 149,292 rows, 9,330 steps.
  - **Order changed:** acdual runs first, then the exec2 seeds 1 and 2.
  - **Memory cap 0.77 for all three runs.** The cap is a resource setting; it does not change the training
    computation. Both seed runs launched on 2026-11-16 with the 0.72 cap stopped at their own cap (seed 1 at step
    ~180, seed 2 at step ~1,160), and so did exec2 seed 0 at step ~7,190.
  - An earlier start of this run at 01:58 was stopped by mistake after a few minutes and deleted. Its checkpoints were
    never evaluated.
