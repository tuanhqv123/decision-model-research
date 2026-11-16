# All figures available (2026-11-12)

Every number comes with its denominator. The AndroidWorld tables are built with `benchmarks/aw/aw_report.py` from saved traces. The traces (without screenshots) are in `benchmarks/results/2026-11-12-armsAB/androidworld_armsAB.tgz` and `benchmarks/results/2026-11-12-maskenv/androidworld_maskenv.tgz`. Scoring standard: official AndroidWorld, 116 tasks, seed 30, 1 run (pass@1); SR = total score / number of scored tasks.

## 1. AndroidWorld: the runs

| Model | Harness / environment | Score | SR | Paired against the run before it |
|---|---|---|---|---|
| v3h | V8 | 17/113 | 15.0% | – |
| v4 | V8 | 15.5/115 | 13.5% | – |
| v4 | V11 (answer fix, policy logging, fresh_wait) | 20.5/115 | 17.8% | against v4/V8: 5 wins, 0 losses |
| mix 0.3·v3 + 0.7·v4 | V11 | 18/114 | 15.8% | against v4/V11: 6 wins, 9 losses |
| v4 | V11 + mask | 20.5/113 | 18.1% | 2 wins, 2 losses |
| **v4** | **V12 + environment fixes (SIM, Chrome)** | **25/115** | **21.7%** | against v4/V11: 5 wins, 1 loss (p = 0.22) |
| v4 | V14 (adds answer format, hidden keyboard) + Markor/OsmAnd fix | running the 39 affected tasks; the other 77 tasks keep the V12 result | – | – |

**Tasks excluded because of environment errors (all fixed):**
- OsmAndMarker: the `map_markers_db` file is empty and owned by root;
- SimpleSmsReplyMostRecent: SIM error;
- SystemCopyToClipboard: Clipper error, happened only once.

## 2. AndroidWorld by official difficulty (`task_metadata.json`: easy 61 / medium 36 / hard 19)

### v4 + V11 (arm A)
| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | avg optimal steps | excluded |
|---|---|---|---|---|---|---|---|---|---|
| easy | 61/61 | 19.0 | 19 | 31.1 | 8.7 | 5.4 | 10.1 | 5.0 | 0 |
| medium | 35/36 | 1.0 | 1 | 2.9 | 12.3 | 7.0 | 12.4 | 9.1 | 1 |
| hard | 19/19 | 0.5 | 0 | 2.6 | 18.3 | - | 18.3 | 18.1 | 0 |
| all | 115/116 | 20.5 | 20 | 17.8 | 11.3 | 5.5 | 12.6 | 8.4 | 1 |
### merge 0.3 + V11 (arm B)
| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | avg optimal steps | excluded |
|---|---|---|---|---|---|---|---|---|---|
| easy | 61/61 | 17.0 | 17 | 27.9 | 8.8 | 5.8 | 10.0 | 5.0 | 0 |
| medium | 35/36 | 1.0 | 1 | 2.9 | 13.4 | 8.0 | 13.6 | 9.1 | 1 |
| hard | 18/19 | 0.0 | 0 | 0.0 | 22.1 | - | 22.1 | 18.5 | 1 |
| all | 114/116 | 18.0 | 18 | 15.8 | 12.3 | 5.9 | 13.5 | 8.4 | 2 |
### v4 + V11 + mask
| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | avg optimal steps | excluded |
|---|---|---|---|---|---|---|---|---|---|
| easy | 60/61 | 19.0 | 19 | 31.7 | 8.4 | 5.5 | 9.8 | 5.1 | 1 |
| medium | 35/36 | 0.0 | 0 | 0.0 | 12.1 | - | 12.1 | 9.1 | 1 |
| hard | 18/19 | 1.5 | 1 | 8.3 | 12.4 | 7.0 | 12.8 | 18.5 | 1 |
| all | 113/116 | 20.5 | 20 | 18.1 | 10.2 | 5.5 | 11.2 | 8.5 | 3 |
### v4 + V12 (env fixes)
| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | avg optimal steps | excluded |
|---|---|---|---|---|---|---|---|---|---|
| easy | 61/61 | 21.0 | 21 | 34.4 | 8.5 | 5.5 | 10.1 | 5.0 | 0 |
| medium | 36/36 | 3.0 | 3 | 8.3 | 13.0 | 6.7 | 13.6 | 9.0 | 0 |
| hard | 18/19 | 1.0 | 0 | 5.6 | 19.3 | - | 19.3 | 18.5 | 1 |
| all | 115/116 | 25.0 | 24 | 21.7 | 11.6 | 5.7 | 13.2 | 8.4 | 1 |

**By MobileRL-style complexity level**, ceil(complexity), with v4 + V12:

| Level | Successes | SR | Mean steps on success |
|---|---|---|---|
| C1 | 19/45 | 42.2% | 5.3 |
| C2 | 5.5/40 | 13.8% | 7.2 |
| C3 | 0/9 | 0% | – |
| C4 | 0.5/21 | 2.4% | – |

## 3. AndroidControl (v4 ckpt_2482)

**AC-High** (InfiGUI scorer, test set 1,543 tasks / 8,444 steps):
- correct per step: 5,213/8,444 = 61.7%;
- tasks with every step correct: 344/1,543 = 22.3%;
- number of times the model asked to stop wrongly (the test set has no stop step): 1,171/8,444.

**AC-Low:**
- correct per step: 7,544/8,444 = 89.3%;
- tasks with every step correct: 932/1,543 = 60.4%.

**Validation** (827 rows / 137 tasks):
- correct per step: 520/827 = 62.9%;
- tasks with every step correct: 14/137;
- wrong done declarations: 64/690;
- wrong back actions: 16/798.

**Replacing the image with a gray image** (validation, same v4):

| | Real image | Gray image |
|---|---|---|
| Correct per step | 519/827 | 337/827 |
| Correct action type chosen | 80.0% | 50.8% |

## 4. Trying the pattern with the current model (v4, not trained)

| | Validation | AC sample, 2,012 rows |
|---|---|---|
| **M1: choose by choice** (argmax) | 286/430 | 821/1,224 |
| M1: re-choose with one noul per candidate | 261/430 | 772/1,224 |
| M1: combine the two ways | 285/430 | 823/1,224 |
| M1: correct answer is in the top-5 | 388/430 | 1,084/1,224 |
| **M3: AUROC** of the noul "did the last step get closer to the goal" | 0.68 (10 negative samples) | **0.76** (1,550 positive / 26 negative) |

## 5. Retraining (contract v6): not finished

| At ckpt_250, on validation | Run 1 | Run 2 | Run 3 | v4 |
|---|---|---|---|---|
| Correct per step | 489 | 485 | 492 | 520 |
| Wrong done declarations (out of 690) | 67 | 108 | 106 | 64 |
| Wrong back actions (out of 798) | 44 | 23 | 35 | 16 |

- **Runs 1 and 2** were stopped by the early-warning rule. Afterwards the data of the v3 run itself refuted this rule: a mid-run metric does not predict the final metric (AGENTS.md M17).
- **Run 3** stopped at step 630 to yield the shared GPU to a customer. The run will be rerun on GPU_HOST_A.
