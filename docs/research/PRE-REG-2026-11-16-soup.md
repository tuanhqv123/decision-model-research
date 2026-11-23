# Pre-registration: weight mixing (model soup) for AC (11/16)

Written before seeing any result of the mixes.

**How AC is scored.** Use the original AndroidControl action set, which has no "terminate" action:
- The test file (`infigui_test.json`) has only click, swipe, type, open, wait, system_button, long_press.
- `status` is an action added by `preprocess_ac.py` at preprocessing time.

So every model is scored with `--exclude-actions terminate`, both AC-Low (`--instruction`) and AC-High.

This is a different way of scoring from condition 2 in `PRE-REG-2026-11-14-exec2.md` (argmax, "terminate" allowed). The result "exec2 not met" under that registration stays.

**The mixes.** Uniform weight average (`avg_weights.py`). All three models are initialised from `LiquidAI/d1-2B` `cfe5760`:
- S1 = average of exec1 `ckpt_6000` and exec2 `ckpt_7000`;
- S2 = average of exec2 `ckpt_7000` and DMA-2B `v4 ckpt_2482`.

**How to choose.** Use only AC-Low-style validation: `validation_v4` with `--instruction` and "terminate" forbidden, computed on 690 action rows.
- Choose the model with the highest number of correct steps among exec2, S1 and S2.
- On a tie, keep exec2.
- A difference under 1% (under 7 rows) also keeps exec2, because that is the noise level.

**On test.** Only the chosen model runs AC-Low and AC-High, each exactly once. Report it together with exec2 and DMA-2B, all scored the same way.

## Result (11/16)

**Selection on validation.** Number of correct steps on 690 action rows:

| Model | Correct steps |
|---|---|
| exec2 | 601 |
| S1 | 606 (+5) |
| S2 | 601 |
| DMA-2B | 587 |

S1 beats exec2 by under 7 rows, so under the recorded rule **keep exec2**. No test was run for any mix.

**Test.** All scored with the original AndroidControl action set, with no "terminate", on 8,444 steps, InfiGUI scorer:

| Model | AC-Low | AC-High |
|---|---|---|
| **exec2 `ckpt_7000`** | **90.50** (7,642) | **74.21** (6,266) |
| DMA-2B `v4 ckpt_2482` | 89.34 (7,544) | 69.73 (5,888) |

Paired per step (McNemar):
- AC-Low: exec2 right and DMA-2B wrong on 227 steps, the opposite on 129 steps, p = 2.3e-7.
- AC-High: 734 against 356, p = 8.6e-31.

**Notes:**
- exec2 never chooses "terminate" on AC-High, so its AC-High is the same whether or not "terminate" is allowed.
- If DMA-2B is allowed "terminate", its AC-High is only 61.7.
- Under the scoring of `PRE-REG-2026-11-14-exec2.md` ("terminate" allowed), exec2 is still recorded as not meeting condition 2.

Files: `benchmarks/results/2026-11-16-ac-noterm/`.
