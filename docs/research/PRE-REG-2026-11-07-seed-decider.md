# PRE-REGISTRATION — the seed decider: 22.5k vs 15k at 3 more seeds each

*Frozen before the runs. This is the experiment that decides the only surviving grounding hypothesis.*

## The hypothesis under test

`compact_ctrl22k` (cur15k duplicated to 22,500 rows) vs `compact_cur15k` (15,000 rows), both at 938 steps, 720 px,
track V, same init/lr/warmup.

Current evidence, 2 seeds each:

| recipe | grounding | mean | within-spread |
|---|---|---|---|
| 15k | 56.05, 58.30 | 57.17 | 2.25 |
| 22.5k | 59.19, 60.09 | 59.64 | **0.90** |

Difference **+2.47**, pooled within-recipe sd **1.73**, **t = 1.42, not significant**.

`grounded|type` separates them more cleanly: 63.8 / 66.3 vs 70.2 / 70.6.

## The runs

Seeds **2, 3, 4** for each recipe = **6 runs**, ~1.6 h each, run 2-at-a-time across the two hosts (≈5 h wall).

Everything else identical: `--track V --no-memory --steps 938 --micro 1 --accum 16 --lr 1e-5 --warmup 30
--grad-ckpt --freeze-embed --mem-frac 0.40 (B) / 0.62 (A)`, 720 px, eval on the same 398 rows.

## Pass condition (frozen)

Pooling to **5 seeds per recipe**:

- **PASS:** mean grounding difference ≥ **+2.0** with a **two-sample t ≥ 2.78** (df = 8, p < 0.05 two-sided).
- **PASS-SECONDARY:** `grounded|type` difference ≥ **+3.0** with t ≥ 2.78. (This is the more stable measure and
  the one the mechanism predicts.)
- **KILL:** mean difference < +1.0, or t < 2.0 → the 22.5k recipe is **not** better than 15k; the grounding line
  closes and the paper reports only what survived (loss mechanism, base-vs-fine-tune, candidate collapse, cost
  window).

No threshold moves after seeing the numbers.

## What each outcome means for the paper

| outcome | the sentence the paper can say |
|---|---|
| PASS | "Duplicating the training decisions to 1.5× improves element grounding by X points (5 seeds, p<0.05) at zero inference cost." |
| PASS-SECONDARY only | "Placement improves (grounded-given-type +X, 5 seeds) while overall grounding is unchanged." |
| KILL | "No grounding recipe tested here beats the baseline beyond seed noise; the contribution is the loss-mechanism analysis and the gate's cost window." |

## Why this and not something else

It is the only grounding lever with (a) a mechanism, (b) two agreeing seeds, (c) zero inference cost, and (d) an
ambiguous statistical verdict that one more batch of runs resolves definitively. Every other lever was either
inside noise, cost latency, or failed outright.


## POWER CORRECTION (found before the runs finished — 5 seeds is underpowered)

My first pass pooled the seed variance across *all four* runs, which mixes the between-recipe difference into the
variance and **overstates it**. The correct within-recipe sds are:

| recipe | seeds | sd |
|---|---|---|
| 15k | 56.05, 58.30 | **1.59** |
| 22.5k | 59.19, 60.09 | **0.64** |

Pooled within-recipe sd = **1.21** (not 1.73). With the observed effect of **+2.47**:

| seeds per recipe | SE | t | t_crit (95%, df=2n−2) | verdict |
|---|---|---|---|---|
| 2 (current) | 1.21 | 2.03 | 4.30 | NS |
| 3 | 1.41 | 1.75 | 2.78 | NS |
| 4 | 1.22 | 2.02 | 2.45 | NS |
| **5 (my original plan)** | **1.09** | **2.26** | **2.31** | **NS — a coin flip** |
| **8** | 0.86 | 2.86 | 2.14 | **SIG** |

**So the 5-seed plan I pre-registered would almost certainly have returned "not significant" even if the effect is
real.** That is a design error, caught before the result rather than after.

**Corrected plan:** run to **8 seeds per recipe** (6 more runs beyond the 4 already done, ≈5 h at two hosts), and
keep the same PASS rule: mean difference ≥ +2.0 with t ≥ 2.14 (df=14). If the true effect is smaller than +2.47 the
required n grows further, so the honest alternative is to **declare the effect indeterminate at this n** and close
the line rather than keep adding seeds indefinitely.
