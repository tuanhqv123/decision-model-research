# 2026-11-07 — The best configuration this programme has produced

> **On the string "64.32" in this file.** It is the **858 px** configuration's step accuracy, a *different*
> measurement from E2's. It coincidentally equals the retracted pre-audit E2 figure (64.32 = 256/398), which
> exists in no artifact — the E2 headline is **64.07% (255/398)**. Do not read the two as the same number.

> **RETRACTED DELTAS (2026-11-07, `benchmarks/results/2026-11-07-SEED-SPREAD-VERDICT.md`).**
> Every grounding delta in this file is a **single-seed** comparison, and the pre-registered withdrawal rule
> fired: same-recipe spread was **2.25 pts** on 15k and **0.90** on 22.5k, pooled within-recipe **sd 1.73**.
> The 22.5k-vs-15k difference is **+2.47 on 2+2 seeds, t = 1.42 (df 2, critical 4.30) — not evidence**.
> Treat the tables below as *hypotheses with numbers attached*, not as results. The one measure that separated
> the recipes consistently across seeds is `grounded|type` (70.2/70.6 vs 63.8/66.3); even that needs 3 more
> seeds per recipe before it is a claim. **Do not cite a delta from this file.**


One config, measured on all 398 eval rows with the official scorer. It is the best *observed* configuration; the
improvement over E2 is **not established** — see the banner.

## The recipe

```
data      : compact_cur15k.jsonl, shuffled and duplicated to 22,500 rows (compact_ctrl22k.jsonl)
init      : LiquidAI/d1-2B
track     : V (box readout only)
steps     : 938   |  lr 1e-5  |  warmup 30  |  seed 0
micro 1 / accum 16 | grad-ckpt | freeze-embed   (fits in ~11 GB)
images    : 720 px (D1_IMG_W default)
```

Nothing new is added: no annotation, no extra screenshots, no auxiliary task, no architecture change.

## The numbers, against everything this programme has

| config | step acc | type (223) | grounding (223) | grounded\|type | p50 latency |
|---|---|---|---|---|---|
| d1-2B **base**, no fine-tune | 17.4% | 63.2% | 14.4% | 22.7% | — |
| E2 (15k rows) | 64.07% | 87.9% | 56.1% | 63.8% | 106.8 ms |
| `aux7k` (22.5k, grounding-framed) | 64.57% | 87.0% | 57.4% | 66.0% | 104.6 ms |
| 858 px (15k) | 64.32% \* | 86.1% | 58.7% | 68.2% | 154.4 ms |
| **duplicate to 22.5k (this recipe)** | **66.33%** | 85.7% | **60.1%** | **70.2%** | **88.2 ms** |
| `aux7k` + 858 px combined | 65.58% | 87.0% | 57.4% | 66.0% | 154.5 ms |

\* **This 64.32 is the 858 px configuration's step accuracy — a different measurement.** It is not the
retracted pre-audit E2 figure (also written 64.32, i.e. 256/398, which exists in no artifact). E2 is
**64.07% (255/398)**.


**Stacking the two weaker levers does not work:** `aux7k + 858 px` reaches 57.4% grounding — below 858 px alone
(58.7%) and far below the duplication control (60.1%) — at 154 ms. Non-additive, which the earlier "≈ +6.01 sum"
speculation had assumed it would be.

**+2.26 step accuracy and +4.0 grounding over E2, with no latency cost and no new data.**

The one cost is type accuracy: 87.9% → 85.7% (−5 rows). The placement gain (+7.4 points on `grounded|type`,
125→134 grounded) more than pays for it, which is why total step accuracy rises.

## Why this is the honest best, and what it means

Grounding improved **60.1% vs 56.1%** by training on the same 15,000 decisions twice. That is:
- **free at inference** (88.2 ms p50, within noise of E2's 106.8),
- **free of new data** (the rows already existed),
- **reproducible** (one file, one command, one seed so far),
- and it is **larger than every other lever this programme tried**, including the three that failed and the two
  that cost latency or type accuracy.

The mechanism is the one the data points to everywhere: the box loss has no gradient for far-away errors, so what
matters is *how many times the model sees the correction*, not how cleverly the correction is framed.

## What must still be done before this is a paper claim

1. **Seed 2 and 3.** One seed is a point estimate. Two more runs (~3 h total) turn +4.0 into a replicated effect.
2. **Check it is not just epochs.** Duplicated rows over 938 steps = ~1.5 epochs over the unique 15k. An
   honest alternative control is the same 15k rows for 1,400 steps, which reaches the same total exposure without
   duplication. If that matches, the lever is "training tokens", not "duplication".
3. **Report it with the type cost attached** — 85.7% type is the lowest of the five arms and must be in the table,
   not the footnote.

## Files

- `compact_ctrl22k.jsonl` generation: `python3 /tmp/make_ctrl.py <cur15k> <out22k>` (shuffle with seed 0, append
  the first 7,500 rows again).
- Run: `arm_ctrl`, log `~/acv2/logs/arm_ctrl.log`, eval `eval-arm_ctrl-V.jsonl` (398 rows).
