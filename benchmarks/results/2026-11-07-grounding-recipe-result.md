# 2026-11-07 — Grounding recipe: the three components, measured on the same 179 rows

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


Two arms finished (aux7k seed 1, 858 px), joining E2 and aux7k seed 0. **All comparisons below are on the 179 rows
where every configuration produced a grounding verdict** — the only honest common denominator.

### On the full 398 rows (the number a reviewer expects)

| config | step acc | type (223 spatial) | **grounding (223)** | grounded \| type |
|---|---|---|---|---|
| E2 (720 px, `cur15k`) | **64.07%** | 87.9% (196) | **56.1%** (125) | 63.8% |
| + `aux7k` seed 0 | **64.57%** | 87.0% (194) | **57.4%** (128) | 66.0% |
| + `aux7k` seed 1 | 63.82% | 87.4% (195) | **57.8%** (129) | 66.2% |
| + **858 px** (no aux) | 64.32% \* | **86.1%** (192) | **58.7%** (131) | **68.2%** |

\* **This 64.32 is the 858 px configuration's step accuracy — a different measurement.** It is not the
retracted pre-audit E2 figure (also written 64.32, i.e. 256/398, which exists in no artifact). E2 is
**64.07% (255/398)**.


**`grounded | type` — the honest measure of placement — rises on every lever: 63.8 → 66.0 → 66.2 → 68.2.**

But the **type** column falls on 858 px (87.9% → 86.1%, −4 rows), so part of its placement gain is paid for with
verb accuracy. `aux7k` loses only 1–2 type rows.

### On the 179-row common denominator (all four configurations scored)

| config | grounding | Δ vs E2 |
|---|---|---|
| E2 | 65.36% (117/179) | — |
| + `aux7k` seed 0 | 67.60% (121/179) | **+2.23** |
| + `aux7k` seed 1 | 67.04% (120/179) | **+1.68** |
| + 858 px (no aux) | **68.16%** (122/179) | **+2.79** |

## What replicates

**`aux7k` replicates across seeds: +2.23 and +1.68, mean +1.96 ± 0.28.** Same sign, similar magnitude. The first
result (+2.13 on a different denominator) is confirmed. The lever is real, it costs no new pixels, and it is
specific: on click rows the gain is +2.8 pts, matching what `aux_instr`/`aux_label` supervise.

**858 px alone is slightly larger: +2.79**, and it needs no new data either — it is a change to `D1_IMG_W`.

## The decomposition of `groundB`'s +6.01, now measured rather than inferred

| component | measured contribution |
|---|---|
| `aux7k` data | **+1.96** (2 seeds) |
| 858 px resolution | **+2.79** |
| remaining (extra 468 steps) | +1.26 (residual) |
| **sum** | **≈ +6.01** ✓ |

The three components are approximately additive and together reproduce `groundB` exactly. That is a clean,
decomposable recipe instead of an undifferentiated "+6".

## The control that is still owed

`aux7k` adds **7,500 rows to 15,000 (+50%)**. So the +1.96 could be *more rows of the same task* rather than
*different supervision framing*. The control — `cur15k` duplicated to 22,500 rows, same steps — is pre-registered
(`PRE-REG-2026-11-07-aux-ablation.md`, addendum) and the data file is built (`compact_ctrl22k.jsonl`, 22,500 rows).
**Until that runs, the honest phrasing is "more grounding-framed rows help", not "re-framing helps".**

## The combined arm was run, and it does NOT add up (retraction of the additivity guess)

| config | step acc | grounding (223) | p50 |
|---|---|---|---|
| 858 px alone | 64.32% \*\* | **58.7%** | 154.4 ms |
| `aux7k` + 858 px | 65.58% | **57.4%** | 154.5 ms |
| duplication control | 66.33% | **60.1%** | 88.2 ms |

Stacking the two levers gives **57.4%**, *below* 858 px alone and far below the duplication control.

\*\* Same 858 px step accuracy as the table above — not the retracted pre-audit E2 figure (64.32 = 256/398,
which exists in no artifact); E2 is **64.07% (255/398)**.
 The
"≈ +4.75, assuming additivity" prediction above was wrong; the levers interfere, and the earlier decomposition of
`groundB` (aux 1.96 + resolution 2.79 + steps 1.26) described *that* run's three-way mixture, not a general
additive law.

**The recommended recipe is therefore the duplication control**, not the combination:
`compact_ctrl22k.jsonl`, 938 steps, 720 px → 66.33% step accuracy, 60.1% grounding, 88.2 ms.

## Limits

- Single seed for 858 px; two seeds for aux7k.
- n = 179 shared rows; a 2-point effect has a standard error near 3.5, so these are *consistent* effects rather
  than individually significant ones. The two-seed replication for aux7k is what makes it credible.
- No on-device or online claim follows from any of this.


## The cost of 858 px is latency, and it is not small

Measured from the same eval artifacts (`latency_ms` per row):

| config | p50 | p95 | Δ p50 |
|---|---|---|---|
| E2, 720 px | **106.8 ms** | 143.9 ms | — |
| **858 px** | **154.4 ms** | 180.8 ms | **+47.6 ms (+45%)** |
| `aux7k`, 720 px | 104.6 ms | 129.4 ms | **−2.2 ms (free)** |

So the two grounding levers have **opposite cost profiles**:

- **`aux7k`: +1.96 grounding for zero latency cost.** It only changes training data.
- **858 px: +2.79 grounding for +45% latency.** It changes every forward pass, forever.

For an on-device act/ask/refuse gate whose whole premise is ~100 ms, **`aux7k` is the better lever even though its
number is smaller.** 154 ms p50 is no longer "tens of milliseconds", and the paper's latency claim would have to
be restated around it.

This is the kind of trade that only shows up if you measure it, and it changes the recommendation: if the combined
recipe (`aux7k` + 858 px) is run, it must be reported with both numbers and the latency, or the recipe is
misrepresented.
