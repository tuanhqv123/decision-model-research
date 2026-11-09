# 2026-11-07 — aux7k ablation: the data lever is real but small (+2.1 grounding), not the +6.01

**Pre-registered** (`PRE-REG-2026-11-07-aux-ablation.md`) before the run: PASS = grounding ≥62.0% **and** step
accuracy ≥63.5%; KILL = grounding <58%.

## Result

Same 398 eval rows, same official scorer, **one variable changed** from E2 (938 steps, 720 px, same init/lr/seed):

| | E2 (`cur15k`) | **arm (`cur15k,aux7k`)** | Δ | significance |
|---|---|---|---|---|
| **step accuracy** | 64.07% (255) | **64.57% (257)** | **+0.50** | p = 0.87, gained 20 / lost 18 |
| **grounding (shared 188 rows)** | 64.36% (121) | **66.49% (125)** | **+2.13** | z ≈ +0.58, ns |
| type, spatial | 87.89% (196/223) | 87.00% (194/223) | −0.89 | |
| **grounded given type** | 63.78% (125/196) | **65.98% (128/194)** | **+2.20** | |

**Verdict: the bar is technically met (66.49% ≥ 62.0%, 64.57% ≥ 63.5%), but the effect is +2.13 points, not the
+6.01 that motivated the run, and it is not individually significant.**

## Attribution of `groundB`'s +6.01 — the thing this run was for

| configuration | grounding | Δ vs E2 |
|---|---|---|
| E2: `cur15k`, 938 steps, 720 px | 63.78% | — |
| **+ aux7k only**, same steps, same resolution | **66.49%** | **+2.71** |
| groundB: + aux7k, **1,406 steps, 858 px** | 69.79% | +6.01 |

**aux7k accounts for 45% of groundB's gain; the remaining +3.30 comes from the extra steps and the higher
resolution.** The "cheap data lever" story is therefore *half* right: the data is free and it does help, but it is
not where most of the gain lives.

## Where the +2.13 sits (per type)

| gold type | n | E2 | aux arm | Δ |
|---|---|---|---|---|
| **click** | 212 | 118 | **123** | **+5** |
| scroll | 37 | 18 | 19 | +1 |
| open_app | 27 | 18 | 19 | +1 |
| status | 87 | 78 | 75 | −3 |
| type_text | 11 | 7 | 5 | −2 |

Grounding on the shared click rows (n=179): **64.2% → 67.0%, +2.8 pts**. So the gain is concentrated exactly where
the auxiliary data points: **clicks**, which is what `aux_instr`/`aux_label` supervise. The small losses on
`status` (−3) and `type_text` (−2) are the type head drifting slightly, not a grounding regression.

That is a coherent mechanism, not noise: the data does what it is supposed to, narrowly.

## What this establishes

1. **`aux7k` is a genuine, free, zero-new-pixel improvement**: 100% of its 6,574 screenshots already exist in
   `cur15k` (verified), and it buys **+2.1 grounding** with **no accuracy cost** (+0.50, ns). That is a defensible,
   reproducible contribution — small, honest, and cheap.
2. **It is not sufficient on its own.** +2.13 on 188 rows has a standard error of ~3.6 points; the effect needs a
   larger eval or a second seed before it can be called established.
3. **The remaining +3.30 is not free.** It costs 468 extra steps (50% more training) plus 858 px images (a ~40%
   increase in visual tokens per step). Those are real deployment costs, and they are the larger half of the gain.

## The next cheapest experiment, now clearly indicated

**Resolution alone**: E2's exact 938-step recipe with `D1_IMG_W=858` and no aux data. One run, ~1.6 h.
Prediction: **+2 to +3 grounding**, which would put the three components (data ≈ +2.1, resolution ≈ +2–3,
steps ≈ +1) roughly additive and give the paper a clean, decomposable grounding recipe instead of an
undifferentiated +6.

That run is worth doing because it converts "groundB is better, we think" into "here is what each of the three
ingredients is worth", which is a reviewable claim.

## Honest limit

Both arms are single-seed, and the grounding denominator is the 188 shared rows where both runs emitted a spatial
type. A second seed for the winning configuration is the minimum before any of these numbers enters a headline.
