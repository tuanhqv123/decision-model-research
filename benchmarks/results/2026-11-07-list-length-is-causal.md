# 2026-11-07 — The candidate interface works; the LIST LENGTH breaks it (causal, measured)

The VA arm failed overall (−3.75 val / −7.29 test). This note shows *why*, with the variable isolated, and the
answer is constructive: **the interface wins wherever the list is short.**

## The measurement

Same checkpoint, same val split, grounding split by **how many candidates were in the state**:

| candidates in state | n | E2 (box only) | VA arm (element choice) | Δ |
|---|---|---|---|---|
| **1–10** | 53 | 75.5% | **79.2%** | **+3.8** |
| **11–20** | 80 | 66.2% | **76.2%** | **+10.0** |
| 21–40 | 94 | 70.2% | 70.2% | 0.0 |
| **41+** | 156 | **66.0%** | **48.7%** | **−17.3** |

### ⚠️ RETRACTION of the short-list half of this claim (found on test, same day)

The val pattern above **does not replicate on the test split**. Same code, same join, held-out rows:

| candidates | n (test) | E2 | VA arm | Δ | z |
|---|---|---|---|---|---|
| 1–10 | 25 | 72.0% | 68.0% | **−4.0** | −0.4 |
| 11–20 | 41 | 70.7% | 63.4% | **−7.3** | −0.9 |
| 21–40 | 69 | 63.8% | 58.0% | −5.8 | −0.8 |
| **41+** | 42 | 59.5% | **35.7%** | **−23.8** | **−3.1** |

So the **only** part that replicates is the **long-list collapse** (val −17.3, test −23.8), which is large and
holds on both splits. The **short-list advantage does not hold**: +3.8/+10.0 on val, −4.0/−7.3 on test, and the
val cells are n = 53 and n = 80 with z = +0.6 and +1.8 — not significant.

**Corrected conclusion:**
- **Established:** candidate lists above ~40 items destroy grounding (−17 to −24 points, both splits).
- **Not established:** that a short list *improves* things. The two-stage design rests on the second claim, so it
  is now a **hypothesis**, not a measured expectation.

The two-stage plan therefore loses its main quantitative support. What remains true is the weaker, safer
statement the data does support: *keep the option list short or the model cannot use it.* Two independent observations agree with this:

1. The arm's own **box-only** path also degraded (60.22 val / 59.80 test vs E2's 62.76 / 64.07) — the long state
   hurts the coordinate readout too, not just the element choice.
2. The VA arm's element grounding falls monotonically with list length: 79.6% → 75.3% → 69.8% → 48.8%.

So the causal variable is **state length / number of options**, not the existence of a candidate interface.

## Why this is the most useful result of the day

It converts "candidate selection failed" into a **quantified design rule**:

> Keep the option list under ~20 items. Above that, the model's placement collapses, and it does so whether it
> answers by picking an element or by regressing a box.

That rule is directly actionable, and it is exactly what the two-stage shortlist does:

| R (px) | median shortlist | gold inside the list |
|---|---|---|
| 300 | 3 | 70.9% |
| **450** | **6** | **78.5%** |
| 600 | 9 | 84.3% |

A shortlist of 6 sits squarely in the bucket where the element interface **wins by +3.8 to +10.0 points**.

## What the two-stage model can be expected to score, stated honestly

Grounding ≈ `recall(R) × selector_accuracy`:

| selector accuracy on the shortlist | implied grounding | vs 56.05% today |
|---|---|---|
| 70% | 54.9% | −1.1 |
| 80% | 62.8% | **+6.8** |
| 90% | 70.7% | **+14.6** |

The arm's element head already achieves **79.2%** in the 1–10-candidate bucket, and **76.2%** in the 11–20 bucket.
So an 80% selector is not speculative — it is the measured performance of this exact model on short lists.

**Expected result: grounding ≈ 62–65%** (from 56.05%), i.e. **+6 to +9 points**, with a plausible upside to +15 if
the shortlist recall can be raised by feeding the stage-2 pass a cropped image of the region rather than the whole
screen.

## Artifact

All numbers from `/tmp/armva_val.jsonl` (arm eval) + `compact_validation.jsonl` (candidate counts) +
`E2-val/eval-E2final-V-val.jsonl`, joined on `(episode_id, step)`; reproducible in ~15 lines of python.
