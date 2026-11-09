# PRE-REGISTRATION — candidate-selection grounding head (2026-11-07)

*Frozen before the run. Pass conditions are not moved afterwards.*

## Why this arm, in one paragraph with numbers

Grounding is **56.05%** (125/223) while action type is **84.17%**. The 143 errors decompose as 71 element-wrong /
27 spatial-type-wrong / 36 non-spatial-type-wrong / 9 value-wrong. Miss distances: median **437 px**, 86.8%
beyond 150 px, only 4.4% within 50 px. So the model picks **a different element**, not a nearby pixel. Separately:
the gold point lies **inside some a11y candidate on 8,399/8,400 training rows (99.99%)** and **471/474 val rows
(99.37%)**, and on the 398 eval rows the same measurement gives **97.2%** (206/212). A head that *chooses* among
candidates therefore has a ~99% recall surface, against a box head that reaches 56%.

**And the interface is not redundant with the box.** On the 398 test rows:

| | step acc |
|---|---|
| V (box) alone | 64.07% |
| VA (candidate choice) alone | 55.53% |
| **union** | **71.36% (+7.29 vs V)** |

VA-only-correct 29 rows, V-only-correct 63, McNemar p = 0.0005. The two interfaces fail on **different rows**.

## Hypothesis

> **H1: with a matched training budget, a candidate-selection head reaches materially better grounding than the
> 56.05% box head, and the union of the two is better than either.**

The previous candidate run (E3-300-VA) got 47.5% element grounding, but it was trained on a **different dataset**
(`cur24k`) for **1,500 steps** in **mix** mode, while the box baseline is `cur15k` / **938 steps** / V-only. That
comparison is confounded by budget and data, so "the candidate interface fails" was never actually tested.

## Arm (one variable: the interface, everything else matched)

| | baseline (E2) | arm |
|---|---|---|
| init | `LiquidAI/d1-2B` | same |
| data | `compact_cur15k` | **same** |
| steps / lr / warmup / seed | 938 / 1e-5 / 30 / 0 | **same** |
| track | `V` (box readout) | **`VA`** (box + `target_el` choice) |
| gold supervision | `target_box` edges | box edges **+ `target_el`** (82.0% of target rows have an exact match) |
| micro / accum / mem-frac | 1 / 16 / 0.38 | same |
| eval | AC-v2 398 + val 827, official scorer | same |

The ONLY change is the track flag, which adds the candidate list to the state and the `target_el` question. This
isolates "does giving the model a selection interface help" from every other difference.

## Primary endpoint and pass condition (frozen)

- **Primary: grounding on the 223 spatial test rows**, official `score_row` (point inside the smallest a11y node
  containing the gold point).
- **PASS:** grounding ≥ **62.0%** (i.e. ≥ +6.0 pts over 56.05%, ≈ +13 rows), **and** step accuracy does not fall
  below 62.0% (a drop > 2 pts means the interface broke the action head).
- Secondary: the **union** of the arm and E2 must be ≥ **70%** step accuracy. If the union does not improve on
  V alone, the interfaces are redundant and the line is closed.

## Prediction (falsifiable)

Grounding 56.05% → **62–70%**; step accuracy 64.07% → **62–66%**; union ≥ 71%. Reason for the wide band: 36
candidates median (p90 76) means the choice is over a long list, and the literature's small-model results on
long candidate lists are the weak point.

## Kill criteria

1. Grounding < 58% → the candidate head does not beat the box head; H1 is false, reported as a negative, and the
   entire "selection interface" line is closed.
2. Step accuracy < 60% → the state length (median 1,632 chars, p90 2,968) crowds out the action question; report
   as an interface cost, not an accuracy result.
3. Any run whose gold candidate is truncated out of the state → invalid; must be detected and reported, not
   tolerated.

## Known confound to report either way

`build_state` for the VA track appends up to 76 candidate lines. Median state 1,632 chars, p90 2,968 chars
(~408 / ~742 tokens). If the model's state budget truncates, the gold candidate can be cut — **28.9% of gold
candidates sit at index ≥ 30**. Any truncation must be measured, not assumed away.

## Cost

One GPU run, 938 steps, ~1.2–1.6 h on a 5090-class card, plus a 30-step smoke. No new data is generated; the
candidate lists are already in `compact_*.jsonl`.

## What invalidates the experiment

- Different data, steps, init, or lr between baseline and arm.
- Evaluating test more than once.
- Choosing the pass threshold after seeing the result.
