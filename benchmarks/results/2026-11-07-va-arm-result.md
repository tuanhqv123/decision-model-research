# 2026-11-07 — Candidate-selection arm: FAILS the bar, but yields two real findings

**Pre-registered** (`PRE-REG-2026-11-07-candidate-head.md`) before the run. Recipe: `d1-2B`, init = E2 box
checkpoint, `cur15k`, 938 steps, lr 1e-5, `--track VA`, `D1_MAX_CANDS=14`.

## The result

| | E2 (box) | VA arm | Δ |
|---|---|---|---|
| **step accuracy (val 827)** | **62.76%** (519) | **59.01%** (488) | **−3.75 pts**, McNemar **p = 0.019** (gained 67, lost 98) |
| **step accuracy (test 398)** | **64.07%** (255) | **56.78%** (226) | **−7.29 pts**, McNemar **p = 0.0013** |
| grounding (shared spatial rows) | val **68.41%** (262/383) | val 63.97% (245/383) | **−4.44 pts** |
| grounding (test, shared) | **65.54%** (116/177) | 55.37% (98/177) | **−10.17 pts** |
| type accuracy | 80.53% | 77.99% | −2.5 |

**It fails both pre-registered clauses** (grounding ≥62% — it reached 63.61% on its own denominator — and step
accuracy ≥62% — it reached 59.01%). Per the pre-registration, a drop below 58% grounding would close the line;
it did not drop that far, so the line is **parked, not closed**, but no claim survives as written.

## Where the loss is, exactly

| gold type | n | E2 | arm | Δ |
|---|---|---|---|---|
| **click** | 420 | 253 | 224 | **−29** |
| **type_text** | 53 | 32 | 26 | **−6** |
| scroll | 86 | 41 | **51** | **+10** |
| status | 137 | 120 | 119 | −1 |
| open_app | 45 | 38 | 35 | −3 |
| navigate_back | 29 | 20 | 19 | −1 |
| wait | 56 | 15 | 14 | −1 |

Bucketed: **spatial gold −35 rows**, **non-spatial +4 rows**.

**So the arm did not damage reading — it damaged placing.** Scroll improved (+10, the biggest single gain), and
non-spatial decisions were flat-to-positive, while click placement lost 29 rows.

## Finding 1: adding the candidate list breaks the box head (mechanism, confirmed on test)

The arm still trains and reports a box readout. Measured on the same checkpoint, with the element path never
consulted:

| split | E2 box | arm's **box-only** path | Δ |
|---|---|---|---|
| val 827 | 62.76% | **60.22%** | −2.54 |
| **test 398** | **64.07%** | **59.80%** | **−4.27** |

Both splits, same direction, same size order. The coordinate readout the model already had **degrades by 2.5–4.3
points just from having ~2,100 candidate tokens in front of the image**.

So the loss is **not** "the element interface is worse". It is: **putting ~2,100 extra state tokens in front of
the model degrades the coordinate readout it already had.** The candidate list competes with the image for
attention. This is measurable, reproducible, and would have been missed by looking only at the element path.

## Finding 2: the two interfaces carry different information — but the union is an ORACLE, not a method

| | val 827 |
|---|---|
| E2 alone | 519 (62.76%) |
| arm alone | 488 (59.01%) |
| **union (val)** | **586 (70.86%)** — **+8.10 pts over E2** |
| **union (test)** | **279 (70.10%)** — **+6.03 pts over E2** |

98 rows where only E2 is right, 67 where only the arm is right. Two interfaces on the *same base model* disagree
on 165 of 827 rows.

**But 70.86% requires the gold label to choose between them.** The deployable versions do not work:

| arbiter | val 827 | Δ vs E2 | gained/lost | p |
|---|---|---|---|---|
| always E2 | 62.76% | — | — | — |
| pick higher `box_conf` | 63.36% | **+0.60** | 50 / 45 | 0.68 |
| pick higher action `conf` | 62.64% | −0.12 | 23 / 24 | 1.00 |
| **oracle union** | **70.86%** | +8.10 | — | **not deployable** |

The free confidence signal is at ~50/50 precision on the disagreements (62.4% accuracy against a 59% chance
rate; it gains 50 and loses 45). So the +8.10 is a **ceiling on what a much better arbiter could win**, not a
result. This is the third time identical arithmetic has appeared in this programme, and it now has a name:
*the arbiter, not the headroom, is the binding constraint*.

## What is ruled out, and what is now the indicated fix

Ruled out by this run:
- "Candidate selection with a 14-candidate cap, warm-started from a box checkpoint, at the same step budget, is
  better than the box interface." **False, −3.75 with p = 0.019.**

Indicated by the mechanism (not yet tested):
- The degradation is **state length**, so the fix is to give the candidate list its own budget rather than
  competing with the image — a **two-stage** design: box head predicts a region (as E2 does), then a *separate*
  short state lists only the candidates inside that region for a selection pass. That keeps each forward pass
  short, and it is the one design that uses both findings (union + the state-length mechanism).
- Also untested: the cap. 14 candidates drops **58.6% of distractors**; on a card with ≥20 GB free the same arm at
  40–64 candidates would test the interface without the memory handicap.

## Two measurement caveats that must travel with these numbers

1. **The grounding denominators differ.** The arm's own summary reports grounding over the rows where *its* type
   prediction made grounding defined (393 val rows); E2's is over 422. Quoting 63.61% against E2's 68.41% would
   be an apples-to-oranges comparison. **Every grounding number in this note is on the shared 383-row
   intersection** (E2 68.41%, arm 63.97%), and the test-side comparison is on their shared 177 rows.
2. **The eval ran pre-patch code, so `el_idx` is absent** (0/827 rows). Step accuracy and grounding are unaffected,
   but the per-candidate post-mortem that R4 asked for cannot be done on this artifact. Re-running costs ~2.5 min
   and is worth it only if the two-stage design needs this checkpoint — which it does not (stage 1 is unchanged
   E2).

## Honest status of the pre-registration

| clause | threshold | measured | verdict |
|---|---|---|---|
| grounding | ≥ 62.0% | 63.61% (own denominator) / 63.97% (shared) | **passes the number, but the denominator differs from E2's** |
| step accuracy | ≥ 62.0% | **59.01%** | **fails** |
| union (secondary) | ≥ 70% | **70.86%** | **passes** |

Because the primary clause fails, **no claim is made**. The union number is recorded as a measurement, not as a
result of this arm.
