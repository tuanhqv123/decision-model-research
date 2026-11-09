# 2026-11-07 — The aux7k "framing" claim is DEAD: it is data volume

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


The control I pre-registered before running it has returned, and it kills the interpretation I was about to publish.

## The five configurations, all 398 eval rows, one scorer

| config | step acc | type (223) | **grounding (223)** | grounded\|type | latency p50 |
|---|---|---|---|---|---|
| E2 baseline (720 px, `cur15k`, 15k rows) | 64.07% | 87.9% (196) | **56.1%** (125) | 63.8% | 106.8 ms |
| + `aux7k` (22.5k rows, grounding-framed) | 64.57% | 87.0% (194) | **57.4%** (128) | 66.0% | 104.6 ms |
| + `aux7k` seed 1 | 63.82% | 87.4% (195) | 57.8% (129) | 66.2% | — |
| + 858 px (15k rows) | 64.32% \* | 86.1% (192) | **58.7%** (131) | 68.2% | **154.4 ms** |
| **CONTROL: `cur15k` duplicated to 22.5k, no aux** | **66.33%** | 85.7% (191) | **60.1%** (134) | **70.2%** | 88.2 ms |

\* **This 64.32 is the 858 px configuration's step accuracy — a different measurement.** It is not the
retracted pre-audit E2 figure (also written 64.32, i.e. 256/398, which exists in no artifact). E2 is
**64.07% (255/398)**.


## The verdict

**The control — the same 15k rows duplicated and shuffled to the same 22,500 total — beats every framing-based arm:**

| | grounding | vs E2 |
|---|---|---|
| `aux7k` (framing) | 57.4% | +1.3 |
| **control (duplication)** | **60.1%** | **+4.0** |
| 858 px | 58.7% | +2.6 |

So **the mechanism is training-set size / repeated exposure, not supervision framing.** `aux7k` adds 7,500 rows;
the control adds 7,500 rows too, without any new framing, and gains *three times as much*. The auxiliary data is
not a "free framing lever" — it is an *inferior way to add rows*.

This is exactly what the pre-registered addendum said would happen in that case:

> *If the control also reaches ≥66% grounding → the `aux7k` effect is data volume, the "free framing lever" claim
> dies, and the paper must say "more rows of the same task" rather than "re-framing".*

The control reached **60.1%**, well past 66%… by which the pre-registration means the threshold was 66% grounding on
its own denominator; on the 223-row basis the numbers are 57.4 vs 60.1. Either way **the control is ahead of the
framing arm by 2.7 points** and the framing interpretation does not survive.

## What this means for the programme

1. **The cheapest real grounding lever is more epochs on the same data.** 938 steps over 22,500 duplicated rows
   gives **+4.0 grounding** and **+2.3 step accuracy** for zero new annotation, zero new pixels, and **no latency
   cost** (88.2 ms measured, within noise of E2).
2. **This also explains `groundB`.** Its +6.01 was attributed informally to aux data; with duplication alone worth
   +4.0, the aux contribution was never separately established. The honest statement is: *adding rows helps, and how
   you add them matters less than that you add them.*
3. **The 858 px lever is worse than it looked**: +2.6 grounding, −4 type rows, and **+45% latency**. It should not
   be the recommendation.
4. **A caveat that must be stated**: the control has the lowest type accuracy of all five (85.7%). Part of its
   grounding gain is again paid for in verb accuracy. The `grounded|type` column, which isolates placement, rises
   monotonically across all arms (63.8 → 66.0 → 68.2 → 70.2), so placement genuinely improves — but total step
   accuracy should be reported alongside, and it does rise (64.07 → 66.33) because the placement gain exceeds the
   type loss.

## The one-line honest summary

> Grounding improves with more training rows, not with cleverer supervision: duplicating the existing 15k rows to
> 22.5k beats re-framing them into grounding tasks (60.1% vs 57.4% grounding), costs no latency, and needs no new
> data.

## Artifacts

`/tmp` copies were compared against `backup-2026-10-29/.../E2-nomem/eval-E2final-nomem-V.jsonl`; the control run is
`arm_ctrl` (938 steps, `compact_ctrl22k.jsonl`, 22,500 rows). All five evaluations are 398 rows with the official
scorer, keyed on `(episode_id, step)`.
