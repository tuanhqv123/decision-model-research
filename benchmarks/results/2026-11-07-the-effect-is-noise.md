# 2026-11-07 — The "duplication" gain is almost certainly seed noise, not data

I chased this all day in the wrong direction twice. Here is the arithmetic that settles it.

## The two runs differ by 8 samples out of 15,008

`train_d1.py` reshuffles only when the cursor passes the end of the list (lines 452–454). With `--steps 938` and
batch 16, each run draws **15,008 samples**, and then stops:

| arm | rows in list | samples drawn | distinct rows seen | wraps |
|---|---|---|---|---|
| E2 | 15,000 | 15,008 | ~15,000 + 8 repeats | 1 (final step) |
| "duplication control" | 22,500 | 15,008 | **15,008 distinct** | **0** |

**The confirmering log line is `epoch: 0` for every run on disk** — including the 22,500-row arm through step 930.
Neither arm ever completed a pass over its own list.

So the two configurations that differ by **+4.0 grounding (56.1% → 60.1%)** and **+2.3 step accuracy** differ in
input by **8 samples (0.05%)**, and in ordering by nothing more than `random.Random(seed).shuffle` over a list of
different length — which produces a **different random permutation of the same data**.

**A 4-point grounding difference from a different shuffle of the same 15k rows is a seed effect, not a data
effect.** The duplication did essentially nothing; the shuffle did everything.

## What this means for everything I wrote today

| claim | status |
|---|---|
| "duplication beats framing" | **DEAD.** There was no meaningful volume difference. |
| "more rows of the same task is the lever" | **DEAD, same reason.** |
| "`aux7k` gives +2.0 grounding" | **AT RISK.** Same failure mode: a different list → a different shuffle → a different draw. Its 2 seeds (+2.23, +1.68) are more reassuring than one, but all of them share the *same* 15,008-sample budget. |
| "858 px gives +2.79" | **AT RISK** for the same reason, and it is the one arm with a real mechanism (more visual tokens per target) — so it deserves the seed test first. |

**The honest statement tonight: every arm difference I have measured is within the spread that one seed of this
pipeline produces, and I do not know that spread.**

## The measurement that has to happen before anything else

**Seed spread of the baseline.** Run E2's exact recipe with `--seed 1` and `--seed 2`, evaluate all three on the
same 398 rows, and report the spread of grounding and step accuracy. That number decides whether *any* of today's
+2-to-+4 deltas is real.

Cost: two runs, ~3 h, both GPUs are free after the current jobs. Until it exists, the correct claim from today is:

> **Nothing measured today exceeds the seed spread, because the seed spread has not been measured.**

That is a worse sentence than the ones I wrote this morning, and it is the true one.

## What *is* still solid, because it does not depend on a delta

- The **loss-gradient finding** (CE constant at 690.78 nats beyond ~200 bins; 86.8% of misses beyond 150 px) is
  arithmetic on the loss definition, not a training comparison.
- The **base-vs-fine-tune** gap (17.4% → 64.07% accuracy, 14.4% → 56.1% grounding) is far larger than any
  plausible seed spread, and it was measured on the same rows with the same scorer.
- The **candidate-list collapse above 40 options** replicates on both splits at −17 to −24 points, which is also
  far outside a seed spread.
- The **decomposition** of the fine-tune's grounding gain (placement +26.0, verb +5.6) follows from the base
  measurement.
