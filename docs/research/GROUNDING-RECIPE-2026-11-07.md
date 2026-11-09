# Grounding recipe — what to change, ranked by measured evidence (2026-11-07)

A single page a future session can act on without re-reading twelve notes. Every claim has either a measured
number with its artifact or is marked UNMEASURED.

## The core finding

`grounding = P(type correct) × P(grounded | type correct)`. On AC-v2 398, our fine-tune stands at 87.9% and 63.8%.
The second factor is where the remaining room is, and the loss that trains it **has no gradient where its errors
are** (CE constant at 690.78 nats beyond ~400 bins; misses are median 437 px).

## Ranked levers

| # | lever | measured effect | cost | status |
|---|---|---|---|---|
| 1 | **`aux7k` auxiliary grounding data** (re-frame existing screenshots' instructions/labels into grounding targets) | **+2.13 grounding**, +2.8 on click rows (n=179), accuracy +0.50 (ns) | 0 new pixels; +2,500 rows of JSON | **measured once, seed 2 running** |
| 2 | **858 px images** (vs 720) | groundB suggests +2–3 (residual after aux) | ~40% more visual tokens/step | **running** |
| 3 | **more steps** (938 → 1,406) | E2@705→938 was +3.52 paired on the same rows | 50% more compute | measured elsewhere |
| 4 | **bounded coordinate loss** (spatial softmax instead of per-edge Gaussian) | UNMEASURED | code change + 1 run | not started |
| 5 | **contrastive vs own distractor** (ConClick-style) | published +11.3 SSPro with 8,584 examples; transfer unknown | negative mining | not started |
| ❌ | candidate selection over a long list | **−10.17 grounding** (measured, failed) | — | closed above ~40 options |
| ❌ | two-stage shortlist | needs ≥71.4% selector just to break even; proxy did not replicate | 2× latency | parked |

## The one rule the data does support about interfaces

| candidates in state | val Δ | test Δ |
|---|---|---|
| 1–10 | +3.8 | −4.0 |
| 11–20 | +10.0 | −7.3 |
| 21–40 | 0.0 | −5.8 |
| **41+** | **−17.3** | **−23.8** |

Only the 41+ collapse replicates. **Do not put more than ~20 options in the state.**

## What to run next, in order

1. **Seed 2 of `aux7k`** (running) — turns +2.13 from a point estimate into a replicated one. If seed 2 disagrees
   in sign, the data lever is not established and lever #1 drops below #3.
2. **858 px alone** (running) — completes the decomposition of `groundB` (+6.01 = aux 2.71 + steps/resolution 3.30).
3. **`aux7k` + 858 px together** — checks additivity. Prediction 68–70%.
4. Only then consider a loss change (#4/#5), because #1–#3 are cheaper and already have evidence.

## Honest limits that must go with any of these numbers

- All arms are single-seed so far.
- The grounding denominator is the **188 shared rows** where both runs emit a spatial type; quoting 56.05% and
  66.49% side by side without saying that would overstate precision.
- `aux7k` uses **only** screenshots already in `cur15k` (6,574/6,574 verified) — it is a supervision framing
  change, not new data. Any framing of it as "more data" is wrong.
- Nothing here is an on-device or online result.
