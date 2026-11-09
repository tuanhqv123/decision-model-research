# 2026-11-07 — The one thing that already worked: auxiliary grounding data (+6.0 grounding, +0.75 accuracy)

After a day of failures, this note records the configuration in this programme that **measurably improved
grounding**, and what it implies. It is not new — it is `groundB`, run on 2026-10-29 — but it was never given its
due because the day's attention went to the box-vs-candidate question.

## The comparison, same 398 rows, same scorer

| run | recipe | step acc | grounding |
|---|---|---|---|
| **E2** | `cur15k`, 938 steps, 720 px | 64.07% (255) | 63.78% (125/196) |
| **groundB** | `cur15k` **+ `aux7k`**, 1,406 steps, 858 px | **64.82%** (258) | **69.79%** (134/192) |
| Δ | | **+0.75** | **+6.01** |

`aux7k` is 7,000 auxiliary GUI-grounding samples built from **the same screenshots** the training set already
uses (`make_aux.py`), in two flavours:
- `aux_instr`: state = the step's low-level instruction ("Click on the search icon"), target = that element's box;
- `aux_label`: state = an element's visible label ("Search"), target = that element.

No new images, no new annotation, no external dataset. It is a **re-framing of labels we already had** into extra
grounding practice.

**Verified, not assumed:** of the 6,574 unique screenshots in `aux7k`, **6,574 (100.0%) already appear in
`cur15k`**. So the auxiliary set adds zero pixels — only additional (state, target) pairs over screens the model
already trains on. Any gain it produces is therefore attributable to *supervision framing*, not to data volume.

## Why this matters more than the candidate-interface work

| approach | grounding Δ | evidence class |
|---|---|---|
| **`aux7k` auxiliary grounding** | **+6.01** | **measured, same scorer, same rows** |
| candidate selection (VA, 14-cap) | −10.17 | measured, failed |
| two-stage shortlist | unknown, ≥71.4% selector needed to even break even | unmeasured; the only proxy (short-list buckets) **did not replicate** |
| loss redesign (spatial-softmax) | unknown | unmeasured, no isolated published delta |

**The grounding lever with a measured effect is data, not interface.** That is the honest state of the evidence,
and it contradicts where the last 8 GPU-hours went.

## What the +6.01 probably consists of

`groundB` changed three things at once (aux data, +468 steps, 858 px images), so the +6.01 is not attributable
without an ablation. The cheapest decomposition, in order of cost:

1. **aux7k only** — rerun E2's exact 938-step recipe with `--train-set cur15k,aux7k`. One 4090 run, ~1.6 h.
   **This is the highest-value remaining experiment in the programme**: if it reproduces +6, grounding is a data
   problem and the paper has a clean, cheap, reproducible lever.
2. **resolution only** — E2 recipe with `D1_IMG_W=858`. Same cost. Prediction: +1–2, not +6.
3. both, to check additivity.

## Why the ablation matters for the paper

If (1) reproduces, the paper's grounding story becomes: *"element selection improves 6 points from re-framing
existing labels into grounding practice, with no new data and no architecture change"* — a reproducible,
zero-cost contribution. That is worth more than any of today's interface work, and it is one 1.6-hour run away
from being established rather than inferred.

## Pre-registration for that run

`docs/research/PRE-REG-2026-11-07-aux-ablation.md`: PASS = grounding ≥ 62.0% (halfway to groundB's 69.79) with
step accuracy ≥ 63.5%; KILL = grounding < 58% (then the +6.01 was the steps or the resolution, not the data).
