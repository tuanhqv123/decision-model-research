# 2026-11-07 — WHY grounding plateaus at 56%: the box loss has no gradient where the errors are

This is the mechanism. It is arithmetic, not opinion, and it explains a plateau the programme has been stuck on.

## The loss, exactly

`target_box` is supervised as four edges, each a per-edge cross-entropy against a Gaussian-smoothed target on a
**0–999 grid**, with **σ = 4 bins**.

For a 1080×2400 screenshot:
- one grid cell = **1.08 px** horizontally, **2.40 px** vertically;
- σ = 4 bins = **4.3 px / 9.6 px** — i.e. the tolerance is a few pixels, far tighter than any element.

## What that costs when the prediction is wrong by a realistic amount

Cross-entropy of the true edge distribution under the model's, as a function of how far the model is off:

| offset (bins) | offset in y (px) | CE (nats) | Δ CE per bin |
|---|---|---|---|
| 0 | 0 | 2.16 | — |
| 4 | 9.6 | 2.81 | 0.40 |
| 16 | 38.4 | 10.21 | 5.96 |
| 33 | 79.2 | 36.24 | 26.0 |
| 57 | 136.8 | 103.74 | 23.4 |
| 100 | 240.0 | 314.71 | 200.0 |
| **200** | **480** | **690.78** | **0.00** |
| **415** | **996** | **690.78** | **0.00** |

**Past roughly 400 bins the CE is a constant.** The gradient is *exactly zero*, because the Gaussian target and
the model's distribution no longer overlap in the numerical range the softmax preserves — and under bf16 (which
this model trains in) the underflow starts far earlier than the float64 arithmetic above shows.

## Why this is the whole story

Our measured error distribution: median miss **437 px**, mean 663 px, **86.8% of misses beyond 150 px**, only
**4.4% within 50 px**.

So the loss is:
- **sharp** where our errors are rare (within a few px),
- **flat** where our errors are almost all of them (hundreds of px).

A model trained on this objective is told, in precise numerical terms, how to fine-tune a click that is already
almost right, and told **nothing** about a click that landed on the wrong widget. Elements that are far apart are
indistinguishable to the gradient. That is a direct, sufficient explanation of a plateau at 56% with 84% type
accuracy: the verb is supervised with a discrete choice (which works), the placement with a loss that vanishes
whenever the choice is wrong (which doesn't).

## The three fixes this implies, in order of expected value

1. **Make the placement a discrete choice too.** Pick among the ~24–36 a11y candidates. The gold point is inside
   a candidate on 8,399/8,400 training rows; a wrong candidate is a *wrong class*, so the gradient never vanishes.
   This is also the interface that already beat the box head at matched weights (+6.03 pts, p = 1.5e-3).
2. **If keeping a coordinate head, bound the loss**: clamp or use a softmax over image patches (spatial softmax)
   rather than a per-edge Gaussian, so that a far-away prediction still carries a gradient towards the right region.
3. **Contrastive term against the chosen wrong element** (ConClick: +11.3 ScreenSpot-Pro with 8,584 examples).
   A pairwise margin between the gold candidate and the model's own mistaken candidate has a gradient exactly
   where the box loss has none.

## What this does NOT claim

- It is not a claim that σ=4 is "wrong" for precision. For a model that already clicks within 10 px it is fine.
- It is not measured that fixing the loss alone closes the gap — the candidate interface changes the *state* too.
  The pre-registered arm (`PRE-REG-2026-11-07-candidate-head.md`) isolates that.

## Provenance

Arithmetic in this note: `python3` with numpy, log-sum-exp form to avoid underflow; reproducible in four lines.
Inputs: σ = 4, grid 1000, image 1080×2400, and the miss distribution from
`error_budget.py` on `E2-nomem` (median 437 px, 86.8% > 150 px).
