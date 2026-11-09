# 2026-11-07 — The two-stage design: a 6-item list that contains the answer 78.5% of the time

This is the constructive output of the failed VA arm. It uses the mechanism we measured (state length, not the
interface, caused the loss) and it needs **no gold label at inference**.

## The measurement

Stage 1 is the model we already have (E2, box path, ~103 ms, 1,537-token state). It emits a point. Stage 2 shows
the model only the candidates **within R pixels of that point**, so the list is short.

Measured on the 223 gold-spatial eval rows, using **E2's own predicted point** (no gold anywhere):

| R (px) | median list | p90 | max | gold inside the short list | with a nearest-neighbour fallback |
|---|---|---|---|---|---|
| 200 | 2 | 4 | 12 | 61.4% | **65.5%** |
| 300 | 3 | 7 | 20 | 69.5% | **70.9%** |
| **450** | **6** | **11** | 48 | **78.5%** | **78.5%** |
| 600 | 9 | 18 | 84 | 84.3% | 84.3% |

E2's own grounding on the same rows is **56.05%**.

## Why this is the right shape, and the previous run was not

| | VA arm (ran, failed) | two-stage (this) |
|---|---|---|
| state length | ~2,164 tokens, every step | stage 1 ~1,537; stage 2 ~6 candidates ≈ 200 tokens |
| damage to the box head | **−4.27 pts** (measured on the box-only path) | none — stage 1 *is* the box head, unchanged |
| candidate list | 14 of 36, dropping 58.6% of distractors | ~6 of 25, all of them near the prediction |
| ceiling | 97.99% (unreachable — the model could not read the long list) | **78.5%** at R=450 |
| gold used at inference? | no | no |

The key difference is that **stage 2 never competes with the image**. The long-context failure is structural: a
2,100-token list placed in front of the screenshot degrades coordinate prediction by 4 points. A 200-token list of
six nearby candidates does not have to.

## What the numbers say the ceiling is worth

+22.5 points of grounding headroom (56.05% → 78.5%) is available from a *second selector pass* whose input is six
items. Even a selector that is right only half the time on those lists adds ~+11 points — which would take
grounding from 56% to ~67%, past every number this programme has produced.

## The honest risks

1. **R=450 px is 45% of the screen height.** The list is short because it is *near*, not because it is *small*.
   On a dense screen (p90 list = 11, max = 48) the stage-2 state grows; the design must cap at ~20 and verify
   the cap keeps the gold (the same `GOLD_KEEP` mechanism already built).
2. **Two forward passes = ~206 ms**, which breaks the 103 ms single-pass claim. Must be stated: *"single pass on
   84% of steps, two passes when the box head is confident enough to shortlist."*
3. **It only works where an a11y tree exists.** On the online AndroidWorld loop there was none. This is an
   offline/AC-v2 result unless a pixel-based detector supplies candidates.
4. The 78.5% is a **ceiling** — it assumes the selector always picks correctly from the short list. It is not a
   measured accuracy.

## Next experiment, pre-registered

`docs/research/PRE-REG-2026-11-07-two-stage.md` (written before the run):

- Stage 2 trained from the **E2 checkpoint**, VA track, but with the state reduced to the shortlist for R=450.
- Training data: for each spatial training row, the shortlist is built from the **gold-free** filter that will be
  used at inference — i.e. from E2's predicted point on that row, **not** from the gold point. Using the gold
  point would make training and inference distributions differ, which is exactly the skew that R4 found.
- PASS: grounding ≥ 68% (i.e. +12 over 56.05%) **and** step accuracy ≥ 62%.
