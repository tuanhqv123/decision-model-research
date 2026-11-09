# 2026-11-07 — one day, ten retractions, three things that survived

The most useful output of this day is not a number. It is that the programme's core methodological problem is now
**measured**: two runs of the identical recipe differ by **0.90 points** of grounding, and the baseline differs from
its own duplicates by **4 points**, which is the size of every effect we were chasing.

## What survived (and why each is safe)

| finding | number | why it is not seed noise |
|---|---|---|
| **The box loss has no gradient where our errors are** | CE is a numerically identical **constant (690.78 nats)** for offsets ≥ 200 bins; ΔCE/bin = 0 beyond ~400 px. Misses: median **437 px**, 86.8% > 150 px | arithmetic on the loss definition; no training run involved |
| **What the fine-tune actually did** | base **17.4% → 64.07%** step acc; grounding **14.4% → 56.1%**; grounded-given-type **22.7% → 63.8%** | a 40-point gap, ~20× the seed spread, same scorer, same rows |
| **Two thirds of the grounding gain is placement, not verb choice** | placement **+26.0**, verb **+5.6**, interaction +10.1 of the +41.7 | derived from the base measurement |
| **Candidate lists above ~40 destroy placement** | **−17.3** val, **−23.8** test | replicates on both splits, 8–12σ |
| **The decision-level claim** | gate beats always-act *and* always-ask for `C_err/C_ask` in **2.11–5.79** (2.08–4.19 on the next checkpoint) | arithmetic on saved risk/coverage artifacts; unaffected by which checkpoint produced them |
| **The label rule bug** | 12 eval rows were supervised with a point the scorer can never award; fixed; scoreable ceiling **97.99%** not 99.99% | code inspection + re-derivation |

## What was retracted, and by what

| claim | killed by |
|---|---|
| VA beats box at matched weights (+6.03) | a 938-step rerun: **−7.29, p=0.0013** |
| short candidate lists help (+10.0) | test split: **−4.0 / −7.3** |
| candidate coverage ceiling 99.99% | the scoreable rule: **97.99%** |
| gold ranks top-5 by text 82.5% | tie-handling bug: **38.8%** |
| ensemble +2.01 | matched 3-voter size: **+0.50**; 16-cell correction: **p=0.17** |
| union +8.10 | it is an **oracle**; free arbiter **+0.60, p=0.68** |
| `aux7k` is a free framing lever (+2.0) | control: duplicating rows beats it (**60.1 vs 57.4**) |
| duplication is a data-volume win (+4.0) | `epoch: 0` in every run — the arms differ by **8 samples of 15,008** |
| the levers are additive (+4.75) | the stacked arm: **57.4%**, below 858 px alone |
| **all grounding deltas today** | **seed spread measured: range 4.04, sd 2.12** (rule pre-committed at 2.0) |

## The methodological finding, which is the day's real result

**Every arm was compared at n=1 seed against a pipeline whose own seed spread is ±2 points.** With `--steps 938`
and batch 16 the trainer draws 15,008 samples and never completes a pass (`epoch: 0` everywhere), so *changing the
data list only changes the shuffle*. That is why a 0.05% input difference produced a 4-point "effect".

**Rule now in `AGENTS.md`:** no AC-v2 grounding or accuracy delta may be reported without **≥3 seeds and an
interval**.

## What to run next, in order

1. **`seed1` finishing now** → completes the 3-seed baseline (56.05 / 59.19 / 60.09 so far), giving a defensible
   spread instead of an estimate.
2. **Only then** pick ONE lever and run it at 3 seeds: `858 px` is the best candidate because it has a real
   mechanism (more visual tokens per target) and a measured cost (**+45% latency, 106.8 → 154.4 ms p50**) that
   must be reported with any gain.
3. **Do not** run another single-seed arm on AC-v2. That is how the nine retractions happened.

## The one-sentence state of the programme

> The fine-tune is real and large, the loss cannot see the errors it is supposed to fix, the gate's cost window is
> defensible — and every attempt to squeeze another 2–4 points out of AC-v2 this day landed inside the pipeline's
> own seed noise, which is now measured rather than suspected.
