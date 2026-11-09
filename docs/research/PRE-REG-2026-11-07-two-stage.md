# PRE-REGISTRATION — two-stage shortlist selector (2026-11-07)

*Frozen before the run. Do not move the bars after seeing results.*

## Why this design, from measured failures

1. The VA arm **failed** (−3.75 val, p=0.019; −7.29 test, p=0.0013) but its **box-only path also failed**
   (−2.54 val, −4.27 test). So the long candidate state damages the coordinate head; the interface was not the
   problem.
2. A shortlist built from **E2's own predicted point** (no gold) at R = 450 px has **median 6 candidates** and
   contains the gold element on **78.5%** of the 223 spatial eval rows, against E2's 56.05% grounding.
3. Therefore: keep the long-context-free stage 1, and give stage 2 only ~6 items.

## Arm

| | baseline (E2) | arm |
|---|---|---|
| stage 1 | E2 box head, unchanged | **same weights, unchanged, frozen** |
| stage 2 | — | selector over the shortlist, `--track VA`, `D1_MAX_CANDS=20` |
| training data | `cur15k` | same rows; shortlist per row built from **E2's predicted point on that row** (gold-free) |
| steps / lr / warmup / seed | 938 / 1e-5 / 30 / 0 | same |
| micro / accum / grad-ckpt / freeze-embed | 1 / 16 / on / on | same |
| eval | AC-v2 398 + val 827, official scorer | same |

**Critical anti-skew rule.** Every training shortlist must be produced by *the same function* used at inference,
i.e. filtered by the stage-1 prediction, never by the gold point. If a shortlist is built from gold, training
sees lists that inference will never produce (the R4 defect class).

**This forces a prerequisite step that is easy to skip and fatal to omit:** we do **not** have stage-1
predictions for the 15,000 training rows. So the arm needs

> **Step 0 (prerequisite):** run the frozen E2 checkpoint over all 15k training rows, one forward pass each,
> and dump its predicted point per row (`eval_d1.py --track V --save-points`). Cost ≈ **50 min GPU, one time**.
> Only then can training shortlists be built from the model's own prediction.

The alternatives are all wrong and are recorded here so they are not re-invented:
- building shortlists from the **gold** point → train/inference mismatch, the exact R4 defect;
- building them from an **oracle near-miss** → optimistic bias with no inference analogue;
- assuming the gold point is "close enough" → it is not: at R=450 the gold-filtered shortlist is 78.5%, but the
  *prediction*-filtered one on training rows is unknown until Step 0 is measured.

## Pass condition (frozen, and corrected after arithmetic check)

**A flaw found in the first version of this bar, fixed before any run.** The 78.5% shortlist recall is a
*ceiling* that assumes a **perfect** selector. Grounding is `0.785 × selector_accuracy`:

| selector accuracy over the R=450 list | implied grounding | vs 56.05% today |
|---|---|---|
| 50% | 39.2% | **−16.8** |
| 60% | 47.1% | **−9.0** |
| 70% | 54.9% | −1.1 |
| 80% | 62.8% | +6.8 |
| 100% (the ceiling) | 78.5% | +22.5 |

So the original bar (grounding ≥68%) required a selector at **≥86.6%** on six items — near the ceiling, and
unrealistic for a 1.92B model that scores 56% today. **That bar would have failed by construction and told us
nothing.**

**Corrected PASS condition:** grounding ≥ **62.5%** on the 223 spatial test rows (≥ +6.5 over 56.05%), which
corresponds to a selector at **~80%** on the shortlist, **and** step accuracy ≥ 62.0%.

**Diagnostic that must accompany the number:** report the selector's accuracy *on the shortlist* separately from
the end-to-end grounding. If the selector reaches ≥80% but end-to-end grounding does not rise, the loss is in the
shortlist recall (R too small), not in the selector — and vice versa. Without that split, a null result is
uninterpretable.
- **Partial:** grounding 62–68% with step accuracy ≥62% → report as a gain, not a headline.
- **KILL:** grounding < 58% (below today's 56.05% baseline by any real margin is already covered by the PASS bar;
  the kill line is here so a *regression* closes the line rather than being re-tuned) → the two-stage line is
  closed; the remaining option is the loss redesign (spatial-softmax or contrastive), not more interface work.

## Secondary, must be reported

- **Latency:** p50/p95 passes per step. Two passes ≈ 206 ms typical; must be stated as
  "single pass on most steps, two when the shortlist is triggered".
- **Shortlist recall** on val and test (predicted ≈ 78.5% at R=450); if it falls below 75% the design is
  mis-specified regardless of accuracy.
- **Ablation:** R ∈ {300, 450, 600} for shortlist recall, chosen on val only.

## Cost

One training run (938 steps, ~1.5 h) plus three cheap evals. No new data. Requires ≥20 GB free VRAM (the previous
run needed 11.8 GB; the shortlist state is smaller, so 14 GB should suffice).

## What invalidates the experiment

- Shortlists built from gold at train time.
- A different stage-1 checkpoint than E2.
- Evaluating test more than once.
- Changing the pass threshold after the result.


## Third skew found before the run: the SHORTLIST distribution differs between train and eval

Stage 1 is the E2 checkpoint. Its accuracy on the rows it was **trained on** is 69.5% (1,180/1,698 sampled
training rows), versus **64.1%** (255/398) on the held-out eval rows. Consequently the shortlist it produces is
easier on train:

| rows | shortlist recall at R=450 |
|---|---|
| training rows (stage-1 predictions on seen data) | **91.3%** |
| eval rows (held-out) | **78.5%** |

A stage-2 selector trained on training-row shortlists would therefore learn on lists where the answer is present
91% of the time and be evaluated on lists where it is present 78% of the time. **That is a 13-point shift in the
task difficulty itself**, and it will bias the end-to-end number upward if not handled.

**Mitigation, mandatory for this arm:**
1. Report **shortlist recall separately on train and val** as a first-class diagnostic, not a footnote.
2. Train stage 2 on **val-sourced shortlists as well** if val is not used for model selection, or better:
   split the 15k training rows by **episode** and build shortlists for the held-out episode-fold using a
   checkpoint that never saw those episodes. Cheap version: reuse the existing val split (137 episodes) as a
   shortlist-shift probe.
3. Pre-commit that **any end-to-end gain smaller than 13 points is partly explainable by this shift** and must be
   decomposed before being claimed.
