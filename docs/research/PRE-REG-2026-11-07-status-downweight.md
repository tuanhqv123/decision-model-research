# Pre-registration — status down-weighting (stage 2, arm 1)

*Written 2026-11-07 before the run. Pass conditions are frozen here; they are not moved after seeing results.*

## Why this arm

The model emits `status` 98 times on the 398 test steps and 21 of those are premature (gold is a UI action) —
5.3 accuracy points, the size of the whole gap to Mem-W-8B. The labels are **synthesized**: one completion
label per episode, generated from the final screen (`preprocess_ac.py`). 22% of the 15k training decisions are
`status`, against a 21.9% base rate in the eval set.

Two zero-GPU attempts to fix this by post-processing (hand routing, per-class logit adjustment, a calibrated
`p(status)` gate) have all failed. The `p(status)` gate is **monotone-harmful at every threshold** (best rule =
never switch; `benchmarks/results/2026-11-07-status-gate.md`). That closes the scoring branch and leaves exactly
one hypothesis worth GPU time:

> **H1: the model's premature terminations are caused by over-supervision on synthesized `status` rows.**
> Down-weighting that class during training should reduce premature terminations more than it costs true ones.

## Arm (single variable)

Retrain from the **base** `LiquidAI/d1-2B`, E2's exact recipe and schedule, with one change: the training file
keeps every non-`status` row and a random 20% of `status` rows (seed 0).

| | Baseline (E2) | Arm |
|---|---|---|
| init | `LiquidAI/d1-2B` | same |
| data | `compact_cur15k` (15,000 rows) | `compact_cur15k_status20` (12,361 rows) |
| `status` share | 22.0% (3,300) | 5.3% (661) |
| steps / lr / warmup / micro / accum | 938 / 1e-5 / 30 / 2 / 8 | same |
| track, freeze-embed, grad-ckpt, mem-frac | V, true, true, 0.92 | same |
| eval | AC-v2 398 + val 827, official scorer | same |

Choosing a from-base retrain rather than a continuation keeps the comparison single-variable: E2 and the arm
have the same init, schedule, and evaluation; only the class mix differs.

## Primary endpoint and pass condition (frozen)

- **Paired Δ step accuracy on the 827-row validation split**, episode-clustered bootstrap (the project's
  `paired_eval.py`), arm vs E2.
- **PASS:** Δ ≥ **+1.5 points** on val, **and** no gold action type loses more than **1 row**.
- Report per-type Δ so a `status`↔action trade cannot masquerade as a win.
- Test (398) is evaluated **once** after the pass/fail decision and reported with its interval.

## Prediction (falsifiable)

Premature `status` 21 → 8–12; `status` accuracy 89.7% → 83–87%; net test Δ between **+1 and +4 points**. If the
net is negative, H1 is false and the next arm is a learned progress readout, not a data re-weighting.

## Kill criteria

1. Smoke (30 steps) fails to run, OOMs, or does not decrease loss → stop, debug cheaply, no full run.
2. Full run completes but val Δ < 0 → stop this line; do not tune the weight post hoc.
3. Any gold type loses >1 row → the arm does not pass even if total Δ is positive; recorded as a trade.

## Cost

One GPU run, ~1.2–1.6 h on a 5090, plus a 30-step smoke (~5 min). No model calls on the Mac.

## What invalidates the experiment

- Different steps, lr, or init between baseline and arm.
- Choosing the down-weight fraction after seeing val.
- Evaluating test more than once.

## Execution deviation (recorded after the run, 2026-11-07)

The frozen table above specifies `micro 2 / accum 8 / mem-frac 0.92` (the baseline's exact recipe). The run was
executed with **`micro 1 / accum 16 / mem-frac 0.38`**: the shared 5090 had ~13.9 GB free, and `micro 2` with
gradient checkpointing does not fit. The **effective batch is unchanged (16 samples/step)**, but the gradient
noise profile differs, so the executed experiment is a **two-variable change** (class mix + microbatch), not the
single-variable one this document froze.

Consequences, stated before the numbers are read:

- The pass/fail decision is still valid as a test of *this* arm — it failed both clauses by wide margins and the
  cross-split sign flip is not a microbatch artefact one would expect — but a *positive* result would have needed
  a matched-recipe replication before being claimed.
- Any future arm must either run at `micro 2` on a free card or re-run the baseline at `micro 1` so the comparison
  is matched.
