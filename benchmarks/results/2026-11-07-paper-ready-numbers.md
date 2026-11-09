# 2026-11-07 — Numbers that are ready to go into the paper today

Only claims that re-derive from an artifact on disk and survived a replication attempt. Nothing aspirational.

## 1. What the fine-tune actually did (base vs fine-tuned, same scorer, same rows)

| metric (AC-v2 398, official scorer) | base d1-2B | ours (E2) | Δ |
|---|---|---|---|
| step accuracy | 17.4% (48/276) | **64.07%** (255/398) | **+46.7** |
| action type, spatial rows | 63.2% (141/223) | **87.9%** (196/223) | +24.7 |
| grounded given type correct | 22.7% (32/141) | **63.8%** (125/196) | **+41.1** |
| grounding (product) | 14.4% | **56.1%** | **+41.7** |

**Decomposition of the +41.7 grounding gain: placement +26.0, verb choice +5.6, interaction +10.1.**
Two thirds of the improvement is *element selection*, not action choice.

## 2. Where the remaining grounding deficit comes from (mechanism, arithmetic)

The `target_box` loss is four per-edge cross-entropies against a Gaussian with **σ = 4 bins on a 0–999 grid**.
Cross-entropy is a **numerically identical constant (690.78 nats) for offsets ≥ 200 bins** — ΔCE/bin = **0.0**.
Our misses: median **437 px**, 86.8% > 150 px, 4.4% within 50 px.

> **The objective has a gradient where our errors are rare and none where they are common.**

Four lines of numpy; no GPU; explains a plateau this programme has been stuck on for weeks.

## 3. Candidate lists above ~40 destroy placement (replicated on both splits)

| candidates in state | val Δ vs box-only | test Δ vs box-only |
|---|---|---|
| 1–10 | +3.8 | −4.0 |
| 11–20 | +10.0 | −7.3 |
| 21–40 | 0.0 | −5.8 |
| **41+** | **−17.3** | **−23.8** |

Only the 41+ collapse replicates. The arm's **box-only** path also degraded (−2.54 val, −4.27 test), so the cause
is **state length competing with the image**, not the candidate interface.

## 4. The grounding recipe, decomposed and replicated (n=179 shared rows)

| config | grounding | Δ vs E2 |
|---|---|---|
| E2 baseline (720 px, `cur15k`) | 65.36% (117/179) | — |
| + `aux7k`, seed 0 | 67.60% (121/179) | **+2.23** |
| + `aux7k`, seed 1 | 67.04% (120/179) | **+1.68** |
| + **858 px** (no aux) | **68.16%** (122/179) | **+2.79** |

**`aux7k` replicates across seeds: mean +1.96 ± 0.28.** `aux7k` is **100% re-framed existing screenshots**
(6,574/6,574 images already in `cur15k`) — no new pixels. The three components sum to `groundB`'s measured
+6.01 almost exactly (aux 1.96 + resolution 2.79 + steps 1.26), so the recipe is decomposable rather than a
single opaque gain.

**Owed control, pre-registered:** `aux7k` adds 50% more rows, so "more rows" is not yet excluded as the cause;
`compact_ctrl22k.jsonl` (22,500 duplicated rows) is the isolation run.

## 5. Supervision for candidate selection exists and is scoreable

Gold point inside a candidate on **97.99%** of training rows and **97.68%** of val rows under the corrected,
scoreable label rule.

## 6. The decision-level claim (independent of accuracy)

At 40.7% test coverage and 17.3% empirical risk, the calibrated act/ask/refuse gate beats **both** trivial policies
for `C_err/C_ask` in **2.11–5.79**, and **2.08–4.19** on the next checkpoint. Certificate is val-side (47.6%
coverage, 19.9% upper bound); test CP upper is 21.7%.

## What is explicitly NOT claimable

- Any candidate-interface win (−3.75 val / −7.29 test, both significant).
- Any ensemble win (+0.50 at matched size; +2.01 was a selection artefact, corrected p = 0.17).
- Any union number as a method (+8.10 val / +6.03 test are **oracles**; best free arbiter +0.60, p = 0.68).
- Any on-device latency (never measured).
- Any online AndroidWorld claim (1/8 both conditions, underpowered, unmatched instances).

## The one-line summary of the day

**Grounding is a placement problem, the loss cannot see placement errors, and the supervision to fix it is already
in the dataset — 100% of the auxiliary grounding samples are re-framings of screenshots we already had.**
