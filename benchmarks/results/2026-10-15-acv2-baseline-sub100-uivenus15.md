# AC-v2 baseline subset-100 — UI-Venus-1.5-8B (Step 0 anchor)

> **Source:** Kaggle kernel `trnanh140999/ac-v2-sota-baseline` (subset-100, COMPLETE 2026-10-15).
> Protocol: GUI-Libra native-agent (goal only, T=greedy). Scoring: eval_official.py port
> (a11y-tree element grounding + equivalences). Backend: HF (not llama.cpp).

## Numbers (n=100, leaderboard subset of the 398-step AC-v2)
| metric | value |
|---|---|
| step accuracy (Pass@1) | **0.950** (95/100) |
| grounding accuracy (spatial steps, n=96) | 0.000 |
| ECE (conf vs correct, scalar-bin) | **0.0426** |
| Brier (scalar) | 0.0501 |
| latency median / p95 (ms) | 40893 / 42962 |
| coverage–risk | t>=0.3/0.5/0.7: cov=1.00 risk=0.050 ; t>=0.9: cov=0.58 risk=0.069 |

## Why this matters for the paper (claim C2)
- The frontier baseline (UI-Venus-1.5-8B) is **confident and already well-calibrated** (ECE 0.043)
  and hits **0.95 step accuracy** on this subset — but at **~41 s per step**.
- At ~41 s and 8B weights it **cannot** run on-device within a **≤100 ms / risk ≤10%** budget
  (paper claim C2: "big LLM can't meet the risk budget on-device").
- **Grounding 0.000 (n=96)** is a known red flag: the model selects the right action *type*
  but does not land on the exact element/point (a11y grounding). Worth a scorer inspection.

## Compare against (our Step 3 single-pass decision model)
- Target: comparable step accuracy at **single-pass ms latency**, with a **calibrated p(succ)**
  head we gate on (act/ask/refuse). The baseline's calibrated confidence is *not* produced by a
  dedicated p(succ) head — it's a generative-model conf proxy.
- Notes: `benchmarks/run_acv2.py`, `benchmarks/calib_metrics.py`; data `benchmarks/data/acv2/`.

---
> **RETRACTED (2026-10-28): SCORER BUG.** The `score_row` port applied the
> key-subset fallback to SPATIAL steps (click/long_press/type_text), so any
> click with the right action TYPE counted as correct regardless of where the
> predicted point landed. The "step accuracy" above is therefore TYPE-ONLY
> accuracy, not the official full-step number (grounding was 0.000 in that run).
> The scorer was fixed and differentially validated against the original
> `eval_official.py` (312/312 agreement, 0 mismatches, 2026-10-28). This file's
> accuracy, ECE, and coverage-risk numbers are invalid; the latency numbers
> stand. Do not cite this file as an anchor result.
