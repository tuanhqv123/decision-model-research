# d1-2B two-stage decision protocol — full AC-v2 (398 steps), Kaggle T4

*2026-10-28. Kernel: `trnanh140999/d12b2s-acv2-kaggle` (latest run, TAG `d12b2s-full-v`, modality=vision, THR=0.5, zero-shot — no fine-tuning).*

## Headline numbers (this run: 73/398)

| Metric | Value |
|---|---|
| Step accuracy | **18.34%** (73/398) with p_done≥0.5 → terminate gate |
| Grounding (spatial rows) | 27.0% (n=163) |
| Completion recall (87 gold status) | **10/87 (11.5%)**, mean p_done(status)=0.237, median 0.187 |
| False-completion (311 non-status) | 7/311 (2.25%), mean p_done(non-status)=0.123, median 0.073 |
| ECE (action-prob confidence convention) | **0.362** |
| Brier (action-prob confidence) | 0.280 |
| Mean conf if correct / if wrong | 0.622 / 0.528 (weak separation) |
| Rows with conf=None (box-not-found bug, pre-patch) | 51/398 (12.8%) — scored as misses |
| Wall time | ~1524 s (25 min) on T4 |

**By gold type (this run):** click 43/212 (20.3%) · status 10/87 (11.5%) · scroll 9/37 · open_app 6/27 · navigate_back 3/20 · type_text 1/11 · wait 1/4

**Run-to-run variance (same code, T4 bf16 nondeterminism, compile skipped):** earlier full run (v2) scored **61/398 (15.33%)**; this run (v3) **73/398 (18.34%)**. Treat single-run numbers as ±~3 pts.

## Baselines for comparison

| System | Step acc |
|---|---|
| d1-2B two-stage zero-shot (this) | 18.34% (15.33–18.34 band) |
| d1-2B flat vision zero-shot | 15.83% |
| d1-2B flat a11y zero-shot | 28.39% (different interface) |
| Best trivial (always-status) | 21.86% |
| Qwen3-VL-2B zero-shot (SOTA small-model bar) | 45.48% |
| Mem-W-4B (SOTA break target) | 63.07% |

## The p_done gate analysis (the interesting part)

Zero-shot noul completion head shows a **real but weak signal**: p_done separates status from non-status (mean 0.237 vs 0.123; median 0.187 vs 0.073), but at the principled threshold 0.5 it catches only 10/87 status while mis-firing on 7 non-status.

Threshold sweep **fit on the eval set (analysis only — NOT a deployable claim; would be overfit)**:

| thr | status caught | false-fire | net step acc |
|---|---|---|---|
| 0.05 | 77/87 | 220 (45 was-right) | 23.9% |
| **0.10** | **62/87** | **115 (23 was-right)** | **25.6%** |
| 0.15 | 52/87 | 71 (14 was-right) | 25.4% |
| 0.50 (default) | 10/87 | 7 (0 was-right) | 18.3% |

Even at the eval-fit best (0.10), ~25.6% — still **20 pts below** the small-model bar (45.48). The gate converts ~60 status steps but breaks ~15–25 previously-correct non-status steps.

## Honest conclusions

1. **Zero-shot two-stage does not break any SOTA point.** At the principled 0.5 threshold it is ~18% — below the 21.86% trivial baseline. The "free points" hypothesis is dead at zero-shot.
2. **The completion head has separable signal** (p_done status > non-status, medians 0.187 vs 0.073) — it's miscalibrated/untrained, not blind. This is the Q1 evidence we want: a zero-shot single-pass decision head under-confirms completion on final states.
3. **Training the completion head is now MANDATORY** (was conditional): 1,242 final-state completion rows + intermediate-step negatives are ready locally (`benchmarks/data/curated/android_control_completion.json`).
4. **Action head calibration is poor zero-shot** (ECE 0.36, mean conf 0.528 on wrong answers) — overconfidence on misses; per-type temperature / proper-scoring training (Q2) is the paper's mechanism question, and this run quantifies the "before".

## Bug found + fixed

- `run_d12b_kaggle.py` (pre-patch): in vision mode, when the `box` was not found (`found < 0.5`), `conf_final` was left `None` → `ece_binary` crashed at the end of the run (kernel marked ERROR after DONE_RUN; all 398 rows + JSONL were intact). Fix: `conf_final = conf` always set in the action branch (mirrors local `run_d12b_acv2.py`), plus end-of-run cleanup of `/kaggle/working` (images+json) so the output dataset stays small.

## Artifacts

- JSONL (per-step): pulled to `/tmp/kout3/j/data/results/d12b2s-full-v.jsonl` — copy to `benchmarks/results/d12b2s-full-v-2026-10-28.jsonl` (done in this commit).
- Kernel: `benchmarks/kaggle/run_d12b_kaggle.py` (patched).
- Logs: `kaggle kernels logs trnanh140999/d12b2s-acv2-kaggle`.