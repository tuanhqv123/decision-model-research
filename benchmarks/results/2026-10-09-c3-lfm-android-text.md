# C3 Baseline — LFM2.5 decision head on raw AndroidControl (text-only)

Date: 2026-10-09 · Task: `Run LFM AndroidControl bench`

## Setup
- **Model:** `mlx-community/LFM2.5-1.2B-Instruct-4bit` (Liquid LFM2.5, d1 lineage), on-device MLX, 4-bit.
- **Data:** raw `smolagents/android-control` (Apache-2.0), 7 test shards = **322 episodes**, **300 decisions** (k=4, seed=0).
- **Task:** next-action selection. State = goal + current step instruction (text only; no screenshot/vision). Correct action = ground-truth action; distractors = real actions from other episodes in the sample.
- **Scoring:** LLM2Jev-style bracket-numeric decoding from next-token logits.

## Results (k=4; chance = 0.25)
| metric | value |
|---|---|
| rows | 300 |
| mean accuracy | **0.573** |
| mean confidence (max_prob) | 0.674 |
| mean n_confidence | 0.565 |
| ECE (n_bins=5) | **0.1003** |
| Brier | 0.5777 |
| Brier decomposition | rel=0.098, res=0.266, unc=0.749 |

## Coverage–risk (selective prediction) — the key curve
| threshold | coverage | accuracy | risk | n_act |
|---|---|---|---|---|
| 0.3 | 0.997 | 0.572 | 0.428 | 299 |
| 0.5 | 0.733 | 0.659 | 0.341 | 220 |
| 0.7 | 0.427 | 0.766 | 0.234 | 128 |
| 0.9 | 0.197 | 0.898 | 0.102 | 59 |

Monotone: raising the confidence threshold trades coverage for accuracy (risk 0.43 → 0.10). This is exactly the calibration-reliability behavior the harness is designed to expose.

## Cascade distribution
- **ask: 233 (77.7%)**
- **act: 67 (22.3%)**

The gate asks a lot because the head is only moderately confident — a real, actionable signal (gate thresholds need tuning for a target-risk operating point).

## Interpretation & caveats
- **Accuracy 0.573 >> chance 0.25** — the text-only LFM2.5 head is a genuine next-action prior, not noise.
- **Overconfident:** mean confidence 0.674 vs accuracy 0.573 → ECE ~0.10. This is the quantity our calibration-aware head targets.
- **Coverage–risk works end-to-end on real data** — the harness is validated (C3 milestone).
- **Text-only:** no screenshot grounding, so this under-measures a visual agent; it's a *decision-head prior*, not the vision baseline. Vision grounding (ScreenSpot / Qwen2.5-VL) is blocked on this box (mlx_lm strips the vision tower).
- `k=4` with distractors from real data; ECE is binning/`confidence_fn` sensitive (n_bins=5, max_prob) — report both when citing.
