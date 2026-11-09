# G2/G3 + G4 — text vs multimodal single-pass decision + risk-gate (AndroidControl-Curated)

> **⚠️ RETRACTION (2026-10-16):** the accuracy values in this file (all = 0.5828) are **NOT a real result**.
> They exactly equal the majority-class baseline (always-predict 'click' = 0.5828 on the 1,630-row test).
> The frozen-encoder + single-pass over-K=7 classification head was a **degenerate prior-only solution** —
> it learned the class prior, not the task. The 'action-type classification' interface was the wrong task
> framing (compare: SOTA mobile agents use multi-turn RL over trajectories, e.g. MobileRL). Keep only the
> **calibration (ECE) contrast** (multimodal CE ECE 0.029 < text CE 0.093) and the **G4 risk-gate finding**;
> discard the accuracy values as evidence. See VERIFICATION GATE in AGENTS.md.

Date: 2026-10-16. Base: `google/embeddinggemma-2` (768-d), frozen. Benchmark LOCKED:
AndroidControl-Curated `android_control_high_bbox.json` (8,435 bbox-corrected), coarse action-type
K=7, episode split seed 123 (train 6,805 / test 1,630).

## Encoding
- Text embeddings and multimodal (goal + screenshot) embeddings encoded on Kaggle T4 kernel
  `trnanh140999/embedgemma-curated-t4` (frozen embeddinggemma-2, bf16), cached to
  `benchmarks/data/curated/emb/{S_*_t,S_*_m}.npy`. Both are 768-d, normalized.
- Train-only head (Linear 768->K + calib 2->16->1 over (max_prob, margin)); 60 epochs, Adam 2e-3.
- Runner: `benchmarks/_g5_train_emb.py` (train-only, uses cached embeddings; no encoder/download).

## Results (test 1,630)
| mode | objective | acc | ECE | Brier | conf_mean |
|---|---|---|---|---|---|
| text | ce | 0.5828 | 0.0930 | 0.2298 | 0.553 |
| text | proper | 0.5828 | 0.0386 | 0.2423 | 0.621 |
| text | joint | 0.5828 | 0.0315 | 0.2420 | 0.614 |
| multimodal | ce | 0.5828 | 0.0293 | 0.2388 | 0.567 |
| multimodal | proper | 0.5828 | 0.0332 | 0.2434 | 0.616 |
| multimodal | joint | 0.5828 | 0.0298 | 0.2433 | 0.613 |

## Key findings
1. **Accuracy is identical (0.5828) across text and multimodal.** The screenshot adds ~nothing to
   action-type accuracy on this LOCKED benchmark (consistent with G1 zero-shot: image adds ~nothing).
   The decision head saturates at ~0.58 regardless of modality.
2. **Multimodal CE already calibrated (ECE 0.029) vs text CE (ECE 0.093).** The image input alone
   (even with CE) gives a more calibrated readout on this head — likely because the vision features
   provide a sharper action-discriminative signal that the confidence head can read.
3. **Q2 mechanism holds in both modalities:** proper-scoring/joint reduce ECE relative to the
   within-mode CE baseline, at equal accuracy.

## G4 — risk-gate coverage/risk (T4 kernel `trnanh140999/g4-riskgate-t4`, text-state)
Runner: `benchmarks/_g4_riskgate.py` / notebook `benchmarks/colab/g4_riskgate_t4.ipynb`.
Risk gate: ACT if `p_success >= tau` else ASK/REFUSE. `risk` = error rate among ACTed; `coverage` = fraction ACTed.

| objective | tau | coverage | n_act | risk |
|---|---|---|---|---|
| ce | 0.40 | 1.000 | 1630 | 0.417 |
| ce | 0.45 | 0.981 | 1599 | 0.407 |
| ce | 0.50 | 0.868 | 1415 | 0.364 |
| ce | 0.55 | 0.543 | 885 | 0.283 |
| ce | 0.60 | 0.163 | 266 | 0.207 |
| proper | any | 1.000 -> 0.000 | 1630 -> 0 | (no split) |

**Critical G4 finding:** proper-scoring produced a **near-constant confidence** (range [0.602, 0.632],
mean 0.621) — calibrated on average (ECE 0.039) but **useless for risk-gating** (coverage jumps 1.0 -> 0.0,
no intermediate tradeoff). CE (uncalibrated, range [0.409, 0.649]) **does give a usable coverage/risk
tradeoff** (tau=0.55: cover 54% at risk 0.28; tau=0.60: cover 16% at risk 0.21).

## Interpretation / risk to claim
- **Q2 caveat this session:** proper-scoring collapses the confidence spread about the mean accuracy,
  so while it is well-calibrated in expectation, it loses the *per-example* separation needed to
  abstain on the risky subset. This is a real tradeoff to surface in the paper (calibration <-> risk-gating).
- The identical accuracy across modalities + weak G1 zero-shot means **multimodal is not clearly
  adding value on this benchmark**; the image benefit we can defensibly claim is **calibration (lower ECE
  for CE), not accuracy**.

## Files
- `benchmarks/_g5_train_emb.py`, `benchmarks/_g4_riskgate.py`
- `benchmarks/colab/g4_riskgate_t4.ipynb`
- `benchmarks/results/2026-10-16-g2g3-text-vs-multimodal.csv`
