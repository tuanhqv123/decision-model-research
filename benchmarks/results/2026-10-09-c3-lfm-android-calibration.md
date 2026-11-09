# C3 follow-up — Post-hoc temperature scaling on LFM2.5 (AndroidControl, text)

Date: 2026-10-09 · On-device MLX (M1 Pro), text-only LFM2.5, real Apache-2.0 data.

## Setup
- Model: `mlx-community/LFM2.5-1.2B-Instruct-4bit`.
- 300 AndroidControl next-action decisions (k=4). Calibration split 120 / eval 180.
- Temperature T fit on the calibration split by minimizing NLL over the correct option.
- Same inference protocol (bracket-numeric logits) for both uncalibrated and calibrated.

## Result
| | accuracy | ECE (n_bins=5) | Brier | mean confidence |
|---|---|---|---|---|
| **Uncalibrated** (T=1.0) | 0.544 | **0.1272** | 0.6155 | 0.6717 |
| **Calibrated** (T=1.75) | 0.544 | **0.0551** (-57%) | **0.5871** | 0.5497 |

Mean confidence drops to 0.550, closely tracking accuracy 0.544 — the head is no longer overconfident.

## Coverage–risk (calibrated vs uncalibrated)
| threshold | uncal cov / risk | cal cov / risk |
|---|---|---|
| 0.5 | 0.750 / 0.370 | 0.489 / 0.273 |
| 0.7 | 0.422 / 0.263 | 0.206 / 0.108 |
| 0.9 | 0.189 / 0.118 | 0.067 / 0.000 |

At confidence ≥0.7, calibrated risk drops from **0.263 → 0.108** (and to 0 at ≥0.9). Calibration lets the operator act at lower confidence with less risk — the core point of the calibration-reliability framing.

## Interpretation / caveats
- **Post-hoc temperature scaling**, applied at readout time on the frozen head. It is real, runnable on-device, and fixes overconfidence — but it is **not** the stronger claim of a *calibration-aware-*trained* head.
- Accuracy unchanged (0.544) — temperature scaling only re-shapes confidence, it doesn't add accuracy.
- ECE is binning/confidence_fn sensitive (n_bins=5, max_prob) — report both when citing.
- Text-only (no vision); this is the decision-head, not the vision baseline.
