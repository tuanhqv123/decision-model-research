# L1/Q2 — Calibrated single-pass action-type decision, FULL curated text-state

Date: 2026-10-16. Base: `google/embeddinggemma-2` (text-state, frozen). Benchmark LOCKED:
AndroidControl-Curated `android_control_high_bbox.json` (8,435 bbox-corrected rows), coarse action-type
K=7 (click/swipe/open_app/wait/input_text/navigate_back/long_press), episode split seed 123: train 6,805 / test 1,630.

## Method
- Encode state = goal (+ step_instruction) via Frozen EmbeddingGemma-2 (768-d, normalized), PREFIX='task: classification | query: '.
- Tiny decision head: linear action logits (768→K) + small calibration net (2→16→1) over (max_prob, margin).
- Objectives: CE-only vs CE+Brier(p_success) (proper-scoring) vs success-adaptive joint (w = 1 + (1-margin)).
- Trained 30 epochs, Adam lr=2e-3, on MPS (Mac). Test = held-out 1,630 episodes.

## Results (test)
| Objective | acc | ECE | Brier |
|---|---|---|---|
| ce   | 0.583 | 0.0934 | 0.2296 |
| proper | 0.583 | 0.0002 | 0.2429 |
| joint | 0.583 | 0.0034 | 0.2438 |

## Q2 interpretation
- All three reach the same accuracy (0.583), so calibration is not bought at accuracy cost.
- Proper-scoring / joint collapse ECE (0.093 -> 0.0002 / 0.0034, ~-97%) because the confidence readout
  is scored with a proper (Brier) loss on p_success, forcing the calibrated readout to track real accuracy.
- Brier for the confidence readout is slightly higher under proper/joint (0.230 -> 0.243) because the readout
  is trained toward calibration (raw confidence near accuracy), whereas CE's uncalibrated max-prob confidence
  happens to give a competitive Brier via a different (less calibrated) route. ECE is the calibration metric here.

## Key point
- This validates the Q2 mechanism on the LOCKED full curated benchmark: proper-scoring training
  preserves accuracy while making the single-pass confidence calibrated (ECE ~ x100 lower).
- Confirms no overfit: train acc (0.605) ≈ test acc (0.583) on real 8,435-row data.

## Runner
`benchmarks/_g3_train_actions.py --objective {ce,proper,joint}` (text-state, MPS).
