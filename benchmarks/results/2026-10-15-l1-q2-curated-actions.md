# L1 / Q2 — Calibrated action-type decision on AndroidControl-Curated (2026-10-15)

## What this is
Locked benchmark swap after the user pushed back that data is abundant and I was
stuck on a micro 232-row a11y choice-over-K subset (real cause of the earlier
"overfit" framing — wrong interface, not the model). The correct, data-adequate
benchmark is **AndroidControl-Curated** (bbox-corrected, `android_control_high_bbox.json`).

## Data reality (verified, resolves the "no data" confusion)
- `android_control_high_bbox.json` = **8,435 steps** with `gt_action` (raw), `instruction` (goal),
  `step_instruction`, `gt_min_bbox` / `gt_max_bbox`, `gt_coordinate`.
- Every curated image (`android_control_images/<ep>/step_<n>.png`) maps to a **local**
  screenshot (`acv2_shots/images/<ep>/screenshot_<n>.png`) — **8,435 / 8,435 (0 missing)**,
  1,543 episodes, 9,987 local PNGs. **No 6.4GB download needed on the Mac.**
- Coarse action-type distribution (K=7): click 5065, swipe 1211, input_text 632,
  open_app 608, wait 567, navigate_back 343, long_press 9.
- Spatial (groundable) rows with coord+bbox: 5,074. Non-spatial: 3,361.

## Benchmark locked
- **Data**: AndroidControl-Curated high split. Split by episode (seed 123): train 6,805 / test 1,630.
- **Task L1 (this run)**: single-pass action-type choice over K=7 classes; state =
  `goal + step_instruction` (text) [+ optional image], frozen `google/embeddinggemma-2`
  encoder + tiny linear head. Zero output tokens.
- **Task L2 (next)**: grounding E_bbox for spatial rows. (Not run yet.)

## Result: Q2 objective comparison (text-state, n=1630 test)

| objective | acc | ECE | Brier |
|---|---|---|---|
| CE | 0.583 | 0.094 | 0.230 |
| proper-scoring (CE + Brier p_succ) | 0.583 | 0.038 | **0.210** |
| joint (success-adaptive) | 0.583 | **0.011** | 0.240 |

### Reading
- Accuracy identical across objectives (all fit CE on the action logits) = 0.583;
  this is a **baseline** (frozen encoder + tiny head, text-state), not SOTA.
- Train acc ≈ test acc (0.605 vs 0.583) → the earlier "overfit" is gone once data is large
  and the interface is right. Confirms the problem was the 232-row subset, not the model.
- **Q2 supported**: joint drops ECE 0.094 → 0.011 (hard/easy success-weighted Brier);
  proper-scoring (CE+Brier) gives the lowest **Brier 0.210** and ECE 0.038. The proper-scoring
  family beats plain CE on calibration. This is a measured confirmation on full data,
  not theory (cf. notes/device-2026-research.md).

## Next (T4 — needs GPU for image encode)
- Multimodal: state = goal+image, encode 8,435 images on T4 (MPS too slow — the multimodal
  run aborted).
- LoRA the encoder / larger head for real accuracy; then grounding E_bbox (L2) + G4 risk-gate
  on-device.

## Files
- builder: `benchmarks/build_curated_actions.py`
- runner: `benchmarks/_g3_train_actions.py`
- data: `benchmarks/data/curated_actions{,_train,_test}.jsonl`
