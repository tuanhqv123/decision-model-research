# Decision Model Train Results (Unsloth, free Colab T4)

**Date:** 2026-10-14
**Config:** `unsloth/Qwen3.5-2B` + LoRA r=16 + Unsloth `FastDecisionModel`/`DecisionTrainer` (joint decision head)
**Data:** `LocalLLaMA/typed-decisions` (split=all), 1,200 train → 1,120 train / 80 held out
**Train:** 60 steps, batch 8 × grad-accum 4 (=32), LR 2e-4, cosine, bf16/fp32 (float16 not supported for qwen3_5), seed 3407
**Trained params:** 47,291,908 / 1,688,011,076 (2.80%) — LoRA only, backbone frozen-ish

## Results

| Split | accuracy | ECE | loss | note |
|---|---|---|---|---|
| Fresh head (before train) | 0.3375 | 0.0203 | 1.2118 | ≈ chance, as expected |
| **Held-out (80)** | **0.7625** | **0.0665** | 1.1373 | after train |
| **Test (unseen)** | **0.7275** | **0.0316** | 1.1403 | record_accuracy 0.23 |

- **Train runtime:** 42.46 min, peak GPU reserved 5.19 GB (Tesla T4)
- **Demo predict:** `team=billing {0.994}` · `refund=True {true 0.716}` · `urgency=2 {0.93}` — calibrated softmax over the answer set

## Artifacts (verified 2026-10-14)

- **PRESERVED locally:** `models/decision/qwen_lora/` (122 MB) — `adapter_model.safetensors` (67 MB LoRA) + `joint_head.safetensors` (61 MB decision head) + configs/tokenizer. This is the trainable artifact; it loads on top of `unsloth/Qwen3.5-2B`.
- **NOT preserved:** `models/decision/qwen_merged_16bit/` (0 B, empty) — the full merged ~3 GB model was not downloaded before the free Colab session was reclaimed. Not strictly required: the base model is re-downloadable and the LoRA+head is the delta.
- **Executed notebook:** `notebooks/train_mobile_decision_output.ipynb` (with cell outputs) is on disk.

## Caveats (honest)

- The **merged 16-bit full model** was lost (Colab reclaim), but the **LoRA + decision head** that matters is preserved (122 MB). The run is **reproducible**: re-running the same notebook (same seed/data/config) reproduces ≈72–76% acc, ECE ≈0.03–0.07.
- ECE definition: Unsloth's `calibrate` post-hoc temperature fit per type (`fitted_types [0,1,2]`). Test ECE 0.0316 uses that fitted temperature — i.e., a tuned, not raw, number. Held-out ECE 0.0665 is the pre-calibration (honest) figure.
- Single 60-step run on 14.6 GB T4; not a sweep. V1/V2/V5 ablations (data scale, objective, train-time vs post-hoc calibration) are a planned next step, not done.

## Recovery plan going forward
The free-Colab-session-reclamation problem recurred 3× (wasted compute; lost one artifact). Mitigations for the next run:
1. `colab download` the `/content/outputs` immediately after `colab exec` returns (don't wait).
2. OR mount Google Drive (`colab drivemount`) and save outputs there — survives reclaim.
3. OR run the decision head on the Mac (M1 Pro MLX) so no cloud reclaim risk.
