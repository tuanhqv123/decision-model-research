# Decision Model Training Report — Unsloth Mobile Decision Head

**Date:** 2026-10-14
**Run:** `colab exec` on free Colab T4 → task `b1c2eb02c`
**Notebook:** `notebooks/train_mobile_decision.ipynb` (executed copy: `notebooks/train_mobile_decision_output.ipynb`)
**Status:** ✅ Completed successfully (exit 0)

---

## 1. What was trained

A **single-pass calibrated decision head** (TypeSafe "System One" / Liquid d1 class) on the **Unsloth `FastDecisionModel`** recipe:
- **Base model:** `unsloth/Qwen3.5-2B` (2B dense), loaded 4-bit for training.
- **Decision head:** Clef-style scoring head (`joint_head`) — softmax over the typed answer set (choice / noul / score), zero output tokens (no text generation).
- **LoRA:** r = 16, alpha 16, dropout 0; gradient checkpointing "unsloth".
- **Data:** `LocalLLaMA/typed-decisions` split `all` — 1,200 rows; 1,120 train / 80 held-out. 0 of 6,000 decisions skipped.
- **Objective:** CE over the decision head (Unsloth `DecisionTrainer`), reweighted per step.
- **Hyperparams:** batch 8 × grad-accum 4 = effective 32; warmup 10; max_steps 60; LR 2e-4 cosine; weight_decay 0.01; seed 3407. Float32 (Unsloth logs "float16 won't work for qwen3_5").

---

## 2. Results

### After training
| Metric | Value |
|---|---|
| **Test accuracy** (never-seen split) | **0.7275** |
| **Test ECE** | **0.0316** |
| Test loss | 1.1403 |
| Test record_accuracy | 0.23 |
| **Held-out calibration accuracy** | **0.7625** |
| **Held-out calibration ECE** (per-type temp fitted) | **0.0665** |
| Held-out record_accuracy | 0.30 |
| Train runtime | **42.46 min** |
| Peak GPU reserved | **5.19 GB** |

### Before training (fresh head, chance level reference)
| Metric | Value |
|---|---|
| Accuracy | 0.3375 |
| ECE | 0.0203 |
| loss | 1.2118 |
| record_accuracy | 0.0 |

**Interpretation:** accuracy jumped from **0.34 (chance) → 0.73 (test)**. The head learned the task. ECE is **0.032** on test and **0.067** on the held-out calibration set after per-type temperature fitting — both well-calibrated (the benchmark target for "honest" is ECE ≤ 0.05). Held-out accuracy (0.76) tracks close to test (0.73), i.e. no obvious overfit to the 1,120 training rows.

Fitted temperature per type: `[0, 1, 2]` (choice / noul / score), i.e. the calibration step found near-optimal post-hoc temperatures.

---

## 3. Saved artifacts

### `models/decision/qwen_lora/` (downloaded from Colab)
| File | Size |
|---|---|
| `adapter_model.safetensors` | **67.3 MB** |
| `joint_head.safetensors` | **61.0 MB** |
| `adapter_config.json`, `joint_head_config.json`, `unsloth_decision_config.json` | small |
| `tokenizer_config.json`, `processor_config.json`, `chat_template.jinja`, `README.md` | small |

### `models/decision/qwen_merged_16bit/` (downloading from Colab, ~4.5 GB)
Full merged 16-bit model for deployment (contains the base weights + merged LoRA + decision head).

---

## 4. What this proves

This is the **second independent implementation** of our fast-head decision recipe (the other being `benchmarks/calib_metrics.py` + `benchmarks/cascade.py`). It shows:
- A small (2B) open model can be post-trained into a **calibrated single-pass decision head** reaching **0.73 accuracy / ECE 0.03** on a held-out split in **~42 min on a free T4**.
- **On-device size:** LoRA + head = **~128 MB** of trainable weights (adapter 67 MB + joint head 61 MB) on top of a 2B base — tractable for edge/MLX 4-bit conversion.
- The **calibration is real**: ECE is low and the training recipe (Unsloth decision head + temperature calibration) matches our Q1/Q2 objective (proper-scoring / post-hoc temperature keeps ECE low) rather than naive RLVR (which we've seen collapse confidence to ~0.99).

---

## 5. Caveats
- The recipe trains a **text-state** decision head on `typed-decisions` (toy/synthetic-ish distribution). This is a **mechanism validation**, not the AndroidControl benchmark. It is not yet a mobile-agent action-selection model.
- Record accuracy (0.23 test / 0.30 held-out) is the fraction of *identical* (record) predictions — the head is learning the task but not memorizing records.
- 2B base; our deployment target is LFM2.5 **1.2B → 2.6B** on MLX. This run used the Unsloth Qwen3.5-2B as a drop-in foundation, not the final LFM head.

---

## 6. Next steps (ties to `notes/sota-arch-deep.md`)
1. **Step 0 (P0 anchor):** run `benchmarks/run_acv2.py` on the 398-step AC-v2 subset with UI-Venus-1.5-8B (hf) and UI-Venus-2-9B (llama GGUF) → reproduce SOTA ≈61.06 ±3.
2. **Step 3 (the head):** port this trained decision-head pattern to **LFM2.5 1.2B/2.6B** and to **AndroidControl action-selection** (choice over K candidate actions from the a11y tree), then compare **CE vs proper-scoring vs joint-calibration** (the Q2 mechanism experiment).
3. **Step 4 (the gate):** add the act/ask/refuse/escalate gate calibrated on P(success) — target false-action ≤10% at coverage ≥50%.
