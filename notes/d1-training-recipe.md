# d1 / decision-model training recipe, and why your frozen-encoder + linear head collapsed

Author: evidence-gathering subagent. All claims are sourced; vendor claims are labeled.
Purpose: answer (1) real d1/d1-omni recipe, (2) what single-pass decision-head training + calibration looks like,
(3) why a frozen embedding encoder + linear head collapsed to the majority prior, (4) a legitimate recipe for our
GUI-action task.

---

## Coverage Status

- [x] Liquid d1-3B training method — read verbatim from the vendor blog + model card + shipped code (d1-3B).
- [x] d1-omni-600M training method — read verbatim (blog + model card + modeling code).
- [x] Single-pass decision-head mechanics + per-type temperature calibration code — read from `modeling_d1.py`, `config.json`, `runner.py`.
- [x] Linear-probe-vs-fine-tune / shortcut / majority-baseline literature — multiple primary papers + explainer s.
- [ ] d1's exact training data size and per-step hyperparameters (LR, epochs, LoRA rank, batch) — `BLOCKED`, not disclosed publicly. Vendor only says "fine-tuned checkpoints with different random seeds and data mixtures before merging."
- [x] GUI-action post-training recipes: GUI-Libra (read paper §), UI-Venus (abstract), QUAN-Transcribe (recipe read).
- NOTE: `semantic scholar` / `openalex` and all web_search providers were rate-limited or key-less; I routed around this by
  fetching primary pages/APIs directly (arXiv, Hugging Face API, Liquid blog). Evidence below is primary-source.

---

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | Liquid AI blog, "Open d1" | https://www.liquid.ai/blog/open-d1 | d1-3B is a **full fine-tune** (post-train) of LFM2.5-VL-3B; recipe = weight-average base → multi-seed fine-tune → merge; "training on long inputs, shuffling answer options, and fixing shortcuts in the data made a bigger difference than more advanced techniques." d1-omni = fine-tune encoder → staged audio/vision (adapters+LoRA, frozen vision tower) → full fine-tune + merge + weight-average. | self-reported (vendor) | medium (vendor, but concrete/technical) |
| 2 | HF model card, LiquidAI/d1-3B | https://huggingface.co/LiquidAI/d1-3B | "3B decision model built on LFM2.5-VL-3B… returns calibrated, typed answers in **one forward pass with zero output tokens**." 3.12B params; SigLIP2 NaFlex 400M vision; ctx 32,768; Decision Index 0.2.1 = 48.57. | self-reported (vendor) | medium |
| 3 | HF model card, LiquidAI/d1-omni-600M | https://huggingface.co/LiquidAI/d1-omni-600M | "587M decision model built on LFM2.5-Encoder-350M… zero output tokens." Text answers "calibrated with per-type temperatures stored in `config.json`; image and audio answers are the model's softmax as trained." | self-reported (vendor) | medium |
| 4 | d1-omni-600M `config.json` | https://huggingface.co/LiquidAI/d1-omni-600M/raw/main/config.json | Per-type temperatures: `choice:2=1.7465`, `choice:3-5=1.3999`, `choice:6-10=1.1751`, `choice:11+=1.3725`, `noul:2=1.6663`, `score:3-5=1.7301`, `score:6-10=1.0`, defaults `choice/score/noul=1.0`. Also `head_layers=2`, `projector_hidden_size=2048`. | primary (code) | high |
| 5 | d1-omni-600M `modeling_d1.py` | https://huggingface.co/LiquidAI/d1-omni-600M/raw/main/modeling_d1.py | Trunk encoder + 2-layer `DecisionHead` on marker positions; calibration applied as `z = z / self.config.temperatures.get(temperature_key(q), self.config.temperatures.get(q.type, 1.0))` **only for text** (`calibrate=True`); image/audio `calibrate=False`. `p = softmax(z)`. | primary (code) | high |
| 6 | d1-3B `runner.py` | https://huggingface.co/LiquidAI/d1-3B/raw/main/runner.py | d1-3B reads the answer off **log-softmax at the answer slot** (zero output tokens); reads probabilities at answer position; calibration is an optional object (None by default when loaded via `D1Model`). | primary (code) | high |
| 7 | GUI-Libra, arXiv:2602.22190 | https://arxiv.org/abs/2602.22190 | 81K GUI reasoning data / 170K steps / 9K traj; **action-aware SFT** reweights tokens toward action & grounding (long CoT hurts grounding); conservative GRPO with KL trust region + success-adaptive negative-gradient scaling. 13 action types. +15.6%/+12.2% AndroidWorld, +4.0%/+8.7% Online-Mind2Web (4B/8B). | primary (paper) | high |
| 8 | UI-Venus, arXiv:2508.10833 | https://arxiv.org/abs/2508.10833 | RFT on Qwen2.5-VL with "several hundred thousand high-quality training samples"; 7B/72B get 94.1%/50.8% and 95.3%/61.9% on ScreenSpot-V2/Pro. | primary (paper) | high |
| 9 | QUAN-Transcribe-1.0 model card | https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0 | Built on Google EmbeddingGemma-2 audio encoder (top 2 of 12 layers fine-tuned) → **6-layer transformer reader (d=768) trained with CTC**; ~2,800h Vietnamese; ~3h on 8×H100; 6.87% avg WER, 3,713× real time. "Freeze (or lightly tune) encoder + task-specialized **deep** head." | self-reported (model card) | medium-high |
| 10 | "Probing Classifiers" (Multigrid) | https://multigrid.ai/learn/probing-classifiers | "Report the majority-class baseline alongside this. …a probe that beats chance but not the majority baseline is not evidence of anything." | secondary (explainer) | medium |
| 11 | "Is the Shortcut in the Probe or in the Frozen Encoder?" (Lacuna) | https://lacuna.tiptreesystems.com/hypothesis/is-the-shortcut-in-the-probe-or-in-the-frozen-encoder-7a8a8f641401053f | "Linear probes latch onto easy shortcut cues… while ignoring harder, more reliable ones." | secondary (blog) | medium |
| 12 | Alain & Bengio, "Understanding intermediate layers using linear classifier probes", arXiv:1610.01644 | https://arxiv.org/abs/1610.01644 | Canonical linear-probe setup: freeze net, train one linear layer on its features — it *reads* pre-existing features, does not learn new ones. | primary (paper) | high |
| 13 | Kunz & Kuhlmann, "Classifier Probes May Just Learn from Linear Context Features", COLING 2020 | https://aclanthology.org/2020.coling-main.450.pdf | Probe accuracy is not proof the feature is "there"; probes can pick up shallow linear/context cues. | primary (paper) | high |
| 14 | Guo et al., "On Calibration of Modern Neural Networks", ICML 2017, arXiv:1706.04599 | https://arxiv.org/abs/1706.04599 | Temperature scaling (single-parameter Platt) is surprisingly effective at calibrating deep nets; modern nets are often over-confident / miscalibrated. | primary (paper) | high |
| 15 | Mukhoti et al., "Calibrating Deep Neural Networks using Focal Loss", arXiv:2002.09437 | https://arxiv.org/abs/2002.09437 | Training with focal loss (with temperature-aware term) produces calibrated, less over-confident models. | primary (paper) | high |
| 16 | Fernando & Tsokos, "Dynamically Weighted Balanced Loss…", IEEE TNNLS | https://openalex.org/W3120093105 | Class-imbalanced learning + confidence calibration: naive CE on imbalanced data is biased to the majority class; reweighting helps balance and calibration. | primary (paper) | high |
| 17 | Thulasidasan et al., "On Mixup Training: Improved Calibration and Predictive Uncertainty", arXiv:1905.11001 | https://arxiv.org/abs/1905.11001 | Mixup improves calibration / reduces over-confidence. | primary (paper) | high |
| 18 | d1-3B `config.json` | https://huggingface.co/LiquidAI/d1-3B/raw/main/config.json | d1-3B has **no** `temperatures` dict (unlike d1-omni); `model_type=lfm2_vl`, full 30-layer text stack + vision projector. So d1-3B's "calibrated" claim is not from an in-config temperature scaler. | primary (code) | high |

---

## Findings

### (a) The real d1 / decision-model training recipe

**It is a FULL fine-tune (post-training) of the backbone — not a frozen-encoder + linear probe.** Evidence:

- HF metadata shows d1-3B carries the `base_model:finetune:LiquidAI/LFM2.5-VL-3B` tag and 3.12B params (= the whole backbone), and its config is `model_type: lfm2_vl` with the full 30-layer hybrid text stack plus vision projector and projector [2][18]. It is not a probe on top of a frozen feature extractor.
- Vendor training recipe for **d1-3B**, verbatim: *"We averaged the weights of LFM2.5-2.6B and the text backbone of LFM2.5-VL-3B to create a better base model. We then fine-tuned checkpoints with different random seeds and data mixtures before merging them again. Training on long inputs, shuffling answer options, and fixing shortcuts in the data made a bigger difference than more advanced techniques."* [1]
- Vendor training recipe for **d1-omni-600M**, verbatim: *"We first fine-tuned LFM2.5-Encoder-350M on decision tasks, then added audio and vision in stages. For audio, we trained a FastConformer encoder with an adapter… then fine-tuned the audio encoder with a frozen text backbone. For vision, we took the encoder from LFM2.5-VL-450M and trained an adapter plus LoRA updates to the backbone. Those updates were active only when the input included images, and the vision encoder stayed frozen. We then fine-tuned the full model, merged the LoRA updates, and averaged the weights with the previous checkpoint to regularize the final model."* [1]

So both variants update the backbone weights. The omni variant uses *stage-wise* training with adapters/LoRA for new modalities (and a frozen vision tower), but it ends with a **full fine-tune + merge + weight averaging** — again not a probe.

**What "single-pass / System One" actually is.** The model consumes the state + named questions once and emits the answer distribution directly from the backbone logits at a marker/answer position; there are **zero generated output tokens** [2][3][6]. Mechanically:
- d1-3B: reads `log-softmax` at the answer slot of a causal decoder (a hybrid "tree" over the shared prompt trunk), then `readout()` maps option logits → probabilities/confidence [`runner.py`] [6].
- d1-omni-600M: `Trunk` encoder → a 2-layer `DecisionHead` over per-question marker positions, and for text (`calibrate=True`) divides the logits by a per-type temperature before softmax [`modeling_d1.py`] [5].

Both are best described as *a task-specialized decision head on a fine-tuned backbone, trained end-to-end with cross-entropy-like supervision over the option distribution*, then calibrated.

**Calibration.** Vendor states text answers for d1-omni are "calibrated with per-type temperatures stored in `config.json`; image and audio answers are the model's **softmax as trained**" [3]. The per-type temperature values (from the shipped config) [4] and the code that applies them only to text [5] are reproduced above. d1-3B, by contrast, has **no** in-config temperature scaler [18] — its "calibrated" description is a vendor training claim (the optional `calibration` object is left `None` when loaded via `D1Model` [6]). **Honest caveat:** the d1 score of 48.57 on Decision Index 0.2.1 and the 82.9/74.1 benchmark means are vendor-reported (official-scorer), no independent replication [1][2][3].

### (b) Why your frozen encoder + linear head collapsed to the majority prior

This is **both** an architecture/capacity problem **and** a task-framing/imbalance problem. Breaking it down:

1. **A linear probe cannot learn features — it can only read what is already linearly separable in frozen features.** Alain & Bengio's linear probes freeze the net and fit one linear layer on its features; they measure *presence*, not *usability*, and learn no new representation [12]. Kunz & Kuhlmann show a probe's accuracy is not proof a feature is genuinely encoded — it can satisfy on shallow linear context cues [13]. With a frozen `google/embeddinggemma-2` (general-purpose multimodal 768-d embedding, image–text matching), the K=7 GUI-action classes are not linearly separable in that embedding space.

2. **Class imbalance + plain CE biases the head toward the majority class.** Your majority ('click', ~58%) means the head can hit ~58% for free. On imbalanced data, CE-trained classifiers are frequency-biased and drift to predicting the dominant label; reweighting is the standard fix [16]. This is precisely why you got exactly the always-predict-majority baseline.

3. **The option-order / shortcut problem (the killer the d1 team explicitly fought).** Liquid AI's #1 lesson was *"shuffling answer options, and fixing shortcuts in the data made a bigger difference than more advanced techniques"* [1]. If options appear in a fixed order and one is the majority, a probe learns the positional/majority shortcut instead of the content. Your linear head had no incentive to look at the screenshot content.

4. **Frozen-encoder + head is not inherently doomed — but a *linear* head on a *generic* encoder is.** The counter-example is QUAN-Transcribe: it freezes (lightly tunes) EmbeddingGemma-2's audio tower and adds a **6-layer Transformer CTC reader** (d=768), trained on ~2,800h — and gets SOTA [9]. That works because the encoder is *domain-aligned* (audio→tokens) and the head is deep and nonlinear. Your setup was the weak case: linear head + generic multimodal embedding + 6.8k imbalanced samples.

> **Inference (mine, not a single source):** the collapse is dominated by the combination of (i) no feature learning from a linear head, (ii) 58% majority prior under CE, and (iii) fixed option order. Any one of these on its own can cause exactly the observed "equals majority baseline" result. The strongest single piece of direct evidence against your exact framing is the majority-baseline check [10] and the shortcut lesson [1][11][13].

### (c) Recommended legitimate recipe to beat the majority baseline on a GUI-action task

Ranked by feasibility for a ~7-way GUI-action decision on a small VLM, all grounded in the sources above.

**1. Stop using a linear probe on a frozen generic embedding. Fine-tune a small VLM/LFM backbone (LoRA or full SFT) with a small decision head.**
- Use a *multimodal* base so the model can *see* the screenshot: e.g. LFM2.5-VL-3B (3.1B, what d1-3B is built from [2]), or Qwen2.5-VL-3B (what UI-Venus and GUI-Libra use [7][8]) to keep GPU needs low. LoRA on a 3B VLM fits a single ~24GB GPU; full SFT is heavier but what the vendor actually did [1]. Even LoRA updates the backbone so features *adapt* to the task (this is the key difference from your frozen probe).
- Add a small task head. Either a d1-style read-off-at-answer-slot head (reads logits over the K options, zero generated tokens) [5][6], or a 1–2 layer `DecisionHead` over a CLS/marker token like d1-omni [5].

**2. Fix the task framing (this is the direct fix for the collapse).**
- **Shuffle option order per sample** during training — the single most important data-side fix per the vendor [1]. It removes the positional/majority shortcut.
- **Balanced / reweighted supervision.** If 'click' is 58%, don't train plain CE. Either resample or use a class/imbalance-aware loss (dynamically weighted balanced loss) [16], or — better for GUI — follow GUI-Libra's **action-aware SFT** which reweights tokens to emphasize the *action* and *grounding* tokens (long CoT reasoning actually *hurts* grounding) [7].

**3. Get enough data.** 6.8k is far below what these models use. Reference scales:
- GUI-Libra: 81K curated reasoning samples / 170K steps / 9K trajectories (built from GUI-Odyssey, AMEX, AndroidControl, AitW, GUIAct, MM-Mind2Web…) [7].
- UI-Venus: "several hundred thousand high-quality training samples" via RFT [8].
- d1-scored datasets alone are ~24k rows (DecisionBench) [2].
Even a narrow task wants ≥ tens of thousands of labeled (state, action, grounding) examples, class-balanced, with heterogeneous screenshots/apps.

**4. Calibrate (so your decision is honest and your probabilities usable).**
- Post-hoc **per-type temperature scaling** on a held-out split, exactly the d1-omni pattern: one `T` per (question-type, #options) group, e.g. choice:2, choice:3-5, choice:6-10, noul:2, score:3-5 [3][4][5]. This is the canonical, cheap, effective method [14].
- Optionally also train with **focal loss** (Mukhoti uses a temperature-aware focal term for calibration) [15] or **mixup** [17] to avoid over-confidence, and monitor **ECE + Brier** alongside accuracy. Note: for calibration to matter you need a proper scoring objective between CE and the calibration step; CE is proper, so post-hoc temperature scaling is the safe route.

**5. Evaluate the right way.**
- Always report the **majority-class baseline** and only claim a real result once accuracy materially exceeds it (and ECE is low) [10]. A probe that merely ties the majority baseline is "not evidence of anything" [10].

**Concrete starting recipe (values not vendor-disclosed for d1 → marked as chosen, not claimed):**
- Base: LFM2.5-VL-3B or Qwen2.5-VL-3B (LoRA rank 16–64 on attention + MLP; lr ~1e-4 for LoRA; full SFT lr ~2e-5); optimizer AdamW; cosine schedule with ~3% warmup; 2–4 epochs; batch = as large as fits; fp16/bf16 + gradient checkpointing. (These are MY standard choices, **not** Liquid's disclosed values — the vendor did not publish hyperparameters.)
- Data: ≥20–50k balanced (state, K-option) samples, options **shuffled**, mixed screenshot + instruction; action tokens **reweighted** per GUI-Libra [7].
- Head: d1-style single-pass readout at the answer slot [5][6]; train the head jointly with the backbone.
- Loss: CE over the K options (proper) [14]; add focal/mixup regularization for calibration [15][17] if over-confident.
- Calibration: per-type temperature scaling on held-out set [3][4][14].
- Eval: accuracy vs. majority baseline, ECE, Brier, per-class recall on the minority actions.

---

## Honesty / risk notes
- d1's exact data size, per-step hyperparameters, and loss formulation are **not publicly disclosed**; the recipe above is the vendor's *qualitative* method plus independent GUI-post-training evidence. I did not fabricate any hyperparameter as vendor-provided.
- d1 benchmark numbers (48.57 Decision Index, 82.9/74.1) are vendor/self-reported; no independent replication found.
- QUAN-Transcribe and UI-Venus/GUI-Libra use different tasks (ASR / GUI navigation) than your K-way decision head, so their data-scale numbers are guidance, not guarantees for your task.

## Sources
1. Liquid AI, "Open d1: Edge decision models for text, vision, and audio" — https://www.liquid.ai/blog/open-d1
2. LiquidAI/d1-3B model card — https://huggingface.co/LiquidAI/d1-3B
3. LiquidAI/d1-omni-600M model card — https://huggingface.co/LiquidAI/d1-omni-600M
4. LiquidAI/d1-omni-600M config.json — https://huggingface.co/LiquidAI/d1-omni-600M/raw/main/config.json
5. LiquidAI/d1-omni-600M modeling_d1.py — https://huggingface.co/LiquidAI/d1-omni-600M/raw/main/modeling_d1.py
6. LiquidAI/d1-3B runner.py — https://huggingface.co/LiquidAI/d1-3B/raw/main/runner.py
7. R. Yang et al., "GUI-Libra…" arXiv:2602.22190 — https://arxiv.org/abs/2602.22190
8. Z. Gu et al., "UI-Venus Technical Report…" arXiv:2508.10833 — https://arxiv.org/abs/2508.10833
9. qnguyen3, "QUAN-Transcribe-1.0" — https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0
10. "Probing Classifiers: Finding What a Layer Knows" (Multigrid) — https://multigrid.ai/learn/probing-classifiers
11. "Is the Shortcut in the Probe or in the Frozen Encoder?" (Lacuna) — https://lacuna.tiptreesystems.com/hypothesis/is-the-shortcut-in-the-probe-or-in-the-frozen-encoder-7a8a8f641401053f
12. G. Alain & Y. Bengio, "Understanding intermediate layers using linear classifier probes", arXiv:1610.01644 — https://arxiv.org/abs/1610.01644
13. J. Kunz & M. Kuhlmann, "Classifier Probes May Just Learn from Linear Context Features", COLING 2020 — https://aclanthology.org/2020.coling-main.450.pdf
14. C. Guo et al., "On Calibration of Modern Neural Networks", ICML 2017, arXiv:1706.04599 — https://arxiv.org/abs/1706.04599
15. J. Mukhoti et al., "Calibrating Deep Neural Networks using Focal Loss", arXiv:2002.09437 — https://arxiv.org/abs/2002.09437
16. K. R. M. Fernando & C. P. Tsokos, "Dynamically Weighted Balanced Loss…", IEEE TNNLS — https://openalex.org/W3120093105
17. S. Thulasidasan et al., "On Mixup Training: Improved Calibration and Predictive Uncertainty", arXiv:1905.11001 — https://arxiv.org/abs/1905.11001
18. LiquidAI/d1-3B config.json — https://huggingface.co/LiquidAI/d1-3B/raw/main/config.json
