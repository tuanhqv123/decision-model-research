# QUAN-Transcribe / EmbeddingGemma 2 — lessons for our decision model

> Why I studied this: QUAN-Transcribe-1.0 is a **single-pass, non-autoregressive** model (one forward pass, zero autoregressive decode) that is simultaneously **strong** (WER 6.87%, beats Gemini-Flash & MAI-Transcribe) and **fast** (3,713× real-time on H100), at only **356M params**. It is the closest real-world validation of our own decision-model philosophy for a different modality. EmbeddingGemma 2 is its encoder backbone and is itself a low-latency multimodal embedding model. Goal: extract what transfers to our calibrated single-pass decision head on AndroidControl.
> Source: `qnguyen3/QUAN-Transcribe-1.0` model card + `github.com/qnguyen3/QUAN-Transcribe` README [quan2026transcribe]; `google/embeddinggemma-2` model card [google2026embeddinggemma2].

## What QUAN-Transcribe-1.0 is
- Vietnamese ASR. **Encoder** = EmbeddingGemma 2 audio encoder (USM-style conformer), **only top 2 of 12 layers fine-tuned**. **Reader** = 6-layer transformer (d=768) trained with **CTC**. Output = 14,917 Gemma tokenizer tokens (~1 per Vietnamese syllable).
- **One forward pass per utterance; no autoregressive decoder.** Streaming supported (reader attention covers only current 0.48 s chunk + earlier; encoder attention chunked with left context only; reader conv is causal).
- Greedy decoding default; optional 4-gram KenLM + CTC beam search adds accuracy (~15 ms/utterance on CPU).
- Runs bf16 on CUDA, fp32 on MPS/CPU. Base encoder is **gated** (Gemma terms + HF login).

## Transferable lessons (map to our decision head / Step 3)
| # | Lesson from QUAN / EmbeddingGemma | Transfer to our decision model |
|---|---|---|
| L1 | **Strong pretrained encoder + fine-tune only top layers** (2/12) — huge efficiency win. | Use our base model (LFM2.5) as the encoder and fine-tune only top layers / a light head; freeze most of the backbone. Cuts train cost; keeps capability. |
| L2 | **Light reader head** (6-layer d=768) over encoder features does all the task-specific work. | Our choice-over-K + P(success) head **is** the reader — keep it light over a mostly-frozen encoder. |
| L3 | **Learned mix of layer outputs** (not just last layer) into the reader. | Aggregate several encoder-layer features (learned weighted sum) as the latent for the decision head → richer, more robust latent. |
| L4 | **Native multimodality, single shared embedding space** (text+image+audio → 768-d) + **selective encoder loading** (load only needed modality). | For action selection, encode goal (text) + screenshot (vision) into ONE representation. Use a vision-capable backbone / selective modality loading for on-device memory. |
| L5 | **Matryoshka (MRL) truncation** 768→256 near-lossless (6× storage cut). | Truncate latent dims for on-device; also lets us trade head capacity vs. latency/size. |
| L6 | **bf16, NOT fp16** (EmbeddingGemma returns NaN in fp16). | Use bf16 for our calibrated head — precision bugs silently corrupt probabilities/ECE. |
| L7 | **Classification (CTC/tokens) single-pass beats generation on speed** — no autoregressive decode. | Validates our **zero-output-token** decision philosophy: classify over K choices, never generate. |
| L8 | **Optional lightweight head for extra accuracy** (n-gram LM) on the fast path. | Our **act/ask/refuse gate** = the analogous "accuracy-boost" path: greedy (argmax) fast when confident, ask/refuse when uncertain. |
| L9 | **Streaming via chunked + causal attention** → low latency. | For on-device, process screenshot state with chunked attention so first-decision latency is bounded. |
| L10 | **Gated / Gemma-terms licensing** on the encoder side. | Verify our base model license (LFM Open License v1.0) before releasing fine-tuned weights; note modality-encoder license separately. |

## Calibration angle (the part QUAN does NOT do, and why we win there)
QUAN is optimized for WER (accuracy), not calibration. It is a strong existence proof of "single-pass + fast + small can beat big generative models" (L7) — **exactly our headline mechanism story** (C2: on-device speed + risk control). Our differentiator: we ADD a calibrated P(success) head + act/ask/refuse gate (Q2/Q3), which QUAN lacks. So we cite QUAN as a **sibling fast single-pass system** and build on its encoder/reader lessons to make our head strong, while owning the calibration axis.

## Concrete changes to make in `models/decision/head.py` + `notes/decision-head-step3.md`
1. **Layer-mix latent (L3):** read several top encoder-layer states (learned weighted sum) instead of only the last layer → feed to action + P(success) heads.
2. **Freeze encoder, train head (+ optionally top 1-2 layers) (L1/L2):** LoRA or full freeze; train the reader/decision head. Matches QUAN's top-2-finetune pattern.
3. **bf16 (L6):** set `torch_dtype=bfloat16` for all forward passes; never fp16 for calibration metrics.
4. **Gate = L8:** keep greedy argmax fast path; P(success) head triggers act/ask/refuse when prob < threshold.
5. **Latent truncation (L5):** optional MRL-style dim reduction for on-device.
6. **Single-pass framing (L7):** document that our head is classification over K, never generation — same as QUAN's CTC.

> **Skeptical flag:** QUAN's "3,713× RT", WER table, and training hours are **vendor self-scored** (no independent replication found). Treat as a design reference, not an evidence claim. Its speed claim also relies on a tuned H100 + bf16, not an edge device.
