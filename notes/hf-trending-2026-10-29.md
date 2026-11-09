# HF Hub survey, 2026-10-29 (Hub API: trending, "flint", GUI/computer-use orgs). Model-card numbers are self-reported.

## What is trending (trendingScore, top of the list)
DeepSeek-V4.1-Flash-1017 (multimodal) · Nanbeige4.3-33B · **RecursiveSI/flint** · Spark-X2.6-4B · LiquidAI/LFM3-1.7B ·
MiniMax-M3.1-Flash · Qwen3.8-27B (7.5M downloads) · Nemotron-3.5-Bolt-E4B-4GB (on-device) · LFM3-Omni-2B · baidu Unlimited-OCR-2.

## Flint (RecursiveSI, 2026-10-21, Apache-2.0): NOT a GUI model
39.8B MoE, ~2.5B active, text-only, a coding agent (card: SWE-bench Verified 74.2%, Terminal-Bench 2.1 54.9%).
Architecture trend worth noting: 35 of 49 layers sliding-window (513 tokens), 14 full-attention layers with no positional
encoding, 256 experts / 6 active + 1 shared, per-head QK-norm + attention sinks. Same direction as LFM3 (conv + SWA + few global).

## GUI / computer-use lane (organisations checked: inclusionAI, Tongyi-MAI, mPLUG, ByteDance-Seed, stepfun, Hcompany, xlangai, zai-org, microsoft, Qwen)
- **Holo4** (H Company, 2026-09-25): 27B dense (Qwen3.8 base, CC-BY-NC), 35B-A3B MoE (Qwen3.6 base, Apache-2.0), Holotron4-30B-A3B.
  Card: OSWorld 85.2% (27B). Computer-use, web/desktop/mobile. No AndroidControl number on the card.
- **MAI-Gym / MAI-Gym-9B** (Tongyi-MAI, 2026-10-16, arXiv 2610.20939): open platform for GUI-agent eval + ONLINE RL on mobile/computer/web,
  up to 20,000 concurrent environments, GRPO/DAPO/GSPO/PPO via Slime. Framework code "coming soon" (not released yet).
- Small GUI models seen: UI-TARS-2B-SFT variants, Qwen3-VL-4B GUI fine-tunes (community). Nothing new at 2B that publishes AC-v2.

## Embedding / retrieval
- google/embeddinggemma-2 (3,005 likes, 308k downloads): multimodal embedding, 740M, 768-d, Apache-2.0.
- openbmb/Trident-Qwen3VL-2B (2026-10-12): feature-extraction on a Qwen3-VL-2B backbone (not read in detail).

## What it means for us (see chat 2026-10-29)
1. Frontier GUI agents are 27B+ generative models trained with online RL; small on-device 1–4B is a different lane.
2. Qwen3-VL-2B zero-shot has AC-v2 Ground 59.6 > our fine-tuned d1-2B (56.5): LFM3-VL's vision side is the weaker part for GUI.
3. Retrieval for memory could use EmbeddingGemma 2 / Trident (test retrieval precision first; TF-IDF memory cost -6 points).
4. MAI-Gym is the likely route to online RL / AndroidWorld once its code is out.
