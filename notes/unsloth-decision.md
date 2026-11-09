# Unsloth "Train your own Decision Model" (2026-10-13)

Source: https://unsloth.ai/docs/new/train-your-own-decision-model-with-unsloth
(bibkey: `unslothdecision2026`). Fetched + analyzed 2026-10-13.

## What it is

Unsloth shipped first-class **decision-model training** (`FastDecisionModel` +
`DecisionTrainer`, `pip install --upgrade unsloth` or git). Fine-tunes
Qwen3.5-0.8B/2B/4B, Llama-3.2-3B, Gemma 4, Laya with **LoRA + a Clef-style
scoring head**: one forward pass reads input + all questions/options, the head
scores every option for every question, returns `{answer, probabilities}`.
**No text generation.** = exactly the Jev/Laya/Clef architecture class.

## Their numbers (from the docs page)

| model | result | VRAM | time |
|---|---|---|---|
| Qwen3.5-0.8B | typed-decisions 36→73%; BANKING77 7→74%; CLINC150 19→76%; holdout 78% | 4GB | 42 min |
| Qwen3.5-2B | 33→78%; holdout 81% | 8GB | 40 min |
| Qwen3.5-4B | 75-78% after 60 steps (1.7 ep) | ~8GB | ~2h T4 (fp32) / 25 min A100 (2 ep) |
| Llama-3.2-3B | 79% | 4.1GB | 30 min |
| Gemma 4 E4B | 77% | 14.4GB | 49 min |
| **Laya (fine-tuned)** | 77% | 2.5GB | 10 min |

Training recipe: 4-bit base, LoRA r=16, lr 2e-4 (head 1e-4), 2 epochs,
12 mixed sources + `typed-decisions`; test = 3000 rows (2000 typed-decisions
+ 500 BANKING77 + 500 CLINC150), decontaminated.

**Calibration is built in:** `FastDecisionModel.calibrate(model, tokenizer,
eval_items)` on the holdout → "after 60 steps: 80-84% held-out accuracy,
**ece 0.02-0.06**." Calibration is saved with the head (load = calibrated).

## Data

`LocalLLaMA/typed-decisions` (Jev's own org!) — Apache-2.0, tags:
`system-one, calibration, structured-decisions`. Row = `{state, questions,
gold}`; question types: `choice` (option dict), `noul` (yes/no), `score`
(level list). Configs: `all`, `customer_service`, `invoice_processing`,
`security_incidents`, `agent_trace_observability`. Updated 2026-10-13 (40k
downloads). **Raw, downloadable — usable for our own training.**

## Free compute path (user directive)

Official free notebooks (T4, no account paywall):
- Colab: `colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb`
- Kaggle: `Kaggle-Qwen3_5_(4B)-Decision.ipynb` (T4)

Full notebook fetched 2026-10-13; key cells: install (unsloth git +
transformers 5.2.0 + install-kernels), `FastDecisionModel.from_pretrained(
"unsloth/Qwen3.5-4B", max_seq_length=2048, load_in_4bit=True)`, LoRA r=16,
`build_dataset` / `split_holdout(seed=3407)`, `DecisionTrainer`
(bs 8×GA 4, 60 steps, lr 2e-4 cosine, T4→fp32), `calibrate`, `evaluate` on
test split, `predict` API, `save_pretrained` (LoRA+head+calibration) /
`save_pretrained_merged` (16-bit).

**Run via MCP**: the `mcp` server (mcp.appllama.io) — OAuth pending
(2026-10-13); on completion it exposes the Colab control tools; plan:
duplicate the Qwen3.5-4B Decision notebook → run on free T4 → download the
saved `qwen_lora` (adapters+head) + calibration metrics.

## Relevance to our program

1. **More evidence the space moved** (scoop watch): open decision-head
   training is now a pip feature with an official free notebook. Our C1
   (train-time calibration vs post-hoc) must position against this too.
2. **It's a tool for us, not just a competitor**: train the 4B decision head
   on free T4 (2h), export LoRA+head → candidate **fast head** for our T3
   cascade (text AndroidControl protocol) + independent cross-check of our
   MLX LoRA path (todo #14). Two independent implementations of the same
   recipe = stronger paper claim.
3. **Laya** (Cloudflare's decision model) is fine-tunable in 10 min —
   smallest open decision model yet; add to competitive set.
4. Their `agent_trace_observability` config = decision questions over agent
   traces — close to our mobile-agent use; inspect rows when we pull the data.