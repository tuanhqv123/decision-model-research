# Hugging Face release: DMA-2B

Published model: https://huggingface.co/tuanhqv123/DMA-2B (public), Hub commit `a7c8adc5`.

* Weights: the final checkpoint of the training run on the full AndroidControl training split (optimizer step 7,000).
  `model.safetensors` SHA-256 `284229c8a5a56b8a692b7d8b49b2c27097358dfb7748b3097299cd735593f209`, checked against the
  Hub's LFS hash after upload. Base remote code, tokenizer, config and LICENSE are unchanged copies of
  `LiquidAI/d1-2B@cfe5760`; the base model card is replaced by `README.md`.
* Files in this folder are the card files as uploaded; `ac_common.py` on the Hub is `benchmarks/train/ac_common.py`.
* License (LFM Open License v1.0, Section 4): license copied, changed files listed in `NOTICE.md`, attribution kept;
  Section 7: Liquid AI names are used only to describe the origin, and the model name avoids "LFM" and "Liquid".
* Verification: `example.py` was run on the GPU host with the released weights. It reproduces the decision recorded in
  the AndroidControl evaluation for test episode 27, step 1: `click`, p = 0.999, element [13], p = 0.987; this is the
  human's action.

## Demo Space

https://huggingface.co/spaces/tuanhqv123/DMA-2B-demo is a **static** Space (`release/hf-DMA-2B-demo-static/`). It shows
three AndroidControl test episodes in the AC-Low setting. Every step was run through the released weights on a GPU, and
each step is marked with whether the decision matches the human's action.

The episodes were chosen by a fixed rule: the first two episodes of 4 to 6 steps (by episode id) that are fully right and
use different apps, and the first episode with exactly one wrong step. The data are built by the script
`build_space.py` in that folder.
