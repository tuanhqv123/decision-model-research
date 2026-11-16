---
title: DMA-2B Android Decision Model
emoji: 📱
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
  - tuanhqv123/DMA-2B
short_description: A 2B model that picks phone actions in one pass
tags:
  - gui-agent
  - android
  - decision-model
---

Step through three AndroidControl test episodes with [DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B) and see, at
each step, the probability the model gave to every action and to the on-screen elements, and whether its decision
matches the human's. Every step was run through the released model on a GPU; `data.json` was built with the model
repository's `ac_common.py` and `example.py`. Episodes were chosen by a fixed rule (the first two 4–6-step episodes that
are fully right, and the first with exactly one wrong step).

The page code is Apache-2.0. The model is a fine-tune of LiquidAI/d1-2B under the LFM Open License v1.0 (see the
model card). Screenshots, instructions and accessibility data come from the AndroidControl test split
(google-research, Li et al., NeurIPS 2024).
