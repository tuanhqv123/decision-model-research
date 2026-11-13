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

Step through three AndroidWorld tasks completed by [DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B) and see, at
each step, the probability the model gave to every action and to the on-screen elements. All answers are replayed
from our evaluation run (model on a GPU); `data.json` was built from the recorded inputs and outputs with the
model repository's `ac_common.py` and `example.py`.

The page code is Apache-2.0. The model is a fine-tune of LiquidAI/d1-2B under the LFM Open License v1.0 (see the
model card). The screenshots were captured from the AndroidWorld emulator (google-research/android_world).
