---
title: DMA-2B Android Decision Model
emoji: 📱
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
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

Step through three AndroidWorld tasks solved by [DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B) and see, at
each step, the probability the model gave to every action and to the on-screen elements. The recorded answers come
from our evaluation run on a GPU; the button re-runs the same forward pass on this Space's CPU (about one minute).

The app code is Apache-2.0. The model is a fine-tune of LiquidAI/d1-2B under the LFM Open License v1.0 (see the
model card). The screenshots were captured from the AndroidWorld emulator (google-research/android_world).
