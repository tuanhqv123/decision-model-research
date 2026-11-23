---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: LiquidAI/d1-2B
base_model_relation: finetune
language:
  - en
tags:
  - decision-model
  - single-pass
  - gui-agent
  - mobile-agent
  - android
  - androidcontrol
  - custom_code
model-index:
  - name: DMA-2B-seed1
    results:
      - task:
          type: gui-action-selection
          name: AndroidControl-Low (step success rate, InfiGUI-R1 scorer)
        dataset:
          type: androidcontrol
          name: AndroidControl test, low-level instructions (8,444 steps)
        metrics:
          - type: step_success_rate
            name: SR
            value: 90.94
      - task:
          type: gui-action-selection
          name: AndroidControl-High (step success rate, InfiGUI-R1 scorer)
        dataset:
          type: androidcontrol
          name: AndroidControl test, high-level goals (8,444 steps)
        metrics:
          - type: step_success_rate
            name: SR
            value: 73.46
---

# DMA-2B, training seed 1

This is **training seed 1** of [DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B). The released DMA-2B is seed 0.
All seeds use the same data, recipe and checkpoint step (7,000); only the training seed, and with it the order of the
training data, differs. This repository lets anyone reproduce the seed table and check that the released model is a
typical run of the recipe.

For what the model does, its inputs and outputs, usage, training data and limitations, read the
[DMA-2B card](https://huggingface.co/tuanhqv123/DMA-2B). Everything there applies here; only the weights differ.

## Results on AndroidControl

Step success rate (%) on the AndroidControl test set (8,444 steps, 1,543 episodes), InfiGUI-R1 scorer, AndroidControl's
own action space (no "done"). ESR: episodes in which every step is correct.

| Seed | Low SR | High SR | High ESR |
|---|---|---|---|
| 0 (released as DMA-2B) | 90.50 | 74.21 | 30.65 |
| 1 **(this repo)** | 90.94 | 73.46 | 29.55 |
| 2 | 90.63 | 74.24 | 31.89 |
| **Mean ± s.d.** | **90.69 ± 0.23** | **73.97 ± 0.44** | **30.70 ± 1.17** |

This seed: Low 7,679/8,444 steps correct, High 6,203/8,444. Per-step predictions and
scores of every seed are in the dataset [DMA-2B-results](https://huggingface.co/datasets/tuanhqv123/DMA-2B-results).

## How to use

Exactly as DMA-2B, with this repository's id:

```bash
pip install "transformers>=5" torch torchvision pillow
git clone https://huggingface.co/tuanhqv123/DMA-2B-seed1 && cd DMA-2B-seed1
python example.py
```

## Weights

`model.safetensors` is the step-7,000 checkpoint of seed 1, sha256 `571d3b9a37c8bf4f04be6651b0eebb33b9cbad470a139c17fa446585b7e9c3a0`. Optimizer: 8-bit AdamW,
learning rate 1e-5 with 30 warm-up steps and cosine decay, batches of 16 rows, one RTX 4090.

## License

LFM Open License v1.0, inherited from the base model `LiquidAI/d1-2B`; see [LICENSE](./LICENSE) and
[NOTICE.md](./NOTICE.md).
