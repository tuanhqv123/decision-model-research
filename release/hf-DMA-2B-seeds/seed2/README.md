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
  - name: DMA-2B-seed2
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
            value: 90.63
      - task:
          type: gui-action-selection
          name: AndroidControl-High (step success rate, InfiGUI-R1 scorer)
        dataset:
          type: androidcontrol
          name: AndroidControl test, high-level goals (8,444 steps)
        metrics:
          - type: step_success_rate
            name: SR
            value: 74.24
---

# DMA-2B, training seed 2 (revision `seed2`)

This branch holds **training seed 2** of [DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B). The default branch
(`main`) is the released model, seed 0.
All seeds use the same data, recipe and checkpoint step (7,000); only the training seed, and with it the order of the
training data, differs. These branches let anyone reproduce the seed table and check that the released model is a
typical run of the recipe.

For what the model does, its inputs and outputs, usage, training data and limitations, read the
[DMA-2B card](https://huggingface.co/tuanhqv123/DMA-2B) on the `main` branch. Everything there applies here; only the
weights differ.

## Results on AndroidControl

Step success rate (%) on the AndroidControl test set (8,444 steps, 1,543 episodes), InfiGUI-R1 scorer, AndroidControl's
own action space (no "done"). ESR: episodes in which every step is correct.

| Seed | Low SR | High SR | High ESR |
|---|---|---|---|
| 0 (released as DMA-2B) | 90.50 | 74.21 | 30.65 |
| 1 | 90.94 | 73.46 | 29.55 |
| 2 **(this branch)** | 90.63 | 74.24 | 31.89 |
| **Mean ± s.d.** | **90.69 ± 0.23** | **73.97 ± 0.44** | **30.70 ± 1.17** |

This seed: Low 7,653/8,444 steps correct, High 6,269/8,444. Per-step predictions and
scores of every seed are in the dataset [DMA-2B-results](https://huggingface.co/datasets/tuanhqv123/DMA-2B-results).

## How to use

Exactly as DMA-2B, with `revision="seed2"`:

```python
model = AutoModel.from_pretrained("tuanhqv123/DMA-2B", revision="seed2", trust_remote_code=True,
                                  dtype=torch.bfloat16)
```

or, to run the example:

```bash
pip install "transformers>=5" torch torchvision pillow
git clone -b seed2 https://huggingface.co/tuanhqv123/DMA-2B DMA-2B-seed2 && cd DMA-2B-seed2
python example.py
```

## Weights

`model.safetensors` is the step-7,000 checkpoint of seed 2, sha256 `a069e5dd22609d9670f9650348e021a3e6203e4d8283aa9884e6c326d39616e3`. Optimizer: 8-bit AdamW,
learning rate 1e-5 with 30 warm-up steps and cosine decay, batches of 16 rows, one RTX 4090.

## License

LFM Open License v1.0, inherited from the base model `LiquidAI/d1-2B`; see [LICENSE](./LICENSE) and
[NOTICE.md](./NOTICE.md).
