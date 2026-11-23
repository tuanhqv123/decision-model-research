#!/usr/bin/env python3
"""Model cards and NOTICE files of the DMA-2B seed branches (revisions `seed1`, `seed2` of tuanhqv123/DMA-2B), from
benchmarks/results/2026-11-16-ac-noterm/seeds_summary.json. Writes release/hf-DMA-2B-seeds/seed{1,2}/{README,NOTICE}.md."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
S = json.load(open(os.path.join(ROOT, "benchmarks/results/2026-11-16-ac-noterm/seeds_summary.json")))
SHA = json.load(open(os.path.join(HERE, "sha256.json")))  # {"1": "...", "2": "..."} of model.safetensors


def card(seed):
    v = S["seeds"][str(seed)]
    rows = "\n".join(
        f"| {k}{' (released as DMA-2B)' if k == '0' else ''}{' **(this branch)**' if k == str(seed) else ''} | "
        f"{x['low']['sr']:.2f} | {x['high']['sr']:.2f} | {x['high']['episode_sr']:.2f} |"
        for k, x in sorted(S["seeds"].items()))
    m = lambda k: f"{S[k]['mean']:.2f} ± {S[k]['sd']:.2f}"  # noqa: E731
    return f"""---
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
  - name: DMA-2B-seed{seed}
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
            value: {v['low']['sr']:.2f}
      - task:
          type: gui-action-selection
          name: AndroidControl-High (step success rate, InfiGUI-R1 scorer)
        dataset:
          type: androidcontrol
          name: AndroidControl test, high-level goals (8,444 steps)
        metrics:
          - type: step_success_rate
            name: SR
            value: {v['high']['sr']:.2f}
---

# DMA-2B, training seed {seed} (revision `seed{seed}`)

This branch holds **training seed {seed}** of [DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B). The default branch
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
{rows}
| **Mean ± s.d.** | **{m('low_sr')}** | **{m('high_sr')}** | **{m('high_episode_sr')}** |

This seed: Low {v['low']['exact']:,}/8,444 steps correct, High {v['high']['exact']:,}/8,444. Per-step predictions and
scores of every seed are in the dataset [DMA-2B-results](https://huggingface.co/datasets/tuanhqv123/DMA-2B-results).

## How to use

Exactly as DMA-2B, with `revision="seed{seed}"`:

```python
model = AutoModel.from_pretrained("tuanhqv123/DMA-2B", revision="seed{seed}", trust_remote_code=True,
                                  dtype=torch.bfloat16)
```

or, to run the example:

```bash
pip install "transformers>=5" torch torchvision pillow
git clone -b seed{seed} https://huggingface.co/tuanhqv123/DMA-2B DMA-2B-seed{seed} && cd DMA-2B-seed{seed}
python example.py
```

## Weights

`model.safetensors` is the step-7,000 checkpoint of seed {seed}, sha256 `{SHA[str(seed)]}`. Optimizer: 8-bit AdamW,
learning rate 1e-5 with 30 warm-up steps and cosine decay, batches of 16 rows, one RTX 4090.

## License

LFM Open License v1.0, inherited from the base model `LiquidAI/d1-2B`; see [LICENSE](./LICENSE) and
[NOTICE.md](./NOTICE.md).
"""


def notice(seed, base_notice):
    return base_notice.replace("Full fine-tune (seed 0,", f"Full fine-tune (seed {seed},")


if __name__ == "__main__":
    base_notice = open(os.path.join(ROOT, "release/hf-DMA-2B/NOTICE.md")).read()
    assert "Full fine-tune (seed 0," in base_notice
    for seed in (1, 2):
        d = os.path.join(HERE, f"seed{seed}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "README.md"), "w").write(card(seed))
        open(os.path.join(d, "NOTICE.md"), "w").write(notice(seed, base_notice))
    print("cards written")
