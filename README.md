# DMA-2B: single-pass action selection for Android

DMA-2B is a 2B **decision model** for phone control. At each step it reads the screen (screenshot plus accessibility
elements), the goal and, optionally, a step instruction, and answers typed questions in **one forward pass**: which
action, which element, which scroll direction, which app, which text span to type. It generates no tokens, so its
output cannot be malformed, every answer comes with a probability, and one decision costs one forward pass.

It is a fine-tune of [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) on the full AndroidControl training
split. Weights: [huggingface.co/tuanhqv123/DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B).
Paper: [`papers/latex/dma-2b-androidcontrol/main.pdf`](papers/latex/dma-2b-androidcontrol/main.pdf).

## Results

AndroidControl test split, 8,444 steps, step success rate (%), InfiGUI-R1 evaluation code. DMA-2B is scored in
AndroidControl's own action space (which has no "done" action); the base model as originally evaluated.

| Model | Low | High |
|---|---|---|
| d1-2B (base, not fine-tuned) | 59.8 | 40.2 |
| **DMA-2B** | **90.5** | **74.2** |

- On AndroidControl-High, DMA-2B is the best reported 2B model (UI-TARS-2B 68.9, MAI-UI-2B 67.3) and is above several
  7B–8B generative agents (AgentCPM-GUI-8B 69.2, OS-Atlas-7B 71.2, UI-TARS-7B 72.5, GUI-Owl-7B 72.8, MobileRL-9B
  73.9). Four 3B–7B models are higher (UI-Genie-7B 74.5, GUI-SAGE-3B 75.4, UI-Venus-Navi-7B 76.1, GUI-SAGE-7B 76.8);
  the best reported value is 79.1. The full comparison, with sources, is Table 1 of the paper. Other models' numbers
  come from their papers, with different prompts and scorers, so the comparison is approximate.
- Confidence: on AndroidControl-High, a threshold on the action probability chosen on held-out episodes acts on 80%
  of the steps with an error rate of 19% (target 20%).
- One training seed; the paper lists the limits.

## Reproduce

Data (AndroidControl train split, rebuilt from the original release, test episodes excluded):

```bash
python3 benchmarks/train/build_ac_full_v2.py --raw <android_control tfrecords> --out ac_full.jsonl --split train
python3 benchmarks/train/build_exec_data.py benchmarks/train ac_full.jsonl data/compact_exec_v1.jsonl --seed 0
python3 benchmarks/train/upweight_exec.py --k 3 --extra-only data/compact_exec_v1.jsonl data/compact_execcont.jsonl
```

Training (one RTX 4090, about 16 h; base weights LiquidAI/d1-2B at revision cfe5760):

```bash
python3 benchmarks/train/train_d1.py --data data --init LiquidAI/d1-2B --contract v4 --track VA \
  --train-set exec_v1,execcont --epochs 1 --micro 1 --accum 16 --lr 1e-5 --warmup 30 \
  --freeze-embed --grad-ckpt --no-memory --seed 0 --save-every 500 --keep-ckpts --out runs/dma2b
```

Evaluation (AndroidControl-Low; drop `--instruction` for High) and scoring:

```bash
python3 benchmarks/train/eval_d1.py --ckpt runs/dma2b/ckpt_7000 --data actest --val --val-set actest_v4 \
  --track VA --instruction --exclude-actions terminate --tag low --out out
python3 benchmarks/train/score_actest.py --pred out/eval-low-VA-val.jsonl --test infigui_test.json --out out/perstep-low.jsonl
```

Paper numbers, tables and figures: `benchmarks/train/paper_ac_final.py`, then `tables/make_numbers.py`,
`tables/make_ac_table.py` and `figures/make_figures.py` in the paper folder; build with `tectonic main.tex`.
Result summaries: `benchmarks/results/2026-11-16-ac-noterm/`.

## Layout

```
benchmarks/train/   data builders, trainer, evaluation and scorers (with tests)
benchmarks/results/ result summaries
serving/            HTTP API for the model (one decision per request)
papers/latex/       the paper (LaTeX, figures, tables)
docs/, notes/       research notes and decision records
```

## License

Code: see the repository license. Model weights: a fine-tune of LiquidAI/d1-2B under the LFM Open License v1.0
(see the model card).
