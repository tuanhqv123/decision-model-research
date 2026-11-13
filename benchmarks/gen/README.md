# Q1 generative control arm (`gen_v1`)

Question (PLAN-2026-11-10 §3 Q1): at the same model family and size, the same training rows, the same inputs and
the same harness, what does a single-pass decision model (R1: `LiquidAI/d1-2B`, contract `v1`) lose or gain
against a model that **generates** the action? This arm fine-tunes the public base `LiquidAI/LFM3-VL-2B`
(revision `23387beda43c60310348e83e25f78f33c3140d3a`) to emit one compact JSON action.

| File | Role |
|---|---|
| `gen_common.py` | prompt (R1's `build_state` + the frozen `gen_v1` instruction), target JSON, 0-999 grid, parser, confidence, contract files |
| `gen_model.py` | load, chat-template prompt, supervised example (loss mask), answer-only loss, greedy engine |
| `make_gen_data.py` | row -> (prompt, image, target); `--check` is the data gate |
| `train_gen.py` | the SFT trainer (R1's rows, order, schedule, optimizer, frozen embedding) |
| `eval_gen.py` | greedy generation on validation, scored by eval_d1's row shape + `run_acv2.score_row` |
| `cpu_check.py` | CPU-only verification of the template, the mask, token counts, one forward/backward, save/reload, API |
| `../../serving/gen_api.py` | FastAPI server with d1_api's endpoints and fields (+ `tokens_generated`, `raw`, `parse_error`) |

## Environment

`transformers>=5.21.0` (model card), which needs Python >= 3.11. Use the same stack as R1's trainer:
Python 3.12, `torch==2.6.0` (cu124), `torchvision==0.21.0`, `transformers==5.22.2`, `tokenizers==0.23.3`,
`torchao==0.10.0`, `pillow`, `numpy`, and `fastapi`/`uvicorn` for serving. Pin the base with
`huggingface_hub.snapshot_download("LiquidAI/LFM3-VL-2B", revision="23387beda43c60310348e83e25f78f33c3140d3a",
local_dir="<MODELS>/LFM3-VL-2B")` and pass that directory as `--init` if the GPU host has no Hub access.

Placeholders: `<PY>` python of that stack; `<REPO>` this repository (or a copy holding `benchmarks/gen`,
`benchmarks/train`, `serving`); `<ACDATA>` the directory holding `compact_cur15k_v1.jsonl`,
`compact_validation_v1.jsonl` and `imgs/`; `<RUNS>` a writable run directory; `<BASE>` either
`LiquidAI/LFM3-VL-2B` or the local snapshot directory.

## 0. Data gate (CPU, about a minute per split)

```bash
cd <REPO>
<PY> benchmarks/gen/make_gen_data.py --data <ACDATA> --split cur15k_v1 --check
<PY> benchmarks/gen/make_gen_data.py --data <ACDATA> --split validation_v1 --check
```
Pass = `CHECK_PASS hard_failures=0` (every row converts; every target re-parses to itself). Measured 2026-11-10:
14,965 / 827 rows; `navigate_home` 0 rows in both; decoded gold point outside its own gold box 2/8,387 and
1/474 spatial rows (zero-width/height boxes: the grid's ceiling).

## 0b. CPU check of the real model (optional, no GPU)

```bash
CUDA_VISIBLE_DEVICES="" nice -n 19 <PY> benchmarks/gen/cpu_check.py --model <MODELS>/LFM3-VL-2B --data <ACDATA> \
    --forward --backward --generate --roundtrip <RUNS>/roundtrip_tmp --out <RUNS>/cpu_check.json
```
`--data` needs a few `imgs/<ep>/<step>.jpg` present; pass = `CPU_CHECK_PASS`.

## 1. 30-step smoke (GPU)

```bash
cd <REPO>
CUDA_VISIBLE_DEVICES=0 <PY> benchmarks/gen/train_gen.py --data <ACDATA> --out <RUNS>/gen_smoke \
    --init <BASE> --train-set cur15k_v1 --steps 30 --micro 1 --accum 16 --lr 1e-5 --warmup 30 --seed 0 \
    --grad-ckpt --freeze-embed --optim adamw8bit --mem-frac 0.55
CUDA_VISIBLE_DEVICES=0 <PY> benchmarks/gen/eval_gen.py --ckpt <RUNS>/gen_smoke/ckpt --data <ACDATA> --val \
    --val-set validation_v1 --limit 50 --tag gen_smoke --out <RUNS>/gen_smoke/eval
```
Pass = `TRAIN_DONE`; logged `loss` falls and `tok_acc` rises over the 30 steps; `mem_gb` (same
`torch.cuda.max_memory_allocated` metric R1 logs) about 10.5-12; the eval prints `SUMMARY` with
`parse_fail_rate` well below 1 (the format is learned within tens of steps). The smoke's step accuracy means
nothing: with `--warmup 30 --steps 30` the learning rate never leaves the warm-up ramp.

## 2. Full run (GPU) — R1's recipe

```bash
CUDA_VISIBLE_DEVICES=0 <PY> benchmarks/gen/train_gen.py --data <ACDATA> --out <RUNS>/gen_s0 \
    --init <BASE> --train-set cur15k_v1 --steps 938 --micro 1 --accum 16 --lr 1e-5 --warmup 30 --seed 0 \
    --grad-ckpt --freeze-embed --optim adamw8bit --mem-frac 0.55
```
Seeds 1 and 2: `--seed 1 --out <RUNS>/gen_s1`, `--seed 2 --out <RUNS>/gen_s2` (same seed = same row order as R1's
seed). Output: `<RUNS>/gen_s0/ckpt` with `contract.json` = `{"name": "gen_v1", "tracks": ["V"]}`.

## 3. Validation eval (GPU)

```bash
CUDA_VISIBLE_DEVICES=0 <PY> benchmarks/gen/eval_gen.py --ckpt <RUNS>/gen_s0/ckpt --data <ACDATA> --val \
    --val-set validation_v1 --tag gen_s0 --out <RUNS>/gen_s0/eval
```
Reports `step_acc`, `type_acc`, `grounding`, `per_type`, `parse_fail_rate` + `parse_errors`, `tokens_generated`,
`no_eos_within_budget`, `latency_ms`, and the trivial baselines `always_click_type` / `always_status` on the same
rows. PRE-REG-2026-11-10 §4's kill rule applies unchanged: step accuracy must beat always-click by >= 10 points.

## 4. Serving (GPU)

```bash
cd <REPO>
GEN_MODEL=<RUNS>/gen_s0/ckpt GEN_MEM_FRAC=0.25 D1_IMG_W=720 D1_API_TOKEN=<TOKEN> \
    <PY> -m uvicorn gen_api:app --app-dir serving --host 127.0.0.1 --port <PORT>
curl -s 127.0.0.1:<PORT>/health      # contract must read gen_v1, tracks ["V"]
```
The harness needs no change: point `D1_URL` at the port and keep `track: V`. A request with `track: VA` is refused
(400). The server refuses a checkpoint whose `contract.json` is not `gen_v1` or whose `gen_prompt.json` hash differs
from `gen_common.INSTRUCTION`.

## Parity with R1 and the asymmetries that cannot be removed

Matched by construction (and pinned by `tests/test_gen_*.py`): the 14,965 `cur15k_v1` rows and their draw order
for a given seed; the state string (`build_state(goal, "V", None, None, history)`, no memory, no app intent);
the picture (`train_d1.load_image`, 720 px; identical image-processor settings — pixel bounds, 16-px patches,
2x2 merge, normalisation — and d1's 1.64 MP cap does not bind at 720 px, so a 720x1600 screenshot is 1,100 image
tokens here; not re-counted inside d1's engine, whose processor source file is a different revision);
938 steps x 16; lr 1e-5, 30 warm-up steps, cosine; torchao AdamW8bit, weight decay 0, bf16 stochastic rounding;
clip 1.0; bf16 weights; gradient checkpointing; frozen tied embedding / LM head; the scorer path.

Not matched (disclose): initial weights (d1-2B carries Liquid's decision post-training); the loss (weighted
per-question CE with Gaussian-smoothed box edges vs. token CE on the JSON); coordinate supervision (four box edges
vs. the centre's digits); string supervision (R1's pointer can only copy from the state and is unsupervised on
14.3% of training `open_app` rows, generation is supervised on all and can emit any string); the prompt wrapping
(d1's question slots vs. chat template + instruction); inference (one pass vs. prefill + decode); confidence
(9-way choice probability vs. product of token probabilities under the full vocabulary); the possibility of an
unparseable answer (gen only); learning-rate tuning (1e-5 was chosen for d1, not for this arm).
