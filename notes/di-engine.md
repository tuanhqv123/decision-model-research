# Plugging our model into the Decision Index (status)

We built and validated an on-device **MLX engine** so our LFM2.5 head runs through the real
Decision Index harness (`apolinario/decision-index`) — the field's benchmark for typed decision
engines. File: `benchmarks/decision-index/decision_index/engines/mlx_engine.py`; registered as
engine `mlx`.

## What's done
- **Engine built** (`MlxEngine`): implements the same policy as the reference `transformers`
  engine — every option is scored by the summed log-probability of its option-key tokens over a
  fixed prompt; option probabilities are the softmax over those scores. No option filtering; no
  truncation (over-length prompts are `Unsupported`).
- **Validated** against the kit's `validate()` on both primitives:
  - `choice` → `{"type":"choice","choice":k,"probabilities":{...}}` (sums to 1.0),
  - `noul` → `{"type":"noul","noul":p}`.
  - Our LFM2.5 correctly chose `refund` (0.85) and `true` (0.92) on a test state, and the
    response passed DI validation.
- **Environment:** Python 3.11 venv at `benchmarks/decision-index/.venv` with
  `mlx`/`mlx-lm` 0.32 + the kit installed (`pip install -e ".[transformers,rebuild]"`).

## What's NOT done (and why)
We did **not** get a real Decision Index score for our model. The blockers, in order:

1. **Rebuild is all-or-nothing per normalizer.** `suite rebuild --only <ids>` can't isolate a
   single benchmark: the `knowledge` normalizer (HellaSwag, MMLU, ARC, GPQA, WinoGrande) reads
   several source files together, so a `--only` build fails with `FileNotFoundError` on a source
   it didn't download. The partial `work/` build downloaded GSM8K + HellaSwag and then stopped.
2. **Full rebuild is heavy + gated.** The complete suite is ~7 GB of downloads and ~17 GB of
   working space; the **HLE** source is gated on Hugging Face (needs your token + accepting the
   terms). ANLI is CC BY-NC; RAGTruth contexts are non-redistributable.
3. **Hand-building rows** to bypass the rebuild means reconstructing `freeze.py`'s
   `_evaluation` (run_id, payload_sha256 over the exact model payload, deterministic group hashes,
   provenance) — brittle and low-value.

## How to get a real DI score (when we choose to)
- **Local, open subset:** rebuild the suite for the MIT/CC-BY benchmarks only (MMLU-Pro 57, BBH 58,
  HellaSwag 29, WinoGrande 28, GSM8K 30…) in the Python 3.11 env, then
  `python -m decision_index suite sample --n 300` → `run --engine mlx` → `score`.
- **Full board / private:** `hf-job` on one RTX PRO 6000, or the maintainers' submission, which
  needs the **gated HLE** accept + HF auth. Our `MlxEngine` drops in unchanged via
  `--engine decision_index.engines.mlx_engine:MlxEngine`.

## Bottom line
We understand exactly how the field benchmarks decision models (see
[`how-they-benchmark.md`](./how-they-benchmark.md)), and our on-device LFM2.5 is
**benchmark-compatible** (engine validated). Producing a board number is gated by a heavy, partly
gated rebuild — not by our model or our harness.
