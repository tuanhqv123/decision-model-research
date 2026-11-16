# NOTICE

This repository contains a **modified version** of **LiquidAI/d1-2B** (Liquid AI, Inc.), which is licensed under
the **LFM Open License v1.0**. The full license text is in [LICENSE](./LICENSE), copied unchanged from the base
repository. This notice records which files were changed, as Section 4(b) of the license requires.

## Base model

* Source: https://huggingface.co/LiquidAI/d1-2B
* Revision: `cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41`
* License: LFM Open License v1.0

## Modified files

| File | Change |
|---|---|
| `model.safetensors` | **Modified.** Full fine-tune (seed 0, one epoch over 116,139 rows, released at optimizer step 7,000; the tied input embedding / output head stayed frozen) on the AndroidControl training split, as an Android GUI action selector. See README.md for data, recipe and evaluation. |
| `contract.json` | **Added by the fine-tune.** Names the question set this checkpoint was trained on. |
| `README.md` | **Replaced.** This repository's model card replaces the base model card. |

## Unchanged files

The remote-code files (`*.py` from the base repository, loaded with `trust_remote_code=True`), the tokenizer and
processor files, `config.json`, `calibration.json`, `chat_template.jinja`, `assets/` and `LICENSE` are byte-identical
copies from the revision above.

## Added files (not part of the base model)

| File | Content |
|---|---|
| `NOTICE.md` | this notice |
| `ac_common.py` | input building and answer decoding, copied verbatim from the fine-tuning project's code (standard library only) |
| `example.py` | runs the model on one AndroidControl test step and decodes the answer |
| `examples/ac_test_ep27_step1.json`, `examples/ac_test_ep27_step1.jpg` | that step: goal, step instruction, history, accessibility elements, app list, screenshot, recorded decision |

## Attribution

The base model, its architecture and its remote code are the work of Liquid AI. Training data: AndroidControl
(Li et al., NeurIPS 2024 Datasets and Benchmarks), which is not redistributed in this repository except for the one
test step used by the example (screenshot, accessibility elements and instruction of episode 27, step 1).

"Liquid AI", "LFM" and "d1" are names of Liquid AI, Inc.; they are used here only to describe the origin of the
work (license Section 7).
