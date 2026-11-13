# Hugging Face release: DMA-2B (2026-11-13)

Published model: https://huggingface.co/tuanhqv123/DMA-2B (public), Hub commit `31169f0a`.

* Weights: v4 checkpoint 2482 (`model.safetensors` md5 `fa8ba04b1a19f472d0e124230b86069c`, identical to the local backup
  `backup-2026-11-12/ckpt-v4_s0_2482`). Base remote code, tokenizer, config and LICENSE are unchanged copies of
  `LiquidAI/d1-2B@cfe5760`; the base model card is replaced by `README.md`.
* Files in this folder are the card files as uploaded; `ac_common.py` on the Hub is `benchmarks/train/ac_common.py`
  at this commit.
* License check (LFM Open License v1.0, Section 4): license copied, changed files listed in `NOTICE.md`, attribution
  kept; Section 7: Liquid AI names used only to describe the origin; the model name avoids "LFM"/"Liquid".
* Hub conventions followed: `license: other` + `license_name: lfm1.0` + `license_link: LICENSE`, `base_model` +
  `base_model_relation: finetune` (the model appears in the base model's fine-tune tree), `library_name: transformers`,
  `pipeline_tag`, `model-index` results. The newer `.eval_results/` format needs a Hub benchmark dataset;
  AndroidControl and AndroidWorld are not Hub benchmarks, so it is not used.
* Verification: `example.py` run on the GPU host from the local folder and again from a fresh `snapshot_download` of the
  Hub repo; both reproduce the recorded AndroidWorld decision (click p = 0.988, element [6] p = 0.776, tap (407, 860)).
* Not released: v3 (better AC-High SR, no `answer` action) and v6 (no gain over v4).

## Demo Space (2026-11-13)

* https://huggingface.co/spaces/tuanhqv123/DMA-2B-demo — **static** Space (`release/hf-DMA-2B-demo-static/`): replays three
  AndroidWorld episodes DMA-2B completed (SystemWifiTurnOnVerify, NotesRecipeIngredientCount, SimpleSmsReply; seed 30,
  harness C run `aw116_v4g_s0`) with the recorded action / element / app / scroll probabilities. `data.json` is built
  from the recorded inputs with the model repo's `ac_common.py` and `example.py`.
* A Gradio Space that also re-runs the model live on CPU is in `release/hf-DMA-2B-demo-gradio/`. Hugging Face refused to
  create it (HTTP 402: Gradio and Docker Spaces on free cpu-basic need a PRO subscription). Tested on the GPU host with
  the GPU hidden: all 16 recorded steps render; a live CPU forward pass (float32) reproduces the recorded decision
  (answer step: p 0.594 vs 0.586 recorded). Measured CPU time: about 46 s with 2 threads, 15 s with 16 threads.
* Model card updated: AC-Low, AC-High and AndroidWorld tables per checkpoint (Type, Grounding, SR), and what each
  metric means.
