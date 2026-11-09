# Deployment of a Small Decision Model to Mobile / Edge

**Topic:** Deploying a small "System One" decision model (e.g. Liquid d1-3B / d1-omni-600M, or our own post-trained model) to mobile / edge.
**Anchors:** Jev (TypeSafe, proprietary frontier), d1-3B / d1-omni-600M (Liquid, open blueprint).
**Status:** Phase 1 research — evidence gathering for `notes/deployment.md`.

> Citation rule: every claim below carries a `[bibkey]` and a full URL (see **Sources**, bottom). Primary = vendor blog / Hugging Face model card / official docs / arXiv paper. Vendor marketing statements are labeled **claim, not evidence**. New bibkeys are proposed for the parent to fold into `references/bib.bib` — I did **not** edit `references/bib.bib` or `references/ledger.md`.

---

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | Liquid AI, "Open d1" (blog + HF blog) | https://www.liquid.ai/blog/d1-open | Full latency table (Apple M5 Pro / Jetson AGX Thor / Orin / Orin Nano / RTX 4090 / MI325X); llama.cpp day-one + NVFP4 | primary | high |
| 2 | d1-3B HF model card | https://huggingface.co/LiquidAI/d1-3B | Arch, latency, Decision Index table, CUDA-graph warm-up note | primary | high |
| 3 | d1-omni-600M HF model card | https://huggingface.co/LiquidAI/d1-omni-600M | 587M param breakdown; float16 vs bfloat16 answer change; no reported speed | primary | high |
| 4 | NVIDIA Jetson AI Lab, "d1-3B" | https://www.jetson-ai-lab.com/models/d1-3b/ | Run d1-3B on Jetson (JetPack 7.2, PyTorch container, BF16), warmup latency | primary | high |
| 5 | LiquidAI/d1-3B-GGUF (HF repo) | https://huggingface.co/LiquidAI/d1-3B-GGUF | Official GGUF: BF16/F16/Q4_K_M/Q8_0 + mmproj; `llama-server -hf …:Q8_0`; `/v1/systemone` endpoint | primary | high |
| 6 | LiquidAI/d1-omni-600M-GGUF (HF repo) | https://huggingface.co/LiquidAI/d1-omni-600M-GGUF | Official GGUF: BF16/F16/Q8_0 + mmproj (no Q4_K_M published) | primary | high |
| 7 | Liquid docs, llama.cpp (on-device) | https://docs.liquid.ai/deployment/on-device/llama-cpp | GGUF quant levels; llama-server / llama-cli usage | primary | high |
| 8 | Liquid docs, iOS & Android (llama.cpp) | https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile | XCFramework / NDK in-process; mobile tuning (mmap, threads, n_gpu_layers, Q4_0 default) | primary | high |
| 9 | Liquid docs, MLX | https://docs.liquid.ai/deployment/on-device/mlx | mlx-lm on Apple Silicon (Metal, unified memory) | primary | high |
| 10 | Liquid docs, Model Library / formats | https://docs.liquid.ai/lfm/models/complete-library | For d1-3B & d1-omni-600M: HF=✓, GGUF=✓, **MLX = –, ONNX = –** | primary | high |
| 11 | NVIDIA, "Introducing NVFP4" | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | NVFP4 format; ≤1% accuracy loss vs FP8 (DeepSeek-R1-0528) | primary | high |
| 12 | Apple ML Research, "On Device Llama 3.1 with Core ML" | https://machinelearning.apple.com/research/core-ml-on-device-llama | Core ML export (coremltools), fused SDPA, stateful KV cache, int4 quant; methodology (TTFT / extend throughput) | primary | high |
| 13 | arXiv 2508.16785, "Interpreting the Effects of Quantization on LLMs" | https://arxiv.org/html/2508.16785v3 | 4/8-bit: calibration effect generally minor; 1–4% accuracy but outliers; calibration task/model-dependent | primary | high |
| 14 | arXiv 2509.21173, "Quantization is not systematically destructive" | https://arxiv.org/html/2509.21173v4 | W8A8 sweep: ~40% of runs improve calibration (lower ECE); 8.5% improve both acc+calibration | primary | high |
| 15 | ACL 2025, "Quantized Can Still Be Calibrated" | https://aclanthology.org/2025.acl-long.1473.pdf | Analytic upper bound of calibration error (UBCE) for quantized LLMs | primary | medium |
| 16 | arXiv 2309.13866, "On Calibration of Modern Quantized Efficient Neural Networks" | https://ar5iv.labs.arxiv.org/html/2309.13866 | Calibration tracks quantization quality; worse at 4-bit activation | primary | medium |
| 17 | arXiv 2609.11940, "The Battery Price of Edge AI" | https://arxiv.org/abs/2609.11940 | On-device LLM ~3× less energy-efficient than batched server inference/token; quant×energy non-monotonic; battery-aware routers | primary | high |
| 18 | arXiv 2506.19884, "MNN-AECS: Energy Optimization for LLM Decoding" | https://arxiv.org/html/2506.19884 | Memory-bound decode dominates mobile energy | primary | medium |
| 19 | mlc-ai/web-llm | https://github.com/mlc-ai/web-llm (+ arXiv 2412.15803) | In-browser LLM via WebGPU | primary (repo) | high |
| 20 | wllama-webgpu (WASM+WebGPU bindings for llama.cpp) | https://www.npmjs.com/package/@reeselevine/wllama-webgpu | Browser/WASM llama.cpp bindings, WebGPU backend | secondary | medium |
| 21 | mauludsadiq/State-Pack | https://github.com/mauludsadiq/State-Pack | KV-cache state packing across agent steps / shared system prompt | secondary | medium |
| 22 | tryhamster, batching parallel AI agent decisions | https://tryhamster.com/skills/batching-and-parallelizing-decisions | Batch all independent decisions in one call against one state snapshot | secondary | medium |
| 23 | Sapirex d1 deployment guide | https://sapirex.com/en/liquid-ai-d1-edge-decision-model-deployment-guide-implementing-zero-token-routing-with-llama-cpp/ | d1-3B Q4_K_M ≈ 2.2–2.8 GB VRAM & ~28 ms; claims ">98% fidelity" at Q4_K_M | secondary | medium |
| 24 | StudioTV, d1-3B VRAM requirements | https://studiotvai.com/vram-requirements/d1-3b | d1-3B BF16 ≈ 5.82 GiB; ~1.57 GiB @ 4-bit; ~0.50 GiB KV/32K-request | secondary | medium |
| 25 | Liquid AI, "LLM2Jev / Jev" (context) | https://typesafe.ai/blog/introducing-system-one-models-and-jev | Jev: 70–500 ms, typed decisions, "can't hallucinate" | primary (vendor) | medium |

---

## 1. d1's deployment: llama.cpp day-one, NVFP4, NVIDIA DGX→RTX→Jetson, Apple M5 Pro

**d1-3B deployment is a llama.cpp story, day-one.** Liquid shipped an official GGUF at launch (`LiquidAI/d1-3B-GGUF`: `d1-3B-BF16.gguf`, `d1-3B-F16.gguf`, `d1-3B-Q4_K_M.gguf`, `d1-3B-Q8_0.gguf` plus `mmproj-*` for vision) and it runs through plain `llama-server` with a dedicated `/v1/systemone` endpoint `[d1gguf][d1modelcard]`:
```
llama-server -hf LiquidAI/d1-3B-GGUF:Q8_0
curl http://127.0.0.1:8080/v1/systemone -d '{"state": "...", "questions": {"refund": {"type":"noul",...}}}'
```
`decision_type` in the GGUF header is `lfm2-d1`; `context_length` 32 768.

**Latency (single question, warm, one request at a time), reported by Liquid and mirrored by NVIDIA [liquidAI2026opend1][d1modelcard][jetsond13b]:**

| Device | 1 question | 3 questions (1 pass) | 3.4K-token state | 384px image | 64 states packed |
|---|---|---|---|---|---|
| Apple M5 Pro (`mps`) | 30 ms | 41 ms | 640 ms | 62 ms | 78 /s |
| Jetson AGX Thor | 16 ms | 20 ms | 220 ms | 35 ms | 262 /s |
| Jetson AGX Orin 64 GB | 26 ms | 35 ms | 560 ms | 83 ms | 110 /s |
| Jetson Orin Nano | 50 ms | 73 ms | 1 640 ms | 202 ms | 38 /s |
| NVIDIA RTX 4090 | 8 ms | 21 ms | 102 ms | 17 ms | 475 /s |
| AMD MI325X | 9 ms | 14 ms | 44 ms | 18 ms | 1 106 /s |

Key interpretations:
- **Every measured device answers a single question in < 50 ms**; the RTX 4090 is < 10 ms `[liquidAI2026opend1][d1modelcard]`.
- **Batching questions over one state is cheap**: 3 questions take ~1.3× a single question (Thor 16→20 ms; M5 Pro 30→41 ms) `[liquidAI2026opend1]`. This is the central mobile/edge advantage: the state is encoded **once** and read for all questions.
- **3.4K-token state dominates latency** (640 ms on M5 Pro, 1 640 ms on Orin Nano). For "interact on mobile," keep states short or cache the encoded state.
- **Inference cost scales with input length** (single forward pass), not with answer token count (there are none).

**"Runs the full NVIDIA stack, from DGX to Jetson"** is a vendor claim; the verifiable part is the measured Jetson/RTX numbers (NVIDIA co-authored the Jetson evaluation) and the official Jetson AI Lab guide that runs d1-3B on Jetson AGX Thor / AGX Orin 64 GB / Orin Nano with JetPack 7.2 and the NVIDIA PyTorch container (BF16 checkpoint) `[jetsond13b]`.

**NVFP4.** Liquid states "Native support for llama.cpp across Apple, AMD, Qualcomm, and NVIDIA with NVFP4" `[liquidAI2026opend1]` — **claim, not evidence; no independent benchmark of d1 in NVFP4 published.** NVFP4 itself is a real NVIDIA format: 4-bit float with E4M3 FP8 scaling factors and a two-level micro-block strategy, on Blackwell; NVIDIA reports ≤1% accuracy degradation vs FP8 on DeepSeek-R1-0528, produced via TensorRT Model Optimizer `[nvfp4]`. **NVFP4 is a desktop/data-center/Blackwell path, not a mobile path** — phones use integer 4-bit (INT4) via GGUF `Q4_0`/`Q4_K_M` or Core ML/MLX int4.

**Jetson system-on-module context** (for power/size framing of "edge"): Jetson Thor is ~2 070 FP4 TFLOPS / 128 GB / 40–130 W, ~7.5× AGX Orin compute @ 3.5× efficiency; AGX Orin ~275 TOPS @ 15–60 W; Orin Nano 8 GB is 7 W / 15 W / 25 W modes (max) `[jetsonthor]` — these are a different class from a phone; the phone-relevant numbers are the M5 Pro and (inferred) Apple/Android NPU/GPU.

## 2. Size / footprint tradeoff: d1-omni-600M (footprint) vs d1-3B (quality)

| | d1-3B | d1-omni-600M |
|---|---|---|
| Params | 3.12B (3 123 483 888) | 587M (587 161 089) |
| Backbone | LFM2.5-VL-3B (decoder-only VLM; SigLIP2 NaFlex 400M vision) | LFM2.5-Encoder-350M (bidirectional; +94M SigLIP2 vision +112M FastConformer audio; 381M shared trunk+head) |
| Context | 32 768 tokens | 16 384 tokens (images: state+question cut to 896) |
| Vocab | 128 000 | 65 536 |
| Decision Index 0.2.1 | **48.57** | 15.95 |
| Text-benchmark mean | **82.9** | 78.4 |
| Official GGUF | BF16/F16/Q4_K_M/Q8_0 | BF16/F16/Q8_0 (no Q4_K_M) |

Sources `[d1modelcard][d1omnicard][d1gguf][d1omnigguf][liquidAI2026opend1]`.

Tradeoff read:
- **d1-3B = quality.** Best decision model < 10B on Decision Index 0.2.1 (48.57), above Decider 35B-A3B (47.11) despite being 12× smaller; text mean 82.9 > Decider 4B (81.1) `[liquidAI2026opend1][d1modelcard]`.
- **d1-omni-600M = footprint + multimodal (audio).** 78.4 text mean beats Decider 2B (77.1) at a quarter of the params; adds **audio** (up to 30 s speech) which d1-3B lacks `[d1omnicard]`. But Decision Index 15.95 is far behind d1-3B — the 600M model is a **routing/triage** model, not a top-quality judge `[liquidAI2026opend1]`.
- **Weights footprint.** d1-3B BF16 ≈ 5.8–6.2 GiB; ≈1.6 GiB @ 4-bit; d1-3B `Q8_0` GGUF ≈ 2.7–2.9 GB `[studiotv][d1gguf]`. d1-omni-600M is ~587M F32 (~2.35 GB safetensors; ~4.7 GB repo) → roughly ~0.3–0.6 GB at int4/Q8, i.e. **goes into under-2-GB mobile budgets** `[d1omnicard][d1omnigguf]`. Sapirex (secondary) quotes d1-omni-600M Q4_K_M < 1.5 GB and d1-3B Q4_K_M 2.2–2.8 GB VRAM `[sapirex]` — flag as **claim/estimate**, not measured on our hardware.
- **Vision encoder pull-through:** removing images from an image question drops d1-3B from 74.1 to 45.1 on the 11-image benchmark means — the image is doing the work; a vision decision model needs the projector loaded too `[d1modelcard]`.

## 3. Conversion paths to mobile

**Reality check (important, from primary docs):** in Liquid's model-format matrix, for **d1-3B and d1-omni-600M only `HF` (transformers, `trust_remote_code`) and `GGUF` are published. MLX = –, ONNX = –** `[liquidmodels]`. So the **only** officially published on-device path for d1 today is **llama.cpp / GGUF**. If we want Core ML or ONNX Runtime or MLX for d1, we must convert ourselves (see gaps).

Per-target options:

- **llama.cpp / GGUF (officially supported, mobile-ready).** `llama.xcframework` ships with Metal + `mtmd` multimodal and slices for iOS/macOS/visionOS/tvOS; Android via `examples/llama.android` NDK module (`AiChat`/`InferenceEngine`) or a CMake subdirectory + JNI. In-process C API: `llama_model_load_from_file` → `llama_init_from_model` → sampler chain → tokenize → decode loop `[mobileguide]`. For d1 the "decode loop" is a single pass to read the answer distribution from one forward pass, served by `/v1/systemone` `[d1gguf]`.
  - **Mobile tuning from Liquid docs:** keep `n_ctx` small (KV cache grows linearly); `n_threads = activeProcessorCount - 2`; `n_gpu_layers = 99` on Apple for Metal, CPU is the Android default (Vulkan/OpenCL Adreno need device testing); weights are memory-mapped by default (`use_mmap`) so they count as file-backed pages, not RSS (helps iOS jetsam / Android LMK); `Q4_0` is the recommended phone default (llama.cpp repacks into Arm-optimized kernels at load), `Q4_K_M` when quality matters `[mobileguide]`.
- **Core ML (Apple).** Apple's canonical LLM recipe: trace/`coremltools` export; **fused SDPA** op (set deployment target macOS 15+ / iOS 18+; maps to one fused GPU kernel); **stateful KV cache** via `ct.StateType` to avoid copy overhead; **block-wise int4 weight quantization**. Apple's worked example gets Llama-3.1-8B to ~16 tokens/s (stateful KV) and ~33 tokens/s after int4 quantization on an M1 Max, and defines the metrics TTFT (time to first token) and extend throughput (tokens/s) `[coreml]`. This is the methodology/template for a Core ML deployment of a decision model — but d1 requires the custom `D1Model`/`D1OmniModel` code (`trust_remote_code`) to be converted, which is non-trivial and not released (medium/high effort, unverified for d1).
- **ONNX Runtime.** Liquid publishes ONNX for the *generative* LFM family (`*-ONNX`, including LFM2.5-VL-3B-ONNX), and its job is "production/edge, cross-platform CPUs/GPUs/accelerators" with FP32/FP16/Q4/Q8 `[liquidmodels]`. But **d1 decision models have no published ONNX**, so an ONNX Runtime mobile path means converting the d1 forward pass ourselves (drop-in for the `system_one` path). Note: d1-omni-600M's `causal:false` bidirectional encoder and audio FastConformer add conversion complexity.
- **MLX (Apple Silicon).** `mlx-lm` loads MLX checkpoints (e.g. `mlx-community/LFM2-…-8bit`), 3–8 bit + BF16, Metal acceleration via unified memory `[mlxdocs]`. **No d1 MLX published** `[liquidmodels]` → self-convert. Good option for Mac/local dev; not directly Apple phone NPU.
- **WebGPU / WASM (browser).** For an in-browser or lightweight client: **WebLLM** runs LLMs in-browser with WebGPU and no server `[webllm]`; **wllama-webgpu** is WASM+WebGPU llama.cpp bindings (WebGPU backend, split/parallel model files) `[wllama]`. Both are token-generating engines, but a GGUF-backed d1 decision pass could in principle be driven through a llama.cpp WASM build (unverified for `/v1/systemone`). Works best for desktop browsers; mobile browser WebGPU is uneven.

**Recommended target stack for "our" decision model on mobile:** llama.cpp/GGUF (only published, low risk) → optionally Core ML on Apple phone, ONNX Runtime for cross-device, WebGPU/WASM for browser/embeddable. Each beyond llama.cpp needs our own conversion (gap).

## 4. Quantization (4-bit / 8-bit) and its effect on calibration

A decision model's whole value is a **calibrated** probability, so quantization's effect on calibration (not just accuracy) is the key question. Findings:

- 4-bit/8-bit weight quantization **generally has a minor effect on LLM calibration** (ACE / adaptive calibration error) and on confidence; accuracy typically drops only 1–4%, but there are real outliers — Llama-2-7B −14% on toxicity, Qwen-3B −7% on HellaSwag at 4-bit; 4-bit models tend to be slightly *less* confident. Effect is **model- and task-dependent** (some tasks/mistral variants *improve* calibration when quantized), so it must be evaluated per-model/per-task `[quanteffect]`.
- A large W8A8 sweep found **~40% of runs get lower ECE (better calibration) after quantization**, and 8.5% improve both accuracy and calibration — quantization is "not systematically destructive" `[quantnotdestructive]`.
- Smaller efficient CNNs: calibration degrades with precision, worst at 4-bit activation `[quantcnncal]`.
- Analytic framing: there's a provable upper bound on calibration error for quantized LLMs (UBCE) you can compute cheaply instead of measuring `[quantcal]`.
- **d1-specific precision hazard (primary, very relevant):** for **d1-omni-600M** Liquid says the model was trained in **float32**; float16 is faster and keeps the float32 outputs (same top answer on all 243 text / 214 image / 416 audio rows), but **bfloat16 changed the top answer on 0.8% of text and 1.7% of audio rows — avoid bfloat16 on the 600M model** `[d1omnicard]`. Also: "text answers are calibrated with per-type temperatures stored in `config.json`; image and audio answers are the model's softmax as trained" — so d1-omni re-calibrates text with temperature but not vision/audio; a quantization-induced shift could interact with those temperatures `[d1omnicard]`.

**Practical guidance for our model:** quantize weights (prefer int4/Q8 or FP4 on capable hardware); **always re-measure calibration (ECE/Brier/reliability) after quantization**, because it can go either way; keep calibration temperature values and re-fit them after quantizing; beware activation outliers — d1-omni's bidirectional encoder + non-Softmax intermediate layers may be more outlier-prone.

## 5. Single-pass latency measurement methodology (input → output, no token generation)

Liquid's own methodology is the template, and it's directly applicable because a decision model **doesn't generate output tokens**:

- **Metric = end-to-end latency from input to output** (they state this explicitly: "Since decision models don't generate output tokens, we measure end-to-end latency, from input to output") `[liquidAI2026opend1]`. There is no TTFT/tokens-s the way you'd measure a generative LLM; instead it's a single call-and-return.
- **Warm calls, one request at a time** (avoid cold-start noise); report **median of 20 runs** in bf16 for the GPU rows `[d1modelcard]`.
- **Warm up the exact shapes you serve.** On NVIDIA, `model.compile(mode="reduce-overhead")` runs single questions as **CUDA graphs** (the RTX 4090's 8 ms row uses it; without it one question is 16 ms). "The first call with a new shape pays for kernel selection or compilation, so warm up the shapes you serve" `[d1modelcard]`. On Jetson the reported numbers are **after warmup** `[jetsond13b]`.
- **Report throughput for packed/batched states separately from single-request latency** (the "64 states, packed" column). The 3.4K-token and image columns isolate input-length scaling. This is the right matrix for a mobile agent: single-request p50 for interaction, throughput for batch triage, input-length latency for state size.

For **our** benchmark we should: (1) define a fixed, realistic prompt/state length and option set; (2) warm up N iterations, discard them; (3) measure wall-clock of one `system_one` call (input bytes → typed answer), median over ≥20 runs plus p95/p99; (4) repeat across quantization levels and devices; (5) optionally use `llama-bench -m model.gguf -p 512 -n 128` as a hardware pre-check, though that measures prefill/decode tok/s for generative models and is only a proxy `[mobileguide]`.

## 6. Integrating into a mobile app / agent loop ("interact on mobile")

**Streaming vs one-shot.** A decision model has **no tokens, so there is nothing to stream** — the natural model is **one-shot**: send state + questions, get typed answers. The only streaming-like concern is *model load / cold start* on first use (load GGUF off disk/mmap) `[mobileguide]`.

**State packing = question batching (this replaces LLM KV-cache reuse).** Because the model reads the state **once** for all named questions (`system_one(state, questions, images, audio)`), the mobile optimization is to **encode the state once and ask every question the agent needs in a single call**, not fire one call per question. Evidence: 3 questions over one state ≈ 1.3× one question `[liquidAI2026opend1]`; the API returns `{answers: {name: answer}, usage: {input_tokens: n, output_tokens: 0}}` `[d1modelcard][d1omnicard]`.

**Multi-question / multi-request batching.** `system_one_batch([(state, questions, …)])` packs many independent requests with no padding — this is the "64 states, packed" throughput column (up to 1 106/s on MI325X) `[d1modelcard]`. For a mobile agent, batch all decisions that can be decided against the same snapshot simultaneously `[batching]`.

**State packing analog for multi-turn.** For a **generative** front-end, State-Pack caches the transformer KV state after the base prompt and only processes new info per step; ~1 000 agents sharing a system prompt compute the base KV once `[statepack]`. For a *decision* model there is no autoregressive decode to cache, but the analogous win is (a) cache the **encoded state** (or a prefix embedding) so a new question set doesn't re-encode a long state, and (b) keep states short (3.4K tokens → 640 ms on M5 Pro) `[liquidAI2026opend1]`.

**Real-time interactive loop is demonstrated.** Liquid's **System One Arcade** runs d1-3B **in a loop over live camera input**, reading answers from **one pass per frame** (gesture games, live content moderation), and an Isaac Sim hardware-in-the-loop demo with the model served on a Jetson `[liquidAI2026opend1][hfblog]`. This is the direct precedent for "interact on mobile": one forward pass per frame/turn, no token streaming.

**Recommended mobile loop:** load model once (mmap) → on each turn, build a compact state (truncate < ~1K tokens when possible) → ask all needed questions in one `system_one` call → branch on typed answers (noul / choice / score) directly in Swift/Kotlin, with a confidence threshold to escalate to cloud if needed (Sapirex suggests routing to a bigger model when top confidence < ~75%; that's a **claim/pattern**, not evidence) `[sapirex]`.

## 7. Realistic on-device memory & battery constraints

**Memory.**
- **d1-3B** BF16 weights ≈ 5.8–6.2 GiB; at 4-bit ≈ 1.6 GiB; Q8_0 GGUF ≈ 2.7–2.9 GB; KV cache is only ~0.50 GiB per 32K-token request in a generative context `[studiotv]` (a decision model needs far less than this since there's no decode growth, though it still pays an attention state over the input).
- **d1-omni-600M** ≈ 587M params → ~0.3–0.6 GB at int4/Q8 → fits in typical phone app budgets (< 1.5–2 GB) `[d1omnicard][sapirex]`.
- **Memory-mapping.** llama.cpp `use_mmap=true` means weights count as file-backed pages, not app RSS — far friendlier to iOS jetsam / Android LMK. Keep `n_ctx` small; KV cache grows linearly with it `[mobileguide]`.
- **Battery / energy.**
  - **On-device LLM inference is ~3× less energy-efficient per token than batched server inference**, and ~88–90% of the *environmental* impact is the device's embodied carbon, not the electricity (over 18 model configs × 2 phones + server) `[batteryprice]`.
  - **Energy-per-token vs quantization bit-width is non-monotonic** — there are "sweet spot" bit-widths on each phone, i.e. the most energy-efficient quant is not simply the smallest `[batteryprice]` (implication: measure energy, don't assume a smaller model saves battery).
  - **8/18 configs sit on the accuracy-vs-energy Pareto front** — allows a battery-aware model router (pick the smallest model that meets accuracy for the task, or use the decision model's own confidence to skip heavy processing) `[batteryprice]`.
  - The dominant energy cost is the **memory-bound decode phase** (relevant to a generative wrapper, but for a one-pass decision model the energy is dominated by reading the whole state once — input-length bound) `[mnnenergy]`.

**Bottom line for mobile:** a decision model's *absence of token decode* removes the dominant mobile energy cost (decode reads weights repeatedly). For a short-state decision call on d1-omni-600M the energy should be roughly a single prefill-scale pass — but this is **inference, not yet measured on our target phone**; and because quant vs energy is non-monotonic, benchmark energy (mAh/decision) empirically `[batteryprice]`.

---

## Coverage status

**Checked directly (primary):** Liquid "Open d1" blog + HF blog + d1-3B/d1-omni-600M model cards (arch, latency tables, Decision Index, precision notes); HF API for d1-3B, d1-3B-GGUF, d1-omni-600M, d1-omni-600M-GGUF (verified format availability and siblings); NVIDIA Jetson AI Lab d1-3B guide; Liquid llama.cpp / mobile / MLX / model-library docs; Apple Core ML Llama research post; NVIDIA NVFP4 blog; arXiv quantization/calibration & battery papers.

**Uncertain / inferred:**
- **No independent Apple-phone (iPhone/Android NPU) latency or battery measurement exists for d1.** The only on-device numbers are Apple **M5 Pro** (a laptop SoC) and Jetson modules. Phone-specific latency/battery is an open measurement task. *(inference / gap)*
- **d1-omni-600M GGUF header reports `context_length=128000` and `causal:false`** while the model card says 16 384 — likely a GGUF header artifact; do not trust the GGUF context field, trust the model card. *(uncertainty)*
- **d1-omni-600M-GGUF publishes only BF16/F16/Q8_0, no Q4_K_M** — if we need <1 GB for phone, we must quantize the 600M ourselves.
- **No Core ML / ONNX / MLX export of d1** exists; those are custom conversion work (effort/feasibility unverified). *(gap)*
- **NVFP4 on d1 and ">98% fidelity at Q4_K_M" (Sapirex)** are **claims, not evidence**; neither is independently benchmarked for d1. *(claim)*
- Battery figures are for generative LLM inference, not one-pass decision models; transferability is an *inference*.

**Not completed:** I could not (a) run d1 on actual Apple/Android hardware (no device access), (b) induce/measure quantization-calibration shift on d1 specifically (no model run), (c) confirm whether llama.cpp's WASM build supports `/v1/systemone` for a browser path.

## Open questions

1. **What base model will we post-train?** d1 paths are d1-3B (LFM2.5-VL-3B) and d1-omni-600M (LFM2.5-Encoder-350M). If we want a small on-device decision model, is the 600M encoder-backbone the right starting point (audio + cheap), or a ~1B decoder for quality footprint balance?
2. **Which runtime for the phone?** Only llama.cpp/GGUF is published. Do we commit to llama.cpp (fast, proven, but no NPU accel on most phones) or invest in a Core ML / ONNX export to hit the Apple Neural Engine / Android NNAPI? This is the biggest remaining engineering fork.
3. **Mobile latency budget:** what's an acceptable end-to-end decision latency on a 2024–2026 phone (e.g. < 20 ms like Jetson Thor, or only < 100 ms)? We have no phone measurement.
4. **Actionable calibration after quantization:** what's the right ECE/Brier threshold for "safe to use on-device," and does d1-omni's per-type temperature calibration survive int4?
5. **Energy measurement:** what's mAh per decision on a phone, and is there a battery-aware router (use d1-omni-600M for most, only escalate to a bigger model when confidence is low)?

## Sources

1. Liquid AI, "Open d1: Edge decision models for text, vision, and audio" — https://www.liquid.ai/blog/d1-open [`liquidAI2026opend1`]
2. LiquidAI/d1-3B model card — https://huggingface.co/LiquidAI/d1-3B [`d1modelcard`]
3. LiquidAI/d1-omni-600M model card — https://huggingface.co/LiquidAI/d1-omni-600M [`d1omnicard`]
4. NVIDIA Jetson AI Lab, "d1-3B" — https://www.jetson-ai-lab.com/models/d1-3b/ [`jetsond13b`]
5. Hugging Face blog, "Multimodal open d1 decision models for the edge" — https://huggingface.co/blog/LiquidAI/open-d1 [`hfblog`]
6. LiquidAI/d1-3B-GGUF — https://huggingface.co/LiquidAI/d1-3B-GGUF [`d1gguf`]
7. LiquidAI/d1-omni-600M-GGUF — https://huggingface.co/LiquidAI/d1-omni-600M-GGUF [`d1omnigguf`]
8. Liquid docs, llama.cpp — https://docs.liquid.ai/deployment/on-device/llama-cpp [`llamacpp`]
9. Liquid docs, iOS & Android (llama.cpp) — https://docs.liquid.ai/deployment/on-device/llama-cpp/mobile [`mobileguide`]
10. Liquid docs, MLX — https://docs.liquid.ai/deployment/on-device/mlx [`mlxdocs`]
11. Liquid docs, Model Library (formats/matrix) — https://docs.liquid.ai/lfm/models/complete-library [`liquidmodels`]
12. NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference" — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ [`nvfp4`]
13. Apple ML Research, "On Device Llama 3.1 with Core ML" — https://machinelearning.apple.com/research/core-ml-on-device-llama [`coreml`]
14. Sajjad et al., "Interpreting the Effects of Quantization on LLMs" — https://arxiv.org/html/2508.16785v3 [`quanteffect`]
15. "Quantization is not systematically destructive" (W8A8 sweep) — https://arxiv.org/html/2509.21173v4 [`quantnotdestructive`]
16. "Quantized Can Still Be Calibrated" — https://aclanthology.org/2025.acl-long.1473.pdf [`quantcal`]
17. "On Calibration of Modern Quantized Efficient Neural Networks" — https://ar5iv.labs.arxiv.org/html/2309.13866 [`quantcnncal`]
18. Coignion, "The Battery Price of Edge AI" — https://arxiv.org/abs/2609.11940 [`batteryprice`]
19. "MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices" — https://arxiv.org/html/2506.19884 [`mnnenergy`]
20. mlc-ai/web-llm — https://github.com/mlc-ai/web-llm (arXiv 2412.15803) [`webllm`]
21. @reeselevine/wllama-webgpu — https://www.npmjs.com/package/@reeselevine/wllama-webgpu [`wllama`]
22. mauludsadiq/State-Pack — https://github.com/mauludsadiq/State-Pack [`statepack`]
23. tryhamster, "Cutting cost by batching parallel AI agent decisions" — https://tryhamster.com/skills/batching-and-parallelizing-decisions [`batching`]
24. Sapirex, "Liquid AI d1 Edge Decision Model Deployment Guide" — https://sapirex.com/en/liquid-ai-d1-edge-decision-model-deployment-guide-implementing-zero-token-routing-with-llama-cpp/ [`sapirex`]
25. StudioTV, "D1 3B VRAM Requirements" — https://studiotvai.com/vram-requirements/d1-3b [`studiotv`]
26. NVIDIA Jetson Thor / Orin / Orin Nano spec pages — https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/ [`jetsonthor`]
27. TypeSafe AI, "Introducing System One Models & Jev" — https://typesafe.ai/blog/introducing-system-one-models-and-jev [`typesafe`]
