# Citation Ledger

Every claim in this project must be traceable to a source below. Rule: **no unsourced claim survives** — if you write a number, an architecture detail, or a comparison in a note/paper, add or point to a ledger entry with a bib key and the exact source.

## Convention

- Each entry = `bibkey` + full citation + URL + the key quotes/facts it supports + retrieval date.
- Notes reference sources by `bibkey` so we can always look back and never lose a citation.
- Keep the `bib.bib` file in sync (this is the machine-readable copy).
- Distinguish **primary sources** (vendor blog / HF cards / arXiv paper) from **secondary commentary**.

## Primary sources (captured verbatim)

### `liquidAI2026opend1` — Liquid AI, "Open d1" (Oct 7, 2026) — PRIMARY
- URL: https://www.liquid.ai/blog/d1-open
- Raw capture: `references/raw/d1-open.txt`
- Released d1-3B and d1-omni-600M (open weights).
- **Decision Index v0.2.1 (public split): d1-3B = 48.57**, ahead of every model < 10B, on par with Decider 35B-A3B (12x size). d1-omni-600M = 15.95.
- Architecture: d1-3B from **LFM2.5-VL-3B** (decoder-only VLM, text+image). d1-omni-600M from **LFM2.5-Encoder-350M** (bidirectional encoder; adds audio FastConformer + vision from LFM2.5-VL-450M).
- d1-3B training: **weight-averaged** LFM2.5-2.6B + text backbone of LFM2.5-VL-3B; fine-tuned seeds/data-mixtures then merged. Long inputs, shuffling answer options, fixing data shortcuts mattered most.
- Latency (single question): RTX 4090 8 ms, Jetson AGX Thor 16 ms, AGX Orin 26 ms, **Orin Nano 50 ms**, Apple M5 Pro 30 ms.
- Linux: llama.cpp native support, NVFP4. Runs Dell/AWS/Qualcomm/NVIDIA.
- Text benchmark mean: d1-3B **82.9**, d1-omni-600M **78.4** (7 public benchmarks: SQuAD2.0, Civil Comments, MASSIVE, PubMedQA, BoolQ, XNLI, PAWS-X).
- Note: Decision Index v0.3 has a private vision split (not reported).

### `typesafe2026systemone` — TypeSafe AI, "Introducing System One Models & Jev" (Sep 15, 2026) — PRIMARY
- URL: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Raw capture: `references/raw/jev-intro.txt`
- **System One Model** = returns type-safe structured values + calibrated probabilities in a single parallel query (no token generation).
- Stack: **new model architecture + parallel sampler + RLCD** (Reinforcement Learning for Calibrated Decisions).
- Jev: unstructured state in → typed probabilistic decisions out. "Can't hallucinate" (schema matching guaranteed; no type errors). Author: Diogo Almeida (ex-OpenAI), SF, founded 2024.
- Speed: **70–500 ms** end-to-end (claims 40x–200x faster); cost input **$0.042 / MTok**, output FREE.
- Workflow evals: claims **193.6x faster, 444.6x cheaper** (ref = avg of GPT-6 Astra + Fable 5.1).
- Naming: Kahneman "Thinking, Fast and Slow"; Jev named after William Stanley Jevons.

## Secondary / literature (papers 2024→now)

| bibkey | Title | Type | Note |
|---|---|---|---|
| `llm2jev` | LLM2Jev: LLMs Are Already Jev-Style Decision Models—When and How to Fine-Tune Them | arXiv | Direct fine-tuning path for "build our own" |
| `jevVsLLMs` | JEV versus LLMs: Accuracy, Cost and Calibration on Seven Political Science Replications | arXiv | Independent calibration/cost evidence |
| `jevScientific` | Jev for Scientific Decisions | arXiv | Semantic decision component eval |
| `searchJev` | SearchJev: A Fast and Calibrated System-1 Model for Search Agents | arXiv | Agent-oriented decision model |
| `jevWikipedia` | Jev (AI model) | Wikipedia | Background facts |

## Full reference aggregation

Each note in `notes/` carries its own numbered source table (bibkey + title + URL + type + confidence). The **lossless** record is:
- `references/INDEX.md` — auto-generated, every source URL from every note (98 unique), by file & line.
- `references/bib.bib` — machine-readable BibTeX for the citable primary/academic sources.
- `references/raw/` — verbatim vendor-page captures (d1-open, Jev intro).

Note → source table mapping:
| Note | Sources it cites |
|---|---|
| `notes/d1.md` | 21 sources, incl. Liquid blog, d1-3B/d1-omni cards, GGUF/w8a8, llama.cpp PR, Jetons guides, backbone repos |
| `notes/jev.md` | 20 sources, incl. TypeSafe blog+docs, independent audits (TrueStandard, DecisionEval, coin-flip, JevAdvBench) |
| `notes/llm2jev.md` | 31 sources, incl. LLM2Jev, OpenJev-RLCD, option-bias, distillation |
| `notes/benchmark.md` | Decision Index (v0.3), BenchLM, reproduction kit, independent eval boards |
| `notes/calibration.md` | 29 sources (DI methodology, ECE/Brier canon, quant calibration) |
| `notes/deployment.md` | 27 sources (GGUF/CoreML/MLX/WebGPU, quantization, battery) |

## Appended sources (from Phase 1 research)

The above `INDEX.md` + `bib.bib` are the append. Key primary anchors already captured verbatim in `raw/`: `liquidAI2026opend1` (d1-open) and `typesafe2026systemone` (Jev intro).

## Open questions carried to Phase 2
- Exact base open model & license terms for fine-tuning (read LFM Open License v1.0 full text).
- Choose d1-3B-style decoder (3B) vs d1-omni-style encoder (600M) as our base.
- Independent (non-vendor) Decision Index / latency replication — currently vendor self-scored.
- Decide mobile conversion path: llama.cpp/GGUF (only published) vs custom CoreML/ONNX/WebGPU.

Sources retrieved: 2026-10-09. Raw captures in `references/raw/`; aggregate in `references/INDEX.md` + `bib.bib`.

## 2026-10-13 competitive sweep (goal repositioning)

Retrieved 2026-10-13 during goal consolidation (`notes/GOAL.md`). Full analysis:
`notes/competitive-2026.md`. Bibkeys added to `bib.bib`:

| bibkey | Source | Type | Key fact |
|---|---|---|---|
| retrac2026 | arXiv:2602.02486 (Microsoft) | primary (preprint) | Re-TRAC: small+mechanism beats 229B on BrowseComp; our claim-structure template |
| typedbench2026 | arXiv:2610.11392 (Oct 2026) | primary (preprint) | class-level calibration benchmark for System One models; takes "first calibration benchmark" positioning |
| agentabstain2026 | arXiv:2607.10059 | primary (preprint) | best of 17 frontier agents: 59.5% on act/don't-act pairs |
| cora2026 | arXiv:2604.09155 | primary (preprint) | conformal risk-controlled mobile GUI shield (post-hoc) |
| caag2026 | IJCOPE 2(5), May 2026, Dutta | primary (journal, low-venue) | act/ask/abstain + CE+α·ECE loss; **simulated results only** |
| seerguard2026 | arXiv:2607.15550 | primary (preprint) | world-model pre-action screening |
| conflictgui2026 | arXiv:2609.03438 | primary (preprint) | conflict-aware GUI termination |
| unicr2025 | arXiv:2509.01455 | primary (preprint) | calibrated P(correct) + risk-budget refusal |
| thx012026 | HAL-X AI, PyPI + HF (Apache-2.0) | primary (vendor, open) | **open RLCD recipe**: proper-score reward (log+spherical+RPS); ECE 0.003–0.066; selective automation |
| decider2026 | Mapika HF + GitHub (open) | primary (independent, open) | full open Jev-repro recipe; **published calibration failure: hard ECE 0.156 vs bar 0.08** |
| thisthat2026 | arXiv:2609.23886 | primary (preprint) | 2B typed decision, 30.9 ms laptop GPU |
| kotobadecisions2026 | github.com/kotoba-lang | primary (open) | Jev-shape on 3 backbones, measured calibration |
| typedlm2026 | neurono-ml.github.io | secondary (docs) | documents class + miscalibration risk |
| conu2024 / conformalabstain2024 / sconu2025 / cic2026 | arXiv/AACL | primary | conformal post-hoc guarantee line |
| probecal2025 | cited via caag2026 | UNVERIFIED - read before citing | tool-use calibration probing |

**Consequence:** our positioning moved from "(C) first calibration-reliability
benchmark + (A) first open recipe" to "Q1 breakdown-regime study + Q2
mechanism comparison + Q3 on-device venue". See `notes/GOAL.md` §8 decision log.

## 2026-10-13 AndroidControl comparison plan

Plan: `docs/plans/2026-10-13-androidcontrol-comparison-plan.md`. Bibkeys added:

| bibkey | Source | Key fact |
|---|---|---|
| androidcontrol2024 | arXiv:2406.03679 (NeurIPS 2024 D&B, Google) | official protocol: Pass@1 step/type/grounding accuracy; 15,283 demos; fine-tuning + data scale is what works |
| androidcontrolcurated2025 | arXiv:2510.18488 | curated Easy/Hard; best models ~60% cap claim |
| sota2androidcontrol | sota2.com leaderboard (accessed 2026-10-13) | "the old" scoreboard, May 2026: Mem-W-8B 68.59, GPT-5+UGround 61.3, Qwen3-VL-2B 45.48 (open-generation, screenshot protocol) |
| limac2025 | ICLR 2025 | LiMAC: small on-device app control (old on-device method) |
| bmoca2024 | arXiv:2404.16660 | B-MoCA mobile benchmark |
| androidworld2024 | arXiv:2405.14573 | interactive emulator benchmark (stretch, needs vision) |

## AC-v2 official protocol (2026-10-13)
| bibkey | claim / asset | source | type | retrieved |
|---|---|---|---|---|
| guilibra2026 | GUI-Libra paper; AC-v2 (398 steps) introduced here; official eval harness (prompt, scorer, data) in github.com/GUI-Libra/GUI-Libra | arXiv:2602.22190 + repo (cloned /tmp/GUI-Libra) | primary | 2026-10-13 |
| ugroundoffline2025 | AC-v2 screenshots (AndroidControl_images.zip) + processed test steps | HF demisama/UGround-Offline-Evaluation | primary | 2026-10-13 |
| memw2026 | Mem-W decode settings (T=0, top-p 1.0, max 64 tok/step); AC-v2 High/Low settings; retrieval M sweep 63.1→70.8 | arXiv:2605.09317 (html, /tmp/memw.html) | primary | 2026-10-13 |
| memw2026-breakdown | **The only public source for the AC-v2-High Type/Grounding split.** Mem-W Table 2, AC-v2-High Acc/Type/Ground: Mem-W-8B 68.59/81.16/71.30; Mem-W-4B 63.07/74.62/70.85; UI-Venus-1.5-8B 61.06/74.62/**61.06** (Ground == Acc → suspected transcription error); base Qwen3-VL-8B 54.80/69.35/67.71; base Qwen3-VL-4B 49.30/63.57/69.06. **Grounding denominator not stated.** GUI-Libra rows are copied from GUI-Libra with `–` for Type/Ground | arXiv:2605.09317 §4 Table 2 (html) | primary | 2026-11-05 |
| guilibra2026-grounding-correction | **Retraction of a claim we previously carried.** GUI-Libra reports **no** AC-v2 Type/Grounding breakdown (Tables 3–4 are Pass@1/Pass@4 only). The "76.2" previously attributed to "GUI-Libra-8B AC-v2 grounding" is GUI-Libra **Table 8: ASFT-3B, reasoning mode, measured on ScreenSpot-v2** (pre-RL). AC-v2 grounding figures from GUI-Libra: none. Do not use 76.2 in an AC-v2 column | arXiv:2602.22190 §6.3 Table 8 (html) | primary | 2026-11-05 |

## Unsloth decision-model training (2026-10-13)
| bibkey | claim / asset | source | type | retrieved |
|---|---|---|---|---|
| unslothdecision2026 | FastDecisionModel/DecisionTrainer recipe; Qwen3.5-4B 75-78% @60 steps, ECE 0.02-0.06; Laya fine-tunable 10min; free T4 notebooks | unsloth.ai docs page (fetched) | primary (vendor docs) | 2026-10-13 |
| typeddecisions2026 | LocalLLaMA/typed-decisions dataset (Apache-2.0, system-one/calibration tags) | HF repo metadata (gated: false) | primary | 2026-10-13 |
## SOTA architecture deep-read (2026-10-14) — notes/sota-arch-deep.md
| bibkey | claim / asset | source | type | retrieved |
|---|---|---|---|---|
| uivenus15 | UI-Venus-1.5-8B: Qwen3-VL-8B fine-tune (Apache-2.0), 4-stage training, AC-v2 High 61.06 (leaderboard), grounding bottleneck, card metadata says "9B params" | arXiv:2602.09082 + HF inclusionAI/UI-Venus-1.5-8B | primary | 2026-10-14 |
| **uivenus15-androidworld** | **The highest published 2B AndroidWorld result: UI-Venus-1.5-2B = 55.6%** (Table 2, "end-to-end models"). Also 8B 73.7, 30B-A3B 77.6; Table 2's neighbouring row prints MAI-UI-2B at 49.1. **Eval step budget NOT STATED; seeds NOT STATED; eval harness not named.** Merge ablation (verbatim reading): 30B-A3B is 71.0 ScreenSpot-Pro / 75.5 AndroidWorld **before** merging; Linear Merge costs −2.9 and −2.3; TIES-Merge gives 69.6 (−1.4) and 77.6 (+2.1). | arXiv:2602.09082 Table 2 + §3.4 (html v2) | primary | 2026-11-09 (A5 T2) |
| uivenus2 | UI-Venus-2-9B: Qwen3.5-9B init, multi-teacher on-policy distillation, keypoint+multi-model-voting reward, GGUF+mmproj vision, AndroidWorld 80.2% (vendor self-scored), OSHarm ASR 11.3% / OSBlind ASR 48.8%, license pending/conflicting, card "1.47M params" metadata bug | arXiv:2609.00028 + HF inclusionAI/UI-Venus-2-9B | primary | 2026-10-14 |
| memw2026 | Mem-W decode T=0/top-p 1.0/max 64 tok; latent compressor + memory bank; self-distill + outcome-aware RL; 68.59 author-reported | arXiv:2605.09317 | primary | 2026-10-13 (ledger) |

## QUAN-Transcribe-1.0 + EmbeddingGemma 2 (2026-10-15)
- **primary**: qnguyen3/QUAN-Transcribe-1.0 model card https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0 + repo https://github.com/qnguyen3/QUAN-Transcribe
  - Single-pass, non-autoregressive (one forward pass, no decoder), CTC reader (6-layer d=768), encoder = EmbeddingGemma 2 audio conformer (top 2/12 fine-tuned). 356M, WER 6.87% avg, 3,713x RT on H100, trained ~3h on 8xH100, streaming 0.48s chunk. **vendor self-scored**.
- **primary**: google/embeddinggemma-2 https://huggingface.co/google/embeddinggemma-2 (Apache 2.0)
  - 740M = 270M text + 170M vision + 300M audio encoders; unified 768-d embedding space; MRL truncation to 128/256/512 (near-lossless to 256); 8K context; task-instruction prefixes; selective encoder loading; bf16 not fp16.
- **lessons note**: notes/quan-transcribe-lessons.md (L1-L10 transfer to our decision head).

## LFM3 + d1-2B release (2026-10-28)
| bibkey | claim / asset | source | type | retrieved |
|---|---|---|---|---|
| liquidAI2026LFM3 | LFM3 family (Oct 27, 2026): one 1.73B tri-hybrid base (13 gated conv k7 + 5 SWA w256 + 6 global attn; head-wise gated attn, partial RoPE, sandwich norm) → 6 models: text 1.7B / VL 2B / Audio 2B / Omni 2B / d1-2B / d1-omni-2B. 22T tokens, last 3.5T mid-train = curated agentic traces + 256K ctx extension. WebArena-Verified 38.42 vs LFM2.5-2.6B 3.02; beats LFM2.5-2.6B with 35% fewer params; ~1GB at 4-bit; hardware-in-the-loop arch search over 14 device/backends. | https://www.liquid.ai/blog/lfm3 + https://huggingface.co/blog/LiquidAI/lfm3 | primary (vendor) | 2026-10-28 |
| liquidAI2026opend1 | d1-2B: 1.92B decision model on LFM3-VL-2B; single forward pass, zero output tokens; NEW `box`/`boxes` question types (object grounding, fractions + `found`<0.5→None) + text/number/date extraction. Decision Index v0.3 51.1 (d1-3B 49.0); ImajevBench 75.5 (d1-3B 64.0); DecisionBench 77.1; Decision Bench v4 85.2; JevBench 77.9; vendor board 63.2 (d1-3B 58.8). 5.7 ms H100 / 3.8 ms MI325X / M5 Pro mps: 18.6 ms 1-q, 26.5 ms 3-q, 45.2 ms 384px img, 53.5 ms box, 109 states/s batched; model.compile() ~1 min. set_size=16 default. 128K ctx, answers less reliable past ~64K tokens. GGUF support pending llama.cpp#31275. | https://huggingface.co/LiquidAI/d1-2B (card) + https://www.liquid.ai/blog/open-d1 | primary (vendor) | 2026-10-28 |
| lfm3vl2b2026 | LFM3-VL-2B: 1.92B = 1.73B LFM3 LM + 164M in-house conv/attn vision encoder (2.2x fewer FLOPs than Qwen3.5-2B ViT) + 25M projector; 128K ctx; first LFM video input (2 fps); thinking mode. 62-bench avg 71.4 vs LFM2.5-VL-3B 57.6 — BUT wins 60/62 only with thinking ON; thinking OFF = 62.1 avg, wins only 39/62 (orcarouter analysis). | https://huggingface.co/LiquidAI/LFM3-VL-2B + https://www.orcarouter.ai/blog/lfm3-vl-2b-vs-lfm2-5-vl-3b | primary (vendor) + secondary analysis | 2026-10-28 |

**Caveats (all three):** no technical report/paper — blog + model cards only; all numbers vendor self-scored; NO AndroidControl/AC-v2 numbers published by Liquid for any d1-2B or LFM3 model (gap = our Phase 0 measurement).

## SOTA training-method deep-reads + data audit (2026-10-28)

### Trend research 2026-10-29: Memory / RL / CoT (deep-read, see notes/trend-memory-rl-cot-2026.md)
| bibkey | claim / asset | source | type | retrieved |
|---|---|---|---|---|
| cotACL2026 | *Does CoT Help Mobile GUI Agents?* (ACL 2026 Findings, BUPT+Tsinghua): empirical study of 6 reasoning vs non-reasoning VLM pairs. Static benchmarks: reasoning ~no/marginal gain (AndroidControl +0.8%; ScreenSpot −29.7%); AndroidWorld: model-specific (Claude +6.3%, Gemini/Qwen3-VL DROP); reasoning & non-reasoning fail on DIFFERENT tasks (T→F 0–36%, F→T 7–11% — gains & losses counterbalance); **token cost 3.11×–14.78×** (ScreenSpot 37.6→238.5); optimal reasoning budget = **128 token** (0: 70.32 → 128: 71.55 → unconstrained 70.41); **most errors due to benchmark** (47.8% weak-eval, 14.9% static-input, 11.9% unclear instr) — supports benchmark-audit thesis; recommends interactive testbeds + adaptive reasoning | aclanthology.org/2026.findings-acl.392 / arXiv 2503.16788 | primary | 2026-10-29 |
| magmaR1-2026 | Magma-R1 (AndroidControl-Curated): long CoT SFT *hurts* grounding; action-aware SFT (reweight action/grounding tokens) + one-stage GRPO + dense 2D-Gaussian grounding reward fixes it. 2.4k curated → 3-4B matches 235B. (alias row exists as magma-r1-2025)| arXiv 2510.18488 | primary | 2026-10-29 |
| guilibra2026 | GUI-Libra (NeurIPS 2026): "standard SFT with long CoT often hurts grounding"; action-aware SFT (ASFT, mix reasoning-then-action + direct-action, token reweight α_a/α_g) + conservative GRPO with KL trust region (β=0.001) to stabilize partial verifiability; success-adaptive negative-gradient scaling. AC-v2-High: GUI-Libra-3B 57.3 / 4B 62.3 / 7B 59.3 / 8B 64.3; Grounding (High) GUI-Libra-8B 76.2, 4B 74.0; AndroidWorld (20 steps): 3B 25.2 / 4B 42.6 / 8B 42.6. Confirms CoT-grounding tension + KL importance | arXiv 2602.22190 | primary | 2026-10-29 |
| memw2026 | Mem-W (LV-NUS, Shuicheng Yan): latent-memory-native agents; trajectory-to-latent compressor maps obs-action segments to fixed latent blocks woven into policy embedding (not text summaries); self-distillation + outcome-aware supervision; gains up to +30 on 4 nav benchmarks. Best fit for our latent-memory direction over text-summary (which we tested at −6.0) | arXiv 2605.09317 | primary | 2026-10-29 |
| mementogui2026 | MementoGUI: learned working + episodic multimodal memory; **ablation: removing ROI reference images from working memory drops AMS 67.69→64.22** (visual grounding in memory is NOT redundant with text summary) | arXiv 2605.18652 | primary | 2026-10-29 |
| comem2026 | CoMEM / Auto-scaling Continuous Memory (UCSD): compress each GUI trajectory into fixed-length **8 embeddings** (VLM-as-encoder), sharply cut context while keeping fine-grained visual cues (widget size/pos); limitation: retrieval drifts under extreme UI shifts | arXiv 2510.09038 | primary | 2026-10-29 |
| focusmem2026 | FocusMem: factorize content / readout / trust in latent GUI memory; one fixed memory block per trajectory loses detail + serves different roles; separates episodic vs working memory | arXiv 2608.04530 | primary | 2026-10-29 |
| androtmem2026 | AndroTMem: anchored memory using compact causally-linked intermediate-state anchors for subgoal-targeted retrieval; long-horizon diagnostics (AndroTMem-Bench 1,069 tasks / 34,473 steps) | arXiv 2603.18429 | primary | 2026-10-29 |
| memguibench2026 | MemGUI-Bench: memory-centric mobile GUI benchmark — only **5.2–11.8%** of tasks in existing benchmarks are memory-related, none cross-session → memory is under-measured | arXiv 2602.06075 | primary | 2026-10-29 |
| uiagile2026 | UI-AGILE: RL is bottlenecked by sparse/ineffective reward (binary correct/incorrect gives no learning signal on complex interfaces) + precise inference-time grounding | arXiv 2507.22025 | primary | 2026-10-29 |
| hyperclick2026 | HyperClick: self-critiqued RL adding a **confidence-alignment reward** to correctness reward, training policy to output correct + calibrated | arXiv 2510.27266 | primary | 2026-10-29 |
| taskstate2026 | Task-State Representation (TSR, training-free): decouple persistent task state from transient screen obs to avoid forgetting/hallucinating/stale interfaces in long-horizon | arXiv 2607.00502 | primary | 2026-10-29 |
| memguia2026 | MemGUI-Agent: end-to-end proactive context management; TSR 5.56→72.22 with task-specific experience (long-horizon consistency) | arXiv 2606.19926 | primary | 2026-10-29 |
| whatmemory2026 | *What Memory Do GUI Agents Really Need?*: passive records → active task-driving states; "retrieving a value does not reveal its current role in the workflow" | arXiv 2606.31612 | primary | 2026-10-29 |
|---|---|---|---|---|
| magma-r1-2025 | Magma-R1 (AndroidControl-Curated paper): single-stage GRPO, NO SFT; dense 2D-Gaussian grounding reward exp(−‖c−p‖²/2σ²) ("binary hit/miss = no effective learning signal"); action-type proportional (stratified) sampling vs long-tail P_orig; 2.4k curated samples, 60h H20 ≈ $60, matches Qwen3-VL-235B (AC-Curated-Hard: Type 84.2 / Grounding 84.8 / SR 75.3 vs 88.2/83.6/76.5); ~30% of AndroidControl has ambiguities/factual errors, "benchmark was actively teaching incorrect behaviors", larger raw data reduced accuracy; paper says 3B MobileVLM base but released checkpoints are 4B Qwen2.5-VL — unresolved mismatch | arXiv:2510.18488 + github batwBMW + HF batwBMW | primary | 2026-10-28 |
| mobilerl2025 | MobileRL: reasoning-free SFT (97.9k steps, 2 epochs, LR 1e-5→1e-6, Llama-Factory) → reasoning SFT (23.6k) → AdaGRPO (SPA shortest-path reward, AdaPR positive replay, FCF failure curriculum; KL β=0.001, n=16, LR 1e-6). Stage gains Qwen2.5-VL-7B on AndroidWorld: base 27.6 → +RFSFT 50.2 (+22.6) → +RSFT 56.6→56.8 → +AdaGRPO 72.0. Online RL cost: hundreds of Dockerized AVDs, >10 min/step, 100 steps >25h, 4×1TB RAM machines — NOT feasible for us (no emulator) | arXiv:2509.18119 | primary | 2026-10-28 |
| acv2-completion-audit | Local data audit (this session): (a) android_control_train.jsonl 12,226 trajectories have NO success flag, NO terminate/status action (last actions of 500 sample: click 342/wait 81/scroll 68); (b) curated 8,435 has no status class; (c) smolagents/android-control HF schema (episode_id, goal, screenshots_b64, actions, step_instructions) also has NO success flag; (d) in the 398-step AC-v2 eval, **85/87 status rows are the LAST step of their episode** (350 episodes; 0 episodes with >1 status row; status rows: step_instruction="terminate", goal_status="successful") → completion labels are synthesizable by position (last step → complete=1) | local: benchmarks/data/acv2/500_steps_filtered.json + benchmarks/data/android_control_train.jsonl + HF smolagents/android-control metadata | primary (our verification) | 2026-10-28 |
| trainanalysis2026 | Synthesis: "just train" is NOT sufficient; conditions C1 curate (20–30% noise) / C2 balance / C3 synthesize completion / C4 objective-matches-primitive / C5 conservative+calibrated; plan A(data prep)→B(two-stage zero-shot 398)→C(learnability probe, gate: ≥32% AND status recall ≥30%)→D(full fine-tune, target ≥45.48); RL deferred | notes/training-analysis-2026-10-28.md | ours | 2026-10-28 |
| acv2-datamodel-audit | AC-v2/UGround data model (verified 2026-10-28): T-action trajectory has T+1 screenshots; screenshot_i = pre-action state for action i; screenshot_T = final state after last action; eval status screenshots = state after last UI action (ep=4114: action step4→status img screenshot_5); visual check of 3 final screens shows goal achieved; completion set = 1,242 rows built locally (android_control_completion.json), 10 length-mismatch episodes excluded, zero overlap with curated; a11y noise: 100% gold in some node, 79.7% labeled, 74.5% in ≥2 overlapping nodes; a11y bbox field = bbox_pixels (dict) not bbox | local: benchmarks/data/acv2/500_steps_filtered.json, acv2_shots/images (6.4GB UGround), visual inspection | primary (our verification) | 2026-10-28 |
| d12b2s-full-v-2026 | d1-2B two-stage zero-shot full AC-v2 on Kaggle T4 (2026-10-28): latest run 73/398 (18.34%), prior run 61/398 (15.33%) — bf16 T4 nondeterminism ±~3pts; completion recall 10/87 (mean p_done status 0.237/med 0.187 vs non-status 0.123/med 0.073); false-completion 7/311 @thr0.5; eval-fit thr sweep best ≈0.10 → 25.6% (analysis only, overfit); ECE(action-prob) 0.362, Brier 0.280; 51/398 conf=None box-not-found bug (fixed); zero-shot two-stage does NOT break any SOTA point → completion-head training mandatory | benchmarks/results/2026-10-28-d12b2s-full-vision-twostage.md + d12b2s-full-v-2026-10-28.jsonl + kaggle kernels logs trnanh140999/d12b2s-acv2-kaggle | ours (measured) | 2026-10-28 |
| decisionfield-update-2026 | Decision-model field 2026-10-28 re-verified: JevBench 232 open systems; ImageJevBench 64 image models; Jev DI Vision 39 models (top: amos-1-large 72.4 ECE 0.024; small: d3-nano-2B 54.7, Intern-Decision-2B 47.8 ECE 0.032, RSI-Jev v5.0-VL-3B 57.3 ECE 0.023); d1-2B/d1-3B absent from vision board; new entrants: Typical 1.7B/4B (Oz Labs/CSIRO, text-only, built-in abstain, noul label-invariant), Strands Decider 2B (AWS, Qwen3.5-2B torso + 1M pointer head + LoRA r16, text-only, open data+scripts), CoPE-B-S1 (Zentropi safety), GLiNER2.5-Decide 340M (Fastino), ej 11MB; Mapika decider-2b-vision (Qwen3.5-2B VLM, image + lettered options → calibrated probs, Apache-2.0, NO box primitive, hard-ECE 0.18-0.31). d1-2B's unique AC-v2 asset = native box/boxes grounding; de-risking test = decider-2b-vision zero-shot on 398 ax protocol before fine-tune | benchmarkheaven.com/jev-models + /image-jev-bench, metatext.io Jev DI Vision, typical.ozlabs.ai, strandsagents.com blog, HF Mapika/decider-2b-vision + Mapika/decider-2b v12 | primary (fetched) | 2026-10-28 |

## Grounding-precision deep-reads 2026-11-02 (via Feynman alpha, alphaXiv)
| bibkey | claim / asset | source | type | retrieved |
|---|---|---|---|---|
| v2p2026 | V2P: Visual Attention Calibration for GUI Grounding via Background Suppression and Center Peaking (Zhejiang Univ + Inclusion AI/Ant Group). Diagnoses 2 failure modes: **background distraction** (attention leaks to non-target) + **center-edge confusion** (uniform target modeling). Fixes = training-time: (a) Suppression Attention (L_Sup_Attn, weighted penalty on attention mass over non-target patches), (b) Fitts-Gaussian Peak Modeling (L_Action_Attn, KL to 2D Gaussian centered on target centroid, **σ ∝ element width/height**, σ_factor=1.0, lr=5e-6). On GUI-Actor, V2P-7B: ScreenSpot-Pro 44.6→**52.5**; ablation: remove FGPM −5.0, remove SA −3.2 (total −8.2). Attribution: 50.5% of gains = background distraction, 35.7% = center-edge confusion. Small-element gain **+10.0**. Sensitivity: ScreenSpot-Pro peaks at σ=1.0. Model trained Qwen2.5-VL-Instruct 7B/3B. Relevance: our `train_d1.py` box loss uses fixed σ=4 and NO suppression — should scale σ with element size + add suppression | arXiv 2508.13634 / 2601.06899 | primary | 2026-11-02 |
| conclick2026 | ConClick: Element-Wise Contrastive Learning Improves GUI Grounding (RUOTIAN ZHANG et al.). Systematically evaluates 10 grounding models, 5 datasets, 161,640 predictions. **Key finding: 74–88% of incorrect clicks hit a non-target ELEMENT (not background)**; far-element is the dominant error category; different models repeatedly select the **same distractor** (55.5–79.1% agreement). Method = element-wise contrastive: build target–distractor pairs (4,292 in CONCLICK-DATA, distractor = GUI-Libra-8B / GTA1-7B mis-clicks matched to OmniParser-v2 elements), add listwise ranking + pairwise margin selection loss on top of coordinate loss (contrastive operates on attention-derived region-selection scores). Results: Qwen3-VL-8B ScreenSpot-Pro 54.71→**66.03**; MMBench-GUI L2 83.72→86.28; OSWorld-G 63.33→70.20. Inference-time RECLICK (revisit + cross-view support) → ScreenSpot-Pro **72.36**. Relevance: error profile = exactly our measured E2 miss pattern (66% far-element); but operates on attention scores → need to check d1-2B readout feasibility (slot-logit vs attention map) | arXiv 2610.13268 | primary | 2026-11-02 |
| safeground | SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration (QINGNI WANG, YUE FAN, XIN ERIC WANG). Uncertainty-aware GUI grounding: distribution-aware UQ captures spatial dispersion of stochastic samples -> captures spatial spread; calibration derives a test-time threshold with statistically guaranteed FDR control. Applied on ScreenSpot-Pro; improves system-level accuracy by up to 5.38% pp over Gemini-only inference. **Relevance: the risk-controlled deferral/FDR idea is a calibration-first framing analogous to our act/ask/refuse gate; supports 'risk control needs calibration' (Q1/Q3)** | arXiv 2602.02419 | primary | 2026-11-02 |
| seg2click2026 | Seg2Click: Learning Pixel-Level Action Masks for GUI Visual Grounding (East China Normal Univ). Argues **single-point coordinate supervision is flawed**: many valid click points inside a target, single-point penalizes any deviation regardless of staying inside; visual-token footprint can span target+neighbor+background but gets one relevance score (granularity bottleneck). Method = box-supervised **sub-token action-field prediction**: query-conditioned pixel-shuffle decoder outputs an executable-region mask at processed resolution; **centroid of largest connected component = click**; no verifier/zoom. Results: Seg2Click-9B 92.06 / 94.10 / **55.22** (ScreenSpot / v2 / Pro); 0.8B beats coordinate generation by +24 correct clicks and GUI-Actor by +4.87 on ScreenSpot-Pro. Relevance: d1-2B's box slot *could* supervise the whole executable region (Seg2Click-style) instead of collapsing to center-point | arXiv 2610.16234 | primary | 2026-11-02 |
| magmar1dense2026 | (cross-ref) dense 2D Gaussian grounding reward exp(−‖c−p‖²/2σ²), "binary hit/miss = no effective learning signal"; ~30% of AndroidControl has ambiguities/factual errors | arXiv 2510.18488 | primary | 2026-11-02 |

## Value-brief sources (2026-11-07)
Added for `outputs/calibrated-decision-value.md`; every entry below was fetched and read during the citation
pass, and the claims it supports are stated in the brief.

| bibkey | claim / asset | source | type | retrieved |
|---|---|---|---|---|
| mammo-triage-2026 | Prospective **paired within-woman** noninferiority study, n = 31,301: workload −63.6%, CDR +15.2% (95% CI 6.6–24.4), recall +14.8% (9.0–20.6) and **not noninferior** — the strongest outcome-level abstention result found; direction is inverted (machine withholds escalation to the human) | Nature Medicine 2026, https://www.nature.com/articles/s41591-026-04277-x | primary, peer-reviewed | 2026-11-07 |
| routellm2025 | Router decides before any call; 3.66× cost saving at 95% GPT-4 quality on MT-Bench, 1.41× @92% MMLU, 1.49× @87% GSM8K | ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/file/5503a7c69d48a2f86fc00b3dc09de686-Paper-Conference.pdf | primary, peer-reviewed | 2026-11-07 |
| tmlr-criteria | "Are the claims … supported by accurate and convincing evidence?" is the first criterion; no-SOTA and no-novelty are explicitly forbidden as rejection reasons; the remedy for an overclaim is to reduce the claim | https://jmlr.org/tmlr/acceptance-criteria.html | primary, venue policy | 2026-11-07 |
| icml-reviewers | "Soundness is distinct from impact … even if its contributions are modest or incremental" | https://icml.cc/Conferences/2026/ReviewerInstructions | primary, venue policy | 2026-11-07 |
| neurips-checklist | Reviewers are instructed not to penalise honesty about limitations; "no"/"n/a" is not grounds for rejection | https://neurips.cc/public/guides/PaperChecklist | primary, venue policy | 2026-11-07 |
| uist-authors | "User Studies Are Not Required … papers must support the claims they make with evidence"; threshold benchmarking is an accepted strategy | https://uist.acm.org/2026/author-guide/ | primary, venue policy | 2026-11-07 |
| apple-hig-ml | Act / ask / withhold with confidence thresholds; "verify that your confidence values correspond to the quality of your results"; balance benefit against cost of correction. **No numeric threshold published** | https://developer.apple.com/design/human-interface-guidelines/machine-learning | primary, vendor guidance | 2026-11-07 |
| google-gemini-ask | Enumerated always-ask action classes (communications, data deletion, purchases, financial transactions, web-form submission). **No confidence threshold or measured ask rate published** | https://support.google.com/gemini/answer/16596215 | primary, vendor policy | 2026-11-07 |
| elyaniv-wiener-2010 | Coverage Φ = E[g(X)] and selective risk R = E[ℓ·g]/Φ — the two primitives every later metric summarises | JMLR 11:1605–1641, https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf | primary, peer-reviewed | 2026-11-07 |
| traub-augrc-2024 | AURC/AUGRC reorder the leaderboard on 5 of 6 datasets — the selective metric is not decision-stable | NeurIPS 2024, https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf | primary, peer-reviewed | 2026-11-07 |
| conformal-beyond-exchange | Coverage ≥ 1−α−Σw·d_TV under shift: assumption-free in form, unusable in practice (TV term not estimable in production) | arXiv:2202.13415 (Ann. Statist. 2023) | primary, peer-reviewed | 2026-11-07 |
| crc-2024 | Finite-sample control of expected loss for monotone losses; the guarantee is over the calibration draw and the test point, i.e. marginal and one-shot | arXiv:2208.02814, ICLR 2024 | primary, peer-reviewed | 2026-11-07 |
| agentabstain-2026 | 263 paired tasks / 42 sandbox envs / 8 scenarios; 17 frontier LLMs in 4 harnesses; best 59.5% paired accuracy — abstention capability is largely independent of task capability | arXiv:2607.10059 | primary, preprint | 2026-11-07 |
| typedbench-2026 | Realised cost under asymmetric cost matrices — the only calibration→cost experiment found (non-GUI decision tasks) | arXiv:2610.11392 | primary, preprint | 2026-11-07 |
| maiui2026 | MAI-UI (Tongyi Lab, Alibaba). Qwen3-VL backbones 2B/8B/32B/235B-A22B. Two-stage: self-evolving SFT data pipeline (manual-driven task generation, human + model trajectory synthesis, iterative MLLM-judge rejection sampling, failed trajectories reused as their longest correct prefix) then online agentic RL (GRPO, group 16, clip_low 0.2 / clip_high 0.3, no KL, action-repetition penalty, experience replay of last 8 successful trajectories, difficulty-stratified curriculum) on containerised AVDs. ACTION SPACE (their Table 1, 12): click, long_press, type, swipe, drag, system_button(back/home/menu/enter), wait, terminate, answer, ask_user, mcp_call. SCALE: 512 concurrent env instances across 10 ECS servers; 50-step horizon. ABLATIONS: parallel envs 32->512 = +5.2 pts; step budget 15->50 = +4.3 pts; device-cloud collaboration +33% on-device. ANDROIDWORLD (Table 6 + Table 9): 235B-A22B-RL 76.7, 32B-RL 73.3, 8B-RL 70.7, **2B-RL 49.1**; **SFT-ONLY rows (Table 9): 2B-SFT 45.1, 8B-SFT 64.7, 32B-SFT 69.8** -> online RL gain +4.0 / +6.0 / +3.5. Baselines: Qwen3-VL-2B 36.4, ScaleCUA-3B 23.7, Ferret-UI-Lite-3B 28.0, UI-TARS-2 73.3, Gemini-2.5-Pro 69.7. CRITICAL for us: 2B-SFT 45.1 e2e implies **92.3% per-step**, so RL is worth only +0.8 per-step points at 2B — the SFT supervision, not the RL, is what separates MAI-UI from a single-step-trained model. Agent-user interaction worth +32.1 absolute (37.5% vs 5.4%) on their interaction subset. Weights Apache-2.0 on HF (Tongyi-MAI/MAI-UI-2B, 4.25 GB). | arXiv:2512.22047 | primary (read 2026-11-08) | 2026-11-08 |
