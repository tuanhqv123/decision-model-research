# LFM3 + d1-2B — the new Liquid generation (2026-10-28)

Read from primary sources on 2026-10-28: `LiquidAI/LFM3-1.7B-Base` card, `LiquidAI/d1-2B` card
(full), HF blog "Build specialized multimodal models with LFM3", Liquid blog /lfm3, Liquid docs.
Citations: `[liquidAI2026LFM3]` `[liquidAI2026opend1]` `[lfm3vl2b2026]` (see `references/ledger.md`).
**All vendor self-scored; no technical report exists — blog + cards only.**

## What shipped (Oct 27, 2026)

One new backbone, **LFM3-1.7B-Base** (1.73B, 256K ctx, 22T tokens), from which 6 models branch:

| model | in → out | built on |
|---|---|---|
| LFM3-1.7B | text → text | base |
| LFM3-VL-2B (1.92B) | text+image+video → text | base + 164M in-house ViT + 25M projector, 128K ctx |
| LFM3-Audio-2B | text+audio → text/speech | base + audio enc/dec + LiquidCodec |
| LFM3-Omni-2B (2.2B) | everything | base + both encoders |
| **d1-2B (1.92B)** | text+image+video → **calibrated typed answers, 0 tokens** | LFM3-VL-2B |
| d1-omni-2B | +audio → same | LFM3-Omni-2B |

## Architecture (why "break records")

- **Tri-hybrid layer mix** in 24 layers: **13 gated short conv** (kernel 7, up from 3),
  **5 sliding-window attention** (256-token window), **6 global attention**. Conv is cheap
  near-local processing; only 6/24 layers pay full attention.
- **Head-wise gated attention**: sigmoid gate per head (head can switch off → fewer activation
  outliers → better quantization).
- **Partial RoPE**: global layers rotate only the high-frequency half → the other half matches
  on content regardless of distance.
- **Sandwich norm**: normalize block input AND output.
- Design chosen by **hardware-in-the-loop architecture search over 14 device/backend combos**
  (Raspberry Pi 5, phones, laptops, cloud CPUs, GPUs, Qualcomm Hexagon NPU) — not just accuracy.
- Training: 22T tokens; **last 3.5T = mid-training on curated agentic traces + tool calls**,
  which also extends context to 256K.

Result: LFM3-1.7B **beats LFM2.5-2.6B with 35% fewer params**; the record numbers are on
**agentic tasks** (WebArena-Verified **38.42 vs 3.02** for LFM2.5-2.6B; beats Qwen3.5-9B on long
browser/business tasks). Vendor table avg over 29 benches: 49.37 vs Qwen3.5-9B 47.24,
Qwen3.5-4B 39.27, MiniCPM5-2B 39.19, Gemma-4-E4B 31.74. Weak spots: HLE-no-tools 7.17,
RULER 128K 65.56. Edge: ~1GB at 4-bit, decode 1.8–2.8× faster than next model on 3 devices,
128K KV = 1.6GB (vs 4.3GB Qwen3.5-4B).

## d1-2B — the decision model that matters for us

- Replaces d1-3B with **a third fewer params and higher scores everywhere**: Decision Index v0.3
  **51.1** (d1-3B 49.0), ImajevBench **75.5** (64.0), DecisionBench 77.1 (73.1),
  Decision Bench v4 85.2 (83.6), JevBench 77.9 (76.2), vendor board **63.2** (58.8).
- **NEW: `box` / `boxes` question types** — object grounding in one picture, returned as
  picture fractions (top/bottom/left/right) + `confidence`; `found < 0.5` → None.
  Plus `text`/`number`/`date` extraction with spans, `number` estimates with p10–p90 interval.
- Latency: 5.7 ms/decision H100, 3.8 ms MI325X, **M5 Pro mps**: 18.6 ms (1 q), 26.5 ms (3 q),
  45.2 ms (384px image), 53.5 ms (box), **109 states/s batched**; `model.compile()` ~1 min once.
- API: `system_one(state, questions, media)` / `system_one_batch(...)`; `set_size=16` default
  questions per pass; no sampling parameters (single forward pass).
- Deploy: transformers (custom code, `trust_remote_code`), GGUF (llama.cpp support pending
  ggml-org/llama.cpp#31275), MLX 4-bit for LFM3-VL-2B (weights 1.59GB, QAD-trained).
- Caveat from card: **answers get less reliable past ~64K tokens of state.**

## Honest caveats (carry into the paper)

1. **No paper/technical report** — only blog + cards. Numbers = vendor self-scored claims.
2. **The vision headline is thinking-dependent**: LFM3-VL-2B beats LFM2.5-VL-3B 60/62 only with
   reasoning ON; with it OFF: 62.1 avg, wins only 39/62. d1-2B (decision, 0 tokens) has NO
   reasoning — its quality ceiling on visual UIs is unproven anywhere public.
3. **No AndroidControl / AC-v2 number from Liquid for any of these models.** The d1 board is
   their own decision suite (tickets, invoices, objects) — nothing measures d1-2B on mobile GUI
   next-action selection. That is exactly the gap our Phase 0 measurement fills.

## What this changes for us

- Old plan "build grounded next-action decision model from scratch" is obsolete as stated:
  **a single-pass decision model with a grounding primitive already exists and is open**
  (d1-2B, LFM Open License v1.0).
- Our value-add, if we keep it honest: (a) the **first independent AC-v2 number** for d1-2B /
  LFM3-VL-2B; (b) fine-tune for AndroidControl if the zero-shot number is below the 63.07/64.3
  targets; (c) the **act/ask/refuse risk gate** on its calibrated outputs — no vendor number for
  that axis either; (d) on-device latency profile on our Mac.
- Base-model decision for Phase 1 (fine-tune) flips from LFM2.5/EmbeddingGemma to the
  **LFM3 family** (1.7B text for a11y-tree-only variants; LFM3-VL-2B/d1-2B for the visual
  protocol).

## Phase 0 results (2026-10-28)

### A. d1-2B zero-shot, VISUAL protocol (goal + screenshot, box interface) — DONE
`benchmarks/run_d12b_acv2.py` → `benchmarks/results/d12b-acv2-d12b-vision-2026-10-28.md`.
**First independent AC-v2 number for d1-2B** (official scorer, 398 steps, M5 Pro mps):

| metric | d1-2B zero-shot | SOTA range |
|---|---|---|
| step accuracy (Pass@1) | **15.83%** (63/398) | 45.48 (Qwen3-VL-2B) → 68.59 (Mem-W-8B) |
| grounding accuracy (n=165 spatial) | 27.9% | — |
| ECE / Brier | 0.131 / 0.155 | — |
| latency median/p95 (1280px img, mps) | 2.29s / 5.30s | card: 45ms @384px |
| per gold type | click 46/212 · open_app 8/27 · scroll 6/37 · nav_back 3/20 · **status 0/87** · type_text 0/11 · wait 0/4 | — |

Diagnosis: (1) action TYPE is decent (~60%+ on click steps), (2) pixel grounding is weak —
boxes biased to the screen top, wide (27.9% on spatial steps), (3) **task-completion detection
is 0/87** — with goal-only (no history) the model never decides "terminate". Verdict: zero-shot
d1-2B does NOT break any SOTA point on the visual protocol; fine-tuning or a different
interface is required.

### B. d1-2B, A11Y choice-over-K interface (text candidates from the accessibility tree) — in progress
`benchmarks/run_d12b_acv2_ax.py` (`--modality ax`, then `vision_ax`). The decision contract's
native form: goal + labeled element list → CHOICE over candidates + CHOICE over 8 action types.
Candidate picker (bbox-dedup, no cap, systemui filtered) reaches **0.97 coverage** (gold element
present in the candidate list) on the 223 spatial steps — the ~3% is an a11y-data ceiling.
If the choice interface recovers most of the grounding gap, the remaining levers are
task-completion detection (a noul head) and the rare types (status/type_text).

Targets to beat: Qwen3-VL-2B 45.48 · UI-Venus-1.5-8B 61.06 · Mem-W-4B 63.07 · GUI-Libra-8B 64.3 ·
Mem-W-8B 68.59.