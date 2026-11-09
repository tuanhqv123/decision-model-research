# SOTA Deep-Dive: Who Wins AndroidControl, and How

*Created 2026-10-13 from the live leaderboard, the Mem-W paper + cloned repo,
and the UI-Venus technical report. This is the system we must run, then improve.*

---

## 1. Who is SOTA (AndroidControl High v2 leaderboard, May 2026) [sota2androidcontrol]

| Rank | System | Backbone | Pass@1 step acc | Type | Grounding |
|---|---|---|---|---|---|
| **1** | **Mem-W-8B** | UI-Venus-1.5-8B + latent memory | **68.59** | 81.16 | 71.3 |
| 2 | GUI-Libra-8B | (own) | 64.3 | — | — |
| 3 | **Mem-W-4B** | Qwen3-VL-4B + latent memory | **63.07** | 74.62 | 70.85 |
| 4 | GPT-5 + UGround | proprietary | 61.3 | — | — |
| 5 | UI-Venus-1.5-8B (no memory) | UI-Venus-1.5-8B | 61.06 | 74.62 | 61.06 |
| … | Qwen3-VL-32B / GPT-4o+UG / Qwen3-VL-8B | | 59.05 / 57.0 / 54.8 | | |
| … | Qwen3-VL-2B | | 45.48 | 64.07 | 59.64 |

Newer general agent (Aug 2026, not yet on this leaderboard): **UI-Venus-2-9B**
— AndroidWorld **80.2%** (27B: 84.0%) [uivenus22026]; open weights + **GGUF with
vision projector** (runs on this Mac).

**Key observation:** memory (Mem-W) adds **+7.5 pts over its own backbone**
(68.59 vs 61.06). Memory is where the current SOTA is won.

## 2. How the SOTA works (Mem-W, arXiv 2605.09317) [memw2026]

Repo cloned locally: `/private/tmp/pi-github-repos/runtime-lSYFki/.../` (also
`github.com/bingreeky/Mem-W`). Checkpoints: `lunovelle/Mem-W-8B`,
`lunovelle/Mem-W-4B` (full VL model repos, open).

Architecture — **frozen VLM backbone + learned latent memory**:
```
                    memory bank (22,346 web + 4,972 mobile trajectories)
                                 │  top-M retrieval (frozen encoder + faiss/CLIP)
                                 ▼
        past trajectories ──► [ Q-Former compressor ] ──► K latent tokens each
        current episode     ──► [ SAME compressor ]   ──► K latent tokens per W-step chunk
        (expired prefix)                                 (working memory)
                                 │
                                 ▼  woven in front of normal context (source embeddings)
        ┌────────────────────────────────────────────────────┐
        │  [latent memory tokens] + [screenshot + last L steps] │
        │            → FROZEN VLM (UI-Venus-1.5-8B / Qwen3-VL-4B)
        └────────────────────────────────────────────────────┘
                                 │
                                 ▼
                          next action (generated)
```
- Only the **compressor** (Q-Former, K learned queries → K×d tokens) is trained;
  the backbone stays frozen.
- Training: **Stage 1** self-distillation (teacher = same frozen agent with a
  long raw window; student = latent-augmented; action loss + KL). **Stage 2**
  outcome-aware policy gradient on full-episode rollouts (success/fail reward),
  KL-anchored to Stage 1.
- Inference code: `transformers` backend only (README: "can be slow"), deps are
  CUDA-oriented (flash_attn, nvidia-*), faiss + CLIP for retrieval, playwright
  for live web evals.
- **No AndroidControl eval script in the repo** (only GUI-Odyssey / Mind2Web /
  MMINA live evals) — the AC-v2 number comes from the paper (Table 2). We must
  build the offline AC eval ourselves to match the leaderboard protocol.

## 3. Can we RUN it on this Mac? (honest feasibility)

| Path | What | Feasible on M1 Pro 32GB? | Speed estimate |
|---|---|---|---|
| **UI-Venus-2-9B Q4_K_M GGUF + mmproj via llama.cpp (mtmd)** | newer open agent, vision | ok YES — 378GB disk, GGUF+mmproj exist (bartowski) | ~1–3 s/step |
| **UI-Venus-1.5-8B (the SOTA backbone) via transformers + MPS** | exact 61.06 model | ok YES (fp16 ≈ 16GB, fits) | ~2–5 s/step (3–4 h for full 3,051) |
| **Mem-W-4B (lunovelle) via transformers + MPS** | the 63.07 SOTA | warn PORT NEEDED — strip flash_attn→sdpa, MPS backend, build memory bank (we have the mobile train data), write AC eval | slower than backbone (retrieval + compressor) |
| **Mem-W-8B** | the 68.59 SOTA | warn same port, ~17GB fp16, tight but fits | slow |
| vLLM backend (their plan) | — | no CUDA | — |

**Execution tiers (fixed):**
- **T1 — anchor (run NOW):** UI-Venus-2-9B GGUF (llama.cpp) + UI-Venus-1.5-8B
  (transformers MPS) on the AC offline protocol, subset first (~300 steps),
  then full 3,051. Measure step/type/grounding + **output confidence (logprobs)
  → ECE + coverage-risk + latency**. Deliverable: our own "current best"
  numbers on our machine + the SOTA's calibration profile (the paper's hook:
  does SOTA even have honest confidence?).
- **T2 — replicate SOTA:** port Mem-W-4B to MPS; build the mobile memory bank
  from AndroidControl train split (success/fail episodes — raw data, no
  fabrication); write the AC-v2 offline eval; target ≈63.
- **T3 — IMPROVE (our contribution, the paper):**
  - (a) **Trust layer on SOTA:** measure SOTA confidence → show it is
    miscalibrated → add our **calibrated risk-budget gate (act/ask/refuse)**
    → "SOTA + honest dial": at risk ≤ 10%, coverage ≥ X%; or same accuracy at
    lower risk. Nobody has done this on the current SOTA (CAAG = simulated;
    CORA = older VLMs).
  - (b) **Fast cascade:** our 1.2B head handles the high-confidence easy steps
    at ~18 ms; SOTA handles the hard remainder → ≥ same accuracy at 5–10×
    lower average cost, on-device.

## 4. THE EXACT LEADERBOARD PROTOCOL (found it — GUI-Libra repo)

The AC-v2 harness is public: `github.com/GUI-Libra/GUI-Libra` →
`evaluation/offline_evaluation/AndroidControl/` [guilibra2026]. Copied locally:
- `benchmarks/data/acv2/500_steps_filtered.json` — **the 398-step leaderboard subset**
  (350 episodes; actions: 212 click, 87 status, 37 scroll, 27 open_app, 20
  navigate_back, 11 type_text, 4 wait; each row embeds the full **accessibility
  tree** + ground-truth action).
- `benchmarks/data/acv2/eval_official.py` — their scorer (verbatim copy).

**Inputs (native-agent "High" variant):** screenshot + **goal only** (no step
instruction, no a11y tree in the prompt, interaction history empty) +
their 8-type action-space system prompt (Click/Write/LongPress/Scroll/Wait/
NavigateBack/OpenApp/Terminate) + `<answer>` JSON template with
`point_2d` = **pixel** coordinates.

**Scoring (eval_official.py):**
- Spatial steps (click/long_press/type_text): find the **smallest a11y node
  containing the ground-truth point** (the true target element); predicted
  point must fall inside that node's bbox. type_text also needs exact text.
  → element-level, not pixel-level, grounding.
- Non-spatial: all shared GT fields must match (scroll direction, app name…).
- **Equivalences:** click↔open_app (clicked element's text contains app name),
  click↔navigate_back (element text contains "back").
- **Pass@1** = greedy; **Pass@4** = 4 samples, correct if any (eval_best_of_n).

**Decode:** Mem-W paper: T=0, top-p 1.0, max 64 tokens. (GUI-Libra code
default T=0.5 — we run T=0 and note it.)

**P0 (harness validation) = reproduce UI-Venus-1.5-8B ≈ 61.06 within ±3**
on this exact 398-row protocol with our runner (`benchmarks/run_acv2.py`,
writes official-format plan/ans files so their eval.py cross-checks us).

Runner: `benchmarks/run_acv2.py` — backends `hf` (transformers Qwen3-VL + MPS,
full logprobs → calibration) and `llama` (llama.cpp mtmd HTTP). Adds
ECE/Brier/coverage-risk on top of their metrics.

## 5. What "improve" must beat (pass conditions, updated)

- **P0 (anchor):** we reproduce a leaderboard-credible number within ±3 pts of
  the published SOTA row (proves our eval harness is correct).
- **P1 (trust):** SOTA + our gate achieves risk ≤ 10% at coverage ≥ 50%, and
  beats SOTA-alone at the same coverage on executed-success.
- **P2 (speed):** cascade system: same step-accuracy as SOTA-alone (±2 pts)
  at ≥ 5× lower average latency/cost on M1 Pro.
- **P3 (honesty):** SOTA-alone's own confidence is miscalibrated (ECE reported);
  our gate's dial is calibrated (ECE ≤ 0.05 on the same rows).
- **P4 (repro):** one command regenerates everything.

## 6. Assets to download (started 2026-10-13)

| Asset | Size | Purpose | status |
|---|---|---|---|
| `demisama/UGround-Offline-Evaluation` images zip + processed steps | ~0.5 GB | **the 398 shots** (official protocol) | downloading (b9a1e1aad) |
| `bartowski/UI-Venus-2-9B-GGUF` Q4_K_M + mmproj-bf16 | ~6.5 GB | T1 fast path (llama.cpp mtmd) | downloading (bf62800dc) |
| `inclusionAI/UI-Venus-1.5-8B` (bf16) | ~16 GB | T1 exact 61.06 backbone (P0) | downloading (bb96cf077) |
| `lunovelle/Mem-W-4B` | ~9 GB | T2 | pending |
| AndroidControl full test images (13GB) | | T3 cascade data + history | pending (text done) |

## 7. Risks

1. llama.cpp mtmd may not fully support UI-Venus-2's vision stack → fallback:
   transformers+MPS (slow but exact).
2. Mem-W port could burn days (custom compressor + faiss + bank) → T1 results
   stand on their own; T2 is time-boxed.
3. Leaderboard protocol details (exact prompt, token budget, matching
   tolerance) are not fully public → P0 (±3 pts reproduction) is the check
   that we got it right; document our protocol exactly in the paper.
4. All SOTA systems are generative (no native probabilities) → their
   "confidence" = token logprob; our ECE measurement must use the same
   extraction for both sides (fair).
5. **UI-Venus-1.5-8B is a Qwen3-VL-8B fine-tune** (config verified):
   needs transformers ≥ 4.57 (we have 5.19 in `benchmarks/decision-index/.venv`
   + torch 2.14.1 MPS). bf16 ≈ 16GB fits in 32GB.