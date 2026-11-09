# Decision 2026-10-15 — Build on EmbeddingGemma 2 (optimized recipe)

> Synthesis of 3 subagent audits (emb2-api, aw-facts, novelty-evidence) + G1 empirical (pending).
> HARD-CONSTRAINT respected: heavy compute on Kaggle T4; G1 (light 440M encode) runs locally.

## 1. Base model decision: `google/embeddinggemma-2`
- Multimodal embedding model (text+image+audio+video -> **768-d unified**), **Apache-2.0**, **NOT gated** (no HF token needed).
- 740M total = 270M text core (130M transformer + 140M embedder) + 170M vision + 300M audio.
- **Optimization:** load **text+image only = 440M** via `config_kwargs={"audio_config": None}`. bf16 (never fp16). 
- Loading path: transformers `AutoModel` + `AutoProcessor.apply_chat_template` (works with transformers that has `embedding_gemma2`); sentence-transformers v6.1.0+ also supported.
- Task-prefix for quality: `task: classification | query: ...` (symmetric task). Applies to text; image passed without prefix.
- MRL truncation 768->256 (near-lossless, 3x smaller) optional for on-device.

## 2. Novelty framing (the "ko nhảm" check — critical)
Evidence: none of the 8 competitors (d1, decider, TypedBench, AgentAbstain, THX-01, CORA, CAAG, Re-TRAC) does a **controlled CE-only vs proper-scoring vs joint calibration comparison**. 
- **(a) calibrated single-pass decision head = NOT our differentiator** (d1/decider/THX-01 already do it) -> do NOT claim it.
- **(b) CE vs proper-scoring calibration-objective comparison = GENUINELY UNCONTESTED** -> this is our strongest, most defensible claim (Q2).
- **(c) on-device act/ask/refuse risk gate = open but NOT pristine** (decider runs Apple-Silicon MPS but no risk gate; CORA/CAAG have gate but no on-device). Scope narrowly as "reliability-per-cost of a risk-gated act/ask/refuse on Apple-silicon / no-GPU".
- **Q1** (when calibration breaks under hard/OOD) also unstudied by all 8.

=> **Paper claim = Q2 (calibration mechanism) + Q1 (breakdown regime) + narrow Q3 (on-device risk-gate reliability).** Cite decider (single-pass+on-device MPS+calibrated, no objective comparison), CORA (conformal act/abstain gate, GPU post-hoc), CAAG (act/ask framing, simulated, generative, CE+ECE not proper-scoring), THX-01 (RLCD proper-scoring, no objective comparison, no on-device).

## 3. Benchmark/venue plan
- **AndroidControl** (current, offline single-step, fast): use for **G2 (model) + G3 (Q2 calibration comparison)**.
- **AndroidWorld** (Google DeepMind, ICLR 2025, arXiv 2405.14573): **live Android emulator**, 116 tasks x 20 apps, dynamic instantiation, reward = task completion via device state. Exposes UI elements (a11y/tap targets) -> a decision model can pick from K candidates. Apache-2.0, runs on laptop (2GB/8GB), needs Android Studio emulator (Pixel 6, API 33). Use for **G4 (risk-gate, live act/ask/refuse)** — the ideal venue. **Model trained on AndroidControl transfers** (same action-selector over UI state+goal).
- Feasibility: AndroidWorld setup = heavier (emulator harness); decide after G2/G3 prove the model + calibration.

## 4. Optimized build recipe (applies QUAN L1/L2/L3/L7/L8 + EmbeddingGemma best practices)
1. **Encoder:** EmbeddingGemma 2 text+image (440M), frozen, bf16. Encode state = goal(+image) via AutoProcessor; candidates = action text.
2. **Reader/decision head (train only this):** layer-mix of a few top encoder states (L3) -> choice-over-K logit + P(success) head. Single-pass classification over K (L7), zero output tokens.
3. **Objective (Q2):** CE vs proper-scoring (Brier) vs joint — the mechanism comparison (our novelty).
4. **Cold-start: real acc + ECE/Brier on AC-v2** (G2/G3), then **risk scope**: act/ask/refuse gate (L8) + coverage/risk.
5. **On-device:** MRL truncate to 256d + export (CoreML/ONNX) for the latency/risk claim.

## 5. Status
- [x] Subagent audits (emb2-api, aw-facts, novelty-evidence). 
- [x] **G1 empirical** (46 samples, zero-shot): multimodal top-1 = 17/46 (36.96%) vs text-only 16/46 (34.78%); random=16.7%. Image adds ~nothing at zero-shot (red flag for zero-shot premise).
- [x] **DATA/BENCHMARK LOCKED 2026-10-15** — see §6.
- **[INVALIDATED 2026-10-16]** ~~G2 single-pass choice-over-K head (real acc)~~ — the acc = 0.583/0.5828 numbers were **EXACTLY the majority-class baseline** (always-predict 'click' = 0.5828 on test). Not a real result. See the VERIFICATION GATE in AGENTS.md. Only the calibration/risk-gate caveats remain useful.
- [x] **G3/Q2 calibration comparison** — DONE. Multimodal embeddings WERE encoded on T4 (kernel `embedgemma-curated-t4` COMPLETE, cached to `benchmarks/data/curated/emb/`), so G2/G3 now cover BOTH text and multimodal (see §5c).
- [x] **G4 risk-gate + coverage/risk** — DONE on T4 (kernel `g4-riskgate-t4`). KEY caveat: proper-scoring collapses confidence range [0.602,0.632] -> useless for risk-gating; CE gives a usable coverage/risk tradeoff. See `benchmarks/results/2026-10-16-g2g3-g4-text-multimodal.md`.
- [~] G4 on-device feasibility — embeddings are 768-d (MRL trim to 256 optional); head is tiny (few k params). Not yet exported to CoreML/ONNX; single forward, zero output tokens.

### 5b. Multimodal on T4 — status & blocker (2026-10-16)
- Kernel `trnanh140999/embedgemma-curated-t4` (v8/v9/v10) RUNS but each run re-downloads the 6.4GB
  `android_control_images.tar.gz` + re-extracts, which is very slow (~40+ min, GPU idle during network/extract).
  v9/v10 never produced `results.csv` in a sane window, and `kaggle kernels output` returns empty while RUNNING
  (working dir not synced mid-run).
- **Verified by reading sentence-transformers v6.1.0 source + local test:** multimodal `encode` input accepted
  as `{'text':..., 'image': PIL}` and message-dict forms — all 4 ran (768-d output). So format is NOT the bug.
- **Root cause of slowness = re-downloading/extracting 6.4GB every run.**
- **RESOLVED (2026-10-16):** the v10 kernel actually ran to COMPLETE and produced the multimodal
  + text embeddings (S_*_t/m.npy, 768-d) before its last cell; the multimodal encode DID succeed.
  Those embeddings were cached to `benchmarks/data/curated/emb/` and reused for the train-only
  G2/G3 run (`benchmarks/_g5_train_emb.py`), so no re-download/encode was needed afterwards.
- **Decision (updated):** G2/G3 now have BOTH text and multimodal results from the same saved embeddings;
  multimodal is NOT accuracy-boosting but is calibration-boosting (see §5c).

## 6. DATA + BENCHMARK LOCK (2026-10-15 — resolves the "no data" confusion)
User pushed back: data is abundant, so stop locking a micro subset. **True** — the real cause of the
earlier "overfit" was a wrong interface (choice-over-K on a 232-row a11y subset), not the model.
- **Data = `android_control_high_bbox.json`** (AndroidControl-Curated, bbox-corrected): **8,435 steps**
  with `gt_action`, `instruction` (goal), `step_instruction`, `gt_min_bbox`/`gt_max_bbox`, `gt_coordinate`.
  Every curated image maps to a **local** screenshot (0 missing; 1,543 episodes, 9,987 PNGs). No 6.4GB download on Mac.
- **Benchmark/task L1 = action-type choice** (K=7: click/swipe/open_app/wait/input_text/navigate_back/long_press),
  state = goal(+step_instruction [+image]), frozen embeddinggemma-2 + tiny linear head, zero tokens.
  Episode split (seed 123): train 6,805 / test 1,630. (L2 = grounding E_bbox for the 5,074 spatial rows, next.)
- **L1/Q2 text-state result** (full 8,435; see `benchmarks/results/2026-10-16-l1-q2-fulltext-embedgemma.md`):
  | obj | acc | ECE | Brier |
  | CE | 0.583 | 0.0934 | 0.2296 |
  | proper | 0.583 | 0.0002 | 0.2429 |
  | joint | 0.583 | 0.0034 | 0.2438 |
  → Q2 supported strongly: proper-scoring/joint collapse ECE (0.093 → 0.0002/0.0034, ~−97%) at SAME accuracy.
    Brier of the confidence readout is slightly higher under proper/joint (0.230→0.243) because the readout is
    trained toward calibration (confidence tracks accuracy) rather than minimizing Brier via mis-calibration.

### 5c. G2/G3/G4 results — text vs multimodal (2026-10-16, full curated)
> **⚠️ RETRACTION (2026-10-16):** the accuracy numbers below are **NOT meaningful** — they exactly equal the
> majority-class baseline (always-predict 'click' = 0.5828). The frozen-encoder + single-pass K=7
> classification head was a **degenerate prior-only solution**, and the 'action-type classification' interface
> was the wrong task framing. Only the **calibration (ECE) contrast + risk-gate caveat** are worth keeping;
> the accuracy values must NOT be cited as evidence. See VERIFICATION GATE in AGENTS.md.

See `benchmarks/results/2026-10-16-g2g3-g4-text-multimodal.md` and `2026-10-16-g2g3-text-vs-multimodal.csv`.
Train-only on cached T4 embeddings (768-d), 60 epochs. **Accuracy identical (0.5828) across text and
multimodal** — but this equals the always-click baseline, so it is not evidence of learning. **Multimodal CE
has lower ECE (0.029) than text CE (0.093)** — the image helps calibration, not accuracy.
\n**G4 risk-gate caveat (important for the paper):** proper-scoring made confidence nearly constant
(range [0.602, 0.632], ECE 0.039) = well-calibrated on average but **unusable for per-example risk-gating**
(coverage jumps 1.0 → 0.0). CE (range [0.409, 0.649]) gives a real coverage/risk tradeoff (τ=0.55: cover 54%
at risk 0.28; τ=0.60: cover 16% at risk 0.21). Surface calibration ↔ risk-gating tradeoff honestly.


