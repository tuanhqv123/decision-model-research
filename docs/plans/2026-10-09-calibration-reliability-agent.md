# Phase 2 Plan — Calibrated Reliability for Decision Models & Mobile Agents

> **Goal:** Build the missing **calibration-reliability benchmark** (C) and a **calibration-aware-trained decision head** (A) that wins on it under a coverage constraint. Honest framing: we **lose raw accuracy** to large-VLM mobile agents but win **calibrated trust + on-device latency**.
>
> **Specs this implements:** `notes/mobile-agents.md`, `notes/action-selection-feature.md`, `notes/calibration.md`, `AGENTS.md` Phase 2.

---

## 1. Contribution (the defensible claim)

1. **C — a benchmark** for *calibrated* decision-making by decision engines *and* mobile agents: coverage–risk curves, an act/ask/refuse/escalate cascade, cost/latency-adjusted scoring. There is **no standard evaluator** for this today (SafeGround/CORA do grounding-level FDR; OS-Kairos/AgentAbstain do per-step confidence; no unified calibrated-reliability benchmark).
2. **A — a decision head** trained with a calibration-aware objective that (a) matches SOTA on *calibrated trust* and (b) runs on-device in a single pass (zero output tokens). Winning axis = **reliability per unit latency/cost**, not raw accuracy.

**Explicit non-goals:** beat SOTA on raw task success; single-pass *provable* finite-sample calibration guarantees (likely not achievable — flagged as the riskiest claim).

---

## 2. Workstream C — Calibration-reliability benchmark

### Deliverables
- `benchmarks/calib_metrics.py` — ECE (stated binning), Brier + decomposition, coverage–risk (accuracy at coverage c), FDR control procedure.
- `benchmarks/cascade.py` — act/ask/refuse/escalate gate with a utility/cost model (`u_act`, `u_ask`, `u_refuse`, `c_escalate`).
- `benchmarks/task_suite.yaml` — frozen task set + entry points (below).
- `benchmarks/run_baselines.py` — runs a model/engine through the suite, emits coverage–risk + cascade metrics.
- `benchmarks/report.md` — the SOTA baseline measurement (publishable on its own).

### Task set (frozen)
| Domain | Source | Notes |
|---|---|---|
| Typed decisions | Decision Index public suite (110k requests / 42 benches, MIT kit `apolinario/decision-index`) | Run locally or as HF Job; per-model GPU cost |
| Decision calibration | JevBench-style + a curated OOD/refusal set | Coverage for act/ask/refuse |
| GUI grounding | ScreenSpot-Pro; AndroidControl/AITW subset | Grounding-level uncertainty |
| Mobile agent control | AndroidWorld (emulator) — optional, heavier | Reproducible environment |
| On-device | MLX/llama.cpp latency harness on M1 Pro (+ optional real phone) | input→output, no token decode |

### Baseline engines to measure (first honest numbers)
A VLM point-decoder (e.g. UI-R1-style / a small VLM), a decision-model choice head (`decider`/System-One-style over candidates), and an abstain-aware variant (SafeGround/CORA-style thresholding). All go through the same harness.

**Resources (C):** mostly engineer-time + modest GPU for running baselines through Decision Index (see §5). No heavy training.

---

## 3. Workstream A — Calibration-aware decision head

### Deliverable
A small decision head (base TBD: **600M encoder** vs **3B decoder**) that:
- input: GUI/app state (UI XML + screenshot/embeddings) + task + candidate action set,
- output: **calibrated P(action_i)** + confidence, **one forward pass**.

### Method (ranking of effort vs value)
1. **Calibration-aware objective** — train with proper-scoring readout (RLCD-style: score the committed distribution with a strictly proper score; calibrate-then-reinforce), plus option-shuffling and data-shortcut removal (d1 recipe). This is the real novelty over plain SFT/RLVR.
2. **Anchoring / selective prediction head** — optional conformal-style calibration on a held-out set to build coverage–risk curves.
3. **Distil** a reference decision model (teacher-calibration drives student quality) if a strong teacher is available.

### Base-model decision (gates later steps)
| | 600M encoder | 3B decoder |
|---|---|---|
| Train | **Local MLX on M1 Pro, free** | Cloud LoRA (rent) |
| Footprint | ~0.3–0.6 GB → fits phone | ~1.6 GB @4-bit |
| Decision quality | weaker (compares to d1-omni 15.95 DI) | stronger (d1-3B 48.57 DI) |
| On-device | yes, easily | yes, heavier |
| **Recommend** | **600M** for the mobile use case | 3B only if paper needs stronger accuracy |

---

## 4. Task breakdown + timeline

### Phase C (weeks 1–4)
- [ ] **C1 (w1–2):** Define metric + cascade; implement `calib_metrics.py` + `cascade.py`; write unit tests on synthetic data.
- [ ] **C2 (w2–3):** Curate `task_suite.yaml`; pin sources/licenses; download/convert data.
- [ ] **C3 (w3–4):** `run_baselines.py`; run 2–3 baselines through the suite; produce coverage–risk + cascade curves.

### Phase A (weeks 4–8)
- [ ] **A1 (w4–5):** Build state→candidate-decision conversion; pick base (600M/3B) + license check.
- [ ] **A2 (w5–7):** Train calibration-aware objective (LoRA, option-shuffle, shortcut removal); validate calibration on held-out set.
- [ ] **A3 (w7–8):** On-device port (MLX/llama.cpp); measure latency; run through Workstream C harness + Decision Index.

### Phase Paper (weeks 8–12)
- [ ] **P1 (w8–10):** Results tables (coverage–risk, ECE per binning, cost-adjusted).
- [ ] **P2 (w10–12):** Write paper; `verifier` adds inline citations + URL checks.

---

## 5. Real resource requirements

### Hardware we already have
| Resource | Spec | Use |
|---|---|---|
| Mac | **Apple M1 Pro, 10-core, 32GB, Metal 4, 466GB free** | MLX LoRA on ~600M (free, local); on-device latency proxy; llama.cpp/GGUF |
| Stack | Python 3.9.6, torch 2.8.0, uv, ollama | Install `mlx-lm`, `llama.cpp`|

### Hardware to rent (cloud GPU) — only for 3B train + big eval

**Preferred: use free tiers first (budget ≈ $0).**

| Resource | What | Cost |
|---|---|---|
| **Kaggle Notebooks (free)** | T4 16GB, **~30h/week** GPU quota, resume-checkpointing; phone-verified account for internet/HF/git | **$0** — primary for 3B LoRA + full Decision Index run (chunked/resumed) |
| **Google Colab (free)** | T4 16GB (some regions L4 24GB); ~12h sessions | $0 — quick experiments |
| **Our Mac (M1 Pro)** | MLX/Llama.cpp train + on-device latency | $0 |
| **Fallback (only if out of quota)** | RTX 4090 (~$0.59/hr) for 3B LoRA; A100 (~$1.49/hr) for a big eval | ~$2–40 |

> Grounding: LoRA trains ~0.1% of params; QLoRA runs a 3B in ~8GB VRAM and even 7B on a 16GB T4. The full Decision Index (≈132k req) is GPU-hour-heavy but the kit supports resume, so it fits in Kaggle's weekly quota. **Nothing on Colab/Kaggle can measure on-device latency** — that's the M1 Pro or a real phone.

### Data
| Data | Source | License to verify |
|---|---|---|
| Decision Index suite (110k requests / 42 benches) | `apolinario/decision-index` (MIT kit) | suite text from other datasets — respect their terms |
| JevBench / decision sets | public boards | verify |
| ScreenSpot-Pro | public | verify |
| AndroidControl / AITW | public | verify |
| AndroidWorld | Google DeepMind env | verify |
| Mobile-agent task data (curated) | ours | n/a |

> On-device/mobile latency uses the **M1 Pro** (Kaggle/Colab can't do it). If the big Decision Index run exceeds Kaggle's weekly quota, fall back to renting an A100 (~$1.49/hr) for a few hours.

### Personnel
- **1 research lead** (you) owns framing, paper, synthesis.
- **1 ML/modelling** (can be you + subagents for data/lit) — could be solo for MVP.
- Optional **1 edge/mobile** for on-device port + real-device latency.
- Subagents at each stage (researcher for lit/data, reviewer for critique) — already in our toolchain.

---

## 6. Timeline summary

| Milestone | Duration | Notes |
|---|---|---|
| Benchmark spec + harness (C1) | **~2 weeks** | metric + cascade + tests |
| Data + baselines (C2–C3) | **~2 weeks** | first honest SOTA numbers |
| Decision head (A1–A2) | **~3 weeks** | depends on base decision |
| On-device + eval (A3) | **~1 week** | M1 Pro + optional phone |
| Paper (P1–P2) | **~3 weeks** | overlaps |

**Total: ~8–12 weeks (solo) / ~6–8 weeks (1–2 people).**
**Fast path (benchmark-first, publish C alone):** ~5–6 weeks.

---

## 7. Risks & mitigation

| Risk | Mitigation |
|---|---|
| **Riskiest claim** — single-pass guaranteed calibration | Don't claim provable guarantees in one pass. Report coverage–risk curves empirically; use held-out recalibration + calibration-aware loss. |
| We **lose raw accuracy** to big VLM agents | Reframe win as *reliability per unit latency/cost*; be explicit in paper. |
| **Grounding** is the hard part, not the head | If grounding fails without a VLM, consider hybrid: cloud VLM extracts candidate set → local calibrated head scores (CORE-style). |
| Base 600M too weak to be credible | Use 3B if paper needs accuracy; keep 600M for the mobile/product path. |
| License on open weights | Read LFM Open License v1.0 in full before training/distribution; verify each dataset's license. |
| Benchmarks drift / no independent replication | Pin versions; report the specific Mean (82.9 vs 77.1) and flag self-scored numbers. |

---

## 8. Decision points (need your call)
1. **Paper (novel result) or product (working agent)?** → decides how much to invest in benchmark+paper vs product polish.
2. **Base: 600M or 3B?** → **600M** recommended for mobile; **3B** if paper needs accuracy.
3. **Scope of benches:** include AndroidWorld (emulator, heavier) or ScreenSpot/AndroidControl only?

---

*Sources: gathered 2026-10-09 via web search; grounded in `references/`. Cloud/training numbers from compute-market, JarvisLabs, FinLora benchmarking, MLX tutorials, Decision Index kit.*
