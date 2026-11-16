# Trend 2026: Memory, RL, CoT for GUI Agents — Pros/Cons (tied into the project's decision)

> Date: 2026-10-29. Compiled from primary sources (arXiv/ACL), every claim has a bibkey. Purpose: answer the user's question
> — "are CoT / RL / memory really suitable, pros/cons?" — to decide the build direction for
> d1-2B (the project's single-pass decision model). No guessing, only what can be verified.

---

## 1. CoT / Reasoning — CONCLUSION: NOT worth investing in for our axis

### Most important source: "Does Chain-of-Thought Reasoning Help Mobile GUI Agents?" (ACL 2026 Findings, Zhang/Gao/Xu, BUPT+Tsinghua)
This is the **first empirical study** (6 VLM pairs: Gemini, Claude, AgentCPM-GUI, GLM-4.5V, Qwen3-VL-4B/8B) on
ScreenSpot + AndroidControl (static) + AndroidWorld (interactive).

**Main results — very important for the project:**
1. **Static benchmarks: reasoning barely improves, even lowers.**
   - AndroidControl: Gemini Thinking is only 0.8% higher (54.4%).
   - ScreenSpot: drops by up to **29.7%** in many setups.
2. **AndroidWorld (interactive): model-specific.** Claude Thinking +6.3%, Gemini/Qwen3-VL **drop**.
3. **Reasoning and non-reasoning FAIL on DIFFERENT tasks** — the gains and losses **cancel each other out**.
4. **Token cost rises 3.11× → 14.78×** (ScreenSpot: 37.6 → 238.5 tokens) — raises latency + money, with no clear gain.
5. **A small reasoning budget (128 tokens) is best**; more gives **diminishing returns** (76.83 → 77.81 (128) → 76.90 (unconstrained)).
6. **Most errors come from the benchmark** (47.8% weak-evaluation; 14.9% static-input) **not from the model** — supports the
   project's thesis: static benchmarks have errors, do not blame the model.

### Magma-R1 (arXiv 2510.18488) — supporting: CoT can LOWER grounding
The paper says explicitly: long CoT SFT *hurt* grounding. Their solution is **action-aware SFT** (reweight tokens toward
action/grounding) + RL. → That means **CoT is not a "compass"**, it even needs to be corrected.

### GUI-Libra (arXiv 2602.22190) — confirms the same thing
- "standard SFT with long CoT often **hurts grounding**."
- ASFT + conservative GRPO (KL trust region) is what preserves grounding when CoT is present.

### 🎯 Pros/Cons of CoT for the PROJECT
| Pros | Cons |
|---|---|
| May help hard / multi-step tasks (Claude Thinking +6.3% AndroidWorld) | **Raises tokens 3–15×** → breaks our "0-token, low latency" axis |
| Can be sustained with a small budget (128 tokens) | Long CoT **lowers grounding** (Magma-R1, GUI-Libra) |
| — | Results are **model-specific**, inconsistent |
| — | Reasoning/non-reasoning fail on different tasks → not a "general solution" |

**CONCLUSION FOR THE PROJECT:** CoT is **NOT suitable** for our "calibrated single-pass 0-token" axis. It breaks
latency, lowers grounding, and is inconsistent. Our axis (direct decision, reading logits, calibrated
probabilities) is **stronger** on latency/calibration — and the ACL 2026 evidence shows that **reasoning is not a
necessary condition to be good at GUI**.

---

## 2. RL — CONCLUSION: WORTH INVESTING IN, this is the path with the most evidence

### Trend 2026: RL is the main "engine" of GUI agents
| Paper | How RL is done | Result |
|---|---|---|
| **MobileRL** (ICLR 2026) | online AdaGRPO, sparse terminal reward, difficulty-adaptive replay/filter/spa | 27.6 → 50.2 (SFT) → 56.8 (reason SFT) → **72.0** (RL) |
| **GUI-Libra** (NeurIPS 2026) | conservative GRPO + **KL trust region** (against partial verifiability) | base 30.4 → **42.6** AndroidWorld |
| **Magma-R1** (AC-Curated) | one-stage GRPO, **dense 2D-Gaussian grounding reward** | 2.4k curated → 3-4B = 235B |
| **UI-AGILE** (arXiv) | RL + inference-time grounding; names the problem of "ineffective/sparse reward" | — |
| **HyperClick** (arXiv) | Self-critiqued RL: correctness + **confidence-alignment reward** | — |

### What the successful RL papers share (important):
1. **The reward must be DENSE, not binary.** Magma-R1: `exp(−‖c−p‖²/2σ²)` in place of hit/miss. UI-AGILE/MobileRL also
   complain that "sparse reward = no effective signal".
2. **Partial verifiability is a big problem.** GUI-Libra: many actions are right but only 1 is verified → needs a **KL trust
   region** (β=0.001) to stabilise.
3. **Class imbalance.** Magma-R1: stratified sampling by action type.
4. **RL is MOST effective after a good SFT.** MobileRL: SFT → +22.6, RL → +15.2. GUI-Libra: base 3.5 → ASFT 13.0 → +RL 25.2.

### 🎯 Pros/Cons of RL for the PROJECT
| Pros | Cons |
|---|---|
| Strongest evidence for raising **grounding** (Magma-R1: 2.4k → 235B) | **Online RL needs an emulator + large compute** (MobileRL: 2k tasks, many AVDs) |
| GUI-Libra does "offline RL" (no online data needed) → still +12.2 | **Reward design** is hard (dense, handling partial verifiability) |
| Can raise **both AC-v2 and AndroidWorld** | Risk of overfit / losing calibration (RLVR inflates confidence to ~0.99) |
| Fits **closing our grounding gap 56.5 → 70+** | — |

**CONCLUSION FOR THE PROJECT:** RL is **the most worth investing in** — especially a **dense grounding reward** (Magma-R1) + a **KL trust region**
(GUI-Libra) against partial verifiability. This is the path with evidence for raising AC-v2 and AndroidWorld.

---

## 3. Memory — CONCLUSION: WORTH doing, BUT the right mechanism must be chosen

### Trend 2026: from text-summary → latent/continuous memory
| Paper | Mechanism | Score |
|---|---|---|
| **Mem-W** (2605.09317, LV-NUS) | **latent memory token** (trajectory-to-latent compressor) — memory native in the embedding | +30.0 |
| **MementoGUI** (2605.18652) | working + episodic memory, multimodal (ROI image) | UI-Venus-1.5-8B: AMS 54.58→67.69 |
| **CoMEM / Auto-scaling** (2510.09038) | continuous memory, 8 embeddings/traj (keeps visual cues) | shrinks context, keeps the image |
| **FocusMem** (2608.04530) | factorize content/readout/trust in latent memory | — |
| **AndroTMem** (2603.18429) | anchored memory (intermediate-state anchors) | — |
| **MemGUI-Agent** (2606.19926) | proactive context management | TSR 5.56→72.22 |

### Important findings from ablations:
- **MementoGUI**: removing the ROI image from working memory → AMS 67.69→64.22. → **Visual grounding in memory is
  necessary, not just a text summary.** (This CONTRADICTS what we tried with TF-IDF text memory — it explains why
  we got **−6.0**.)
- **Benchmarks under-represent memory**: MemGUI-Bench has only 5.2–11.8% memory-related tasks → **memory is not yet
  measured correctly** in current benchmarks.
- **From passive records → active task-driving states** (2606.31612): "retrieving a value does not reveal its current
  role in the workflow" — memory must know the state, not only store it.

### 🎯 Pros/Cons of memory for the PROJECT
| Pros | Cons |
|---|---|
| **Needed for multi-step** (AndroidWorld, long-horizon) | Our single-pass model needs a **change of mechanism** — it is not simply adding a prompt |
| **Latent/continuous memory** keeps visual cues, shrinks context (good for on-device latency) | Complexity: needs a **trajectory compressor / encoder** |
| Strong evidence: Mem-W +30, MementoGUI +13 AMS | **Memory must be trained** (self-distillation, outcome-aware supervision) |
| — | Pure text memory (like our TF-IDF) is **not enough** — we tested it, −6.0 |

**CONCLUSION FOR THE PROJECT:** Memory is **worth doing** but **must be latent/multimodal**, not a text summary (we
tried text → −6.0). The simplest way for d1-2B: a **compact latent memory token** (following Mem-W) or a **ROI
crop** (following MementoGUI) — to keep the visual cue. This is the direction so as not to lose on multi-step.

---

## 4. Summary for the project (priority order)

| Direction | Worth doing? | Reason | How |
|---|---|---|---|
| **RL + dense grounding reward** | ✅ **PRIORITY #1** | Strongest evidence for **closing the grounding gap 56.5 → 70+**, raises both AC-v2 & AndroidWorld | Magma-R1 (2D-Gaussian reward) + GUI-Libra (KL trust region) |
| **Latent/multimodal memory** | ✅ #2 | Needed for multi-step, does not lose to them; text memory already failed | Mem-W latent token or MementoGUI ROI crop |
| **CoT/reasoning** | ❌ **NO** | ACL 2026: no consistent improvement, lowers grounding, raises tokens 3–15× which breaks latency | Drop — our axis is direct decision + calibration |

### Key points for the paper
- **CoT is not a requirement to be good at GUI** — the ACL 2026 evidence + Magma-R1 + GUI-Libra all point out that CoT can
  **lower** grounding. Our "0-token calibrated" axis is **stronger** on latency + calibration.
- **What we need so as NOT to lose on multi-step = memory (latent/multimodal) + RL (dense reward)**, NOT
  imitating reasoning.
- **Static benchmarks have errors** (ACL 2026: 47.8% weak-eval; Magma-R1: ~20% annotation errors) → supports the paper's content
  on auditing benchmarks.

---

## Bibkey / Source

- [cotACL2026] Zhang, Gao, Xu. *Does Chain-of-Thought Reasoning Help Mobile GUI Agents? An Empirical Study.* Findings of ACL 2026. arXiv 2503.16788 / aclanthology.org/2026.findings-acl.392
- [magmaR1] *AndroidControl-Curated + Magma-R1.* arXiv 2510.18488.
- [GUIlibra] Rui Yang et al. *GUI-Libra.* arXiv 2602.22190 (NeurIPS 2026), gui-libra.github.io
- [MobileRL] *MobileRL: Online Agentic RL for Mobile GUI Agents.* arXiv 2509.18119 (ICLR 2026)
- [MemW] Guibin Zhang, ... Shuicheng Yan. *Mem-W: Latent Memory-Native GUI Agents.* arXiv 2605.09317
- [MementoGUI] *MementoGUI: Learning Agentic Multimodal Memory Control.* arXiv 2605.18652
- [CoMEM] Wu et al. *Auto-scaling Continuous Memory for GUI Agent.* arXiv 2510.09038
- [FocusMem] Zhang et al. *FocusMem: Factorizing Content, Readout, Trust.* arXiv 2608.04530
- [AndroTMem] *AndroTMem.* arXiv 2603.18429
- [MemGUI-Bench] *MemGUI-Bench.* arXiv 2602.06075
- [UIAGILE] *UI-AGILE: Advancing GUI Agents with Effective RL.* arXiv 2507.22025
- [HyperClick] *Enhancing Trustworthy GUI Grounding via Self-Critiqued RL.* arXiv 2510.27266
- [MemGUI-Agent] *MemGUI-Agent: End-to-End Long-Horizon Mobile GUI Agent.* arXiv 2606.19926
- [TSR] *A Task-State Representation for Long-Horizon Mobile GUI Agents.* arXiv 2607.00502
- [WhatMem] *What Memory Do GUI Agents Really Need? From Passive Records to Active Task-Driving States.* arXiv 2606.31612
