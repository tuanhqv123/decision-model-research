# SOTA Architecture Deep-Read: Reuse + Enhance for a Calibrated On-Device Decision Head

**Date:** 2026-10-14
**Status:** deep-read of Mem-W / UI-Venus-1.5 / UI-Venus-2 / GUI-Libra, distilled for reusable architecture + enhancement points.
**Sources:** arXiv 2605.09317 (Mem-W) `[ABS]`, arXiv 2602.22190 (GUI-Libra) `[ABS]`, HF `inclusionAI/UI-Venus-2-9B`, HF `inclusionAI/UI-Venus-1.5-8B` `[CARD]`, plus repo-internal notes/benchmarks `[REPO]`.

> Provenance legend: `[ABS]` live arXiv abstract · `[CARD]` live HF model card · `[REPO]` repo synthesis · `[INFER]` our analysis. Vendor self-scored numbers are flagged.

---

## 0. Bottom line

All four SOTA systems are **autoregressive generative VLMs**: they decode the next action as `(action_type, value, point_2d)` tokens. **None** emits a native calibrated probability over a closed candidate-action set, and **none** has an abstain/ask/refuse interface. Their "confidence" is at best a post-hoc mean-token-logprob, which mixes reasoning tokens into the score and is not a distribution over candidate actions.

**This is exactly the gap our ≤2.6B calibrated single-pass decision head + act/ask/refuse/escalate gate targets.**

The most reusable things are **not** the SOTA architectures per se, but these training/verification mechanisms:

1. **Mem-W** — trajectory→latent Q-Former compressor + latent weaving; self-distillation + KL-anchored outcome-aware RL.
2. **GUI-Libra** — action-aware SFT reweighting; KL trust region in partially-verifiable RLVR; success-adaptive scaling; the AC-v2 offline harness/scorer.
3. **UI-Venus-2** — multi-teacher on-policy distillation with action-aware supervision; keypoint-grounded + multi-model-voting reward; GGUF + mmproj vision-capable deployment.
4. **UI-Venus-1.5** — the actual Qwen3-VL-8B backbone Mem-W wraps; useful as a teacher and the P0 anchor baseline.

**Honest headline we can defend:** *reliability-per-cost on-device at matched accuracy*, not raw-accuracy SOTA.

---

## 1. Target benchmark + protocol (fixed)

From the GUI-Libra AC-v2 offline harness, ported into `benchmarks/run_acv2.py` `[REPO]`:

- **398-step leaderboard subset** (`500_steps_filtered.json`, 350 episodes). Action distribution: 212 click, 87 status, 37 scroll, 27 open_app, 20 navigate_back, 11 type_text, 4 wait.
- **Inputs (native "High v2"):** screenshot + **goal only** (no step instruction, no a11y tree in prompt, empty interaction history) + 8-type system prompt (`Click/Write/LongPress/Scroll/Wait/NavigateBack/OpenApp/Terminate`) + `<answer>` JSON template with `point_2d` = **pixel** coords.
- **Scoring (`eval_official.py` port):** spatial steps (click/long_press/type_text) graded **element-level**: find the smallest a11y node containing the ground-truth point, then require the predicted point to fall inside that node's bbox. `type_text` also requires exact text. Non-spatial steps require all shared GT fields to match. Equivalences: click↔open_app (element text contains app name) and click↔navigate_back (element text contains "back").
- **Pass@1** = greedy (we run **T=0**); **Pass@4** = best-of-4. Mem-W paper: T=0, top-p 1.0, max 64 tokens/step. GUI-Libra code defaults T=0.5 — we run T=0 and note it.

**Crucial structural fact:** the strict "High v2" protocol gives the model only screenshot+goal and requires a pixel coordinate from scratch. There is **no injected candidate-action K-set**. Our single-pass head is a `choice` over K; on AC-v2 High, K must come from an external extractor (Android accessibility tree, SoM/segmentation, or an on-device grounding extractor). The AX tree is embedded in each eval row (ground truth), so it can supply an "oracle" K — but that is arguably out-of-protocol. **This is the single biggest framing risk** and is resolved by the two-path evaluation plan (§10).

**Element-level scoring makes choice-over-K natively compatible:** a head that predicts a UI element (an a11y node) can emit that node's bbox center as `point_2d`, and the official scorer accepts it.

---

## 2. Mem-W (arXiv 2605.09317) `[ABS][REPO]`

### Architecture components & roles

- **Frozen VLM backbone** (UI-Venus-1.5-8B for Mem-W-8B; Qwen3-VL-4B for Mem-W-4B) — fully frozen.
- **Learnable Q-Former trajectory→latent compressor** — the *only* trained component. K learned queries encode a trajectory/episode chunk into K×d latent tokens.
- **Procedural memory bank** (~22,346 web + ~4,972 mobile trajectories) — episodic prior; top-M retrieval via frozen encoder + faiss/CLIP.
- **Working memory** — in-session expired-prefix chunks, compressed by the same compressor.
- **Latent-token weaving** — memory tokens woven in front of / into the normal context, so the frozen VLM sees `[memory latents] + [screenshot + last L steps]` as one continuous embedding sequence.

### Training recipe

- **Stage 1 — self-distillation:** teacher = same frozen agent with a long raw window; student = latent-augmented agent; objective = action loss + KL to teacher.
- **Stage 2 — outcome-aware policy gradient:** full-episode rollouts scored by success/failure; KL-anchored to Stage-1 policy. Only the compressor is updated; backbone stays frozen.
- Data = memory-bank trajectories (web + mobile). Mem-W repo ships **no AndroidControl eval script**; the AC-v2 number comes from the paper's table.

### Sizes / latency / where it runs

- 8B ≈ 17 GB fp16 (tight on a 32 GB Mac); 4B ≈ 9 GB.
- Repo is CUDA-oriented (flash_attn, faiss+CLIP, playwright). **Not runnable as-is on Apple silicon** — needs flash_attn→sdpa port, MPS backend, rebuilt bank, and a written AC-v2 eval.
- Latency estimate: backbone alone ~2–5 s/step on M1 Pro; Mem-W adds retrieval + compressor. **No measured on-device Mem-W latency exists.**

### Reusable modules we can adopt

1. **Q-Former latent-memory compressor + weaving** — as a caching/memory conditioner for multi-step loops. On single-step AC-v2 High, the working-memory path is inert; the **procedural-bank retrieval-as-prior** is what carries Mem-W's +7.5 pts over its backbone. We can replicate a compact distilled prior or use the compressor for long-horizon extensions.
2. **Self-distillation** — train our head to be decision-faithful to a stronger teacher (e.g., UI-Venus-1.5-8B).
3. **Outcome-aware RL + KL anchor** — Stage-3 objective for our head, optimizing task success rather than just step-match.

### Weaknesses our head + gate targets

- **No calibrated action probability.** Greedy T=0; "confidence" = mean token-logprob, uncalibrated and contaminated by reasoning tokens.
- **Decode latency on the critical path.** Up to 64 action/reasoning tokens per step + retrieval + compressor forward = seconds/step.
- **On-device infeasible as published.** CUDA + faiss/CLIP + ~17 GB.
- **Redundancy on this benchmark.** AC-v2 High has empty interaction history, so the working-memory channel has no in-session content; the leaderboard gain comes from a retrieval-augmented prior — where a compact distilled prior can substitute.
- **Unverified.** 68.59 / 63.07 are author-reported, not independently reproduced.

---

## 3. UI-Venus-1.5-8B (arXiv 2602.09082) `[CARD][REPO]`

### Architecture components & roles

- **Qwen3-VL-8B fine-tune** (family: 2B/8B dense, 30B-A3B MoE). Unified end-to-end GUI agent; Apache-2.0.
- Four-stage pipeline: (1) Mid-Training ~10B tokens across 30+ datasets for GUI semantics; (2) Offline-RL; (3) Online-RL full-trajectory rollouts; (4) Model Merge unifying grounding/web/mobile specialists.

### Sizes / latency / where it runs

- 8B bf16 ≈ 16 GB; transformer+MPS ~2–5 s/step on M1 Pro (≈3–4 h for the full 3,051). vLLM/SGLang intended. **Measured-on-Mac latency not published** — repo estimate.
- Card metadata lists the "8B" model as **"9B params"** — metadata inconsistency.

### Reusable modules

- **It is the actual SOTA backbone Mem-W wraps**, so it is our **T1/P0 anchor** (reproduce ≈61.06 ±3).
- **Specialist merge (grounding/web/mobile)** — clean recipe for composing capabilities; we can emulate at small scale by distilling multiple specialists into one head.

### Weaknesses our head + gate targets

- **Grounding is the bottleneck.** Leaderboard row: Pass@1 **61.06**, Type **74.62**, Grounding **61.06**. The model knows what to do but not exactly where. A choice-over-grounded-K-set head removes coordinate hallucination.
- **Manual temperature heuristics, not calibration.** Card: agentic tasks temp 1.0 + reasoning; grounding temp 0 + no reasoning. That's a manual switch, not calibrated confidence. T=1.0 encourages sampling, bad for greedy Pass@1 and hides miscalibration.
- **Autoregressive decode with reasoning** — latency scales with reasoning-token count.
- **No ask/refuse / risk gate.**
- **Caveat:** AC-v2 61.06 is a leaderboard row `[REPO]`, not on the model card. Card AndroidWorld 77.6 / ScreenSpot-Pro 69.6 are **30B-A3B self-scored**; ScreenSpot-Pro 68.4 (8B) is a vendor eval result.

---

## 4. UI-Venus-2-9B (arXiv 2609.00028) `[CARD][REPO]`

### Architecture components & roles

- Initialized from **Qwen3.5-9B**; unified closed-loop reason→act agent across mobile/web/desktop.
- Three-stage training: (1) Multimodal Mid-Training over simulated mobile/web/OS with human+discriminator filtering; (2) Offline RL per domain (Grounding/CAPTCHA/Mobile/Web/Computer); (3) Multi-teacher On-policy Distillation consolidating specialists, with action-aware supervision concentrated on executable behaviors.
- **Verification for RL reward:** task completion judged on **task-relevant visual keypoints** + **multi-model voting** (robust to reward hacking). Data: web pool 4,000+ domains / 19 categories, 45k tasks from InSTA-150k-v3.

### Sizes / latency / where it runs

- 9B (also 27B). Served via vLLM (`--reasoning-parser qwen3`, max 262k context) or SGLang.
- **GGUF + mmproj vision path exists** (Q4_K_M + mmproj-bf16 ≈ 6.5 GB) via llama.cpp/LM Studio/Ollama; repo estimate ~1–3 s/step on this Mac. **Estimate, not measured.**
- **License pending:** Apache-2.0 was *intentionally not carried over* → **do not redistribute / build production on these weights without per-checkpoint diligence.**
- **Metadata bug:** card lists "Model size 1.47M params" — inconsistent with 9B name / Qwen3.5-9B base. Treat as a metadata error.

### Reusable modules

1. **Multi-teacher on-policy distillation with action-aware supervision** — the most directly transferable recipe for a small calibrated head.
2. **Keypoint + multi-model-voting reward** — robust, reward-hacking-resistant success signal for our outcome-aware RL.
3. **Human–discriminator filtering** — maps directly to building **ask/refuse** labels from invalid/ambiguous interactions.
4. **GGUF + mmproj vision path** — the only vision-capable on-device backend available on this Mac (llama.cpp mtmd).

### Weaknesses our head + gate targets

- **No calibrated probability.** Temp 1.0 + full reasoning history maxes out decode cost. No abstain/ask/refuse.
- **Safety gap:** OSHarm ASR **11.3%** / OSBlind ASR **48.8%** (vs Qwen3.5-9B 25.3/79.4) — roughly **half of benign-looking-but-harmful instructions still get executed**. Strongest concrete motivation for an act/ask/refuse gate.
- **Vendor self-scored** (AndroidWorld 80.2/84.0, WebVoyager 90.8) — not independently verified and **not** AC-v2 comparable. UI-Venus-2 is not on the AC-v2 leaderboard.
- **Redundancy:** huge compute budget; we only need the decision slice.
- **License blocker** if we build on its weights.

---

## 5. GUI-Libra (arXiv 2602.22190) `[ABS][REPO]`

Primarily a **training recipe**, not a new architecture. Post-trains a native generative GUI agent (GUI-Libra-8B) to reason *and* act. Diagnoses two problems directly relevant to us:

1. Standard SFT with CoT **hurts grounding** (reasoning tokens degrade coordinate grounding).
2. Step-wise RLVR under **partial verifiability** — multiple actions can be correct but only one is verified, so offline step metrics weakly predict online task success.

### Training recipe

- **Data construction + filtering** → curated **81K GUI reasoning dataset**, action-aligned.
- **Action-aware SFT:** mixes reasoning-then-action and direct-action data; **re-weights tokens to emphasize action and grounding**.
- **Partially-verifiable RL (RLVR) + KL trust region:** identifies KL regularization as *critical* for offline→online predictability; adds **success-adaptive scaling** to downweight unreliable negative gradients.
- Result: consistent gains in step-wise accuracy *and* end-to-end task success, without costly online data collection.

### Reusable modules — our most directly reusable *training* recipe

1. **The AC-v2 offline harness + scorer** — already ported; our benchmark.
2. **8-type action space + system prompt + `<answer>` JSON template** — already in `run_acv2.py`.
3. **Action-aware SFT reweighting** — adopt in our head training; separates reasoning from action/grounding so we don't repeat CoT-hurts-grounding.
4. **KL trust region + success-adaptive scaling** — exactly what prevents RLVR-driven calibration collapse (our own prior: naive RLVR can push confidence to ~0.99, Brier ~0.43).

### Weaknesses our head + gate targets

- **No calibrated probability.** Generative token decode; T=0.5 default in code.
- **Partial verifiability handled as a loss trick, not as uncertainty.** It downweights noisy negatives instead of representing the candidate distribution — the head can't say "these two are both plausible, ask."
- **Offline→online gap explicitly admitted.** Same reason a **risk gate** is the right lever, not just step-accuracy.
- **Latency/cost.** Autoregressive reasoning + action decode; no on-device single-pass claim.

---

## 6. Cross-cutting reusable-module adoption table

| Module | Source | How we adopt |
|---|---|---|
| **One-pass calibrated choice-head mechanics** | Mapika `decider` (Jev-class): project hidden state onto one label token per option, softmax over valid options at fitted per-type temperature; confidence = `(n·p_max−1)/(n−1)` | This *is* our head. 1.2B→2.6B, MLX 4-bit, zero output tokens. |
| **K-set = AX tree** | AutoDroid-style element ids; on-device AccessibilityService | Feed head a `choice` over K real affordances; avoids coordinate hallucination. |
| **Latent memory compressor + weaving** | Mem-W Q-Former | Cheap memory module for multi-step loops; retrieval-as-prior for single-step. |
| **Self-distillation + outcome-aware RL (KL-anchored)** | Mem-W Stage 1+2 | Distill long-context teacher → compact student; reinforce on success with KL to keep calibration stable. |
| **Action-aware SFT reweighting + KL trust region + success-adaptive scaling** | GUI-Libra | Our CE / proper-scoring / joint-calibration training recipe. |
| **Keypoint-grounded + multi-model-voting reward** | UI-Venus-2 | Robust RL reward for grounding / task success. |
| **Eval harness / 8-type space / element-grounding scoring / equivalences** | GUI-Libra | Already ported; our benchmark. |
| **GGUF + mmproj vision path** | UI-Venus-2-9B | Our only vision-capable on-device backend (llama.cpp mtmd). |
| **FDR-controlled threshold + expected-utility gate** | SafeGround / CAAG + our `cascade.py` / `calib_metrics.py` | act / ask / refuse / escalate on the calibrated distribution. |
| **Safety eval (OSHarm / OSBlind)** | UI-Venus-2 | Score the refuse path on benign-but-harmful tasks. |

**The composition we claim as our own:** a calibrated one-pass scorer over a GUI-affordance K-set feeding an act/ask/refuse/escalate gate. No published direct prior art composes these pieces this way.

---

## 7. The targetable gaps

| Gap | Mem-W | UI-Venus-1.5/2 | GUI-Libra | Our calibrated single-pass head + gate |
|---|---|---|---|---|
| **Calibrated action probability over K** | None (greedy T=0) | None (point pred.) | None (point pred.) | **Changes the interface**: softmax over K candidates, one forward pass, zero output tokens. |
| **Abstain / ask / refuse** | None | None (OSBlind ASR 48.8%) | None | **Adds it**: act/ask/refuse/escalate gate on calibrated max-prob. |
| **Decode latency** | 64 tok/step + retrieval + compress | ~1–5 s/step | ~s/step | **~18 ms** for our 1.2B head (repo-recorded baseline, not live trained-head measurement). |
| **On-device (M1 Pro, no GPU)** | Port needed (CUDA/faiss/CLIP) | GGUF+mmproj ok ~1–3 s/step estimate | CUDA-oriented | **Native MLX 4-bit**, text-state first. |
| **Calibration training** | Self-distill + outcome RL (no calib obj.) | RL + distill (no calib obj.) | RLVR + KL (no calib obj.) | **CE vs proper-scoring vs joint-calibration** objective comparison. |
| **Verifier for RL reward** | outcome success/fail | keypoint + multi-model voting | demonstrated-action match | **Adopt UI-Venus-2 keypoint/multi-model voting.** |

---

## 8. Honesty ledger — vendor self-scored / unverified

| Claim | Status | Note |
|---|---|---|
| Mem-W-8B AC-v2 **68.59**, Mem-W-4B **63.07** | **Author-reported** `[ABS]/[REPO]` | No AC eval script shipped; not independently reproduced. |
| GUI-Libra-8B **64.3** | **Leaderboard-sourced** `[REPO]` | Abstract gives no single AC number; not our run. |
| UI-Venus-1.5-8B AC-v2 **61.06** | **Leaderboard-sourced** `[REPO]` | Not on the model card. Card AndroidWorld 77.6 (30B-A3B) / ScreenSpot-Pro 69.6 (30B-A3B) / 68.4 (8B) — vendor self-scored. |
| UI-Venus-2-9B AndroidWorld **80.2%** / ASR **11.3/48.8%** | **Vendor self-scored** `[CARD]` | Authors' own runs; AndroidWorld ≠ AC-v2 protocol. |
| UI-Venus-2-9B **"1.47M params"** | **Unverified/inconsistent** `[CARD]` | Contradicts 9B name / Qwen3.5-9B base; likely a card metadata bug. |
| UI-Venus-1.5-8B card **"9B params"** | **Inconsistent** `[CARD]` | Name says 8B; metadata says 9B. |
| UI-Venus-2-9B license | **Pending/conflicting** `[CARD]` | Apache-2.0 NOT carried over; do not assume redistribution rights. |
| Mem-W compressor internals (K queries → K×d tokens; faiss/CLIP top-M) | **Partially verified** `[REPO]` | From paper + cloned repo; repo lacks AC eval so end-to-end not verified. |
| GUI-Libra 81K dataset composition / exact loss weights | **Abstract-level only** `[ABS]` | Recipe verified at mechanism level; hyperparameters not. |
| SOTA "confidence" being miscalibrated | **Hypothesis** | Wait for Step-0 ECE measurement; not yet measured. |
| Latency numbers for SOTA backbones | **Repo estimates** `[REPO]` | Not live measurements. ~18 ms for our 1.2B head is a recorded baseline, not a live measurement of the trained head. |
| LFM Open License v1.0 | **Source-available, not OSI** | Read full terms before releasing fine-tuned weights. |

---

## 9. Recommended build order (us vs SOTA)

Ground truth on the 398-step subset: Mem-W-8B **68.59** · GUI-Libra-8B **64.3** · UI-Venus-1.5-8B **61.06** · Qwen3-VL-2B **45.48**.

- **Step 0 — Pin the harness and reproduce a leaderboard-credible baseline (P0).** Run `benchmarks/run_acv2.py` with `--backend hf` on UI-Venus-1.5-8B and `--backend llama` on UI-Venus-2-9B GGUF, on the exact 398-row protocol. Target UI-Venus-1.5-8B ≈61.06 ±3, cross-checked against `eval_official.py`. Document T=0, max-token, prompt settings.
- **Step 1 — Measure SOTA's calibration profile.** `run_acv2.py` already computes logprob-derived confidence. Add **ECE/Brier/coverage–risk/median/p95 latency** output. This gives the empirical hook: does SOTA even have honest confidence?
- **Step 2 — Port Mem-W-4B, time-boxed (optional but valuable).** Strip flash_attn→sdpa, build the mobile procedural memory bank from the AC-v2 train split (real data only), write the AC-v2 eval. Target ≈63.07. Risk: could burn days; if exceeded, fall back to Step-1 numbers (T1 stands on its own).
- **Step 3 — Build the calibrated single-pass head.** Base = LFM2.5 **1.2B → 2.6B**, MLX 4-bit, zero output tokens. Input: goal + step_instruction + K=4 real same-split candidate actions (text-state first). Output: probability vector over K. Train with GUI-Libra's **action-aware reweighting**, **KL trust region**, **success-adaptive scaling**. Compare three objectives on identical data: (a) CE-only, (b) CE + proper scoring (log/spherical/RPS), (c) CE + joint-calibration (α·ECE). Add a **separate calibrated P(success) head** rather than reading confidence off action logits — partial verifiability makes action logits peaked.
- **Step 4 — Add act/ask/refuse/escalate gate.** Gate on calibrated P(success) + entropy + wrong-action cost model: **act** only if P(success) ≥ τ and utility(act) ≥ utility(ask/refuse); **ask** if ambiguous/invalid; **refuse** if P(success) very low / high risk. Calibrate τ to a coverage constraint: **false-action ≤10% at coverage ≥50%**. Report ECE (15 bins), Brier, reliability diagram, coverage–risk curve.
- **Step 5 — Fast cascade.** 1.2B/2.6B head handles high-confidence easy steps at ~18 ms; SOTA VLM handles the hard remainder. Target: same step-accuracy ±2 pts at ≥5× lower average latency on M1 Pro.
- **Step 6 — Pass@4 reranker / SHIELD arm (separate report).** Keep the SOTA agent, let it sample 4 actions, use our calibrated head to rank/select the highest-confidence one (or gate all four). Report leaderboard Pass@4 improvement separately.
- **Step 7 — Ablations + final table + one-command repro (P4):** V1 small head ≈ big LLM at ≥10× cheaper; V2 proper-scoring vs CE (the Q2 mechanism); V3 gate vs no-gate; V4 data scale (2k vs 12k; 1.2B vs 2.6B); V5 train-time calibration vs post-hoc temperature.

### Pass conditions
- **P0** anchor: reproduce SOTA ±3.
- **P1** trust: risk ≤10% at coverage ≥50%, better than SOTA-alone.
- **P2** speed: ≥5× faster at ±2 pts.
- **P3** honesty: ECE ≤0.05, with bin count/definition, Brier decomposition, reliability diagram, coverage–risk curve.
- **P4** repro: one command regenerates everything.

---

## 10. Framing the improvement claim (honest axes)

We do **not** out-argmax the 68.59 generative VLM on the strict High-v2 protocol. Our axes:

1. **Risk-controlled executed success** — at fixed coverage, our gate lowers false-action rate while preserving executed-success (something SOTA cannot even express natively).
2. **Reliability-per-cost on-device** — ≥5× cheaper, ≤100 ms, on the M1 Pro.
3. **Calibration as a first-class output** — the model is honest about uncertainty via ECE/Brier/coverage–risk, not just pass@1.

### Two-path evaluation plan
- **Strict vision-only AC-v2 High run** (screenshot + goal only) for any leaderboard-parity claim. Uses UI-Venus-2 GGUF+mmproj (llama.cpp mtmd) or UI-Venus-1.5-8B (transformers+MPS).
- **A11y-guided text-state run** for the on-device path. Uses the AX tree as the K-set. Since the tree is an extra input modality, **do not present this as beating SOTA on the unmodified High protocol** — document the modality difference in the paper.

**The riskiest claim:** single-pass calibrated ECE / FDR. No prior evidence shows one latent read matches an ensemble of stochastic samples (SafeGround requires over-sampling for FDR guarantees; agent abstention is empirically near-unsolved, best frontier 59.5%). Mitigate by: training with a calibration-aware objective, class-balancing with a `none-of-these/refuse` label, validating on a separate on-device set, and reporting coverage–risk curves rather than a single operating point.

---

## 11. Top-line takeaway

The benchmark number is the point; calibration is the mechanism. The SOTA is won by a **frozen backbone + cheap learned latent-memory compressor + self-distillation/outcome-aware RL** (Mem-W) and by **post-training that fixes CoT-hurts-grounding and stabilizes partial-verifiability RL** (GUI-Libra). Neither gives a calibrated action probability, and all are GPU-only generative decoders.

Our edge: reuse the **latent-memory + KL-anchored RL** (for the agent loop), the **action-aware/KL-trust-region recipe** (for the head), and the **GGUF+mmproj vision backend**, then compose a **one-pass calibrated choice-head over an AX-tree K-set with an act/ask/refuse/escalate gate** — a composition with no direct prior art. The honest headline we can defend is **reliability-per-cost on-device at matched accuracy**, not raw-accuracy SOTA.
