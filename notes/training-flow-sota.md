# The SOTA Training Flow — how the researchers THINK (distilled from MobileRL / UI-Venus / GUI-Libra)

> Written 2026-10-27 after deep-reading three primary papers. Purpose: replace the wrong
> "frozen-encoder + linear classify-action-type" framing with the actual SOTA training design.
> Sources: MobileRL (arXiv 2509.18119, ICLR 2026), UI-Venus (2508.10833), GUI-Libra (2602.22190).

---

## 0. The single biggest misunderstanding we had

We framed the task as **"classify the action type (click/swipe/wait) of ONE step."** That is NOT
what these systems do, and it is why our model collapsed to the majority prior (click=60% → always
predict click = 0.5828 = our result).

Real framing: **a mobile-GUI task is a multi-step sequential MDP with a SPARSE terminal reward.**
- State = screenshot + UI/XML hierarchy.
- Action = a grounded atomic op (Tap on element `[x1,y1,x2,y2]`, Type `text=...`, Swipe `dir+dist+[box]`,
  LongPress `[box]`, Launch, Back, Home, Wait, Finish) — **the action carries coordinates, not just a type.**
- Reward = binary 0/1, delivered **only when the whole task succeeds** (or horizon reached).
- Objective = maximize task-success probability over the trajectory. (MobileRL §2, Eq. for `θ*`.)

So the unit of value is **per-episode task success (SR)**, not per-step type accuracy. And the
"action" is grounded (has coordinates). This is exactly the grounding problem — see §2, §3.

---

## 1. MobileRL — the 3-stage flow (reasons matter)

The pipeline is **staged warm-up, THEN RL** because online RL in mobile emulators is expensive to
sample. Every stage exists to reduce the cost/stability of the next.

| Stage | What it does | Data | Result (Qwen2.5-VL-7B) |
|---|---|---|---|
| **Base (no train)** | — | — | AndroidWorld 27.6 |
| **1. Reasoning-Free SFT** | Teach the raw action format/grounding from expert demos | 97.9k human annot. + 23.6k AndroidControl | **50.2 (+22.6)** |
| **2. Reasoning SFT** | Add intermediate reasoning traces (make the policy "think like a person") | boosted reasoning corpus from an Instruct model | **56.8 (+6.6)** |
| **3. AdaGRPO (online RL)** | Optimize for task success with sparse terminal reward | 2,000 AndroidWorld + 1,103 AndroidLab tasks | **72.0 (+15.2)** |

(with GLM-4.1V-9B: 7.7 → 48.1 (+40.4) → 66.2 (+18.1) → 80.2 (+14.0).)

**The flow order is the lesson.** Grounding/action-format SFT gives the biggest jump; reasoning SFT
adds a bit more; RL adds a final large chunk on top of a good base. **If you skip SFT and RL cold,
it barely works (27.6).** Our earlier "just train a linear head" skipped all of this.

### How they get the "reasoning" (the part that makes it think like a human)
Expert demos only contain the final action sequence — no intermediate reasoning. So they **bootstrap**
reasoning with an off-the-shelf Instruct model:
1. For each task `x` with expert action `a*`, the Instruct model generates diverse `(reasoning c_k, action a_k)`.
2. Only keep pairs where `a_k == a*` (correct action) → `D_R`.
3. SFT an initial reasoning policy `π₀^R` on `D_R`.
4. Iteratively refine: `π_t^R` proposes candidates; keep the ones matching `a*`; pick the best
   explanation `c*`; fine-tune `π_{t+1}^R`. (2 epochs.)

This is literally **teaching the model the human decision process** ("I see the screen, I reason,
I decide which element to act on") before letting it explore on its own.

### AdaGRPO = GRPO + 3 difficulty/efficiency adapters (all about the sparse-reward problem)
- **SPA (Shortest-Path Reward Adjustment):** don't reward all successful trajectories equally —
  shorter successful paths get higher reward (`R^SPA = r(τ_i)(1 - α (T_i - T_min)/T_i)`). Counteracts
  bias toward verbose/long solutions; matches user preference for efficiency. (α=1.0.)
- **AdaPR (Difficulty-Adaptive Positive Replay):** successful-but-hard trajectories are rare and
  high-value. Keep the top-κ by advantage in a replay buffer, mix with fresh on-policy samples
  (`q(τ)=γ p_B(τ)+(1-γ) p_on(τ)`). Reuse scarce informative positives.
- **FCF (Failure Curriculum Filtering):** tasks with all-zero reward for 2 consecutive epochs enter a
  cooldown; sampling weight `w_task=exp(-f)`; then permanently removed. Stops wasting compute on
  unsolvable tasks (heavy-tailed difficulty).
- KL term `β D_KL(π_θ ‖ π_ref)`, β=0.001 to keep from drifting too far.

**Ablation (the diagnostic gold):** removing AdaPR & SPA → training collapses after ~30 steps.
   So the **combination** is what stabilizes. This is the answer to "why did ours collapse" — no RL,
   no reward shaping, no difficulty adaptation, no KL trust region.

---

## 2. UI-Venus — go straight to RFT + a fine-grained action reward + sparse-action balancing

UI-Venus takes Qwen2.5-VL and applies **RFT (GRPO) directly** (no SFT warmstart reported). Its
central claim (verbatim): *"SFT is not appropriate in discriminative tasks like UI agent... SFT
assigns widely-used cross-entropy loss penalty to ANY predicted points within and without the box
except the center point."* → a strict point-matching loss is the WRONG loss; use an interval/set
reward that gives equal credit to any point inside the box.

### Two separate branches (reward conflict makes them push apart)
- **Grounding branch** (localize the element): reward = `format + point-in-box` (`r_g = 1{x1≤xc≤x2 & y1≤yc≤y2}`).
- **Navigation branch** (multi-step): 4-part **additive** action-wise reward (Eq. 11):
  `R = R_format·w1 + (R_type + R_coord + R_content)·w2`
  1. **R_format** — output follows `<think>…</think><action>…</action>` template.
  2. **R_type** — predicted action type matches ground truth (1/0).
  3. **R_coord** — stepwise partial-credit by pixel distance: `α` if `d<δ2`, `0.5α` if `δ2≤d<δ1`, 0 otherwise.
     Scroll rewards start+end+direction. (Reason: a rigid threshold gives zero gradient on "get closer.")
  4. **R_content** — `γ` if token-F1 ≥ 0.5 (for Type text), 0 otherwise.

### Sparse Action Enhancement (§3.3.3) — the direct fix for class imbalance
Data is long-tailed (Click/Scroll frequent; LongPress/CallUser rare). To make rare actions learnable
without inflating common ones, they increase the **diversity of contexts** that lead to the rare action:
- Build a **thought pool** per step: rollout the policy R times, keep only `(thought)` whose predicted
  **action exactly matches ground truth**.
- For a **sparse-action step**, construct **M historical variants** by sampling prior thoughts from each
  pool — conceptually the **Cartesian product C_1 × … × C_{n-1}** — holding the rare action fixed but
  diversifying all the reasoning history that precedes it.
- **Common actions are untouched.** So the model learns "this rare action is reachable from many
  different situations," not just a memorized (state→action) pair.

> **Lesson for us:** don't just oversample the minority class. **Enrich the valid histories/contexts
> that lead to the rare action, keep the action fixed, and upsample those.** This is contextual
> class-balancing.

### Training hyperparameters (exact)
- lr = **4e-7**, rollout sample = **8**, global batch = 128 (grounding) / 256-512 (nav), 128-512 PPU-GPUs,
  ~1 day (7B) / 8.5 days (72B) per **epoch**. Long horizon, low LR.

---

## 3. GUI-Libra — action-aware SFT → conservative KL-regularized RL

GUI-Libra argues generic RLVR post-training breaks on GUI for two reasons:
1. **Standard CoT SFT hurts grounding** (long reasoning tokens dominate the loss and push toward
   semantic abstraction away from pixel-precise localization). Measured: **grounding accuracy has a
   clear NEGATIVE correlation with response length.**
2. **Partial verifiability** — `r̃(s,a)=1{a=ã(s)}` is reliable when positive (this action IS correct)
   but **ambiguous when negative** (there may be other valid actions not demonstrated).

### Action-aware SFT (ASFT) — §5.2
Three levers:
- **Mixed supervision** (reasoning-then-action AND direct-action only) so the model learns both modes.
- **Action-aware token reweighting:** split `<answer>` tokens into action vs grounding (`point_2d`);
  `L_ASFT` upweights them: `α_a=2` (action), `α_g=4` (grounding). Grounding gets the highest weight
  because it determines execution.
- So reasoning gets weight 1, action 2, grounding 4.

### Conservative GRPO — §5.3
- **KL trust region** (β≈0.001-0.005) — the single most important anti-collapse lever. Without KL,
  training reward climbs but eval degrades (reward hacking), policy entropy collapses, offline→online
  correlation weak (Pearson 0.63). With small KL: entropy stable, **Pearson 0.89**, robust alignment.
  **Too large also bad** (β=0.05 → over-conservative, AndroidWorld drops). Moderate is the sweet spot.
- **SNGS (Success-adaptive Negative Gradient Scaling):** instead of treating every `r=0` as equally
  negative (overfits the single demonstrated action), scale negative advantages by a state-conditioned
  reliability `λ_g(s) = min(λ0 + κ·p̂_g(s), 1)` where `p̂_g` = group success rate. Keep negatives to
  stabilize but don't over-penalize valid-but-uncredited actions.
- Reward (multiplicative AND): `r̃ = w_fmt·r_fmt + (1-w_fmt)·r_acc`, `w_fmt=0.1`, with
  `r_acc = r_act · r_val · r_g` (type AND value-F1>0.5 AND point-in-box).

### Data curation (relevant for small datasets)
- Aggregate 8 GUI datasets; drop <3 or >50 steps, drop compound actions → 19K traj / 170K steps.
- Unified format (Fig 2): system prompt (action list) + user instruction + interaction history +
  current screenshot → `<think>…</think><answer>…JSON…</answer>` (action_type, description, target,
  value, point_2d). **13 action types**: Click, Write, Terminate, Swipe, Scroll, NavigateHome, Answer,
  Wait, OpenAPP, NavigateBack, KeyboardPress, LongPress, Select.
- **SFT filtering (§4.1.4):** (1) agreement filtering by re-prediction (run model 10×, drop steps with
  <0.3 match to annotation); (2) bbox-verification filtering. → 81K SFT steps / 9K traj.
- **RL filtering (§4.1.5):** downsample early steps (home-screen actions dominate) + downsample mobile
  domain → 40K-step RL subset, more balanced.
- **Key quote on imbalance:** *"Click accounts for around 60% of steps, followed by Write, Terminate,
  and Swipe, while LongPress and Select are rare. This imbalance makes rare actions difficult to learn
  from SFT alone, **motivating a subsequent RL stage.**"*

---

## 4. The distilled FLOW (what we should actually do)

Synthesizing all three papers into one build order for a **single-pass / grounded next-action
decision model** — and being honest that the SOTA flow is generative multi-step RL, not single-pass:

```
STEP 0 — REFRAME THE TASK
  Task = grounded next-action choice: state(screenshot + instruction + history) → score K grounded
  candidate actions (action_type + point_2d/bounds). Gold = candidate matching the GT action+coordinate.
  Metric = per-episode task-success-style accuracy, reported against majority/random baselines.
  (NOT "classify click vs swipe" — that is prior-dominated and meaningless.)

STEP 1 — CLEAN + VALIDATE DATA  (GUI-Libra: ~20% label error in AndroidControl)
  - Agreement filtering (re-predict N times, drop low-agreement steps).
  - Bbox-verification filtering (point_2d inside predicted box).
  - Downsample repetitive clicks / early home-screen steps.

STEP 2 — FIX IMBALANCE (UI-Venus Sparse Action Enhancement / GUI-Libra ASFT)
  - For rare actions (long_press, paste, copy, scroll-up): generate MULTIPLE valid reasoning/history
    contexts that end in that same action (Cartesian product of thought pools), upsample only those.
  - Leave dominant click/tap at natural rate.

STEP 3 — ACTION-AWARE SFT WARM-START (GUI-Libra ASFT)
  - Mix reasoning-then-action and direct-action supervision.
  - Token reweight α_a≈2 (action), α_g≈4 (grounding); reasoning weight 1.
  - lr 1e-5, small effective batch. NOT plain CoT-SFT.

STEP 4 — (SINGLE-PASS READOUT) 
  - If building a true single-pass decision model: score K candidates in one forward pass
    (d1-style readout at the answer slot, or a 1-2 layer DecisionHead over a marker token), zero tokens.
  - Keep a SEPARATE P(success) head trained on [predicted==gt] for the risk/abstain gate —
    do NOT read success from argmax-softmax (it's class-membership prob, not correctness prob).

STEP 5 — CONSERVATIVE GRPO / CALIBRATION  (if RL; else the calibration guardrails)
  - KL trust region β≈0.001 (the #1 anti-collapse lever). SNGS for negative re-scaling.
  - Reward: additive partial-credit action-wise (UI-Venus) is gentler for a small model;
    validate against a strict all-correct measure so you don't game the metric.
  - If using proper-scoring for calibration: KEEP a reference/teacher temperature + KL to prevent
    the confidence-spread collapse (see notes/action-collapse-diagnosis.md).

STEP 6 — EVALUATE HONESTLY
  - Report accuracy vs majority AND random baseline + ECE + per-class recall on minority actions.
  - Report offline→online correlation (Pearson) if you do RL; stop at best-correlated checkpoint.
```

---

## 5. What this means for our "single-pass decision model" thesis (honest)

- The SOTA is **generative, multi-step RL over trajectories**, using reasoning traces + history +
  sparse terminal reward. It is NOT a single-pass over-K choice.
- Our project's core claim (single-pass, zero-token, calibrated, low-latency) is a **different, weaker
  in raw task-success** framing. To be defensible it must be justified on a benchmark that rewards
  **calibration + latency + structured output**, not raw SR; and it must be compared to the RL-SOTA SR
  metric + proper baselines (majority/random).
- The parts of the SOTA flow that DO transfer directly to a single-pass model:
  1. **Reframe to grounded next-action choice** (the action carries coordinates).
  2. **Clean + validate labels** (20% error).
  3. **Context-based class balancing** for imbalance.
  4. **Action-aware supervision + separate P(success) head** for calibration.
  5. **KL trust region + reward shaping** if any RL, to avoid collapse.
- The parts that do NOT transfer trivially: multi-turn trajectory RL, reasoning traces at inference
  (which a single-pass model cannot emit), online emulator sampling.

---

## 4b. Magma-R1 — the CLOSEST precedent to what we want (small model, grounded next-action, cheap)

Magma-R1 (AndroidControl-Curated, arXiv 2510.18488) is the single most relevant reference for us.
- **What it does:** given screenshot + task (+ past actions), emit `{"bbox_2d":[x1,y1,x2,y2],"action_type":...}`
  for the NEXT single action. That is **exactly** a grounded next-action decision.
- **Method:** one-stage GRPO (no SFT stage), post-training a small multimodal base.
- **Reward — the key design:** a **dense 2D-Gaussian grounding reward**, NOT binary hit/miss:
  `R_grounding = exp(-‖c_pred − p_gt‖²/(2σ²))` where `c_pred` = center of predicted box, `p_gt` = GT point.
  Rationale: binary reward gives "no effective learning signal" for near-misses; the Gaussian gives
  **reward density, smoothness, physical intuition** (slightly-off click >> miss completely).
- **Class imbalance fix:** **Action-Type Proportional Optimization** — stratified sampling by action type
  toward a target distribution that over-samples `type`/`scroll`/`swipe` (rare) relative to the empirical
  distribution (clicks dominate).
- **The radical insight (its headline):** ~30% of AndroidControl has ambiguous/wrong labels, and the
  benchmark was "actively teaching incorrect behaviors." So **data quality > quantity**: trained on just
  **2.4k curated samples / ~60 h H20 GPU (≈$60)**, Magma-R1-3B matches Qwen3-VL-235B (200× larger) and
  Infi-GUI-R1 (31k samples).
- **⚠ Base-model discrepancy:** paper says "3B MobileVLM-base Magma-R1-3B" but the HF-released checkpoints
  are `Magma-R1-4B-*` tagged `qwen2_5_vl`. Could not verify a released 3B MobileVLM checkpoint — needs follow-up.
- **Grounding eval lesson:** use **bounding-box intent alignment** (is the predicted center inside the GT
  box / on the element), NOT exact-point matching — exact-point "assesses geometric proximity, not whether
  the model's interaction intent is correct."

**Why this matters for us most:** Magma-R1 is a SMALL model doing grounded next-action selection, optimized
with a dense per-step grounding reward, on a tiny curated dataset, for ~$60. It is the closest existing
proof that *our* path (small model, single grounded decision, cheap) works — and it validates that the
DENSE grounding reward is what makes a small model precise/calibrated at grounding, rather than a huge
reasoning chain or a whole multi-turn RL infrastructure.

---

## Coverage / honesty
- MobileRL: read full main body + Appendix B (hyperparameters, action space, pruning) + ablation §3.3.
- UI-Venus: read full body incl. §3.3.3 sparse-action, §3.3.4 reward, Table 4/§4.1.
- GUI-Libra: read §1/§3/§4/§5 (ASFT, partial-verifiability, KL, SNGS, reward, pipeline) + §6.
- Reward constants (w1/w2, δ1/δ2/δ3, α/β/γ) not numeric in UI-Venus text; GUI-Libra α_a=2/α_g=4,
  β=0.001-0.005, lr 1e-5/1e-6 are stated. AndroidControl's ~20% label-error is GUI-Libra's finding.
