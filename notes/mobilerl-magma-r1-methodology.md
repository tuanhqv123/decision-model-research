# How MobileRL & Magma-R1 actually train — a deep-read of the training methodology

**Scope.** This note deep-reads the FULL training methodology of MobileRL (arXiv:2509.18119) and
Magma-R1 (introduced in AndroidControl-Curated, arXiv:2510.18488), i.e. *how the researchers think
about and design the training pipeline*, not just the results table. All claims are cited to the
two papers / released repos by arXiv ID and section. Sources are in the numbered list at the end and
in the summary file. **Nothing here is invented**; where I could not verify something I say so.

> **CAVEAT on the MobileRL venue.** The task prompt calls MobileRL an "ICLR 2026" paper. The arXiv
> page I read (v1 2025-09-10, v2 2025-10-24) lists only the arXiv submission history; the page text
> does **not** state ICLR 2026 acceptance, and I did not verify that. Treat "ICLR 2026" as
> **unconfirmed** in this note.

---

## 0. One-line summary of how each team thinks

- **MobileRL (Tsinghua / Z.AI / GLM team)** thinks about **agentic RL as a cold-start + sample-budget
  problem.** Their whole pipeline is engineered around one fact: *interacting with an emulator is
  extremely expensive and rewards are sparse terminal 0/1 signals.* So they deliberately do **two
  SFT stages first** to shrink the amount of on-policy exploration needed before RL, then run a
  **reward-shaped, difficulty-adaptive, replay-augmented GRPO** that spends a fixed compute budget on
  the *most learnable* tasks and rewards *efficient* (short) solutions.
- **Magma-R1 (BMW ArcherMind / BA TechWorks)** thinks about **grounding precision as an RL reward
  design problem**, and, more radically, about **benchmark/data quality as the real bottleneck.** They
  argue a 3–4B model on ~2.4k *clean* samples can match a 235B model because the *reward is dense and
  the data is clean*, not because the model is big or the pipeline is long. Their training is a much
  simpler **one-stage GRPO with a dense Gaussian grounding reward + stratified action-type sampling.**

---

## 1. MobileRL — exact pipeline, stage by stage

MobileRL = **three ordered components** (§1, §2.1): `reasoning-free SFT → reasoning SFT → online
agentic RL (AdaGRPO)`.

### Stage 1 — Reasoning-Free SFT

- **What it is:** Supervised fine-tuning on expert demonstrations, **without** reasoning/chain-of-thought.
- **Why (verbatim, §2.1):**
  > "In agentic RL training, sampling in virtual-device environments is usually inefficient; thus,
  > starting online RL directly from a base model was found to be excessively time-consuming.
  > Therefore, we perform SFT with expert demonstration data obtained by following the data collection
  > protocol of [Xu et al., 2024 = AndroidLab], supplemented with the training split of the publicly
  > available AndroidControl dataset [Li et al., 2024]. Note that this data is reasoning-free."
- **Data (Appendix D.1 / D.5, Table 7):** 97.9k training **steps** total = 21.3k AndroidControl-Low
  + 14.3k AndroidControl-High + 62.2k human annotation. Human annotation follows AndroidLab's protocol
  [Xu et al. 2024], plus model-driven self-exploration (online rollouts, select correct action
  sequences) (§D.2). AndroidControl (8,444 test / 690 val / 74,714 train samples) is converted from
  accessibility tree → XML and points → bounding boxes (§D.5).
- **Loss:** standard next-token supervised cross-entropy over the action-sequence tokens.
- **Checkpoint:** the base VLM, e.g. Qwen2.5-VL-7B-Instruct or GLM-4.1V-9B-Base, after 2 epochs.
- **Hyperparameters (§B.1):** 2 epochs, cosine decayed LR 1e-5 → 1e-6, **Llama-Factory** with packing
  mode, full-resolution images (no compression).
- **Reasoning for ordering it FIRST:** It "builds a strong **action foundation** from expert
  demonstrations" (§2.2 Summary). It's the cheapest way to get the model to emit syntactically valid,
  roughly-correct actions before any expensive rollouts.

### Stage 2 — Reasoning SFT (the "reasoning warm-up")

- **What it is:** SFT on the *same* expert data but now **augmented with intermediate reasoning**
  (planning rationales). It's built by an *iterative reasoning-refinement* loop, not by hand.
- **Why (verbatim, §2.1):**
  > "To further construct a stronger reasoning policy initializer, we perform reasoning SFT via an
  > iterative reasoning refinement strategy over the expert dataset. Manually collected expert
  > demonstration datasets for mobile use often contain only the final action sequence, omitting
  > intermediate reasoning. Training solely on such 'black-box' trajectories yields opaque policies.
  > We leverage an off-the-shelf instruction model to bootstrap a reasoning-augmented training set from
  > raw demonstrations, yielding a structured and transparent policy initialization."
- **The 3-sub-step loop (§2.1):**
  1. **Bootstrap sampling:** for each task `x` with expert answer `a*`, an Instruct model `M` generates
     diverse `(c_k, a_k)`; keep `(x, c_k, a*)` whenever `a_k == a*`.
  2. **Supervised fine-tuning:** train an initial reasoning policy `π^R_0` on this set.
  3. **Iterative refinement:** at iteration `t`, `π^R_t` proposes candidates; those matching `a*` are
     scored by correctness, the best explanation `c*` is added to `D_new`, and `π^R_{t+1}` is obtained by
     fine-tuning.
- **Data (Table 7):** 23.6k training steps = 7.2k AndroidControl-Low + 4.1k AndroidControl-High +
  12.2k human annotation. *Note this is much smaller than Stage 1* — because it's expensive to
  bootstrap reasoning traces.
- **Loss:** SFT cross-entropy on `(screenshot, XML, task, reasoning, action)` sequences.
- **Checkpoint:** the Stage-1 model; the "reasoning warm-up model used for agentic RL training."
- **Hyperparameters:** same as Stage 1 (2 epochs, cosine LR 1e-5→1e-6, Llama-Factory, full-res images).
- **Reasoning for ordering it SECOND (the key design move):** Reasoning SFT exists to (a) handle
  **long/compositional instructions**, (b) **reduce costly on-policy trials** in the simulator, and
  (c) let the team use **open or human-labeled datasets without relying on proprietary models**
  (verbatim, §1 abstract). So it's placed *before* RL to shrink the exploration burden RL has to pay.

### Stage 3 — Online Agentic RL with AdaGRPO

- **What it is:** GRPO (group-relative policy optimization) on **multi-turn** rollouts in real Android
  emulators, extended by three mechanisms: SPA, AdaPR, FCF.
- **Setup (§2.1 Problem Formulation):** finite-horizon MDP; state = screenshot + UI hierarchy from XML
  (Appendix C preprocessing); action space = finite set of atomic ops; episodic **terminal reward
  `R(τ) ∈ {0,1}`** (success/failure), assigned only at the end. Objective = maximize success probability.
- **GRPO baseline (verbatim §2.1):** replaces PPO's learned value baseline with a group-relative
  baseline: advantage `Â_{i,t} = (R(τ_i) − μ_R)/σ_R` over a group of `G` trajectories sampled from the
  same initial state. Loss is the clipped PPO-style objective on the importance-sampling ratio, plus a
  KL term `β·D_KL(π_θ ‖ π_ref)`.
- **AdaGRPO's three strategies (§2.2):**

  **(a) Shortest-Path Reward Adjustment (SPA).**
  - Problem: broadcasting the terminal 0/1 reward to every timestep *biases toward longer
    trajectories* because they contribute more gradient terms (verbatim §2.2.1).
  - Fix (Eq. 2): `R_SPA(s_t,a_t) = r(τ_i)·(1 − α·(T_i − T_min)/T_i)`, where `T_min` = length of shortest
    successful trajectory in the current group, `α ∈ (0,1]` controls penalty. Reward is *not* given for
    early failure.
  - Verbatim rationale: "In this formulation, shorter sequences are not automatically considered
    better; unsuccessful early terminations still receive a reward of 0. This adjustment encourages the
    policy to prefer shorter, successful paths without sacrificing the success rate."

  **(b) Difficulty-Adaptive Positive Replay (AdaPR).**
  - Problem: in sparse-reward environments, "successful yet challenging rollouts are **rare but highly
    informative**" (verbatim §2.2.2).
  - Mechanism: each iteration, insert the **top κ trajectories by advantage** into a replay buffer `B`;
    each update samples a mini-batch from a **mixture** `q(τ) = γ·p_B(τ) + (1−γ)·p_on(τ)`; at most `γ·M`
    of the highest-advantage trajectories come from `B` to preserve on-policy diversity.
  - Purpose: "amplifies their learning signal and stabilizes policy updates."

  **(c) Failure Curriculum Filtering (FCF).**
  - Problem: uniform sampling wastes budget on *persistently unsolvable* tasks (heavy-tailed
    difficulty), and "hinders the collection of positive advantage data." (verbatim §2.2.3)
  - Mechanism: a task producing all-zero rewards for **two consecutive epochs** enters a **3-epoch
    cooldown**; during cooldown sampling probability is reduced by `w_task = exp(−f)` (`f` = number of
    consecutive failure epochs); after cooldown, the task is **permanently removed**. "We simplify it to
    target only the exclusion of the most difficult subset of data."
  - Purpose: "reallocat[es] computational budget toward feasible instances."

- **Loss/reward:** trajectory-level advantage computed from the (SPA-shaped) terminal reward, then GRPO.
  **AndroidWorld** uses rule-based verifiable rewards directly; **AndroidLab** has no rule-based reward,
  so they train a **VLM reward model** (§D.3.2) on GLM-4.1V-9B-Thinking, scoring full screenshot-sequence
  montages against step-by-step action descriptions, ~86% accuracy on a curated 1000-trace set.
- **Checkpoint:** the Stage-2 "reasoning warm-up" model.
- **Hyperparameters (§B.1, Table 4):** max prompt len 16384; max response len 4096; **train batch 256**;
  validation batch 256; **FSDP**; PPO micro-batch 4/GPU; **LR 1e-6**; **grad clip 1.0**; **clip ratio 0.2**;
  **PPO epochs 1**; **sampling temp 1.0**; max new tokens 4096; **n = 16** samples per task; **max turns 50**;
  max pixels 5,000,000 / min 65,536; **KL β = 0.001**; **SPA α = 1.0**; **replay buffer size 256**;
  **γ = 1.0**; **κ = 0.25**.
- **Implement details (§B.3):** additionally **prune lowest-advantage trajectories**, keeping a **max
  positive:negative ratio of 1:2**, to reduce noise; this "keeps policy entropy higher during training"
  and "delays premature convergence."
- **The infrastructure that makes RL feasible (§1, §E):** a Verl-based distributed framework that runs
  **hundreds of Dockerized Android virtual devices** across multiple machines, sustaining **>1,000
  concurrent environments**. Cost reality (§E): each AVD needs >5 GB RAM and >30 GB storage; they ran
  4 CPU machines with 1 TB RAM each just for rollouts; only **~256 stable parallel rollouts**; **each
  step >10 min**; **100 RL steps >25 hours**. This is *why* sample efficiency (FCF/AdaPR) matters so much.

---

## 2. MobileRL design philosophy — why the ordering, why RL last

**Central premise (§1):** the *hard* part of mobile agentic RL is not the algorithm, it's that
`(i)` correct rollouts are **rare and expensive** under sparse positive signals, `(ii)` task difficulty
is **heavy-tailed and unstable**, and `(iii)` large-scale sampling is **bottlenecked** by emulator cost.
Everything follows from that.

**Why not RL from the base model directly?** Verbatim: "starting online RL directly from a base model
was found to be excessively time-consuming" (§2.1). Because each rollout is >10 min/step and only ~256
parallel, you simply cannot afford the exploration a raw base model needs. So the two SFT stages exist
to *pre-pay* the exploration in cheap offline imitation, not in costly rollouts.

**Why reasoning-free SFT (Stage 1) BEFORE reasoning SFT (Stage 2)?** Stage 1 produces a working action
emitter cheaply and at scale (97.9k steps). Reasoning SFT is *much smaller* (23.6k) and *much more
expensive* (needs a bootstrapped reasoning corpus). Ordering reasoning-free first gives the model a solid
**"action foundation"** (their words) that the reasoning stage can then annotate and polish, rather than
trying to learn both the action format and the reasoning simultaneously from scratch.

**What problem does each stage solve (their §2.2 Summary, verbatim):**
> "Reasoning-free SFT helps build a strong action foundation from expert demonstrations, while
> reasoning SFT adds intermediate reasoning to improve instruction following and policy transparency.
> On top of this initialization, agentic RL with AdaGRPO addresses the challenges of sparse terminal
> rewards, heavy-tailed task difficulty, and expensive sampling."

**Why does reasoning SFT need to exist before RL?** Reasoning SFT is the "transparency" and
"instruction-following" layer that offline imitation can't give but that lets RL start from a policy that
already (a) parses long compositional instructions and (b) emits plans; this **reduces on-policy trials**
(verbatim §1). RL then *fine-tunes* the planning with a real success signal, rather than trying to
*teach* it from scratch with a reward that only fires at the end.

**Why does RL come LAST?** RL is the only stage with a *ground-truth success signal* (`R ∈ {0,1}`), and
it's the only stage that can correct the model when its learned actions fail in the wild (recovery from
errors). But RL is also the most expensive and the most unstable (sparse reward, heavy-tailed difficulty).
By the time RL starts, the model already imitates expert behavior and reasons; RL only needs to *select
among* near-correct behaviors and *shorten* them, not find them from scratch.

**Is reasoning-free SFT even necessary?** They explicitly ask this (§3.3, "Is Reasoning-Free SFT Still
Necessary?"). Verbatim: "our experiments show that incorporating Reasoning-Free SFT consistently improves
performance on AndroidWorld. It suggests that without explicit reasoning traces, (Reasoning-Free) SFT
contributes to enhancing final results for MobileRL." So **yes** — the cheap stage still helps, even with
a reasoning stage on top.

**Ablation evidence on the design (§3.3, Figure 3, verbatim observations):**
- **FCF under constraints:** "With a 100-step budget (>40 hours), FCF plays a key role in filtering.
  Removing it biases early sampling toward overly hard tasks, yielding many negatives and a lower reward
  ceiling."
- **AdaPR & SPA are necessary for stability:** "FCF only (w/o AdaPR & SPA). Training is initially stable
  but collapses after about 30 steps, indicating that AdaPR and SPA are necessary for stabilizing the
  training."
- **AdaPR timing:** "After about seven steps (once the replay buffer is populated), the gap between w/o
  AdaPR and the full MobileRL method grows, showing the benefit of replay."
- **SPA timing:** "Noticeable gains of SPA appear after roughly 60 steps, likely because the lack of
  step-wise control leads to overly long trajectories." SPA has the *smallest* effect on accuracy but is
  the clearest on **step efficiency** (Figure 5: MobileRL w/ SPA completes tasks in fewer steps across
  every complexity group, and wins the "both correct" and "only one correct" comparisons).

**Stage-wise success-rate gains (Table 3a, Qwen2.5-VL-7B):** base 27.6 → +Reasoning-Free SFT 50.2
(+22.6) → +Reasoning SFT 56.8 (+6.6) → +AdaGRPO 72.0 (+15.2) on AndroidWorld. For GLM-4.1V-9B-Base:
base 7.7 → +RFSFT 48.1 (+40.4) → +RSFT 66.2 (+18.1) → +AdaGRPO 80.2 (+14.0). **The single biggest jump is
always the cheap reasoning-free SFT** — confirming the "pre-pay exploration offline" thesis.

---

## 3. Magma-R1 — what it is, how it differs from MobileRL

**Where it's defined.** Magma-R1 is the model proposed in **AndroidControl-Curated** (arXiv:2510.18488);
the paper's headline is actually *benchmark purification*, with Magma-R1 as the demonstration that
"data quality > quantity."

**Training method (verbatim §2.2):** "we employ a policy optimization-based reinforcement learning
framework to train Magma-R1... We opt for an efficient policy gradient algorithm, which we term
**Generative REINFORCE with Policy Optimization (GRPO).**" Objective = the clipped-PPO-style GRPO
objective (Eq. 1); advantage `A_i = (r_i − mean{r})/(std{r} + δ)` (batch-normalized in-batch baseline).
It's **RL only** — there is **no SFT stage** described; it post-trains directly on a pretrained
multimodal base with GRPO.

**Reward signal — the key design:** a **smooth 2D-Gaussian-kernel grounding reward** (§2.2.1), replacing
binary hit/miss:
> "we design a smooth reward function based on a 2D Gaussian kernel, which provides a dense and
> continuous reward signal for grounding tasks." `R_grounding(B_pred, p_gt) = exp(−‖c_pred − p_gt‖²/(2σ²))`,
> where `c_pred` is the center of the predicted box and `p_gt` the ground-truth point.
- Three stated advantages: **Reward Density** (any prediction gets reward in (0,1]); **Smoothness**
  (continuous & differentiable, "smooth optimization landscape"); **Physical Intuition** ("a click
  slightly off-center but still on the target element is far better than a click that misses entirely").
- This is a *per-step grounding* reward, **not** an end-to-end task-success reward. The output prompt
  (from their GitHub `prompt.py`) is: given a screenshot + task (+ past_actions), emit
  `{"bbox_2d":[x1,y1,x2,y2], "action_type": ...}` for the *next single action*. So Magma-R1 optimizes
  **grounding/precision of one action**, while MobileRL optimizes **multi-turn task success**.

**Balanced learning / class imbalance (§2.2.2):**
> "GUI operation datasets commonly suffer from a severe class imbalance problem, where 'click/tap'
> actions constitute the vast majority, while other critical actions like 'type', 'scroll', and 'swipe'
> are relatively rare. This long-tailed distribution causes the model to over-focus on high-frequency
> actions during training, neglecting the learning of low-frequency yet crucial skills."
- Fix: **Action Type Proportional Optimization** — stratified sampling from data pools by action type into
  a target distribution `P_target` (which over-samples `type`, `scroll`, etc. relative to the empirical
  `P_orig`), so each batch better approximates `P_target`.

**Base model — and an important discrepancy.** The paper (§3.1, "Compared Models") says: "Magma-R1
(Ours): Our proposed model, based on a **3B-parameter multimodal model** [Wu et al., 2024 = MobileVLM]."
The abstract calls it **Magma-R1-3B**. **But** the HF API for the `batwBMW` org returns two released
checkpoints, both named **`Magma-R1-4B-...`** and both tagged `qwen2_5_vl` (and a later
`..._Based_Qwen3VL` with `apache-2.0`). So there's a genuine **mismatch** between the paper's stated
"3B MobileVLM-base Magma-R1-3B" and the publicly released "4B Qwen2.5-VL/Qwen3-VL-base Magma-R1-4B"
checkpoints. I could **not** verify a released 3B MobileVLM-base checkpoint. This is **uncertain** — flag
it for the parent.

**The actual headline result (verbatim abstract):** "Magma-R1-3B, post-trained on just **2.4k curated
samples** using **60 hours of an H20 GPU (approximately $60)**. Despite being **200 times smaller** in
parameters, this model delivers performance comparable to Qwen3-VL-235B." On AndroidControl-Curated-Hard,
Magma-R1 gets Type 84.2%, Grounding 84.8%, SR 75.3% vs. Qwen3-VL-235B's 88.2/**83.6**/76.5; on Easy it's
91.3/94.2/**88.0** vs 85.1/82.9/74.5. It matches **Infi-GUI-R1 trained on 31k** samples.

**Why this design (their reasoning):**
1. **Reward sparsity** is the core problem in GUI grounding; a binary reward gives "no effective learning
   signal" for near-misses (§2.2.1). So they make the reward **dense and continuous**.
2. **Data imbalance** is the other core problem; the action distribution is long-tailed, so they **stratify
   the sampling** (§2.2.2).
3. **Data quality trumps quantity** (their central thesis): "training with larger, seemingly
   higher-quality datasets from the benchmark sometimes led to reduced accuracy" (§1), because ~30% of
   AndroidControl had ambiguities/factual errors. They show "the benchmark was actively teaching incorrect
   behaviors" (§3.4.1) and that fixing the benchmark (not scaling the model/data) is what unlocks SOTA.
   Hence a small, clean, RL-optimized model beats a big model on dirty data.

**Why the single-stage GRPO + dense reward is enough for them but probably not for a full agent:** Magma-R1
optimizes **grounding precision on curated single-turn data**, not long-horizon planning with sparse,
deferred success. Its reward is *immediate and dense* (one box per step), so it doesn't need the staged
warm-up or the difficulty/replay machinery that MobileRL needs for long-horizon, sparse-reward, multi-turn
tasks.

---

## 4. The "why it works" insights (verbatim kernels)

**MobileRL:**
- Sparse reward ⇒ replay rare successes: "In sparse-reward mobile environments, successful yet challenging
  rollouts are rare but highly informative; effectively leveraging them enhances the learning signal and
  stabilizes policy updates." (§2.2.2)
- Difficulty-adaptive sampling ⇒ don't waste budget: "To avoid repeatedly sampling tasks that yield zero
  reward—which wastes computation and hinders the collection of positive advantage data, we propose failure
  curriculum filtering." (§2.2.3)
- Failure curriculum as a *simplified* curriculum: "This method can be regarded as a variant of curriculum
  sampling... To avoid excessive hyperparameter tuning, we simplify it to target only the exclusion of the
  most difficult subset of data." (§2.2.3) — i.e., they deliberately avoid a fussy curriculum and just
  drop hard dead-ends.
- Shortest-path reward as incentive for efficiency: uniform terminal reward "biases training toward longer
  trajectories, since they contribute more gradient terms" ⇒ SPA. (§2.2.1)
- Pre-pay exploration offline: "reasoning SFT enhances the handling of long and compositional instructions,
  reduces costly on-policy trials in mobile simulators, and enables the broad use of open or human-labeled
  datasets without relying on proprietary models." (§1 abstract)
- RL is for error-recovery the imitator can't do: imitation "suffer[s] from limited behavior coverage and
  poor error recovery"; RL with verifiable rewards is the alternative (§1).

**Magma-R1:**
- Dense reward solves sparsity: binary "hit-or-miss" rewards give "no effective learning signal"; a
  2D Gaussian reward gives "reward density," "smoothness," and "physical intuition." (§2.2.1)
- Benchmark/data quality, not model scale, is the bottleneck: "benchmark quality, not model architecture
  limitations, was the primary bottleneck preventing the practical deployment of on-device GUI agents"
  (§3.4.1); "the benchmark was actively teaching incorrect behaviors" (§3.4.1); "Our work demonstrates
  that data quality fundamentally trumps data quantity in training high-performance GUI agents" (§5).
- "data quality > quantity" as a falsifiable claim: trained on 2.4k curated samples, matches models
  trained on >31k less-curated samples.

---

## 5. Action space & how grounding is trained

**MobileRL action space (Table 5, §B.2):** a **finite atomic set** — `Tap`, `Type`, `Swipe`
(direction up/down/left/right; dist short/medium/long; optional bbox), `Long Press`, `Launch(app)`,
`Back`, `Home`, `Wait`, `Finish(message?)`. Grounding boxes are **aligned to the XML accessibility-tree
bounds.** It's a merged space from AndroidLab + AndroidControl (§B.2). State = screenshot + condensed XML
(Appendix C: drop off-screen nodes except when `remain_nodes=True`, drop non-functional nodes, keep only
`True` boolean attributes, drop index/resource-id/package, keep last component of class, keep full bounds;
example: `TextView;;Audio Recorder;[221,1095][858,1222]`). At inference the agent emits the **relative
bbox** format (`is_relative_bbox: true`) and a one-line pseudo-code action, in a
`do(action=..., element=[x1,y1,x2,y2])` or `finish(message=...)` form, wrapped in
`<answer>...</answer>` (AW_SYSTEM_PROMPT_THINKING also requires a `<think>...</think>` rationale first).

**How grounding is trained in MobileRL:** grounding is **learned in the SFT stages** (the model is
supervised to output correct element bboxes from the expert trajectories + XML), and then **further
refined indirectly by RL** through the *terminal success reward*, not by a dense coordinate reward —
during RL, the agent wins only if the whole trajectory succeeds, and a correct box is a means to that end.
There is **no per-step grounding reward** in MobileRL's RL (unlike Magma-R1). Grounding quality comes from
reading the XML bounds in the state and from expert imitation.

**How grounding is trained in Magma-R1:** grounding is trained **directly via the dense Gaussian RL
reward** (§2.2.1) — the model predicts a bbox and is rewarded by how close its center is to the
ground-truth point. That's the *only* reward. This is a fundamental difference: MobileRL rewards
*episodic success* and gets grounding "for free"; Magma-R1 rewards *per-step grounding precision*.

**Action-space lesson for building a small grounded decision model:** keep the action space **small,
closed, and XML-anchored** (MobileRL) OR **closed + JSON `{bbox_2d, action_type}`** (Magma-R1). Both
teams deliberately make the *output space* tractable so a sub-9B / sub-4B model focuses on choosing and
grounding rather than figuring out the actor's API.

---

## 6. Training-design lessons for a small (<3B) grounded next-action decision model

Extracted from how the two teams actually think (some of this is my **synthesis/inference** — marked as
such, not verbatim):

1. **Order SFT before RL, and make the first SFT cheap but large.** MobileRL's biggest single jump is the
   cheap 97.9k-step reasoning-free SFT (e.g. +22.6 AndroidWorld for Qwen-7B; +40.4 for GLM-9B). For a
   <3B model, get a legal-action emitter and XML-grounded boxes working in SFT first; don't ask RL to
   discover the action format. *[synthesis]* The reason is purely cost: RL rollouts in any live env are
   the bottleneck, so pay for coverage offline.

2. **Add a reasoning/planning SFT, but keep it small and bootstrapped.** MobileRL's reasoning SFT is only
   ~24k steps, built by self-refinement, and adds +6.6/+18.1 on AndroidWorld. It's cheap insurance for
   "long, compositional instructions" and makes the policy "transparent." Doing it after the action
   foundation is what keeps it stable. *[verbatim reasoning + synthesis]*

3. **Design the reward against the failure mode you actually have.**
   - **If your task is long-horizon with a terminal success signal** (agentic), use **sparse terminal
     reward + trajectory-level advantages** (GRPO), and **shape it for efficiency** with something like
     SPA so long trajectories aren't accidentally rewarded; *do not* broadcast the terminal reward to every
     step (it biases toward longer, more verbose paths — MobileRL §2.2.1).
   - **If your task is a single next-action grounding step** (like Magma-R1), use a **dense, continuous,
     distance-based reward** (2D Gaussian). Binary hit/miss gives "no effective learning signal" for
     near-misses. *[verbatim]*
   - Rule of thumb: dense reward for localization precision, sparse+shaped reward for end-to-end success.

4. **Handle class imbalance / long tails explicitly.** MobileRL prunes negative trajectories (max 1:2
   pos:neg) and filters unsolvable tasks; Magma-R1 stratifies sampling by action type toward a balanced
   `P_target`. For a small grounded model, the "type" distribution is heavily `click`-dominated, so
   over-sample the rare `type`/`swipe`/`long_press` actions. *[verbatim both]*

5. **Treat difficulty as a curriculum but keep it boring.** MobileRL's FCF is deliberately *simple*:
   "To avoid excessive hyperparameter tuning, we simplify it to target only the exclusion of the most
   difficult subset of data." You don't need an elaborate curriculum — just drop/reweight the dead-ends
   (2 epochs of all-fail → cooldown ×3 → drop). *[verbatim]*

6. **Replay rare successes.** In sparse-reward RL, replay a buffer of high-advantage trajectories mixed with
   on-policy samples (AdaPR: `q = γ·p_B + (1−γ)·p_on`, push top-κ by advantage). It "stabilizes policy
   updates" and only matters once the buffer fills (~7 steps in MobileRL). *[verbatim]*

7. **RL is for correction and calibration, not for discovering behavior.** Both teams treat RL as the final
   *fine-tuning* layer over an already-working policy. MobileRL says imitation alone "suffer[s] from limited
   behavior coverage and poor error recovery." Put RL last; let it push an already-competent policy toward
   better success rate and shorter paths. *[verbatim + synthesis]*

8. **For grounding calibration specifically (a <3B model making grounded next-action choices):** the densest,
   most directly-controllable signal is a **coordinate-distance reward** (Magma-R1's Gaussian), because it
   gives *graded* feedback on how close a box center is to the target. This is the mechanism that makes a
   small model *well-calibrated* at grounding without needing a huge reasoning chain. Add SFT on
   XML-anchored + point-mapped data (MobileRL) so the model knows the coordinate frame. *[synthesis from
   §2.2.1 + §C/D]*

9. **Reward-model vs rule-based reward — a reliability caveat MobileRL states in their Limitations (§E):**
   "we believe that evaluating whether a broader range of tasks has been successfully completed without
   relying on rule-based rewards remains an important challenge for future work." Their training curve for
   AndroidLab (reward-model-based) is less smooth than AndroidWorld (rule-based). **For a small model, if
   you can define a rule-based verifiable reward, use it; a learned reward model injects bias**
   ("a general and unbiased verification model is of significant importance"). *[verbatim]*

10. **Watch your compute envelope.** MobileRL: ~256 stable parallel rollouts, >10 min/step, 100 steps
    >25 h, each AVD >5 GB RAM / >30 GB storage. Magma-R1: 60 h on one H20 GPU ≈ $60 for 2.4k samples.
    For a <3B model, the gradient is toward **high-quality, small data + cheap single-GPU RL**, which is
    exactly Magma-R1's play, versus MobileRL's heavyweight multi-emulator infra. *[verbatim both]*

11. **Don't trust your benchmark blindly; it may be teaching the wrong thing.** Magma-R1's deepest insight
    (and the hardest to replicate honestly): "the benchmark was actively teaching incorrect behaviors"
    (§3.4.1); ~30% of AndroidControl had ambiguities/multiple valid actions/wrong ground truth. If your
    "grounded next-action" data has wrong or ambiguous labels, RL/SFT will bake in the error. Validate a
    subset and use bounding-box intent alignment (not exact-point matching) for grounding evaluation —
    exact-point matching "assesses geometric proximity, not whether the model's interaction intent is
    correct" (§2.1.1). *[verbatim]*

---

## Coverage Status

**Directly read (verified):**
- MobileRL full text + all appendices A–E (arXiv:2509.18119 v2 HTML): training method, AdaGRPO details,
  hyperparams (Table 4), action space (Table 5), data (Table 7), reward model, resource limits, ablations.
- MobileRL released repo (THUDM/MobileRL): configs (`AW_SYSTEM_PROMPT_THINKING`,
  `is_relative_bbox`, `temp 0.0`, `max_rounds 50`), agent/env scaffolding.
- AndroidControl-Curated / Magma-R1 full text (arXiv:2510.18488 v1 HTML): GRPO objective, Gaussian reward
  (Eq. 3), action-type stratified sampling (Algorithm 1), purification pipeline, results/ablation.
- Magma-R1 released repo (githubzuoyi/Magma_R1) + HF API (`batwBMW` org): eval prompt (action space
  `bbox_2d` + `action_type`), and the released model IDs/sizes/tags.

**Uncertain / could not verify:**
- **"ICLR 2026"** claim for MobileRL (arXiv page does not state this); treat as unconfirmed.
- **Magma-R1 base model / size.** Paper says "3B multimodal model (MobileVLM)" and calls it `Magma-R1-3B`;
  the HF-released checkpoints are named `Magma-R1-4B-AndroidControl` and tagged `qwen2_5_vl` (plus a
  later `_Based_Qwen3VL`). I could not verify a released 3B MobileVLM-base checkpoint. **Discrepancy —
  needs follow-up.** This matters for the parent's <3B target.
- Training code for Magma-R1's GRPO loop / MobileRL's full AdaGRPO loop is **not public** (MobileRL uses a
  closed `agentrl.worker`; Magma-R1 repo releases only eval + benchmark, not the training loop). So exact
  GRPO batch sizes / epochs for Magma-R1 are **not stated in what I read** (only the algorithm + reward +
  sampling are given).
- MobileRL's exact stage durations (number of RL steps) and final stage-wise compute totals beyond the
  "100 steps >25 h" note are not fully enumerated.

**Not completed:** I did not fetch the Magma-R1 (2510.18488) PDF vs HTML for any differences, and did not
pull the HF dataset card for AndroidControl_Curated (would refine the "2.4k curated samples" composition
further). These are follow-ups if the parent wants them.

---

## Quick evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | MobileRL (Xu et al.) arXiv v2 | https://arxiv.org/html/2509.18119v2 | 3-stage pipeline (RSFT→RSFT→AdaGRPO), SPA/AdaPR/FCF, 80.2% AndroidWorld / 53.6% AndroidLab | primary (paper) | high |
| 2 | MobileRL Appendix B (hyperparams) | https://arxiv.org/html/2509.18119v2 | Table 4 hyperparams; Table 5 action space; resource limits | primary | high |
| 3 | MobileRL released repo | https://github.com/THUDM/MobileRL | AW_SYSTEM_PROMPT_THINKING (think+answer format), is_relative_bbox, temp 0.0, max_rounds 50 | primary (code) | high |
| 4 | AndroidControl-Curated / Magma-R1 arXiv v1 | https://arxiv.org/html/2510.18488 | GRPO + 2D-Gaussian grounding reward + action-type proportional sampling; 2.4k curated samples vs 31k | primary (paper) | high |
| 5 | Magma-R1 HF API (batwBMW org) | https://huggingface.co/api/models?author=batwBMW | Released checkpoints are `Magma-R1-4B-*`, tagged `qwen2_5_vl` (3B MobileVLM base not found) | primary (artifact) | high (for what exists) |
| 6 | Magma-R1 eval prompt / GitHub | https://github.com/githubzuoyi/Magma_R1 | Action space = `{"bbox_2d":[x1,y1,x2,y2],"action_type":...}`; evaluation via bounding-box intent | primary (code) | high |
| 7 | MobileRL-9B model card | https://huggingface.co/xuyifan/MobileRL-9B | (referenced in repo; not independently fetched) | secondary | low |
| 8 | Qwen2.5-VL / GLM-4.1V backbones | https://arxiv.org/abs/2502.13923 , https://arxiv.org/abs/2507.01006 | Base models used by MobileRL | primary | medium |
