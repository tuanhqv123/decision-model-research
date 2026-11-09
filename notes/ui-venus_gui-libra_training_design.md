# Training-Design Deep-Dive: UI-Venus (2508.10833) & GUI-Libra (2602.22190)

**Sub-agent research note — how the researchers THINK about the training pipeline.**
Sources read: full HTML v2 of both arXiv papers (verbatim quotes with section numbers). Both verified on arXiv.
UI-Venus: https://arxiv.org/abs/2508.10833v2 (Ant Group, 14 Aug 2025)
GUI-Libra: https://arxiv.org/abs/2602.22190v2 (UIUC/Microsoft/UNC, 25 Feb 2026, 57pp)

---

## 0. TL;DR — the two design philosophies in one line each

- **UI-Venus** = "RFT is a *discriminative-supervision* fix for SFT's wrong loss, so go *straight* to GRPO and craft a **fine-grained action-wise reward**; then fix the *data* (cleaning, self-evolving history, sparse-action upsampling) because RL only learns what the distribution lets it."
- **GUI-Libra** = "Generic RLVR post-training breaks on GUI because of **reasoning-vs-grounding interference** and **partial verifiability**; so rebuild SFT as **action-aware** and make RL **conservative** (KL trust region + success-adaptive negative scaling) so that offline step-metrics actually track online success."

The two papers are complementary and directly relevant to a `<3B` grounded next-action model on AndroidControl: **UI-Venus gives the reward-engineering and class-imbalance machinery; GUI-Libra gives the reasoning/grounding trade-off and the RL-stability (KL/SNGS) machinery.**

---

## 1. UI-Venus — the ENTIRE training pipeline (stage by stage)

### 1.1 Base and the big design call: RFT, not SFT, and NO separate mid-training/SFT stage reported

UI-Venus releases **four** models — `UI-Venus-Ground-7B/72B` and `UI-Venus-Navi-7B/72B` — all trained **with GRPO on the Qwen2.5-VL model** (§1). Crucially the report does **not** describe an intermediate SFT stage or a pretraining stage of its own: they take Qwen2.5-VL and apply **reinforcement fine-tuning (RFT) directly**. The paper's §1 gives the *why* of skipping SFT:

> "Although SFT is effective in many generation tasks with strong instruction-following ability, sometimes it is not appropriate in discriminative tasks like UI agent. For example, in UI grounding tasks, a predicted point is considered correct as long as it falls within the ground-truth bounding box. In contrast, SFT assigns widely-used cross-entropy loss penalty to any predicted points within and without the box except the center point."

This is the central thesis: **SFT's per-token cross-entropy is the wrong loss for anything that only needs to be "inside a box."** It over-penalizes every non-center coordinate. RFT with a set-based/interval reward gives *equal credit* to all points inside the box. **Note (inference):** UI-Venus therefore treats grounding and navigation as *pure RL problems from a frozen-ish VLM init*, relying on the RL reward to shape behavior, rather than an SFT-warmstart-then-RL recipe. This contrasts sharply with GUI-Libra which explicitly argues you *need* an SFT stage first (see §4).

Also in §1, the separate-grounding-vs-navigation decision is justified by *reward conflict* — see §5.

### 1.2 Stage structure overall

There is no explicit "stage 1 / stage 2" in UI-Venus. The methodology (§3) organizes around **two task branches** that share the same GRPO engine but have entirely different data pipelines and reward functions:

1. **UI Grounding branch**: data collection → data cleaning → GRPO w/ `format` + `point-in-box` reward.
2. **UI Navigation branch**: data collection → 3-stage data pipeline (Filter / Trace-Reconstruction / Iterative Trace-Generation) → **Self-Evolving Trajectory History Alignment** + **Sparse Action Enhancement** (applied between epochs) → GRPO w/ 4-part `action-wise` reward.

### 1.3 UI Grounding data + cleaning

- Sources: Widget Captioning, UI RefExp, SeeClick, ShowUI, OmniAct (mobile+desktop+web).
- **627k raw** open-source grounding samples → **~107k** used for training after cleaning (Table 1).
- Cleaning is **manual**, and §3.2.2 explicitly says automated cleaning *failed*:
  > "Failed attempts for automatic grounding data cleaning include: (1) Using open-source models to perform the above filtering and rewriting operations; (2) RFT with hard samples selected by rejection sampling strategies. Although these approaches failed to work..."
- Noise estimate (~§3.2.2): "approximately **40%** of the current open-source grounding data contain significant noise issues, including prompt ambiguity and box shifts." Root cause: HTML/A11Y-derived boxes can **shift after re-render**; nested elements break the one-to-one box–instruction correspondence (many-to-one / one-to-many).
- Cleaning steps: remove repeated prompts (downsample), manually filter ambiguous prompts, **relocate offset boxes**, rephrase unmeaningful instructions.

### 1.4 UI Navigation data

Sources (Table 2): GUI-Odyssey, Aguvis-AITZ, **AndroidControl**, AITW, AMEX + **~20k self-collected** Chinese mobile-APP samples (`Navi*`). Also **integrates grounding data** to improve click accuracy (§3.3.1). After the pipeline: **~350k** high-quality nav samples.

**Unified action space** (Table 3): they follow UI-TARS but *modify it* (e.g., redefine `CallUser` for information-retrieval) so mixed-source data maps to one space.

**3-stage data pipeline (§3.3.2):**

1. **Data Filtering** — remove overly short traces; **standardize scroll-direction definitions** across datasets; use an MLLM to summarize each action and integrate into an overall trace description to compare against the task (catches "more/less operations than required" or traces not following the task); categorize by app/subtask and **resample** to prevent overfitting to specific scenarios.
2. **Trace Reconstruction** — information-retrieval tasks often *lack an explicit answer* at the final step (e.g., "What's the total price?"). They MLLM-generate the answer from the last screenshot and **insert a `CallUser` step carrying the answer before `Finish`**, teaching the agent to *report* the answer, not just navigate to it.
3. **Iterative Trace Generation** — use the trained UI-Venus-Navi itself to generate new trajectories on real cloud emulators with hand + MLLM instructions (Chinese/English). Filter by three layers: (a) **rule-based** (short w/ abnormal exit, repeated invalid actions), (b) **ORM-based** (an outcome reward model scores the whole trajectory; drop low scores), (c) **annotator-based** (humans pick correct; **failures are fixed by annotators and added back with their valid prefix** → learning from failures).

### 1.5 GRPO preliminaries (§3.1)

Standard GRPO, group-relative advantage:
`Â_i = (r_i − mean({r_1..r_G})) / std({r_1..r_G})`, objective with clip (ε) and KL(β) toward a reference (Eq. 1). No critic.

### 1.6 Training hyperparameters (exact, Table 4 + §4.1)

**Grounding:**
- lr = **4e-7**; rollout sample = **8** (7B) / **10** (72B); global batch = **128**; **128 PPU-GPUs** (EasyR1 framework, github.com/hiyouga/EasyR1); ~**1.5 days** (7B) / **10 days** (72B) for ~one epoch.

**Navigation:**
- lr = **4e-7**; rollout sample = **8**; global batch = **256** (7B) / **512** (72B); **256 / 512 PPU-GPUs**; ~**1 day** (7B) / **8.5 days** (72B) per epoch.

> Practical signal: UI-Venus uses a **very low LR (4e-7)** and small rollout (8) with **long horizon** per epoch. Very different from GUI-Libra's 1e-5 (SFT) / 1e-6 (RL) — see §6.

---

## 2. UI-Venus — the FOUR-part action-wise reward function (§3.3.4, Eq. 8–11)

This is the heart of the paper. They explicitly reject a *single* IoU/distance reward for navigation:

> "While effective for coarse-level correctness, such rewards provide limited guidance for fine-grained UI manipulation... In UI navigation tasks, the agent model is required to select the correct action type and generate precise action parameters... even small parameter errors can cause task failure."

So they factor the reward **per action dimension**:

**Final reward (Eq. 11):**
`R = R_format · w1 + (R_type + R_coord + R_content) · w2`

(weights w1/w2 trade structure vs. parameter precision. **Not numeric-value-reported in the paper** — I note this as unreported.)

**1. Format Reward (`R_format`)** — checks the output follows the template (reasoning block in `<think>`, action block in `<action>`, in the right order). Presence/ordering of tags → base reward. Rationale: "the reasoning process provides useful historical context for subsequent decision-making."

**2. Action Type Reward (`R_type`)** — binary 1/0 match of predicted vs. ground-truth action type. "Direct and interpretable."

**3. Coordinate Reward (`R_coord`)** — the key innovation. **Point-target actions (Click, LongPress):** stepwise (partial-credit) reward on pixel distance `d` (Eq. 8):

```
R_coord = α          if d < δ2
        0.5·α        if δ2 ≤ d < δ1
        0            otherwise
```

> "Prior methods often assess correctness by checking whether this distance is within a fixed threshold, which can be somewhat rigid and insufficient for distinguishing near-misses from large deviations. To address this, we introduce a stepwise reward strategy that grants higher scores for predictions closer to the target, while still awarding partial credit for reasonably accurate locations."

**WHY they did it:** a fixed-ε binary reward treats a 5px miss and a 200px miss identically (both 0); the model gets *no gradient signal* to get closer. The stepwise function gives **partial credit for nearby but not exact** predictions, "encourages the model to incrementally improve its coordinate parameter localization." (Note: this is a **shaped/continuous dense-ish grounding reward** inside the RL — contrasts with UI-Venus's *own* pure-grounding branch which uses binary point-in-box, see §3.2.3. So: binary for pure grounding, stepwise for grounding-inside-navigation.)

**Scroll reward (`R_scroll`)** for Scroll actions (Eq. 9) — checks **both endpoints AND direction**:

```
R_scroll = 1.5·β   if d_start < δ3 and d_end < δ3 and direction match
          β        if d_start < δ3 and direction match
          0.5·β    if d_start < δ3 OR direction match
          0        otherwise
```

> "we incorporate both spatial accuracy at the start and end positions as well as the correctness of the scrolling direction. The direction refers to the scrolling orientation (up, down, left, right), and a direction match indicates that the predicted orientation exactly matches the ground-truth."

**WHY:** scroll is not a point — it has a start, an endpoint, and a *semantic direction*. A scroll that starts in the right place but scrolls the wrong way is worse than one that starts slightly off but scrolls the right way. The tiers reward start-location + direction most, end-location adds bonus.

**4. Content Reward (`R_content`)** — for text-input actions (Type) using **token-level F1** (Eq. 10):

```
R_content = γ   if F1-score ≥ 0.5
            0   otherwise
```

**WHY:** text isn't exact-match in general (typing "hello world" vs "hello world " or a substring). F1 captures precision/recall overlap; the ≥0.5 threshold gives credit for substantial-but-imperfect text.

---

## 3. UI-Venus — "Sparse Action Enhancement" (and Trajectory History Alignment) — §3.3.3

**This is the piece most directly relevant to your collapse-on-majority problem.**

Observation: the action distribution in the data is **long-tailed** (Figure 4): `Click`/`Scroll` frequent, `LongPress`/`CallUser` rare.

> "For sparse actions (e.g., LongPress), the agent model often struggles to learn these actions effectively due to their low frequency in the data. In certain complex trajectories, these actions can be pivotal, and missing them may cause the entire task to fail."

### The exact mechanism (Algorithm 1 + Cartesian product)

The mechanism is coupled to **Trajectory History Alignment**, which runs **between epochs**. The alignment step's thought-pool is *reused* as a generator of variants:

1. **Rollouts for thoughts (Eq. 5):** for each step *n*, run `R` rollouts of the current policy to get candidate `(thought, action)` pairs: `{(t^(r)_k,n, a^(r)_k,n)}_{r=1..R} ~ p_θ(t,a | x_k,n, H_k,n−1)`.
2. **Action-exact-match filter (Eq. 6):** keep only rollouts whose *predicted action exactly matches ground truth*: `C_k,n = { t^(r)_k,n | a^(r)_k,n = a^(gt)_k,n }`. This forms a **thought pool** per step (thoughts that led to the *correct* action).
3. **History Alignment (Eq. 7):** for step *n*, replace each prior thought `t_k,i` with `Select(C_k,i)` (e.g., the candidate whose length is closest to a target length) so the history is written in the *model's own* reasoning style, not whatever crowd-sourced/expert/synthetic source produced it.
4. **Sparse Action Enhancement:** for a step whose action is sparse (`a_n ∈ {LongPress, CallUser, ...}`), construct **M historical variants** by sampling thoughts from each prior pool and concatenating: `H^(m)_n−1 = [(t'^(m)_1, a_1), ..., (t'^(m)_n−1, a_n−1)]`.
   > "Conceptually, the augmented set corresponds to the **Cartesian product C_1 × ⋯ × C_n−1**. This increases both the diversity and frequency of reasoning patterns that lead to sparse actions. **For common actions, we do not perform this augmentation.**"

**WHY it works (their reasoning + my read):**
- The map from *reasoning* to a rare action is what the model needs to learn. By holding the rare action fixed and **varying the whole prior reasoning history** (the Cartesian product of every step's thought pool), you get *many* distinct "paths" that all terminate in the same rare action. That **reweights the training distribution** toward the rare action *and* teaches the model that the rare action is reachable from many contexts.
- It only fires for sparse actions (Click/Scroll are untouched), so it doesn't inflate common actions further — i.e., **it's targeted class balancing at the trajectory-history level**, not naive oversampling.
- It's **"free"** in the sense that it *reuses* rollouts the alignment step already paid for — no extra environment interaction.

> **Synthesis for the collapse-on-majority problem:** the lesson is not "oversample the minority class." It's **"varify and enrich the *contexts* that lead to the minority action, keeping the action fixed, so the model learns that action as a function of diverse states rather than memorizing a few (state→action) pairs."** For AndroidControl next-action selection, the analogous recipe: for rare actions (long-press, paste, copy, scroll-up), generate **multiple plausible histories/observation-contexts that legitimately precede the target action** and upsample those, while leaving the dominant `click`/`tap` samples at their natural rate. This attacks the collapse by **making the minority action predictable from many states**, not by adding raw copies.

### Self-Evolving Trajectory History Alignment (the "evolving" part)

> "we iteratively refine the thought–action histories between training epochs, aligning the form and level of detail of the historical reasoning with the agent's evolving decision-making patterns... After each training epoch, we perform global trajectory refinement by re-inferring reasoning traces with the current model."

The problem it addresses: history thoughts come from heterogeneous sources (crowd-sourced, expert, synthetic) → **inconsistent style/detail/abstraction** → the model is confused by its *own* history. "Shorter thought sequences may miss critical intermediate observations, while overly verbose ones may obscure key decision-making milestones."

---

## 4. GUI-Libra — the training design (ASFT → conservative GRPO)

### 4.1 The framing / two "fundamental issues"

> "(i) standard SFT with CoT reasoning often hurts grounding, and (ii) step-wise RLVR-tyle training faces **partial verifiability**, where multiple actions can be correct but only a single demonstrated action is used for verification."

**Why long CoT hurts grounding — §5.1 (measured):**
- Fine-tune Qwen2.5-VL-3B/7B; sample at temp 1.0; bin responses into 30-token bins.
- **Grounding accuracy has a clear *negative correlation* with response length** (Figure 4a) for both base and CoT-SFT. Longer → worse.
- CoT-SFT broadens the length distribution, producing many >250-token responses "associated with particularly severe performance drops."
- Three-variant comparison (Figure 4b): Grounding-only SFT = modest gain; SFT-without-CoT = slight degradation; **SFT-with-long-CoT = substantial drop.** The driver is *excessively long reasoning sequences*, not reasoning per se.

> **Their insight:** "as models may prioritize high-level semantic logic over precise spatial execution." Reasoning tokens **dominate the loss** and push the policy toward semantic abstraction, away from pixel-accurate localization.

### 4.2 Action-aware SFT (ASFT) — §5.2

Two mechanisms:

**(a) Mixed reasoning-then-action + direct-action supervision.** Strip reasoning (`<answer>…</answer>` only) for a subset of data so the model learns *both* modes.

> "Two complementary supervision modes: (i) reasoning-then-action and (ii) direct action prediction... it increases the amount of action-centric learning signal... it reduces reliance on verbose intermediate reasoning, alleviating grounding degradation induced by long CoT traces."

**(b) Action-aware token reweighting.** Split the `<answer>` tokens into **action tokens** (everything except `point_2d`) and **grounding tokens** (the `point_2d` field). The ASFT objective (Eq. 3):

```
L_ASFT(θ) = −E_{(x_t,c_t,a_t,g_t)~D_mix}
            [ log π_θ(c_t|x_t) + α_a·log π_θ(a_t|x_t,c_t) + α_g·log π_θ(g_t|x_t,c_t,a_t) ]
            / ( |c_t| + α_a|a_t| + α_g|g_t| )
```

- `α_a` and `α_g` upweight action and grounding tokens. Defaults (Table in §6.1): **α_a=2, α_g=4**. So grounding tokens get 4×, action 2×, reasoning 1×.
- Special cases: α_a=α_g=1 → plain SFT; α_a=α_g≫1 → CoT-free-ish; α_g≫α_a≫1 → grounding-only.
- **WHY:** "the action sequence contains both semantic components (action description, action type, value) and spatial components (coordinates), which can be weighted differently at the token level." Grounding gets the highest weight because it determines execution.

**Result (§6.3, Table 8):** ASFT lifts grounding accuracy in reasoning mode (7B: 79.0%→83.4%); but a residual reasoning-vs-no-reasoning gap remains (ASFT-3B: 4.8 pts; ASFT-7B: 3.4 pts), which is then **eliminated by RL** (RL gives higher grounding in *reasoning* mode than no-reasoning: GUI-Libra-7B 89.3% vs 88.5%).

### 4.3 Reinforcement learning under partial verifiability — §5.3

The core theory (Def. 5.1): the *offline* step reward `r̃(s,a) = 1{a = ã(s)}` is **partially verifiable**:
> "positive feedback is reliable, while negative feedback is ambiguous because A*(s) often contains valid actions beyond ã(s)."

They define two quantities controlling offline→online predictability (Theorem 5.2):
- **occupancy mismatch** `C(π)` = how far the policy's *online* state distribution drifts from the *offline* distribution `d_μ`; and
- **off-demo validity mass** `η̄_π` = probability mass the policy puts on *valid-but-not-demonstrated* actions.

Bound: `J(π) ≥ 1 − H·C(π)·(1 − M_off(π) − η̄_π)`.

**Why standard RLVR is easier to predict (Corollary 5.3):** standard RLVR is typically **single-step (H=1) and fully verifiable (η̄_π ≡ 0)** → one-step matching directly reflects correctness, no error accumulation. Multi-step GUI has **error accumulation + distribution shift + partial verifiability** → offline matching is a *weak* proxy.

### 4.4 KL trust region — §5.3.2 (why KL)

> "Many RLVR pipelines omit KL regularization for efficiency... we find it is crucial in partially verifiable, multi-step GUI settings. Intuitively, a **KL trust region constrains policy drift**, which in turn helps control the two quantities governing offline-to-online predictability."

Theoretical bounds (Eq. 8/9): a per-state KL constraint `KL(π‖π_ref) ≤ ε` bounds occupancy mismatch: `C(π) ≤ C(π_ref) + H√(2ε)/ρ`, and bounds off-demo mass: `η̄_π ≤ δ̄ + √(ε/2)`.
So a **KL trust region simultaneously limits state-distribution shift AND constrains how much probability mass can move away from the demonstrated action** → keeps `M_off(π)` a stable proxy for `J(π)`.

**Empirical (§6.4, Fig 9/10):**
- Without KL, **training reward climbs but eval degrades** (= reward hacking); **policy entropy collapses** (premature collapse/overfitting); offline↔online correlation weak (Pearson r=0.63, Spearman 0.53, rank not significant).
- With small KL (0.001/0.005), entropy stays stable; **Pearson 0.89 (p<1e-4), Spearman 0.83** — strong, robust offline→online alignment.
- **KL coefficient trade-off (§6.5, Table 9):** increasing KL improves **Pass@4** (keeps entropy) but can slightly drop **Pass@1**; **moderate (0.001) is best** — AndroidWorld 13.0→25.2; too large (0.05) → over-conservative (AndroidWorld ~20.0); zero → collapse.

> **WHY they say KL matters:** Unlike math RLVR where *dropping* KL helps, GUI agents need it precisely because rewards are ambiguous and errors compound — without it the policy drifts off-support and overfits the single demonstrated action. **This is a directly actionable calibration lesson for a small model.**

### 4.5 Success-adaptive Negative Gradient Scaling (SNGS) — §5.4

The problem: under partial verifiability, `r̃=0` is **ambiguous** — it conflates genuinely-wrong actions with *valid-but-uncredited* ones. Treating every non-match as equally negative overfits the demonstrator's single choice.

> "negative updates in policy-gradient methods such as GRPO remain useful for stabilizing training and avoiding premature collapse. Therefore, rather than suppressing all negative gradients equally, SNGS rescales them using a state-conditioned reliability signal estimated from the GRPO sampling group."

Mechanism: group success rate `p̂_g(s) = (1/G)Σ r̃_k`; scaling factor (Eq. 10):
```
λ_g(s) = min(λ0 + κ·p̂_g(s), 1)
```
Modify only negative advantages (Eq. 11): `Ã_k = A_k` if `A_k ≥ 0`, else `λ_g(s)·A_k`. With κ>0, λ_g rises with p̂_g — when the policy is *already* concentrated on the demo action, non-matches are more likely genuinely wrong → downweight negatives *less*, recovering standard GRPO. Model-specific (λ0, κ): 3B (0.9,0.5), 7B (1.4,-0.5), 4B (0.5,1.5), 8B (0.5,2.0) (Appendix B / Table 14).

**Effect (Table 10, 4B):** AndroidWorld 39.1→42.6 (+3.5), WebArena-Lite-v2 22.2→24.4 (+2.2); small/offline gains; minor tradeoff on AC-Low Pass@1. SNGS is **sensitive to hyperparameters**, so ablations use KL-only GRPO to isolate other effects.

### 4.6 Reward function implementation (§5.5, Eq. 12)

```
r̃(s,a) = w_fmt·r_fmt + (1−w_fmt)·r_acc,   w_fmt = 0.1
```
- `r_fmt` = valid `<think>…</think><answer>…</answer>` + parseable JSON.
- `r_acc = r_act · r_val · r_g` (multiplicative):
  - `r_act`: action_type matches.
  - `r_val`: word-level **F1 > 0.5** on the text value.
  - `r_g` (**grounding**): predicted point `u` **inside the demonstrated bounding box** `b*`: `r_g = 1{u ∈ b*}`.
- **w_fmt=0.1** → reward mostly about *action correctness*. Note it's an **AND (multiplicative)** of all three → a step scores 1 only if type AND value AND grounding all correct. (Contrast UI-Venus's **additive** `R_type + R_coord + R_content`, which gives partial credit per dimension. This is a real design fork: GUI-Libra is *stricter/AND*, UI-Venus is *credit-per-component/OR-ish*.)

### 4.7 GUI-Libra data curation (§4) — relevant for small datasets

- Aggregate GUI-Odyssey, AMEX, **AndroidControl**, AITZ, AITW, GUIAct, MM-Mind2Web + Chinese GUIAct subset. Initial clean: drop <3 or >50 steps, drop compound actions → **19K trajectories / 170K steps**.
- **Unified format** (Fig 2): input = system prompt (action list) + user instruction + interaction history + current screenshot; output = `<think>…</think>` + `<answer>…JSON…</answer>` (action_type, action_description, action_target, value, point_2d). **13 action types**: Click, Write, Terminate, Swipe, Scroll, NavigateHome, Answer, Wait, OpenAPP, NavigateBack, KeyboardPress, LongPress, Select.
- **Reasoning augmentation:** prompt GPT-4o/o4-mini/GPT-4.1 with GUI-specific guidelines (observation/reflection/planning) + format + action requirements. Finds reasoning quality is **sensitive to generator**; and importantly **does NOT force** the generator to follow the dataset action — treats annotation as a reference, may pick a different valid action.
- **SFT filtering (§4.1.4):** (1) **Agreement filtering via re-prediction** — run Qwen3-VL-8B-Instruct 10 stochastic times per input, measure match to annotation (type + coord proximity); **discard steps with re-prediction accuracy < 0.3**. (2) **Coordinate alignment via bbox verification** — prompt Qwen3-VL-32B to predict a bbox from `action_target`, keep step only if `point_2d` falls inside; this also yields **bbox annotations for RL**.
  → **81K SFT steps / 9K trajectories**.
- **RL filtering (§4.1.5):** downsample **early steps** (many trajectories share home-screen/landing actions — early-step bias) and downsample **mobile** domain → **40K-step RL subset**, more balanced.
- **Action distribution note:** "Click accounts for around 60% of steps, followed by Write, Terminate, and Swipe, while LongPress and Select are rare. This imbalance makes rare actions difficult to learn from SFT alone, **motivating a subsequent RL stage.**"

### 4.8 Overall pipeline (§5.6, Fig 5)

**Stage 1 (SFT):** ASFT — mixed reasoning/direct-action + action-aware reweighting (α_a=2, α_g=4), lr 1e-5, eff. batch 256.
**Stage 2 (RL):** conservative GRPO — VERL + EasyR1, lr 1e-6, global batch 128, group size 8, **clip ε=0.2**, **KL β=0.001 (0.005 for 7B)**, 300 iterations, rollout batch 256, temp 1.0 / top_p 0.98, 8×NVIDIA B200 (~16h 4B, ~20h 8B). SNGS on top.

### 4.9 Grounding-vs-navigation mixing is a real trade-off (§6.6, Table 11)

Mixing 20K grounding samples into RL nav training → **helps grounding** (ScreenSpot-v2 +2.3/+4.1, ScreenSpot-Pro +7.1/+5.8) but **hurts navigation** (MM-Mind2Web −5.2/−0.8, AC-v2 High −1.5... actually mixed; and AC-v2 Low −2.8/−2.3). "These results reveal **competing optimization pressures**: adding direct grounding supervision strengthens spatial alignment, but can reduce performance on reasoning-intensive navigation tasks." This corroborates UI-Venus's decision to keep grounding/nav separate (see §5).

---

## 5. KEY INSIGHTS / design philosophy (verbatim)

**Why grounding and navigation need DIFFERENT training (UI-Venus §1, "two motivations for publishing the grounding and navigation models separately"):**
> "(1) A More Efficient Grounding Model: ... explicitly prompting the model to output its thinking process significantly enhances its observation and planning capabilities in UI navigation tasks. However, UI-Venus achieves approximate performance on UI grounding tasks with and without thinking... (2) **Reward Conflicts**: If we attempt to train an agent using both grounding and navigation tasks, a feasible approach is to convert grounding data into single-step click actions in the navigation task. However, **using these two reward mechanisms simultaneously can lead to unstable training, ultimately degrading performance on both tasks.**"

**The grounding-vs-navigation conceptual split (GUI-Libra §3, "High-level vs. low-level"):**
> "A low-level task can be completed with a single atomic interaction... In contrast, a high-level task requires a multi-step interaction trajectory... **We view grounding as a special case of low-level decision making**, where the agent localizes the target UI element by predicting its interaction coordinates. In our paper, we mainly focus on high-level navigation tasks."

**The grounding↔navigation asymmetry (my synthesis, grounded in both):**
- **Grounding = localization reward.** It is **single-step, fully verifiable** (`r_g = 1{u ∈ b*}` / point-in-box), and benefits from **C**o**T-free/aversion** because long CoT hurts pixel precision (§5.1). Best reward = **binary point-in-box or stepwise distance**; no multi-step credit assignment needed.
- **Navigation = multi-step decision.** It needs **reasoning/history** (UI-Venus uses thought–action history; GUI-Libra shows CoT boosts online generalization — removing CoT at inference drops AndroidWorld 25.2→18.3 / 13.0→8.7). It faces **partial verifiability + error accumulation**, so it needs **conservative RL (KL, SNGS)** and **history alignment**, and higher **policy entropy**.
- Hence the training data/policy are essentially **orthogonal axes**: navigation wants a *reasoning-enabled, entropy-preserving* policy; grounding wants a *precise, reasoning-suppressed* policy. A single reward mixture tries to satisfy both and fights itself.

**GUI-Libra's stronger grounding-in-reasoning result (§6.3):** their ASFT+RL recipe achieves **higher** grounding in *reasoning* mode than no-reasoning (89.3% vs 88.5%) — evidence that the interference is *solvable* by design rather than fundamental, but it took both action-aware SFT *and* conservative RL.

**Why they avoid costly online RL (GUI-Libra §2.2 / overall):** "online RL is expensive to scale, requiring costly environment interaction and robust infrastructure." Their whole pitch: "carefully designed post-training and data curation can unlock significantly stronger task-solving capabilities **without costly online data collection**." And UI-Venus notes the "hallucination gap between the model's internal reasoning (think) and its final response (answer)" as an open problem (§5 Future Work).

---

## 6. MOST IMPORTANT — concrete TRAINING-DESIGN lessons for a `<3B` multimodal decision model doing **grounded next-action selection on AndroidControl**

These are the actionable takeaways. I mark each as **direct (paper-stated)** or **inferred (my synthesis)**.

### 6.1 Data reweighting for class imbalance — the single most relevant lever
- **Target the *context*, not raw copies (direct, UI-Venus §3.3.3 Sparse Action Enhancement).** For rare actions (long_press, paste, copy, scroll-up), build **multiple valid historical/observation contexts** that lead to the *same* rare action (the Cartesian-product idea), and upsample only those. Leave the dominant `click/tap` at natural frequency. This teaches the rare action as a *function of diverse states*, fighting collapse-on-majority without inflating the majority.
- **Two dataset-level imbalance fixes (direct):**
  - **Downsample the majority/over-represented slices.** GUI-Libra downsamples **early steps** (home-screen actions dominate) and **mobile-domain** trajectories for its RL set (§4.1.5). AndroidControl is a mobile/domain-heavy pool — consider downsampling the share of repetitive navigational `click`s and identical starting screens.
  - **Note the stated cause of imbalance in AndroidControl-like data (direct):** "Click accounts for around 60%... LongPress and Select are rare. This imbalance makes rare actions difficult to learn from SFT alone, **motivating a subsequent RL stage.**"
- **Filter to a smaller, cleaner set, not a bigger noisier one (direct, §6.5):** "focusing on a smaller but cleaner and less biased dataset can generalize better than using a larger, noisier corpus." For a <3B model, **data quality beats quantity** — agreement filtering (re-predict N times, drop <0.3 agreement) + bbox-consistency filtering.

### 6.2 Reward design for grounding
- **Use a localization reward that gives credit for "inside the box / near the box," not only exact center (direct).** UI-Venus grounding: **point-in-box** (`1{x1≤xc≤x2 and y1≤yc≤y2}`) — `R_point-in-box = 1 if inside else 0`; the exact rationale is that SFT cross-entropy wrongly penalizes all non-center points (§3.2.3).
  - For *next-action* (which is grounding next to a nav element), the **stepwise coordinate reward** (Eq. 8) is the better design: partial credit as distance decreases (`α` for `d<δ2`, `0.5α` for `δ2≤d<δ1`), so the model is pushed to *refine* coordinates instead of only being told pass/fail.
  - **For AndroidControl specifically**, note GUI-Libra's finding (~20% label error): "AndroidControl contains roughly **20% errors** in action types and coordinates." **Validate your labels** before trusting a reward on them — a reward that rewards a wrong box teaches the wrong thing. Their fix: bbox-verification filtering.
- **Reward composition fork (contrast/direct):**
  - UI-Venus: **additive** `R_type + R_coord + R_content` → partial credit per dimension, gentler, good when you want the model to make *incremental* progress (multi-step nav).
  - GUI-Libra: **multiplicative AND** `r_act · r_val · r_g` → strict (all three must be right to score). For a *small* model that must nail both action AND grounding in a single step, the multiplicative reward is tighter but can be **sparse/harsh**; if you see the model never improving, switch to an additive/partial-credit version (UI-Venus style) to densify the signal.
- **Downweight `format` (direct):** both papers keep format/parse reward small (UI-Venus `w1`; GUI-Libra `w_fmt=0.1`) so the reward is dominated by **action correctness**, not tag-shape.

### 6.3 SFT vs RL ordering
- **Two valid recipes, pick by data budget:**
  - **UI-Venus path (skip SFT):** go *straight* to GRPO on the VLM, rely on a rich reward. Works when data is clean and reward is dense. **For a <3B model, this is risky** — RL from a cold VLM with a sparse/harsh reward and no SFT warmstart tends to collapse (see §6.4).
  - **GUI-Libra path (ASFT → conservative RL):** do **action-aware SFT first** (warm the model to the action format, base grounding, and reasoning), **then** KL-regularized GRPO. This is the safer, proven route at 3B scale and is what I'd recommend for `<3B` AndroidControl: SFT to get the *skeleton*, RL to *refine*.
- **The demonstrable fact that SFT-then-RL matters (direct, §6.5 Table 9, Qwen2.5-VL-3B):** base 3.5 → SFT 5.2 → ASFT 13.0 → ASFT+RL(KL=0.001) **25.2** on AndroidWorld. RL adds a large chunk *on top of* a good SFT; SFT alone is weak.
- **If you include reasoning in SFT, do it action-aware (direct):** standard CoT-SFT hurts grounding. Use **mixed reasoning/direct** supervision + token reweighting (`α_a≈2, α_g≈4`). Isolation evidence: SFT-with-long-CoT causes a *substantial* grounding drop while grounding-only SFT gives a *modest gain*.

### 6.4 Calibration / RL stability (the collapse guardrails)
- **Do NOT drop KL (direct).** The single best anti-collapse lever for a small model: a **moderate KL trust region (β≈0.001–0.005)** keeps policy entropy from collapsing. Without it you get reward hacking (training reward up, eval down) and entropy collapse; with it, offline→online correlation jumps to Pearson 0.89. **Too large a KL is also bad** (over-conservative, β=0.05 degrades AndroidWorld to ~20); **moderate is the sweet spot.**
- **Scale the *negative* gradients, don't kill them (direct, SNGS).** Under partial verifiability, `r=0` is ambiguous. Downweight negative advantages by a state-conditioned reliability factor `λ_g = min(λ0 + κ·p̂_g, 1)` (p̂_g = group success rate). This prevents overfitting the single demonstrated action while still keeping negatives to stabilize. **But SNGS is hyperparameter-sensitive** — isolate it in ablations (run KL-only GRPO as the baseline).
- **Keep group size / rollout modest (both):** UI-Venus rollout 8, GUI-Libra group size 8, global batch 128 (RL). For a <3B model you don't need huge groups; **what stabilizes is the KL term, not more sampling.**
- **Low LR + long horizon (UI-Venus) vs higher LR (GUI-Libra):** UI-Venus uses lr **4e-7** for RL and trains ~1d/8.5d per *epoch* (few, expensive, long-horizon steps). GUI-Libra uses lr **1e-6** for RL / **1e-5** for SFT, 300 iterations, 16–20h on 8×B200. For a small model, GUI-Libra's numbers are more accessible; note SFT and RL want very different LRs (1e-5 vs 1e-6).

### 6.5 A concrete "starter recipe" distilled for `<3B` grounded next-action on AndroidControl
*(Inferred synthesis grounded in the two papers' mechanisms and numbers.)*
1. **Clean + validate AndroidControl labels** (it has ~20% label error — GUI-Libra). Apply agreement filtering (re-predict N=10, drop <0.3) and/or a bbox verifier to get a trustworthy `point_2d` and a bounding box per step.
2. **Build a balanced next-action set:** frame each (screen, history) → (action_type, point_2d, value) as a single "grounding next to nav" step. **Downsample repetitive 'click' + early-step/home-screen samples**; **upsample sparse actions** (long_press, paste, copy, scroll-up) via **multi-context augmentation** (generate alternative valid histories leading to the same rare action — UI-Venus Cartesian-product idea).
3. **ASFT warmstart:** mix reasoning-then-action and direct-action; reweight tokens (`α_a≈2, α_g≈4`), lr 1e-5, small effective batch (256). This must NOT be plain CoT-SFT.
4. **Terminal objective / reward:** for each step, reward = format(small) + action-type + **stepwise coordinate** (partial credit) + value-F1. Start **additive** (UI-Venus) for denser signal; validate against a strict "all-correct" measure to check you're not gaming the metric. Don't let reward run on a mass-`click` slot machine — reward *grounding* for those steps.
5. **Conservative GRPO:** KL β≈0.001 (tune to 0.005 if 7B-ish), clip 0.2, group 8, global batch 128, lr 1e-6, ~300 iters; add **SNGS** with κ>0; **evaluate the offline→online correlation** (Pearson) of your checkpoints and stop at the best-correlated one.
6. **Decide on reasoning at inference:** if you keep CoT, you *must* have done action-aware SFT + RL, or grounding will degrade. For a small model doing next-action, consider **no-reasoning (direct-action) inference** but **train with CoT** — training-with-CoT beats training-without-CoT even when inference is CoT-free (§6.5).

**Let me be explicit about what is NOT verified / still to check:**
- UI-Venus does not report numeric values for reward weights `w1/w2`, the stepwise thresholds `δ1/δ2/δ3`, or scale constants `α/β/γ` — I list the formulas but **cannot confirm the numbers** (they're not in the text).
- Neither paper gives exact `ε` (clip) for UI-Venus; GUI-Libra uses clip 0.2.
- UI-Venus's 4-part reward is for its *navigation* model; its *pure grounding* model uses only `format + point-in-box`. So "point-in-box" is the grounding reward, "stepwise coordinate" is the nav reward.
- The claim that UI-Venus uses **no SFT stage** is an **inference** from §1 ("trained with GRPO on the Qwen2.5-VL model") + the SFT-vs-RFT framing; the report never explicitly enumerates a mid-training/SFT stage, which is a notable absence given the task's prompt assumption of "mid-training? SFT? Then RL?" — UI-Venus is best read as **pure RFT from the base VLM**, while GUI-Libra is **SFT→RL**.

---

## Coverage Status

**Checked directly (read from paper HTML):**
- UI-Venus: full main body + appendices (data pipelines, reward functions Eq.8–11 Algorithm 1, hyperparameters Table 4 & §4.1, navigation/grounding rationale, prompts). ✅
- GUI-Libra: §1 intro, §3 preliminaries, §4 data curation/filtering, §5 (ASFT §5.2, partial-verifiability theory §5.3, KL §5.3.2, SNGS §5.4, reward §5.5, pipeline §5.6), §6 experiments incl. ablations §6.5–6.6, Appendix B implementation tables (action space, SFT Table 13, RL Table 14), Appendix D grounding-as-single-step note. ✅

**Verified availability/format of artifacts (not yet checked):**
- UI-Venus **code** at https://github.com/inclusionAI/UI-Venus (claimed in paper) — **not opened** in this pass.
- GUI-Libra **dataset/code/models** at https://gui-libra.github.io (claimed in paper) — **not opened** in this pass; dataset "GUI-Libra-81K"/40K-RL subsets are named but **not quantitatively verified** (I only have the paper's own token/step stats).
- UI-Venus's exact `δ1/δ2/δ3`, `α/β/γ`, `w1/w2` — **unavailable in text**.

**Unresolved / open questions:**
- Whether UI-Venus's reward constants are in the GitHub repo (would need to open the repo).
- GUI-Libra's full ablations on SNGS across 7B (they note sensitivity) — partly covered, but model-specific (λ0,κ) values are only in Appendix B, not all ablated.
- For the parent's AndroidControl task, the **exact AndroidControl label-error rate** and the **reward constants** are the two things to pull from the actual repos/datasets.

## Sources

1. Gu et al. — *UI-Venus Technical Report: Building High-performance UI Agents with RFT* — https://arxiv.org/abs/2508.10833v2 (Ant Group).
2. Yang, Wu, Wang, Chen, Yang, Cheng, Yao, Peng et al. — *GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL* — https://arxiv.org/abs/2602.22190v2 (UIUC/Microsoft/UNC).
3. UI-Venus code repo (claimed): https://github.com/inclusionAI/UI-Venus — not opened this pass.
4. GUI-Libra project page (claimed): https://gui-libra.github.io — not opened this pass.
