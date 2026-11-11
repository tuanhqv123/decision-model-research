# Prior-art slice B — decoder-free GUI action policies (2026-11-11)

Scope: work that produces a GUI action **without autoregressive text decoding** — per-type classifier,
element/candidate scorer, contrastive retrieval, latent/predictive head. Question: has anyone already
done what contract v3 claims?

## 0. Verdict

| Claim | Verdict |
|---|---|
| **T-A** matched comparison: one-pass keeps AR competence at lower latency, loses it beyond a task length | **DUPLICATES (offline half) / CONFIRMS-NOVEL (live + length axis).** GUI-JEPA runs exactly the matched experiment — same inputs, same judge, same split, same GPU — a 0.50B decoder-free policy against a Qwen2.5-VL-7B SFT they train themselves, and reports 92.4%/86.1% of its in-distribution accuracy at 2.8–4.0× speed (§4.3, Table 1, Table 3). What is *not* published: the same comparison on a live benchmark, and the length-dependent crossover. |
| **T-B** discrete option set + verified-failure masking removes period-1 repeats, shortens wasted steps | **DUPLICATES (different carrier).** The mechanism and its effect are published twice on *generative* agents: VLAA-GUI Loop Breaker (§3.3: same action + unchanged screen ≥ τ → blacklist the repeated action, force a different one) and Absorbed-in-Inertia R3 (§3.2: same screen–action pair k times unchanged → reset context, reroute; −16.9 to −42.7% recurrence, all CIs below zero). No decoder-free paper does it; the free space is the distribution-level form (masking the softmax, no prompt, no context reset) plus a decoder-free measurement of period-1 repeats. |
| **T-C** offline confidence fails online; env-verified labels restore AUROC ≥ 0.70 | **CONFIRMS-NOVEL.** No source gates a policy on env-verified no-effect labels. Nearest: GUI-JEPA's top1–top2 margin flags its own step errors at **AUROC 0.74 offline** (a number we will be compared to, and the offline half of our own T-C), and Absorbed-in-Inertia's activation readout separates recurrence at AUC 0.72–0.95 — but that is a probe on internals, not a confidence signal, and it is not used online as a gate. |

Single most dangerous source: **GUI-JEPA** (arXiv 2610.36877). It occupies the T-A claim *and* is the closest
structural match to contract v3 (element-id choice over a11y candidates + span copy from the goal + one batched
forward pass, zero tokens). Second: **VLAA-GUI** (arXiv 2604.21375) for T-B.

## 1. Sources read

Full text read this session (HTML, not abstract) except where marked.

| id | title | venue+year | URL | citations | what it is |
|---|---|---|---|---|---|
| JEPA | GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy | arXiv preprint, Oct 2026 | https://arxiv.org/abs/2610.36877 · https://arxiv.org/html/2610.36877v1 | not shown on abs page | Decoder-free policy: scores a11y candidates by cosine on candidate-conditioned JEPA latents + a MiniLM text matcher |
| VDROID | V-Droid: Advancing Mobile GUI Agent Through Generative Verifiers | **MobiCom '26** (DOI 10.1145/3795866.3796681); arXiv v5 Feb 2026 | https://arxiv.org/html/2503.15937v5 | ~37 (scixplorer snippet, UNVERIFIED) | Prefill-only LLM verifier scores enumerated a11y action candidates; one token read out, none decoded |
| LWAM | Latent World Action Model for GUI Agents | arXiv preprint, Oct 2026 | https://arxiv.org/abs/2610.14228 · html/2610.14228v1 | not shown | Linear action head `w_kᵀh` over candidate actions on LMM hidden states; transition head dropped at inference |
| LIMAC | Lightweight Neural App Control (LiMAC / AcT) | **ICLR 2025** | https://arxiv.org/html/2410.17883v1 | not shown on abs | Action Transformer: 10-way type classifier + InfoNCE contrastive click-target selection over UI-element embeddings; VLM only for text arguments |
| JEV | Jev-Mobile: Jev as an Executor for Mobile GUI Agents | arXiv preprint, Sep 2026 (single author) | https://arxiv.org/html/2609.30186 | not shown | Typed decision service picks one candidate id, or DONE / BLOCKED; a separate VLM sets local goals |
| ISCORE | IntentScore: Intent-Conditioned Action Evaluation | arXiv preprint, Apr–May 2026 | https://arxiv.org/html/2604.05157v3 | not shown | 13M dual-encoder reward model; cosine score per candidate action; no LLM in the loop |
| DECOUP | Decoupling What from Where: How Should a Small GUI Grounding Model Receive the Action Type? | **Workshop** (2nd Workshop on Grounded and Faithful VLMs, VLM4RWD), 2026 | https://arxiv.org/html/2610.07444v1 | not shown | Matched 5-seed study of an 8-class action-type **classification head** feeding a grounding model |
| GACTOR | GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents | **NeurIPS 2025** | https://proceedings.neurips.cc/paper_files/paper/2025/hash/16130af940e9dabb43c726119bd3b42e-Abstract-Conference.html (abstract only read) | not shown | Attention action head over patch tokens: grounding with no coordinate text (plan is still generated) |
| LEX | Lexical Coupling in GUI Element Grounding | **Findings of EMNLP 2026** | https://arxiv.org/html/2608.21794v1 | not shown | Audit: every grounded step recast as same-screen candidate ranking; BM25 beats 5 dense encoders at R@1 |
| INERTIA | Absorbed in Inertia: Activation Analysis for Computer-Use Agents | arXiv preprint, Sep 2026 | https://arxiv.org/html/2609.37176v1 | not shown | Measures screen–action recurrence; failure notice vs steering vs R3 reset-reroute-restore |
| VLAA | VLAA-GUI: Knowing When to STOP, RECOVER, and SEARCH | arXiv preprint, Apr 2026 | https://arxiv.org/html/2604.21375v2 | not shown | Modular framework; **Loop Breaker** = verified no-change + repeat → blacklist action, force another |
| VERIGUI | Don't Act Blindly: Robust GUI Automation via Action-Effect Verification | arXiv preprint, Apr 2026 | https://arxiv.org/abs/2604.05477 (abstract only read — see note) | not shown | Action-effect verification, synthetic failure trajectories, GRPO with asymmetric verification reward |
| GUARD | Do GUI Agents Know When Not to Act? Conflict-Aware Termination | arXiv 2609.03438; EMNLP 2026 main | https://arxiv.org/abs/2609.03438 (abstract only) · https://aclanthology.org/2026.emnlp-main.1028.pdf | not shown | Inference-time termination steering for infeasible instructions (CONFLICTGUI / CONFLICTGUARD) |
| BLOG | "A 'screen did not change' notice barely stops computer-use agents repeating" | vendor/grey blog, Oct 2026 | https://blolabel.ai/blog/screen-unchanged-notice-barely-stops-repeats | n/a | Secondary analysis of INERTIA + a shipped phone agent whose loop detector is dead code |

Rows INERTIA and VLAA are **not** decoder-free — they are in the table because they are the only measurements
of the mechanisms T-B and T-C claim, and rule 3 forbids me from putting their numbers in a decoder-free comparison.

## 2. Findings

### 2.1 GUI-JEPA — the closest thing to contract v3 (VERIFIED)

Mechanism. Screenshot → frozen ViT-L/16 JEPA encoder (303M, pretrained on 189K unlabelled GUI screenshots,
I-JEPA objective) → 196 patch tokens. A 39.9M residual predictor maps the screen latent to one latent per
candidate, conditioned on a 16-d action vector + ROI-pooled element feature. A cosine "structural head"
(pooled candidate latent vs CLIP-embedded goal+history) is added to a 33M MiniLM cross-encoder score of the
candidate's text. Argmax over candidates. **No token is generated** (§3, §3.2–3.4).

- **Element list:** yes, a11y. "visible a11y nodes flagged clickable, long-clickable, editable or scrollable,
  **capped at the 80 smallest by area**"; every node gives a click candidate, every editable node also gives an
  `input_text` candidate; plus 8 global actions (4 scroll directions, back, home, wait, open_app). `open_app`'s
  argument is *"the longest training-vocabulary app name in the text condition"* — a string match, not a choice
  head over a launchable-app list (§3.1).
- **Span copy:** yes, and it is structurally the same idea as our typed-text head — a MiniLM-L6 ranker picks
  "the top span of the goal" as the `input_text` argument; switching it from step instructions to goal spans gave
  **+2.85 IDD AC-High** (13.5 → 52.5 input_text accuracy) (§4.4).
- **History carries the effect of the previous action:** partially, and this matters for T-B. State is
  `h_t = (previous action vector, mean latent change from the previous frame)`. So a change signal is present but
  as a **continuous latent delta**, not a verified binary changed/no-change flag. Conditioning on previous action
  and frame change was worth **+8 AC-High points** at pilot scale (§5.2).
- **Termination:** none. Verbatim: *"The policy is a step-level scorer; online task completion, which needs a
  termination head and a planner, is left to future work"* (Limitations).
- **Benchmarks:** AndroidControl only. No AndroidWorld, no AndroidLab. IDD: AC-Low 84.06 / AC-High 60.99;
  full test 78.64 / 53.75 (Table 1). Same-protocol Qwen2.5-VL-7B LoRA reference: 91.01 / 70.87 → 92.4% / 86.1%
  relative, 6.0% of the parameters, and **51.5 ms (bf16) / 74.3 ms (fp32) per step vs 206.7 ms vLLM, 506.7 ms HF**
  (Table 3, medians over 100 real AC-High steps, one H20, batch 1).
- **Confidence:** top1−top2 margin flags its own errors at **AUROC 0.74**, proposed as a routing signal to a 7B
  (§4.6, Appendix J). This is the number our T-C offline arm will be read against.
- **Data / learning:** 13,594 episodes, 74,714 steps from the AndroidControl train split; listwise CE over
  candidates; ~132M trained of 0.50B. **Imitation only, no RL.**
- Honest limits they state: a11y is a privileged input; 1.6–2.6% of test clicks have no candidate and count as
  errors; latency excludes screen capture and a11y retrieval; some design choices selected on test IDD.

### 2.2 V-Droid — the only decoder-free *decision* with an AndroidWorld 116-task number (VERIFIED)

Mechanism. Rule-based extractor builds the action set from the Android **accessibility XML** (no screenshot):
click / long-press / scroll / type / clear per interactive node, plus default actions `open apps, wait,
navigate home, navigate back, complete task, answer users' question` (§3.1). Each candidate is wrapped in a
prompt; a Llama-3.1-8B-Instruct-4bit verifier with a trainable MLP head scores it **by prefill only** — the score
is the likelihood of the 'Yes' token, no decoding (§3.2–3.3). Argmax executes; batching + prefix caching.

- **Element list:** yes — ≈**20 interactive elements on average** per screen (Fig. 3), and they verify
  **50.3 actions per step** on average (§6.5), so the candidate set is the element list × action types.
- **History:** a "working memory" of action history + trajectory, updated every step by a **step-level summary
  written by GPT-4** (§3.4). Table 4 is the direct experiment for our Q4: LLM-written memory → 59.5% SR;
  rule-based memory that extracts UI before/after changes → **46.1%**; action history only → **40.0%**, with the
  stated failure mode *"the agent struggles to retain key contextual information, leading to repeated actions"*.
  So effect-carrying text memory is worth +19.5 points over bare action history in *this* architecture — the
  strongest existing counterweight to 2610.30205's "failure records raise repeats in small LMs".
- **Termination:** `complete task` is **one more candidate action**, scored by the same verifier — a learned
  status choice, not a separate head, and not trained with constructed negatives. Trajectories average 11.6 steps.
- **Benchmarks:** AndroidWorld **59.5%** (116 tasks, in-domain), AndroidLab 38.3%, MobileAgentBench 49.0%;
  decision latency 0.74 s, step latency 4.3 s of which 3.03 s is the GPT-4 memory update. Table 3 (1.1K steps of
  training data): verifier 47.4 vs a **Selector** (SFT to output the correct action number) 35.8 vs a Generator
  27.4 — the cleanest published evidence that scoring beats selecting-by-generation at equal data.
- **Data / learning:** 110K human+agent-annotated action pairs after four rounds; P³ pairwise **preference**
  (ranking) loss on top of SFT. **No RL.**
- **Effect mask:** none. Failures are handled by *training* (self-correcting pairs on erroneous states, 2.5% of
  data, reverse action = navigate_back) — not by suppressing a re-issue at run time.

### 2.3 LWAM — decoder-free *rescoring* inside a generative proposer (VERIFIED)

Mechanism. A frozen M3A-style agent proposes K candidate actions; a LoRA'd Qwen3-VL-8B encodes
`(screenshot, goal, history H_t, candidates)` in one prompt with candidates in **randomly permuted order**;
`ActionHead(h_t)_k = w_kᵀ h_t`, `softmax` over K, CE against the recorded action (§3.1.1). A transition head
predicts the next-state latent with an action-aware contrastive loss (current-state latent and a FIFO memory bank
as negatives); **the transition head is dropped at inference** (§3.2).

- **Element list:** inherited from the base agent's candidates — i.e. a11y + screenshot via M3A. Size not stated.
- **History:** `H_t` text history; **no effect flag**.
- **Termination:** inherited from the base agent, which is generative. Not modelled by the head.
- **Benchmarks:** AndroidWorld Table 2. Qwen3-VL-8B base 27.68 → **40.18**; GPT-5.6 base 54.46 → **61.61**;
  Gemini-3.6-Flash 58.93 → 63.39; GUI-Owl-7B 15.18 → 22.32; Qwen3-VL-32B 41.96 → 46.42.
  **Number of task instances, seeds and budget are not stated** — the Easy/Medium/Hard SR columns imply the
  116-task suite but the count never appears; do not pair this with our seed-30 numbers.
- **Ablation that matters to us:** Group I (base) 27.68 → Group II (**action head only**) 36.36 → Group III
  (+transition head) 38.18 → Group IV (+contrastive) 40.18 (Table 4). So a decoder-free candidate head alone
  recovers **+8.7 points** of the world-model gain on AndroidWorld. That is the single most useful number in this
  slice for sizing what an element/action scorer can buy inside an existing harness.
- **Data / learning:** LoRA on Qwen3-VL-8B, CE + contrastive, 3 epochs, 8 h on one A6000. **No RL.**

### 2.4 LiMAC / AcT — the earliest decoder-free "what + where" (VERIFIED)

Mechanism. An Action Transformer over a sequence of pretrained embeddings: goal, every UI element
(`[attr; text; image]`, CLIP fine-tuned with InfoNCE), and past actions. Two heads: a **10-way action-type
classifier** (`L_type`, CE, §3.3) and, for clicks, an **InfoNCE contrastive selection over all element
embeddings in the episode** (§3.5). A fine-tuned VLM is called *only* when the type is `input-text` or
`open-app`, forced to start from the predicted type (§3.4). ~500M params.

- **Element list:** yes — every UI element in the observed tree, no cap stated; AITW has no tree, so they
  OCR it (and show the method degrades gracefully when the tree is missing).
- **History:** the whole episode's element sets + action embeddings, interleaved; **no effect signal**.
- **Termination:** not addressed. The 10 types include "other essential commands" but no verified done head
  (whether a `finished`/`status` type exists is **not stated** in the text I read — UNVERIFIED).
- **Benchmarks:** AndroidControl 63.1 overall, AITW 72.2 overall (relaxed accuracy, Table 1); modular ablation
  Table 2 (AcT type + AcT click + Florence2 text = 63.1 at 0.34 s/step). **No AndroidWorld.**
- **Learning:** supervised + contrastive, offline. No RL.
- Why it matters: it is the ancestor of both GUI-JEPA and our contract (type head → element choice → text
  argument by a small model), and it already reports the modular decomposition we would use.

### 2.5 Jev-Mobile — a decoder-free *executor* with 0.79 on AndroidWorld (VERIFIED)

Mechanism. Two-tier: a Qwen3.8-Max VLM sets a *local goal* (and exact text values) and hands control to "Jev",
a remote typed decision service that chooses **one candidate id**, or returns **DONE / BLOCKED** (§3.1–3.3).
Candidates are regenerated from the raw a11y tree after every action; old ids are invalidated.

- **Element list:** yes, "deterministic candidates from the raw tree" — visible, enabled, valid-bounds nodes ×
  click / long-press / scroll / focus, plus Back, Home, Enter, Open app. No cap stated; long/noisy trees are
  named as a failure mode.
- **History carries effect:** **yes, and explicitly.** Each submitted step records "the chosen ID, actual action
  parameters, execution status, **before/after observation IDs**, and a **deterministic change description**"
  (§3.3). This is the nearest existing instance of our "action + verified changed/no-change" trajectory line —
  but it feeds a *VLM planner*, and the decision model receives only the delegation-local history.
- **Termination:** the decision model returns `DONE` (return control) or `BLOCKED`; task success is judged by
  AndroidWorld after the VLM issues `finish`. So there are **two** termination paths, one learned by the typed
  decision service, one by the planner.
- **Benchmarks:** full AndroidWorld suite: **0.79 SR** vs SeeAct-V 0.78 and step-wise VLM 0.84; mean total time
  per success 132.67 s vs 197.21 s; executor time 5.16 s vs 94.30 s; model cost $0.0727 vs $0.2737 (Table 1).
  **Task count, seeds and step budget are not stated** — "the full AndroidWorld task suite" is the only
  description; do not read 0.79 as comparable to a seed-30 116-task run.
- **Learning:** none. No training, no RL; it is a service-level composition. (So it is not a competing trained
  policy — it is proof that a decoder-free *chooser* can carry 79% of AndroidWorld when a planner supplies goals.)

### 2.6 IntentScore — decoder-free candidate scoring, and a warning (VERIFIED)

Mechanism. A 13M-parameter dual encoder: a state vector (frozen SigLIP screenshot + MPNet text + a 2-layer GRU
over the last K=3 steps) and an intention-aware action vector (action description + code + coordinates + the
candidate's own planning rationale). Score = temperature-scaled cosine; argmax over N sampled candidates.
~1 ms/candidate on CPU, **no LLM call** (§3.2).

- **Element list:** no — candidates are N=3 LLM-sampled actions (desktop, pyautogui code).
- **History:** last 3 steps of `(screenshot, action, code, coords)`; **no effect flag**.
- **Termination:** not modelled.
- **Benchmarks:** OSWorld 361 tasks: Agent S3 45.2% → **52.1%** (+6.9), override rate 46.6% (Table 5).
  Offline AgentNet: 97.5% adjacent-step discrimination, 99.7% real-incorrect detection (Table 1).
- **The warning we must not skip:** Table 2, "Architecture & paradigm" — replacing the dual encoder with a
  Transformer encoder drops Hard accuracy to 0.755, and **"Binary classification 0.073"**. And Table 3:
  contrastive alignment *alone* gives a correct-minus-incorrect score gap of **0.005** — i.e. relevance and
  correctness come apart, and only an explicit margin-ranking objective on labeled negatives produced separation
  (0.046). If our per-decision confidence is trained on relevance/imitation alone, this is the published failure
  mode it will land in. It is also the best available template for T-C: **construct the negatives from labeled
  incorrect steps and rank**, do not train a binary head.
- **Learning:** InfoNCE + margin ranking on 398,828 offline steps. No RL.

### 2.7 Decoupling What from Where — the action-type head, measured honestly (VERIFIED)

Mechanism. Two stages: **Stage 1 is a pure classifier** — frozen Qwen2-VL-2B features, mean-pooled, 3-layer MLP
(1536→1024→256→8), class-weighted CE over an 8-class space (*click, double-click, type, scroll, drag, hotkey,
wait, finished*). Stage 2 grounds to a coordinate string with LoRA, receiving the type five different ways
(auxiliary loss, hard routing, additive embedding, prepended token, prompt word) under matched data, compute,
adapters and decoding. Five seeds, episode-clustered bootstrap, seed-level paired tests (§3.2–3.3).

- **Element list:** no (coordinate grounding).
- **History:** none.
- **Termination:** `finished` exists as a class in the taxonomy, but the evaluation metric is hit@0.10 on
  grounding; no termination result is reported in the text I read (UNVERIFIED for `finished` accuracy).
- **Numbers:** the three conditioning mechanisms gain **+5 to +7 hit@0.10** over the flat baseline
  (p ≤ 0.03) — but the gain is largely an artifact of their serializer clamping AITW's off-screen type point to
  the origin; retraining without that class lifts the baseline by **~7 points**, after which **no mechanism beats
  it by more than 0.002** and the intervals exclude a 2-point effect. Deployment: with a **Stage-1 classifier at
  84% accuracy**, the pipeline's margin over the flat baseline is **+0.016, 95% CI [−0.017, +0.052]** — not
  established. And *"a wrong type collapses every model conditioned at inference."*
- Why this is the sharpest threat to the *shape* of contract v3: we put a type choice in front of an element
  choice. This is the only matched, multi-seed, cluster-bootstrapped test of that decomposition at 2B scale, and
  its verdict is that the type signal pays only when the type is right, which a 84%-accurate classifier mostly
  isn't.

### 2.8 GUI-Actor — decoder-free *grounding* only (abstract read)

An attention action head aligns a dedicated token with relevant visual patch tokens and proposes one or more
action regions in a single forward pass; a verifier picks among them. GUI-Actor-7B: 40.7 (Qwen2-VL) / 44.6
(Qwen2.5-VL) on ScreenSpot-Pro, above UI-TARS-72B (38.1); fine-tuning only the ~100M action head with a frozen
backbone matches prior SOTA. The *plan* is still generated text — this is a component, not a policy.
No AndroidWorld, no termination, no history.

### 2.9 Lexical Coupling — the measurement trap for element choice (VERIFIED)

Recasts **every grounded step as a same-screen candidate-ranking task** and asks what embedding similarity
measures. AndroidControl: 58,078 grounded steps, **4.56M candidates, median 62 per step, max 1,657**;
clean-main n = 32,646.

| retriever | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Random | 0.022 | 0.107 | 0.212 | 0.089 |
| Jaccard | 0.547 | 0.703 | 0.741 | 0.624 |
| **BM25** | **0.557** | 0.707 | 0.742 | **0.632** |
| MiniLM-L6 | 0.489 | 0.709 | 0.768 | 0.592 |
| MPNet | 0.467 | 0.716 | 0.783 | 0.583 |
| BGE-large | 0.520 | 0.724 | 0.785 | 0.616 |
| E5-large | 0.507 | 0.707 | 0.767 | 0.603 |
| Qwen3-Emb-0.6B | 0.481 | 0.680 | 0.740 | 0.577 |

(Table 1.) BM25 beats all five dense encoders at R@1. Label-type stratification (Table 2, all 58,078 rows):
text-only 21,539 rows BM25 0.624; cd-only 8,441 rows 0.396; **resource-only 9,669 rows tops out at 0.124
(Qwen3); pseudo and none sit at chance**. Masking the target's exposed text costs **22–38 R@1 points**
(Appendix G). A task-supervised MiniLM-L6 control reaches R@1 0.694 but retains LCC 0.877 and collapses to
**0.151** under target-text perturbation.

Consequence for us: our element head is a learned scorer over a11y text, and it will inherit exactly this. An
offline element-choice number on AndroidControl that does not report (a) a lexical baseline, (b) an
element-text-masked ablation and (c) R@1 *stratified by label type* is not evidence of grounding — it is
evidence of label copying. That is a reporting obligation, not a footnote.

### 2.10 Absorbed in Inertia — what a verified-failure signal does (VERIFIED, 97 OpenComputer tasks)

Inertia = exact screen–action recurrence **and** the reasoning acknowledges failure. Recurrence appears in up to
66% of failed trajectories; 55–89% of recurrences come with acknowledged failure (for 3 of 4 models). Pre-action
activations separate recurrence at AUC 0.72–0.95.

Table 1, relative change in screen–action recurrence per executed action, with task-level SE:

| intervention | UI-TARS | GUI-Owl | OpenCUA | EvoCUA |
|---|---|---|---|---|
| Failure notice | −4.6 ± 8.4 | −5.0 ± 16.6 | −23.4 ± 16.9 | **+16.9 ± 17.6** |
| Activation steering | −5.6 ± 6.9 | −0.2 ± 15.7 | **+39.0** ± 19.0 | −0.9 ± 13.0 |
| Permanent reset | −14.6 ± 8.0 | −28.8 ± 14.5 | −2.3 ± 16.7 | +6.9 ± 18.0 |
| **R3 (reset, reroute, restore)** | **−16.9 ± 5.6** | **−38.0 ± 9.9** | **−42.7 ± 16.1** | **−33.8 ± 11.3** |

Task score moves +0.60 / +1.20 / −2.50 / −0.90 points — **not distinguishable from zero**. Table 2 (next-action
replay at 2,000-ish trigger states, none executed): unmodified agents propose the same action again **79.8 /
81.0 / 87.5 / 84.8%**; **selective reset** (delete only the duplicates, name the repeated action in the prompt)
still **70.2 / 60.7 / 62.5 / 73.9%**; full reset **0.8 / 2.4 / 3.6 / 0.0%**.

Three readings we cannot avoid:
1. Telling a model "this did nothing" in context is close to useless (CIs include zero, one arm got worse).
2. Removing the repeated action *from the record* is not enough either — 61–74% still repeat it.
3. What works is changing the *input distribution* the next action is computed from. A softmax mask on
   (screen, action) is a third mechanism of exactly that class, and is untested here.

### 2.11 VLAA-GUI — Loop Breaker (VERIFIED, OSWorld 361 tasks / WAA 154 tasks)

Tier 1: same action + no visible change `n_t^a ≥ τ_a` → switch interaction modality. Tier 2: same screen state
recurs `n_t^o ≥ τ_o` → switch strategy. **Tier 3: an external judge signals a loop → "a hard directive is
injected at step t+1 that blacklists the repeated action and forces the Manager to select from the remaining
actions"** (§3.3). Completion is a separate two-level check (UI-observable criteria + a verifier model), and
"no visible change" has an explicit rule: *"wait(1) before re-check; do not repeat immediately"* (Table 1).

Ablation, OSWorld 100-step: Gemini 3 Flash −Loop Breaker 68.77 → 66.95; Sonnet 4.6 71.67 → 71.63 (noise).
WAA 100-step: −Loop Breaker 61.0 → 55.8, the second-largest ablation after the verifier. Reported summary:
*"the Loop Breaker nearly halves wasted steps (4.9% → 2.8%) for loop-prone backbones."*

This is T-B's mechanism with the *exact* logic we plan (verified no-effect → do not re-issue → take the next
option), minus (a) the softmax and (b) the decoder-free carrier. It is published, April 2026, on the dominant
desktop benchmark.

### 2.12 VeriGUI, ConflictGuard — adjacent, generative (abstract only)

- **VeriGUI**: Thinking–Verification–Action–Expectation loop that predicts the expected outcome and detects
  failure, plus **Robust SFT with synthetic failure trajectories and GRPO with asymmetric verification rewards**;
  builds a robustness benchmark on AndroidControl. This is the only work I found that *manufactures* failed
  experience deliberately — the design move T-C needs (construct the no-effect/negative set rather than wait for
  it). I read only the abstract (rule 1: treat "significantly reduces failure loops, maintains competitive
  standard task performance" as **UNVERIFIED**; no numbers available to me).
- **ConflictGuard**: inference-time feasibility verification + conditional action modulation to make agents stop
  on infeasible instructions; five agents, CONFLICTGUI benchmark. Abstract only (**UNVERIFIED**). It is about
  *should-not-act* termination, adjacent to a `status` head but prompt-level.

### 2.13 What the grey literature shows (secondary, one blog)

BLOG is not a paper; it is a derivative reading of INERTIA plus a shipped open phone agent ("phoneshell"). Its
own finding is the interesting part: the shipped agent already emits "THE SCREEN DID NOT CHANGE … try something
different rather than repeating it" after any no-effect action, and its `StuckDetector` class (repeat limit 3,
A-B-A-B-A-B rule) is **defined and called nowhere** (`grep -rn "StuckDetector"` returns only the class). Its
advice to builders: *"do not stop at a stronger warning. Count unchanged repeats per screen and action, and at a
small threshold call the model with only the task and the current screen."* Everything in this paragraph is
**UNVERIFIED** (single vendor blog, no independent check), but it is corroboration that the naive form of the
verified-failure signal is already deployed and already fails.

## 3. What this changes for us

1. **T-A must be re-aimed or it is a repeat.** GUI-JEPA already ran the matched comparison (same inputs, same
   judge, same split, same GPU, two arms, one trained by them) and already reports the relative loss
   (92.4%/86.1%) and the latency (2.8–4.0×). If we publish T-A as "one pass keeps competence at lower latency",
   the reviewer's first citation is 2610.36877 §4.3. What remains unbought: (i) a *live* benchmark, (ii) the
   **length axis** (their data is single-step exact-match; F4 says our success is 3/10 at ≤3 steps and 1/38
   beyond — that gradient is the paper), (iii) end-to-end latency including environment time. Our T-A text should
   say so explicitly and cite GUI-JEPA as the offline precedent.
2. **The v3 architecture is not novel as a shape; its completeness is.** Element-id choice over an a11y list,
   span copy from the goal, and a global `open_app` are all in GUI-JEPA. What no one has is the full contract:
   + a **launchable-app choice head** (GUI-JEPA string-matches the longest training-vocabulary app name — a
   brittle shortcut that our F7 "an app name that cannot be copied from the goal" failure is exactly about),
   + a **status head**, + an **effect-carrying history line**, + a live 116-task evaluation. Position the
   contribution as the first *complete* decision contract evaluated online, not as a decoder-free agent.
3. **T-B is a citation, not a claim.** VLAA-GUI's Loop Breaker blacklists a verified-no-effect repeat and forces
   the next option; INERTIA's R3 resets and reroutes (−17 to −43% recurrence). Both are published. We keep T-B
   only as a *measurement* on a decoder-free policy plus the softmax-mask form, and we must say what the mask
   buys that a blacklist does not — the only defensible difference is that it excludes one (screen, action) pair
   from the model's own distribution instead of instructing the model in text, and INERTIA Table 2 is the
   evidence that text instructions do not work (61–74% still repeat after selective deletion).
4. **T-B's control arm is now more important, not less.** V-Droid Table 4 gives the mechanism a size: bare action
   history 40.0% vs effect-carrying memory 59.5%, with "repeated actions" named as the failure. That is a
   decoder-free, online, 116-task precedent that effect in the record matters — we must plan to *replicate* it,
   not discover it.
5. **T-C gets a template and a benchmark to beat.** IntentScore's ablation is the recipe: relevance-trained
   scores do not separate correct from incorrect (CI gap 0.005), margin ranking on labeled negatives does
   (0.046), and a naive binary head collapses (0.073). We should build the no-effect signal the same way
   (rank against verified negatives), and report against GUI-JEPA's **AUROC 0.74 offline** top-margin number,
   which is the closest published value to our ≥0.70 gate.
6. **Our offline element-choice numbers need three new columns or they are not evidence**: a lexical baseline
   (BM25 R@1 = 0.557 is the bar on AndroidControl), a label-type stratification, and an element-text-masked
   ablation (target-text removal costs 22–38 R@1 points). LEX makes this an expectation, and it directly
   threatens the "copy a span of the GOAL" head too — that head is *by design* a lexical match.
7. **Termination: the gap is real.** No decoder-free paper has a done head; GUI-JEPA states it as future work;
   V-Droid and Jev treat completion as one more option. This is the one column of the comparison table where the
   whole field is empty and our contract has a cell. If anything in v3 is defensible as new, it is a `status`
   head trained with constructed negatives and measured on 116 tasks.
8. **Do not compare these numbers to ours.** None of them states task-instance counts or seeds (LWAM, Jev),
   two are desktop (IntentScore OSWorld, VLAA OSWorld), one is text-only XML (V-Droid), one is offline
   (GUI-JEPA), and one has a planner in the loop (Jev). Every AndroidWorld figure above is reported with its
   stated scaffold and marked "task count not stated".

## 4. Searches run

Sources searched: arXiv (abs + HTML full text), NeurIPS proceedings, ICLR proceedings, ACL Anthology, MobiCom
DOI, alphaXiv/Scholar-Inbox/SciXplorer/DeepLearn indices, Semantic Scholar-adjacent aggregators, GitHub, one
vendor blog.

Queries and outcomes:

| query | outcome |
|---|---|
| `GUI agent action classification head without text decoding element ranking` | found GUI-JEPA, Decoupling What from Where, GUI-Actor, Ponder & Press (rejected: generative) |
| `V-Droid verifier-free GUI agent sequential decision making process pairwise` | found V-Droid (MobiCom v5) + MSR page |
| `Jev-Mobile accessibility identifier selection typed decision model GUI agent` | found Jev-Mobile arXiv + 3 secondary write-ups |
| `LiMAC "A Simple yet Effective Action" mobile agent action transformer UI tree` | found LiMAC ICLR 2025 |
| `"GUI agent" non-autoregressive single forward pass no text generation action selection 2026` | GUI-JEPA again; found **LWAM** (2610.14228), GUI-Libra (RL, generative), GTA1 (test-time scaling, generative) |
| `mobile GUI agent candidate action reranking listwise scoring action head AndroidWorld` | null for a decoder-free AndroidWorld reranker; returned benchmarks/leaderboards only |
| `GUI agent termination "done" head learned stop prediction task completion classifier` | found **VLAA-GUI**, **ConflictGuard**; nothing decoder-free |
| `GUI agent detect no-effect action screen unchanged repeat action masking loop avoidance` | found **VeriGUI**, VLAA-GUI Loop Breaker, **BLOG**, an open GitHub android agent with screenshot-diff + stuck detection |
| `IntentScore scoring candidate actions generative GUI policy verifier` | found IntentScore (2604.05157 v3) |
| `"UI element" retrieval ranking BERT encoder classify element index GUI agent instead of generating coordinates` | found **LEX** (2608.21794) plus GUI retrieval papers (GUI-ReRank etc., out of scope: retrieving GUIs, not actions) |
| `master thesis "GUI agent" action classifier no language model decoder embeddings android element selection` | **null** — no thesis, no workshop paper, no HuggingFace card describing a decoder-free GUI policy. Returned GUICourse, surveys, SeeClick, CogAgent, UI-Act-model (GitHub, generative) |

Nulls worth recording explicitly:

- **No decoder-free paper reports AndroidWorld 116-task SR with zero generated tokens end to end.** V-Droid is
  the closest (action *choice* is prefill-only) but it generates text for `type`/`open_app`/`answer` (12.4% of
  actions) and uses a GPT-4-written memory summary every step.
- **No decoder-free paper uses a verified-failure / no-effect mask.** Not in GUI-JEPA, V-Droid, LWAM, LiMAC,
  Jev-Mobile, IntentScore, Decoupling, GUI-Actor.
- **No decoder-free paper trains a done/status head**, with or without constructed negatives.
- **No decoder-free paper evaluates a launchable-app choice** as a head (GUI-JEPA string-matches; Jev delegates
  the app name to the VLM).
- I found **no thesis, no workshop-length paper other than DECOUP, and no HuggingFace model card** for a
  decoder-free GUI policy; the entire visible population is the table above. Low-citation status is therefore
  the norm here, not a hidden set I failed to reach: every entry except V-Droid, LiMAC and GUI-Actor is a 2026
  preprint with no citation count displayed.
- LAMO (ACL 2026 Findings, "lightweight GUI policy executor + stronger planner") was surfaced by Jev-Mobile's
  related work and **not read**; if it contains a decoder-free executor it belongs in this table. Flagged.

## 5. Follow-up (2026-11-11)

### 5.1 LAMO — "Towards Scalable Lightweight GUI Agents via Multi-role Orchestration" (Findings of ACL 2026, pp. 22359–22388, DOI 10.18653/v1/2026.findings-acl.1122, https://aclanthology.org/2026.findings-acl.1122.pdf) — read in full.

**Verdict: NOT decoder-free (SILENT for this slice). Read and closed.**

- **No decoder-free executor.** LAMO-3B is Qwen2.5-VL-3B-Instruct post-trained (PWCE-SFT then GRPO) and it
  **emits tokens**. §3 formulates the step exactly as generation, `y_k = M_θ(y_<k | I, o)`; §4.4 defines the
  output as `<think>CoT</think> <answer>action decision</answer> <tool_call>atomic action a_t</tool_call>`, and
  the action space `A` is PyAutoGUI-style text (`pyautogui.click`, `write(message=…)`). §4.3's ATA reward parses
  the tool class and value as strings. There is no per-type classifier, no element ranker, no candidate scorer.
  "Policy executor" means *a 3B MLLM that executes a planner's instruction*, not a decoder-free policy.
- **AndroidWorld: reported — Table 4.** LAMO-3B in Policy Executor Mode with **Gemini-2.5-pro as planner = 60.3**
  and with **GPT-5 as planner = 77.6**; Gemini-2.5-pro alone (end-to-end reasoning) = 31.0, i.e. the +94.5%
  relative gain quoted in §5.4. Table 4's other rows (UI-TARS-72B 46.6, Agent-S2 54.3, Mobile-Agent-V3 73.3,
  UI-Ins-7B + GPT-5 74.1) are leaderboard entries from their own papers. **Task count, seeds and step budget
  are not stated** — AndroidWorld is named only as "the online mobile environment"; do not align 77.6 with a
  seed-30 116-task run.
- **Learning:** SFT + **GRPO** (multi-task, rule-based rewards) — so it is also the one lightweight-agent entry
  in this slice that uses RL, and therefore outside our imitation-only comparison in any case.
- Consequence: the flag opened in §4 is resolved as a null. LAMO is not a competitor to contract v3; it is a
  second instance of the pattern already visible in Jev-Mobile — a *generative* 3B executor plus a *large*
  planner, with the planner doing the work (31.0 → 60.3 is the planner's contribution, not the executor's).

### 5.2 GUI-JEPA: one pass per step, or one per candidate? — **one batched pass per step.**

**Verdict: single-pass in our sense (1 forward pass per step over the screenshot), with per-step cost linear in the candidate count — not N-pass prefill like V-Droid.** Verbatim, in order of specificity:

1. §3 (Method, first paragraph): *"Since no token, pixel or line of code is generated, **a decision is one batched
   forward pass whose cost grows with the number of candidates**."*
2. §3.3 (residual latent predictor): *"**All candidates of a step form one batch.**"* — the 39.9M predictor takes
   the screen latent plus |C| candidate conditions and returns |C| candidate-conditioned latents in one call.
3. §1 (Introduction): *"letting lightweight heads score all candidates against the instruction in a **single
   batched pass**, without decoding a token."*
4. §4.2 (Examples): *"it picks a menu entry among 15 candidates, opens the app named in the goal, and types a book
   title copied from the goal, **each with one forward pass**."*

The screenshot itself is encoded **once** per step and not per candidate: the ViT-L/16 encoder is frozen after
JEPA pretraining and *"its patch tokens are cached, so policy training never runs the backbone"* (§3.2); at
inference the encoder runs on the screenshot, the predictor is applied per candidate **as a batch over that one
latent**, and the structural head and the MiniLM text matcher are likewise batched over candidates (§3.4).

What this means against V-Droid. V-Droid re-prefills an 8B LLM prompt once **per candidate** (50.3 verified
actions per step, §6.5), amortised only by shared-prefix KV caching; its 0.44 s verification is the price of
~50 prefills of a large model. GUI-JEPA pays one ViT-L pass (303M params, frozen) plus one batched pass through a
39.9M predictor — which is why 51.5 ms/step is possible (§4.5, Table 3). So the two are not the same regime, and
"single-pass" is fair for GUI-JEPA.

The caveat we must carry into our own contract: **GUI-JEPA's step cost is not fixed** — it scales with the
candidate count, capped at 80 elements (+ editables + 8 global actions) in §3.1, and §4.5 attributes roughly
half of an fp32 step to "the batched predictor and head, which bf16 speeds up threefold, and **its cost grows
with the number of candidates**". Our cap of 40 elements is therefore a latency parameter, not just an input
format, and any latency claim of ours must state the candidate count it was measured at, exactly as GUI-JEPA
does (100 shared AC-High steps, batch 1, one H20).
