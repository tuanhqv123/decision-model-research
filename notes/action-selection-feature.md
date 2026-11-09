# Feature deep-dive: calibrated, on-device next-action selection with a confidence gate

**Scope.** How SOTA mobile/GUI agents predict the *next action* today; how (if at all) they model confidence/uncertainty; where the candidate-action set K comes from; prior art for a one-pass "choice-head over GUI affordances"; and defensible improvement directions for a single-pass calibrated action-scoring head + act/ask/refuse/escalate gate.

**Bottom line up front.** No system in the surveyed SOTA does what we propose. Every current action predictor is an **autoregressive token decoder** that emits a `(type, coordinate)` point prediction (UI-R1, Hi-Agent, MIRAGE, Aria-UI, UI-Venus) — even when it restricts output to an enumerated element id (AutoDroid). The pieces of our design exist *separately*: (i) one-pass calibrated distributions over a fixed option set (TypeSafe System One / Jev, and its open reproduction `decider`), (ii) confidence gating for GUI agents (OS-Kairos) and uncertainty-aware GUI grounding with calibrated thresholds (SafeGround), and (iii) the act/ask/abstain(**/escalate**) decision vocabulary (CAAG, MARC). The **gap** is that no one has (i) built an explicit K-set of *GUI affordances*, (ii) scored each as a **calibrated** distribution in **one forward pass / zero output tokens**, and (iii) gated act/ask/refuse/escalate on that distribution. This note documents what exists and which claim is most vs. least defensible.

---

## 1. How SOTA produces/predicts the NEXT ACTION today

### UI-R1 — rule-based RL over MLLMs for GUI *action prediction* (AAAI 2026) [ui-r1]
- **Input:** one GUI screenshot + a *low-level* text instruction (single-state, e.g. "Click the menu icon in the top left corner"), as in AndroidControl.
- **Policy model:** Qwen2.5-VL-3B.
- **Candidate/action space:** *not* a candidate set. It is a generative action space of 5 types `{Click, Scroll, Back, Open_App, Input_Text}` plus a click coordinate `C=[x,y]`, transcribed as text tokens.
- **Output format (dictated as `<think>`/`<answer>` tags):**
  `[action: enum[click, open_app, scroll, navigate_back, input_text], coordinate: [x, y]]`
- **Loss / reward (Group Relative Policy Optimization, GRPO):** rule-based reward
  `R = R_T + R_C + R_F`
  - `R_T` (action-type reward) = 1 if predicted type == ground-truth type, else 0.
  - `R_C` (coordinate accuracy) = 1 if predicted coord is *inside* the ground-truth bounding box, else 0 (deliberately not IoU — they argue action-correctness, not element-location, is the GUI goal).
  - `R_F` = format reward for emitting the `<think>`/`<answer>` structure.
  - Efficient variant `UI-R1-E-3B` adds DAST (difficulty-adaptive token-length-budget reward `R_L`) and a "NOTHINK" stage that drops the `think` block for direct grounding.
- **Data:** only **136** high-quality samples (three-stage select: quality/difficulty/diversity over ScreenSpot-mobile + filtered AndroidControl).
- **Results:** UI-R1-3B gains +22.1 (ScreenSpot), +6.0 (ScreenSpot-Pro), +12.7 (AndroidControl) avg over base Qwen2.5-VL-3B; UI-R1-E-3B reaches ScreenSpot-Pro avg 33.5, ~SOTA among 3B GUI-specialized models.
- **Key mechanics to note:** it *still generates reasoning tokens then an answer* (autoregressive); the reward is a *binary* type match + point-in-box; there is **no candidate set, no calibrated probability, and no confidence**. It learns to pick a coordinate, not to score options.

### Hi-Agent — hierarchical VL agent (2025) [hi-agent]
- **Architecture:** two trainable, jointly-optimized policies. High-level reasoning model `π_h` emits a *semantic subgoal* `g_t`; low-level action model `π_l` executes the atomic action. Both instantiated as **Qwen2.5-VL-3B-Instruct**.
- **Low-level action model input:** current screenshot `s_t` **+** subgoal `g_t`.
- **Low-level action model output:** an atomic action `a_t` = `(action_type, coordinate)` — normalized `(x,y)` taps, long-presses, swipes, variable-length text entry, function buttons (HOME/BACK/ENTER), and task-completion signal.
- **Observation space is deliberately minimal:** RGB screenshots only — **no** structured UI annotations, **no** bounding boxes, **no** Set-of-Marks.
- **Training:** GRPO with **alternating joint optimization**; `π_l` gets a binary step reward `r_l = 1` iff `π_l` completes `g_t` else 0 (normalized to advantage across the batch). A **foresight advantage** propagates low-level execution feedback up to `π_h`. Data = >1,200 trajectories auto-generated from a hierarchical oracle (Qwen2.5-VL-72B reasoning + 7B action).
- **Results:** 87.9% task success on Android-in-the-Wild (vs AppAgent 17.7%, Filtered BC 54.5%, DigiRL 71.9%); competitive zero-shot on ScreenSpot-v2; scales on AndroidWorld. ~22h on 4× A800-80GB.
- **Key mechanics to note:** the low-level "action model" is still a **generative VLM** producing action tokens — a point prediction `(type, coord)`, **not** a distribution over a candidate set, and it is *dependent* on the high-level model's subgoal. No calibration/confidence anywhere.

### MIRAGE — implicit reasoning + generative world model (2026) [mirage]
- **Goal/question it answers:** kill the *textual thought* bottleneck ("long textual thoughts ... slower, supervision more costly, deployment less efficient") while keeping reasoning.
- **Mechanism:** replace the `<THOUGHT>` text block with `N` continuous **latent slots** `⟨lat⟩` in the decoder context; the action is still serialized as text tokens and decoded. The latent thoughts are **never detokenized**.
- **How it avoids textual thoughts / what the "action head" is:**
  - **Stage 1** warm-up trains on explicit structured thoughts `[observation][rationale][predict]<ACTION_DESC><ACTION>` via next-token CE.
  - **Stage 2** swaps the `<THOUGHT>` block for latent slots, refines them with **APLR** (Approximate Parallel Latent Refinement — K=3 Jacobi-style parallel rounds approximating serial latent chain-of-thought; provably exact for the first K slots, bounded tail error).
  - **World-model head:** a BLIP-2-style **Q-Former** aligns the gathered latent states with *stop-gradient* next-frame visual features from the frozen vision encoder (masked per-patch cosine). This makes latents transition-predictive and prevents collapse.
  - **Joint objective:** `L = λ·L_ce + (1−λ)·L_wm`, `λ=0.8`. `L_ce` is cross-entropy over serialized action tokens (the actual "action head" is the VLM's LM head over action tokens, used at inference).
  - Backbones Qwen3-VL-4B (9 latents) / 8B (6 latents).
- **Results:** AndroidControl low-level EM 68.48→77.59 and action acc 75.15→91.09 while average tokens drop 115.67→18.92 (4B); AndroidWorld success 42.9→52.6 (4B), 47.6→57.8 (8B) while tokens drop ~103→31 (4B) / ~108→27 (8B). Matches explicit-CoT at 3–5× lower decoded-token budget; >75% fewer generated tokens on AndroidControl.
- **Key mechanics to note:** this is the **strongest existing step toward low-token action emission**, and it is the natural SOTA to beat on *latency/cost*. But it is still (a) **autoregressive token decoding**, (b) a **point prediction**, and (c) **no candidate set, no calibrated distribution, no confidence gate**. The action "head" is a plain LM head over serialized action tokens.

### Set-of-Marks / grounding approaches — where the K candidate elements come from
- **Set-of-Mark (SoM)** [som]: use off-the-shelf segmentation (SEEM/SAM/Mask-DINO/OpenSeeD) to partition the screenshot into regions, overlay alphanumeric marks / masks / boxes, and feed the *marked* image to the LMM so it can *refer to a marked region* rather than emit a raw coordinate. This makes the candidate set **explicit in the image**.
- **ScreenSpot / ScreenSpot-Pro** [screenspot, screenspot-pro]: grounding *benchmarks* with annotated element boxes + element types (Text/Icon). They measure point/box grounding accuracy, and they are a source of *annotated* candidate elements (not a runtime extractor). ScreenSpot-Pro targets high-res professional UI, where elements are small and dense — the regime where extractor-free point prediction degrades.
- **Aria-UI** [aria-ui]: a large multimodal model for **pure-vision** GUI grounding that *eschews HTML/AXTree* (patch-based grounding) — explicitly rejecting the external-accessibility representation.
- **GUI-Actor** [gui-actor]: coordinate-free grounding that critiques single-point prediction (weak spatial-semantic alignment, ambiguous supervision targets).
- **Takeaway:** the field is split between (i) *extract marks/boxes externally and let the model choose* (SoM, and any accessibility/DOM pipeline) and (ii) *let the model directly emit a coordinate/point from pixels* (UI-R1, Hi-Agent, MIRAGE, Aria-UI, UI-Venus). This split is the crux of where K comes from (Section 3).

### AutoDroid — functionality-aware UI representation [autodroid]
- **How it bridges UI → LLM:** converts each Android GUI state to a **simplified HTML** representation using 5 tags (`<button>`, `<checkbox>`, `<scroller>`, `<input>`, `<p>`) with properties `ID, label, onclick, text, direction, checked, value`. HTML is chosen because LLMs are better at HTML than natural-language UI descriptions. The **`onclick`** property encodes *functionality* inferred offline from a UI Transition Graph (UTG) built by random exploration.
- **How it restricts the action space (the K-set):** the LLM must output `- id=<id number> - action=<tap/input> input text=<...> (id=-1 for task completion)`. So the candidate action set is the **enumerated set of existing on-screen UI elements**; the model *selects an element id* — it is a **discrete choice over K**, though still via text generation of the id, not a calibrated distribution.
- **Capabilities/limits:** knowledge injection via exploration-based memory + synthesized simulated tasks; multi-granularity query optimization (prune/merge the DOM, summarize scrolled content) to cut LLM cost.
- **Results:** 90.9% per-action accuracy, 71.3% task success with GPT-4; beats GPT-4 baselines by 36.4–39.7% and cuts query cost ~51.7%.
- **Key mechanics to note:** this is the classic system that already treats action selection as *choosing among enumerated affordances*, but via generative id output on a frozen LLM (GPT-4/Vicuna), with **no calibrated probability** and **no confidence gate**.

### MobileUse (reflection) and K²-Agent — do they improve action *selection* or plan/memory?
- **MobileUse** [mobileuse]: hierarchical **reflection** — self-monitor / detect / recover from errors across temporal scales (a single action → whole task), with **reflection-on-demand** (selective, not after every step) plus a **proactive exploration** module for cold-start. Results: AndroidWorld 62.9%, AndroidLab 44.2% (SOTA at the time). **Answer:** it improves **robustness / error recovery / memory**, not calibrated action *selection*; the reflection layer is an *after-the-fact* correction, not a pre-action confidence gate.
- **K²-Agent** [k2-agent]: hierarchical split of **declarative** know-what (high-level reasoner via a Summarize-Reflect-Locate-Revise self-evolution loop, bootstrapped from one demo) and **procedural** know-how (low-level executor trained with **C-GRPO** — curriculum-guided GRPO with a balanced sample pool from decoupled rewards + dynamic demonstration injection). Results: 76.1% on AndroidWorld from raw screenshots + open backbones; dual generalization (declarative knowledge transfers across base models; procedural skills on unseen ScreenSpot-v2/AitW). **Answer:** it improves **planning/knowledge (high-level)** and **skill execution (low-level RL)** — it is not a calibration/choice-head, and its low-level executor is an RL-trained action *generator*, not a one-pass scorer.

---

## 2. How confidence/uncertainty is modeled today (action selection & agents)

**Main finding: essentially none of the mobile GUI action predictors (UI-R1, Hi-Agent, MIRAGE, AutoDroid) emit or use a calibrated confidence signal for the next action.** Confidence/uncertainty work exists in three adjacent strands:

**(A) Confidence/adaptive-interaction for GUI agents directly.**
- **OS-Kairos** [os-kairos] (ACL Findings 2025): the closest thing to our gate. It targets **over-execution** — an agent acting fully autonomously without assessing action confidence. It predicts a **confidence level at each interaction step** and decides **act autonomously vs. seek human intervention**. Two mechanisms: (i) *collaborative probing* to annotate per-step confidence, (ii) *confidence-driven interaction* that uses the scores for adaptive interaction. Results: +24.59%–87.29% task success on their curated set and on AITZ/Meta-GUI. Mechanism is text-confidence annotation + a decision rule — not a calibrated distribution over candidate actions, and not one-pass.

**(B) Uncertainty-aware *grounding* with calibrated thresholds and safe deferral.**
- **SafeGround** [safeground] (arXiv 2602.02419): the most directly on-point for "calibration for GUI and a confidence gate." It estimates **spatial uncertainty** by aggregating **multiple stochastic grounding samples** into a **patch-level probability distribution**, then calibrates a **test-time decision threshold with statistically-guaranteed false-discovery-rate (FDR) control**, enabling **selective prediction and safe deferral**. On ScreenSpot-Pro it improves system-level accuracy by up to **5.38 pts** over Gemini-only inference. **Caveat:** it gets uncertainty by *over-sampling* (multiple stochastic passes) — not a single forward pass — and it is *grounding-level* (coordinate correctness), not full action selection over a K-set.

**(C) Generic agent/LLM abstention & calibration (not GUI-specific).**
- **AgentAbstain** [agentabstain] (arXiv 2607.10059): the strongest evidence that **agentic abstention is unsolved**. 263 paired should-act/should-abstain tasks across 42 MCP sandboxes, 8 abstention scenarios (pre-execution vs. runtime). Across 17 frontier models / 4 harnesses the **best is 59.5% paired accuracy** (Gemini 3.1 Pro), mean 45.7%; 13 of 17 below 50%. Crucially, **abstention capability scales largely independently of task-solving capability** (mean act accuracy 80.6% vs abstain 59.1%; Opus 4.7 is the only model whose abstain (79.0%) exceeds its act (76.5%)). Identifies *post-hoc abstention* — committing an irreversible action *before* recognizing the trigger. Conclusion: merely scaling models does not close the act/abstain gap; it must be explicitly designed/trained.
- **Do GUI Agents Know When Not to Act? (CONFLICTGUI)** [conflictgui]: benchmark on **instruction-internal conflicts** and **instruction–GUI context conflicts**, revealing severe **execution-biased overcompliance** — the GUI-specific instance of "knows how to act but not when not to."
- **CAAG — Calibrated Act–Ask–Abstain Gating** [caag]: a behavior-level control layer. A **lightweight calibration head** `g_φ` on a **frozen backbone**: `π(a|s,h)=softmax(g_φ(f_θ(s,h)))`, over `A={act, ask, abstain}`. An **expected-utility gating rule** `U = E[R] − λc·C_tool − λl·C_latency − λr·C_risk`, and act only if `P̂≥τ ∧ U(act)≥U(ask),U(abstain)`. Trained with `L_CE + α·L_ECE` + temperature scaling; `τ` chosen from a validation coverage constraint on false-action rate. Includes **memory-triggered reflection** (only when low confidence / repeated failure / stalled progress), cutting reflection overhead ~40–60%. **⚠️ Critical caveat:** the paper reports numbers from **"simulated analytical evaluation"** and states *live validation remains essential future work* — the venue is the low-tier IJCOPE journal. Its *architecture* is highly relevant (calibration head + utility-gated act/ask/abstain on a frozen backbone = exactly our shape); its *quantitative* claims (WebArena 21.6% TSR vs 18.2% uncalibrated; ECE 0.089 vs 0.187; FAR 17.1% vs 24.3%; ablations) are **unverified/low-confidence**.
- **ProbeCal — Uncertainty Calibration for Tool-Using Language Agents** [probecal]: identifies prompt design and execution-trace selection as two primary areas of tool-use calibration and proposes **probing** to align model probabilities with tool-use effectiveness. Foundational for "tool-use calibration," cited by CAAG.
- **Uncertainty-aligned RL for agentic tool calling** [unc-rl-tool] and **Risk-Aware Action Selection for Uncertainty-Calibrated Tool-Using Agents** [risk-aware]: both treat tool-call decisions under uncertainty; the latter proposes an *uncertainty-calibrated, risk-aware control architecture* for multi-step execution.
- **The Confidence Dichotomy** (ACL 2026) and **Confidence Trajectory Analysis** thesis [conf-dichotomy]: systematic study of calibration *dynamics* in tool-integrated agents — confidence competes with / diverges from performance; step-wise UQ ignores trajectory-level compounding.
- **Conformal / selective prediction for abstention** [cap, conformal-abstain]: CAP (Conformalized Abstention Policies) and "Learning Conformal Abstention Policies" — context-adaptive risk management with finite-sample guarantees for LLMs/VLMs; and "Mitigating LLM Hallucinations via Conformal Abstention."

**Synthesis for our gate:** confidence is modeled *today* either (i) by **annotating/deriving a text-level confidence and thresholding it** (OS-Kairos, CAAG), (ii) by **aggregating multiple stochastic samples** to get an uncertainty distribution and calibrating a threshold with FDR guarantees (SafeGround), or (iii) **not at all** (all the actual action predictors). **Nobody obtains calibrated ECE over a GUI candidate-action set in a single forward pass.**

---

## 3. Where the candidate-action set K comes from

Two competing paradigms:

**(a) Model *proposes* the action from pixels (generative grounding).** UI-R1, Hi-Agent, MIRAGE, Aria-UI, UI-Venus. The model emits `(type, coordinate)` directly from the screenshot. **There is no explicit K-set** — K must be *constructed retroactively* (e.g., by a grounding/segmentation post-hoc step) if you want a distribution over candidates. Pros: no separate perception stage, generalizes to unseen layouts, no extractor noise/bias (Hi-Agent explicitly removes UI annotations and SoM). Cons: off-element clicks; no natural notion of "the set of things I could have done," so a calibrated distribution over actions is awkward; small/dense elements (ScreenSpot-Pro regime) degrade.

**(b) External extractor enumerates clickable elements → K; model *selects* among them.** AutoDroid (accessibility/DOM → HTML, restricted to element **ids**), SoM (segmentation marks), ScreenSpot-style annotator, and the on-device accessibility tree (Android AccessibilityService / iOS AX). Pros: K is grounded to real, actionable elements (no invalid actions); selection reduces to a **discrete choice over a fixed set**, which is exactly what a calibrated choice-head/softmax can score in one pass; on-device the accessibility tree is *freely available* and cheap. Cons: adds an extractor stage + latency; extractor can miss / mis-bbox / add noise — and, notably, UI-R1 discards part of AndroidControl because its element annotations are **"unfiltered and misaligned"** — evidence that accessibility-derived element sets are noisy and action-space semantics vary; SoM segmentation marks are imperfect on dense UIs; the availability/quality of the AX tree varies by app (Flutter/RN/webviews often expose poor or missing semantics).

**On-device implication:** For an *on-device, one-pass, calibrated* action-scoring head, the **accessibility tree is the natural and most defensible K source** (free, low-latency, gives clickable bounds + content semantics). The trade-off is the same one Hi-Agent/UI-R1 accept against it: AX-less "pure-vision" generalizes but can't yield a clean K for a one-pass scorer. Coupling the decision head to an AX-derived K also means **a grounding/extractor error becomes a decision error**, so the gate's calibration inherits extractor noise — a real risk to manage.

---

## 4. Prior art for a "decision model / choice-head over GUI affordances"

**The one-pass calibrated choice-head already exists as a model class — but not for GUI affordances.**
- **TypeSafe AI — System One models / Jev** [jev]: the flagship "System One" model. It **evaluates a state and returns typed answers (choice / score / yes-no) and probabilities** in one round trip, no prose to parse, no schema to validate. Trained with **RLCD** (reinforcement learning for calibrated decisions). This is the vendor-origin of the exact model family we're building.
- **`decider` (Mapika)** [decider]: an open, independent reproduction of the System One class. **Reads a state + typed questions (Choice 2–255 options, Score 2–10 levels, Noul yes/no) and returns a probability distribution per question from one forward pass — no decoding, no parsing, no out-of-set answers.** Mechanics: `decider/model.py` reads the hidden state at each answer slot, projects it onto one label token per option (A–J, then K–Z, then two-letter tokens up to 255), and **softmaxes over the valid options at a fitted temperature**. `confidence` for a Choice = `(n·p_max − 1)/(n − 1)` (0 = uniform, 1 = one option). Includes a **confidence-gated routing** example and a **hierarchical beam over Choice probabilities**. Deployment: CUDA/MPS/CPU; **GGUF Q4_K_M 1.3 GB (2B) / 2.7 GB (4B)**; ~0.12–0.31 s per 40–120-token request on 8 CPU threads for the 2B; a community **Snapdragon X Elite NPU** port runs the 12B at ~2.2 s. Calibration is trained directly (v10 RL on games with known probability laws, scoring predicted outcomes by **log score**). **Limitation for us:** it is a **text/JSON-state** system — `decider-2b-vision` is the vision variant but reads Visual7W, **not GUI grounding/affordances**.
- **How it compares to our goal:** it already scores a fixed option set as a *calibrated* distribution in one pass, with no output tokens — the head architecture is essentially what we want. What it does **not** do: consume a GUI screenshot, build an affordance K-set, and gate act/ask/refuse/escalate.

**Adjacent but not the same.**
- **SafeGround** produces a calibrated *grounding* distribution + FDR-controlled deferral, but via **multiple stochastic samples** (not one pass) and at the grounding level, not over numbered affordance options.
- **CAAG + OS-Kairos** implement an act/ask/abstain gate over a confidence signal, but the confidence comes from a **calibration head on a frozen generative backbone that still emits candidate actions by decoding** — it does not score a fixed K in one pass.
- **AutoDroid** restricts to an enumerated id set (discrete choice over K) but on a *generative* frozen LLM with no calibration/confidence.

**Conclusion on §4:** The *mechanics* of a one-pass calibrated choice head exist (System One/Jev/decider); the *mechanics* of a GUI confidence gate and calibrated grounding exist (OS-Kairos, SafeGround); and the *mechanics* of restricting selection to an enumerated affordance set exist (AutoDroid, SoM). **The composition — a calibrated one-pass scorer over a GUI-affordance K-set feeding an act/ask/refuse/escalate gate — has no published direct prior art.**

---

## 5. Improvement directions vs. SOTA

**(a) Latency / cost — MOST defensible.**
- MIRAGE already proves the mechanism: moving reasoning out of the decoded token stream cut average tokens per step from ~115 → ~19 on AndroidControl low-level and ~103 → ~31 on AndroidWorld, and it is *faster* first-to-last-token [mirage]. A one-pass scorer over K (no action tokens at all) goes strictly further — zero decoded action tokens, one forward pass. The System One class proves the runtime: `decider` 2B does a whole decision in **0.12–0.31 s on CPU** and runs on an NPU [decider]. So the latency benefit is grounded in measured numbers from both ends (token-count reduction; one-pass runtime), not speculation.
- Also: removing the external *extractor* from the critical path (if K comes from the AX tree) vs. the cost of *generative* grounding that still needs the VLM to produce coordinates. A one-pass scorer over AX-K skips the multi-token grounding decode, additional win on the extractor free anyway.

**(b) Calibration / trust — RISKIEST, but valuable.**
- The hard evidence is *against* easy wins: AgentAbstain shows the best frontier agent reaches only **59.5%** paired act/abstain accuracy and that **abstention is largely independent of task-solving capability** [agentabstain]; SafeGround achieves guaranteed FDR only by **over-sampling** (multiple stochastic passes), not one pass [safeground]; CAAG's claimed calibration gains are **simulated, not empirically validated** [caag]. So claiming a calibrated ECE/FDR in a **single forward pass** (no sampling) is the **riskiest claim** — there is no evidence a single latent/representation read yields uncertainty as good as an *ensemble* of stochastic samples, and distribution shift across apps is real (AndroidControl's own annotations are "misaligned"; UI-R1 had to filter them [ui-r1]).
- Mitigation to make it defensible: combine **in-policy data-driven calibration** (temperature scaling / post-hoc recalibration on a small on-device set), **train the head with a calibration-aware objective** (as System One's RLCD and CAAG's `L_CE + α·L_ECE` do), validate on **ScreenSpot-Pro + AndroidControl + AndroidWorld** with an explicit ECE + FDR report, and report act/ask/refuse/escalate **coverage–risk curves** rather than a single operating point.

**(c) Error handling — the act/ask/refuse/escalate cascade.**
- The decision *vocabulary* and *architecture* are already specified by others: MARC defines exactly `ANSWER / CLARIFY / RETRIEVE / TOOL / DELIBERATE / ABSTAIN / ESCALATE`, a pre-decision `pre_capability`, uncertainty-source attribution (ambiguity / missing_evidence / capability_limit / evidence_conflict / safety), and post-decision `post_answer_confidence` with low/med/high bands [marc]; CAAG gives the **expected-utility gate** `U = E[R] − λc·C_tool − λl·C_latency − λr·C_risk` and a coverage-constrained threshold [caag]. **Defensible contribution:** applying this *cascade* specifically to a calibrated GUI-action distribution, and making the "escalate" branch hand off to cloud (matching MARC's ESCALATE) rather than just act/abstain. **Caveat:** the *cost coefficients* and *thresholds* are deployment-specific and unpublished for GUI; there is **no validated evidence** that the cascade improves end-to-end GUI success (CAAG's numbers are simulated; OS-Kairos reports large gains but is the only empirical GUI-confidence data point) [caag, os-kairos]. So the cascade is a *sound design*, but its *quantified benefit* is unproven.

**(d) On-device capability — defensible, with a real dependency.**
- One-pass zero-output decision heads are demonstrably on-device friendly: `decider` 2B Q4_K_M is 1.3 GB and runs ~0.12–0.31 s on CPU and on Snapdragon X Elite NPU [decider]. The AX tree is available on-device for free [autodroid]. So a small (~1–3B) action-scoring head over AX-K is feasible on phone-class hardware.
- **The risk is the coupling to K (grounding):** if K comes from the accessibility tree, then on-app AX quality (webviews/Flutter/RN often expose poor semantics) becomes a hard dependency — this is precisely why UI-R1/Hi-Agent drop the element-representation entirely [ui-r1, hi-agent]. A robust on-device design needs a fallback (e.g., OCR/SoM marks, or a vision grounding only when AX is unavailable), and that fallback adds the extractor latency the head was meant to avoid.

**Ranking of defensibility:** (a) latency/cost and (d) on-device are the **most defensible** because they rest on measured one-pass runtimes and measured token-reduction gains; (b) calibration/trust is the **riskiest** because the only guaranteed-calibration result (SafeGround) requires sampling, and agent abstention is empirically near-unsolved; (c) the cascade is architecturally well-specified but its quantified benefit is unverified for GUI.

---

## Coverage Status

**Checked directly (contents read, not just titles):** UI-R1 [ui-r1], Hi-Agent [hi-agent], MIRAGE [mirage], AutoDroid [autodroid], OS-Kairos abstract [os-kairos], AgentAbstain [agentabstain], CAAG full text [caag], MARC (action set + confidence semantics) [marc], `decider` README (mechanics, runtime, calibration) [decider], K²-Agent abstract [k2-agent], MobileUse abstract [mobileuse], SafeGround abstract [safeground], Aria-UI [aria-ui]. 

**Read from abstracts/snippets only (mechanism-level detail not fully inspected):** SoM [som] (mechanism from abstract + repo), ScreenSpot(-Pro) [screenspot-pro], GUI-Actor [gui-actor], ProbeCal [probecal], CAP/conformal abstention [cap, conformal-abstain], TypeSafe Jev [jev] (vendor docs, self-reported), UI-Venus 2 [ui-venus-2], CONFLICTGUI [conflictgui], risk-aware/uncertainty-aligned tool call [risk-aware, unc-rl-tool], confidence-dichotomy [conf-dichotomy].

**Uncertain / to be verified:**
- CAAG's quantitative results are **simulated**, not live — treat as low confidence [caag].
- TypeSafe Jev numbers are **vendor self-reported** [jev].
- UI-R1's exact config (learning rate, epochs, GRPO group size) is in the appendix I did not fully read — point/type reward + GRPO + 136 samples is confirmed; hyperparameters not [ui-r1].
- MIRAGE's per-epoch/training-cost numbers are in an appendix I did not read [mirage].
- ScreenSpot's exact dataset size/splits were not verified from the dataset card directly (only the benchmark paper abstract) [screenspot].
- Whether the System One head (letter-logit softmax over options) has been applied to *vision GUI affordances* anywhere: I found only `decider-2b-vision` reading Visual7W, not GUI grounding [decider]. **No source found that does this — treated as an open gap, not confirmed absent.**

**Tasks not completed:** none of the five assigned questions are blocked; each is answered above. Remaining open items are noted as uncertainties rather than skipped.

---

## Source table

| # | Bibkey | Title | URL | Type | Confidence |
|---|--------|-------|-----|------|------------|
| 1 | ui-r1 | UI-R1: Enhancing Efficient Action Prediction of GUI Agents by RL (AAAI'26) | https://arxiv.org/abs/2503.21620 (DOI 10.1609/aaai.v40i21.38816) | primary (peer-reviewed) | high |
| 2 | hi-agent | Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control | https://arxiv.org/abs/2510.14388 | primary (preprint/journal) | high |
| 3 | mirage | MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models | https://arxiv.org/abs/2606.04627 | primary (preprint) | high |
| 4 | autodroid | AutoDroid: LLM-powered Task Automation in Android (MobiCom'24) | https://arxiv.org/abs/2308.15272 (DOI 10.1145/3636534.3649379) | primary (peer-reviewed) | high |
| 5 | som | Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V | https://arxiv.org/abs/2310.11441 | primary (peer-reviewed) | high (mechanism in abstract) |
| 6 | screenspot-pro | ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Displays | https://arxiv.org/abs/2504.07981 | primary (benchmark) | high |
| 7 | screenspot | ScreenSpot: GUI Grounding Benchmark | https://huggingface.co/datasets/ZhuOnR/ScreenSpot | primary (dataset) | medium (sizes not verified) |
| 8 | aria-ui | Aria-UI: Visual Grounding for GUI Instructions | https://arxiv.org/abs/2412.16256 | primary (peer-reviewed) | high |
| 9 | gui-actor | GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents | https://proceedings.neurips.cc/paper_files/paper/2025/file/16130af940e9dabb43c726119bd3b42e-Paper-Conference.pdf | primary (peer-reviewed) | medium (abstract only) |
| 10 | k2-agent | K²-Agent: Co-Evolving Know-What and Know-How for Hierarchical Mobile Device Control | https://arxiv.org/abs/2603.00676 | primary (preprint/ICLR'26) | high |
| 11 | mobileuse | MobileUse: A Hierarchical Reflection-Driven GUI Agent for Autonomous Mobile Operation | https://arxiv.org/abs/2507.16853 | primary (NeurIPS'25) | high |
| 12 | os-kairos | OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents | https://aclanthology.org/2025.findings-acl.348/ | primary (peer-reviewed) | high |
| 13 | safeground | SafeGround: Uncertainty-Aware GUI Grounding with Safe Deferral | https://arxiv.org/abs/2602.02419 | primary (preprint) | high |
| 14 | agentabstain | AgentAbstain: Do LLM Agents Know When Not to Act? | https://arxiv.org/abs/2607.10059 | primary (preprint) | high |
| 15 | conflictgui | Do GUI Agents Know When Not to Act? Conflict-Aware Termination | https://arxiv.org/abs/2609.03438 | primary (preprint) | medium (abstract only) |
| 16 | caag | Calibrated Act–Ask–Abstain Gating for Agentic Language Models | https://ijcope.org/wp-content/uploads/2026/05/Calibrated-Act%E2%80%93Ask%E2%80%93Abstain-Gating-for-Agentic-Language-Models-in-Resource-Constrained-Interactive-Tasks.pdf | secondary (proposal; results simulated) | low (quantitative) / medium (architecture) |
| 17 | marc | MARC: A Control and Uncertainty Disclosure Profile for Generative Models and Agents | https://www.ietf.org/archive/id/draft-c4tz-marc-02.html | standards draft (informational) | high (spec) |
| 18 | jev | TypeSafe AI — System One Models & Jev | https://typesafe.ai/blog/introducing-system-one-models-and-jev | self-reported (vendor) | medium |
| 19 | decider | decider: one-pass typed decisions with calibrated probabilities (Mapika) | https://github.com/Mapika/decider/blob/main/README.md | self-reported (open-source) | high (mechanics/runtime) |
| 20 | probecal | Uncertainty Calibration for Tool-Using Language Agents | https://aclanthology.org/2024.findings-emnlp.978.pdf | primary (peer-reviewed) | medium (abstract only) |
| 21 | unc-rl-tool | Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned RL | https://arxiv.org/pdf/2606.06976v1 | primary (preprint) | medium (abstract only) |
| 22 | risk-aware | Risk-Aware Action Selection for Uncertainty-Calibrated Tool-Using LLM Agents | https://ak495867.vercel.app/assets/papers/Risk_Aware_Action_Selection_Agents.pdf | secondary (grey) | low (no venue/verification) |
| 23 | conf-dichotomy | The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents | https://aclanthology.org/2026.acl-long.520.pdf | primary (peer-reviewed) | medium (abstract only) |
| 24 | cap | CAP: Conformalized Abstention Policies | https://proceedings.mlr.press/v304/tayebati26a.html | primary (peer-reviewed) | medium (abstract only) |
| 25 | ui-venus-2 | UI-Venus-2 Technical Report: foundation GUI agent | https://arxiv.org/abs/2609.00028 | primary (tech report) | medium (abstract only) |

Note: bibkeys 21–24 were surfaced by search and read at abstract/snippet level; their mechanism-level claims are labeled accordingly in the Confidence column and in the Coverage Status section. Excluded from the table are any items for which no verifiable URL surfaced.
