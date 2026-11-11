# C — Termination / stop-head prior art, grey literature (2026-11-11)

Scope: sources the LIT note (`docs/research/2026-11-11-LIT-termination-one-model.md`) did **not** cover —
workshop/short papers, theses and tech reports, Chinese-language work, in-policy robot termination heads,
option/RL termination literature, and negative results. Everything already in the LIT note is marked
"already in LIT note" and is not treated as a finding.

Labels: **VERIFIED** = number read in the source's own text this session. **UNVERIFIED** = snippet or
secondary only. **INFERRED** = my reasoning. No cross-harness comparison is made anywhere below.

---

## 0. Verdict

- **T-A (matched comparison)** — **DUPLICATES (in part) / CONFIRMS-NOVEL (narrow).**
  The latency-vs-competence trade-off for a typed, single-pass "System One" decision model inside a live
  AndroidWorld agent is now *published and measured* (Jev study, §2.1): decision latency −73.9…−81.5 %,
  success 57.47 → 44.25 (T3A) and 50.00 → 31.90 (FastAgent), with the end-to-end latency win collapsing to
  must cite it and cannot claim "first standalone decision-model result on AndroidWorld".
  What remains unclaimed is the *matched* arm (same family, size, rows, harness, one-pass vs generation),
  which no source runs. Also: Jev loses **even on the shortest tasks** (−12.5 pp T3A, −15.3 pp FastAgent in
  the shortest-budget stratum) and more on the longest, which cuts against T-A's "keeps short-task
  competence" half while leaving its length gradient intact.
- **T-B (closed-loop dynamics / masking)** — **CONFIRMS-NOVEL for the GUI claim; DUPLICATES for one
  design sub-claim.** No source does verified-failure masking *by re-ranking the policy's own option
  distribution* on a GUI benchmark; the nearest thing is VLAA-GUI's deterministic Loop Breaker
  (**already in LIT note**), which forces a mode change rather than taking the runner-up of the model's
  distribution. Searched, none found.
  Separately, SparkVLA (§2.5) already measures the design choice our `status` head embodies — stop as an
  *independent* head vs stop competing inside the same candidate ranking — and reports 42.53 → 47.12 SR and
  80.37 → 96.65 stop accuracy. That is a mechanism argument we would otherwise have had to discover.
- **T-C (confidence under own-action shift)** — **CONFIRMS-NOVEL for the claim; SILENT on the shift itself.**
  No GUI paper measures "offline-valid confidence fails to flag no-effect actions online". But "a learned
  readout on the policy's own features predicts task failure / transition validity" is now established
  three times outside our setting (SAFE ROC-AUC 82.26/77.04 seen/unseen; VLAConf failure AUROC 0.876;
  StepReflect 82.16 % transition accuracy trained on 882 samples). Our novelty is the *closed-loop shift*
  and the environment-verified label source, not the existence of the head.

Answers to the three questions are in §3.

---

## 1. Sources read

| id | title | venue + year | URL | cites | what it is |
|---|---|---|---|---|---|
| S1 | When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows | arXiv 2610.13092, Oct 2026 (preprint, Nanjing U. + TUM) | https://arxiv.org/pdf/2610.13092 | n/a | 4,500 executions, 5 agent families, ALFWorld + AndroidWorld; swaps the LLM decision-maker for a typed "System One" model (Jev). Closest published analogue of our architecture. **Not in LIT note.** |
| S2 | SAFE: Multitask Failure Detection for Vision-Language-Action Models | **NeurIPS 2025** (UofT + TRI) | https://proceedings.neurips.cc/paper_files/paper/2025/file/392d0d05e2f514063e6ce6f8b370834c-Paper-Conference.pdf | n/a | Scalar failure head on the VLA's own last-layer features, trained on success/fail rollouts, conformal threshold. **Not in LIT note.** |
| S3 | VLAConf: Calibrated Task-Success Confidence for VLA Models | arXiv 2605.29605 v2, Aug 2026 (SUSTech + Zhongguancun) | https://arxiv.org/html/2605.29605 | n/a | Frozen-representation success score (Coin-Flip Net) + 2-param Platt calibration; LIBERO, LIBERO-Pro/Plus, real robot. **Not in LIT note.** |
| S4 | What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents | arXiv 2606.10267 v2, Sep 2026 — **Google DeepMind tech report** | https://arxiv.org/html/2606.10267v2 | n/a | Controlled ablation of Hi-VLA components incl. the **termination condition**; 5 tasks × 200 trials per category + real ALOHA. **Not in LIT note.** |
| S5 | SparkVLA: Stop-Aware Hierarchical VLA with Adaptive Action Chunking | arXiv 2608.16172, Aug 2026 (BIT + Tsinghua) | https://arxiv.org/pdf/2608.16172 | n/a | Stop modelled as one candidate in a unified ranking with action-prefix lengths; trained from offline ordinal preferences only. **Not in LIT note.** |
| S6 | StepReflect: Structured UI Transition Reflection for Mobile GUI Agents | arXiv 2608.05587, Aug 2026 (Concordia) | https://arxiv.org/html/2608.05587v1 | n/a | Learned 8B post-action transition-consistency head; trained on 882 GUI transitions, evaluated on AndroidWorld (1,082 transitions offline + 36 tasks online) and MobileWorld. **Not in LIT note.** |
| S7 | Mobile-Aptus: Confidence-Driven Proactive and Robust Interaction in MLLM-based Mobile-Using Agents | arXiv 2605.28629, May 2026 (SJTU + Meta) | https://arxiv.org/html/2605.28629 | n/a | In-policy confidence score emitted with the action (SFT + DPO), thresholded to request help; OS-Kairos/AITZ/Meta-GUI/AndroidControl + 50 real-device tasks. **Not in LIT note.** |
| S8 | AgentStop: Terminating Local AI Agents Early to Save Energy in Consumer Devices | **CAIS '26** (ACM Conf. on AI and Agentic Systems), May 2026 (UMass + Brave) | https://arxiv.org/pdf/2605.15206 | n/a | GBDT supervisor on already-produced logprob features that preemptively kills doomed trajectories; FRAMES, SimpleQA, SWE-Bench Verified, on-device energy. **Not in LIT note.** |
| S9 | When Does Learning to Stop Help? A Cost-Aware Study of Early Exits in Reasoning Models | arXiv 2606.30852, Jun 2026 | https://arxiv.org/html/2606.30852v1 | n/a | 18 task–model settings; learned multi-feature stopper vs scalar exits, with lost-correct risk calibration. The clearest "learned stopper does not beat a scalar / hurts" study. **Not in LIT note.** |
| S10 | Agentic Abstention: Do Agents Know When to Stop Instead of Act? | **NeurIPS 2026 poster** (U. Washington) | https://neurips.cc/virtual/2026/poster/150912 | n/a | 13 LLM-as-agent systems, 2 scaffolds, >28,000 tasks (WebShop, terminal, QA); "timely abstention" measured. **Not in LIT note.** |
| S11 | Who Verifies the Agent's Success Claim? Self-Report Leakage in GUI-Agent Completion Judging | arXiv 2610.29609, Oct 2026 (Nanjing U. + TUM) | https://arxiv.org/html/2610.29609v1 | n/a | Audit of 1,384 GUI-agent runs (AndroidWorld + extension + Themis); 180 dangerous misreports. **Not in LIT note.** |
| S12 | A3: Android Agent Arena … Essential-State Procedural Evaluation | **Findings of ACL 2026** (CUHK + vivo) | https://aclanthology.org/2026.findings-acl.184.pdf | n/a | 100 tasks / 20 live apps; trained essential-state reward model (A3RM) on 9,324 step samples. **Not in LIT note.** |
| S13 | Semantic Early-Stopping for Iterative LLM Agent Loops | arXiv 2606.27009, Jun 2026 (single author) | https://arxiv.org/html/2606.27009 | n/a | Judge-free semantic-convergence halt for Writer→Critic loops; HotpotQA. Marginal; included as a null witness for "free content signal replaces a judge". **Not in LIT note.** |
| S14 | OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents | **Findings of ACL 2025** | https://aclanthology.org/2025.findings-acl.348.pdf | n/a | The over-execution/confidence-threshold precursor. I read only its abstract + the numbers Mobile-Aptus reports for it (§2.7). Numbers UNVERIFIED at source. |
| S15 | The Option-Critic Architecture / The Termination Critic | AAAI 2017 / AISTATS 2019 | https://ojs.aaai.org/index.php/AAAI/article/view/10916 ; http://proceedings.mlr.press/v89/harutyunyan19a.html | high | Learned termination functions β(s). Background only; no GUI, no benchmark effect, and the readout is a *subtask* boundary, not task completion. |
| S16 | VLAA-GUI / SPA-Bench / ConflictGuard / VeriGUI / MobileGUI-RL | — | — | — | **already in LIT note.** One new fact only: VLAA-GUI is a poster in the *Second Workshop on Agents in the Wild* at ICML 2026 (https://icml.cc/virtual/2026/67912) in addition to the arXiv version. |

### 1b. Mechanism / benchmark / effect / transfer (acceptance table)

"in-policy?" = the stop-like readout is computed by the acting policy itself, with no second model call.
"transfer" = would the mechanism survive as a **zero-token** decision head on our one forward pass.

| source | venue/year | in-policy? | mechanism | benchmark | measured effect (table ref) | transfer to zero-token head |
|---|---|---|---|---|---|---|
| S1 Jev study | arXiv 2026 | **yes** (one bounded selection per step; the decision model is external to the agent) | typed choice over supplied candidates, incl. a terminal action | AndroidWorld, 116 tasks × 3 runs | T3A 57.47→44.25, FastAgent 50.00→31.90; decisions −73.9…−81.5 %, task time −10.6 %/+0.7 % (T5, T6, T8); premature completion 14/251, 5.6 % (T3) | **yes** |
| S2 SAFE | NeurIPS 2025 | **yes** (last-layer features of the same forward) | scalar failure head + functional-conformal threshold | LIBERO-10, SimplerEnv, real Franka + WidowX | ROC-AUC 82.26 seen / 77.04 unseen (avg. of 4 settings, T1); <1 ms overhead; token-uncertainty ≈ chance (44–58) | **yes** |
| S3 VLAConf | arXiv 2026 | **yes** (frozen features + 2 Platt params) | phase-conditioned pseudo-count score → success probability | LIBERO, LIBERO-Pro/Plus, real robot | failure AUROC **0.876** vs logistic **0.731** vs DeepSVDD 0.571 vs prior 0.500 (T2) | **yes** |
| S4 Hi-VLA | Google DeepMind tech report 2026 | **no for the detector** (2nd VLM call + privileged state); **yes for the "VLM-horizon" arm** | success detector vs fixed timer vs self-predicted execution length | MuJoCo ALOHA (5×200 trials/category) + real ALOHA | long-horizon 57.39 vs 52.36 vs **43.50**; reasoning 80.89 vs 72.62 vs 72.27 (T5). FP error is the destructive one (App. C) | detector **no** / the finding **yes** |
| S5 SparkVLA | arXiv 2026 | **partly** (extra shallow backbone pass at chunk boundaries) | `Stop` competes with all action-prefix lengths in one ranking; `L_rank + λL_stop`, offline ordinal labels | RoboCerebra (600 episodes), LIBERO | 42.53→**47.12** SR and stop acc 80.37→**96.65** when Stop joins the candidate set (T3); stop acc ≈80 % for *any* independent head | objective **yes**, architecture **no** |
| S6 StepReflect | arXiv 2026 | **no** (separate 8B model) | supervised binary transition-consistency head + rationale | AndroidWorld 1,082 transitions + 36 tasks online; MobileWorld 117 | 82.16 % overall (pos 86.79 / neg 76.12) from 882 training transitions (T1, T2); M3A 38.89→44.44 (T3); GPT-5.2 reflector **hurts** Seed-2.0-Pro 41.03→34.19 | **no** |
| S7 Mobile-Aptus | arXiv 2026 | **yes but token-emitting** (same call, score is generated text) | confidence 1–5 emitted after the action, threshold γ, SFT + DPO | OS-Kairos, AITZ, Meta-GUI, AndroidControl, 50 real-device tasks | STOP-action SR 20.00→66.67 / 5.16→88.02 / 0.00→94.73 (T3–VI); DPO calibration: SR 95.90→**88.62** (T12) | head **yes**, token cost **no** |
| S8 AgentStop | CAIS '26 | **yes-ish** (features already produced; no extra model call) | GBDT on smallest-logprob / length / token-overlap → kill the run | FRAMES, SimpleQA, SWE-Bench Verified | 0.6–0.7 AUC at steps 4–5; 15–20 % energy saved at <5 % utility loss; **1.7B logprobs show no separation** | **yes** (our logits are stronger than logprobs) |
| S9 LearnStop | arXiv 2026 | **no** (probe answer decoded per checkpoint) | logistic stopper on prefix-observable features + lost-correct risk control | GSM8K, MATH-500, MMLU-Pro, AIME-90, GPQA (18 settings) | +0.028 [0.002,0.053] over best scalar (GSM8K-32B); **−0.022** val-selected on GPQA-32B; ≤ scalar on all hard settings (T1) | **no** |
| S10 Agentic Abstention | NeurIPS 2026 | **no** (context engineering) | stopping rules distilled into the prompt | WebShop, terminal, QA; >28,000 tasks, 13 agents | timely recall 26.7→57.4 (Llama-3.3-70B); "some agents never abstain when they should" | **no** |
| S11 Self-report leakage | arXiv 2026 | **no** (post-hoc audit, two model calls) | deterministic evidence→policy auditor, claim-blind | AndroidWorld + extension + Themis, 465 records | 180 dangerous misreports; 78/151 on AndroidWorld-original (T1); withholding the self-report raises recall 13.3–43.4 pp | **no** (but the label-isolation rule transfers) |
| S12 A3 / A3RM | Findings ACL 2026 | **no** (separate 8B judge) | DAPO-trained essential-state judge, 981 pos / 7,260 neg | 100 tasks / 20 live apps | essential-state F1 95.3; task F1 96.0 (T2); positives oversampled ×4 to avoid all-negative collapse | **no** |
| S13 Semantic early-stop | arXiv 2026 | **no** (external embedder + judge) | cosine distance on consecutive drafts + patience window | HotpotQA, 60 questions | −38 % tokens at ΔIS −0.004 (p=0.81); quality-gated variant counter-productive | **no** |
| S15 Option-critic / Termination critic | AAAI 2017 / AISTATS 2019 | **yes** (β on the policy's own state) | learned termination function, control or distinguishability objective | classic RL control tasks | no GUI number, no benchmark success effect | form **yes**, evidence **no** |

---

## 2. Findings

### 2.1 S1 — Jev: a single-pass typed decision model on AndroidWorld (the closest published analogue to us)

**VERIFIED**, Table 5 (3 runs, 116 tasks, seed 0, budgets 10·c steps):

| config | AndroidWorld SR | official score | all-3-runs success |
|---|---|---|---|
| T3A-B (LLM decisions) | 57.47 ± 4.75 | 57.47 | 53/116 |
| T3A-J (Jev decisions) | **44.25 ± 6.05** | 44.83 | 34/116 |
| AutoDroid-B / -J | 18.68 / 16.38 | 19.68 / 17.39 | 18 / 14 |
| FastAgent-B / -J | 50.00 / **31.90** | 50.14 / 33.05 | 38 / 30 |

Paired outcomes (Table 6): T3A 9 gains / **55 losses**; FastAgent 27 / **90**.

Latency (Table 8, per decision): 7.24 s → 1.34 s (T3A, −81.5 %) and 5.40 s → 1.41 s (FastAgent, −73.9 %).
**Per task** (Table 8): 213.00 s → 190.36 s (T3A, −10.6 %) and 119.68 s → 120.50 s (FastAgent, **+0.7 %**).
The decision-level saving does not survive to end-to-end. That is exactly the failure mode plan §6 rule 6
guards against, and it is now a published AndroidWorld measurement.

Length dependence (Fig. 2b, Jev−baseline in pp by step-budget stratum). The paper's own text: "T3A and
FastAgent lose **12.5 and 15.3 pp even in the shortest-budget stratum**, with larger deficits in the
longest-budget stratum." The figure's per-stratum cells read T3A −12.5 / −11.1 / −17.2 and FastAgent
−15.3 / −17.1 with the stratum labels only partly recoverable in extraction, so I quote the text, not the
cells. Reading: a decision model loses competence *on short tasks too*, while the gradient with length still
points the way T-A predicts.

Termination-specific evidence, §5.1 "Errors become failures through termination or failed recovery":
"In the Wi-Fi task, Jev selects an ambiguously labeled switch and **declares completion with the goal
unmet. This terminal judgment ends the opportunity for correction.**" (VERIFIED quote)

Premature-completion count, §5.3 Table 3: **14 / 251 codable executions (5.6 %)**, κ=0.919. Denominator is
the *outcome-selected* set (43 family–task pairs, 258 executions, **both** ALFWorld and AndroidWorld), so
this is a label share, not a rate. Their own recommendation (§6) names our claim: train on "contrasting
successful and failed trajectories … decisions that restore progress or **correctly end a task**", plus
"calibrated confidence in whether any available action advances the goal".

Transfer to a zero-token head: **yes** (one bounded selection per step, no generation of free text), but Jev
*is* the external model, so this is a competitor datum, not a technique donor.

### 2.2 S2 — SAFE: a scalar stop/failure head on the policy's own features (robotics)

**VERIFIED.** Failure detector `f(σ(g(e_t)))` on the VLA's **last-layer hidden state**, trained only on
rollout-level success/fail labels (no per-step labels required), thresholded by functional conformal
prediction calibrated on successful rollouts.

- Table 1, averaged over the four simulation settings: **SAFE-MLP 82.26 (seen) / 77.04 (unseen) ROC-AUC**,
  SAFE-LSTM 81.43 / 78.00. Text: "both outperforming the best baseline by 4–5 % on unseen tasks."
  (Per-column attribution in the extracted table is unreliable; I use only the averaged row and the text.)
- Overhead: **<1 ms, <1 % of inference time** (footnote 2), vs 152–221 % slowdown for 10-sample consistency
  baselines.
- Token-uncertainty baselines are near chance: max-prob 50.25/53.83, avg-prob ~50, max entropy 44–53 (Table 1).
  This is the direct counter-example to "just threshold the policy's confidence".
- Baselines that need multiple samples (STAC, sample-consistency) do better on some unseen cells but are
  excluded from deployment for cost.

Transfer: **yes** — same forward pass's features, one scalar, no second model, no extra decode. It is a
*separate trained head* (post-hoc), which is exactly the shape of our `status`/effect readout. It does
**not** report a closed-loop SR effect of stopping (the paper stops the robot for a human/backtrack, no
intervention study), so it says nothing about T-B/T-C's loop dynamics.

### 2.3 S3 — VLAConf: the head is easy, the calibration is the hard part

**VERIFIED**, Table II (LIBERO, averaged; π0.5 backbone):

| method | ECE↓ | Brier↓ | NLL↓ | **Failure AUROC↑** |
|---|---|---|---|---|
| Success prior (constant) | 0.017 | 0.094 | 0.336 | **0.500** |
| Logistic regression on frozen features | 0.056 | 0.072 | 0.253 | **0.731** |
| DeepSVDD (one-class) | 0.057 | 0.096 | 0.342 | 0.571 |
| VLAConf (CFN + Platt) | 0.062 | 0.068 | 0.255 | **0.876** |

Two lessons we can use directly:
1. A **linear** readout on frozen policy representations already reaches 0.731 — the signal is in the trunk.
2. The gain to 0.876 comes from *phase conditioning* (step index) and 2-parameter Platt calibration, not
   capacity. Ablation VLAConf-NoStep (Table I) is worse in Brier/NLL in most settings.
   **INFERRED:** our head should take step index / normalised progress as an explicit input; a bare linear
   probe on the pooled feature will under-read.
Also VERIFIED: online confidence is not uniformly better than pre-execution — "others remain flat or degrade
near the terminal stage" (§V-B observation 4). That is a null on "more steps ⇒ better readout".

### 2.4 S4 — Google DeepMind Hi-VLA study: termination-condition ablation, incl. the harmful case

**VERIFIED**, Table 5 (MuJoCo ALOHA; 5 tasks × 200 trials per category; ±s.e.):

| termination condition | short-horizon | long-horizon | reasoning |
|---|---|---|---|
| VLM-based horizon (**the planner predicts its own execution length**) | 72.16 ± 0.95 | **43.50 ± 1.14** | 72.27 ± 1.14 |
| Success detector (separate VLM call, privileged state) | 74.65 ± 0.92 | **57.39 ± 1.52** | **80.89 ± 1.17** |
| Fixed horizon (400 steps) | 75.81 ± 0.93 | 52.36 ± 1.54 | 72.62 ± 1.17 |

Read this carefully, because it is the sharpest negative result in the sweep:
- **A same-model, no-extra-call self-prediction of when to stop is worse than a dumb fixed timer**, by
  −8.86 pts on long-horizon tasks. It is the worst of the three on the category that matters.
- A *verified* success detector gains +5.03 (long) and +8.27 (reasoning) over the fixed timer, and loses
  −1.16 on short-horizon. So the stop signal pays only where command sequencing exists.
- Appendix C, detector-error sweep: 10 % detector error "does not hurt performance at all, and in fact
  slightly boost[s] it"; rising **false positives** "start to drastically impact the system performance,
  likely because of the false 'command completed' feedback that causes the VLM to move on"; false negatives
  are moderate *if independent across steps*, but "once a FN occurs, the command may fail to terminate for a
  very long time". Asymmetry: **FP (premature done) is the expensive error; correlated FN is the second.**
- Real ALOHA, Table 1: best hierarchy 12/15 fruits, naive hierarchy 9/15, flat VLA 3/15 — the success
  detector in the real-robot config uses no privileged state, i.e. the effect survives without a simulator
  oracle.

Transfer: the *success detector* implementation is **no** (a second VLM call). The *finding* about where the
stop decision should live is transferable and is the reason to prefer a head over a self-predicted horizon.

### 2.5 S5 — SparkVLA: stop as a candidate, not an independent head

**VERIFIED**, Table 3 (RoboCerebra, SR % / stop accuracy %), all rows share the same low-level executor:

| configuration | SR | Stop Acc. |
|---|---|---|
| (a) independent stop head + fixed chunk | 34.26 | 79.52 |
| (a) independent stop head + RLPD | 37.62 | 80.05 |
| (a) independent stop head + AQC | 40.92 | 80.43 |
| (a) independent stop head + pairwise ranking | 42.53 | 80.37 |
| **(b) stop inside the unified candidate set** | **47.12** | **96.65** |

Main table: 47.12 vs hierarchical baseline HPE 16.55 (+30.57) and reproduced Mem-0 20.29 (+26.83).
Method: `Stop` and every action-prefix length `1..H` are scored by one head; `L = L_rank + λ·L_stop`, where
`L_stop` contrasts the Stop score against a log-sum-exp over continuation scores, `y^stop = 1[t ≥ τ]`.
Trained from **offline ordinal preferences only, no online RL**.

Two things we did not have:
1. **The independent stop head saturates near 80 % stop accuracy no matter how good the chunk selector is.**
   Putting Stop in the same candidate set moves it to 96.65 and buys +4.59 SR. **INFERRED:** our `status`
   option is currently an "independent head" in exactly this sense (a separate choice over action type), and
   this is a concrete argument (and a measurable) that it should compete in the same normalised score space
   as the element/app/scroll choices rather than be thresholded separately.
2. For unsuccessful rollouts where the boundary τ is undefined, SparkVLA's priority design makes "all
   action-prefixes tie above Stop, encouraging continued exploration". I.e. **they deliberately do not train
   the stop label on failed episodes.** That is the opposite of our plan to use premature-done negatives, and
   it is a real design fork (they cannot end a hopeless episode; we want to).

Transfer: **partly**. One scoring pass at chunk boundaries, but on top of a separate shallow backbone pass
(`m_t` from the first `ℓ_m` layers) — not literally the same forward. The ranking-objective idea transfers
to a zero-token head; the architecture does not.

### 2.6 S6 — StepReflect: a *learned, cross-benchmark* transition head on AndroidWorld (second model)

**VERIFIED.** 882 training transitions (517 pos / 365 neg = 41.4 % negative) from Mobile-Eval-E + SPA-Bench;
**no AndroidWorld transition used for training**; test = 1,082 hand-verified AndroidWorld transitions
(613 pos / 469 neg).

Table 1 (offline, same structured input for every model):
- Qwen3-VL-8B base 73.11 overall / positive 67.05 / **negative 81.02**
- + SFT 80.87 / 79.12 / 83.16; + teacher–student distillation 78.37 / 69.17 / **90.41**
- GPT-5.2 zero-shot 70.33; GPT-4o 66.17. Final StepReflect **82.16** overall, positive 86.79, negative 76.12.

Table 2 (objective sweep) is the cleanest published statement of the precision/recall trap in a GUI stop-like
head: GRPO-only 82.90 overall (83.20 / 82.52); DPO-only **81.24 with negative 86.35 but positive 77.32**;
final config gives up 10.2 pts of negative accuracy to gain 9.5 pts of positive accuracy. Their GRPO reward
is explicitly asymmetric: +2.5 / +2.0 for correct Yes / No, −2.0 / −2.5 for FP / FN (penalise false rejection
more).

Table 4 (ablation): description-only input (no explicit pre/post-conditions) → 77.91 overall, positive
**67.21**, negative 91.90. Structured transition spec is worth +4.25 overall and +19.58 positive. **INFERRED:
this is the strongest available evidence that the *representation* of "what should have changed" matters more
than head capacity — directly relevant to whether our trajectory line should carry an expected-effect column.**

Table 3 (online, AndroidWorld, fixed subset of **36 medium-difficulty tasks**):
- M3A: GPT-5.2 reflector 38.89 → **StepReflect 44.44** (+5.55); SFT+GRPO 33.33.
- Agent-SAMA: built-in GPT-5.2 reflector 58.33 → StepReflect 55.56 (−2.77, one task).
- MobileWorld (117 tasks): MAI-UI-8B none 21.37 → GPT-5.2 25.64 → **StepReflect 29.91**.
- MobileWorld: Seed-2.0-Pro **none 41.03**, GPT-5.2 **34.19 (−6.84)**, StepReflect 41.88.
- Calibration finding: "GPT-5.2 issues negative verdicts at 19.3 % … and triggers **42 early terminations
  compared with 19** for StepReflect" — the more conservative verifier terminates more and scores worse.

Note the honest limitation they state: 36-task subset, single run, "does not represent full-benchmark
performance".

Transfer: **no** (separate 8B model, second call). But it is the closest thing in existence to "trained
verifier attached to a GUI agent, measured online on AndroidWorld", and it *fails to help on the one agent
that already has a state-aware reflector* — a warning about headroom.

### 2.7 S7 + S14 — Mobile-Aptus / OS-Kairos: the in-policy confidence-score lineage (token-emitting)

**VERIFIED** (Mobile-Aptus; OS-Kairos numbers are theirs).
Mechanism: SFT the agent to emit `(action, confidence 1–5)` in one call, then DPO with retrieval-mined pairs
to fix over-soliciting. Same model, same call, **but the score is generated tokens** — not free like a head.

Table III–VI, STOP-action SR (i.e. accuracy on the `COMPLETE`/`IMPOSSIBLE` action), base → Mobile-Aptus:
OS-Kairos **20.00 → 66.67**; AITZ **5.16 → 88.02**; Meta-GUI **0.00 → 94.73**; AndroidControl 53.62 → 81.27.
Task SR: OS-Kairos 14.29 → 68.32 (relative to their untrained probe, not a like-for-like baseline).
Real device (Table VIII, 50 instructions, 10-step cap): TSR 26.00 (OS-Atlas-Pro-7B) → 62.00 (Mobile-Aptus),
and against the interactive baseline OS-Kairos 70.00 → 62.00 but with **AIF 0.96 → 0.64** and
RE 86.42 → 93.09.

Table XII ablation — the negative-direction number we want:
| | IP↑ | HSR↑ | AIF↓ | SR↑ |
|---|---|---|---|---|
| Base | – | – | – | 61.36 |
| SFT | 70.75 | 86.87 | 1.59 | **95.90** |
| SFT+DPO w/o retrieval | 74.89 | 87.44 | 1.41 | 91.42 |
| Mobile-Aptus (SFT+DPO+retrieval) | **84.83** | 88.62 | **1.06** | **88.62** |

Sharpening the stop/intervene decision costs **7.28 pts of task success** while cutting interventions from
1.59 to 1.06 per episode. Table IX: prompt-only "output your confidence" is worthless — best prompt baseline
TSR 46.58 vs 68.32 for the trained version. **The signal must be trained, not asked for.**

Transfer: **mechanism yes / cost no.** A confidence head on the same trunk is what we plan; their version
pays tokens per step (+0.09 s/step, Table XI). Their STOP-accuracy column is the closest published analogue
of our `status` head's accuracy on classification-style benchmarks (not AndroidWorld).

### 2.8 S8 — AgentStop: how weak a stop predictor is when it is cheap

**VERIFIED.** GBDT on features already produced by inference (top-10 smallest logprobs, output-token count,
adjacent-step token overlap as a loop proxy), one model per (agent, step); predicts "this run will fail",
used to kill it.

- Web QA, Qwen3-30B-A3B: "**0.6–0.7 AUC in the first 4–5 steps** on both FRAMES and SimpleQA"; ≥20 % energy
  wastage reduction at <5 % utility drop.
- **Qwen3-1.7B: "the logprobs … do not generally share this pattern, as the successful and failed runs show
  more logprobs overlap."** A 1.7B model's confidence does not separate success from failure at all.
  This is the single most uncomfortable number in the sweep for a 2B decision model.
- Loop proxy is a feature, not a mechanism: "Ratio of tokens overlap between any two adjacent agent steps".

Transfer: **yes** (no extra model call; features are free). Caveat: it needs token logprobs, which a
zero-token decision head does not have — it has logits *over discrete choices*, a stronger signal, so the
0.6–0.7 figure is a floor, not an estimate.

### 2.9 S9 — When does learning to stop help: the "learned stopper loses to a scalar" study

**VERIFIED**, Table 1 (paired bootstrap vs the strongest scalar baseline in each row):

| task | model | learned peak | best scalar | learned − best scalar [95 % CI] | val-selected gain |
|---|---|---|---|---|---|
| GSM8K | 32B | +0.157 | Ent. +0.130 | **+0.028 [+0.002, +0.053]** | +0.136 |
| MATH-500 | 8B | +0.047 | Ent. +0.024 | **+0.023 [+0.005, +0.042]** | +0.043 |
| MMLU-Pro | 32B | +0.035 | Conf. +0.048 | −0.013 [−0.032, +0.006] | +0.026 |
| AIME-90 | 32B | +0.016 | Stable +0.026 | −0.010 [−0.033, +0.010] | **−0.009** |
| GPQA | 32B | +0.048 | Conf. +0.051 | −0.002 [−0.036, +0.031] | **−0.022** |

Finding: a learned multi-feature stopper beats scalars only on free-form math; on hard settings it is at or
below a single confidence threshold, and its validation-selected operating point is *negative* on GPQA 32B.
Their mechanism explanation (Fig. 2) is the transferable part: "AIME and GPQA contain many unsolved or
harmful trajectories, where aggressive learned stopping is harder to justify." **INFERRED: AndroidWorld at
~15–25 % SR is an "unsolved trajectories" regime, so this is a direct caution for a stop head trained on it.**
Also VERIFIED: their stopper is hidden-state-free and *worse* than hidden-state probes are claimed to be
(Thought Calibration), i.e. our head has more to work with than theirs.

Transfer: **no** as a method (needs a probe answer = extra decoding per checkpoint). **Yes** as a null.

### 2.10 S10 — Agentic Abstention (NeurIPS 2026): the base rate of bad stopping in agents

**VERIFIED from the abstract page** (poster 150912, Wed Dec 9 2026): 13 LLM-as-agent systems, 2 scaffolds,
>28,000 tasks, WebShop / terminal / QA. "Some agents **never abstain when they should**, while others do so
**only after many unnecessary interactions**." "Larger or more capable models [are] not always … better at
timely abstention." Their fix, CONVOLVE, "distills full interaction trajectories into reusable stopping
rules" and raises Llama-3.3-70B's **timely recall 26.7 → 57.4** and overall recall 83.2 → 100.0, *without
updating parameters*.
Numbers beyond the abstract (per-benchmark tables, the AndroidWorld-adjacent slice) **UNVERIFIED** — I did
not open the paper body (poster page only).

Transfer: **no** (context engineering). Use: it establishes that "stopped eventually" and "stopped at the
right time" are different measurements, and that the gap is large — which is the same distinction as our
overdue vs premature split, on a much bigger sample. It also shows a prompt-side fix can move timely recall
by 30 pts, so a trained head must beat that, not zero.

### 2.11 S11 — Who verifies the success claim: the only AndroidWorld premature-done counting I found

**VERIFIED.** 1,384 in-scope GUI-agent attempts → 465 audit-target records → **180 dangerous misreports**
(an unhedged success claim the trace does not support).

Table I composition:
| slice | total | POS | NEG | median steps |
|---|---|---|---|---|
| AndroidWorld original | 151 | **78** | 73 | 8 |
| AndroidWorld extension | 276 | 88 | 188 | 11 |
| Themis | 38 | 14 | 24 | 17 |
| agent = appagent | 177 | 90 | 87 | 9 |
| agent = mav3 (Mobile-Agent-v3) | 288 | 90 | 198 | 11 |

So: **78/151 = 51.7 %** of audited AndroidWorld-original report-producing runs are premature success claims,
and 180/465 = 38.7 % overall. These are *conditional on reaching the audit stage* (clean passes and
non-auditable runs excluded by construction), so they are not population rates. Both agents are generative
LLM agents, not decision models.

The mechanism result is directly relevant to why a self-report is a bad label: withholding the agent's
self-report raises trace-only dangerous-misreport recall by **43.4 pp** (Qwen3.5-35B-A3B) and **13.3 pp**
(DeepSeek-284B-A13B) — i.e. the agent's own claim contaminates any judge that sees it. **INFERRED: any
"did I finish?" signal that is trained on or evaluated with the agent's own success claim is circular; this
is an argument for environment-verified labels (our checker) independent of T-C's AUROC claim.** Their
auditor reaches F1 0.75–0.78 (465 records) / 0.77–0.81 (held-out Batch 2).

### 2.12 S12 — A3: constructed-negative completion detection works, at 1:7.4 imbalance, in a separate model

**VERIFIED.** A3RM = Qwen3-VL-8B fine-tuned with DAPO on **8,241 step-wise samples: 981 positive / 7,260
negative** (plus 1,083 mined negatives from agent trajectories), positives oversampled ×4 "to prevent the
model from collapsing to a trivial solution of always predicting a negative outcome".
Table 2: essential-state level precision 95.7 / recall 94.8 / F1 95.3 / accuracy 96.6 (vs Gemini-2.5-pro
87.3 / 96.3 / 91.5 / 89.5); task level precision 96.0, F1 96.0, accuracy 98.0.
Benchmark: A3RM-evaluated SR, e.g. T3A+Gemini-2.5-pro 53.0, InfiGUI-R1 27.0, most open models 3–20 %.

Transfer: **no** (separate 8B judge, second call). Use: (i) milestone-state detection is learnable at
~96 % with ~1:7 negatives and **requires** positive oversampling to avoid the all-negative collapse — the
same failure mode StepReflect shows; (ii) the ESAR ≫ SR pattern ("agents possess sufficient semantic
understanding to initiate tasks … but lack the long-horizon robustness required to reach the final state")
is the progress-vs-completion gap our `status` head would sit in.

### 2.13 S13 — Semantic early-stopping (marginal, included as a null)

**VERIFIED.** HotpotQA, 60-question test split: a judge-free stopper (cosine distance between consecutive
draft embeddings < ε for k rounds) cuts operational tokens **38 %** at parity quality (ΔIS = −0.004, p = 0.81);
the quality-gated variant is **counter-productive** because per-round judging dominates cost.
Transfer: **no** (external embedder + judge, multi-agent loop). Use: one more instance of "a free content
signal matches an expensive judge, and adding the judge on top destroys the gain" — the same shape as our
screen-hash rule vs a trained effect head.

### 2.14 S15 — Option-critic / termination critic (background, explicitly not transferable)

**VERIFIED** at the level of the abstracts/known formulation: options are triples `(I, π, β)`; β(s) is
learned to optimise a control or information objective; The Termination Critic argues terminations should
maximise *distinguishability* of the option's effect rather than control value.
Why it is **not** transferable as evidence for our claim: no GUI, no benchmark success number, and β marks a
*subtask* boundary inside one episode, not "the user's task is complete". It is the right citation for the
*form* of the head (a learned scalar on the policy's state) and for nothing else. `INFERRED`: the
distinguishability objective in The Termination Critic is a plausible auxiliary for our effect head
("does this option still change anything?"), which is closer to T-C than to T-B.

---

## 3. What this changes for us

**(a) Is the LIT note's claimed absence still true?**
**Yes, with the fence moved in.** No source trains a completion/stop readout into the same policy and reports
an AndroidWorld effect. But three near-misses now exist that the LIT note did not have:
- **StepReflect** (S6): trained head, AndroidWorld online, 36 tasks — but a separate 8B model, and it is
  *transition consistency* not *task completion*.
- **Mobile-Aptus** (S7): in-policy, same call, STOP-action accuracy 20.00 → 66.67 on OS-Kairos — but by
  emitting tokens, and no AndroidWorld.
- **Jev** (S1): a one-pass decision model measured on AndroidWorld — but not a stop head.
Also, the LIT note's "no standalone decision-model result on AndroidWorld" (plan F10) is now **false**: S1 is
one. That is the single most consequential correction in this report.

**(b) Is there a published "premature done rate" on AndroidWorld for a non-generative agent?**
**No.** Searched; none found. What exists, none of which is that number:
- S11: 78/151 (51.7 %) dangerous misreports among *audit-target* AndroidWorld-original runs, agents
  `appagent` and `mav3`, both generative. Conditional denominator, not a rate. (VERIFIED)
- S1: 14/251 (5.6 %) premature-completion label share in an outcome-selected 43-pair subset spanning ALFWorld
  **and** AndroidWorld, decision model = Jev. Closest thing; wrong denominator, wrong scope. (VERIFIED)
- already in LIT note: SPA-Bench M3A 24.4 %, T3A 36.8 %, AutoDroid 49.4 % — SPA-Bench, not AndroidWorld.
Our own B0/R1 counting will be the first AndroidWorld rate for a decision model. That is now a
*measurable contribution*, not a sanity check.

**(c) Best evidence that a constructed-negative `status` head is learnable at all?**
Ranked, with the caveat each carries:
1. **StepReflect** (S6) — 882 transitions, 41.4 % negative → 82.16 % overall on 1,082 held-out AndroidWorld
   transitions, and cross-benchmark (trained on Mobile-Eval-E + SPA-Bench). Best single datum: it is the same
   task family, the same metric shape (accept/reject a state claim), and it is **not** trained on the eval set.
   Its ablation also shows *how* it fails: DPO-only buys negative accuracy with positive accuracy
   (87.42 vs 75.04 → 76.12 vs 86.79 depending on order).
2. **SAFE** (S2) — 82.26/77.04 ROC-AUC from rollout-level labels only, no per-step annotation, <1 ms overhead.
3. **Mobile-Aptus** (S7) — the largest absolute lift of a stop-action head in a GUI agent (20.00 → 66.67
   OS-Kairos; 0.00 → 94.73 Meta-GUI) via SFT + DPO, but every number is on a classification-style benchmark,
   never a live benchmark.
4. **A3RM** (S12) — 96 % precision with 1:7.4 negatives **only with positive oversampling**; without it the
   head collapses to all-negative. This is the concrete recipe for our class imbalance (AndroidControl has
   ~0 non-final `status` rows).
5. **AgentStop** (S8) — the floor: 0.6–0.7 AUC from cheap confidence features, and **no separation at all for
   a 1.7B model**. Read as: if we rely on the 2B trunk's implicit confidence we should expect ~chance; the
   head must be trained against constructed labels, not thresholded.

**What to add to the plan (each is a change, not a note):**
1. **Cite S1 in plan F10 and §4 T-A.** Remove "a standalone decision-model result on AndroidWorld was not
   found". Add the constraint it imposes: at the shortest budget stratum a decision model lost 12.5–15.3 pp,
   so T-A's "keeps short-task competence" is the half most at risk and should be the first thing Q1 tests.
2. **T-B design change (from S5):** make `status` compete in the same normalised candidate score space as
   element/app/scroll, and log *stop accuracy* (agreement with the true final step) as a primary mechanism
   metric. SparkVLA's 80.37 → 96.65 is a published target; if our `status` sits at ~80 % we now know why.
3. **T-C representation change (from S6 Table 4):** put the expected effect of the *next* action into the
   input, not just the observed changed/no-change of the last one. +19.58 pts of positive accuracy in their
   ablation, at +4.25 overall.
4. **T-C metric change (from S3):** include normalised step index in the head's input (VLAConf-NoStep
   ablation) and report Brier/NLL/ECE alongside AUROC — a head can have AUROC 0.876 and ECE 0.062, and
   AUROC alone will not tell us if the threshold is usable.
5. **Negative-direction guard (from S4 App. C):** our kill rule must be asymmetric. A FP (premature done) is
   costly; an independent FN is cheap but a *correlated* FN (stuck screen ⇒ never done) is expensive. That
   is a testable property of our head on our traces: measure the autocorrelation of `p(status)` along
   no-effect runs.
6. **Report timely vs eventual stopping (from S10):** our metrics already have premature/overdue (from SPA-Bench);
   add "steps wasted after the first correct done point" so a head that stops eventually cannot look good.
7. **Do not train `status` on the agent's own claim (from S11):** label only from the checker. Any AUROC whose
   labels came from the agent's self-report is inflated (recall drop 13.3–43.4 pp when the claim is withheld).
   Our checker labels already satisfy this; state it explicitly in the paper as a design requirement.
8. **Drop from consideration:** S9 (learned stopper needs probe decoding), S4's success detector, S6,
   S12, S13 as *implementations* — all violate the one-model/zero-token constraint. They are evidence only.

**Compensating result in our favour:** S4 Table 5 shows a *self-predicted* horizon is the worst option and a
verified one is the best, and S5 shows an independent stop head caps at ~80 %. Together they say the
interesting object is not "a stop head" but "where the stop decision sits and what verifies it" — which is
exactly T-B's masking story, and it is not yet done on AndroidWorld.

---

## 4. Searches run (queries + sources searched + nulls)

**Surface 1 — workshops / short papers (2025–2026).**
- `GUI agent termination head "done" prediction single policy AndroidWorld completion detection`
- `MobileHCI 2025 GUI agent loop detection repeated action termination workshop paper`
- `workshop paper 2026 "GUI agent" "completion detection" OR "termination" OR "when to stop" short paper`
- `"NeurIPS 2026" OR "ICLR 2026" workshop GUI agents mobile agents paper list stopping looping failure`
- `openreview workshop 2026 "GUI agent" repeated action loop detection OR termination OR completion judge`
- `"workshop" 2026 short paper "loop detection" OR "repetition" mobile agent poster abstract stopping criterion GUI`
- Sources searched: NeurIPS 2026 downloads index (neurips.cc/Downloads/2026), ICML 2026 workshop listings,
  ICLR 2026 workshop listings, OpenReview, ACL Anthology 2026 Findings, MobileHCI '25 adjunct proceedings.
- **Null:** no workshop or short paper proposes a *trained in-policy* stop/completion readout for GUI agents.
  The only workshop-hit "Knowing When to STOP, RECOVER, and SEARCH" is VLAA-GUI (ICML 2026 *Agents in the
  Wild* poster), **already in LIT note**. Two workshop-adjacent items were found but are out of scope:
  *Scale Test-time Compute in Mobile GUI Agents with Parallel Speculative Execution* (EdgeFM '25) and
  *Demo Abstract: On-Device Mobile GUI Agent System with Online Exploration* (SenSys adjunct 2026) — neither
  concerns termination.

**Surface 2 — theses and tech reports.**
- `thesis "mobile GUI agent" termination criterion "task completion" detection university repository 2026`
- (via the above) found: Google DeepMind Hi-VLA study (S4) — the only tech report with a termination-condition
  ablation table; Microsoft/Apple/Meta: nothing on stopping criteria for mobile agents.
- **Null:** no master's/PhD thesis on mobile-agent stopping criteria surfaced in these queries. I did not
  have a thesis-specific index (ProQuest/ETD) available, so this is a search-surface null, not a proof.

**Surface 3 — Chinese-language work.**
- `GUI智能体 终止判定 任务完成判定 循环检测 手机 智能体`
- `GUI智能体 终止动作 完成判定 训练 头 输出 "done" 动作 准确率 手机 智能体 论文`
- `硕士学位论文 移动端 GUI智能体 任务完成 判定 终止 循环 检测 2026`
- Sources searched: general web + alphaXiv zh + arXivDaily zh + 机器之心/腾讯云/掘金/知乎 mirrors.
- **Null:** every Chinese-language item is either a translation of an English arXiv paper (VLAA-GUI, GraceBench,
  ConflictGuard, MAI-Gym, GUI-JEPA) or an engineering blog on agent stop conditions (e.g.
  phppan.com/2025/10/ai-agent-stop — generic LLM-agent stopping strategies, no GUI, no measurement).
  The one Chinese-authored primary line is **OS-Kairos / Mobile-Aptus** (SJTU+Meta), which is published in
  English. **I found no Chinese-language-only paper or thesis on 终止判定 with a trained head.**

**Surface 4 — non-GUI transferable (in-policy readouts).**
- `option-critic termination head semi-Markov option termination function learned stopping`
- `robot manipulation policy learned success classifier head "termination" vla predicts task success from hidden state`
- `hierarchical VLA subtask termination head learned "subtask completion" policy switch success rate long-horizon`
- `learned halting head tool-use agent "when to stop" internal representation probe stopping`
- Found: SAFE (S2), VLAConf (S3), Hi-VLA (S4), SparkVLA (S5), option-critic/termination-critic (S15),
  *On Semi-Markov Suboptimality in Hierarchical RL* (arXiv 2610.05338, not read — no benchmark number in
  snippet, marked **UNVERIFIED** and not used).
- **Null:** no robot paper reports an *end-to-end task-success effect of stopping on the readout* — all report
  detector quality (AUROC/bal-acc/detection time) or a subtask-switch effect. S4 Table 5 is the closest and
  its detector is a second VLM call.

**Surface 5 — negative direction (stop head hurts).**
- `"premature termination" OR "over-eager" stopping head ablation hurts agent performance learned stop policy false positives`
- `GUI agent reinforcement learning "terminate" action reward "premature" OR "early termination" penalty AndroidWorld ablation success rate`
- `"AndroidWorld" "false completion" OR "premature termination" OR "declares done" rate percentage taxonomy failure analysis`
- Found: S4 (VLM-horizon worst; FP sweep), S6 Table 3 (GPT-5.2 reflector −6.84 pts on Seed-2.0-Pro, 42 vs 19
  early terminations), S7 Table XII (DPO calibration: −7.28 pts SR), S9 Table 1 (learned stopper ≤ scalar on
  hard settings; negative val-selected gain), S8 (1.7B model: no signal at all).
  **Already in LIT note:** ConflictGuard gate ablation (24–48 pt feasible-SR loss), MobileGUI-RL, SPA-Bench
  premature/overdue trade-off, VLAA-GUI's sign flip for Gemini at 50 steps.
- **Null:** no paper reports an ablation "add a stop head, remove it" on a GUI live benchmark; every GUI
  negative result is a side effect of a confidence/verification mechanism, not a controlled stop-head ablation.
  **That ablation is unclaimed.** `INFERRED`: it is also the cheapest thing we can run (mask `status` at
  inference on logged trajectories, no retraining).

**Surface 6 — leaderboards, GitHub, HF model cards, vendor blogs.**
- `huggingface model card GUI agent "done" head OR "termination" head small model 2B android action head outputs status`
- GitHub/leaderboard checks surfaced Droidrun (claimed 91.4 % AndroidWorld, vendor blog), Taqwright LIME
  (claimed 97.4 %, 113/116, vendor benchmark page), Midscene report — all single-run vendor claims with no
  seeds, no stop analysis, no termination metric.
- **Null:** no public AndroidWorld leaderboard entry, repo, or model card reports a premature-done rate or a
  stop head. The official AndroidWorld leaderboard could not be enumerated from these queries (it is a
  JS-rendered table) — **UNVERIFIED**, and a gap for whoever holds that surface.
