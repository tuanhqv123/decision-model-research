# Slice D — Calibration / selective prediction / abstention for GUI & mobile agents (2026-11-11)

Scope: what is *outside* the well-known arXiv set (CORA, AgentAbstain, TypedBench, THX-01, decider,
GUI-JEPA margin routing, PACE, SelectiveNet, CRC, SCRC). Targets were unindexed/no-paper results,
leaderboard entries, GitHub, vendor blogs, HF cards, workshop/thesis/non-English papers and
small-venue papers. Every number below was read in the source's own HTML/PDF text this session
unless explicitly marked otherwise.

## 0. Verdict

- **T-A** (matched one-pass vs AR comparison at equal family/size/data/harness, latency win on short tasks, loss past a length): **SILENT**. Nothing in this slice compares a one-pass decision model to an AR agent at matched budget; the only single-forward-pass GUI signal work (PACE, logit gates in MID-Bench) is *confidence scoring around* an AR policy, not a one-pass policy.
- **T-B** (verified-failure masking, next-best option of the model's own distribution): **CONFIRMS-NOVEL** for the exact mechanism. Closest prior art does learned recovery (VeriGUI), falls back to a base agent (trace-reuse, DeltaReplay), or replans — no source found that masks the verified no-effect option and re-argmaxes the same distribution. Caveat: `DecoderFreePolicies` owns the exhaustive search for this.
- **T-C** (offline-valid confidence does not flag no-effect actions online; environment-verified labels restore a usable signal, target AUROC ≥ 0.70): **DUPLICATES the premise and the mechanism; CONFIRMS-NOVEL only for the exact per-action no-effect instantiation.** "Per-step conformal calibration does not transfer" is now a theorem + EMNLP 2026 paper with numbers; "overconfidence under UI shift" is measured; "verified labels fix confidence" is done three ways (episode outcome, execution-feedback hindsight, no-change verification). The exact claim "a *no-effect* gate fitted on env-verified changed/no-change labels reaches AUROC ≥ 0.70 on a held-out task split" is absent.
- **Claim 1** (coverage–risk curve / selective-risk certificate / ask-for-help policy with numbers on AndroidWorld, AndroidLab, MobileWorld, OSWorld, AitW, AndroidControl): **DUPLICATES for ask-for-help policies with numbers** (OS-Kairos, Mobile-Aptus, MID-Bench, InquireMobile, RecAgent) **and for coverage–risk on grounding benchmarks** (PACE, SafeGround, Argus, HyperClick, CEB). **CONFIRMS-NOVEL for a coverage–risk curve or certificate on AndroidWorld.**
- **Claim 2** (confidence from environment-verified labels rather than model probabilities): **DUPLICATES**. CEB fits a step-level critic from hindsight execution-feedback labels; Outcome-Grounded fits verbalized confidence against realized episode outcome; VeriGUI trains a `SUCCESS/NO_CHANGE` verification head on deterministic screen-unchanged pairs. None of them reports a per-action *no-effect* gate AUROC.
- **(a) exact claim "offline-valid confidence does not transfer to online no-effect actions"**: **CONFIRMS-NOVEL as a sentence, DUPLICATES at the level of the finding.** No source states the no-effect version; `2026.emnlp-main.1808` states and proves the general trajectory version.
- **(b) does anyone use verified-effect labels to fit the gate?**: **Partially yes → DUPLICATES.** CEB's critic is a gate and its labels come from post-execution consequences, but the labels are LLM-judge hindsight votes, not a deterministic changed/no-change diff. VeriGUI's deterministic no-change labels train a verification head, not a gate.
- **(c) published coverage–risk number on AndroidWorld?**: **CONFIRMS-NOVEL for a curve/certificate.** VeriCUA publishes an abstention/decision-rate number on AndroidWorld (82% decided, 95.1% decided accuracy, FPR 7.7%, Table 1) but for a post-hoc completion verifier, not a coverage–risk curve and not the agent's own gate. CORA's risk–coverage sweep is on Phone-Harm/MobileRisk, not AndroidWorld; its AndroidWorld table is success rate only. RecAgent reports AndroidWorld SR with a human-in-the-loop module but no coverage axis.
- **Deployment-policy threshold (item 4)**: **CONFIRMS-NOVEL**. Apple HIG (v26, 2026-09-29) says to "set a confidence threshold below which you don't offer results" but gives no number. Android's 2026 Computer Control doc has "hand over control" but no numeric threshold. No new 2026 numeric platform threshold found.

## 1. Sources read

| id | title | venue + year | URL | citations (visible) | what it is |
|---|---|---|---|---|---|
| S1 | Per-Step Conformal Calibration Does Not Transfer to Agent Trajectories | EMNLP 2026 main, pp. 41197–41215 | https://aclanthology.org/2026.emnlp-main.1808.pdf | not shown | Theorem + empirics: per-step conformal coverage collapses over trajectories |
| S2 | Mobile-Aptus: Confidence-Driven Proactive and Robust Interaction in MLLM-based Mobile-Using Agents | arXiv 2605.28629 (2026-05-27) | https://arxiv.org/abs/2605.28629 | not shown | Trained confidence + threshold γ ask-policy; intervention precision/HSR/AIF; OOD UI experiment |
| S3 | OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents | Findings of ACL 2025 | https://aclanthology.org/2025.findings-acl.348.pdf | 6 (Liner) | Per-step confidence via "collaborative probing"; act vs ask threshold; AITZ, Meta-GUI |
| S4 | SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration | ICML 2026 (poster) | https://arxiv.org/abs/2602.02419 | not shown | K-sample spatial-dispersion UQ + Clopper–Pearson/LTT FDR-controlled selective grounding |
| S5 | Uncertainty Quantification for Computer-Use Agents (Argus) | NeurIPS 2026 D&B poster | https://arxiv.org/abs/2606.25760 | not shown | 27-method UQ benchmark, 4 models × 4 GUI datasets; conformal click disks |
| S6 | How Many Pixels Is a Digit Worth? Place-Aware Coordinate Entropy (PACE) | EMNLP 2026 main, pp. 10302–10315 | https://aclanthology.org/2026.emnlp-main.448.pdf | not shown | Single-pass coordinate-digit entropy; AUROC/AUARC + selective accuracy vs coverage |
| S7 | Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction (VeriGUI) | ACL 2026 long (2026.acl-long.1335) | https://aclanthology.org/2026.acl-long.1335.pdf | not shown | `SUCCESS`/`NO_CHANGE` verification head trained on deterministic unchanged-screen pairs; Loop Rate/RSR |
| S8 | Can an Agent Learn to Predict Its Own Failure? Outcome-Grounded Confidence Calibration | arXiv 2610.18614 (2026-10-13, SAP) | https://arxiv.org/abs/2610.18614 | not shown | Verbalized confidence calibrated against episode outcome under GRPO; AUROC/ECE + risk–coverage |
| S9 | Critic Experience Bank: Self-Evolving Step-Level Confidence Estimation for LLM Agents (CEB) | arXiv 2607.12397 (2026) | https://arxiv.org/abs/2607.12397 | not shown | Training-free critic gated by hindsight execution-feedback pseudo-labels; ECE/AUC + risk–coverage |
| S10 | Why Act Without Asking? Evaluating Mobile GUI Agents under Ambiguous Instructions (MID-Bench) | EMNLP 2026 main, pp. 28189–28210 | https://aclanthology.org/2026.emnlp-main.1236.pdf | not shown | Ask-vs-act gates (verbalized / semantic entropy / logit) on AndroidControl + GUI-Odyssey items |
| S11 | InquireMobile: Teaching VLM-based Mobile Agent to Request Human Assistance via RFT | ACL 2026 long, pp. 32241–32255 | https://aclanthology.org/2026.acl-long.1487.pdf | not shown | Trained inquire policy; InquireBench; ISR/SR |
| S12 | Uncertainty-Aware GUI Agent: Adaptive Perception through Component Recommendation and Human-in-the-Loop Refinement (RecAgent) | arXiv 2508.04025; ACL ARR 2026 Jan submission | https://arxiv.org/abs/2508.04025 | 17 (Scholar Inbox) | Decision-uncertainty → user query; AndroidWorld SR |
| S13 | VeriCUA: Verification as Agentic Exploration for Computer-Using Agents | arXiv 2610.40578 (2026-10-29, Google/HKU) | https://arxiv.org/abs/2610.40578 | not shown | Post-hoc verifier with `success/fail/abstain`; AndroidWorld/OSWorld/ScienceBoard |
| S14 | Last Step Matters: Early Uncertainty Cannot Predict Failure in Long-Horizon Agents | EMNLP 2026 main, pp. 10913–10931 | https://aclanthology.org/2026.emnlp-main.475.pdf | not shown | Negative result: early/mid-trajectory uncertainty AUROC < 0.60 |
| S15 | Testing-Driven Reliability Audit of Trajectory-Based Early Outcome Prediction for LLM Agents | arXiv 2609.25647 (2026-09-22) | https://arxiv.org/abs/2609.25647 | 1 (Scholar Inbox) | Leave-one-agent-out calibration-transfer audit, SWE-bench |
| S16 | HyperClick: Advancing Reliable GUI Grounding via Uncertainty Calibration | arXiv 2510.27266 (2025-10-31; ACL ARR 2026) | https://arxiv.org/abs/2510.27266 | not shown | Brier-calibrated verbalized confidence via truncated-Gaussian reward; AP at confidence thresholds |
| S17 | Structured Uncertainty guided Clarification for LLM Agents (SAGE) | Findings of ACL 2026, pp. 40811–40838 | https://aclanthology.org/2026.findings-acl.2028.pdf | not shown | Non-GUI tool-calling; EVPI question selection; coverage + question count |
| S18 | The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents | ACL 2026 long, pp. 11325–11349 | https://aclanthology.org/2026.acl-long.520.pdf | not shown | Non-GUI: verification tools ground confidence, evidence tools induce overconfidence |
| S19 | Thesis Proposal: When Does an Agent Know It Is Lost? | ACL 2026 SRW, pp. 877–887 | https://aclanthology.org/2026.acl-srw.78.pdf | not shown | Thesis (tool-use, no results tables) |
| S20 | Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control | arXiv 2610.31712 (2026-10-22) | https://arxiv.org/abs/2610.31712 | not shown | Deterministic pre/post-action evidence checks; reject-and-fallback on AndroidWorld |
| S21 | Apple Human Interface Guidelines — Machine learning | Apple HIG v26, change date 2026-09-29 | https://developer.apple.com/design/human-interface-guidelines/machine-learning | n/a | Platform design policy on confidence |
| S22 | Android Computer Control | Android Developers, 2026 preview | https://developer.android.com/ai/computer-control | n/a | Platform policy on assistant hand-over |
| S23 | AmbiBench: Benchmarking Mobile GUI Agents Beyond One-Shot Instructions in the Wild | ISSTA 2026 | https://arxiv.org/abs/2602.11750 | not shown | Mobile ambiguity benchmark; interaction quality |
| S24 | CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation | arXiv 2604.09155 (2026-04-10) | https://arxiv.org/abs/2604.09155 | 0 (arXiv.gg) | Known to repo; read only to bound what its AndroidWorld numbers are |
| S25 | Who Verifies the Agent's Success Claim? Self-Report Leakage in GUI-Agent Completion Judging | arXiv 2610.29609 (2026-10-21) | https://arxiv.org/abs/2610.29609 | not shown | Controlled aware/blind audit: agent self-report biases judge recall; report-isolated auditor |

## 2. Findings

### S1 — Per-Step Conformal Calibration Does Not Transfer to Agent Trajectories (EMNLP 2026)
This is the single most dangerous source for T-C. Claim, VERIFIED from the paper's own abstract and Table 1:

- Nominal miscoverage ε = 0.10, per-step split conformal: **29.5%** trajectory miscoverage at T=3 (HotpotQA), **36.4%** (MuSiQue, T=4), **43.1%** (FinanceBench, T=5), **60.9%** (GAIA-L1, T=8), **83.5%** (BrowserGym-WorkArena-L2 web-navigation, T=15); mean **50.7%** (Table 1).
- Theoretical lower bounds in three regimes: i.i.d. resampling (R1), memoryless covariate shift, and state-dependent Markov transitions (R3). The i.i.d. regime "accounts for the bulk of the empirical gap"; the R3 bound is claimed as the first separation of the trajectory-union event from i.i.d. resampling under marginal-stationary non-trivial Markov dynamics. The R3 lower bound at T=15 is **82.70%**, R1 **79.41%** (Table 1).
- A polynomial α-spending wrapper restores coverage to **8.9–9.7%** across the five benchmarks (target 10%), within 1.1 pp (Table 1).
- Process-reward-model step verifiers, calibrated per step at their empirical (1−ε)-quantile, produce trajectory miscoverage **31.5–62.5%**; substituting the wrapped score gives **9.6–10.6%** (Table 2).
- Scope: five agentic-NLP benchmarks, no GUI, no AndroidWorld, no AndroidControl. Metric is conformal coverage, not AUROC, and not "no-effect actions".

What it does to us: the sentence "a threshold calibrated on single steps does not retain its guarantee over an agent trajectory" is published, proved, and quantified. We cannot present trajectory non-transfer as a finding. Our residual novelty is narrower: (i) the *specific* failure mode is idempotent/no-effect actions, not generic compounding; (ii) we report *discrimination* (AUROC) of a gate, not coverage of a prediction set; (iii) environment-verified labels, not an α-spending schedule, are the fix.

### S2 — Mobile-Aptus (arXiv 2605.28629, 2026)
The most complete *ask-policy with numbers* on mobile. VERIFIED:

- Formalism: model emits `(action, confidence)`; if `c ≤ γ`, request human intervention; else execute (§III, eq. 2–3). Confidence is a 1–5 verbalized score.
- Stage 1: SFT on confidence-annotated data (the OS-Kairos dataset). Stage 2: "confidence bias correction" — semantic-similarity retrieval + DPO on pairs where the probed agent's confidence was bad (§IV-B).
- Main results vs OS-Kairos baseline (Table VII): OS-Kairos dataset IP **70.75 → 84.83%**, HSR **86.87 → 88.62%**, AIF **1.59 → 1.06**, SR **95.90 → 88.62%**; AITZ IP **71.64 → 73.27%**, AIF **5.78 → 5.42**, SR **87.54 → 85.82%**; Meta-GUI IP 93.18 → 93.25, AIF 4.28 → 4.28; AndroidControl IP 77.77 → 77.96, AIF 2.93 → 2.75, SR 79.78 → 75.81. (IP = intervention precision, HSR = human-intervention success rate, AIF = average intervention frequency.)
- Offline: "average improvement over 17% in task success rate"; real-device: +26% TSR with **0.64** intervention steps/instruction (Table VIII: OS-Kairos AIF 0.96 vs Mobile-Aptus 0.64; RE 86.42 → 93.09%; TSR 70 → 62%).
- Trade-off section V-C9: threshold γ=5 forces intervention every step, gives 100% SR but poor HSR; they explicitly argue the right operating metrics for an interactive agent are intervention proportion and HSR, not SR.
- OOD experiment V-C10: both trained on OS-Kairos, evaluated on AITZ; OS-Kairos AIF **6.42**, Mobile-Aptus **5.18** — "OS-Kairos exhibits an extremely high AIF due to overconfidence". This is the closest published statement to "offline/target-split confidence over-fires on a new UI", and it is *confidence trained on one benchmark* → *new benchmark*, not no-effect actions.
- Harness: OS-Kairos dataset, AITZ, Meta-GUI, AndroidControl; base OS-Atlas-Pro-7B; no AndroidWorld, no step budget stated for the offline tables, 10-step cap in the real-device run.

### S3 — OS-Kairos (Findings ACL 2025)
- Per-step confidence, act if `score ≥ γ`, ask human if `< γ` (§4.1, eq. 4). Confidence labels come from "collaborative probing": a layout parser + GPT-4o critic annotate the probed agent's per-step capability boundary; **not** environment verification.
- Metrics: Type, SR, TSR, plus HSR, intervention precision (IP), autonomous precision (AP), relative efficiency (RE) — i.e. it reports precision of the ask decision at a threshold.
- Headline: **24.59%–87.29%** TSR improvements over OS-Atlas-Pro-7B across the curated complex-scenario set, AITZ and Meta-GUI; curated set TSR **88.20%**, SR 95.90%, Type 99.88% (§5.2). Fine-tuned table (Table 3): overall TSR **87.71%** (zero-shot) and **87.29%** absolute gain vs ZS on one benchmark cell.
- No AUROC, no risk–coverage curve, no AndroidWorld/AndroidControl. It is a threshold-swept ask policy with precision-style metrics.

### S4 — SafeGround (ICML 2026)
The strongest *certified* selective-prediction number on a GUI task, but on grounding, not agents.

- Task: single-click grounding on ScreenSpot-Pro (1,581 screenshots). UQ: K=10 stochastic forward passes, spatial density → three scores → fixed-weight `U_COM` (0.6/0.2/0.2).
- Calibration: Clopper–Pearson upper bound on FDR, threshold `τ̂ = sup{τ : FDR_upper(τ) ≤ α}`; δ=0.05; 100 random calibration/test splits; test ratio 0.8.
- AUROC (Table 2): Holo1.5-3B **0.8056** (PC 0.7576), Holo1.5-7B **0.7526** (0.6983), GUI-Actor-2.5VL-7B **0.7793** (PC n/a), UI-TARS-1.5-7B **0.8021** (0.7844), GUI-Actor-2VL-7B **0.8155**, GTA1-7B **0.6344** (0.6114).
- AUARC (Table 3): up to **0.7166** (GUI-Actor-2VL-7B), 0.6705 (Holo1.5-7B), 0.6576 (Holo1.5-3B).
- FDR control (Fig. 3, text): empirical test FDR stays below the target for all risk levels tested; feasible α starts at **0.34** for these models because "very stringent FDR requirements may be infeasible" — no threshold can meet them. Cascading accuracy (Table 1): Holo1.5-7B 58.66% at α=0.34 vs Gemini-only 53.28%.
- Labels: ground-truth bounding regions (offline annotation), **not** environment verification. No AndroidWorld.

### S5 — Argus UQ benchmark (NeurIPS 2026 D&B)
- 27 methods / 7 families, 4 open-weight models × 4 datasets (ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G, UI-Vision-EG) + 8 methods × 3 API vendors.
- Main finding: "selective transfer". Fixed model across datasets: Spearman ρ up to **0.969**, mean **0.705** over 120 pairs. Cross-tier (open→closed): mean ρ **+0.08** over 12 vendor×dataset pairs, 95% CI [−0.219, +0.373] — indistinguishable from no transfer.
- Conformal click disks: locally weighted disks shrink radii **40–60%** when the plug-in UQ is calibrated, but "coverage can degrade under calibration-test or interface mismatch". Concrete undercoverage: Gemini × ScreenSpot-Pro Disk-Fixed coverage **0.926 / 0.884 / 0.795** for targets **0.95 / 0.90 / 0.80** (i.e. 2.4 / 1.6 / 0.5 pp under). Disk radii at α=0.10: SS-Pro 833–1547 px, SS-v2 132–269 px, OSWorld-G in between.
- No AndroidWorld / AndroidLab / MobileWorld. This is the published "coverage does not survive interface shift" result on GUI grounding.

### S6 — PACE (EMNLP 2026)
- Single-forward-pass confidence from place-value-weighted coordinate-digit entropy; no sampling, no training.
- AUROC (Table 1): the paper states PACE has the highest AUROC/AUARC on all three primary agents; the strongest single-pass baseline row I extracted is Coord mean top-1 prob 0.837 (EvoCUA-8B). On MAI-UI-8B the logit-mean confidence is 0.939 while accuracy is 0.494 (gap 0.445) — a concrete statement that response-level model probability is miscalibrated on GUI.
- Selective accuracy at coverage (Table 6): at **5% coverage**, mean selective accuracy **63.8% → 73.2%** (PACE vs uniform coordinate entropy); advantage shrinks with coverage.
- Task: ScreenSpot-Pro / ScreenSpot-v2 single clicks. Already known to the repo; included for the coverage–accuracy numbers and the miscalibration gap.

### S7 — VeriGUI (ACL 2026 long / arXiv 2604.05477)
- The `V_t ∈ {SUCCESS, NO_CHANGE}` head is trained on deterministic "screen unchanged" pairs (Type B synthetic failures, §3.3) plus GRPO with asymmetric verification rewards (Eq. 5: +1.0 correct, −0.5 miss, −2.0 hallucination). This is the closest thing to "verified effect as a training label".
- It is **not** a confidence/coverage gate: no AUROC, no coverage, no calibrated threshold. Outputs are Loop Rate and Recovery Success Rate on a synthetic failure-injection benchmark (Table 2): VeriGUI-3B LR **24.3%**, RSR **51.1%**; VeriGUI-7B LR **15.6%**, RSR **52.5%**; UI-TARS-7B LR 13.4%, RSR 45.5%.
- Online AndroidWorld (Table 3): VeriGUI-7B **25.1%**, VeriGUI-3B **12.6%**; step budget not stated. MiniWoB++ 59.7 / 35.6.
- Also useful as a citation for prevalence: "execution timeouts caused by repeated ineffective actions account for **72.3%** of all failures across 1,265 task executions (Liu et al., 2026)" and AndroidLab's Reasonable Operation Ratio — both cited, not reproduced by me (UNVERIFIED secondary).

### S8 — Outcome-Grounded Confidence Calibration (arXiv 2610.18614, SAP)
Direct prior art for "fit confidence on environment-verified outcome labels".

- Target: `c_t ≈ P(Y=1 | h_t)` where Y is the *episode* outcome label supplied by the environment; confidence is a verbalized `<conf>NN</conf>` token trained with a proper scoring rule under GRPO (Brier default, Eq. 4). Labels come from the rollouts' realized success — verified, but per episode, not per no-effect action.
- Offline pooled AUROC (Table 1): ALFWorld 0.517 → **0.681**, TextWorld 0.578 → **0.722**, WebShop 0.462 → **0.850**; top-10% step success lift 1.1–1.6× → ~**4.3×**.
- Online ALFWorld (Table 4): early-window (k≤2) pooled AUROC +0.048 to +0.101; ECE **0.419 → 0.053–0.115**; Brier 0.250 → 0.044–0.054; top-10% early-step success 0.493 → 0.56–0.66. Risk–coverage curves in Fig. 3 (no table numbers in the text).
- Important honest note in the paper: most of the ECE/Brier drop is the agent learning the base rate; "over all steps the most-confident bin is overconfident", so a controller "should rely on a band of high confidence rather than the very highest values". The gain is between-task (which tasks will fail), not within-task (which attempt).
- No GUI, no AndroidWorld.

### S9 — CEB (arXiv 2607.12397)
Closest mechanism to "verified effect as a label for the gate", and it reports a risk–coverage table.

- Training-free LLM critic; after each trajectory a hindsight LLM votes whether each step was productive, given the full action–observation sequence. Those pseudo-labels populate a memory bank retrieved at scoring time. The paper says it "uses no ground truth step labels" and bootstraps "directly from execution feedback".
- Datasets: Mind2Web (1,009 web trajectories), AMEX (1,000 mobile-GUI episodes), InterCode-Bash (200).
- Calibration: best ECE/Brier/AUC in all 9 dataset×critic cells; **ECE reduced by up to 54%** vs strongest training-free baseline; closes up to **95%** of the gap to a ground-truth oracle.
- Risk–coverage (Table 4, Mind2Web, GPT-5.4 critic; retained GT-positive rate at abstention budget b): at b=0/10/25/50%, Perplexity 0.198/0.178/0.154/0.124, Mean Entropy 0.198/0.175/0.145/0.114, Verbalization 0.198/0.198/0.201/0.219, CoT 0.198/0.205/0.222/0.234, **CEB 0.198/0.212/0.233/0.288**.
- The negative half is as useful as the positive: "Mean Entropy and Perplexity are actively misaligned: retained accuracy drops as their lowest confidence steps are removed (0.198 → 0.114 for ME at b=50%)". That is a published statement that model-probability confidence ranks agent steps wrongly.
- Caveat: the "verified" labels are LLM-judge hindsight votes, not a deterministic screen diff. Mobile GUI appears only as AMEX (ECE/AUC), not as risk–coverage.

### S10 — MID-Bench / Why Act Without Asking (EMNLP 2026, Samsung)
The cleanest ask-vs-act gate study with explicit threshold discipline.

- Three gates over a frozen base agent: PGC (verbalized yes/no), SGC (semantic entropy over k=5 stochastic decodes, τ=0), LGC (mean token log-prob below τ). Gate thresholds chosen by maximizing should-ask F1 on a held-out validation split, then "transferred unchanged to evaluation".
- Motivating numbers (Tab. 1): on an empty instruction, ask rates are 2.4% (Qwen3-VL-4B), 0.7% (GUI-Owl-1.5-4B), 12.3% (UI-TARS-1.5-7B), 0.0% (UI-Venus-Navi-7B); average clarification rate under ambiguity **0.1%**. Recognition vs ask gap: GUI-Owl recognizes 43.8% of ambiguous tasks but asks 0% — "ambiguity signals fail to propagate into action selection".
- Gate quality: LGC gets the highest F1 on Qwen3-VL-4B/GUI-Owl; on UI-TARS all three converge within ±.008 F1; SGC collapses on recall for UI-Venus. Oracle-gate ceiling on Hard Ask+Match is **.48–.61**, not 1.0, because of question quality and re-acting.
- Cross-dataset threshold transfer: re-tuning τ on AndroidControl and evaluating on GUI-Odyssey (and vice versa) gives Ask+Match penalties of **0** (Qwen3-VL-4B, GUI-Owl), **.037** (UI-TARS), **.013** (UI-Venus) — i.e. the threshold *does* transfer across those two offline sets.
- Harness: step-level offline evaluation on AndroidControl/GUI-Odyssey-derived items (n_amb=900, n_clear=300), not AndroidWorld. No coverage–risk curve; it reports Ask+Match, clear-set accuracy, gate precision/recall/F1.

### S11 — InquireMobile (ACL 2026)
- InquireBench: 5 categories / 22 sub-categories for "should I ask?"; the paper says most VLM agents are near-zero. Two-stage RFT (SFT + GRPO), pre-action reasoning.
- Results (Table 2): InquireMobile ISR **52.6%**, SR **7.9%**, score 0.78 on average; best baseline ISR 5.8%. Stage-1-only has higher ISR (27.9%) but much lower SR — i.e. over-asking.
- AndroidControl (Table 3): TM 91.0 / EM 81.2 on Low, 71.9 / 56.3 on High — they check that adding inquiry does not destroy ordinary execution.
- No threshold, no coverage, no AUROC; it is a trained policy, not a calibrated gate.

### S12 — RecAgent / Uncertainty-Aware GUI Agent (arXiv 2508.04025)
- Distinguishes *perceptual* uncertainty (input redundancy, handled by component recommendation) from *decision* uncertainty (ambiguous task, handled by an interactive "ask the user" module). The decision-uncertainty trigger is not a numeric threshold on a model probability; it is an LLM-judged ambiguity check.
- AndroidWorld (Table 1): RecAgent (SoM, GPT-4o) **47.8%** SR; a11y-tree variant 43.5%; baselines e.g. GUI-explorer 47.4%, Mirage-O 42.2%. Ablation (Table 4): no CRM/no RM 40.5% → both 47.8%. Max 30 steps.
- No calibration metric, no coverage, no ask-precision. But it *is* an ask-for-help policy with a number on AndroidWorld.

### S13 — VeriCUA (arXiv 2610.40578, Google/HKU)
- Post-hoc completion verifier that can output `abstain`; it inspects GUI and code state after the episode. On AndroidWorld (Table 1, benchmark's own agent scaffold): VeriCUA precision 90.6, recall 87.9, F1 89.2, accuracy 90.5, FPR **7.7%**, decided **82%**, accuracy on decided **95.1%**. VLM judge: 79.3/69.7/74.2/78.4/15.4/100%/78.4%. Auto-generated verifier: 77.8/84.8/81.2/82.4/16.9/100%/82.4%.
- This is the only AndroidWorld abstention number I found. It is *not* a coverage–risk curve (no risk target, no sweep), and it verifies task completion, not the agent's next action. Episode count not stated in the Table 1 text I read.

### S14 — Last Step Matters (EMNLP 2026, ZJU/Ant Group)
- Deep-research tasks. Verbal confidence "reliably distinguishes failures at trajectory completion, achieving a mean AUROC of **0.85**, whereas all evaluated signals offer limited predictive value earlier in execution, with none exceeding a mean AUROC of **0.60** at 50% trajectory progress."
- Mechanism: "path switching" — agents abandon the current search direction mid-trajectory, breaking the link between early signal and final outcome. Recommendation: use final-step confidence to decide whether to restart, not in-trajectory intervention.
- Not GUI, but it is a published negative result about intermediate confidence, and it is a direct citation against "early step-level confidence can drive intervention".

### S15 — Testing-Driven Reliability Audit (arXiv 2609.25647)
- SWE-bench Verified, frozen dual-head early-outcome predictor, leave-one-agent-out. Broad same-predictor heterogeneity **not** supported (median pairwise corrected-gap differences 0.0180 SUCCESS, 0.0385 FAILURE). Two specific combinations show persistent transfer error: gpt-5-mini/SUCCESS **0.1377**, claude-opus-4.6/FAILURE **0.1107**. TerminalBench did not replicate.
- Takeaway: transfer failure is real but target-specific, not a universal property. Not GUI.

### S16 — HyperClick (arXiv 2510.27266)
- Trains verbalized confidence with a Brier reward against a truncated-Gaussian spatial target (Eq. 6), GRPO. Reports AP at confidence thresholds {0.5,0.75,0.9,0.95} on ScreenSpot-Pro (Table 2): 3B 70.6 / 76.0 / 78.0 / 78.0; 7B 61.3 / 64.6 / 71.2 / 78.7. Baselines e.g. UI-TARS-1.5 37.6 / 37.5 / 37.5 / 40.4.
- Note the metric is AP, not a coverage–risk curve; and it shows the highest-confidence bucket is not monotonically better for the 7B model (78.7 at 0.95 vs 71.2 at 0.90 is fine, but the gain is small). No AndroidWorld.

### S17 — SAGE / Structured Uncertainty (Findings ACL 2026)
- Non-GUI tool-calling. Uses EVPI over tool-parameter domains; "7–39% higher coverage on ambiguous tasks while reducing clarification questions by 1.5–2.7×"; When2Call accuracy 36.5% → 65.2% (3B), 36.7% → 62.9% (7B).
- Included only because it is the only paper I found that reports a *coverage* number for an ask policy. It is tool-calling, not GUI; do not put its numbers next to GUI numbers.

### S18 — Confidence Dichotomy (ACL 2026)
- Non-GUI (search/code tools). ECE relative reduction up to 68% with CAR; AUROC relative improvement up to 17% (Table 2). Core finding: deterministic *verification* tools ground confidence; noisy *evidence* tools induce overconfidence.
- Relevant as external support for the mechanism "deterministic feedback is what makes confidence trustworthy", but it is not a GUI result and not a coverage result.

### S19 — Thesis proposal (ACL SRW 2026)
- Tool-use, trajectory-level confidence; three aims (action-level UQ, action-type aggregation, temporal patterns). No results tables (proposal). Included as the small-venue/thesis null: even the thesis that owns this space has not published the no-effect result.

### S20 — Reuse/Rewrite/Reject (arXiv 2610.31712)
- AndroidWorld trace reuse with deterministic pre/post-action evidence checks; "rejected or failed steps falling back to the base agent". ~80% token reduction across five base agents; most agents ≥1.5× speedup; real-device action-reuse ratios **70.7–79.8%** with >2× speedup.
- Not a confidence gate; it is the closest *deployment* example of "verify the effect, else fall back".

### S21/S22 — Platform policy docs
- Apple HIG v26 (change date 2026-09-29), "Machine learning" page (read via the appstoreguidelines.com v26 mirror because developer.apple.com renders client-side): "For suggestions and proactive features, it's a good idea to set a confidence threshold below which you don't offer results"; also "When confidence is high … simply displays … but when confidence is lower, the feature asks people to confirm". **No numeric threshold anywhere on the page.**
- Android "Computer Control" (2026 preview): describes assistant task automation, permission dialog, and "Hand over control: The assistant can hand over control to the user for manual intervention or let the user manually request to take over. For example, this is useful for transaction confirmation". **No numeric threshold, no coverage/risk language.**
- These confirm the repo's value-brief null for 2026: platform docs give qualitative policy, not numbers.

### S23 — AmbiBench (ISSTA 2026)
- Mobile ambiguity benchmark with four clarity levels and MUSE interaction metrics. Numbers: AutoGLM TSR 65.2% on Detailed instructions; on Ambiguous, TSR for AutoGLM and MobileAgentV2 drops to **zero**; Fairy reaches 26.2% on Incomplete. Two failure modes at low clarity: conservative refusal (low ARR) vs blind high-ARR struggle.
- No calibration, no coverage, no threshold. Useful only to show the ambiguity/ask space is benchmarked outside arXiv-main.

### S24 — CORA (bounding check)
- Read only to bound the AndroidWorld claim. CORA's risk–coverage and safety–helpfulness–interruption frontiers are on **Phone-Harm** (Table 1/2) and **MobileRisk**; its AndroidWorld table is success rate only: core20 **8/20 = 40.0%** for CORA vs AutoGLM 30.0%, GPT-5 55.0% (Table 3). It explicitly frames "risk–coverage" as the right reporting protocol, but not on AndroidWorld.

### S25 — Self-Report Leakage in GUI-Agent Completion Judging (arXiv 2610.29609)
- Controlled aware/blind intervention: the same judge sees the same task and trace, with or without the agent's self-report. Withholding the self-report raises trace-only dangerous-misreport recall by **43.4 pp** (Qwen3.5-35B-A3B) and **13.3 pp** (DeepSeek-284B-A13B), same trend on held-out Batch 2. Report-isolated auditor reaches F1 **0.75–0.78** on the 465-record audit-target set and **0.77–0.81** on held-out Batch 2; under gating it blocks **74–90%** of dangerous misreports at **45–56%** inspection load (16–19% of the full stream after coarse triage).
- Not a confidence paper, and not a coverage–risk curve. Included because it is the empirical case for the repo's stance that self-reported/model-probability signals must be separated from environment evidence, and it is a 2026 GUI-agent result outside the arXiv set already surveyed.

## 3. What this changes for us

1. **T-C cannot be sold as a discovery.** The premise ("a per-step calibrated confidence/threshold does not carry to multi-step execution") is a proved theorem with 83.5% miscoverage at T=15 in EMNLP 2026 main (S1), plus a coverage-under-interface-mismatch result on GUI grounding (S5), plus an OOD overconfidence measurement (S2). We must cite S1 and S5 in the related work and frame T-C as: *the specific no-effect/idempotent sub-case, measured with AUROC on a gate, fixed by deterministic changed/no-change labels*. Anything broader is already taken.
2. **The fix is also partly taken.** CEB (S9) already fits a step-level gate from post-execution-consequence labels and reports a risk–coverage table; Outcome-Grounded (S8) already fits confidence on realized episode outcome and reports AUROC 0.68–0.85 / ECE 0.42→0.05–0.12; VeriGUI (S7) already trains on deterministic `NO_CHANGE` pairs. Our differentiator must be: *per-action* no-effect labels, *deterministic* (not LLM-judge) labels, *GUI*, and a held-out AUROC target, plus the masking policy in T-B.
3. **Claim 1 must be narrowed.** "First ask-for-help policy with numbers" is false (S2, S3, S10, S11, S12). "First coverage–risk curve on a GUI agent task" is false for grounding (S4, S5, S6) and for web (S9). What remains true: no coverage–risk curve, no selective-risk certificate, and no certified abstention policy on **AndroidWorld** (S13 gives a decision-rate number but not a curve; S24 gives SR only). State that precisely, benchmark by benchmark.
4. **Use S4/S5 as the calibration template to beat.** SafeGround gives the exact pattern we would use (Clopper–Pearson upper bound on FDR + LTT threshold, δ=0.05, 100 splits) and reports the "minimum attainable α" caveat we will also hit. Argus gives the transfer/coverage-mismatch caveat. If we publish a certificate, we must say which of these two failure modes we are exposing ourselves to.
5. **Cite the miscalibration evidence, not just the claim.** PACE (S6) shows logit-mean confidence 0.939 vs 0.494 accuracy on MAI-UI-8B (gap 0.445); CEB (S9) shows mean-entropy risk–coverage going *down* (0.198→0.114 at 50% abstention); S14 shows early AUROC <0.60. These are the numbers that make our problem real and are stronger than an abstract assertion.
6. **Deployment-policy null holds.** Apple HIG v26 and Android Computer Control give qualitative "set a threshold" / "hand over control" guidance, no numbers. Do not claim a platform-specified threshold; do claim the absence.

## 4. Searches run (queries + sources searched + nulls)

Web search (engine: default web_search, 8–10 results/query), plus direct HTML/PDF reads of every source listed in §1.

Exact queries:
1. `GUI agent abstention selective prediction coverage AndroidWorld`
2. `mobile agent confidence calibration ask for help AndroidLab benchmark`
3. `conformal prediction GUI agent computer use risk control`
4. `SafeGround GUI grounding uncertainty calibration risk coverage`
5. `"no-effect" action detection GUI agent screen change feedback`
6. `calibration fails distribution shift interactive agent confidence OOD selective risk`
7. `"AndroidWorld" abstain OR abstention OR "risk control" agent`
8. `GUI agent "ask for help" OR clarification request benchmark AndroidWorld`
9. `selective classification GUI agent "coverage" "risk" rejection abstain mobile`
10. `site:developer.android.com confidence threshold automation "abstain" OR "ask the user" numeric`
11. `Apple Human Interface Guidelines automation "confidence" threshold ask user before action 2026`
12. `DeltaReplay mobile GUI agent abstention Adapt AndroidWorld`
13. `"execution feedback" confidence calibration LLM agent "selective" OR "abstain" success prediction`
14. `GUI agent "success prediction" OR "outcome prediction" confidence verifier abstention 2026`
15. `MobileWorld benchmark agent-user interactive "ask" user clarification mobile agents`
16. `"calibrated on offline" fails online interactive agent confidence shift GUI 2026`
17. `"risk-coverage" OR "risk coverage curve" GUI agent mobile agent abstention`
18. `"Structured uncertainty guided clarification" LLM agents 2511.08798`
19. `GUI智能体 置信度 拒答 弃权 风险控制 2026` (non-English)
20. `"confidence" "does not transfer" OR "fails to transfer" agent offline online calibration gap`
21. `"Uncertainty-Aware GUI Agent" adaptive perception component recommendation Hao 2025`
22. `OS-Kairos GUI agent over-execution confidence SITU Meta arxiv`
23. `"selective execution" OR "selective prediction" "AndroidWorld" mobile agent gate threshold`
24. `conformal prediction "AndroidWorld" OR "AndroidControl" mobile GUI agent coverage guarantee`
25. `"risk-coverage" GUI agent OR mobile agent OR screen agent 2026`
26. `workshop thesis "GUI agent" uncertainty calibration abstention 2026`
27. `mobile GUI agent ambiguity clarification benchmark coverage 2026 ask user abstain numbers`
28. `"no change" OR "unchanged screen" confidence calibration agent action failure detection 2026`
29. `HyperClick reliable GUI grounding uncertainty calibration OpenReview`
30. `selective classification GUI agent "coverage" "risk" rejection abstain mobile`

Sources searched: arXiv (abs + HTML + PDF), ACL Anthology (main, Findings, SRW), PMLR, NeurIPS/ICML virtual sites, OpenReview, CEUR-WS, Apple developer docs, Android developer docs, GitHub (repo search results only), HuggingFace papers, plus aggregators (alphaXiv, Scholar Inbox, CatalyzeX) to locate venue/citation metadata.

Nulls (searched, nothing found):
- **No coverage–risk curve, AURC, or selective-risk certificate on AndroidWorld.** CORA reports coverage/risk only on Phone-Harm + MobileRisk; VeriCUA reports a decision rate (82%) + decided accuracy but no sweep/target; RecAgent and VeriGUI report success rate only.
- **No AndroidLab, MobileWorld or AitW coverage–risk number** in any source found. MobileWorld is agent-user-interactive and is the right harness for ask-policy work (and the paper exists: ACL 2026 long 278), but I found no calibration/coverage metric in it; its role here is a benchmark, not a calibration result.
- **No source that publishes the sentence "offline-valid confidence does not transfer to online *no-effect* actions."** The closest are S1 (trajectory non-transfer, coverage), S5 (interface-mismatch coverage), S2 (OOD overconfidence, AIF 6.42 vs 5.18).
- **No source that fits a selective gate on deterministic changed/no-change labels.** CEB is LLM-judge hindsight (not deterministic); VeriGUI's deterministic labels train a verification head, not a gate; S20 uses deterministic checks to accept/reject trace transitions, not to calibrate a confidence threshold.
- **No numeric abstention threshold in Apple/Google platform docs (2026).** Apple HIG v26 and Android Computer Control are qualitative.
- **No thesis/workshop paper with the result.** The only thesis found (ACL SRW 2026, S19) is a proposal with no results.
- **Non-English search (query 19)** surfaced only translations of the English papers already covered (OS-Kairos via Chinese tech press, GUI-AC, Faithful Mobile GUI Agents) — no independent calibration result.
