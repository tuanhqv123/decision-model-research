# What a calibrated act / ask / refuse gate is actually worth

*Research brief. 2026-11-07. Built from four evidence notes: T1 metrics and outcome evidence, T2 audience
requirements, T3 production failure modes, T4 internal audit. Source markers in brackets resolve in the
Sources section; parenthetical locators of the form (T1 §3.1) name the note section each claim was taken
from.*

---

## Executive summary

A calibrated single-pass decision model answers a fixed question with a probability instead of writing a
sentence. "Calibrated" means that when it says 0.8, it is right about 80% of the time. A gate turns that
probability into one of three actions: **act**, **ask the human**, or **refuse**.

The literature supports three claims and undercuts a fourth.

1. **Abstention does change real outcomes — but almost none of that evidence is from GUI agents.** The
   strongest result is a 31,301-woman paired noninferiority trial in which a machine gate removed 63.6% of
   radiologist reading, raised cancer detection by 15.2%, and **failed its own recall-safety criterion**
   (T1 §3.1) [1]. A peer-reviewed router shows the same shape in compute: 3.66× cost saving at 95% of the
   strong model's quality (T1 §3.3) [2]. In GUI agents there is **no peer-reviewed outcome-level abstention
   result at all** (T1 §6, §7).
2. **The gate is already platform policy; the calibration is the missing part.** Apple's HIG describes act /
   ask / suppress [3], and Google enumerates the action classes that must always ask [4]. Neither publishes a
   numeric confidence threshold, a measured ask rate, or an outcome measurement (T2 §A, §B, §E) [3, 4]. Every
   shipped gate found is a hand-authored rule keyed on action class, not a learned confidence threshold
   (T2 §E) [3, 4].
3. **A per-checkpoint certificate is a measurement, not a product property.** Every published guarantee
   requires the target distribution, a density ratio, a stationarity assumption, or a finite conditioning
   class; under shift the bound degrades by a total-variation term that cannot be estimated in production
   (T3 §0–§2) [5, 6]. Group-conditional coverage is provably impossible distribution-free (T3 §6) [7].

And the undercut: **our own program's headline does not reproduce as published.** The internal audit finds no
artifact anywhere for 64.32% (256/398); the two pure-protocol artifacts are 255/398 (in-process) and 254/398
(served), the 56.50% grounding belongs to the *step-705* checkpoint, the candidate-coverage claim of 100% is
96.9%, and the online invalid-action intervals overlap (Fisher p = 0.065) on **task instances that were not
paired** (T4 §1–§2) [8, 9, 11, 12].

The value question is therefore narrower and better than "is calibration interesting". It is:

> **At the same accuracy, does gating on confidence beat both trivial policies — always-act and always-ask —
> at a cost ratio a real product actually faces?**

On our artifacts the answer is yes, in a window. With a measured 36.2% error when acting always (1 − 0.6382
on the pure-protocol served artifact) and 17.3% selective risk at 40.7% coverage, the gate dominates
**always-act** once a wrong action costs more than **~2.0×** an interruption, and still beats **always-ask**
until a wrong action costs **~5.8×** one (T4 §1, rows 1 and 9) [8, 9]; the break-evens are arithmetic on those
two rows, shown in §7. That is a claim a reviewer can grade and a builder can price — and it is the one
contribution shape our existing artifacts can support without new training.

---

## 1. Four kinds of confidence claim, only one of which licenses a decision

The field uses the word "confidence" for four different objects, and they license four different decisions
(T1 §1).

| What is reported | What it licenses | What it does *not* license |
|---|---|---|
| ECE, Brier | "the probability is right on average" | any per-decision risk |
| AUROC(conf, correct), AURC | "confidence orders examples usefully" | any operating point; it is invariant to monotone rescaling, so it cannot detect miscalibration at all |
| **risk at a stated coverage**, conformal/CP certificate | **"at this coverage, error ≤ r with confidence 1−δ"** | anything about a coverage you did not report, or a distribution you did not measure |
| realised cost against a baseline | "acting on the probabilities was better than not acting" | — (almost nobody reports this) |

*Metric definitions and what each licenses: (T1 §1.2) [13, 14, 16]; the coverage-conditioned certificate row:
[6].*

Two structural results matter for framing. First, post-hoc calibration has *limited* impact on selective
performance, because it rarely changes the score ranking (T1 §1.3) [16]. Second, the choice of selective
metric is not decision-stable: AURC and AUGRC reorder the leaderboard on 5 of 6 datasets (T1 §1.3) [14].
Within GUI agents, the modal claim stops at step accuracy with no calibration metric at all (T1 §2.5) [15].

**Implication.** A reliability paper must report coverage-conditioned risk *and* name the estimator. "ECE
improved" is not a deployment claim, and an AUROC is not a certificate.

---

## 2. Does abstention change outcomes?

**Yes in adjacent fields, with effect sizes; no in GUI agents.** The distinction is the reason this program
has a defensible niche.

**Medical screening (peer-reviewed, the strongest result).** A paired noninferiority trial in mammography and
digital tomosynthesis ran standard double reading in parallel with AI-supported screening in which low-risk
cases were called normal and the rest double-read with AI support (T1 §3.1) [1]. Radiologist workload fell
63.6%; cancer detection rose 15.2% (95% CI 6.6–24.4); **recall rose 14.8% (95% CI 9.0–20.6), which failed the
pre-registered noninferiority criterion** [1]. The gate traded human time for sensitivity and lost the safety
margin it promised. That is the exact evidence shape our program needs — and the exact reason a bare AUROC
claim is not enough.

**Compute routing (peer-reviewed).** RouteLLM decides *before* any model is called whether a query goes to the
strong or weak model, and reports 3.66× cost saving at 95% of GPT-4 quality on MT-Bench, 1.41×/1.49× on MMLU
and GSM8K at their stated thresholds (T1 §3.3) [2]. This is the closest analogue to our value claim and it is
favourable: escalation gates pay when the cost asymmetry is large and the quality bar is explicit.

**Robotics (ICRA 2024).** A learned interaction policy that asks for help reports +0.38 task success at a 0.3
expert-interaction rate in Habitat with a simulated expert (T1 §3.5) [21]. Clean shape, simulated human.

**The cautionary result.** A clinical deferral strategy had **under 25% adherence** — the gate's benefit
depends on the person receiving the ask actually following it (T1 §3.2) [20]. Human-factors work shows the
same pressure from the other side: alert override rates of 49–96% [22], clickthrough of a tenth to a third
[24], and automation-bias commission errors in ~6% of cases where clinicians overrode their own correct
judgement (T2 §C.3) [23].

**The GUI gap.** No GUI benchmark reports selective risk as a headline; no GUI-agent paper prices the *ask*;
no paper measures calibration→outcome transfer for a mobile agent; no GUI paper reports threshold transfer
across checkpoints (T1 §6). The nearest artifacts are a grounding-level FDR-controlled threshold (SafeGround,
+5.38 pp over Gemini-only inference) [17], a preprint paired-task abstention benchmark on which 17 frontier
LLMs across four harnesses reach 59.5% paired accuracy [18], and a typed-decision benchmark with a
realised-cost experiment on non-GUI tickets [19] (T1 §2.2, §2.4, §2.7).

---

## 3. Who cares, and what each one needs

**A CCF-A reviewer** grades claim–evidence fit, not the leaderboard. TMLR's stated first criterion is whether
claims are supported by convincing evidence, and it explicitly forbids "not SOTA" and "not novel enough" as
reasons to reject; its remedy for an overclaim is to reduce the claim (T2 §A.1) [25]. ICML separates
soundness from impact (T2 §A.1) [26]; NeurIPS instructs reviewers not to penalise honesty about limitations
(T2 §A.1) [27]; UIST accepts threshold benchmarking as validation, with no user study required (T2 §A.1)
[28]. Precedent that this suffices: Guo et al.'s calibration paper [30] and SelectiveNet [29] are accepted
without an accuracy contribution (T2 §A.3).

What the reviewer will not accept: deployment readiness inferred from ranking metrics; a per-checkpoint
threshold presented as a product property; a certificate that does not re-derive from saved artifacts; and no
cost model (T2 §A.4) [25].

**A builder** needs one number: expected cost per 1,000 gate decisions at the product's own cost ratio,
including the measured ask rate and the dismissal rate (T2 §B.6). Apple's guidance is already stated this way
— compare the cost of correction with the benefit of the feature (T2 §B) [3] — and Google's always-ask list
defines where a gate is mandatory (T2 §B) [4]. But no first-party document gives a numeric confidence
threshold or a latency target; every shipped gate is rule-based (T2 §E) [3, 4].

**An end user** needs the harm avoided to be worse than the interruption. The concrete harms are legible:
message sent to the wrong recipient, wrong purchase, irrecoverable delete, an agent submitting a form on the
user's behalf (T2 §C.3) [4, 31, 32]. The cost side is measured and unflattering: override 49–96% [22],
habituation over days [T2 §C.2], and "are you sure?" producing reflex approval [32] (T2 §C.3–§C.4).
Confirmation is the wrong tool when undo would work; the reversibility rule appears in both platform guidance
and the design literature (T2 §C.3) [31, 32].

**The inference this brief commits to.** The contribution is not "we invented ask-before-acting". It is:
**a learned, calibrated threshold on a single-pass model, reported as risk at coverage with baselines, plus a
measured ask rate and a cost ratio.** That is the exact intersection of what reviewers grade [25] and what
builders price (T2 §B.6), and it is empty in the GUI literature (T2 §A.5, T1 §6).

---

## 4. What has to be true for the gate to be safe

No published method gives a distribution-free guarantee under arbitrary shift (T3 §0) [5]. The strongest
bounds degrade by a total-variation term that cannot be estimated from unlabelled deployment data (T3 §1)
[5, 6]. Three failure modes matter most here.

- **Checkpoint and population change (M1).** Our own transfer measurement — 17.3% risk at one checkpoint,
  23.9% at the next under the same threshold — is a certificate failure, not a demonstrated risk change
  (T4 §4.6) [8]. The audit also notes the certificate is **val-side**: the test-side Clopper–Pearson upper bound
  is 21.7%, so the 20% budget is not certified on test [8].
- **Clustered evaluation units (M2).** Rows are not exchangeable when they share episodes, users, or tasks;
  the effective sample size is the number of independent units, not rows (T3 §1) [34]. Our 398 test steps span
  350 episodes (count of distinct `episode_id` in the test artifact) [9], which is why the paired tests are
  episode-clustered [8].
- **Silent coverage decay (M6).** Coverage can fall from 0.92 to 0.61 while accuracy stays within three points
  of baseline (T3 §0) [33]. A one-shot certificate would not see it.

What a deployer can defensibly do is not a certificate but a *contract plus an instrument*: a named
invariance assumption, an anytime-valid coverage monitor with a stated false-alarm rate, and a re-fit trigger
tied to model updates rather than the calendar (T3 §5–§6) [5, 6, 33]. The honest sentence is: **"certified per
checkpoint, monitored continuously, invalidated by any distributional change we cannot verify we did not
have."**

---

## 5. What this program actually has (after the audit)

The audit re-derived every headline number from disk (T4). The calibration half is solid; the accuracy half
needed restating.

| Claim | Published | Re-derived | Status |
|---|---|---|---|
| AC-v2 pure Pass@1 | 64.32% (256/398) | 64.07% (255/398) in-process; 63.82% served | **cannot reproduce** → restated |
| groundB Pass@1 | 64.82% (258/398) | 258/398 | reproduces |
| Type accuracy | 84.17% | 335/398 | reproduces |
| Grounding | 56.50% (126/223) | 56.05% pure / 57.40% hinted; **56.50 is step 705** | reproduces, wrong checkpoint |
| Type-routed gate AUROC | 0.760 / 0.729 / 0.709 | identical | reproduces |
| Certified 20% risk @ 40.7% coverage | as stated | val-side only; test CP upper 21.7% | reproduces with caveat |
| Learned-fusion risk@50% | 19.1% | 38/199 = 0.191, 95% ±5.5 pp | reproduces, underpowered |
| Candidate coverage | 1.0000 | 216/223 = 0.9686 | **contradicted** |
| Online invalid actions | 28.2% → 14.7%, "non-overlapping" | intervals overlap 19.0–25.0; Fisher p = 0.065; instances unmatched | **contradicted** |
| Online SR | 1/8 both | 1/8 both, Wilson 2.2–47.1 | reproduces, underpowered |
| Online "25% environment crash" | 2/8 per condition | two Clock-task **checker exceptions on a live device**; last recorded step `error: null`, `action_ok: true`; the next episode in the same suite ran ten steps immediately after; stderr repeats `Could not get a11y tree` | **misclassified by the harness, now fixed** [35] |

*Rows 1–8: (T4 §1, §2) [8, 9]; candidate coverage: [12]; online rows: [11]; the crash reclassification: [35].*

Everything in `benchmarks/results/2026-11-03-stage0/` re-derives byte-identically and `verify_paper.py` passes
69/69 — but that script checks stage-0 derived JSON only, not the pure-protocol row, the baselines, the pixel
distances, or any online number (T4 §2 R8) [8, 10].

**What this means for the value claim.** The calibration result survives audit; the headline accuracy is a
one-row drift from a tie; the online half is a harness finding, not a capability result. Nothing here
invalidates the gate argument, and published sentences had to be withdrawn (T4 §2, R1–R9) [8].

---

## 6. The pre-registered value test

The plan committed to delivering only if at least two of five conditions held. They are scored here.

1. **A peer-reviewed outcome-level result where abstention changes success or harm — HOLDS.** The
   mammography trial [1] and RouteLLM [2] (T1 §3.1, §3.3).
2. **A named, concrete user harm an act/ask/refuse gate prevents, with the cost of asking stated — HOLDS.**
   Wrong recipient, wrong purchase, destructive delete, agent-submitted forms [4, 31, 32], each with measured
   override and habituation costs [22, 23] (T2 §C.3).
3. **A reviewer-legible contribution shape our existing artifacts support without new training — HOLDS, with
   a condition.** The shape is "risk at coverage with trivial baselines and a cost ratio" (T2 §A.5) [25]; the
   artifacts support the risk and the baselines [8], and the cost ratio is a parameter we can sweep. What they
   do not yet support is *transfer across checkpoints* — the 17.3% → 23.9% failure is real [8].
4. **One next measurement with a predicted effect size, a cost, and a kill criterion — HOLDS.** §7.
5. **A documented reason the current headline cannot support a CCF-A claim yet — HOLDS.** No GUI outcome
   result exists (T1 §6), the accuracy is a tie [8], the certificate is per-checkpoint (T3 §6) [7], and there
   is no cost model or ask rate in any GUI paper (T1 §6).

**Verdict: the brief passes (5/5).** The honest summary is that the *value* case is strong, the *outcome
evidence* for GUI agents does not yet exist, and the program's own contribution is a measurement plus a
certificate boundary, not a capability claim.

---

## 7. The one next measurement

**Report the first GUI selective-risk table with explicit cost ratios and both trivial baselines.**

Take the artifacts we already have — 398 test rows and 827 validation rows (T4 §1, row 25) [9], with the
type-routed gate and the learned fusion [8] — and report, for each policy, error per 1,000 decisions as a
function of the ratio `C_err / C_ask`:

- **always-act**: `1000 · (1 − acc) · C_err`
- **always-ask**: `1000 · C_ask`
- **gate at coverage c**: `1000 · (c · risk(c) · C_err + (1 − c) · C_ask)`

With the measured numbers (always-act error 36.2% [9]; gate at 40.7% coverage, empirical risk 17.3%; test-side
CP upper 21.7% [8]), the gate beats always-act when `C_err/C_ask > 2.03` (2.17 using the upper bound) and
beats always-ask when `C_err/C_ask < 5.78`. The learned fusion at 50% coverage and 19.1% risk [8] moves the
lower break-even to ~1.9. *(All four break-evens are this brief's arithmetic on the cited rows; the formulas
above are the whole derivation.)*

**Why this one.** It is the number T2 §A.5 says a reviewer will grade and T2 §B.6 says a builder will price;
it needs no GPU; and it converts a 64% tie into a decision-level statement, which is the only kind of claim an
accuracy tie can support.

**Prediction.** The gate dominates both trivial policies over a roughly 2–6× cost window on the current
checkpoint, and that window narrows to nothing on the second checkpoint if the transfer failure is
representative [8].

**Kill criterion.** If the gate does not dominate always-act at `C_err/C_ask ≥ 2` on *both* checkpoints, the
value claim fails for this checkpoint pair and the paper must say so — the certificate, not the cost argument,
is then the contribution.

**Cost.** Hours, CPU only, no model calls.

---

## 8. Open questions

1. **What is the real cost of an ask?** The literature measures override and habituation, not the price of an
   interruption in the product's own utility. The 40% maximum override rate is proposed without derivation
   (T2 §E) [22].
2. **Does the cost argument survive a second checkpoint?** Currently unmeasured; our single transfer pair
   suggests it may not [8].
3. **Is there any GUI agent for which an outcome-level abstention result exists?** None found; the claim is
   "not found", not "does not exist" (T1 §6).
4. **Should the gate be per-action-type?** An irreversible tap and a scroll have different costs; a single
   threshold cannot express that, and no published gate does it (T1 §6 item 8).
5. **What would make us stop?** If the cost-window table is dominated by always-act at realistic ratios, the
   program should pivot from gating to the certificate boundary as the contribution — which is publishable
   and honest, and smaller.

---

## Sources

Numbered sources follow the order of first citation in the brief. Entries **8–12 are internal repository
artifacts, not web sources**; all others are live URLs fetched during the citation pass.

1. *AI-based triage and decision support in mammography and digital tomosynthesis for breast cancer screening: a paired, noninferiority trial* — Nature Medicine, 2026. https://www.nature.com/articles/s41591-026-04277-x — live; headline numbers read on the publisher page (n = 31,301; workload −63.6%; CDR +15.2%, 95% CI 6.6–24.4; RR +14.8%, 95% CI 9.0–20.6; "RR was not noninferior").
2. Ong, I., Almahairi, A. et al. — *RouteLLM: Learning LLMs to Route from Preference Data*, ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/5503a7c69d48a2f86fc00b3dc09de686-Paper-Conference.pdf — live; Table 6 read directly (3.66× @95% GPT-4 quality on MT Bench; 1.41× @92% on MMLU; 1.49× @87% on GSM8K).
3. Apple — *Human Interface Guidelines: Machine Learning* (Confidence section). https://developer.apple.com/design/human-interface-guidelines/machine-learning — live; "confidence thresholds", "verify that your confidence values correspond to the quality of your results", and the cost-of-correction guidance all read on the page.
4. Google — *Manage your tasks & workflows in Gemini Apps*. https://support.google.com/gemini/answer/16596215 — live; always-ask list (sending communications, deleting data, purchases, financial transactions, submitting web forms) read on the page.
5. Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. — *Conformal prediction beyond exchangeability* (arXiv:2202.13415; Ann. Statist. 2023). https://arxiv.org/abs/2202.13415
6. Angelopoulos, A., Bates, S., Fisch, A., Lei, L., Schuster, T. — *Conformal Risk Control*, ICLR 2024 (arXiv:2208.02814). https://arxiv.org/abs/2208.02814
7. Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. — *The limits of distribution-free conditional predictive inference* (arXiv:1903.04684). https://arxiv.org/abs/1903.04684 — source of the distribution-free impossibility of exact group-conditional coverage.
8. Stage-0 derived results and the scripts that regenerate them: `benchmarks/results/2026-11-03-stage0/` (`error_budget.json`, `gate_test938.json`, `gate_test705.json`, `score_fusion.json`, `class_bias.json`, `grounding_audit.json`); `benchmarks/train/{error_budget,gate_routed,score_fusion,class_bias,grounding_audit,paired_eval}.py`. All six JSONs re-derive byte-identically from the prediction artifacts in [9] (T4 §3).
9. Prediction artifacts: `benchmarks/results/2026-10-29-evals/{E2-938-V,E2-705-V,E2-val}.jsonl`; `backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl` (pure, 255/398); `backup-2026-11-06/offline/e2-pure-hist.jsonl` (served-API, 254/398); `backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl` (258/398). Re-checked during the citation pass: 398 rows / 350 distinct `episode_id` in the test artifact; 1 − 254/398 = 0.362 is the always-act error rate used in §7.
10. Provenance scripts: `benchmarks/train/verify_paper.py` (69/69 assertions, stage-0 JSON only) and `benchmarks/repro_manifest.py`.
11. Online run evidence: `backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/index.db` (tables `runs`, `episodes`, `steps`); `backup-2026-11-06/reports/aw_reports.tgz` (`pure8.json`, `fallback8.json`, `pure4.json`, `fallback4.json`, `gb_run1.json`, `gb_run2.json`, `gate_a.log`).
12. Accessibility diagnostic artifact: `benchmarks/results/d12bax-acv2-d12bax-2026-10-28.jsonl` (+ `.md`); `coverage` field = 216 True / 7 False / 175 null → 96.86% of 223 spatial rows.
13. El-Yaniv, R. & Wiener, Y. — *On the Foundations of Noise-free Selective Classification*, JMLR 11:1605–1641, 2010. https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf — live; coverage Φ = E[g(X)] and selective risk R = E[ℓ·g]/Φ read in Definitions 1–2.
14. Traub, J., Bungert, T., Lüth, C., Baumgartner, M., Maier-Hein, K., Maier-Hein, L., Jäger, P. — *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf — live; "substantially changes metric rankings on 5 out of the 6 data sets" (6 datasets × 13 confidence functions) confirmed in the abstract.
15. Galil, I., Dabbah, M. & El-Yaniv, R. — *What Can We Learn From The Selective Prediction And Uncertainty Estimation Performance Of 523 ImageNet Classifiers*, ICLR 2023 (arXiv:2302.11874). https://arxiv.org/abs/2302.11874
16. *What Does It Take to Build a Performant Selective Classifier?* — arXiv:2510.20242 (NeurIPS 2025 per the arXiv comments field). https://arxiv.org/abs/2510.20242 — live; "monotone post-hoc calibration … has limited impact on closing this gap, since it rarely alters the model's underlying score ranking" confirmed in the abstract.
17. *SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration* — arXiv:2602.02419, Feb 2026 (preprint). https://arxiv.org/abs/2602.02419 — live; FDR-controlled test-time threshold on ScreenSpot-Pro, "up to 5.38% percentage points over Gemini-only inference".
18. *AgentAbstain: Do LLM Agents Know When Not to Act?* — arXiv:2607.10059, Jul 2026 (preprint). https://arxiv.org/abs/2607.10059 — live; 263 paired tasks / 42 sandbox environments / 8 abstention scenarios; 17 frontier LLMs in 4 harnesses; best 59.5% paired accuracy (Gemini 3.1 Pro).
19. Sharma, R., Duncan, A. B., Marceau Caron, G., Vollmer, S. J. — *TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models* — arXiv:2610.11392, Oct 2026 (preprint). https://arxiv.org/abs/2610.11392 — live; "realised cost under asymmetric cost matrices", seven generators / nine suites.
20. *Optimizing resource consumption with pre-test probability and the acoustic-based rule-out CAD-score — insights from the FILTER-SCAD trial*, Eur. Heart J. abstract supplement, 2025. https://doi.org/10.1093/eurheartj/ehaf784.4494 — DOI resolves and the abstract is indexed ("adherence … was low, with less than 25% of patients deferred"); the OUP landing page itself is bot-blocked, so the sentence is verified from indexed abstract text rather than a direct page read.
21. *Learning When to Ask for Help: Efficient Interactive Navigation via Implicit Uncertainty Estimation* — arXiv:2305.16502; ICRA 2024. https://arxiv.org/abs/2305.16502 — live; "+0.38 … success … with only a 0.3 expert interaction rate … Habitat … simulated human expert" confirmed in the abstract.
22. *Overriding of Drug Safety Alerts in Computerized Physician Order Entry*, JAMIA (PMC1447540). https://pmc.ncbi.nlm.nih.gov/articles/PMC1447540/ — live; "overridden by clinicians in 49% to 96% of cases"; "Bates et al. propose a maximum override rate of 40% but do not offer an explanation for this figure"; override rise "from about 50% to 75% during a five-year period".
23. *Automation bias: a systematic review of frequency, effect mediators, and mitigators*, JAMIA (PMC3240751). https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751/ — live; Friedman et al.: "in 6% of cases, clinicians over-rode their own correct decisions in favor of erroneous advice".
24. Akhawe, D. & Felt, A. P. — *Alice in Warningland: A Large-Scale Field Study of Browser Security Warning Effectiveness*, USENIX Security 2013. https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/akhawe — live; paper text: users continued through "a tenth of Mozilla Firefox's malware and phishing warnings … a quarter of Google Chrome's … and a third of Mozilla Firefox's SSL warnings".
25. Transactions on Machine Learning Research — Acceptance Criteria. https://jmlr.org/tmlr/acceptance-criteria.html — live; "supported by accurate and convincing evidence" (most important criterion), the no-SOTA / no-novelty prohibition, and "another is for the authors to adjust (reduce) their claims" all confirmed.
26. ICML 2026 — Reviewer Instructions. https://icml.cc/Conferences/2026/ReviewerInstructions — live; "Soundness is distinct from impact … even if its contributions are modest or incremental" confirmed.
27. NeurIPS 2026 — Paper Checklist Guidelines. https://neurips.cc/public/guides/PaperChecklist — live; "answering 'no' or 'n/a' is not grounds for rejection" and "Reviewers will be specifically instructed to not penalize honesty concerning limitations" confirmed.
28. UIST 2026 — Author Guide. https://uist.acm.org/2026/author-guide/ — live; "User Studies Are Not Required … papers must support the claims they make with evidence" and strategy "(3) technical performance evaluations including benchmarking against thresholds or the state of the art" confirmed.
29. Geifman, Y. & El-Yaniv, R. — *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*, ICML 2019 (arXiv:1901.09192). https://arxiv.org/abs/1901.09192
30. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. — *On Calibration of Modern Neural Networks*, ICML 2017 (arXiv:1706.04599). https://arxiv.org/abs/1706.04599
31. Material Design 1 — *Confirmation & acknowledgement*. https://m1.material.io/patterns/confirmation-acknowledgement.html — live; "Confirmation isn't necessary when the consequences of an action are reversible or negligible" confirmed.
32. Nielsen, J. — *Confirmation Dialogs Can Prevent User Errors — If Not Overused*, NN/g. https://www.nngroup.com/articles/confirmation-dialog/ — live.
33. Shihab, I. F. et al. — *Continual Calibration: Coverage Can Collapse Before Accuracy in Lifelong LLM Fine-Tuning* — arXiv:2604.23987, Apr 2026 (preprint). https://arxiv.org/abs/2604.23987 — live; "in the most pronounced case, coverage drops from 0.92 to 0.61, while accuracy remains within three points of baseline"; coverage loss ≈3.4×±0.5× accuracy loss; calibration replay restores coverage "to within two points of nominal at buffer size m = 200".
34. *Conformal Prediction via Transported Beta Laws* — arXiv:2605.19024; PMLR 329:133–168 (2026). https://arxiv.org/abs/2605.19024 — live; calibration-conditional coverage law Beta(k, n+1−k) and explicit treatment of the clustered setting.
35. Internal note: `notes/2026-11-07-clock-checker-not-crash.md` — evidence from `backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/episodes/` that both "crashes" are Clock-task checker exceptions on a live device (last step `error: null`, `action_ok: true`; the next episode ran ten steps immediately after; stderr: `Could not get a11y tree`), with the harness fix and two regression tests in commit `2d68f68a`.

---

## Removed unsupported claims

- None. One row was removed by the citation pass and then **restored with evidence**: the §5 table's
  "25% environment crash" row. The verifier was right that the audit note (T4) predates the finding and marks
  the 2/8 rate "reproduces — keep"; the reclassification rests on a separate note written after that audit,
  `notes/2026-11-07-clock-checker-not-crash.md` [35], which shows the device was alive (the next episode ran
  ten steps immediately after each "crash") and that the failure is `is_successful()` raising while the
  Clock UI animates. The fix is committed (`2d68f68a`) with two regression tests.

---

## Verification notes (citation pass)

- Every URL in Sources was fetched during this pass. All resolve except two partial cases, both flagged in
  place: the FILTER-SCAD OUP landing page is bot-blocked (DOI resolves; abstract text verified from the
  indexing layer), and the USENIX landing page serves metadata only (the quoted clickthrough figures were
  read from the paper PDF).
- Three venue labels in the draft were corrected against the sources: 2305.16502 is ICRA 2024, not a preprint
  [21]; arXiv:2510.20242 carries a NeurIPS 2025 note [16]; arXiv:2605.19024 is in PMLR 329 [34]. None of the
  three changes the argument.
- The draft's third failure-mode bullet was labelled "(M3)"; T3's M3 is base-rate change and the coverage
  collapse it describes is M6 (T3 §1, M6; §0 item 5). The label was corrected to M6.
- The §5 closing line "two published sentences had to be withdrawn" asserted a count that the audit does
  not support (T4 §2 lists ten retract-or-restate items, R1–R10); it now reads "published sentences had to be
  withdrawn", with the item list cited.
- "17 frontier agents" was tightened to "17 frontier LLMs across four harnesses" to match the AgentAbstain
  protocol [18].
- Derived numbers introduced here, all arithmetic on cited rows and not independent measurements: always-act
  error 36.2% = 1 − 0.6382 [9]; the 350-episode count = distinct `episode_id` over 398 test rows [9]; the four
  cost-ratio break-evens in §7.
