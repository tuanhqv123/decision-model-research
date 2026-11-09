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

1. **Abstention changes real outcomes, but not in GUI agents — in the sources checked.** The strongest result
   is a prospective, **paired within-woman** noninferiority study of 31,301 mammograms in which a machine gate
   removed 63.6% of radiologist reading, raised cancer detection by 15.2%, and **failed its own recall-safety
   criterion** (T1 §3.1) [1]. Note the direction: there the machine abstains from *escalating to a human*, so
   the cost traded is reading time, not an interruption — the inverse of our setting. A peer-reviewed router
   shows the same shape in compute: 3.66× cost saving at 95% of the strong model's quality (T1 §3.3) [2]. For
   GUI agents, **no peer-reviewed outcome-level abstention result was found** in the sources checked (T1 §6).
2. **The gate is already platform policy; the calibration is the missing part.** Apple's HIG describes act /
   ask / withhold [3], and Google enumerates the action classes that must always ask [4]. Neither publishes a
   numeric confidence threshold, a measured ask rate, or an outcome measurement (T2 §A, §B, §E) [3, 4]. Every
   shipped gate **found** is a hand-authored rule keyed on action class, not a learned confidence threshold
   (T2 §E) [3, 4].
3. **A per-checkpoint certificate is a measurement, not a product property.** Every published guarantee
   requires the target distribution, a density ratio, a stationarity assumption, or a finite conditioning
   class; under shift the bound degrades by a total-variation term that cannot be estimated in production
   (T3 §0–§2) [5, 6]. **Exact** group-conditional coverage is provably impossible distribution-free (T3 §6)
   [7].

And the undercut: **our own program's headline does not reproduce as published.** The internal audit finds no
artifact anywhere for 64.32% (256/398); the two pure-protocol artifacts are 255/398 (in-process) and 254/398
(served); the 56.50% grounding belongs to the *step-705* checkpoint; the candidate-coverage claim of 100% is
96.9%; and the online invalid-action intervals overlap (Fisher p = 0.065) on **task instances that were not
paired** (T4 §1–§2) [8, 9, 11, 12].

The value question is therefore narrower and better than "is calibration interesting". It is:

> **At the same accuracy, does gating on confidence beat both trivial policies — always-act and always-ask —
> at a cost ratio a real product actually faces?**

On the artifact the gate was fitted to, the answer is yes, in a window: with 64.82% accuracy, 40.7% coverage
and 17.3% empirical selective risk, the gate beats **always-act** once a wrong action costs more than
**2.11×** an interruption, and beats **always-ask** until a wrong action costs **5.79×** one. The same
arithmetic on the served artifact's accuracy (63.82%) gives 2.03. On the next checkpoint the window is
**2.08–4.19**. That is a claim a reviewer can grade and a builder can price — the decision-level statement an
accuracy tie can carry.

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

**In adjacent fields, with effect sizes; not in GUI agents, in the sources checked.** The distinction is the
reason this programme has a defensible niche — and it is a statement about a search, not about the world.

**Medical screening (peer-reviewed, the strongest result).** A prospective paired noninferiority study in
mammography and digital tomosynthesis had **each participant read under both strategies** — standard double
reading, and AI-supported screening in which low-risk cases were called normal and the rest double-read with
AI support (T1 §3.1) [1]. Radiologist workload fell 63.6%; cancer detection rose 15.2% (95% CI 6.6–24.4);
**recall rose 14.8% (95% CI 9.0–20.6), which failed the study's noninferiority criterion** [1]. Two features
matter more than the headline. The design is paired and within-subject, not two parallel arms. And the
direction is inverted relative to an on-device gate: the machine abstains from *escalating to the radiologist*,
so what it trades is human reading time — not a user interruption, which is the cost §7 prices. It is the
right *form* of evidence in the opposite direction.

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

**The GUI gap (as searched).** In the sources checked (T1 §6 lists them), no GUI benchmark reports selective
risk as a headline, no GUI-agent paper prices the *ask*, no paper measures calibration→outcome transfer for a
mobile agent, and no GUI paper reports threshold transfer across checkpoints. The nearest artifacts are a
grounding-level FDR-controlled threshold (SafeGround, +5.38 pp over Gemini-only inference) [17], a preprint
paired-task abstention benchmark on which 17 frontier LLMs across four harnesses reach 59.5% paired accuracy
[18], and a typed-decision benchmark with a realised-cost experiment on non-GUI decision tasks [19] (T1 §2.2,
§2.4, §2.7).

---

## 3. Who cares, and what each one needs

**A CCF-A reviewer** grades claim–evidence fit, not the leaderboard. TMLR's stated first criterion is whether
claims are supported by convincing evidence, and it explicitly forbids "not SOTA" and "not novel enough" as
reasons to reject; its remedy for an overclaim is to reduce the claim (T2 §A.1) [25]. ICML separates
soundness from impact (T2 §A.1) [26]; NeurIPS instructs reviewers not to penalise honesty about limitations
(T2 §A.1) [27]; UIST accepts threshold benchmarking as validation, with no user study required (T2 §A.1)
[28]. Precedent that this suffices: Guo et al.'s calibration paper [30] and SelectiveNet [29] are accepted
without an accuracy contribution (T2 §A.3).

What a reviewer will not accept: deployment readiness inferred from ranking metrics; a per-checkpoint
threshold presented as a product property; a certificate that does not re-derive from saved artifacts; and no
cost model (T2 §A.4) [25].

**A builder** needs one number: expected cost per 1,000 gate decisions at the product's own cost ratio,
including the measured ask rate and the dismissal rate (T2 §B.6). Apple's guidance is already stated this way
— compare the cost of correction with the benefit of the feature (T2 §B) [3] — and Google's always-ask list
defines where a gate is mandatory (T2 §B) [4]. But no first-party document **we found** gives a numeric
confidence threshold or a latency target; every shipped gate found is rule-based (T2 §E) [3, 4].

**An end user** needs the harm avoided to be worse than the interruption. The concrete harms are legible:
message sent to the wrong recipient, wrong purchase, irrecoverable delete, an agent submitting a form on the
user's behalf (T2 §C.3) [4, 31, 32]. The cost side is measured and unflattering: override 49–96% [22],
habituation over days [T2 §C.2, abstract-level only], and "are you sure?" producing reflex approval [32]
(T2 §C.3–§C.4). Confirmation is the wrong tool when undo would work; the reversibility rule appears in both
platform guidance and the design literature (T2 §C.3) [31, 32].

**The inference this brief commits to.** The contribution is not "we invented ask-before-acting". It is:
**a learned, calibrated threshold on a single-pass model, reported as risk at coverage with baselines, plus a
measured ask rate and a cost ratio.** That is the intersection of what reviewers grade [25] and what builders
price (T2 §B.6), and in the sources checked it is absent from the GUI literature (T2 §A.5, T1 §6).

---

## 4. What has to be true for the gate to be safe

No published method gives a distribution-free guarantee under arbitrary shift (T3 §0) [5]. The strongest
bounds degrade by a total-variation term that cannot be estimated from unlabelled deployment data (T3 §1)
[5, 6]. Three failure modes matter most here.

- **Checkpoint and population change (M1).** Our own transfer measurement — 17.3% risk at one checkpoint,
  23.9% at the next under the same threshold — is a **certificate failure**: the guarantee no longer holds. It
  is *not* a demonstrated risk change (z ≈ 1.45; T4 §4.6) [8]. The audit also notes the certificate is
  **val-side**: the certified point is val coverage 47.6% at a val upper bound of 19.9%, while the quoted 40.7%
  is test coverage and the test-side Clopper–Pearson upper bound is 21.7%, so the 20% budget is not certified
  on test [8].
- **Clustered evaluation units (M2).** Rows are not exchangeable when they share episodes, users, or tasks;
  the effective sample size is the number of independent units, not rows (T3 §1, single recent preprint) [34].
  Our 398 test steps span 350 episodes (distinct `episode_id` in the test artifact) [9], which is why the
  paired tests are episode-clustered [8].
- **Silent coverage decay (M6).** Coverage can fall from 0.92 to 0.61 while accuracy stays within three points
  of baseline (T3 §0; **single preprint, not independently replicated**) [33]. A one-shot certificate would
  not see it.

What a deployer can defensibly do is not a certificate but a *contract plus an instrument*: a named
invariance assumption, an anytime-valid coverage monitor with a stated false-alarm rate, and a re-fit trigger
tied to model updates rather than the calendar (T3 §5–§6) [5, 6, 33]. No paper we found demonstrates that
combination end-to-end on a shipped act/ask/refuse gate. The honest sentence is: **"certified per checkpoint,
monitored continuously, invalidated by any distributional change we cannot verify we did not have."**

---

## 5. What this program actually has (after the audit)

The audit re-derived every headline number from disk (T4). The calibration half is solid; the accuracy half
needed restating.

| Claim | Published | Re-derived | Status |
|---|---|---|---|
| AC-v2 pure Pass@1 | 64.32% (256/398) | 64.07% (255/398) in-process; 63.82% served | **cannot reproduce** → restated |
| groundB Pass@1 | 64.82% (258/398) | 258/398 | reproduces |
| Type accuracy | 84.17% | 335/398; class-bias-adjusted **81.41%** (7 fixed / 18 regressed, McNemar p = 0.043) | reproduces; the adjustment **fails** on test |
| Grounding | 56.50% (126/223) | 56.05% pure / 57.40% hinted; **56.50 is step 705** | reproduces, wrong checkpoint |
| Type-routed gate AUROC | 0.760 / 0.729 / 0.709 | identical | reproduces |
| Certified 20% risk @ 40.7% coverage | as stated | certified on val (47.6% coverage, upper bound 19.9%); test risk 17.3%, test CP upper **21.7%** | reproduces with caveat |
| Learned-fusion risk@50% | 19.1% | 38/199 = 0.191; ±5.5 pp binomial (episode-clustered interval wider) | reproduces, underpowered |
| Candidate coverage | 1.0000 | 216/223 = 0.9686 | **contradicted** |
| Online invalid actions | 28.2% → 14.7%, "non-overlapping" | intervals overlap 19.0–25.0; Fisher p = 0.065; instances unmatched | **contradicted** |
| Online SR | 1/8 both | 1/8 both, Wilson 2.2–47.1; but `n_reachable = 0`, `sr_reachable` undefined, timeouts 7/8 and 6/8 | reproduces; the reachable split is **undefined** |
| Online "25% environment crash" | 2/8 per condition | two Clock-task **checker exceptions on a live device**; last step `error: null`, `action_ok: true`; the next episode ran ten steps immediately after | **misclassified by the harness, now fixed** [35] |

*Rows 1–7: (T4 §1, §2) [8, 9]; candidate coverage: [12]; online rows: [11]; the crash reclassification: [35].*

Everything in `benchmarks/results/2026-11-03-stage0/` now re-derives byte-identically — the one exception the
audit found, `paired_705_vs_938.txt`, was regenerated and committed afterwards (`2ddc7cc2`), and its
episode-clustered CI is **[+0.76, +6.35]**, i.e. a *significant* improvement from 244/398 to 258/398 that this
brief cites because it is the one accuracy comparison in the artifact set with a test behind it. `verify_paper.py`
passes 69/69, but it asserts stage-0 derived JSON only: not the pure-protocol row, the baselines, the pixel
distances, the a11y diagnostic, or any online number (T4 §2 R8) [8, 10].

**What this means for the value claim.** The calibration result survives audit. The headline accuracy is
**64.07% against GUI-Libra-8B's 64.30%**, with a 95% bootstrap CI of ±4.7 pp around our value — a tie against
that named comparator, not against the board. The online half is a harness finding, not a capability result.
Nothing here invalidates the gate argument, and several published sentences had to be withdrawn (T4 §2,
R1–R10) [8].

---

## 6. The pre-registered value test

The plan committed to delivering only if at least two of five conditions held. They are scored here.

1. **A peer-reviewed outcome-level result where abstention changes success or harm — HOLDS.** The mammography
   study [1] and RouteLLM [2] (T1 §3.1, §3.3).
2. **A named, concrete user harm an act/ask/refuse gate prevents, with the cost of asking stated — HOLDS.**
   Wrong recipient, wrong purchase, destructive delete, agent-submitted forms [4, 31, 32], each with measured
   override and habituation costs [22, 23] (T2 §C.3).
3. **A reviewer-legible contribution shape our existing artifacts support without new training — HOLDS, with
   a condition.** The shape is "risk at coverage with trivial baselines and a cost ratio" (T2 §A.5) [25]. The
   artifacts support it: the always-`status` (21.86%) and always-`click` (53.27%) baselines are in
   `error_budget.json` [8], and the cost ratio is swept in [36]. What they do not support is *transfer* — the
   certificate fails on the next checkpoint (17.3% → 23.9%).
4. **One next measurement with a predicted effect size, a cost, and a kill criterion — HOLDS.** §7, delivered:
   the predicted effect size is a cost window, and the run costs CPU seconds.
5. **A documented reason the current headline cannot support a CCF-A claim yet — HOLDS.** In the sources
   checked, no GUI outcome result was found (T1 §6); the accuracy is a tie against a named 8B baseline [8];
   the certificate is per-checkpoint (T3 §6) [7]; and no GUI paper prices the ask (T1 §6).

**Verdict: the brief passes (5/5), and the two weakest conditions are 1 and 3** — condition 1 rests on one
trial in an inverted construct plus one routing paper; condition 3 rests on baselines that exist but were
never combined into a single table before §7.

---

## 7. The next measurement, delivered

The measurement proposed in the plan is now computed, from saved artifacts, with no model calls
(`benchmarks/results/2026-11-07-cost-window.md`; script `benchmarks/train/cost_window.py`).

Price 1,000 decisions with `C_ask = 1`, `R = C_err/C_ask`, and a delegated decision that is resolved
correctly with probability `1 − q`:

- **always-act**: `1000 · (1 − acc) · R`
- **always-ask**: `1000 · (1 + q·R)`
- **gate at coverage c**: `1000 · (c · risk(c) · R + (1 − c) · (1 + q·R))`

| Comparison (q = 0) | E2@938 (64.82% acc) | E2@705 (61.31% acc) |
|---|---|---|
| gate beats always-act when | **R > 2.11** | **R > 2.08** |
| gate beats always-ask when | **R < 5.79** | **R < 4.19** |
| learned fusion @50% vs always-act / always-ask | 1.95 / 5.24 | — |

**The lower break-even is a ratio of two estimates, so it is an interval, not a decimal.** Propagating the
always-act accuracy's Wilson interval (254/398 → [0.590, 0.684], i.e. error 0.316–0.410) puts the lower
break-even in **[1.75, 2.41]**; propagating the risk interval (28/162 → [0.122, 0.239]) gives **[1.96, 2.33]**. The point estimate is 2.11
on the fitted artifact and 2.03 on the served one — the two sides of the comparison come from those two
protocols unless stated otherwise, and both are reported here for that reason.

**The perfect-human assumption is load-bearing, and it is the first thing to vary.** If a delegated decision is
wrong with probability `q`, the window moves: `q = 5%` → **2.36–8.14**; `q = 10%` → **2.67–13.7**. It moves in the
gate's favour at the top (a fallible ask is worse than asking perfectly) and against it at the bottom (a
fallible human makes delegation less attractive at low ratios). `q = 0` gives the most favourable **lower** edge and the
least favourable **upper** edge; a fallible human widens the window upward because asking becomes less
reliable than the model's own action.

**What the model omits, deliberately.** The refuse branch (`r·C_refuse` in T2 §B.6's value unit) and the
dismissal-rate tolerance are dropped: this checkpoint's gate has two outputs in practice, because the model
either acts or defers, and no measured dismissal rate exists for it. A three-way gate would need both terms
reinstated, and T2 §B.6 is the formula to use when they are measured.

**Reading.** The gate dominates both trivial policies over roughly a **2.1–5.8×** cost window on the
checkpoint the certificate was fitted to, and **2.1–4.2×** on the next one. The window closes exactly when the
gate's selective risk on the acted subset reaches the always-act error rate (`risk(c) ≥ 1 − acc`) — at the
measured 23.9% that is far from met. Below ~2×, acting is cheap enough that an interruption is not worth
paying for; above ~6×, asking every time is better, because the gate still waves through 40.7% of decisions at
17.3% risk. **The accuracy tie becomes a decision-level statement in that band**, which is the only kind of
claim a tie can support.

**Kill criterion (revised, and why).** The plan's original anchor — "dominates at `R ≥ 2`" — was badly chosen:
the point estimate is 2.03–2.11, so the criterion fires on the very data that motivated it, and 2.0 sits inside
its confidence band. The criterion used here is **`R = 3`**, a deliberately conservative reference in which a
wrong action costs three interruptions: the gate dominates on both checkpoints (2.11 and 2.08 are below 3;
5.79 and 4.19 above). If it did not dominate at 3 on both, the value claim would fail for this checkpoint pair
and the paper would have to say so — the certificate, not the cost argument, would then be the contribution.

**Assumption, not measurement.** No GUI study prices the ask. Real asks are dismissed (49–96% override in
clinical settings), habituate within days, and are not always available. Everything above is therefore a
model of the gate's value under stated assumptions, not a measured user outcome.

---

## 8. Open questions

1. **What is the real cost of an ask?** The literature measures override and habituation, not the price of an
   interruption in the product's own utility. The 40% maximum override rate is proposed without derivation
   (T2 §E) [22].
2. **Does the cost argument survive a second checkpoint?** Measured here: yes, but narrower (2.08–4.19 vs
   2.11–5.79). What is not measured is a *third* checkpoint or a population change.
3. **Is there any GUI agent for which an outcome-level abstention result exists?** None found in the sources
   checked; the claim is "not found", not "does not exist" (T1 §6).
4. **Should the gate be per-action-type?** An irreversible tap and a scroll have different costs; a single
   threshold cannot express that. No paper found reports a per-item harm-weighted loss for an act/ask/refuse
   gate (T1 §6 item 8); the per-action-type version is this brief's inference.
5. **What would make us stop?** If the window is empty, or does not contain the product's ratio, on both
   checkpoints, the programme should pivot from gating to the certificate boundary as the contribution —
   which is publishable and honest, and smaller.

---

## Sources

Numbered sources follow the order of first citation in the brief. Entries **8–12, 35 and 36 are internal
repository artifacts, not web sources**; of the rest, [20]'s publisher landing page is bot-blocked and [24]'s
PDF failed to parse, both noted in place.

1. *AI-based triage and decision support in mammography and digital tomosynthesis for breast cancer screening: a paired, noninferiority trial* — Nature Medicine, 2026. https://www.nature.com/articles/s41591-026-04277-x — live; n = 31,301; workload −63.6%; CDR +15.2% (95% CI 6.6–24.4); recall +14.8% (9.0–20.6), "not noninferior". Design read from the page: "prospective paired noninferiority experimental study"; "each participant received both reading strategies".
2. Ong, I., Almahairi, A. et al. — *RouteLLM: Learning LLMs to Route from Preference Data*, ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/5503a7c69d48a2f86fc00b3dc09de686-Paper-Conference.pdf — live; Table 6 (3.66× @95% GPT-4 quality on MT Bench; 1.41× @92% on MMLU; 1.49× @87% on GSM8K).
3. Apple — *Human Interface Guidelines: Machine Learning* (Confidence section). https://developer.apple.com/design/human-interface-guidelines/machine-learning — live; confidence thresholds, "verify that your confidence values correspond to the quality of your results", and the cost-of-correction guidance.
4. Google — *Manage your tasks & workflows in Gemini Apps*. https://support.google.com/gemini/answer/16596215 — live; the always-ask list (sending communications, deleting data, purchases, financial transactions, submitting web forms).
5. Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. — *Conformal prediction beyond exchangeability* (arXiv:2202.13415; Ann. Statist. 2023). https://arxiv.org/abs/2202.13415
6. Angelopoulos, A., Bates, S., Fisch, A., Lei, L., Schuster, T. — *Conformal Risk Control*, ICLR 2024 (arXiv:2208.02814). https://arxiv.org/abs/2208.02814
7. Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. — *The limits of distribution-free conditional predictive inference* (arXiv:1903.04684). https://arxiv.org/abs/1903.04684 — source of the distribution-free impossibility of **exact** group-conditional coverage.
8. Stage-0 derived results and the scripts that regenerate them: `benchmarks/results/2026-11-03-stage0/` (`error_budget.json`, `gate_test938.json`, `gate_test705.json`, `score_fusion.json`, `class_bias.json`, `grounding_audit.json`, `paired_705_vs_938.txt`); `benchmarks/train/{error_budget,gate_routed,score_fusion,class_bias,grounding_audit,paired_eval}.py`.
9. Prediction artifacts: `benchmarks/results/2026-10-29-evals/{E2-938-V,E2-705-V,E2-val}.jsonl`; `backup-2026-10-29/Model_Training_Server/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl` (pure, 255/398); `backup-2026-11-06/offline/e2-pure-hist.jsonl` (served API, 254/398); `backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl` (258/398). 398 test rows over 350 distinct `episode_id`; 827 validation rows.
10. Provenance scripts: `benchmarks/train/verify_paper.py` (69/69 assertions, stage-0 JSON only) and `benchmarks/repro_manifest.py` (protocol now declared by the caller, not inferred from the value it verifies — T4 R10).
11. Online run evidence: `backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/index.db` (tables `runs`, `episodes`, `steps`); `backup-2026-11-06/reports/aw_reports.tgz` (`pure8.json`, `fallback8.json`, `pure4.json`, `fallback4.json`, `gb_run1.json`, `gb_run2.json`, `gate_a.log`). Both report JSONs record `n_reachable: 0`, `sr_reachable: null`, `coverage: {den: 0}` and timeouts of 7/8 and 6/8.
12. Accessibility diagnostic artifact: `benchmarks/results/d12bax-acv2-d12bax-2026-10-28.jsonl` (+ `.md`); `coverage` = 216 True / 7 False / 175 null → 96.86% of 223 spatial rows.
13. El-Yaniv, R. & Wiener, Y. — *On the Foundations of Noise-free Selective Classification*, JMLR 11:1605–1641, 2010. https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf — live; coverage Φ = E[g(X)] and selective risk R = E[ℓ·g]/Φ in Definitions 1–2.
14. Traub, J., Bungert, T., Lüth, C., Baumgartner, M., Maier-Hein, K., Maier-Hein, L., Jäger, P. — *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf — live; metric rankings change on 5 of 6 datasets.
15. Galil, I., Dabbah, M. & El-Yaniv, R. — *What Can We Learn From The Selective Prediction And Uncertainty Estimation Performance Of 523 ImageNet Classifiers*, ICLR 2023 (arXiv:2302.11874). https://arxiv.org/abs/2302.11874
16. *What Does It Take to Build a Performant Selective Classifier?* — arXiv:2510.20242 (NeurIPS 2025 per the arXiv comments field). https://arxiv.org/abs/2510.20242 — live; "monotone post-hoc calibration … has limited impact on closing this gap".
17. *SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration* — arXiv:2602.02419, Feb 2026 (preprint). https://arxiv.org/abs/2602.02419 — FDR-controlled threshold on ScreenSpot-Pro, "up to 5.38 percentage points over Gemini-only inference".
18. *AgentAbstain: Do LLM Agents Know When Not to Act?* — arXiv:2607.10059, Jul 2026 (preprint). https://arxiv.org/abs/2607.10059 — 263 paired tasks / 42 sandbox environments; 17 frontier LLMs in 4 harnesses; best 59.5% paired accuracy.
19. Sharma, R., Duncan, A. B., Marceau Caron, G., Vollmer, S. J. — *TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models* — arXiv:2610.11392, Oct 2026 (preprint). https://arxiv.org/abs/2610.11392 — "realised cost under asymmetric cost matrices"; seven generators / nine suites.
20. *Optimizing resource consumption with pre-test probability and the acoustic-based rule-out CAD-score — insights from FILTER-SCAD*, Eur. Heart J. abstract supplement, 2025. https://doi.org/10.1093/eurheartj/ehaf784.4494 — DOI resolves; the publisher page is bot-blocked, so "adherence … less than 25%" is verified from indexed abstract text.
21. *Learning When to Ask for Help: Efficient Interactive Navigation via Implicit Uncertainty Estimation* — arXiv:2305.16502; ICRA 2024. https://arxiv.org/abs/2305.16502 — "+0.38 … with only a 0.3 expert interaction rate … Habitat … simulated human expert".
22. *Overriding of Drug Safety Alerts in Computerized Physician Order Entry*, JAMIA (PMC1447540). https://pmc.ncbi.nlm.nih.gov/articles/PMC1447540/ — "overridden by clinicians in 49% to 96% of cases"; a maximum override rate of 40% is proposed without derivation; override rise from ~50% to 75% over five years.
23. *Automation bias: a systematic review of frequency, effect mediators, and mitigators*, JAMIA (PMC3240751). https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751/ — "in 6% of cases, clinicians over-rode their own correct decisions in favor of erroneous advice".
24. Akhawe, D. & Felt, A. P. — *Alice in Warningland: A Large-Scale Field Study of Browser Security Warning Effectiveness*, USENIX Security 2013. https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/akhawe — landing page serves metadata only; clickthrough figures read from the paper PDF.
25. Transactions on Machine Learning Research — Acceptance Criteria. https://jmlr.org/tmlr/acceptance-criteria.html — "supported by accurate and convincing evidence"; no-SOTA / no-novelty prohibition; "adjust (reduce) their claims".
26. ICML 2026 — Reviewer Instructions. https://icml.cc/Conferences/2026/ReviewerInstructions — "Soundness is distinct from impact … even if its contributions are modest or incremental".
27. NeurIPS 2026 — Paper Checklist Guidelines. https://neurips.cc/public/guides/PaperChecklist — "answering 'no' or 'n/a' is not grounds for rejection"; reviewers told not to penalise honesty about limitations.
28. UIST 2026 — Author Guide. https://uist.acm.org/2026/author-guide/ — "User Studies Are Not Required … papers must support the claims they make with evidence"; threshold benchmarking is an accepted strategy.
29. Geifman, Y. & El-Yaniv, R. — *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*, ICML 2019 (arXiv:1901.09192). https://arxiv.org/abs/1901.09192
30. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. — *On Calibration of Modern Neural Networks*, ICML 2017 (arXiv:1706.04599). https://arxiv.org/abs/1706.04599
31. Material Design 1 — *Confirmation & acknowledgement*. https://m1.material.io/patterns/confirmation-acknowledgement.html — "Confirmation isn't necessary when the consequences of an action are reversible or negligible".
32. Nielsen, J. — *Confirmation Dialogs Can Prevent User Errors — If Not Overused*, NN/g. https://www.nngroup.com/articles/confirmation-dialog/
33. Shihab, I. F. et al. — *Continual Calibration: Coverage Can Collapse Before Accuracy in Lifelong LLM Fine-Tuning* — arXiv:2604.23987, Apr 2026 (preprint; no venue). https://arxiv.org/abs/2604.23987 — "coverage drops from 0.92 to 0.61, while accuracy remains within three points of baseline".
34. *Conformal Prediction via Transported Beta Laws* — arXiv:2605.19024; PMLR 329:133–168 (2026). https://arxiv.org/abs/2605.19024 — calibration-conditional coverage law and explicit treatment of the clustered setting.
35. Internal note: `notes/2026-11-07-clock-checker-not-crash.md` — evidence from `backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/episodes/` that both "crashes" are Clock-task checker exceptions on a live device, with the harness fix and two regression tests in commit `2d68f68a`.
36. Cost-window artifact: `benchmarks/results/2026-11-07-cost-window.md`, `benchmarks/train/cost_window.py`, `benchmarks/results/2026-11-07-cost-window.json` — derived from `gate_test938.json`, `gate_test705.json`, `score_fusion.json` and the matching prediction artifacts; inputs and assumptions stated in the file.

---

## Revision record (2026-11-07, post-review)

The reviewer flagged six MAJOR issues; all six are fixed here.

- **W1 (prediction contradicted by its own rows)** — the "narrows to nothing" prediction is replaced by the
  measured second-checkpoint window (2.08–4.19), and the exact condition under which the window closes is
  stated (`risk(c) ≥ 1 − acc`). The kill criterion is re-anchored at `R = 3`, with the reason the original
  `R ≥ 2` anchor was invalid (it sat below the point estimate and inside its confidence band).
- **W2 (perfect-human assumption, dropped refuse branch, mixed protocols)** — the cost model now carries a
  delegated-error rate `q`, with `q = 5%` and `q = 10%` sensitivity; the refuse branch and dismissal tolerance
  are explicitly declared out of scope with a pointer to T2 §B.6; both protocol variants of the break-even are
  reported (2.11 on the fitted artifact, 2.03 on the served one).
- **W3 (three-significant-figure point estimates)** — the lower break-even is reported as an interval from both
  inputs' CIs (2.03–2.41 from accuracy; 1.96–2.33 from risk).
- **W4 (unhedged negative)** — the negative is now stated once, in T1 §6's own formulation ("in the sources
  checked / not found"), at every site, and §6 item 5 no longer says "exists".
- **W5 (trial design and inversion)** — corrected to "paired within-woman" (the source's own wording) and the
  inverted direction is stated next to the "exact shape" sentence.
- **W6 (the tie has no comparator or test)** — the comparator is named (GUI-Libra-8B, 64.30, ±4.7 pp), and the
  one accuracy comparison that *is* significant (244/398 → 258/398, McNemar p = 0.0161, episode-clustered CI
  [+0.76, +6.35]) is cited.
- **Minors** — rows 1–7; the a11y diagnostic added to the `verify_paper.py` scope list; the `paired_705_vs_938.txt`
  regeneration noted; class-bias-adjusted type accuracy (81.41%, p = 0.043) added as a second column; the
  certified val coverage (47.6%) added; the online-SR row now carries the reachability/timeout context;
  R1–R10; "found" hedges restored; preprints flagged inline for [33]/[34]; the habituation citation given a
  number and an abstract-only caveat; "exact" conditional coverage; 5.79; act/ask/refuse unified; the §6
  baselines named; the TypedBench "tickets" detail removed.
