# What a calibrated act / ask / refuse gate is actually worth

*Research brief. 2026-11-07. Built from four evidence notes: T1 metrics and outcome evidence, T2 audience
requirements, T3 production failure modes, T4 internal audit. Source markers in brackets resolve in the
Sources section.*

---

## Executive summary

A calibrated single-pass decision model answers a fixed question with a probability instead of writing a
sentence. "Calibrated" means that when it says 0.8, it is right about 80% of the time. A gate turns that
probability into one of three actions: **act**, **ask the human**, or **refuse**.

The literature supports three claims and undercuts a fourth.

1. **Abstention does change real outcomes — but almost none of that evidence is from GUI agents.** The
   strongest result is a 31,301-woman paired noninferiority trial in which a machine gate removed 63.6% of
   radiologist reading, raised cancer detection by 15.2%, and **failed its own recall-safety criterion**
   (T1 §3.1 [33]). A peer-reviewed router shows the same shape in compute: 3.66× cost saving at 95% of the
   strong model's quality (T1 §3.3 [36]). In GUI agents there is **no peer-reviewed outcome-level abstention
   result at all** (T1 §6, §7).
2. **The gate is already platform policy; the calibration is the missing part.** Apple's HIG describes act /
   ask / suppress, and Google enumerates the action classes that must always ask. Neither publishes a numeric
   confidence threshold, a measured ask rate, or an outcome measurement (T2 §A, §B, §E). Every shipped gate
   found is a hand-authored rule keyed on action class, not a learned confidence threshold.
3. **A per-checkpoint certificate is a measurement, not a product property.** Every published guarantee
   requires the target distribution, a density ratio, a stationarity assumption, or a finite conditioning
   class; under shift the bound degrades by a total-variation term that cannot be estimated in production
   (T3 §0–§2). Group-conditional coverage is provably impossible distribution-free (T3 §6 [7]).

And the undercut: **our own program's headline does not reproduce as published.** The internal audit finds no
artifact anywhere for 64.32% (256/398); the two pure-protocol artifacts are 255/398 (in-process) and 254/398
(served), the 56.50% grounding belongs to the *step-705* checkpoint, the candidate-coverage claim of 100% is
96.9%, and the online invalid-action intervals overlap (Fisher p = 0.065) on **task instances that were not
paired** (T4 §1–§2).

The value question is therefore narrower and better than "is calibration interesting". It is:

> **At the same accuracy, does gating on confidence beat both trivial policies — always-act and always-ask —
> at a cost ratio a real product actually faces?**

On our artifacts the answer is yes, in a window. With a measured 36.2% error when acting always and 17.3%
selective risk at 40.7% coverage, the gate dominates **always-act** once a wrong action costs more than
**~2.0×** an interruption, and still beats **always-ask** until a wrong action costs **~5.8×** one. That is a
claim a reviewer can grade and a builder can price — and it is the one contribution shape our existing
artifacts can support without new training.

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

Two structural results matter for framing. First, post-hoc calibration has *limited* impact on selective
performance, because it rarely changes the score ranking (T1 §1.3 [9]). Second, the choice of selective
metric is not decision-stable: AURC and AUGRC reorder the leaderboard on 5 of 6 datasets (T1 §1.3 [4]).
Within GUI agents, the modal claim stops at step accuracy with no calibration metric at all (T1 §2.5).

**Implication.** A reliability paper must report coverage-conditioned risk *and* name the estimator. "ECE
improved" is not a deployment claim, and an AUROC is not a certificate.

---

## 2. Does abstention change outcomes?

**Yes in adjacent fields, with effect sizes; no in GUI agents.** The distinction is the reason this program
has a defensible niche.

**Medical screening (peer-reviewed, the strongest result).** A paired noninferiority trial in mammography and
digital tomosynthesis ran standard double reading in parallel with AI-supported screening in which low-risk
cases were called normal and the rest double-read with AI support (T1 §3.1 [33]). Radiologist workload fell
63.6%; cancer detection rose 15.2% (95% CI 6.6–24.4); **recall rose 14.8% (95% CI 9.0–20.6), which failed the
pre-registered noninferiority criterion**. The gate traded human time for sensitivity and lost the safety
margin it promised. That is the exact evidence shape our program needs — and the exact reason a bare AUROC
claim is not enough.

**Compute routing (peer-reviewed).** RouteLLM decides *before* any model is called whether a query goes to the
strong or weak model, and reports 3.66× cost saving at 95% of GPT-4 quality on MT-Bench, 1.41×/1.49× on MMLU
and GSM8K at their stated thresholds (T1 §3.3 [36]). This is the closest analogue to our value claim and it is
favourable: escalation gates pay when the cost asymmetry is large and the quality bar is explicit.

**Robotics (preprint).** A learned interaction policy that asks for help reports +0.38 task success at a 0.3
expert-interaction rate in Habitat with a simulated expert (T1 §3.5 [37]). Clean shape, simulated human,
preprint.

**The cautionary result.** A clinical deferral strategy had **under 25% adherence** — the gate's benefit
depends on the person receiving the ask actually following it (T1 §3.2 [35]). Human-factors work shows the
same pressure from the other side: alert override rates of 49–96%, clickthrough of a tenth to a third, and
automation-bias commission errors in ~6% of cases where clinicians overrode their own correct judgement
(T2 §C.3 [13][14][16]).

**The GUI gap.** No GUI benchmark reports selective risk as a headline; no GUI-agent paper prices the *ask*;
no paper measures calibration→outcome transfer for a mobile agent; no GUI paper reports threshold transfer
across checkpoints (T1 §6). The nearest artifacts are a grounding-level FDR-controlled threshold, a preprint
paired-task abstention benchmark on which 17 frontier agents reach 59.5% paired accuracy, and a typed-decision
benchmark with a realised-cost experiment on non-GUI tickets (T1 §2.2, §2.4, §2.7).

---

## 3. Who cares, and what each one needs

**A CCF-A reviewer** grades claim–evidence fit, not the leaderboard. TMLR's stated first criterion is whether
claims are supported by convincing evidence, and it explicitly forbids "not SOTA" and "not novel enough" as
reasons to reject; its remedy for an overclaim is to reduce the claim (T2 §A.1 [1]). ICML separates soundness
from impact (T2 §A.1 [2]); NeurIPS instructs reviewers not to penalise honesty about limitations (T2 §A.1
[3]); UIST accepts threshold benchmarking as validation, with no user study required (T2 §A.1 [5]). Precedent
that this suffices: Guo et al.'s calibration paper and SelectiveNet are accepted without an accuracy
contribution (T2 §A.3 [23][24]).

What the reviewer will not accept: deployment readiness inferred from ranking metrics; a per-checkpoint
threshold presented as a product property; a certificate that does not re-derive from saved artifacts; and no
cost model (T2 §A.4).

**A builder** needs one number: expected cost per 1,000 gate decisions at the product's own cost ratio,
including the measured ask rate and the dismissal rate (T2 §B.6). Apple's guidance is already stated this way
— compare the cost of correction with the benefit of the feature (T2 §B [6]) — and Google's always-ask list
defines where a gate is mandatory (T2 §B [8]). But no first-party document gives a numeric confidence
threshold or a latency target; every shipped gate is rule-based (T2 §E).

**An end user** needs the harm avoided to be worse than the interruption. The concrete harms are legible:
message sent to the wrong recipient, wrong purchase, irrecoverable delete, an agent submitting a form on the
user's behalf (T2 §C.3). The cost side is measured and unflattering: override 49–96%, habituation over days,
and "are you sure?" producing reflex approval (T2 §C.3–§C.4). Confirmation is the wrong tool when undo would
work; the reversibility rule appears in both platform guidance and the design literature (T2 §C.3 [9][11]).

**The inference this brief commits to.** The contribution is not "we invented ask-before-acting". It is:
**a learned, calibrated threshold on a single-pass model, reported as risk at coverage with baselines, plus a
measured ask rate and a cost ratio.** That is the exact intersection of what reviewers grade and what builders
price, and it is empty in the GUI literature (T2 §A.5, T1 §6).

---

## 4. What has to be true for the gate to be safe

No published method gives a distribution-free guarantee under arbitrary shift (T3 §0). The strongest bounds
degrade by a total-variation term that cannot be estimated from unlabelled deployment data (T3 §1 [2][6]).
Three failure modes matter most here.

- **Checkpoint and population change (M1).** Our own transfer measurement — 17.3% risk at one checkpoint,
  23.9% at the next under the same threshold — is a certificate failure, not a demonstrated risk change
  (T4 §4.6). The audit also notes the certificate is **val-side**: the test-side Clopper–Pearson upper bound is
  21.7%, so the 20% budget is not certified on test.
- **Clustered evaluation units (M2).** Rows are not exchangeable when they share episodes, users, or tasks;
  the effective sample size is the number of independent units, not rows (T3 §1 [11]). Our 398 test steps span
  350 episodes, which is why the paired tests are episode-clustered.
- **Silent coverage decay (M3).** Coverage can fall from 0.92 to 0.61 while accuracy stays within three points
  of baseline (T3 §0 [25]). A one-shot certificate would not see it.

What a deployer can defensibly do is not a certificate but a *contract plus an instrument*: a named
invariance assumption, an anytime-valid coverage monitor with a stated false-alarm rate, and a re-fit trigger
tied to model updates rather than the calendar (T3 §5–§6). The honest sentence is: **"certified per
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
| 25% "environment crash" | 2/8 per condition | checker exception on a live device (next episode ran 10 steps) | **misclassified, fixed** |

Everything in `benchmarks/results/2026-11-03-stage0/` re-derives byte-identically and `verify_paper.py` passes
69/69 — but that script checks stage-0 derived JSON only, not the pure-protocol row, the baselines, the pixel
distances, or any online number (T4 §2 R8).

**What this means for the value claim.** The calibration result survives audit; the headline accuracy is a
one-row drift from a tie; the online half is a harness finding, not a capability result. Nothing here
invalidates the gate argument, and two published sentences had to be withdrawn.

---

## 6. The pre-registered value test

The plan committed to delivering only if at least two of five conditions held. They are scored here.

1. **A peer-reviewed outcome-level result where abstention changes success or harm — HOLDS.** The
   mammography trial and RouteLLM (T1 §3.1, §3.3).
2. **A named, concrete user harm an act/ask/refuse gate prevents, with the cost of asking stated — HOLDS.**
   Wrong recipient, wrong purchase, destructive delete, agent-submitted forms, each with measured override and
   habituation costs (T2 §C.3).
3. **A reviewer-legible contribution shape our existing artifacts support without new training — HOLDS, with
   a condition.** The shape is "risk at coverage with trivial baselines and a cost ratio" (T2 §A.5); the
   artifacts support the risk and the baselines, and the cost ratio is a parameter we can sweep. What they do
   not yet support is *transfer across checkpoints* — the 17.3% → 23.9% failure is real.
4. **One next measurement with a predicted effect size, a cost, and a kill criterion — HOLDS.** §7.
5. **A documented reason the current headline cannot support a CCF-A claim yet — HOLDS.** No GUI outcome
   result exists, the accuracy is a tie, the certificate is per-checkpoint, and there is no cost model or ask
   rate in any GUI paper (T1 §6, T3 §6).

**Verdict: the brief passes (5/5).** The honest summary is that the *value* case is strong, the *outcome
evidence* for GUI agents does not yet exist, and the program's own contribution is a measurement plus a
certificate boundary, not a capability claim.

---

## 7. The one next measurement

**Report the first GUI selective-risk table with explicit cost ratios and both trivial baselines.**

Take the artifacts we already have — 398 test rows and 827 validation rows, with the type-routed gate and the
learned fusion — and report, for each policy, error per 1,000 decisions as a function of the ratio
`C_err / C_ask`:

- **always-act**: `1000 · (1 − acc) · C_err`
- **always-ask**: `1000 · C_ask`
- **gate at coverage c**: `1000 · (c · risk(c) · C_err + (1 − c) · C_ask)`

With the measured numbers (always-act error 36.2%; gate at 40.7% coverage, empirical risk 17.3%; test-side CP
upper 21.7%), the gate beats always-act when `C_err/C_ask > 2.03` (2.17 using the upper bound) and beats
always-ask when `C_err/C_ask < 5.78`. The learned fusion at 50% coverage and 19.1% risk moves the lower
break-even to ~1.9.

**Why this one.** It is the number T2 §A.5 says a reviewer will grade and T2 §B.6 says a builder will price;
it needs no GPU; and it converts a 64% tie into a decision-level statement, which is the only kind of claim an
accuracy tie can support.

**Prediction.** The gate dominates both trivial policies over a roughly 2–6× cost window on the current
checkpoint, and that window narrows to nothing on the second checkpoint if the transfer failure is
representative.

**Kill criterion.** If the gate does not dominate always-act at `C_err/C_ask ≥ 2` on *both* checkpoints, the
value claim fails for this checkpoint pair and the paper must say so — the certificate, not the cost argument,
is then the contribution.

**Cost.** Hours, CPU only, no model calls.

---

## 8. Open questions

1. **What is the real cost of an ask?** The literature measures override and habituation, not the price of an
   interruption in the product's own utility. The 40% maximum override rate is proposed without derivation
   (T2 §E).
2. **Does the cost argument survive a second checkpoint?** Currently unmeasured; our single transfer pair
   suggests it may not.
3. **Is there any GUI agent for which an outcome-level abstention result exists?** None found; the claim is
   "not found", not "does not exist" (T1 §6).
4. **Should the gate be per-action-type?** An irreversible tap and a scroll have different costs; a single
   threshold cannot express that, and no published gate does it (T1 §6 item 8).
5. **What would make us stop?** If the cost-window table is dominated by always-act at realistic ratios, the
   program should pivot from gating to the certificate boundary as the contribution — which is publishable
   and honest, and smaller.

---

## Sources

*(filled by the citation pass)*
