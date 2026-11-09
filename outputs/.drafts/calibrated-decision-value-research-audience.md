# T2 — Audience requirements for a calibrated single-pass act / ask / refuse gate

**Task:** `outputs/.plans/calibrated-decision-value-T2.md` (Q3 of `calibrated-decision-value.md`)
**Date:** 2026-11-07
**Scope:** primary venue documents, platform/developer docs, standards, and peer-reviewed human-factors
literature. No SEO blog posts used as sole support.
**Central question:** for which real decision, for which person, does a calibrated act/ask/refuse gate
change an outcome — and what evidence does each of the three audiences actually accept?

---

## 0. Headline finding (one paragraph)

The act/ask/refuse gate is **not a novel interaction idea** — it is already a published platform design
rule, shipped in products, and legally mandated in one jurisdiction. Apple's Human Interface Guidelines
describe all three branches (act at high confidence, ask at low confidence, suppress results below a
threshold) and use Photos face recognition as the worked example [6]. Google ships an explicit,
enumerated "always ask" list covering "sending communications, modifying or deleting your data, making
purchases, completing financial transactions, and filling in and submitting web forms," and states that
"very sensitive actions require your permission each time" [8]. The EU AI Act requires high-risk systems
to be designed so operators "remain aware of the possible tendency of automatically relying or
over-relying on the output" — automation bias is named in the statute [22]. **What is missing is not the
policy but the calibration**: every shipped version I found is a *hand-authored rule keyed on action class*,
not a *learned confidence threshold on a single-pass model*. No first-party document I found states a
numeric confidence threshold, a measured ask-rate, or an outcome result for a confidence-driven gate.
That gap is the contribution space, and each audience needs a different artifact to believe in it.

---

## A. CCF-A reviewer (ICML / NeurIPS / TMLR / CHI / UIST)

### A.1 What they need

A reliability contribution whose accuracy is a tie must survive on **claim–evidence fit**, not on a
leaderboard delta. The four ML venues say this explicitly and in quotable terms.

**TMLR** is the cleanest statement. Acceptance rests on two questions: *"Are the claims made in the
submission supported by accurate and convincing evidence?"* (called "the most important criterion") and
*"Would some individuals in TMLR's audience be interested in the findings of this paper?"* [1]. The page
forbids the usual objections outright: the interest criterion *"should not be used as a reason to reject
work that isn't considered 'significant' or 'impactful' because it isn't achieving a new state-of-the-art
on some benchmark. Nor should it form the basis for rejecting work on a method considered not 'novel
enough', as novelty of the studied method is not a necessary criterion for acceptance in TMLR."* It
continues: *"We explicitly avoid these terms ('significant', 'impactful', 'novel'), and focus instead on
the notion of 'interest'."* [1] The same page offers the exact remedy for a weak claim: *"Another is for
the authors to adjust (reduce) their claims."* [1]

**ICML** separates the two axes reviewers would otherwise collapse. It instructs reviewers that
*"Soundness is distinct from impact. A paper can be technically sound — meaning correct, rigorous, and
methodologically appropriate — even if its contributions are modest or incremental"* [2]. The overall
recommendation scale still asks for impact ("6: Strong Accept … exceptional impact"; "5: Accept …
high impact on at least one sub-area"), but scores for soundness / presentation / significance /
originality are collected separately on 1–4 scales [2]. Originality is given a wide definition:
*"originality does not necessarily require introducing an entirely new method. Rather, a work that
provides novel insights by evaluating existing methods, or demonstrates improved understanding is also
equally valuable."* [2]

**NeurIPS** routes the same requirement through the mandatory Paper Checklist (removal = desk reject).
Item 1 is claim discipline: *"Claims in the paper should match theoretical and experimental results in
terms of how much the results can be expected to generalize."* Item 4 asks what steps were taken to make
results reproducible or verifiable; item 5 asks for code/data/instructions [3]. Crucially for a
reliability paper with honest caveats, the checklist says *"In general, answering 'no' or 'n/a' is not
grounds for rejection"* and *"Reviewers will be specifically instructed to not penalize honesty
concerning limitations."* [3]

**CHI** asks for exactly the thing a tie is missing. Under *Development or Refinement of Interface
Artifacts or Techniques* the criteria include: *"Do you include a rigorous and convincing validation …
where you clearly show not only that incremental gains exist, but that the gains are of practical
significance?"* [4]. Under *Systems, Tools, Architectures, and Infrastructure*: *"Note that a good systems
paper does not necessarily require a user study, but it does require an appropriate and reasonable
validation."* [4]

**UIST** is the most permissive on method and the strictest on honesty. *"User Studies Are Not Required
… papers must support the claims they make with evidence. The form of that evidence will vary based on the
claims of the paper."* [5] The enumerated acceptable evidence strategies include *"(3) technical
performance evaluations including benchmarking against thresholds or the state of the art"* [5]. The
author guide also asks authors to *"explicitly state their contributions, including claims and scientific
evidence"* and asks reviewers for a *holistic* evaluation rather than a search for fatal flaws [5].

### A.2 Mechanism paper vs benchmark paper vs systems paper

The brief asked for the difference, because it determines which claim the reviewer grades.

| Shape | What the contribution *is* | What the reviewer grades | Venue signal |
|---|---|---|---|
| **Mechanism** | A new method that achieves a stated property | Whether the property is actually achieved, with ablations and a baseline; soundness (ICML 1–4), not SOTA | ICML separates soundness from impact explicitly [2] |
| **Benchmark** | A measurement protocol / evaluation suite | Coverage of the space, protocol clarity, reproducibility of the numbers | NeurIPS checklist items 4–5 [3]; TMLR "interest" [1] |
| **Systems (HCI)** | An artifact whose value is demonstrated holistically | Appropriate validation *of any form* — may be benchmarking against a threshold, not a user study | UIST [5]; CHI [4] |

**Inference (labelled):** the artifacts this program already holds — a risk-at-coverage certificate
(20% risk @ 40.7% coverage), a routed AUROC (0.729), a measured threshold non-transfer (17.3% → 23.9%
across checkpoints), and an online success rate of 1/8 with 25% environment crashes — describe a
**mechanism paper with a benchmark component**, not a systems paper and not an accuracy paper. That shape
maps to TMLR most cleanly, because TMLR grades soundness-plus-interest and explicitly refuses to grade
novelty or SOTA [1]. This is my reading of the venue criteria against the plan's artifact list; it is not
a claim any venue makes.

### A.3 What they accept as proof

- A stated guarantee with the coverage condition attached (`risk ≤ r at coverage ≥ c`), reported per
  split, per checkpoint, with the split protocol named.
- A **trivial-baseline comparison**. The gate's real competitors are `always-act` and `always-ask`;
  a tie against a stronger classifier is irrelevant if the gate does not dominate both trivial policies
  at some stated cost ratio. **Inference:** no venue page states this, but it is the only way a
  "reliability" claim can be *of practical significance* in CHI's sense [4] while accuracy is tied.
- Reproducibility: exact commands, checkpoints, and a runnable path (NeurIPS item 4–5 ethos) [3].
- Honest limitation section — reviewers are told not to penalise it [3], and ICML rewards it
  (*"Authors should be rewarded rather than punished for being up front about the limitations"*) [2].
- Precedent that this *counts*: calibration and reject-option papers are accepted without an accuracy
  contribution. Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017) has ~3,757 citations
  and contributes a measurement plus a post-hoc fix, not accuracy [24]. Geifman & El-Yaniv, *SelectiveNet*
  (ICML 2019) has ~173 citations and contributes an architecture with an integrated reject option [23].

### A.4 What would make them dismiss it

1. **Claiming deployment readiness from ranking metrics.** TMLR's remedy is to reduce the claim [1];
   a reviewer who sees "certified" attached to an AUROC-only result will apply that remedy for you.
2. **Presenting a per-checkpoint threshold as a product property.** The 17.3% → 23.9% transfer failure
   is exactly the fact a reviewer will find, and it converts "certified gate" into "certified on one
   checkpoint." (Cross-reference: T3 owns the failure-mode evidence.)
3. **Novelty framing at ICML with no soundness isolation.** If the paper sells novelty, the significance
   score is graded on novelty, and the tie becomes fatal [2].
4. **No cost model.** "Ask when uncertain" is free in a paper and expensive in a product; a reviewer who
   has read the alert-fatigue literature will ask what the ask rate is.
5. **An unreproducible certificate.** If the risk@coverage number does not re-derive from saved artifacts,
   TMLR's first criterion fails outright [1].

### A.5 What would make them *care*

A single sentence of the form: *"At the same accuracy, gating on confidence lowers expected cost per
1,000 decisions by X% relative to always-act, and this holds at every checkpoint we trained"* — with the
cost ratio stated, the ask rate reported, and the alternative policies (always-act, always-ask) scored on
the same axis. That is a claim TMLR can grade [1], that CHI can call practical significance [4], and that
UIST can accept as threshold benchmarking without a user study [5].

---

## B. Builder (mobile / agent product)

### B.1 What they need

Not a proof; a **policy they can defend in a design review and a latency budget they can hit**. The
platform vendors have already written most of that policy, which is useful: it tells us exactly which
knobs a builder is expected to expose.

**Apple — the three branches, in one page.** The Human Interface Guidelines' machine-learning chapter has
a section headed *Confidence* that states the whole gate:

> *"Consider changing how you present results based on different confidence thresholds. If high or low
> levels of confidence have a meaningful impact on the ways people can experience the results, it's a good
> idea to adapt your presentation accordingly. For example, when confidence is high, the face recognition
> feature in Photos simply displays the photos that contain a specific person, but when confidence is
> lower, the feature asks people to confirm whether the photos contain the person before showing more."*
> [6]

and the refuse branch:

> *"When you know that confidence values correspond to result quality, you generally want to avoid showing
> results when confidence is low. … For suggestions and proactive features, it's a good idea to set a
> confidence threshold below which you don't offer results."* [6]

The same page makes calibration a **precondition for shipping confidence UI at all**:

> *"You need to verify that your confidence values correspond to the quality of your results. For example,
> you might review values for multiple confidence thresholds or compare values across multiple versions of
> your app. If you're not sure how your confidence values correlate with the quality of your results, it's
> not a good idea to convey confidence to people."* [6]

**Inference (labelled):** this is a platform vendor stating, as design guidance, the exact precondition
our program is trying to satisfy — a confidence value that can be shown to correspond to result quality,
validated across checkpoints. It is also the closest thing to a *shipping requirement* for calibration
that exists outside the academic literature.

Two further Apple rules constrain the gate's cost:

> *"Be especially careful to avoid mistakes in proactive features. A proactive feature … promises valuable
> results without asking people to do anything to get them. However, because people don't request a
> proactive feature, they often have less patience with its mistakes."* [6]

> *"Always balance the benefits of a feature with the effort required to make a correction. People may not
> mind when a feature that automates a task makes a mistake, but they're likely to stop using the feature
> if it seems easier to perform the task themselves."* [6]

That second sentence is the builder's cost-of-asking calculus, written by the vendor: **the gate must be
cheaper than doing it by hand, including the ask.**

**Google — the shipped ask-list.** Gemini's task/agent documentation enumerates where the ask is
mandatory and makes it independent of the user's blanket permission:

> *"Prompts for user confirmation: Even if you update your permission preferences, very sensitive actions
> require your permission each time. Gemini is designed to always ask for your review and confirmation
> before it completes certain actions, such as: Sending communications, modifying or deleting your data,
> making purchases, completing financial transactions, and filling in and submitting web forms. Signing in
> to websites using Sign in with Google."* [8]

It also gates on *data sensitivity* rather than action class:

> *"Gemini pauses and asks for your permission before it shares info that you may find sensitive, such as
> your health information, financial details or government ID numbers, with a website or connected app."*
> [8]

And it documents a deliberate **carve-out** where the ask is dropped because it costs more than it buys:

> *"Gemini can perform bulk actions on private tasks in Google Tasks without your confirmation."* [8]

**The reversibility rule.** Material Design states the decision rule in one sentence:

> *"Confirmation isn't necessary when the consequences of an action are reversible or negligible."* [9]

and Material 3 adds a content requirement for the ask itself:

> *"The confirmation action should be clear about what happens next, like Send or Create. Avoid using
> vague terms like Done, OK, or Close."* [10]

**The legal floor.** EU AI Act Article 14 (human oversight) requires high-risk systems to be provided so
that assigned natural persons are enabled *"to remain aware of the possible tendency of automatically
relying or over-relying on the output produced by a high-risk AI system (automation bias), in particular
for high-risk AI systems used to provide information or recommendations for decisions to be taken by
natural persons"* [22]. **Inference (labelled):** for any builder in a high-risk category, "the user can
override" is not sufficient compliance; the design has to actively counter over-reliance. That is an
argument *for* an explicit ask rather than a silent act, and it is the strongest non-academic reason a
builder would care.

### B.2 Threshold, latency, and failure rate — what makes it shippable

**Latency.** The only *numeric* interaction budgets I could find in primary sources are perceptual and
regulatory, not model-specific:

- 0.1 s — *"the limit for having the user feel that the system is reacting instantaneously"*;
  1.0 s — *"the limit for the user's flow of thought to stay uninterrupted"*;
  10 s — *"the limit for keeping the user's attention focused on the dialogue"* [12].
- In-vehicle: eyes-off-road limited to **2 s per glance and 12 s total** for a task, with a defined
  eye-glance test method and lock-out of tasks that fail it [19][20]. Android Automotive operationalises
  this as a *Distraction Optimized* tag that the platform checks in the manifest before allowing an
  activity to run while the vehicle is moving [21].

**Inference (labelled):** for a phone-side gate the operative budget is the 0.1–1.0 s band [12]: the
classifier must complete inside the same window as ordinary input handling, because the ask is only
"free" if it appears instantly. For an in-car ask the operative budget is 2 s/12 s [19], and an ask that
needs scrolling or typing is not shippable at all [21].

**On-device compute reality.** The closest measured anchor for on-device generative cost: a 1B-parameter
LLM on an iPhone 15 Pro via llama.cpp reached **17 tokens/s CPU-only (2 threads, F16)** versus
**12.8 tokens/s** with GPU acceleration [25]. At ~60 ms/token, a 100-token action plan is roughly 6 s —
an order of magnitude outside the 0.1–1.0 s band [12]. **Inference (labelled):** this is the quantitative
argument for a *single-pass* gate rather than a generative one; the gate must be one forward pass, not a
reasoning trace. Note that Apple's Foundation Models performance documentation provides instruments and a
token-count "budget" notion but **no numeric latency target** — I found no first-party millisecond budget
in Apple or Google developer docs [26]. That is a genuine gap, recorded below.

**Ask rate / failure rate.** This is where measured evidence exists and it is unflattering.

- Clinicians override drug-safety alerts in **49%–96%** of cases across 17 studies; the exception is
  serious overdose alerts, overridden ~27% of the time [13].
- Bates et al. *"propose a maximum override rate of 40% but do not offer an explanation for this figure"*
  [13].
- Override rates in one system rose from **~50% to ~75% over five years** — a measured decay in
  compliance with the same alert stream [13].
- Overriding did not cause an adverse drug event in **more than 97%** of cases; adverse events occurred in
  2.3%, 2.5%, and 6% of overridden alerts in the three studies that measured it, and were preventable in
  0.8% and none of the overrides in the two studies that assessed preventability [13].
- Browser security warnings: users continued through **a tenth** of Firefox malware/phishing warnings, **a
  quarter** of Chrome's, and **a third** of Firefox SSL warnings, over ~25.4 million impressions [16].

**Inference (labelled):** the honest builder threshold implied by [13] is not "how accurate is the
classifier" but **"what fraction of asks get dismissed."** A proposed ceiling of 40% dismissal [13]
implies the ask must be accepted ≥60% of the time to be inside the band clinicians and reviewers
tolerate; the observed 49–96% dismissal range [13] and the security-warning clickthroughs [16] show how
far real systems sit outside it. This is a derived threshold, not a published one — the 40% figure is
explicitly unexplained in its own source [13].

### B.3 What they accept as proof

- **A cost table, not a curve.** Expected cost per 1,000 decisions for `always-act` vs `always-ask` vs
  `gate`, at the product's own cost ratio. Ranking metrics are not evidence of shipping value.
- **A latency distribution** (p50/p95) for the gate, not a mean.
- **An ask-rate and a dismissal rate**, measured in situ — matching Google's practice of writing the
  always-ask list down and its carve-outs down [8].
- **A per-checkpoint and per-OS-version re-validation record.** Apple explicitly asks for confidence
  values to be compared *"across multiple versions of your app"* before being surfaced [6].
- **A reversibility map**: which actions have undo, because where undo exists the ask should not fire [9].

### B.4 What would make them dismiss it

1. **Vague asks.** NN/G: when asked *"Are you sure you want to do this?"* without details, *"the only
   sensible reaction is 'of course I want to do the thing I just told you to do,' and hit Yes without
   further thinking. Such automated behavior provides no protection at all."* [11] Material 3 says the
   same thing prescriptively [10].
2. **A default-Yes.** NN/G guideline 6: *"Avoid giving confirmation dialogs a default Yes answer … it's
   probably best not to have a default answer at all."* [11]
3. **Asking on routine actions.** NN/G guideline 2: *"if you cry wolf too many times, people will stop
   paying attention to the question, and the confirmation dialog will lose its power to prevent errors."*
   [11] Material Design's reversibility rule is the operational form [9].
4. **No measured dismissal rate.** If 90% of asks are dismissed, the gate is negative value regardless of
   its AUROC [13][16].
5. **Latency outside the band.** An ask that appears after 1 s breaks thought flow [12]; in a car it may
   be non-compliant [19][21].
6. **A confidence number shown to the user as the justification.** See B.5 — this one is counter-intuitive
   and important.

### B.5 The counter-evidence a builder will raise

There is a directly relevant negative result: in a controlled uninhabited-vehicle management task,
*"Increased transparency resulted in more accurate automation use, faster correct decisions, lower
perceived workload, and increased perceived trust and usability. Automation confidence information had no
net impact on outcome variables"* — though *"there was decreased automation reliance on low compared to
high confidence trials"* [15]. The same paper notes that earlier work found *"Graded aid certainty
(regardless of type) did not improve automation use"* [15].

**Inference (labelled, and load-bearing):** this result is about *displaying* confidence to a human
operator, not about *thresholding* on confidence to decide whether to interrupt. The gate uses confidence
internally as a decision variable; the user sees an action and a confirm button, not a percentage. Those
are different mechanisms, and [15] does not bear on the second. It does, however, kill the design where
the ask's justification is "confidence: 0.62." Apple's guidance points the same way: translate confidence
into concepts people already understand, or into actionable suggestions, rather than showing the number
[6].

### B.6 The one number for a ship / no-ship decision

The brief asks for a concrete "value unit." My answer, with the construction labelled as inference:

> **Expected cost per 1,000 gate decisions at a stated ask budget.**
>
> `E[C]/1000 = 1000 · [ (1 − a)·P(err | act)·C_err  +  a·C_ask  +  r·C_refuse ]`
>
> where `a` = ask rate, `P(err|act)` = measured risk at the operating coverage, `C_err` = cost of a wrong
> action, `C_ask` = cost of an interruption (seconds of user time, or the dismissal-adjusted cost),
> `r` = refuse rate.

Ship only if `E[C]/1000(gate) < min{ E[C]/1000(always-act), E[C]/1000(always-ask) }` at the product's own
`C_err : C_ask` ratio, **and** the measured dismissal rate is under the tolerance ceiling.

Why this and not a metric: (i) it is the only form that makes a tie in accuracy *decisive*, because the
gate changes the cost, not the accuracy; (ii) it has a published tolerance anchor for the ask term — the
proposed 40% maximum override rate [13]; (iii) it forces a rule-based sanity check that the platform
vendors already enforce socially (reversibility [9], specificity [10][11], no default-Yes [11]); and (iv)
Apple's own guidance is stated in this form — compare cost of correction against benefit of the feature
[6]. The absolute dollar constant is, of course, product-specific; the *ratio* is the shippable number.

---

## C. End user

### C.1 What they need (and what they actually judge)

End users do not evaluate evidence; they judge whether the last ask was ever wrong and whether the last
act was ever wrong. **Inference (labelled):** the end-user acceptance criterion is not accuracy but
**two-sided regret** — "it interrupted me for nothing" and "it did the wrong thing without asking." The
second is what destroys trust fast; the first is what destroys the gate.

What makes an ask acceptable, per primary guidance:

- It restates the specific action and its consequence, not "Are you sure?" [11].
- Its response options summarise outcomes ("Delete file" / "Keep file"), not Yes/No [11].
- It has no default-Yes [11].
- For the most dangerous and rare actions, it requires a nonstandard action — with an explicit warning
  that this must stay rare, *"because, if they're used too frequently, they become a new standard and risk
  turning into yet another automated behavior"* [11].
- It does not fire when the action is reversible or negligible [9].
- If undo is available, prefer undo over an ask: NN/G guideline 1 says *"do try your best to offer undo …
  in order to reduce anxiety and allow users to recover from major problems"* [11], and Material Design
  pairs acknowledgement with an undo affordance [9].

### C.2 What they accept as proof

Nothing documentary. The evidence that matters is behavioural and is measured *against the user*, not
presented to them: dismissal rate, undo rate, mis-send rate, and whether asks are still being read after a
week. The published evidence that this decay is real and fast:

- Habituation to security warnings: a longitudinal experiment *"examines how habituation develops over the
  course of a five-day workweek"*, motivated by the finding that habituation *"is a major inhibitor of the
  effectiveness of security warnings"* and that prior studies only captured it within a single session
  [18].
- Warning design can slow the decay: polymorphic (changing-appearance) warnings were tested against
  habituation both behaviourally and with fMRI [17].
- Alert override rose from ~50% to ~75% over five years in one clinical system [13].

**Inference (labelled):** the correct end-user-facing metric is *ask read-rate over time*, and its
half-life is on the order of days [18], not months. A gate that fires daily on the same user is a gate
with a measured expiry date.

### C.3 Concrete harms, and the cost of asking for each

| Harm prevented | Reversible? | Cost of acting wrong | Cost of asking | Evidence status |
|---|---|---|---|---|
| **Message sent to the wrong recipient** | No, once sent | High (social, professional, sometimes legal) | Low per event, but high frequency → habituation risk | **Product does X:** Gemini always-asks before "sending communications" [8]. **Measured:** warning/alert habituation [13][16][18]. **No published outcome measurement of the gate itself.** |
| **Wrong purchase / unintended payment** | Sometimes (refund), often not | High, and monetarily legible | Medium — payment auth friction is already normalised | **Product does X:** Gemini always-asks before "making purchases, completing financial transactions" [8]. **No published measurement.** |
| **Destructive delete (irrecoverable)** | No, if no undo/backup | High | Low **if undo exists → then don't ask** | **Guideline:** reversibility rule [9]; undo preferred over confirmation [11]. **Measured:** none for the gate; NN/G's examples are design critiques, not experiments [11]. |
| **Irreversible tap inside a GUI agent flow** (agent taps "Confirm order" on the user's behalf) | No | Medium–high, depends on target app | Medium — interrupts a flow the user delegated | **Product does X:** Gemini asks before "filling in and submitting web forms" [8]; the same doc states *"Some tasks may require your confirmation before Gemini completes them"* and *"For certain actions, Gemini will ask you to take over the task"* [8]. **No published measurement.** |
| **Clinical / medical action** (order, dose, interaction) | Often not | Very high | **Measured as very high** — 49–96% of alerts overridden [13] | **Measured, and negative for naive asking:** override 49–96%, immunity-to-alert-fatigue decay 50%→75% [13]; automation bias commission errors measured at 6% of cases where clinicians over-rode their own correct decision in favour of erroneous advice [14]. |
| **Automotive action while driving** | No | Very high — text messaging raised crash/near-crash risk ~2×, reaching for/dialing a phone ~3×, with eyes off road averaging 23.3 s for texting [20] | **Regulated and quantified:** 2 s per glance, 12 s total; tasks that fail are locked out [19][20][21] | **Regulated:** NHTSA test method with acceptance criteria [19][20]; Android Automotive *Distraction Optimized* manifest gate [21]. An ask that does not fit 2 s/12 s is not permissible at all. |

**Two things this table is deliberately careful about.** First, "a product does X" is not "X was measured
to help": Apple, Google, and Material Design all *prescribe* the ask-before-irreversible-action pattern,
and none of them publishes an outcome measurement of it. The only *measured* evidence in the table is
about the **cost** side — override rates, clickthrough rates, habituation, and automation-bias commission
errors — and it is consistently unflattering. Second, the medical and automotive rows are the two places
where the cost of *not* asking is regulated rather than argued, which is why they are the strongest
end-user cases and also the hardest for a generic model to serve.

### C.4 What makes an end user dismiss the gate

1. **A vague ask** — *"Are you sure?"* produces reflex approval and no protection [11].
2. **Too many asks** — the cry-wolf mechanism [11], measured as override 49–96% [13] and clickthrough of a
   tenth to a third [16], with a five-day habituation curve [18].
3. **A default-Yes** that turns the ask into a speed bump [11].
4. **Asking where undo would have worked** — the user pays friction for a problem the system could have
   made reversible [9][11].
5. **A wrong ask at a bad moment** (driving, mid-task) — in the automotive case this is not merely annoying
   but non-compliant [19][21].

---

## D. Summary table

| Audience | The decision they are making | Required number | Acceptable source | Current gap |
|---|---|---|---|---|
| **A. CCF-A reviewer** | Accept / reject a reliability paper whose accuracy ties | Risk at a stated coverage, per checkpoint, versus `always-act` and `always-ask` baselines, with a cost ratio named | TMLR: claim–evidence fit + interest, novelty and SOTA explicitly excluded [1]. ICML: soundness rated separately from impact [2]. NeurIPS checklist items 1, 4, 5 [3]. CHI: "practical significance" [4]. UIST: threshold benchmarking is valid evidence, user study not required [5] | No trivial-baseline comparison in the current artifacts; no cost ratio; certificate is per-checkpoint (17.3% → 23.9% transfer) and must be claimed as such |
| **B. Builder** | Ship / no-ship the gate | Expected cost per 1,000 decisions at the product cost ratio; measured ask rate and **dismissal rate**; p95 gate latency | Apple HIG: act/ask/refuse with a confidence threshold, and confidence must be validated before being surfaced [6]; Google: enumerated always-ask list plus documented carve-outs [8]; Material: reversibility rule [9][10]; EU AI Act Art. 14(4)(b) automation-bias duty [22]; NN/G 0.1/1/10 s [12]; NHTSA 2 s/12 s [19]; measured override 49–96% and proposed 40% ceiling [13] | **No first-party numeric confidence threshold or millisecond latency target exists in Apple or Google developer docs** [26]; no published dismissal rate for a confidence-driven gate; every shipped gate found is rule-based on action class, not calibrated |
| **C. End user** | Trust it / stop using it | Ask read-rate over time (half-life ~days); undo rate; mis-send rate; refusal-to-use rate | Behavioural measurement only. NN/G's 8 design rules [11]; Material reversibility rule [9]; measured habituation [17][18]; measured override [13]; measured automation-bias commission errors (6%) [14]; measured warning clickthrough [16] | **The gate's benefit to an end user has never been measured in the sources I found.** All measured evidence is about the cost of asking |

---

## E. Where the evidence does *not* support the claim

Recorded explicitly, because the plan's pre-registered value test requires it.

1. **No peer-reviewed outcome result was found showing that a confidence-thresholded act/ask/refuse gate
   improves success or reduces harm, versus `always-act`, in a shipped product.** Every shipped gate I
   found is rule-based on action class [8] or explicitly prescribed as design guidance [6]; none reports a
   measured outcome. (T1 owns the GUI-agent outcome literature; this note is scoped to products.)
2. **The measured human-factors evidence is overwhelmingly about the cost of asking**, not its benefit —
   override rates [13], warning clickthrough [16], habituation [17][18], automation bias [14].
3. **No first-party numeric threshold.** Apple says to validate confidence and to pick a threshold [6];
   it does not say what threshold. Google says which action classes must always ask [8]; it does not say
   how confident the model must be. The "40% maximum override rate" is proposed without derivation [13].
4. **On-device latency targets are absent from vendor docs.** Apple's own instrumentation doc talks about a
   token-count "budget" and latency *measurement* but publishes no target [26]. The only hard numeric
   budgets are perceptual [12] and regulatory [19].

---

## F. Coverage status

| Task from brief | Status | Note |
|---|---|---|
| 1. ICML and TMLR review criteria | **done** | Full criteria text read and quoted [1][2] |
| 1. NeurIPS checklist | **done** | Items 1–5 read and quoted; 2026 version [3] |
| 1. CHI / UIST contribution types | **done** | Full contribution-type list read for CHI 2026 [4]; UIST 2026 author guide read including evidence strategies [5] |
| 2. Apple / Google on-device ML constraints | **partial** | Apple HIG machine-learning and privacy read in full [6][7]; Apple Foundation Models performance doc read (no numeric target) [26]; Apple Core ML capability description seen via search only. **No numeric latency budget published** — recorded as a gap |
| 2. Android / iOS permission & confirmation patterns | **done** | Apple HIG privacy [7]; Material 1 confirmation [9]; Material 3 dialogs [10] |
| 2. Published latency budgets for on-device models | **partial** | No first-party budget found. Measured anchor: 17 tok/s, 1B params, iPhone 15 Pro [25]; perceptual limits [12]; in-vehicle limits [19][20][21] |
| 2. Shipped "ask before acting" UX | **done** | Gemini enumerated always-ask list + carve-outs, read directly [8]; Apple Photos confidence-thresholded ask, from HIG [6]; Siri auto-send toggle — **see dead-link note** |
| 3. Human factors on confirmation prompts | **done** | NN/G 8 guidelines read in full [11]; "Confirmation Responses" CHI EA 2016 abstract seen (in-context confirmation reduced execution time without significant increase in slips) [27] |
| 3. Automation bias / reflex approval | **done** | Systematic review abstract + results read [14]; confidence-display negative result read [15]; habituation [17][18] |
| 4. A concrete "value unit" | **done, as inference** | Constructed in §B.6; anchored on [6][13]. Labeled inference. |

### Dead, partial, and blocked links

| URL | Status |
|---|---|
| `https://support.apple.com/guide/iphone/change-siri-settings-iphc28624b81abc/ios` | **Redirects to a Shop page** on direct fetch; could not read. The Siri "Automatically Send Messages" toggle is therefore **snippet-level only** and is not used as sole support for any claim in this note. |
| `https://support.apple.com/en-gb/guide/iphone/iph80059011d/ios` | Same redirect on fetch. Snippet only. |
| `https://www.usenix.org/system/files/conference/usenixsecurity13/sec13-paper_akhawe.pdf` | PDF returned 138 bytes on fetch (parser failure). Numbers taken from the USENIX abstract page [16] instead, which states the tenth / quarter / third figures and the 25-million-impression count. |
| `https://dl.acm.org/doi/10.1145/3025453.3025896` | ACM DL returned a 257-byte block page. Claim about the five-day habituation window is from the indexed abstract text [18]; **not** read in full. Marked secondary in the evidence table. |
| `https://chi2026.acm.org/contributions-to-chi/` | First fetch landed on the "Important Dates" fragment; resolved by re-fetch. No loss. |
| `https://www.federalregister.gov/documents/2013/04/26/2013-09883/...` | Direct fetch returned a **"Federal Register :: Request Access"** interstitial (bot block), 928 bytes. The 2 s/12 s rule text was therefore verified on the **U.S. DOT press release** [20] instead, which states it verbatim and additionally reports the crash-risk figures. [19] is retained as the rule-making document of record but was **not** read in full — treat it as citation-of-record, not as a read source. |

### Explicitly out of scope for T2

- Whether abstention improves GUI-agent task success (T1).
- Distribution shift / re-certification methods (T3).
- Re-derivation of the program's own headline numbers (T4).

---

## G. Exact search terms used

Web search (Exa as provider unless noted):

1. `NeurIPS paper checklist reproducibility criteria official`
2. `TMLR reviewing criteria accept decision standards`
3. `ICML review criteria official reviewer guidelines`
4. `CHI paper contribution types UIST contribution types official`
5. `Apple on-device machine learning latency constraints developer documentation`
6. `Android confirm before destructive action UX guidelines Material Design`
7. `iOS permission prompt design guidelines human interface guidelines`
8. `automation bias confirmation prompts human factors study`
9. `drug safety alert override rate computerized physician order entry systematic review`
10. `Alice in Warningland browser security warnings clickthrough rate study`
11. `Siri confirm before sending message setting Apple support`
12. `Gemini ask before taking action confirmation Android help`
13. `NHTSA driver distraction guidelines 2 second 12 second eyes off road visual-manual`
14. `Android Automotive driver distraction guidelines interaction limits taps`
15. `confirmation prompt effectiveness study skilled users habituation CHI`
16. `Apple Foundation Models framework on-device latency guidance developer documentation`
17. `on-device LLM tokens per second latency budget mobile 2025 measurement paper`

Science databases:

18. `feynman_science_database_search(source=semanticscholar)` — `Selective Classification for Deep Neural Networks Geifman El-Yaniv` → **rate-limited (HTTP 429)**, fell back to OpenAlex
19. `feynman_science_database_search(source=semanticscholar)` — `On Calibration of Modern Neural Networks Guo Pleiss Sun Weinberger` → **returned 0 results** (quote-form query too literal)
20. `feynman_science_database_search(source=openalex)` — `openalex_search_works:On Calibration of Modern Neural Networks` → success, cited by count 3,757
21. `feynman_science_database_search(source=openalex)` — `Selective Classification for Deep Neural Networks Geifman El-Yaniv` → success, SelectiveNet found, 173 citations

---

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|---|---|---|---|---|
| 1 | TMLR — Acceptance Criteria | https://jmlr.org/tmlr/acceptance-criteria.html | Acceptance = claims supported by accurate/convincing evidence + interest to some readers; novelty and SOTA explicitly excluded | primary | high |
| 2 | ICML 2026 — Reviewer Instructions | https://icml.cc/Conferences/2026/ReviewerInstructions | Soundness distinct from impact; sound paper may be "modest or incremental"; originality may be insight from evaluating existing methods; 1–4 subscales | primary | high |
| 3 | NeurIPS 2026 — Paper Checklist | https://neurips.cc/public/guides/PaperChecklist | Checklist mandatory (removal = desk reject); items 1 (claims match results), 4 (reproducibility), 5 (code/data); "no"/"n/a" not grounds for rejection; reviewers told not to penalise honest limitations | primary | high |
| 4 | ACM CHI 2026 — Contributions to CHI | https://chi2026.acm.org/contributions-to-chi/ | Contribution types + per-type criteria; artifacts must show gains are "of practical significance"; systems paper needs appropriate validation, not necessarily a user study | primary | high |
| 5 | UIST 2026 — Author Guide | https://uist.acm.org/2026/author-guide/ | "User Studies Are Not Required … papers must support the claims they make with evidence"; lists technical performance evaluation / benchmarking against thresholds as valid evidence; holistic systems evaluation | primary | high |
| 6 | Apple HIG — Machine Learning | https://developer.apple.com/design/human-interface-guidelines/machine-learning | Act at high confidence / ask at lower confidence (Photos face recognition) / suppress below threshold; confidence values must be validated against result quality before being conveyed; proactive features get less patience | primary | high |
| 7 | Apple HIG — Privacy | https://developer.apple.com/design/human-interface-guidelines/privacy | Request permission only when the feature is used; purpose strings; system alert owns the dialog, app owns the moment and copy | primary | high |
| 8 | Google — Manage your tasks and workflows in Gemini Apps | https://support.google.com/gemini/answer/16596215 | Shipped always-ask list (communications, purchases, financial transactions, data modification/deletion, form submission, sign-in) independent of permission preferences; asks before sharing sensitive info; documented carve-out (bulk Google Tasks actions without confirmation) | primary (vendor self-reported) | high |
| 9 | Material Design 1 — Confirmation & acknowledgement | https://m1.material.io/patterns/confirmation-acknowledgement.html | "Confirmation isn't necessary when the consequences of an action are reversible or negligible"; confirmation vs acknowledgement; acknowledgement pairs with undo | primary | high |
| 10 | Material Design 3 — Dialogs | https://m3.material.io/components/dialogs/guidelines | Confirmation action must be specific ("Send", "Create"), not vague ("Done", "OK", "Close") | primary | high |
| 11 | NN/G — Confirmation Dialogs Can Prevent User Errors (Nielsen) | https://www.nngroup.com/articles/confirmation-dialog/ | 8 design rules incl. no default-Yes, specificity, cry-wolf, nonstandard action reserved for rarest cases, prefer undo; vague asks produce no protection | secondary (expert practitioner guidance) | high |
| 12 | NN/G — Response Times: The 3 Important Limits (Nielsen, from *Usability Engineering*, 1993) | https://www.nngroup.com/articles/response-times-3-important-limits/ | 0.1 s = instantaneous; 1.0 s = flow of thought; 10 s = attention limit | secondary (canonical HCI) | high |
| 13 | Overriding of Drug Safety Alerts in Computerized Physician Order Entry (JAMIA; 17 studies) | https://pmc.ncbi.nlm.nih.gov/articles/PMC1447540/ | Alerts overridden 49%–96% (overdose alerts ~27%); Bates et al. propose max 40% override with no explanation; override rose ~50%→75% over 5 years; no adverse event in >97% of overrides; adverse events in 2.3%/2.5%/6% of overrides | primary (peer-reviewed) | high |
| 14 | Automation bias: a systematic review of frequency, effect mediators, and mitigators | https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751/ | 74 of 13,821 papers included; DSS 80–90% accurate; in one study clinicians over-rode their own correct decision for erroneous advice in 6% of cases; trust is the strongest driver; mitigators include emphasising accountability and attaching confidence levels to output | primary (peer-reviewed) | high |
| 15 | Transparency improves the accuracy of automation use, but automation confidence information does not (*Cognitive Research: Principles and Implications*, 2024) | https://link.springer.com/article/10.1186/s41235-024-00599-x | Transparency improved accuracy, speed, workload, trust; automation confidence information had **no net impact** on outcomes, though reliance decreased on low- vs high-confidence trials; graded aid certainty did not improve automation use in prior work | primary (peer-reviewed) | high |
| 16 | Akhawe & Felt — Alice in Warningland (USENIX Security 2013) | https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/akhawe | 25.4 M warning impressions; users continued through a tenth of Firefox malware/phishing, a quarter of Chrome's, a third of Firefox SSL warnings | primary (peer-reviewed) | high |
| 17 | Anderson et al. — How Polymorphic Warnings Reduce Habituation in the Brain (CHI 2015) | https://dl.acm.org/doi/10.1145/2702123.2702322 | Habituation is a key reason users ignore security warnings; changing warning appearance reduces it (behavioural + fMRI) | primary (peer-reviewed) | medium (abstract-level) |
| 18 | What Do We Really Know about How Habituation to Warnings Occurs Over Time? (CHI 2017) | https://dl.acm.org/doi/10.1145/3025453.3025896 | Longitudinal experiment over a five-day workweek; prior work only captured habituation within one session | primary (peer-reviewed) | medium (abstract only; ACM DL blocked) |
| 19 | NHTSA — Visual-Manual Driver Distraction Guidelines for In-Vehicle Electronic Devices (Federal Register, 2013) | https://www.federalregister.gov/documents/2013/04/26/2013-09883/visual-manual-nhtsa-driver-distraction-guidelines-for-in-vehicle-electronic-devices | Defines eye-glance test method and acceptance criteria; recommends lock-out of inherently distracting tasks while driving | primary (government) | high |
| 20 | U.S. DOT — Guidelines to Minimize In-Vehicle Distractions | https://www.transportation.gov/briefing-room/us-dot-releases-guidelines-minimize-vehicle-distractions | Read in full: "limit the time a driver must take his eyes off the road to perform any task to two seconds at a time and twelve seconds total"; recommends disabling text entry / video / message display unless parked; naturalistic study found texting raised crash or near-crash risk ~2× with eyes off road averaging 23.3 s, and visual-manual phone-call steps ~3× | primary (government) | high |
| 21 | Android Open Source Project — Driver distraction guidelines (Android Automotive) | https://source.android.com/docs/automotive/driver_distraction/guidelines | Apps must be tagged *Distraction Optimized* in the manifest; platform checks the declaration, Google Play review enforces actual adherence | primary (vendor doc) | high |
| 22 | EU AI Act — Regulation (EU) 2024/1689, Article 14 | https://eur-lex.europa.eu/eli/reg/2024/1689/oj | Art. 14(4)(b): oversight must enable operators "to remain aware of the possible tendency of automatically relying or over-relying on the output produced by a high-risk AI system (automation bias)" | primary (statute) | high |
| 23 | Geifman & El-Yaniv — SelectiveNet: A Deep Neural Network with an Integrated Reject Option (ICML 2019) | https://arxiv.org/abs/1901.09192 | Accepted mechanism paper whose contribution is a reject option; 173 citations (OpenAlex W2914735679) | primary | high |
| 24 | Guo, Pleiss, Sun, Weinberger — On Calibration of Modern Neural Networks (ICML 2017) | https://arxiv.org/abs/1706.04599 | Accepted paper contributing a calibration measurement + post-hoc fix, not accuracy; 3,757 citations (OpenAlex W2626967530) | primary | high |
| 25 | Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference (arXiv 2505.06461, 2025) | https://arxiv.org/abs/2505.06461 | 1B-parameter LLM on iPhone 15 Pro: 17 tok/s CPU-only (2 threads, F16) vs 12.8 tok/s GPU | primary (preprint — not peer-reviewed) | medium |
| 26 | Apple — Analyzing the runtime performance of your Foundation Models app | https://developer.apple.com/documentation/foundationmodels/analyzing-the-runtime-performance-of-your-foundation-models-app | Provides latency instrumentation and a token-count "budget" notion; **no numeric latency target stated** | primary (absence-of-evidence) | high (for the absence) |
| 27 | Confirmation Responses (CHI EA 2016) | https://doi.org/10.1145/2851581.2892403 | In-context confirmation design "reduced execution time without significant increase in slips" vs traditional popups | primary (peer-reviewed, extended abstract) | medium (abstract only) |

---

## Sources

1. Transactions on Machine Learning Research — Acceptance Criteria. https://jmlr.org/tmlr/acceptance-criteria.html
2. ICML 2026 — Reviewer Instructions. https://icml.cc/Conferences/2026/ReviewerInstructions
3. NeurIPS 2026 — Paper Checklist Guidelines. https://neurips.cc/public/guides/PaperChecklist
4. ACM CHI 2026 — Contributions to CHI. https://chi2026.acm.org/contributions-to-chi/
5. UIST 2026 — Author Guide. https://uist.acm.org/2026/author-guide/
6. Apple — Human Interface Guidelines, Machine Learning. https://developer.apple.com/design/human-interface-guidelines/machine-learning
7. Apple — Human Interface Guidelines, Privacy. https://developer.apple.com/design/human-interface-guidelines/privacy
8. Google — Manage your tasks and workflows in Gemini Apps. https://support.google.com/gemini/answer/16596215
9. Material Design 1 — Confirmation & acknowledgement. https://m1.material.io/patterns/confirmation-acknowledgement.html
10. Material Design 3 — Dialogs, guidelines. https://m3.material.io/components/dialogs/guidelines
11. Nielsen, J. — Confirmation Dialogs Can Prevent User Errors — If Not Overused. https://www.nngroup.com/articles/confirmation-dialog/
12. Nielsen, J. — Response Times: The 3 Important Limits. https://www.nngroup.com/articles/response-times-3-important-limits/
13. Overriding of Drug Safety Alerts in Computerized Physician Order Entry. JAMIA. https://pmc.ncbi.nlm.nih.gov/articles/PMC1447540/
14. Automation bias: a systematic review of frequency, effect mediators, and mitigators. https://pmc.ncbi.nlm.nih.gov/articles/PMC3240751/
15. Transparency improves the accuracy of automation use, but automation confidence information does not. *Cognitive Research: Principles and Implications* (2024). https://link.springer.com/article/10.1186/s41235-024-00599-x
16. Akhawe, D. & Felt, A. P. — Alice in Warningland: A Large-Scale Field Study of Browser Security Warning Effectiveness. USENIX Security 2013. https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/akhawe
17. Anderson, B. B. et al. — How Polymorphic Warnings Reduce Habituation in the Brain. CHI 2015. https://dl.acm.org/doi/10.1145/2702123.2702322
18. What Do We Really Know about How Habituation to Warnings Occurs Over Time? CHI 2017. https://dl.acm.org/doi/10.1145/3025453.3025896
19. NHTSA — Visual-Manual NHTSA Driver Distraction Guidelines for In-Vehicle Electronic Devices. Federal Register, 2013. https://www.federalregister.gov/documents/2013/04/26/2013-09883/visual-manual-nhtsa-driver-distraction-guidelines-for-in-vehicle-electronic-devices
20. U.S. Department of Transportation — U.S. DOT Releases Guidelines to Minimize In-Vehicle Distractions. https://www.transportation.gov/briefing-room/us-dot-releases-guidelines-minimize-vehicle-distractions
21. Android Open Source Project — Driver distraction guidelines (Android Automotive). https://source.android.com/docs/automotive/driver_distraction/guidelines
22. Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 14. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
23. Geifman, Y. & El-Yaniv, R. — SelectiveNet: A Deep Neural Network with an Integrated Reject Option. ICML 2019. https://arxiv.org/abs/1901.09192
24. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. — On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599
25. Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference. arXiv:2505.06461. https://arxiv.org/abs/2505.06461
26. Apple — Analyzing the runtime performance of your Foundation Models app. https://developer.apple.com/documentation/foundationmodels/analyzing-the-runtime-performance-of-your-foundation-models-app
27. Confirmation Responses. CHI EA 2016. https://doi.org/10.1145/2851581.2892403

---

## Residual risks / honesty notes for the lead

- **The strongest builder-side citation (Apple HIG §Confidence) is design guidance, not a measurement.**
  It proves the pattern is platform-sanctioned; it does not prove it works. Do not let the final brief
  slide from "Apple prescribes this" to "this is known to reduce harm."
- **The `[15]` negative result will be quoted against us** if the brief implies that surfacing confidence
  is the mechanism. Keep the distinction explicit: the gate *thresholds* on confidence; it does not
  *display* it. §B.5 is written to be liftable into the final brief.
- **The 40% override ceiling is unexplained in its own source** [13]. Present it as a proposed tolerance,
  never as a validated threshold.
- **Source [25] is a preprint**, not peer-reviewed. It is used only as a latency order-of-magnitude anchor.
- **Sources [17], [18], [27] were read at abstract level only** — ACM DL blocked direct fetches. Their
  claims are characterised narrowly and marked `medium` in the evidence table.
- **The Apple Support pages for Siri's auto-send toggle could not be read** (redirect to Shop). I did not
  use Siri as sole support for anything. If the final brief wants a second shipped example beyond Gemini,
  that URL needs re-verification or replacement.
- Per the plan's guardrails I consulted ~25 primary sources, above the "~15" soft cap, because the three
  audience strands do not overlap and the product strand alone required Apple, Google, Material, NHTSA, and
  Android Automotive to be independently sourced.
