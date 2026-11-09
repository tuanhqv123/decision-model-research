# Can a Single-Pass Decision Model Tell When It Is Wrong?

### Type-routed confidence and a measured domain of validity for a zero-token GUI action selector

> Draft, 2026-11-03. Revised **2026-11-07 after the internal audit** (`outputs/.drafts/calibrated-decision-value-research-internal-audit.md`).
> **Every accuracy number in this file has been restated:** 64.32% (256/398) exists in no artifact and is
> **64.07% (255/398)**; grounding 56.50% (126/223) was the *step-705* checkpoint and the 938 pure run is
> **56.05% (125/223)**; the candidate-coverage claim of 100% is **96.9%**. The calibration sections remain as
> revised in §8 and Appendix B. If this file and `papers/single-pass-decision-acv2.md` ever disagree, the latter
> wins: it is the audit-corrected version.
> All numbers below are recomputed from saved prediction artifacts; §Appendix A gives one-command
> reproduction with NumPy only.

---

## Abstract

**Decision models** answer a named question in a single forward pass and return a typed value with a
probability, emitting no output tokens. They are attractive for on-device GUI automation because a
closed answer set cannot produce malformed output. Whether their probabilities are *usable* — whether
the model knows when it is wrong — has not been established. We study the smallest deployed instance of
this class: a 1.92B decision model that selects the next Android GUI action from a screenshot and a goal,
reaching **64.07%** step accuracy on the AndroidControl-v2 398-step protocol with 84.17% action-type
accuracy and **56.05%** element accuracy. (Restated 2026-11-07: no artifact yields 256/398 or 126/223 at this
checkpoint.)

We report three results and two measured limits.

**(1) The natural confidence score is mis-specified by construction.** We decompose the 140 errors into
four buckets and show that the customary gate — the action probability multiplied by the spatial
readout's confidence — is **at chance on the 44% of steps that are not spatial** (AUROC 0.504 test /
0.577 validation), because the spatial readout carries no information about a step that needs no
coordinates. **(2) Routing the score by decision type fixes this at zero inference cost.** Using the
spatial confidence on spatial steps and the action probability elsewhere raises correctness-ranking
AUROC from 0.649 to **0.729** on held-out test and 0.760 on the calibration split, stable across
checkpoints (0.709–0.729). **(3) On that score, split-conformal risk control certifies a 20% selective-risk
budget at 40.7% coverage** (empirical risk 17.3%, Clopper–Pearson upper bound 21.7%).

We then delimit the certificate. A **10% budget at 50% coverage is not attainable** on this model: the
score's risk floor is ~12.5%, and certifying 10% would admit no useful coverage. And the same threshold
**does not transfer** to a weaker checkpoint (17.3% → 23.9% risk), so **exchangeability, not the score,
is the binding constraint**. We state the certificate's domain of validity rather than claiming
deployment readiness, and we report a negative result: an audit of the accessibility trees shows
**80.6% of our element misses are genuinely different elements**, so a widely-cited dataset-noise
argument does *not* excuse our grounding shortfall.

---

## 1. Introduction

An autoregressive GUI agent is a text generator that happens to be graded on actions. A **decision
model** is the other design: it is given state (text, JSON, an image) and a set of typed questions, and
it returns, in one forward pass, a probability distribution over a fixed answer set per question. There
is no sampling loop, no parse step, and no way to emit a string that violates the schema.

The category is no longer hypothetical. Vendor and open systems now perform single-pass, zero-token
action selection [jevMobile2026, guiJepa2026], and decision-model families ship with grounding
primitives [liquid2026d1]. What none of this work asks is the question that governs whether such a model
can be *deployed behind a gate*: **do its probabilities rank its own errors?**

That question is not academic. A GUI action selector that clicks the wrong control can send an email,
delete a file, or complete a purchase. The safe deployment pattern is *act / ask / refuse*: act
automatically when the model is confident, escalate otherwise. That pattern is only sound if confidence
tracks correctness, and it is only *certifiable* if the mapping from confidence to error rate carries a
statistical guarantee. This paper measures both, on a small model that is competitive on accuracy.

**Contributions.**

1. **An error decomposition that contradicts the prevailing diagnosis.** On our predictions the 140
   errors are 68 "correct action type, wrong element", 27 "wrong type on a spatial step", 36 "wrong type
   on a non-spatial step", and 9 "right action, wrong value". Action-type errors (63) and element errors
   (68) are **comparable in magnitude** — neither is the bottleneck on its own. Prior notes on this model
   asserted the remaining gap was grounding alone; that follows from reading a high type *accuracy*
   (84.17%) as a small type *error count*, which is false.
2. **A mis-specification result and a free repair.** The standard gate multiplies an action probability
   by a spatial confidence on every step. On non-spatial steps the second factor is noise, and we show
   the product is at chance there (AUROC 0.504). Routing by decision type recovers the signal
   (0.649 → 0.729 held-out; 0.709–0.729 across checkpoints) with no change to the model and no extra
   forward pass.
3. **A certified operating point with its boundary.** Split-conformal risk control on the routed score
   certifies a 20% selective-risk budget at 40.7% coverage on held-out test. We then show the two ways
   the certificate fails: a hard risk floor that forbids the 10% target, and a breakdown of exchangeability
   when the threshold crosses to a different checkpoint.

We also report two things that do not help us. The grounding audit (§3.4) rejects the hypothesis that our
element errors are mostly measurement artifact. And we concede the framing: single-pass zero-token action
selection is established prior work; we do not claim it.

---

## 2. Setup and non-claims

### 2.1 The model

The subject is **d1-2B**, a 1.92B decision model built on an LFM3-VL-2B backbone, with a 164M-parameter
vision tower and a 25M projector. It exposes *readouts* rather than a text head: each question is
answered by a slot whose logits are projected onto the answer options' token groups and softmaxed over
options. One forward pass produces, for a given screen state:

- `action` — a distribution over eight action types (`click`, `write`, `long_press`, `scroll`, `wait`,
  `navigate_back`, `open_app`, `terminate`), with probability `p(action)`;
- `target_box` — a distribution per box edge over a 1000-bin grid, plus a per-edge confidence and a
  "found" probability; the click point is the box centre; `box_conf` is the readout's confidence;
- optionally `target_el` — a choice over the accessibility-tree candidates visible on screen (the
  "VA" track), and pointer reads for `type_text` / `app_name`.

Because every answer is a distribution, **a group of samples from one question is available for the cost
of one forward pass**. We note this as a property of the interface rather than a contribution: it makes
reward-based refinement of such a model budget-comparable to supervised fine-tuning, unlike an
autoregressive agent which pays one full generation per sample.

### 2.2 Benchmark and protocol

**AndroidControl-v2 (AC-v2), the 398-step / 350-episode leaderboard subset**, greedy, Pass@1. State is the
goal, the harness's interaction history, and the screenshot. A spatial step is credited only if the
predicted point falls inside the **smallest accessibility node containing the gold point**; `type_text`
additionally requires exact text. Non-spatial steps are credited on matching the gold action's keys.
This is the official element-level scorer, ported and differential-tested to zero mismatches over 398
rows. Baselines: always-`status` 21.86%, always-`click` 53.27%.

Our model reaches **64.07%** (255/398) on the pure protocol, type 84.17%, grounding **56.05%** (125/223). A
variant with an extra app-name hint reaches 64.82% (258/398) and is **not** the headline, because the hint
is an input the leaderboard protocol does not provide. (Restated 2026-11-07: no artifact yields 256/398, and
56.50% belongs to the step-705 checkpoint.) For calibration we use an 827-row validation split
of *training* episodes; the 398 leaderboard episodes are used once, as test. **The two splits are not
distribution-matched**, which matters for §6.2: measured action-type rates are `wait` 6.77% validation vs
1.01% test (**6.7×**), `type_text` 6.41% vs 2.76% (2.3×), `status` 16.57% vs 21.86% (0.76×),
`navigate_back` 3.51% vs 5.03% (0.70×). A threshold fitted on the calibration split is therefore fitted on
a different action mix than the one it meets in deployment — an independent reason, beyond checkpoint
strength, for the transfer failure we measure.

### 2.3 What we do not claim

We do not claim single-pass zero-token action selection as novel [jevMobile2026, guiJepa2026]. We do not
claim state-of-the-art accuracy: 64.07% is statistically tied with a rank-A 8B baseline (64.3) — the
±4.7-point interval on 398 rows cannot separate the top of the board. We do not claim a peer-reviewed
AndroidWorld baseline exists. We do not claim on-device latency; every timing we have is on a desktop
GPU. And we do not claim the model is fit to run unattended — §6 is the evidence that it is not.

---

## 3. The errors are not one bucket

### 3.1 Decomposition

Table 1 partitions all 140 errors on test, and replicates on the 827-row validation split.

**Table 1 — Error budget (test n=398; validation n=827).**

| Bucket | test rows | share | worth | val rows | share |
|---|---|---|---|---|---|
| A. spatial gold, **element wrong** | 68 | 48.6% | +17.09 pts | 137 | 44.5% |
| B. spatial gold, type wrong | 27 | 19.3% | +6.78 | 52 | 16.9% |
| C1. non-spatial, type wrong | 36 | 25.7% | +9.05 | 109 | 35.4% |
| C2. non-spatial, value wrong | 9 | 6.4% | +2.26 | 10 | 3.2% |
| **total errors** | **140** | | | **308** | |

Oracle ceilings on test: perfect grounding alone → **81.91%**; perfect action typing alone → **80.65%**;
both → 97.74%.

The consequence is diagnostic. Type errors total 63 on test (45% of all errors) and 161 on validation
(52%). Element errors total 68 and 137. **The two axes are comparable, and neither alone explains the
gap.** We deliberately do not rank them: on validation, p₁=161/827, p₂=137/827, the multinomial standard
error of the difference is ≈0.021 against a difference of 0.029, i.e. **z≈1.4, p≈0.16** — not
significant. The defensible statement is that a plan addressing only grounding leaves roughly half the
headroom unaddressed, and vice versa.

Per gold action type (test): `click` 121/212 (57.1%), `scroll` **18/37 (48.6%)**, `open_app` 20/27,
`navigate_back` 13/20, `type_text` 7/11, `status` 76/87, `wait` 3/4. `scroll` is 9.3% of the benchmark and
13.6% of the errors — the worst ratio on the board, and it is a *type* error, not a grounding error.

Two named slices are worth isolating because they are cheap to attack and semantically coherent:
**premature `status`** — the model declares the task complete when it must act — is 21 rows on test
(+5.28 pts) and 51 on validation; and **spurious `navigate_back`** is 8 rows (+2.01 pts).

### 3.2 Type decisions, not element precision, dominate the late-training gain

A paired comparison of two checkpoints of the same run isolates where continued training helps. From
step 705 (61.31%) to step 938 (64.82%) on the same 398 rows: **+3.52 points, McNemar b=8 / c=22,
p=0.016**, episode-clustered 95% interval **[+0.76, +6.33]** — a shift no independent confidence interval
on 398 rows can detect. Decomposed, the gain is **+12 non-spatial rows and +2 spatial rows**. Grounding saturates early; late training buys action-type
decisions. This is consistent with Table 1's finding that the type axis is first-order, and it is a reason
to measure both axes rather than optimise one.

### 3.3 Why paired testing is not optional here

The leaderboard number is a proportion on 398 rows: its 95% interval is [59.5, 68.8], ±4.7 points.
Independent intervals therefore cannot rank the top five systems, and a +2-point improvement is
invisible. But the comparison we actually make is **paired** — same episodes, same gold, same scorer — so
an exact McNemar test on the discordant pairs, and a bootstrap that resamples **episodes rather than
steps** (steps within an episode are correlated; 398 rows span 350 episodes, so a row-level bootstrap
understates the interval). Every delta in this paper is reported with both.

### 3.4 A negative result: the misses are real

AndroidControl is known to contain defective annotations — a curated re-annotation study reports roughly
30% of the dataset affected [acCurated2025]. If that noise dominated our errors, our 56.05% grounding
would be substantially an artifact of the metric, and the honest move would be to say so. **It is not.**

We joined our predictions to the eval set's full accessibility trees and classified every element miss
(n=198 joined spatial rows, 67 misses, exact-element hit 66.16%):

**Table 2 — Anatomy of the element misses.**

| Category | count | share of misses |
|---|---|---|
| genuinely **different element** | 54 | **80.6%** |
| same widget, parent/child subtree | 12 | 17.9% |
| same label (dataset ambiguity) | 1 | 1.5% |
| IoU ≥ 0.5 with the gold node | 0 | 0.0% |

A tolerance curve — exact 66.16%, +subtree 72.73%, +same-label 72.73% — bounds how much of the shortfall
*any* reasonable relaxation of the criterion could recover: about 6.6 points, not the 20 points needed to
reach the best reported element accuracy. Only 9.6% of rows have a gold label shared by more than one
node. Gold elements are small — median 0.85% of screen area, and **55.1% of spatial rows have a gold
element at or below 1% of screen** — and the median gold element is roughly twice as large when we hit it
(29,106 px²) as when we miss (15,876 px²). The failure is element *discrimination*, and it is genuine.

We report this because it removes an excuse we would otherwise have been entitled to use, and because a
paper about the honesty of a measurement interface should not be built on a metric artefact it cannot
demonstrate.

### 3.5 The two largest type-error slices, and three hypotheses that failed

Because type errors are roughly half the budget (§3.1), we tested the two most coherent slices directly.
All three hypotheses failed, and we report them so that the fixes we did *not* build are on the record.

**`scroll` is a type error, not a direction error.** Of 37 gold-`scroll` steps, 19 are wrong: **16 are type
errors** (the model emits `status` 9 times and `click` 7 times) and only **3 are direction errors**.
Scroll type accuracy is 21/37 (56.8%). This matches the SwipeGen decomposition, where swipe failures are
69% region/scrollable-target and only 6% direction (arXiv 2601.18305). The consequence is *not* to build a
direction model: the ceiling from repairing the type errors is +4.0 points.

**Premature `status` is not an episode-boundary artifact.** We expected the 21 steps where the model
declares the task done to concentrate at the end of episodes, where the screen genuinely looks terminal.
They do not: 87.9% of all evaluated rows sit at their episode's last evaluated step, against 85.7% of the
premature-`status` rows — an enrichment of **0.97×**, i.e. nothing. The rows are *separable* in
probability (median `p(terminate)` 0.671 premature vs 0.882 correct), but the obvious rule — suppress
`status` when grounding evidence is strong — flips 21/21 of the premature rows **and 76/76 of the correct
ones**, a net loss of 55 steps. One structural limit is worth stating: the AC-v2 subset contributes about
one step per episode (398 steps over 350 episodes), so hypotheses about episode position **cannot be tested
on this benchmark** at all.

**Per-class logit adjustment transfers the wrong way.** The standard rare-class repair is an additive
per-class offset on the logits, fitted on held-out data (arXiv 2007.07314). Fitted on the calibration split
it does what it should there — type accuracy 0.8053 → 0.8210 — and then **fails on test**, 0.8417 → 0.8141,
with 18 rows regressing against 7 fixed (exact McNemar p = 0.043, i.e. significantly worse). The cause is
the split mismatch of §5.2: the offsets are fitted against one action mix and applied to another. This is
the mix shift producing a concrete practical failure, and it is the clearest evidence for our claim that a
post-hoc rule belongs to the distribution it was fitted on.

---

## 4. The natural confidence score is mis-specified

### 4.1 The standard gate and where it breaks

For a decision model that answers both "what action" and "where", the obvious confidence is the action
probability, optionally multiplied by the spatial readout's confidence. That product was used in earlier
reporting of this model. It has a structural defect: **when the step needs no coordinates, the spatial
confidence is meaningless**, so multiplying by it injects noise into exactly the steps where the action
head was informative.

**Table 3 — AUROC(score, correct).** Validation is 827 rows; test is the 398 leaderboard rows, reported
for two checkpoints of the same run.

| score | val | test @938 | test @705 |
|---|---|---|---|
| `p(action) × box_conf` (conventional) | 0.677 | 0.649 | 0.662 |
| `p(action)` alone | 0.662 | 0.631 | 0.626 |
| `box_conf` alone | 0.665 | 0.646 | 0.656 |
| **type-routed** | **0.760** | **0.729** | **0.709** |

Within bucket on test: spatial `box_conf` **0.793** (@938) and **0.783** (@705), against `p(action)` 0.687
and 0.688; non-spatial `p(action)` 0.618 / 0.601, against the product **0.504 / 0.526** — chance.

The product does not fail because the model is badly calibrated on average. It fails because it applies a
spatial signal to non-spatial questions. Area under the risk–coverage curve makes the same point
(lower is better): routed 0.209 validation / 0.201–0.239 test, versus 0.243 / 0.244–0.270 for the product.

### 4.2 The repair

Route by decision type. For each row we take the readout that is **semantically responsible** for the
failure mode on that row:

- spatial step → `box_conf`, the confidence that the click point is on the intended element;
- non-spatial step → `p(action)`, the confidence in the action type.

To combine two signals on different scales without leaking test information, we fit a per-bucket empirical
CDF **on the validation split** and apply it unchanged to test; the resulting score is a within-bucket
quantile. One threshold then applies across both buckets. This is a per-row function of the validation
split only — there is no transductive use of test.

The improvement is not a tuning artefact: it appears on the calibration split (0.760) and on **two**
held-out checkpoints (0.729, 0.709). We report the range rather than the maximum.

### 4.3 The honest readout is under-confident, which is the safe direction

`box_conf` on spatial rows is conservative: on validation its mean is 0.419 against an accuracy of 0.601
(ECE 0.183; Brier 0.215 = reliability 0.046 − resolution 0.070 + uncertainty 0.240); on test, mean 0.363
against accuracy 0.574 (ECE 0.222; Brier 0.238). For a risk budget this is the benign failure: an
under-confident score forgoes coverage, whereas an over-confident one certifies risk it does not control.
It also explains why a *product* with it can still rank spatial errors well while being a poor probability.

### 4.4 Can a learned fusion beat the hand-specified routing? Not measurably.

The routing rule fixes its coefficients by hand. The literature on selective prediction says a *learned*
rejection function beats thresholding a pre-built score (SelectiveNet, arXiv 1901.09192), and recent GUI
work measures this directly, finding that **entropy ranks correctness better than top-1 probability,
which beats the top-2 margin** and that uniform aggregation of confidence signals is the failure mode
(PACE, arXiv 2609.24277). We therefore fit an L2 logistic model on eight per-row readout features — action
top-1, top-2 margin, normalised 8-way entropy, `box_conf`, `found`, a spatial indicator, and the two
routing interactions — with hyper-parameters chosen by repeated stratified group cross-validation
**inside** the calibration split (groups = episodes), and score test once.

**Table 3b — learned fusion vs the hand rules.**

| score | AUROC val | AUROC test | test risk @50% cov | test AUPRC |
|---|---|---|---|---|
| `p(action) × box_conf` | 0.677 | 0.649 | 0.266 | 0.771 |
| entropy only (−H) | 0.667 | 0.634 | 0.256 | 0.736 |
| type-routed (hand) | 0.722 | 0.730 | 0.226 | 0.811 |
| **learned fusion** | **0.767** (OOF 0.758) | **0.743** | **0.191** | 0.829 |

The learned model recovers the routing structure on its own — the largest coefficients are the two
interaction terms (`box_conf × spatial` +1.08, `p(action) × non-spatial` +0.50) and the spatial indicator
(−0.46), i.e. it learns to trust the box readout only where it is defined. It improves both the ranking
and the operating risk at 50% coverage (0.226 → 0.191).

**We do not claim the improvement.** The paired, episode-clustered ΔAUROC on the test split is
**+0.013, 95% [−0.020, +0.043]** — inside the minimum detectable effect for 398 rows with 140 positives,
which simulations place around +0.02. We report the point estimate, the interval, and the fact that it
straddles zero. The honest reading is that routing captures most of what these features contain, and that
398 test rows cannot resolve a further gain of this size. A larger evaluation, not a better model class,
is what is needed to settle it.

---

## 5. A certified operating point

We apply split-conformal risk control to the routed score. Let the threshold τ be chosen on the
validation split as the smallest score whose one-sided Clopper–Pearson upper bound on selective risk —
errors among accepted steps — is at most δ, at confidence α=0.10. We then report the same τ applied to
held-out test.

**Table 4 — Risk control, threshold fitted on validation (n=827), α=0.10.**

| budget δ | val coverage | val risk | val upper bound | test coverage | test risk | test upper bound | n accepted | held |
|---|---|---|---|---|---|---|---|---|
| 5% | — | — | — | — | — | — | — | not certifiable |
| 10% | — | — | — | — | — | — | — | not certifiable |
| **20%** | 47.6% | 17.3% | 19.9% | **40.7%** | **17.3%** | 21.7% | 162 | **yes** |

The attainable frontier on validation is 78.0% coverage at a 30% budget, 66.1% at 25%, 47.6% at 20%, and
**nothing at or below 15%**. We report the whole curve rather than a single point because the curve is
the deliverable: an operator with a different tolerance reads a different coverage.

### 5.1 The bound matters as much as the score

Inverting a Clopper–Pearson interval is the most conservative choice available and we made it in Table 4.
Conformal risk control (CRC) instead bounds the *expected* risk directly,
`λ̂ = inf{λ : n/(n+1)·R̂(λ) + 1/(n+1) ≤ δ}` (Angelopoulos et al., arXiv 2208.02814), which is valid but
strictly less conservative: it certifies a budget at a **higher** empirical risk. On the learned fusion
score the difference is not small.

**Table 4b — same calibration split, two bounds, fused score (test n=398).**

| budget δ | bound | τ | val coverage | test coverage | test risk (sd) | held |
|---|---|---|---|---|---|---|
| 20% | CP | 0.713 | 45.1% | 38.4% | 19.6% (3.2) | yes |
| 20% | **CRC** | 0.667 | 51.4% | **45.5%** | 18.2% (2.9) | yes |
| 25% | CP | 0.568 | 63.5% | 60.6% | 22.4% (2.7) | yes |
| 25% | **CRC** | 0.522 | 69.0% | **66.1%** | 23.6% (2.6) | yes |
| 10% | CP | — | — | — | — | not certifiable |
| 10% | CRC | 0.913 | 4.0% | **3.5%** | 0.0% (0.0) | "yes", at 14 steps |

The bound choice is worth **6–7 points of coverage at the same budget** — comparable to the entire gain
from two years of modelling on this benchmark. It also produces the cleanest possible illustration of
§6.1: the 10% budget *is* certifiable under CRC, at 3.5% coverage and 14 accepted steps. A certificate
that accepts fourteen steps is not an operating point; it is the arithmetic of a risk floor.

### 5.2 The splits are shifted, and we can weight for it

Measured action-type marginals differ between calibration and test: `wait` 6.77% vs 1.01%,
`type_text` 6.41% vs 2.76%, `navigate_back` 3.51% vs 5.03%, `status` 16.57% vs 21.86%. Total-variation
distance **0.106**; the likelihood ratios (exact, because action type is a discrete covariate) are
1.43, 1.32, 1.25, 1.05, 0.89, 0.43, 0.15 for navigate_back/status/open_app/click/scroll/type_text/wait.
Importance-weighting costs **9% of the calibration set** (effective size 752 of 827), which is cheap, but
it explains only a small part of the transfer failure: the maximum risk movement available from mix
reweighting is bounded by `TV × spread in per-type risk` ≈ 2 points, against an observed 6.6-point gap.
The rest is genuine difficulty shift and noise — see §6.2.

---

## 6. Boundary conditions: two things this certificate does not do

We promote these out of a limitations list because they are measurements, not caveats, and because both
contradict claims we would otherwise have made.

### 6.1 A 10% budget at 50% coverage does not exist here

The routed score has a **risk floor of about 12.5%**. On test, selective risk is 21.0% at 50.3% coverage,
17.5% at 40.2%, and 12.5% at 20.1%. Consequently a 10% budget cannot be met at any useful coverage, and
certifying it is not merely expensive but impossible at this sample size: with *zero* observed errors,
the Clopper–Pearson bound at α=0.10 reaches 10% only once at least 22 steps are accepted — 2.7% of the
validation split. The top 20 validation rows already contain 2 errors.

This is the feasibility condition of selective conformal risk control, where acceptance itself breaks
exchangeability between calibration and test and a minimum accepted count is required
[scrc2025]. We report the wall rather than a gate that silently returns a few percent coverage: **a
certificate that accepts almost nothing is worse than no certificate**, because it invites the reading
that a risk budget has been met for a system operating at near-zero automation.

The constructive corollary is that the objective for improving this model is now numeric and sharper than
"raise accuracy": **halve the risk at 50% coverage**, from 21.5% to below 10%. That is a property of the
score's discrimination, not of the threshold.

### 6.2 The threshold is a property of the checkpoint, not of the model family

Applying the same τ=0.527 to a weaker checkpoint of the same run (step 705, 61.31%) gives **23.9% risk at
38.9% coverage** — a violation of the 20% budget that the identical threshold satisfied at step 938
(17.3%). Nothing about the score changed; the calibration and test distributions stopped being
exchangeable across the checkpoint boundary.

**We state the size of that evidence honestly.** At 162 and 155 accepted steps the binomial standard
errors of these two risk estimates are 2.9 and 3.4 points, so the standard error of their difference is
about 4.5 points and the observed 6.6-point gap is roughly **1.5σ**. That is *suggestive*, not decisive:
part of what looked like an exchangeability break is sampling noise, and the shift diagnostic in §5.2
(mix reweighting explains ≈2 points) accounts for part of the rest. We report all three quantities —
the gap, its standard error, and the explained component — rather than treating the gap alone as proof.
What survives is the qualitative claim, which is the one that matters: a threshold is a property of the
calibration distribution, and re-using it against a weaker model is not licensed by the certificate.

Two conclusions follow. First, **exchangeability, not the score, is the binding constraint** on
transferring a certificate, and selection-aware or shift-aware conformal variants are the correct
instrument [scrc2025]. Second, this is the mechanism behind a discrepancy in our own earlier reporting:
gate numbers that appeared to be irreproducible were the result of fitting a map on one distribution and
evaluating on another, then averaging two incomparable signals into one score. We therefore state a
certificate as a property of the triple (score, checkpoint, split), and we regard "which of these is it a
property of" as the interesting question, not an implementation detail.

---

## 7. Related work

**Zero-token and single-pass GUI policies.** Jev-Mobile reports a typed decision model as executor at 79%
AndroidWorld with substantially lower per-step executor time than step-wise VLMs [jevMobile2026];
GUI-JEPA establishes a decoder-free GUI policy and, notably, already uses the margin between its top two
scores to flag uncertain steps (AUROC 0.74) so they can be routed to a larger agent [guiJepa2026]. Our
§4 differs in two ways that we consider substantive: we show that the *conventional* product score is at
chance on non-spatial steps — i.e. the failure is structural and not a matter of a better heuristic — and
we attach a distribution-free risk certificate together with the conditions under which it fails. We are
therefore not the first to route by confidence, and we say so.

**GUI agents and accuracy.** The AC-v2 leaderboard is led by autoregressive systems of 4–8B parameters and
above [memw2026, guiLibra2026, uivenus2026]; the best element accuracy reported is far above ours. Our
model is competitive on step accuracy and on action-type accuracy at 1.92B with no generated tokens, and
we do not claim more than that.

**Calibration and selective prediction.** The dissociation between average calibration (ECE) and
informativeness (AUROC, risk–coverage) is well known; we observe it directly, since `box_conf` is
*mis*calibrated on average (ECE 0.18–0.22) yet ranks correctness at 0.79–0.80. Selective conformal risk
control supplies both the guarantee we use and the feasibility condition that explains §6.1
[scrc2025]. Abstention benchmarks show frontier models are weak at knowing when to abstain
[agentAbstain2026] and that post-hoc risk heads with conformal thresholds can control executed harm
[cora2026]; our result is the same instrument applied to a single-pass decision interface, where the
score is free rather than an extra head.

**Proper scoring and RL.** A neighbouring literature trains decision readouts with proper scoring rules
rather than cross-entropy to preserve calibration under reinforcement learning
[prdo2026, thx2026]. Our result is complementary and cheaper: the fix we report requires no retraining,
because it is a re-specification of which readout to trust.

---

## 8. Limitations

1. **Sample size.** 398 test rows, ±4.7 points. We report paired tests and episode-clustered bootstrap
   intervals, and we do not rank differences smaller than the interval.
2. **One model, one benchmark, one family.** Two checkpoints of a single training run, one Android
   benchmark. The routing principle is motivated structurally, but its magnitude is measured once.
3. **Marginal, not conditional, guarantee.** Conformal risk control bounds the average selective risk; it
   does not certify any individual step, and with selection it does not certify coverage conditionally.
4. **The certificate does not transfer across checkpoints** (§6.2) — this is the paper's most important
   single measurement, and it bounds the claim to a stated domain.
5. **The 10% target is unattainable** (§6.1). Prior internal documents state a 10%-budget/50%-coverage
   goal; we show it is not reachable at this checkpoint and we do not report it.
6. **Accuracy is a tie, not a win.** Type accuracy (84.17%) exceeds the best reported 8B value (81.16%),
   but on 398 rows, with a self-reimplemented scorer, against self-reported baselines.
7. **No on-device latency.** All timing is desktop GPU; we make no mobile claim.
8. **No task-level success.** Step accuracy is not task success; an episode-level evaluation on a mobile
   environment is separate work and, in our hands so far, underpowered.
9. **The training mixture was resampled to the test action distribution.** The 15,000 curated training
   decisions are stratified so that their action-type rates match the *evaluation* mix almost exactly
   (click 52.75% train vs 53.27% test; status 22.00% vs 21.86%; scroll 9.00% vs 9.30%; every ratio
   within 0.97–1.09×). That is test-informed data design, not neutral sampling: it is a form of leakage
   that flatters the headline by roughly the distance between the natural training prior (`status` 15.41%
   in the full 88,288-row pool) and the test prior (21.86%). We disclose it rather than remove it,
   because re-running without it changes the headline and we have not measured by how much. Any future
   comparison must either reproduce the same stratification for all systems or drop it for all.
10. **A retracted calibration paragraph.** Earlier drafts of this work reported an isotonic-mapped ECE of
   0.036 and gates holding 55.0% / 36.7% coverage. Those figures do not reproduce from the saved artifacts
    and are withdrawn; §4–§6 replace them. Appendix B records the retraction.

---

## 9. Conclusion

A single-pass decision model can be accurate on GUI action selection and still be unable to say when it
is wrong, because the confidence it is usually credited with is a product of two readouts that answer
different questions. Routing confidence by the question each readout answers recovers the signal for
free, from 0.649 to 0.729 on held-out rows, and that score is strong enough to certify a 20%
selective-risk budget at 40.7% coverage. It is not strong enough to certify 10%, and the certificate does
not survive a change of checkpoint. We prefer to publish the domain where the certificate holds than to
publish a number that does not.

---

## Appendix A — Reproduction

All results are recomputed from two saved prediction files and the evaluation set; NumPy only, no model,
no GPU, no network.

```bash
cd benchmarks/train
python error_budget.py     ../results/2026-10-29-evals/E2-938-V.jsonl
python error_budget.py     ../results/2026-10-29-evals/E2-val.jsonl
python paired_eval.py      ../results/2026-10-29-evals/E2-705-V.jsonl \
                           ../results/2026-10-29-evals/E2-938-V.jsonl
python gate_routed.py      --val ../results/2026-10-29-evals/E2-val.jsonl \
                           --test ../results/2026-10-29-evals/E2-938-V.jsonl --alpha 0.1
python grounding_audit.py  --rows ../data/acv2/500_steps_filtered.json \
                           --pred ../results/2026-10-29-evals/E2-938-V.jsonl
# provenance: asserts every number in this paper against the saved artifacts, exits nonzero on drift
python verify_paper.py
```

`verify_paper.py` performs 63 assertions over `benchmarks/results/2026-11-03-stage0/` and reports
`ALL CHECKS PASS` or the list of drifted claims. It caught one stale artifact during drafting, which is
the reason the check exists.

## Appendix B — Retractions and artifact notes

1. **Isotonic calibration figures withdrawn.** ECE 0.146 → 0.036, AUROC 0.723, and gates holding 55.0% /
   36.7% coverage at 25% / 20% risk do not reproduce from the saved prediction files. Recomputing gives a
   product-score ECE of ~0.29–0.36 and an isotonic fit that collapses to a degenerate threshold. These
   numbers must not be cited. §4–§6 replace them with reproducible equivalents.
2. **"The remaining gap is grounding, not type" withdrawn.** See Table 1 and §3.1: the two axes are
   comparable, and the difference is not significant.
3. **Artifact caveat.** The prediction file used for the analysis is the 258/398 *hinted* variant of the
   best checkpoint. **Superseded 2026-11-07:** the pure per-row predictions *are* on disk — 255/398
   in-process (`E2-nomem`) and 254/398 through the served API (`e2-pure-hist.jsonl`, every readout slot
   populated). So the headline is 64.07 (pure) and 64.82 (hinted), and the error *composition* analysed here
   is the hinted artifact's.
4. **Prior framing.** Single-pass zero-token action selection is prior work; the earlier draft of this
   paper presented it as a contribution. It is not.

## Sources

`jevMobile2026` Jev-Mobile, arXiv:2609.30186 · `guiJepa2026` GUI-JEPA, arXiv:2610.36877 ·
`liquid2026d1` Liquid AI d1-2B model card · `acCurated2025` AndroidControl-Curated, arXiv:2510.18488 ·
`memw2026` Mem-W, arXiv:2605.09317 · `guiLibra2026` GUI-Libra, arXiv:2602.22190 ·
`uivenus2026` UI-Venus, arXiv:2508.10833 · `scrc2025` Selective Conformal Risk Control, arXiv:2512.12844 ·
`agentAbstain2026` AgentAbstain, arXiv:2607.10059 · `cora2026` CORA, arXiv:2604.09155 ·
`prdo2026` PRDO / OpenJev-RLCD, arXiv:2609.38850 · `thx2026` THX-01 RLCD (vendor report)
