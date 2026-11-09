# Gap Analysis: "Calibrated to Be Trusted" — publishedability, collisions, and the honest core

*Audience: author. Date: 2026-11-02. Scope: rigor over encouragement. This is a critical read of the
current draft (`papers/single-pass-decision-acv2.md`) and the AGENTS.md goal, against the late-2026 literature.*
*Every number below traces to a source URL or to the workspace. Vendor numbers are labelled as claims.*

---

## 0. The one-line reality check (read this first)

Your headline — *"a 1.92B single-pass decision model reaches 64.32% on AC-v2, on par with GUI-Libra-8B 64.3"*
*(quoted as the claim being critiqued; 64.32 = 256/398 exists in no artifact — the headline is 64.07%, 255/398)* —
**is no longer a contribution, and two 2026-09/10 papers have already claimed the category framing you are in.**

1. **`Jev-Mobile`** (Rice, `2609.30186`, Sep 2026) is titled **"Jev as an Executor for Mobile GUI Agents"** — it
   is *precisely* your system thesis: a typed decision model (single-pass, no generation) selects actions in the
   a11y space, under a VLM that sets local goals. It reports **79% AndroidWorld** task success, cuts
   execution-model time from 94.30 s to **5.16 s** and model API cost by **73.4%**.
2. **`GUI-JEPA`** (`2610.36877`, Oct 27 2026) calls itself **"the first decoder-free GUI agent policy"** — a
   **0.50B** candidate-scoring action policy that selects actions in **one batched pass, no token decoded**,
   reaching **2.8–4.0× faster** and **86–92% of a 7B VLM's accuracy** with **6% of its parameters**, on
   AndroidControl.

So **"single-pass, zero-token action selection for a mobile GUI" is taken.** A reviewer at any top venue will
find these two within minutes. Your paper must not frame "single-pass decision model for mobile GUI" as its
novelty.

What is **not** taken: the **calibration-mechanism and the risk-controlled act/ask/refuse gate for this class
of model, under distribution shift, at on-device latency.** That is a real but *narrow* intersection, and it is
the only defensible core. Sections 3–4 explain what survives.

---

## 1. Publishability — the minimum evidence bar at a top venue (late 2026)

Top ML venues (NeurIPS/ICML/ICLR) want **mechanism + causality + rigorous statistics**, not a benchmark point.
Top HCI venues (CHI/UIST) want the **human-facing deployment problem + measured on-device cost** with honest
uncertainty. Your claim currently straddles both and satisfies neither. To be accepted, at minimum:

### 1.1 You must lead with a *mechanism finding*, not a leaderboard row
- **Required:** an ablation that changes **one training objective** (CE-only vs CE+proper-scoring vs
  CE+joint-calibration) on **identical data, model, and split** — and shows the confidence's *behavior* changes
  (discrimination, selective-risk-at-coverage, hard/OOD reliability), not just its average ECE. This is the only
  thing in your stack that is (a) not done by anyone and (b) mechanistically interesting. Nobody in the
  references below does this: THX-01 is a recipe on closed data; decider is a full recipe but no regime study;
  TypedBench *evaluates models, not recipes*; 2610.19811 does risk control but no training-objective comparison;
  GUI-JEPA trains with listwise CE but does no calibration/regime study.

### 1.2 Statistical rigor you must ship (non-negotiable)
- **≥3 training seeds**, and report seed spread. `2610.36831` shows a single seed moves ScreenSpot-v2 accuracy by
  up to **2.6 points** — as large as the margins the field reports.
- **Paired intervals** for any model-vs-model comparison: on AC-v2 (~400 test steps) two 7–8B models are
  separated only to ~**1.4 points** by test-set sampling alone. Your 398-step test is far too small to separate
  you from *anything*.
- **Report the trivial baselines.** Majority-class (always-`click`) ≈ 0.58–0.64 on AC-v2 and uniform random
  (1/K) must appear next to your number. Your own AGENTS.md VERIFICATION GATE already documents the 0.5828 =
  majority-class incident — a repeated top-venue rejection reason.
- **A proper scoring rule (Brier) alongside ECE**, plus the **Brier/Murphy decomposition** (reliability vs
  resolution vs uncertainty) so the reader sees whether your model is *well-calibrated* or *just uninformative*.
  `2610.19811` shows a model can have *better* ECE but *worse* AUROC (Kev-base ECE 0.051 vs Kev-ft 0.16 on
  StackOverflow, yet AUROC 0.599 vs 0.81) — aggregate calibration hides discrimination failure.

### 1.3 The system/on-device claim needs measured, not assumed, numbers
- Report **p50/p95 model latency on Apple silicon (M1/M-series, no GPU)**, tokens per task, peak memory, and
  **end-to-end per-task wall time including the emulator**. Your harness plan correctly notes the ~1 s emulator
  settle dominates the whole-task speedup — you must say so. THX-01 and decider report "one NVIDIA GPU"; your
  venue is the on-device/no-GPU one. Nobody in this class reports Apple-silicon reliability + cost.
- If you target CHI/UIST, the *human* framing matters: a wrong tap is irreversible, cost is asymmetric, risk
  budget is the only interface. State the cost matrix explicitly and report the false-action/coverage operating
  points at the budget you actually certify.

### 1.4 Prototype-scale honesty
- You currently flag a **calibration discrepancy** (reported-in-run ECE 0.036 / cov 55%/36.7% not reproducible
  from saved jsonl). This must be **resolved before submission**, not footnote'd. Either re-fit isotonic on val
  and confirm, or retract. A reviewer who finds an internal inconsistency will discount the whole paper. Your
  own `papers/single-pass-decision-acv2.md` already does the right thing by separating reproducible from
  reported-in-run — keep that, but complete it.

### 1.5 Venue choice
- **ML (NeurIPS/ICML/ICLR)** if and only if the mechanism study (CE vs proper-scoring vs joint, hard/OOD split)
  and the **confidence-informativeness** result (Section 3) carry the paper.
- **CHI/UIST** if and only if the **on-device risk-budget gate and latency/cost** carry it, with a real human
  risk asymmetry argument.
- Mixed claims that are strong at neither will be desk-rejected. Pick one axis as the headline.

---

## 2. Prior-art collision map — claim-by-claim, taken vs open

| Work | What they claimed | What they did **not** do | Collision with your paper |
|---|---|---|---|
| **Jev / TypeSafe** | Frontier System One model; RLCD post-training; "can't hallucinate" (schema-guaranteed); calibrated; "193× faster / 444× cheaper" | RLCD **unpublished**; no independent replication; no AC-v2/GUI action-selection eval | Framing collision (category), not result |
| **d1-2B / d1-3B** (Liquid) | Blueprint; native `box`/`boxes` grounding; Decision Index 51.1 (v0.3); ~5.7 ms H100 | **No AndroidControl/AC-v2 number published**; vendor self-scored | This is *your base*, not a collision — but it means your "on par with GUI-Libra" number is the *first independent AC-v2 measurement of d1-2B* |
| **THX-01** (`doofz/THX-01`, Oct 2026) | 322M, non-autoregressive, ~10 ms; **RLCD** with strictly proper scoring reward (log+spherical+RPS); ECE 0.003–0.066; **selective automation ~75% zero-error / 90% at ≥99.5% acc** | Closed data; **no regime study** (when/why calibration breaks); NVIDIA GPU only; **not GUI action selection** | **"First open calibration recipe" is TAKEN.** Your Q2 (recipe mechanism) overlaps its headline |
| **TypedBench** (`2610.11392`, Oct 2026) | First benchmark for **calibration, framing sensitivity, cost** in System One models; finite-sample noise floor | **Benchmarks existing models; does not study training recipes** | **"First calibration-reliability benchmark" is TAKEN.** Your Q1 is a recipe question, so you are complementary, not colliding — but you cannot claim "first calibration benchmark" |
| **AgentAbstain** (`2607.10059`, Jul 2026) | First systematic framework for **do tool-using agents know when not to act**; 263 paired tasks, 8 abstention scenarios; best of 17 frontier models = **59.5%** | Text/LLM tool agents, **not** single-pass decision models; **not** mobile GUI action selection; no training-objective study | Collides on "abstention" framing only |
| **CORA** (`2604.09155`, Apr 2026) | **Post-policy, pre-action** conformal risk-controlled shield over a VLM policy; tune-able safety knob; distribution-free guarantee | **Post-hoc** — does **not train the decision model**; risk-control over *frozen* scores; uses a VLM policy, not a single-pass decision model | **Partially collides** with your gate. Your differentiator must be that the *model itself* produces calibrated confidence, and that this beats post-hoc conformal on coverage-at-budget |
| **CAAG** (`ijcope v2i5`, May 2026) | Calibrated Act–Ask–Abstain **gating**: CE + α·ECE loss + expected-utility gate + memory-triggered reflection | **Simulated results only**; targets **generative LLM agents**, not single-pass decision models | **Top framing collision** (act/ask/refuse + calibration loss) — but no real experiments, and generative rather than single-pass |
| **decider / decider-2b** (Mapika) | Open recipe: CE+proper-scoring on ~95 datasets, then PPO+proper-log-score+KL; per-type temperature; in-task ECE 0.038 / held-out 0.083 | **Self-admitted failure on hard items: ECE 0.156 vs its own 0.08 bar**; no GUI action selection | **"First open calibration recipe" is TAKEN.** Its *hard-item failure* is your Q1 evidence (quotes 0.18–0.31 across versions) |
| **Mem-W** (`2605.09317`, May 2026) | Latent memory-native GUI agent; AC-v2-High **68.59** (Mem-W-8B), +7.53 over UI-Venus-1.5-8B | Autoregressive VLM + latent memory; **no calibration/risk gating**; not a decision model | Accuracy baseline you must beat or cede (you cede it) |
| **MobileRL** (`2509.18119`, ICLR 2026) | 3-stage: reasoning-free SFT → reasoning SFT → **online agentic RL (AdaGRPO)** with sparse terminal reward; 9B = 80.2% AndroidWorld / 53.6% AndroidLab; 7B = 72.0%/42.5% | Autoregressive multi-turn; **no single-pass claim; no calibration gate** | **The strongest "your 64% is not SOTA" reference** — the frontier is multi-turn RL, not single-pass |
| **GUI-Libra** (`2602.22190`, NeurIPS 2026) | Action-aware SFT + conservative GRPO (KL trust region); AC-v2-High **64.3** (8B), Grounding 76.2 | Autoregressive VLM; **no calibration; no single-pass** | Your "on par" baseline (64.3) — you tie it but don't beat it |
| **UI-Venus-1.5-8B** | 61.06 AC-v2-High; grounding via RL (Gaussian box reward) | Autoregressive; **no calibration/gating** | Baseline; you are above it |
| **Re-TRAC** (`2602.02486`, Microsoft, ICML 2026) | Recursive trajectory compression; small 4B/30B beat 229B–358B on deep-search | **Template only** (named failure → mechanism → "small+mechanism ≥ big") — **not a GUI/calibration claim** | **Use as your structural template**, not as a competing result |
| **RIGHT CLICK, WRONG BOX** (`2610.36831`, Oct 27 2026) | **Box scoring under-counts correct clicks**: on Gitea ~1/5, on Oscar ~1/3 of box-rejected clicks are outcome-correct; ScreenSpot-v2 web ~14.4% | **Native mobile widgets: ~1.0% of box-rejected clicks are outcome-correct** | **Directly governs your metric.** NOTE: your AGENTS.md cites this to justify the AC-v2 plateau as a "grading artifact" — but the ~1/5–1/3 figures are for **web** (Gitea/Oscar). On **native mobile** the out-of-box equivalence is ~1%. So the artifact argument is much **weaker** on AC-v2 (mobile) than your notes imply. Use this carefully |
| **GUI-JEPA** (`2610.36877`, Oct 27 2026) | **"First decoder-free GUI action policy"**; candidate-conditioned JEPA latents; 0.50B; 2.8–4.0× faster; 86–92% of 7B accuracy; 6% params; no token decoded | **No calibration; no act/ask/refuse gate; no risk budget**; step-level scorer (no terminal/abstention head); takes the a11y tree as privileged input | **COLLISION ON THE FRAMING.** It is the strongest "single-pass zero-token action selection" claim in the field. You must not compete on this axis |
| **Jev-Mobile** (`2609.30186`, Sep 2026) | **Jev as an executor for mobile GUI agents**; typed decision model selects a11y actions under a VLM; 79% AndroidWorld; −32.7% time / −73.4% cost | **No calibration study; no risk-budget gate; no act/ask/refuse-gate experiment**; uses hosted Jev (not trained); compares to only 2 baselines; success **below** Step-wise VLM (0.84) | **COLLISION ON THE SYSTEM.** Your "decision model as mobile GUI executor" thesis is its title. Your only defensible difference is the **calibrated, risk-controlled gate** — which it does not have |

**What remains open, precisely:**
- **Q1 (breakdown regime):** no systematic study of *when/why* the confidence in this class becomes
  *uninformative about correctness* (rather than merely miscalibrated) on hard/OOD/ambiguous-element inputs.
  Evidence of breakdown exists (decider hard-ECE 0.156–0.31; THX-01 DAIR 0.284; AgentAbstain 59.5%;
  `2610.01006` confidence-overconfidence past a knowledge boundary) but is **unanalyzed for the GUI decision-model
  class**.
- **Q2 (mechanism):** CE-only vs proper-scoring vs joint-calibration on **identical data/model/splits** with an
  easy/hard split. **Not done by anyone. This is your most defensible scientific contribution.**
- **Q3 (on-device gate):** a risk-controlled act/ask/refuse gate on **Apple silicon / no-GPU** at ms latency.
  **Uncharted** (all prior work reports NVIDIA GPU).
- **Novel sub-claim (the one that defeats the "calibration proves nothing" objection):** a trained confidence that
  **separates knows-vs-doesn't-know** (AUROC/rank calibration on the OOD/hard split), rather than one that is only
  *average*-calibrated. This directly answers `2610.01006`'s finding that Jev's confidence "does not show what
  the model does not know."

---

## 3. The honest weakness — strongest objection, and the experiment that defeats it

### 3.1 The strongest objection (construct it yourself, don't wait for a reviewer)

> "Your whole thesis is that a single-pass confidence is a trustworthy act/ask/refuse signal. But the published
> evidence says the opposite for exactly your class. `2610.01006` shows Jev — the model you're building against —
> is *average*-calibrated on familiar closed-choice tasks yet **assigns up to 0.80 confidence to a salient option
> when it has no answer-relevant information**, and **exceeds accuracy by 0.21–0.33 past a knowledge boundary**,
> a gap that recalibration does not close. That is exactly your 'hard/OOD regime.' So your confidence is the
> *wrong object* to gate on: it predicts correctness on the familiar split but does **not** report missing
> knowledge. Meanwhile `2610.19811` shows that even a *well-calibrated* System One score gives no risk guarantee
> under naive thresholding — naive exceeds risk budget 42% of the time; you'd need LTT-style finite-sample
> certification. And on the task itself, `2610.36831` shows box-scored AC-v2 under-counts correct clicks, so your
> 64% is measured against a grading rule that is itself wrong. On top of that, the frontier methods (MobileRL,
> GUI-Libra, Mem-W) are **multi-turn autoregressive RL agents** that beat 64% by construction; a single-pass
> over-K classifier is a strictly weaker policy shape. So your 'parity with GUI-Libra-8B (64.3)' is parity with a
> *non-frontier* baseline, and your calibration claim is *uninformative by the field's own newest evidence*."

### 3.2 The experiment that defeats (most of) it, and the expected effect size

**Experiment E1 — "confidence is a signal of knowing, not just of being right on average."**
Take the AC-v2 test (and a shifted AndroidWorld trajectory set), partition each into **in-distribution** (IDD)
and **out-of-distribution** subsplits (follow the IDD/OOD-union definition used by `GUI-JEPA` and the
AndroidControl paper). For each of {CE-only, CE+proper-scoring, CE+joint-calibration} trained on identical data,
measure:
- **AUROC of confidence (or p(action)) for predicting binary correctness** on IDD **and** on OOD/hard.
- **Do-it-wrong control:** inject "ambiguous / no-signal" steps (a target element that appears twice, an
  instruction with two valid actions — the exact regime where `2610.36831` shows box scoring fails). Show that
  **CE-only confidence rises / stays flat on these while proper-scoring confidence falls** — this is the
  "knows-vs-doesn't-know" separation that `2610.01006` says Jev lacks.

**Expected effect size (grounded):** `2610.01006` reports the *confidence-vs-knowledge* gap as **+0.21–0.33**
(confidence minus accuracy past a boundary), with confidence-vs-correctness AUROC drops of **~0.85 → below** on
OOD. If your proper-scoring model keeps confidence-vs-correctness AUROC **≥0.75 on OOD** while CE-only collapses
to **~0.60** (or confidence rises on no-signal items while proper-scoring falls), that is a 0.15–0.25 AUROC
separation — comfortably publishable *as a mechanism finding*. If you can show the gate at a **10% false-action
budget** keeps **≥40% coverage on OOD** while CE-only's gate must retreat to ~15% coverage to hold the same
budget, that's your headline operating-point win. **Caveat:** you must verify these are directionally real with
≥3 seeds and paired bootstrap CIs, because single-seed AUROC differences of 0.05 are noise.

**Experiment E2 — defeat "post-hoc conformal already does this."**
Compare your trained-head gate against **CORA-style post-hoc conformal** and against **LTT** (the exact method
`2610.19811` certifies). Report coverage-at-budget and achieved risk on IDD and OOD, with CIs. Two honest
outcomes, both publishable: (a) trained-head *beats* conformal/LTT coverage at the same certified risk (novel,
why training the gate matters), or (b) trained-head is **no better** — in which case the contribution collapses to
"a post-hoc gate is sufficient and training a proper score buys you nothing," which is itself a solid negative
result but *not* the paper you drafted. Design for (a) but be honest if you get (b).

**Expected effect size:** `2610.19811` shows LTT certifies at only ~7.8% budget-exceedance vs 42% for naive. Your
win must be **coverage**, not risk-compliance: e.g. trained-head achieves 55% coverage at α=0.10 where LTT
achieves 35–40% on the same data. A ≥10-point coverage gain at matched risk, with CIs excluding zero, is the bar.

**What it does NOT defeat:** the "single-pass is a weaker policy shape than multi-turn RL" objection. You cannot
beat MobileRL 80% on AndroidWorld with a single-pass over-K head. You must **explicitly concede** this and pivot
the contribution to the reliability-per-cost axis (Re-TRAC template: small + mechanism ≥ big *on the reliability
axis*, not raw success). If you frame 64% as "SOTA," you lose. If you frame it as "the frontier is multi-turn
RL; a single-pass calibrated gate can be the *safety/abstention layer* on top (as Jev-Mobile/CORA suggest), and
here is the first study of whether the gate's confidence is honest," you have a defensible niche.

---

## 4. What is actually novel and defensible (testable on one 4090, 2B model, <8 h)

Three narrow, unclaimed, one-4090 claims. These are the ones to build the paper around:

**Claim N1 (mechanism, highest value):** *For single-pass decision models on mobile GUI action selection, CE-only
fine-tuning leaves confidence that is average-calibrated but uninformative about correctness on hard/OOD/
element-ambiguous inputs — the exact failure `2610.01006` shows for Jev — whereas a proper-scoring objective
produces confidence that ranks correctness better and holds a selective-risk budget at higher coverage.*
- **One 4090, <8 h:** 2B model, CE vs CE+proper-scoring vs CE+joint on identical data (train split), 3 seeds,
  3 objectives × 3 seeds = 9 short fine-tunes. Measure AUROC(conf, correctness) and coverage-risk curve on the
  hard/OOD split; report Brier decomp + Smooth ECE.
- **Why defensible:** nobody has compared objectives on *identical* data/model/splits in this class, and it
  directly answers the `2610.01006` objection. This is the paper's spine.

**Claim N2 (gate vs post-hoc):** *A confidence gate trained into the head can certify a false-action budget at
higher coverage than a post-hoc conformal shield (CORA) or LTT (2610.19811) on the same single-pass model.*
- **One 4090, <8 h:** train the head (as in N1), then wrap the *same* frozen model in (a) post-hoc conformal/CORA,
  (b) LTT, and (c) the trained-gate. Report coverage-at-budget (α=0.05/0.10/0.20) with paired bootstrap CIs.
- **Why defensible:** CORA and 2610.19811 are post-hoc; nobody has shown that *training the objective* the risk
  budget was designed for beats the post-hoc layer. Either direction is a contribution. **Risk:** if you're not
  better, E2's honest negative result is the outcome.

**Claim N3 (on-device shift):** *The single-step-box-graded confidence overfits to the benchmark's grading rule;
under the shift to multi-step, outcome-graded online evaluation (AndroidWorld), the gate's informativeness
collapses — and a shift-aware recalibration (weighted conformal) recovers a certified budget on Apple silicon.*
- **One 4090, <8 h:** carry the model's confidence through a small set of live AndroidWorld rollouts (or use the
  emulator server), measure whether the AC-v2-trained risk budget holds; if not, apply weighted/swap conformal
  under covariate shift (Tibshirani `2019`, Angelopoulos & Bates `2024`) and report the recovery. On-device
  latency measured on M1/M-series.
- **Why defensible:** nobody in this class has measured the AC-v2 → AndroidWorld calibration breakdown or done
  on-device shift-aware risk control. This is the Q1×Q3 intersection.

**If you must cut to ONE:** cut claim N2 (it's the riskiest and the most likely to return a null). Keep N1 (the
mechanism) and N3 (the shift + on-device venue). Together they make a coherent, novel, honest paper.

---

## 5. Measurement traps — why ECE is unreliable, and what to report instead

### 5.1 Why bare ECE fails
- **Binning sensitivity.** ECE varies arbitrarily with the number/width of bins (`Guo et al. 2017`,
  `Nixon et al. 2019`). Report ECE across a *grid* of bin counts to show sensitivity, or use a binning-free
  estimate: **Smooth ECE** (KDE-based reliability diagrams, `Blasiok & Nakkiran ICLR 2024` — this is what
  `2610.01006` uses for exactly this reason).
- **Aggregate hides discrimination.** A low global ECE can coexist with *poor* class separation
  (`2610.19811`: Kev-base ECE 0.051 vs Kev-ft 0.16, but Kev-base AUROC 0.599 is *worse*). Report ECE **and**
  AUROC of the score for correctness together.
- **Sample size.** AC-v2 test is ~400 steps. At p=0.64, 95% CI width ≈ 1.96·√(0.64·0.36/400) ≈ **±4.8 points**.
  You cannot resolve a 1–2 point claim here. Use the **full test set** (GUI-JEPA uses 8,444 steps), or report
  Wilson/bootstrap CIs and accept the wider interval.
- **Class imbalance.** AC-v2 skews to `click`. A top-probability ECE is dominated by the majority class. Report
  **per-class ECE** and class-conditional reliability diagrams; also report AUROC and Brier pooled and per class.
- **Ambiguous ground truth.** Box grading under-counts correct clicks (`2610.36831`), and multiple valid actions
  exist (MobiBench). On **native mobile** the artifact is ~1%, but on steps with duplicated/equivalivalent
  elements the box rule is genuinely wrong — those are exactly the hard/OOD steps you want to probe. Report
  outcome-corrected accuracy alongside box accuracy on the web-like items (if any), and on mobile at least note it.

### 5.2 What to report instead of / alongside ECE
1. **Brier score + Brier (Murphy) decomposition** into reliability / resolution / uncertainty — shows whether a
   model is *calibrated* (low reliability term) or *uninformative* (low resolution term). This is the single most
   useful addition.
2. **Reliability diagrams with pointwise (bootstrap) CIs**, using adaptive or Smooth KDE bins.
3. **AUROC of confidence for correctness** (rank calibration / discrimination) — the number that answers
   "does confidence track knowing?"
4. **Coverage–risk (selective prediction) curves** with bootstrap CIs, at the operating points you certify
   (Geifman & El-Yaniv selective classification).
5. **Selective risk at a fixed budget**, reported as (coverage, achieved-risk, 95% CI).

### 5.3 Concrete sample-size requirements (2B model)
- **To resolve a ~4-point accuracy difference from a baseline (e.g. Mem-W-8B 68.59 vs your 64.5) with 80%
  power, p<0.05:** need ≈ **1,900 test decisions per model** (two-proportion). AC-v2's ~400-step test is far too
  small. Use the full AC-v2/AndroidControl test set (~8.4k steps) or aggregate across the public test subsets.
- **Per-class ECE / reliability bins:** target **≥20–30 samples per bin** × ~10–20 bins → **≥600–1,000**
  confidence-labeled decisions **per condition** (you have multiple conditions: objectives, splits).
- **Risk-budget certification sample requirement (from `2610.19811` Appendix B.4):** certifying a threshold at
  budget α and error δ needs **at least ⌈log δ / log(1−α)⌉ selected (executed) examples in the action region**.
  For α=0.05, δ=0.05 → **≥59 act actions**; for α=0.10 → **≥29 act actions**. If coverage is ~50%, you therefore
  need **≥118 calibration decisions for α=0.05** in that subset. For three operating points, scale up.
- **Annotation trust:** if you claim a reliability diagram or ECE on the OOD/hard split, the ambiguity of the
  split's labels matters — apply the `2610.36831` outcome-equivalence reasoning or at least label the
  high-equivalence items separately.

---

## 6. Domain-shift / OOD — how to study calibration breakdown AC-v2 → AndroidWorld

**The right frame:** single-step box-graded AC-v2 → multi-step, outcome-graded, live AndroidWorld is a
**covariate + label-shift** (different apps, device state, longer horizon, and the grading rule changes). A
calibration study must measure the *breakdown* and *recover* it, not just measure it once.

**Published methodology to follow:**
1. **Define IDD vs OOD explicitly.** The AndroidControl paper and `GUI-JEPA` define an IDD/OOD-union split
   (IDD = in-domain; OOD = the unseen apps/tasks/categories). Reuse that split so your OOD is comparable. Report
   IDD and OOD separately for every metric — never pool them.
2. **Measure calibration under shift with the right estimator.** Report Smooth ECE, reliability diagrams, Brier
   decomposition, and **AUROC(conf, correctness)** on IDD **and** OOD. The breakdown will typically appear as a
   reliability-diagram *bowl* (overconfident on OOD/hard), and AUROC drop (confidence stops ranking correctness).
3. **Risk control under shift using weighted conformal.** The guarantee behind conformal risk control and
   LTT requires calibration-from-P and test-from-P (`2610.19811` limitations section explicitly flags this and
   cites Tibshirani et al. 2019 / Angelopoulos et al. 2024 for shift). Under covariate shift, use **weighted
   conformal prediction** (importance-weight the calibration scores by the density ratio) or **swap/weighted
   Learn-then-Test**, so the certified false-action budget is (approximately) preserved. Report the risk before
   and after the weight correction — the gap is your "breakdown" evidence.
4. **Decompose the shift.** Use the classic OOD-uncertainty reference: `Ovadia et al. 2019, "Can you trust your
   model's uncertainty?"` (models are typically overconfident under dataset shift) and the epistemic-vs-aleatoric
   caution of `2610.01006` (a calibrated 0.5 may be a fair coin *or* an unknown fact — you must use targeted
   "is this settled / is there enough evidence" probes, not just confidence, to distinguish). In the mobile-GUI
   setting the *relabel* reason is element ambiguity, so add the `2610.36831` outcome-equivalence analysis on
   OOD items to separate "confidence is honest but the box is wrong" from "confidence is overconfident."
5. **Actionable protocol for the harness:** train on AC-v2 train + a subset of AndroidWorld templates, hold out
   templates by task family (your `plan-androidworld-harness.md` already does this), and report calibration and
   risk budget on **held-out task families** (the OOD split) with paired CIs. Carry the model to the emulator;
   compute the gate's false-action and coverage on the OOD set; if the budget is violated, apply weighted
   conformal and show recovery, and report the latency on Apple silicon.

---

## 7. The three things that most increase acceptance probability

1. **A mechanism ablation (CE vs proper-scoring vs joint) on identical data/model/splits, measured on a
   hard/OOD split, showing the confidence becomes an *informativeness* signal (AUROC/selective-risk) rather than
   merely an *average-calibrated* one.** This is the only unclaimed, causal, publishable claim, and it directly
   answers the `2610.01006` objection. (Claim N1.)
2. **Reporting the shift-aware calibrated gate on Apple silicon, with measured latency/cost, and demonstrating
   either that it beats post-hoc conformal/LTT on coverage-at-budget or honestly reporting when it does not.**
   Nobody in this class reports on-device reliability + cost, and it converts the CALIBRATION-from-theory claim
   into a SYSTEM claim. (Claims N2 + N3.)
3. **Rigorous statistics: ≥3 seeds, paired bootstrap CIs, majority-class baseline, Brier decomposition, and
   per-class + OOD diagnostics.** Your workspace VERIFICATION GATE already encodes this; actually shipping it is
   what separates "accepted" from "preprint." This is cheap and de-risks the whole paper.

## 8. The three things that most risk rejection

1. **Claiming any SOTA/parity on AC-v2 accuracy.** Mem-W-8B (68.59), GUI-Libra-8B (64.3), and — importantly —
   the 0.50B `GUI-JEPA` (which reaches 86–92% of a 7B VLM with 6% params) all sit on the accuracy axis you
   cannot own. A reviewer will see them in 30 seconds and discount the rest. **Concede the axis explicitly** and
   claim the reliability-per-cost axis instead.
2. **Claiming "single-pass decision model for mobile GUI" as novelty, or "first calibration benchmark."** Both
   are already taken (`Jev-Mobile` for the system; `TypedBench` for the benchmark; `THX-01`/`decider` for the
   recipe). Any such sentence is a magnet for a rapid rejection.
3. **Reporting AC-v2 ECE or a calibration number on a ~400-step test with a single seed and no CIs, and/or
   relying on the box-graded artifact as the core evidence.** The field's newest work (`2610.01006`,
   `2610.19811`, `2610.36831`) has *already* established that (a) aggregate ECE is uninformative, (b) box grading
   is systematically biased (though ~1% on mobile), and (c) confidence is not self-knowledge. Reporting a bare ECE
   without AUROC, Brier decomposition, seed/CI, and OOD breakdown invites the "you measured the wrong thing" review.

---

## Sources

- Jev-Mobile: `https://arxiv.org/abs/2609.30186`
- GUI-JEPA: `https://arxiv.org/abs/2610.36877`
- Right Click, Wrong Box: `https://arxiv.org/abs/2610.36831` (box-scoring grading artifact)
- Calibration and Risk-Budgeted Decisions with System One Models: `https://arxiv.org/abs/2610.19811`
- Beyond Answer Confidence (confidence ≠ self-knowledge): `https://arxiv.org/abs/2610.01006`
- TypedBench: `https://arxiv.org/abs/2610.11392`
- AgentAbstain: `https://arxiv.org/abs/2607.10059`
- CORA: `https://arxiv.org/abs/2604.09155`
- CAAG (IJCOPE): `https://doi.org/10.55041/ijcope.v2i5.306`
- THX-01: `https://huggingface.co/doofz/THX-01`
- Mem-W: `https://arxiv.org/abs/2605.09317`
- MobileRL: `https://arxiv.org/abs/2509.18119`
- GUI-Libra: `https://arxiv.org/abs/2602.22190`
- Re-TRAC: `https://arxiv.org/abs/2602.02486`
- Smooth ECE: `Blasiok & Nakkiran, ICLR 2024`
- OOD uncertainty: `Ovadia et al. 2019, "Can you trust your model's uncertainty?"`
- Weighted conformal under covariate shift: `Tibshirani et al. 2019`; `Angelopoulos & Bates 2024`
- Learn-Then-Test / risk control: `Angelopoulos et al. 2025`
