# T1 — Metrics and outcome evidence (Q1 + Q2)

**Owner:** researcher (T1)
**Brief:** `outputs/.plans/calibrated-decision-value-T1.md`
**Date:** 2026-11-07
**Question:** When a paper claims a small model's confidence is "usable", what is actually measured — and is there
any peer-reviewed evidence that abstention (act / ask / refuse) improves a real outcome rather than only a
ranking metric?

**Reading depth disclaimer (applies to every entry).** Unless a passage is quoted, I read the *abstract* and,
for a small number of sources, targeted excerpts of the *body* (TypedBench §3.5/§4 suites, Nature Medicine
abstract block, three benchmark READMEs). I did **not** read any paper end to end. Claims are labelled
`read` (excerpt inspected) vs `abstract` (abstract only) in the Caveat column. No number below is inferred from
prose; every number is either quoted or taken verbatim from the abstract I read.

---

## 0. Headline answer in one paragraph

The literature measures confidence in **four separable ways that license four different, progressively stronger
decisions**: (a) a proper score or calibration error (Brier, ECE) says the number is a probability *on average*;
(b) a ranking metric (AUROC of confidence vs correctness, AURC/E-AURC/AUGRC) says confidence *orders* examples
well, which is threshold-free and says nothing about a specific operating point; (c) risk at a fixed coverage,
or a conformal/Clopper–Pearson certificate, says *at this coverage, error ≤ r with confidence 1−δ*, which is the
first thing that licenses a deployment decision; (d) a realised-cost experiment says acting on the probabilities
was actually *better than not acting*, which almost nobody reports. Most "usable confidence" claims in the
GUI/mobile-agent literature stop at (a) or (b) — frequently at step accuracy with no calibration metric at all.
There **is** peer-reviewed outcome-level evidence that abstention changes real outcomes, but it is in adjacent
fields (mammography screening triage, LLM routing, learning-to-defer), and the best of it is explicitly
*negative*: abstention raised cancer detection *and* broke the recall noninferiority criterion [33]. Two recent
preprints go further and show that the standard machinery can be *anti-useful*: using a model's full probability
vector instead of its top answer can double realised cost [11], and certified risk bounds overrun their budget
in 9–30% of grouped deployments [8].

---

## 1. Definitions and metrics actually reported

### 1.1 The two primitive quantities: coverage and selective risk

El-Yaniv & Wiener define a selective classifier as a pair (f, g) with selection function g: X → {0,1}; coverage
φ(f,g) = E_P[g(x)] and selective risk R(f,g) = E_P[ℓ(f(x),y)·g(x)] / φ(f,g) [1]. The same two quantities appear
verbatim in the later deep-learning work [5]. Everything else in this section is a scalar summary of the
risk–coverage curve that these two quantities trace out.

**Decision licensed:** "at the coverage you will actually run, error is r" — *if* r is reported with an
interval. **Not licensed:** anything about coverage you did not report.

### 1.2 Metric-by-metric: what each one licenses and what it does not

| Metric | Formal content | Decision it licenses | What it does **not** license | Src |
|---|---|---|---|---|
| **Brier score** | Mean squared error of the probability vector vs the one-hot outcome; a strictly proper scoring rule | "On average this forecaster's probabilities are better/worse than that one's" | No threshold, no per-item risk, no coverage; two models can tie on Brier with opposite calibration/refinement profiles | [11], [3] |
| **ECE** | Binned \|accuracy − mean confidence\| per bin | "Reported probability ≈ observed frequency, in aggregate" | Not a per-decision risk; estimator is binning- and n-dependent. TypedBench reports **ECE relative to a matched, perfectly-calibrated null** (`ECE/null`, ≈1 under perfect calibration, "up to about 1.5 by chance") precisely because raw ECE has no reference point | [11] |
| **AUROC(conf, correct)** | Rank separation of correct from incorrect by confidence | "Confidence *orders* examples usefully" — this is the metric a selective classifier actually needs | No operating point. Base-rate free: two deployments with the same AUROC have different absolute error at the same coverage. Invariant to any monotone rescaling of confidence, so it **cannot detect miscalibration at all** | [5], [9] |
| **Selective risk at coverage c** | R(g) restricted to the accepted set | The first metric that licenses a real decision: "at c coverage, error ≤ r on this split" | Single working point; no interval unless reported; a cherry-picked c; carries no guarantee under shift | [1], [5] |
| **Risk–coverage curve / AURC** | Area under selective risk vs coverage | "Model A dominates B across all thresholds" — comparative, not absolute | Two RC curves need not dominate (partial order); AURC is base-rate dependent and not comparable across datasets without normalisation | [1], [5], [4] |
| **E-AURC** | AURC − AURC of the optimal (loss-ordered) ranker | Cross-dataset comparison of *excess* risk over an oracle ranker; range [0,1] | Subtracting the oracle removes the irreducible component but adds no deployment guarantee; it is still a ranking summary | [4], [5] |
| **AUGRC** (2024 proposal) | Area under a generalised risk–coverage curve meeting 5 stated requirements | Traub et al. argue AURC/E-AURC fail their own 5 requirements (task alignment, interpretability, flexibility) and propose AUGRC, "directly interpretable as the average risk of undetected failures"; on a 6-dataset × 13-confidence-function benchmark **it changes the metric ranking on 5 of 6 datasets** | Still a ranking/aggregate; still not a per-deployment certificate | [4] |
| **SGR risk target** (Geifman & El-Yaniv 2017) | Learn a threshold so that, at test time, the classifier rejects to grant a user-set risk **with high probability** | A genuine certificate: "2% error in top-5 ImageNet guaranteed with probability 99.9% at almost 60% coverage" | The guarantee is distribution-conditional and assumes the calibration/test exchangeability; it is a per-model, per-distribution claim | [2] |
| **Conformal risk control (CRC)** | Controls E[loss] for any monotone loss, tight up to O(1/n); extends to distribution shift, quantile, multiple/adversarial risk | A finite-sample, distribution-free bound on the accepted-set risk at confidence 1−δ | Requires exchangeability; the guarantee is about the *calibration→test* draw, not about the product over time | [6] |
| **Clopper–Pearson (CP) upper bound** | Exact binomial upper limit `Beta⁻¹(1−δ; x+1, k−x)` for x errors in k accepted | The admissible (never-anti-conservative) binary-loss bound used inside selective risk control, e.g. by the ICLR 2026 "SAFER" paper [7] | Admissible ≠ tight; and it certifies only under the exchangeability premise — see §4 | [7], [8] |
| **FDR-controlled threshold** (GUI grounding) | Threshold on an uncertainty score with false-discovery-rate control | "Among clicks I accept, the wrong-click rate ≤ α" — the strongest claim made in the GUI-grounding literature so far | FDR ≠ per-item correctness; a rare-but-catastrophic error is not bounded by an average rate | [32], [45] |

**Key structural point.** (a)→(b) are *ranking* statements, (c)→(d) are *decision* statements. Moving from (b)
to (c) requires an n, a split, and an exchangeability assumption. Moving from (c) to (d) requires a cost model
and, ideally, a realised-cost measurement. Almost no GUI-agent paper takes the (c)→(d) step.

### 1.3 The metric critique literature (this matters for framing)

- Traub et al. (NeurIPS 2024) enumerate why the field's favourite multi-threshold metrics fail, and show that
  swapping AURC for their AUGRC **reorders the leaderboard on 5 of 6 datasets** [4]. Any claim of the form
  "our confidence is more usable, measured by AURC" is therefore metric-choice-fragile.
- Galil, Dabbah & El-Yaniv (ICLR 2023) benchmark 523 pretrained ImageNet classifiers on AUROC, ECE, AURC and
  coverage-at-accuracy-constraint jointly, and find the metrics do not move together (distillation regimes
  dominate; a ViT reaches 99% top-1 selective accuracy at 47% coverage while an EfficientNet-V2-XL cannot meet
  the same constraints "at any level of coverage") [5]. Selection quality is a **property of the
  architecture+training regime**, not a free by-product of a good classifier.
- "What Does It Take to Build a Performant Selective Classifier?" formalises the **selective-classification gap**
  versus a perfect-ordering oracle and decomposes it into five sources; critically, "monotone post-hoc
  calibration — often believed to strengthen selective classifiers — has **limited impact** on closing this gap,
  since it rarely alters the model's underlying score ranking" [9]. This is the cleanest statement in the
  literature that **calibration ≠ selective performance**.
- TypedBench (2026) is the only benchmark I found that scores a decision model on calibration *and* selective
  prediction *and* **realised cost** on the same items [11] — see §2.7.

---

## 2. GUI / mobile agent evidence

### 2.1 Status of every system named in my brief

All fifteen named items exist and resolve to a live URL — none had to be discarded as non-existent. **But the
review-status distribution is the finding**: of the 15, roughly half are 2026 arXiv preprints or vendor pages,
and the single most on-point item for our question (AgentAbstain) is a preprint.

| # | System | What it is | Venue / status | URL |
|---|---|---|---|---|
| [15] | AndroidControl | 15,283 episodes, 833 apps, 40 categories; high- and low-level instructions | NeurIPS 2024 D&B **spotlight** (peer-reviewed) | [15] |
| [16] | AndroidWorld | Live emulator, 116 hand-crafted tasks across 20 apps, dynamically parameterised | ICLR 2025 (peer-reviewed) | [16] |
| [17] | ScreenSpot-Pro | 23 professional apps, 5 categories, 3 OS; best model 18.9% at release | ICLR 2025 workshop / arXiv 2504.07981 | [17] |
| [18] | GUI Odyssey | 8,834 episodes, 6 devices, 212 apps, 1.4K app combos, 6 cross-app task types | ICCV 2025 (peer-reviewed) | [18] |
| [19] | MobileRL | Online agentic RL for mobile GUI agents | ICLR 2026 (peer-reviewed) | [19] |
| [20] | UI-Venus 1.5 / 2 | Ant Group foundation GUI agent; 1.5: 77.6% AndroidWorld, 69.6% ScreenSpot-Pro; 2: 84.0 AndroidWorld, 80.5 OSWorld-Verified, 93.4 WebVoyager | **vendor page, self-reported**; arXiv 2609.00028 | [20] |
| [21] | GUI-Libra | Action-aware SFT + partially verifiable RL | NeurIPS 2026 poster (peer-reviewed) | [21] |
| [22] | Mem-W | Latent memory-native GUI agents | arXiv 2605.09317 (**preprint**) | [22] |
| [23] | Jev-Mobile | Low-frequency VLM planner + high-frequency typed decision executor | arXiv 2609.30186 (**preprint**) | [23] |
| [24] | GUI-JEPA | Decoder-free, candidate-conditioned JEPA action policy; 2.8–4.0× faster than Qwen2.5-VL-7B on AndroidControl (author-reported) | arXiv 2610.36877 (**preprint**) | [24] |
| [25] | THX-01 | 322M non-autoregressive decision model; 98.4% ticket accuracy, ECE 0.003, ~10 ms | **vendor** (HAL-X): HF model card + blog + API docs | [25] |
| [26] | AgentAbstain | 263 paired tasks / 42 MCP sandbox envs / 8 abstention scenarios; 17 LLMs × 4 harnesses | arXiv 2607.10059 (**preprint**) | [26] |
| [11] | TypedBench | Calibration + framing sensitivity + **realised cost** for typed decision models | arXiv 2610.11392 (**preprint**), 8 Oct 2026 | [11] |

### 2.2 The benchmark that matters for Q2: AgentAbstain

Protocol (read from abstract): paired-task design; each pair = one should-act task and one should-abstain
variant produced by a controlled perturbation of instruction, tool, or environment state; 263 pairs across 42
executable sandbox environments; 8-scenario agent-native abstention taxonomy spanning pre-execution reasoning
and runtime discovery; environment/task generation automated (AbstainGen) with deterministic replay + semantic
LLM judges, and **three independent annotators rate 94–98% of sampled tasks as well-designed**.

Results (read): across **17 frontier LLMs in 4 agent harnesses**, the best agent (Gemini 3.1 Pro) achieves only
**59.5% paired accuracy** (correct on *both* the act and abstain sides). Two claims bear directly on our
program: *"abstention capability is largely independent of general task-solving capability, indicating that
scaling task-solving alone will not close this gap"*, and a named failure mode, **"post-hoc abstention"**, in
which agents execute irreversible actions *before* recognising the abstention trigger [26].

**What this measures and what it does not.** It measures a decision (act vs not-act) at the level of *task
outcome*, which is stronger than any accuracy metric. It does **not** measure calibration of the abstention
decision in a risk-controlled sense (the paper calls it "the calibrated ability" but the headline metric is
paired accuracy, not risk-at-coverage with a guarantee), and it does not measure the cost of asking. It is a
preprint — and the closest thing to a peer-reviewed version of this specific experiment does not exist yet.

### 2.3 Peer-reviewed GUI abstention / ask-for-help results

| Work | Venue (peer-reviewed?) | What it measures | Number | Caveat |
|---|---|---|---|---|
| **OS-Kairos** [28] | Findings of ACL 2025 — **yes** | Per-step confidence annotation + confidence-driven choice between autonomous action and human intervention | "24.59%~87.29% improvements in task success rate" on a curated complex-scenario dataset and on AITZ and Meta-GUI | Improvement over what baseline, and at what interruption rate, is not stated in the abstract; a percentage *range* that wide (3.5×) is not an effect size until the protocol is read |
| **ConflictGUI / ConflictGuard** [27] | EMNLP 2026 main — **yes** | Whether agents terminate instead of executing when the instruction is infeasible/self-contradictory; new benchmark over instruction-internal conflicts and instruction–GUI-context conflicts | Reports that "agents that perform well on feasible tasks often continue to execute blindly"; ConflictGuard "improves average conflict task success rate significantly" | **No number in the abstract** — must be read from the paper body; I did not |
| **Morae** [31] | arXiv 2508.21456 — preprint (user study) | Proactive pausing at decision points for blind and low-vision users, on real web tasks | "helped users complete more tasks and select options that better matched their preferences, as compared to baseline agents, including OpenAI Operator" | The *only* item here with a real end-user population and a preference/agency outcome — but **no numbers in the abstract**, and it is a preprint |

### 2.4 GUI *grounding* with a risk guarantee (closest thing to a certificate)

**SafeGround** [32] (arXiv 2602.02419, preprint) uses distribution-aware uncertainty (spatial dispersion of
stochastic samples) + calibration to derive a test-time threshold with **statistically guaranteed FDR control**,
on ScreenSpot-Pro, across multiple grounding models. Reported: the uncertainty measure beats baselines at
separating correct from incorrect predictions, and "system-level accuracy improv[es] by up to **5.38 percentage
points** over Gemini-only inference" [32].

This is the strongest decision-level claim in the GUI literature I found: an FDR-controlled accept/reject
threshold on a real GUI benchmark. Note the two gaps: (i) the quantity is a *rate among accepted clicks*, not a
cost-weighted outcome; (ii) nothing in the abstract addresses what happens to the threshold under a new
checkpoint — which is exactly the failure mode our own program has already measured (17.3% → 23.9%).

### 2.5 What the mainstream GUI benchmarks actually score

Checked directly (README keyword scan for `abstain|abstention|calibrat|selective|confidence|risk`, zero hits in
all three):

| Benchmark | Scoring quantity | Abstention / calibration metric present? | Src |
|---|---|---|---|
| AndroidWorld | Programmatic task reward → **task success rate** | **No** (README has no abstain/calibrate/selective/confidence/risk text) | [16] |
| AndroidControl | Step-level action accuracy (high-level vs low-level instruction regimes) | **No** | [15] |
| ScreenSpot-Pro | Click-point grounding accuracy (micro-average) | **No** | [17] |

**Inference (mine, from the README scan + abstracts):** the GUI-agent leaderboard currency is *accuracy of an
action*, and the field's SOTA race (UI-Venus 2 at 84.0 AndroidWorld [20]; ScreenSpot-Pro leaderboard in the
90s [17]) is a race on that currency. A model can be SOTA on every one of these and have an unmeasured
confidence signal. This is the single most important sentence for our positioning.

### 2.6 Typed decision models on mobile: Jev-Mobile, GUI-JEPA, THX-01

- **Jev-Mobile** [23] (preprint): VLM sets local goals at low frequency; the accessibility tree enumerates a
  structured action space; **Jev** (a typed decision model) picks one candidate at high frequency. It is
  architecture-level evidence that a small decision model can *drive* a mobile agent, and the paper reports a
  large cost reduction. **Caveat:** the cost-reduction claim is contested by a third-party blog that reports the
  baseline cost rows disagree 38-fold [47] — treat the % as unverified.
- **GUI-JEPA** [24] (preprint): decoder-free latent action scoring; 2.8–4.0× faster than a Qwen2.5-VL-7B agent
  on AndroidControl at similar accuracy **by the authors' own measurement**. An independent review page reports
  that 1.6–2.6% of test clicks have no mapped candidate, candidate enumeration caps at 80, and there is **no
  termination head and no online task-completion evaluation** [48] — i.e. the "should I act at all" head is
  absent by construction.
- **THX-01** [25] (vendor): 322M params, non-autoregressive, returns typed answers; vendor claims 98.4% on an
  internal multilingual ticket benchmark vs 97.4% for "TypeSafe Jev 1.13" and 98.5% for a frontier LLM, with
  **ECE 0.003**, ~10 ms/request. These are **self-reported vendor numbers on an internal benchmark**; the
  comparison set is chosen by the vendor. Usable as existence proof of the product category, not as evidence.

### 2.7 TypedBench: the only calibration→**cost** experiment I found (read from body)

Protocol (read): seven policy-labelled programmatic generators, nine evaluation suites, n = 500 items per suite;
accuracy reported as median and range across paraphrases; differences tested by item-level paired bootstrap
(10,000 resamples) and Holm-corrected McNemar. Models on identical items: **Jev 1.13.0** (hosted, TypeSafe AI),
**Laya 0.3.4** (421M ModernBERT-large encoder + decision head), and **Kev** (LoRA+pointer head on Qwen3.5 bases
at 0.8B / 4B / 9B) [11].

Metrics (read): risk–coverage curves, AURC, and **coverage at ≤5% risk** (Geifman & El-Yaniv 2017); ordinal
proper scores (RPS, quadratic weighted kappa); and **decision value** — "realised cost per 10⁴ decisions under
the argmax and Bayes policies", where Bayes is argmin_j Σ p_t C_tj, reported as **the value of calibration =
cost(argmax) − cost(Bayes)** [11].

Result (read, quoted): on ticket priority with under-prioritising costing 2× over-prioritising per level, Bayes
"doubles the hosted model's cost and moves half of the 9B decoder's correct answers away"; the pattern holds at
every asymmetric cost ratio from 1.5:1. On phishing, where a missed attack costs as much as 20 false alarms, the
hosted model's under-confidence keeps nearly all legitimate mail above the 4.8% flagging line, so Bayes flags
**498 of 500 emails** [11].

**Why this is the load-bearing citation for our brief.** It is a direct, quantitative demonstration that a
*calibrated-looking* small decision model can make the act/ask/refuse decision **worse** when you actually use
its probabilities, and that the sign of the effect depends on the cost matrix and on the base rate. It is a
preprint (Oct 2026) and the "hosted model" and encoders are commercial, so it cannot carry a peer-reviewed claim
— but it is the exact experiment shape our program needs, run on the exact model class we care about.

### 2.8 Other GUI abstention work (existence only, not evaluated)

- **RecAgent** [29] (arXiv 2508.04025, preprint): splits uncertainty into *perceptual* (screen redundancy/noise)
  and *decision* (ambiguous task) and routes the latter to human-in-the-loop refinement; introduces a
  `ComplexAction` dataset. Abstract only; no numbers read.
- **Navi-plus** [30] (arXiv 2503.24180): "Self-Supplement GUI Navigation" — the agent asks follow-up questions
  to complete missing user information. Abstract only.
- **Towards Trustworthy GUI Agents: A Survey** [46] (arXiv 2503.23434): states the field's core limitation as
  "the absence of mechanisms for uncertainty recognition and signaling". Survey, low evidentiary weight, useful
  only as a field-consensus pointer.

---

## 3. Outcome-level evidence in adjacent domains

The question here is narrow: **is there a result where abstention/deferral changes a measured outcome, with a
baseline and an effect size?** Yes — one strong peer-reviewed one, several medium ones, and one that is a
cautionary tale about gates nobody obeys.

### 3.1 Medical screening triage — the strongest outcome result (peer-reviewed)

**AI-based triage and decision support in mammography and digital tomosynthesis: a paired, noninferiority trial
(Nature Medicine, 2026)** [33]. Prospective paired noninferiority trial, **31,301 women**, March 2022 – January
2024, ClinicalTrials.gov NCT04849776. Two reading strategies run **in parallel**: standard double-blind reading,
and partially autonomous AI-supported screening in which *cases classified by AI as low risk were assessed as
normal* and the rest were double read with AI support.

| Outcome | Result | Direction |
|---|---|---|
| Radiologist workload | **−63.6%** | primary outcome; large win |
| Cancer detection rate | **+15.2%** (95% CI 6.6–24.4), 6.3 → 7.3 per 1,000, P<0.001 | win |
| Recall rate | **+14.8%** (95% CI 9.0–20.6), **"not noninferior"** | **safety criterion failed** |
| Subgroups | DBT: workload −65.5%, detection and recall stable; DM: workload −62.1%, detection +1.6/1,000, recall +1.3% | heterogeneity |

This is the single best answer to "does abstention change a real outcome": a machine gate that removes 63.6% of
human reading, with a *higher* cancer detection rate — and it **fails its own pre-registered safety criterion on
recall**. It is the exact shape of evidence our program needs (parallel-arm, pre-registered outcome, effect size
with CI, an explicit noninferiority bar) and the exact reason a bare AUROC claim is not enough. Note the
structural inversion: here the *machine abstains from escalating to the human*, so the cost being traded is
human reading time, not a human interruption.

### 3.2 Deferral that clinicians do not obey — the cautionary result

**FILTER-SCAD** (Eur Heart J abstract supplement, 2025) [35]: a pre-test-probability + CAD-score strategy
intended to defer testing in suspected obstructive CAD; **"adherence to the recommended diagnostic strategy in
the trial was low, with less than 25% of patients deferred."** Abstract-level only; I did not read the trial
report. But the implication is direct and uncomfortable for any act/ask/refuse gate: **the gate's value depends
on the consumer of the gate's output actually following it**, and in a clinical deployment that assumption
failed by roughly 3–4×. Analogous randomised evidence on clinical prediction rules broadly finds process-of-care
effects without consistent patient-outcome effects [42].

### 3.3 LLM routing — abstention-as-escalation with a real cost, peer-reviewed

**RouteLLM** (ICLR 2025) [36]: a router decides, *before* any model is called, whether a query goes to a strong
expensive model or a weak cheap one; router trained on Chatbot Arena preference data. Reported cost-saving ratios
over GPT-4 at fixed quality thresholds: MT Bench **3.66×** at 95% GPT-4 quality; MMLU 1.41× at 92%; GSM8K 1.49×
at 87% (using the paper's stated $24.7 vs $0.24 per million tokens). The project page states **up to 85% cost
reduction while maintaining 95% GPT-4 performance** and ">40% cheaper than commercial offerings" [36].

This is an outcome-level result (dollars, at a fixed quality bar, with a baseline) for a **binary act/ask-style
escalation decision** on a cheap model's confidence. It is the closest well-cited peer-reviewed analogue to our
value claim, and the analogue is favorable: escalation gates *do* pay, when the cost asymmetry is large and the
quality bar is explicit.

### 3.4 Learning to defer — the theory, and its calibration problem (peer-reviewed)

- Mozannar & Sontag (ICML 2020) [12]: learns predictor + rejector jointly from expert decisions; reduction to
  cost-sensitive learning; consistent estimators. This is the formal version of "ask the human".
- Verma & Nalisnick (ICML 2022) [13]: finds Mozannar & Sontag's multiclass framework is **not calibrated with respect
  to expert correctness** and is "not even guaranteed to produce valid probabilities" — i.e. **the L2D
  literature has its own calibration gap**, which the authors patch with one-vs-all classifiers. Very useful for
  us: a deferral model's probabilities need their own calibration argument; it is not inherited.
- Content moderation: "Learning to Defer in Content Moderation: The Human-AI Interplay" [38] models the queue
  explicitly (delayed, congested human review) — the *only* framing I found that treats the cost of asking as a
  queueing system rather than a constant. Preprint. "Towards Unbiased and Accurate Deferral to Multiple Experts"
  [39] (ACM, peer-reviewed) extends deferral to multiple experts with differing expertise.

### 3.5 Robotics — ask-for-help with a measured success/cost trade

**Learning When to Ask for Help** [37] (arXiv 2305.16502, preprint, v3 2024): a lightweight interaction policy
decides to proceed or request expert assistance at estimated uncertainty, using implicit uncertainty from the
navigation policy's feature extractor. Reported: **+0.38 task success at a 0.3 expert-interaction rate** in
Habitat with a simulated human expert [37]. This is the cleanest "abstention buys success, and here is what it
costs" number in the set — but the expert is simulated and the venue is a preprint.

### 3.6 Autonomous driving — the cost of "ask" measured on humans, not on models

The driving literature inverts the question usefully: it measures **what a hand-off costs the human**. Examples
found (existence + framing only; abstracts/snippets, not read in full): takeover response latency and corrective
manoeuvre quality depend systematically on failure severity and driver cognitive load, formalised as a
probabilistic latency model [40]; and a CHI 2024 simulator study (N=215) compares uncertainty-display designs for
fallback drivers [41]. **Relevance:** this is where the *price of asking* is quantified empirically, and it is
non-zero and human-dependent — which is precisely the term missing from every GUI-agent abstention paper I read.

---

## 4. The gap: does anyone say confidence-ranking ≠ deployment safety?

Yes — five independent, explicit statements, all recent. This is the strongest section of evidence for our
framing.

| # | Statement (paraphrased, sourcing below) | Status |
|---|---|---|
| G1 | Monotone post-hoc calibration "has limited impact on closing" the selective-classification gap "since it rarely alters the model's underlying score ranking" — so improving ECE does not improve act/abstain ordering | arXiv 2510.20242 [9], preprint |
| G2 | Calibration and selective performance are **not** driven by the same factors: across 523 ImageNet classifiers, AUROC / ECE / AURC / coverage-at-accuracy-constraint diverge by architecture and training regime; a model that cannot meet an accuracy constraint "at any level of coverage" coexists with one at 99%/47% | ICLR 2023 [5], **peer-reviewed** |
| G3 | AURC vs AUGRC reorders the leaderboard on **5 of 6** datasets — the metric itself is not decision-stable | NeurIPS 2024 [4], **peer-reviewed** |
| G4 | Certified risk bounds **overrun their declared budget in 9–30% of grouped deployments** (unseen machine types / generators) while never overrunning under exchangeable splits: "the failure lies in the broken exchangeability premise, not in the bounds". Uncertified naive thresholding overruns in **49–73%** of synthetic trials and up to **68%** of real-data splits | arXiv 2606.15153 [8], preprint |
| G5 | In multilabel clinical ICU classification, "selective prediction can substantially degrade performance despite strong standard evaluation metrics", driven by class-dependent miscalibration where models are *more* uncertain on correct predictions for underrepresented conditions; "commonly used aggregate metrics can obscure these effects" | arXiv 2603.02719 [10], preprint |
| G6 | Abstention capability is "largely independent of general task-solving capability" across 17 frontier LLMs (best 59.5% paired accuracy) | arXiv 2607.10059 [26], preprint |
| G7 | Under asymmetric costs, using a decision model's full probability vector can be **worse** than taking its top answer (cost doubles; half the 9B decoder's correct answers move away) | arXiv 2610.11392 [11], preprint |

**Synthesis (my inference, labelled).** G1+G2 mean *calibration work does not transfer to selective performance
by default*. G3 means *the selective metric you report may not survive a metric change*. G4 means *the guarantee
you report may not survive a population change*. G5+G6 mean *the aggregate number may hide the failure, and
capability scaling will not fix abstention*. G7 means *even a correct probability vector can make the decision
worse if the cost matrix is wrong*. Together these form a coherent negative result about the field's current
practice — and each is a citable, recent, specific claim, not a rhetorical gesture.

**What I did NOT find:** any paper that measures calibration→outcome transfer *for a GUI/mobile agent*. Not one.
That is the hole our program can occupy.

---

## 5. Master evidence table

| # | Claim | Value | Source | Peer-reviewed? | Protocol / split / n | Caveat |
|---|---|---|---|---|---|---|
| 1 | Selective classifier = (f,g); coverage φ=E[g], selective risk R=E[ℓ·g]/φ | definitions | El-Yaniv & Wiener, JMLR 2010 [1] | **yes** | theory + synthetic/benchmark | foundational; no numbers needed |
| 2 | SGR grants a user-set risk with high probability | 2% top-5 ImageNet error at 99.9% confidence, ~60% coverage | Geifman & El-Yaniv, NeurIPS 2017 [2] | **yes** | ImageNet val, CIFAR | abstract; per-model, per-distribution certificate |
| 3 | SelectiveNet optimises risk–coverage end-to-end | "consistently improved risk-coverage trade-off" | Geifman & El-Yaniv, ICML 2019 [3] | **yes** | several classification + regression datasets | abstract; origin of AURC/E-AURC reporting |
| 4 | AURC/E-AURC fail 5 stated requirements; AUGRC fixes them | **ranking changes on 5/6 datasets** | Traub et al., NeurIPS 2024 [4] | **yes** | 6 datasets × 13 confidence functions | abstract; leaderboard fragility is the point |
| 5 | Calibration and selective performance diverge across models | ViT: 99% top-1 selective acc @ 47% coverage; 95% @ 80%; competing EfficientNet-V2-XL cannot meet it at any coverage | Galil, Dabbah & El-Yaniv, ICLR 2023 [5] | **yes** | 523 pretrained ImageNet classifiers; metrics AUROC, ECE, AURC, coverage | abstract; ImageNet only |
| 6 | Conformal risk control bounds expected monotone loss, tight to O(1/n) | O(1/n) | Angelopoulos et al., ICLR 2024 [6] | **yes** | theory + vision/NLP worked examples | requires exchangeability |
| 7 | Clopper–Pearson is the admissible binary-loss bound in selective risk control | `Beta⁻¹(1−δ; x+1, k−x)` | ICLR 2026 paper ("SAFER") [7]; see also [8] | **yes (venue)** | formula as stated in the paper | **citation not fully verified** — PDF text extraction failed, authors unknown; admissible ≠ tight; premise-bound |
| 8 | Certified bounds fail under grouped deployment | **9–30%** budget overrun under grouped deployment; naive thresholding 49–73% of synthetic trials, ≤68% real splits; CP/WSR certify coverage where Hoeffding certifies none | 2606.15153 [8] | **no (preprint)** | 4 calibration rules; n=200 calibration points; anomalous-sound detection + AI-image forensics | preprint; two signal domains only |
| 9 | Post-hoc calibration barely changes selective ranking | qualitative ("limited impact") | 2510.20242 [9] | **no (preprint)** | synthetic two-moons + vision & language benchmarks | preprint; no effect size in the part I read |
| 10 | Selective prediction can *hurt* in clinical multilabel tasks | qualitative: "substantially degrade performance"; class-dependent miscalibration | 2603.02719 [10] | **no (preprint, v4)** | multimodal ICU data, multiple unimodal/multimodal models | preprint; I did not read the numbers |
| 11 | Using a decision model's probabilities can cost more than its top answer | Bayes **doubles** hosted model cost on ticket priority; moves **half** of the 9B decoder's correct answers away; phishing: flags **498/500** emails at 20:1 cost | TypedBench, 2610.11392 [11] | **no (preprint, Oct 2026)** | n=500/suite; 9 suites; paired bootstrap 10k; Holm McNemar; cost ratios from 1.5:1 | **most decision-relevant source in this note**; preprint; commercial models |
| 12 | Learning-to-defer can be learnt consistently | theory + experiments | Mozannar & Sontag, ICML 2020 [12] | **yes** | cost-sensitive reduction | abstract |
| 13 | L2D probabilities are not calibrated wrt expert correctness | qualitative; "not even guaranteed to produce valid probabilities" | Verma et al., ICML 2022 [13] | **yes** | multiclass L2D theory + experiments | abstract |
| 14 | Reasoning fine-tuning *degrades* abstention | **−24%** average abstention, incl. math/science | AbstentionBench, 2506.09038 [14] | **no (preprint)** | 20 datasets, 20 frontier LLMs, LLM judge with 88% human agreement | abstract; judge-based scoring |
| 15 | AndroidControl dataset scale | 15,283 episodes, 833 apps, 40 categories | Li et al., NeurIPS 2024 D&B [15] | **yes** | human raters; high/low-level instructions | dataset paper |
| 16 | AndroidWorld scores task success only | 116 tasks, 20 apps, dynamic parameters | Rawles et al., ICLR 2025 [16] | **yes** | live emulator, programmatic reward | README keyword scan: no abstention/calibration metric |
| 17 | ScreenSpot-Pro is hard; no abstention metric | 23 apps / 5 categories / 3 OS; best model 18.9% at release | Li et al., ICLR 2025 workshop [17] | workshop | dataset card size category 1K–10K examples (exact count not re-verified) | README keyword scan: no abstention/calibration metric; leaderboard entries now reported in the high 80s–90s, but these are self-submitted and I did not verify them |
| 18 | GUI Odyssey cross-app mobile corpus | 8,834 episodes, 6 devices, 212 apps, 1.4K app combos | Lu et al., ICCV 2025 [18] | **yes** | 6 cross-app task types | dataset paper |
| 19 | MobileRL: online RL SOTA on AndroidWorld/AndroidLab | qualitative in abstract | ICLR 2026 [19] | **yes** | online env sampling, difficulty-adaptive | accuracy-only currency |
| 20 | UI-Venus 1.5 / 2 SOTA numbers | 1.5: 77.6 AndroidWorld, 69.6 ScreenSpot-Pro; 2: 84.0 / 80.5 OSWorld-Verified / 93.4 WebVoyager | Ant Group vendor pages [20] | **no (vendor, self-reported)** | vendor-selected evaluation config | vendor card; not independent |
| 21 | GUI-Libra: CoT hurts grounding; RLVR insufficient for offline→online | qualitative | NeurIPS 2026 [21] | **yes** | 81K SFT corpus + conservative RL | abstract/poster only |
| 22 | Mem-W: latent memory-native GUI agents | qualitative | 2605.09317 [22] | **no (preprint)** | long-horizon web/mobile/desktop | accuracy-only currency |
| 23 | Jev-Mobile: typed executor + VLM planner, large cost cut | ~73% cost cut **claimed** | 2609.30186 [23] | **no (preprint)** | not verified | baseline cost rows disputed 38× by a third party [47] |
| 24 | GUI-JEPA: 2.8–4.0× faster than Qwen2.5-VL-7B on AndroidControl | 2.8–4.0× | 2610.36877 [24] | **no (preprint)** | same-hardware latency, author-reported; no termination head, no online eval [48] | author-reported; privileged a11y-tree input |
| 25 | THX-01: 322M decision model, 98.4% ticket acc, ECE 0.003, ~10 ms | as stated | HAL-X vendor card/blog/API docs [25] | **no (vendor)** | internal multilingual ticket benchmark, 2,843 tickets (per blog) | self-reported; vendor-chosen baselines |
| 26 | Agents do not know when not to act | best **59.5%** paired accuracy (Gemini 3.1 Pro); abstention independent of task skill; "post-hoc abstention" failure mode | AgentAbstain, 2607.10059 [26] | **no (preprint)** | 263 paired tasks, 42 sandbox envs, 8 scenarios, 17 LLMs × 4 harnesses, 94–98% annotator agreement | preprint; paired accuracy ≠ risk-at-coverage; no cost-of-asking term |
| 27 | GUI agents over-comply with infeasible instructions | "significant" improvement with ConflictGuard; no number in abstract | ConflictGUI, EMNLP 2026 [27] | **yes** | new conflict benchmark; 5 agents | numbers not read |
| 28 | Per-step confidence → adaptive human intervention | **24.59%–87.29%** task success improvement | OS-Kairos, Findings ACL 2025 [28] | **yes** | curated complex-scenario set + AITZ + Meta-GUI | range too wide to be an effect size without protocol; baseline unclear from abstract |
| 29 | GUI uncertainty split into perceptual vs decision, HITL refinement | qualitative | RecAgent, 2508.04025 [29] | **no (preprint)** | ComplexAction dataset | abstract only |
| 30 | Agents can ask follow-up questions to complete intent | qualitative | Navi-plus, 2503.24180 [30] | **no (preprint)** | self-supplement navigation task | abstract only |
| 31 | Proactive pausing improves real user outcomes for BLV users | "complete more tasks" + better preference match vs baselines incl. OpenAI Operator | Morae, 2508.21456 [31] | **no (preprint)** | real-world web tasks, BLV participants | **only real end-user outcome in the GUI set**; no numbers in abstract |
| 32 | FDR-controlled accept/reject threshold on GUI grounding | up to **+5.38 pp** system accuracy over Gemini-only | SafeGround, 2602.02419 [32] | **no (preprint)** | ScreenSpot-Pro, multiple grounding models | FDR is a rate, not a cost; no threshold-transfer test |
| 33 | Abstention-like machine triage changes screening outcomes | workload **−63.6%**; cancer detection **+15.2%** (6.3→7.3/1,000, P<0.001); recall **+14.8%, not noninferior** | Nature Medicine 2026 [33] | **yes** | prospective paired noninferiority, **31,301 women**, Mar 2022–Jan 2024, NCT04849776 | parallel-arm, not randomised between arms; the gold-standard outcome evidence |
| 34 | Risk-based deferred testing improves efficiency | "improved clinical efficiency (catheterization without obstructive CAD) with no statistically significant impact on…" | PRECISE RCT, JAMA Cardiology [34] | **yes** | 2,103 participants | I read only a partial abstract sentence; **do not quote numbers from this until re-read** |
| 35 | Clinicians do not follow deferral recommendations | **<25%** of patients deferred | FILTER-SCAD, EHJ abstract [35] | peer-reviewed abstract supplement | trial subanalysis | abstract-level; the "gate ignored" failure mode |
| 36 | Router-based escalation saves real money at fixed quality | **3.66×** (MT Bench @95%), 1.41× (MMLU @92%), 1.49× (GSM8K @87%); "up to 85% cost reduction" | RouteLLM, ICLR 2025 [36] | **yes** | Chatbot Arena preference data; $24.7 vs $0.24 /M tokens | cost ratios depend on the token-price assumption |
| 37 | Asking for help buys robot task success | **+0.38** success at **0.3** interaction rate | 2305.16502 [37] | **no (preprint)** | Habitat, simulated human expert | simulated expert; navigation domain |
| 38 | Deferral in moderation is a queueing problem | qualitative | 2402.12237 [38] | **no (preprint)** | posts, delayed human review | treats cost-of-asking explicitly — rare |
| 39 | Deferral to multiple biased experts can be learnt | qualitative | ACM (FAccT) 2021 [39] | **yes** | synthetic + content-moderation dataset | abstract |
| 40 | Hand-off cost to a human is measurable and load-dependent | response latency model vs failure severity + cognitive load | PLOS/PMC [40] | **yes** | driving-simulator experiments | snippet-level read; no numbers extracted |
| 41 | Uncertainty displays change fallback-driver behaviour | N=215, 3 between-group designs | CHI 2024 [41] | **yes** | high-fidelity simulator | snippet-level read |
| 42 | Diagnostic prediction rules often change process, not outcomes | systematic review of RCTs | Diagn Progn Res 2017 [42] | **yes** | RCTs of diagnostic CPRs | abstract |

`[47]`, `[48]` are secondary/critical commentary and are cited only as caveats, never as evidence.

---

## 6. What is NOT measured anywhere (in the sources I checked)

Phrased as "not found", not "does not exist" — a negative claim needs a wider sweep than I ran.

1. **No GUI/mobile benchmark reports risk at fixed coverage (selective risk / AURC / AUGRC) as a headline.**
   AndroidWorld, AndroidControl and ScreenSpot-Pro score success rate or click accuracy; their READMEs contain
   no abstention, calibration, selective, confidence or risk vocabulary [15][16][17].
2. **No GUI-agent paper reports the price of asking** — interruption latency, user abandonment, or a queue — as
   a term in the objective. Morae [31] changes the interaction and reports user preference, but not a cost;
   SafeGround [32] controls an FDR, not a cost; TypedBench [11] prices decisions, but for tickets/phishing, not
   screens.
3. **No paper measures calibration→outcome transfer for a mobile GUI agent.** The nearest are a grounding-level
   FDR certificate [32], a paired-task abstention benchmark with no guarantee [26], and TypedBench's cost
   experiment on non-GUI tasks [11].
4. **No GUI paper reports threshold transfer across checkpoints or populations.** The one paper that measures
   this generally — bounds overrunning in 9–30% of grouped deployments [8] — is not about GUIs. Our own 17.3% →
   23.9% transfer failure appears to have no published analogue in this domain.
5. **No test–retest or re-certification protocol** is published for any GUI accept/reject threshold.
6. **No peer-reviewed version of an abstention benchmark for tool-using agents.** AgentAbstain is a preprint
   [26]; AbstentionBench is a preprint aimed at LLM answering, not acting [14].
7. **No study isolates the three-way act / ask / refuse choice with distinct costs for ask vs refuse in a GUI
   setting.** All the GUI work I found is effectively binary (act vs don't act, or act vs ask).
8. **No paper reports a per-item harm-weighted loss** (irreversibility as a category, not a constant) for an act/
   ask/refuse gate. Driving research measures hand-off cost to humans [40][41]; nothing carries that into a GUI
   gate.

---

## 7. Direct answers to Q1 and Q2

**Q1 (definition/measurability).** "Usable confidence" is measured in four distinct tiers, and most claims stop
in the weakest two. Within GUI/mobile agents specifically, the modal claim is *step accuracy* with **no
calibration metric at all**; the strongest published claim is a **calibration-then-FDR-controlled-threshold**
result on ScreenSpot-Pro (+5.38 pp) [32]. Under the plan's falsifier — "if the literature only reports ECE and
never a coverage-conditioned risk, our contribution is larger than assumed" — the literature does report
coverage-conditioned risk in the *selective-prediction* subfield (AURC, S-AURC, AUGRC, SGR risk targets,
CRC bounds), so the falsifier does **not** fire; but for GUI agents it effectively does. The contribution is
therefore larger than assumed *within the domain we are claiming*, not relative to the selective-prediction
literature as a whole.

**Q2 (outcome evidence).** Peer-reviewed outcome-level evidence that abstention changes an outcome **exists**
[33][36][37-ish][39][42], and the strongest instance is a 31,301-woman paired noninferiority trial in which a
machine gate cut radiologist workload 63.6%, raised cancer detection 15.2%, **and failed its recall
noninferiority criterion** [33]. For GUI agents, there is **no peer-reviewed outcome-level abstention result**;
the best available is a preprint paired-task benchmark showing frontier agents reach only 59.5% paired accuracy
[26], plus an ACL Findings result with a task-success improvement that the abstract reports only as a 24.59%–
87.29% range [28]. Under the plan's falsifier — "if every result is AUROC-only, the field has a metric-to-outcome
gap" — the falsifier fires for GUI agents and does not fire for the adjacent domains.

---

## 8. Exact search terms used

**Web (Exa) searches, verbatim:**
1. `selective prediction AURC excess AURC E-AURC definition Geifman El-Yaniv`
2. `conformal risk control selective classification Angelopoulos Bates`
3. `risk-coverage curve selective risk at coverage definition paper`
4. `AndroidControl benchmark GUI agent dataset paper`
5. `AndroidWorld dynamic benchmark autonomous agents`
6. `GUI agent abstention ask for help uncertainty`
7. `ScreenSpot-Pro high-resolution GUI grounding benchmark`
8. `GUI-Odyssey mobile GUI agent benchmark paper`
9. `MobileRL mobile GUI agent reinforcement learning`
10. `GUI-Libra GUI agent training framework`
11. `UI-Venus GUI agent model`
12. `"AgentAbstain" benchmark abstention`
13. `"TypedBench" benchmark`
14. `"GUI-JEPA"`
15. `"THX-01" agent benchmark`
16. `"Mem-W" GUI agent memory`
17. `"Jev-Mobile" decision model mobile`
18. `calibration does not predict selective prediction performance 523 ImageNet classifiers`
19. `miscalibration selective prediction ranking correlation weak`
20. `learning to defer to human expert cost-sensitive Mozannar Sontag`
21. `AbstentionBench evaluating abstention large reasoning models`
22. `RouteLLM cost reduction benchmark peer reviewed routing`
23. `Clopper-Pearson upper bound selective risk control conformal guarantee`
24. `selective prediction deferral medical imaging reduces clinical harm effect size`
25. `content moderation human-in-the-loop routing precision recall tradeoff study`
26. `autonomous driving fallback to human driver uncertainty take-over outcome study`
27. `AI triage deferral randomised trial workload reduction radiology`

**Scholarly database searches:** `semanticscholar: "selective classification risk-coverage curve excess AURC"`
(returned 0; API rate-limited), `semanticscholar: "selective classification"` (fell back to OpenAlex),
`openalex: "selective classification"`, `openalex: "overcoming common flaws evaluation selective classification
systems"`.

**Direct fetches:** arXiv abstract pages for 1705.08500, 1901.09192, 2208.02814, 2302.11874, 2305.16502,
2402.12237, 2407.01032, 2410.05407, 2506.09038, 2508.04025, 2508.21456, 2510.20242, 2602.02419, 2603.02719,
2606.15153, 2607.10059, 2610.11392; arXiv HTML body for 2610.11392 (targeted excerpts); ACL Anthology pages for
2025.findings-acl.348 and 2026.emnlp-main.1028; nature.com abstract block for s41591-026-04277-x; raw READMEs
for android_world, android_control, ScreenSpot-Pro.

---

## 9. Coverage status

| Item | Checked directly? | Status |
|---|---|---|
| 1. Metric definitions & what each licenses | Yes — 10 primary sources, abstracts + one body | **done** |
| 2. GUI/mobile agent evidence (15 named systems) | Yes — all 15 resolve to a live URL; read abstract/protocol for 13 | **done** (numbers for ConflictGUI, RecAgent, Navi-plus, Morae not extracted: abstract only) |
| 3. Adjacent-domain outcome evidence | Partly — medical (strong, peer-reviewed), routing (peer-reviewed), robotics (preprint), deferral theory (peer-reviewed), driving (snippet-level only) | **done with one gap** |
| 3b. Content moderation outcome with effect size | Only structural sources; no harm-level effect size found | **needs follow-up** |
| 4. The gap (calibration≠deployment safety) | Yes — 7 explicit statements, 3 peer-reviewed | **done** |
| Peer-review status labelling | Yes, per row | **done** |

**Tooling failures recorded.** (a) Two `fetch_content` calls in `mode:"answer"` (TypedBench, AgentAbstain) returned
"Page answer model returned an empty response". Worked around by fetching with `curl` and parsing locally; no
content was lost. (b) The ICLR 2026 "SAFER" PDF [7] would not yield extractable text, so its **title, authors and
full citation remain unverified**; only the Clopper–Pearson equation quoted in a search snippet is attributable, and
that attribution is flagged in §1.2 and §5. (c) One mis-attribution was caught during verification and corrected:
the ICML 2022 calibrated-learning-to-defer paper is by **Verma & Nalisnick** (two authors), not three. No dead
links were encountered among the 40+ URLs in §5.

**Honest weaknesses of this note.**
1. Almost nothing here is a full-text read. Where a number is missing (ConflictGUI, Morae, RecAgent, PRECISE) it
   is because the abstract did not contain it, and I did not go into the body — the numbers may well exist.
2. The most decision-relevant source (TypedBench [11]) and the most on-point benchmark (AgentAbstain [26]) are
   both **preprints**, one a month old. The brief's pre-registered value test asked for "a peer-reviewed,
   outcome-level result where abstention changes success/harm" — that exists only in medicine [33], not in our
   domain.
3. I did not audit whether the 2026 arXiv identifiers used here (2602/2603/2605/2606/2607/2609/2610/2612) reflect
   real submissions or a synthetic environment. They resolve and serve consistent metadata, which is the check I
   could run; a verifier should re-resolve each before any of them reaches the paper.
4. `[34]` (PRECISE) is cited from a partial sentence; do not propagate a number from it.

---

## 10. Sources

1. El-Yaniv, R. & Wiener, Y. — *On the Foundations of Noise-free Selective Classification*, JMLR 11:1605–1641, 2010 — https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf
2. Geifman, Y. & El-Yaniv, R. — *Selective Classification for Deep Neural Networks*, NeurIPS 30:4878–4887, 2017 — https://arxiv.org/abs/1705.08500
3. Geifman, Y. & El-Yaniv, R. — *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*, ICML 2019 — https://arxiv.org/abs/1901.09192
4. Traub, J., Bungert, T., Lüth, C., Baumgartner, M., Maier-Hein, K., Maier-Hein, L., Jäger, P. — *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*, NeurIPS 2024 — https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf
5. Galil, I., Dabbah, M. & El-Yaniv, R. — *What Can We Learn From The Selective Prediction And Uncertainty Estimation Performance Of 523 ImageNet Classifiers*, ICLR 2023 — https://arxiv.org/abs/2302.11874
6. Angelopoulos, A., Bates, S., Fisch, A., Lei, L., Schuster, T. — *Conformal Risk Control*, ICLR 2024 — https://arxiv.org/abs/2208.02814
7. ICLR 2026 paper listed as *SAFER* — uses the Clopper–Pearson exact upper bound, eq. (4), for risk control — https://proceedings.iclr.cc/paper_files/paper/2026/file/582e9771ac8527cb6390e5e9444a0fee-Paper-Conference.pdf (**PDF text extraction failed — scanned/encoded stream; title, author list and full citation NOT verified. Re-resolve before citing.**)
8. *False Sense of Safety in Selective Signal Classification: Auditing Bound Tightness and Exchangeability for Risk Control*, arXiv 2606.15153, Jun 2026 — https://arxiv.org/abs/2606.15153
9. *What Does It Take to Build a Performant Selective Classifier?*, arXiv 2510.20242, Oct 2025 — https://arxiv.org/abs/2510.20242
10. *An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification*, arXiv 2603.02719, Mar 2026 (v4) — https://arxiv.org/abs/2603.02719
11. Sharma, R., Duncan, A. B., Marceau Caron, G., Vollmer, S. J. — *TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models*, arXiv 2610.11392, 8 Oct 2026 — https://arxiv.org/abs/2610.11392
12. Mozannar, H. & Sontag, D. — *Consistent Estimators for Learning to Defer to an Expert*, ICML 2020 — https://proceedings.mlr.press/v119/mozannar20b.html
13. Verma, R. & Nalisnick, E. — *Calibrated Learning to Defer with One-vs-All Classifiers*, ICML 2022 — https://proceedings.mlr.press/v162/verma22c.html
14. *AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions*, arXiv 2506.09038, Jun 2025 — https://arxiv.org/abs/2506.09038
15. Li, W. et al. — *On the Effects of Data Scale on UI Control Agents* (AndroidControl), NeurIPS 2024 D&B — https://proceedings.neurips.cc/paper_files/paper/2024/file/a79f3ef3b445fd4659f44648f7ea8ffd-Paper-Datasets_and_Benchmarks_Track.pdf
16. Rawles, C. et al. — *AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents*, ICLR 2025 — https://arxiv.org/abs/2405.14573
17. Li, K. et al. — *ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use*, 2025 — https://arxiv.org/abs/2504.07981
18. Lu, Q. et al. — *GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices*, ICCV 2025 — https://openaccess.thecvf.com/content/ICCV2025/html/Lu_GUIOdyssey_A_Comprehensive_Dataset_for_Cross-App_GUI_Navigation_on_Mobile_ICCV_2025_paper.html
19. *MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents*, ICLR 2026 — https://arxiv.org/abs/2509.18119
20. Venus Team, Ant Group — *UI-Venus-1.5* and *UI-Venus-2* (vendor) — https://ui-venus.github.io/UI-Venus-1.5/ ; https://ui-venus.github.io/UI-Venus-2/
21. Yang, R. et al. — *GUI-Libra: Data-Efficient Post-Training for Reliable Reasoning-and-Acting in Native GUI Agents*, NeurIPS 2026 — https://neurips.cc/virtual/2026/poster/153725
22. *Mem-W: Latent Memory-Native GUI Agents*, arXiv 2605.09317 — https://arxiv.org/abs/2605.09317
23. *Jev-Mobile: Jev as an Executor for Mobile GUI Agents*, arXiv 2609.30186 — https://arxiv.org/abs/2609.30186
24. *GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents*, arXiv 2610.36877 — https://arxiv.org/abs/2610.36877
25. HAL-X — *THX-01* (vendor model card, blog, API reference) — https://huggingface.co/doofz/THX-01 ; https://api.hal-x.ai/docs/thx-01/
26. *AgentAbstain: Do LLM Agents Know When Not to Act?*, arXiv 2607.10059, Jul 2026 — https://arxiv.org/abs/2607.10059
27. *Do GUI Agents Know When Not to Act? Enabling Conflict-Aware Termination for Multimodal GUI Agents*, EMNLP 2026 (main) — https://aclanthology.org/2026.emnlp-main.1028/
28. *OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents*, Findings of ACL 2025 — https://aclanthology.org/2025.findings-acl.348/
29. *Uncertainty-Aware GUI Agent: Adaptive Perception through Component Recommendation and Human-in-the-Loop Refinement*, arXiv 2508.04025 — https://arxiv.org/abs/2508.04025
30. *Navi-plus: Managing Ambiguous GUI Navigation Tasks with Follow-up Questions*, arXiv 2503.24180 — https://arxiv.org/abs/2503.24180
31. *Morae: Proactively Pausing UI Agents for User Choices*, arXiv 2508.21456 — https://arxiv.org/abs/2508.21456
32. *SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration*, arXiv 2602.02419, Feb 2026 — https://arxiv.org/abs/2602.02419
33. *AI-based triage and decision support in mammography and digital tomosynthesis for breast cancer screening: a paired, noninferiority trial*, Nature Medicine, 2026 — https://www.nature.com/articles/s41591-026-04277-x
34. *Comparison of an Initial Risk-Based Testing Strategy vs Usual Testing in Stable Symptomatic Patients With Suspected Coronary Artery Disease: The PRECISE Randomized Clinical Trial*, JAMA Cardiology — https://jamanetwork.com/journals/jamacardiology/fullarticle/2808765
35. *Optimizing resource consumption with pre-test probability and the acoustic-based rule-out CAD-score — insights from the FILTER-SCAD trial*, Eur Heart J abstract supplement, 2025 — https://doi.org/10.1093/eurheartj/ehaf784.4494
36. Ong, I., Almahairi, A. et al. — *RouteLLM: Learning LLMs to Route from Preference Data*, ICLR 2025 — https://proceedings.iclr.cc/paper_files/paper/2025/file/5503a7c69d48a2f86fc00b3dc09de686-Paper-Conference.pdf
37. *Learning When to Ask for Help: Efficient Interactive Navigation via Implicit Uncertainty Estimation*, arXiv 2305.16502 — https://arxiv.org/abs/2305.16502
38. *Learning to Defer in Content Moderation: The Human-AI Interplay*, arXiv 2402.12237 — https://arxiv.org/abs/2402.12237
39. *Towards Unbiased and Accurate Deferral to Multiple Experts*, ACM, 2021 — https://doi.org/10.1145/3461702.3462516
40. *Predicting takeover response to silent automated vehicle failures*, PMC7703974 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7703974/
41. *An Eye Gaze Heatmap Analysis of Uncertainty Head-Up Display Designs for Conditional Automated Driving*, CHI 2024 — https://dl.acm.org/doi/10.1145/3613904.3642219
42. *Systematic review of the effects of care provided with and without diagnostic clinical prediction rules*, Diagn Progn Res 2017 — https://doi.org/10.1186/s41512-017-0013-2
45. *Selective Conformal Risk Control*, arXiv 2512.12844 — https://arxiv.org/html/2512.12844v2
46. *Towards Trustworthy GUI Agents: A Survey*, arXiv 2503.23434 — https://arxiv.org/html/2503.23434v2
47. Blolabel — *Jev-Mobile's 73% cost cut rests on a table whose baseline cost rows disagree 38-fold* (secondary, critical) — https://blolabel.ai/blog/jev-mobile-savings-measured-on-different-tasks/
48. Auto Research review page — *GUI-JEPA: One-pass GUI action selection with candidate-conditioned latents* (secondary, critical) — https://daiwk.github.io/auto-research/foundation-models/2610.36877-gui-jepa/README-en/
