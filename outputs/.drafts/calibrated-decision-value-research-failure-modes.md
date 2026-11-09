# T3 — Production failure modes of a calibrated act / ask / refuse gate

**Question (from `outputs/.plans/calibrated-decision-value-T3.md`, i.e. Q5 of the master plan):**
under what conditions does a calibrated act/ask/refuse gate silently become unsafe — and has anyone
published a method that survives it?

**Scope note.** This note covers the *external* literature only. The program's own measured
non-transfer figure (17.3% → 23.9% risk across checkpoints, quoted in the master plan) is used here as
context, **not re-audited** — that is T4's job. Every claim below is tied to a fetched, checked source;
venue is only asserted where I verified it from the PDF itself, a PMLR page, or arXiv metadata.

---

## 0. Bottom line

1. **No published method gives a distribution-free gate guarantee under arbitrary distribution shift.**
   Every method in this literature either (a) requires labelled or unlabelled data *from the target
   distribution*, (b) requires a known/estimable density ratio plus an invariance assumption, (c) requires
   a stationarity/mixing assumption, or (d) restricts the conditioning set to a finite class. That is not a
   gap in my search; several of the papers say so explicitly [2][4][5][8][17].
2. **The guarantees that do exist are marginal, not per-decision and not per-checkpoint.** Conformal risk
   control bounds `E[L]` over the draw of calibration data *and* the test point [6]. Adaptive conformal
   inference explicitly "provides no information about the marginal coverage frequency at a single time
   step" [3].
3. **Under shift, the guarantee degrades by an unobservable quantity.** The two sharpest results are
   coverage ≥ 1−α−Σᵢ wᵢ·d_TV(residual vectors) [2] and E[L] ≤ α + B·Σᵢ TV(Zᵢ, Z_{n+1}) [6]. Both are
   *assumption-free in form and unusable in practice*, because the total-variation term is not estimable
   from unlabelled deployment data.
4. **Group-conditional coverage is provably impossible distribution-free** [7], and measured subgroup
   failures are large — worst-group accuracy moves from 65% to 75% while the second-worst group moves
   77% → 95% at the same abstention rate [18].
5. **Coverage can collapse before accuracy does.** In sequentially fine-tuned LLMs, measured coverage fell
   0.92 → 0.61 while accuracy stayed within three points of baseline [25] (preprint). That is precisely the
   silent-failure mode a per-checkpoint certificate would not catch.
6. **Monitoring is feasible and evidence-backed, but detection ≠ validity.** The best current tools detect
   *changepoints* with anytime false-alarm control [22][21]; they do not tell you the gate's risk.

---

## 1. Failure-mode table

| # | Mode | What breaks | Published evidence (measured) | Remedy | Remedy evidence |
|---|---|---|---|---|---|
| M1 | **Distribution shift / threshold non-transfer** (covariate, concept, checkpoint, population) | Exchangeability fails; the calibrated quantile no longer targets the test residual distribution. Coverage gap is bounded only by a total-variation term nobody can measure in production. | Coverage ≥ 1−α−Σwᵢd_TV(R(Z),R(Zᵢ)) [2]. Risk ≤ α + B·ΣTV(Zᵢ,Z_{n+1}) [6]. Non-adaptive CP "fails to maintain the desired 90% coverage level, incurring large troughs" during time-zone change on election-night data, while ACI holds ≈90% [3]. Under simulated changepoints, standard CP **failed nearly all** marginal/conditional coverage backtests [23]. | (a) Weighted CP/CRC with the likelihood ratio; (b) ACI (online α update); (c) weighted-quantile "beyond exchangeability" conformal; (d) re-calibrate on target labels | (a) works **only if** the ratio is known or estimable and P(Y\|X) is invariant [4][5]; (b) holds long-run average coverage under a stationary, spectrally-gapped state chain, and γ trades stability against adaptation (γ=0.005 in their experiments) [3]; (c) works, at the cost of interval width, and the TV term remains [2]; (d) exact but requires target labels — see M5/§4 |
| M2 | **Correlated / clustered errors** (episodes, users, tasks, screens) | Rows are not exchangeable; the *effective* calibration sample size is the number of independent units, not the number of rows. Nominal n badly overstates evidence. | With *b* independent units each contributing *m* near-identical scores (n=mb), the realised-coverage law is Beta(⌈k/m⌉, b+1−⌈k/m⌉) — i.e. it behaves like *b* independent units [11] (preprint). Standard exchangeability is stated to fail "when groups of observations come from distinct distributions, such as multiple observations on each patient" [9]. Hierarchical exchangeability is a strictly different notion [10]. | Hierarchical/`second-moment coverage` CP [10]; CDF pooling, single and repeated subsampling for two-layer models [9]; cluster-aware resampling | Dunn et al. compare coverage vs. average set size across three approaches and recommend CDF pooling when asymptotic coverage suffices, repeated subsampling when a guarantee is required [9]. Second-moment coverage gives "uniformly small conditional miscoverage rates" at a width cost that is "very mild in cases where the fitted model is accurate" [10]. **No method was found that restores exact marginal coverage from clustered calibration alone.** |
| M3 | **Base-rate / prevalence change** | A *fixed* threshold on a calibrated score delivers a different PPV/FPR mix when prevalence moves. If the gate's action cost depends on precision (which an act/ask/refuse gate's does), the gate's economics change even if its score is perfectly calibrated. | Exact identity: FPR = (p/(1−p))·((1−PPV)/PPV)·(1−FNR). Therefore equal PPV with unequal prevalence *forces* unequal FPR/FNR — applied to Broward County COMPAS data with prevalence 51% (Black defendants) vs 39% (White defendants) [17]. | Group/class-conditional (Mondrian) calibration [27]; prior-shift reweighting with unlabelled target data [5]; prevalence-aware thresholding per segment | Mondrian CP gives per-category validity by construction [27]; label-shift reweighting restores nominal coverage **only** under q(x\|y)=p(x\|y) and with unlabelled target data — and the authors report that using *estimated* rather than oracle weights leaves a "slightly downgraded" coverage [5]. |
| M4 | **Selective risk / abstention feedback** | The accepted subset is not a random sample. Selective classification changes the composition of what the gate still acts on; at low coverage "the more confident the model is on worst-group points, the more likely it is incorrect." | Worst-group accuracy *decreases* with coverage on CelebA; on MultiNLI, full→50% coverage lifts worst group 65%→75% but second-to-worst 77%→95% [18]. | Distributionally-robust training before applying selection; risk control on the *selected* subset (SCRC) | Jones et al. show that training distributionally-robust models so that full-coverage accuracies match across groups makes selective classification "uniformly improve each group" [18]. SCRC formalises a two-stage select-then-risk-control gate: the transductive variant (SCRC-T) preserves exchangeability by choosing thresholds jointly over calibration and test points and gets exact finite-sample guarantees; the calibration-only variant (SCRC-I) gets PAC-style guarantees and is more efficient, at slightly more conservative risk control [13] (preprint). |
| M5 | **Measurement artefacts** (ambiguous ground truth, multiple valid actions, scorer gaming, box/point tolerance) | The number in the certificate is partly a property of the metric/labelling procedure, not the model. A gate certified at 20% risk may be measuring label ambiguity. | Mean ≥3.3% label errors across 10 standard test sets; ≥6% of the ImageNet validation set; only 51% of algorithmically-flagged candidates were truly mislabelled; **ResNet-18 beats ResNet-50 once the mislabelled-test prevalence rises by just 6%** [14]. On re-annotation, **24,889 of 50,000** ImageNet val images needed re-evaluation; ~29% of re-annotated images are multi-object or carry a synonym label; on class *desktop computer* a model picking a correct label **at random** scores 29.9% while NoisyStudent scores 71.1% [15] (preprint). A **one-pixel** translation of boxes drops mAP by **8.4%** (23.1% on small objects) [16] (preprint). Optimising against a proxy scorer raises the proxy while the gold score peaks and then falls [28]. | Multi-reference / set-valued ground truth; tolerant matching with published tolerance; held-out "gold" audit labels; freeze-the-metric and audit it like the model | Beyer et al. show re-annotation *does* change the ranking-relevant conclusions and become the better target [15]; Northcutt et al. show corrected-label evaluation changes model choice [14]. **I found no published method that separates metric error from model error inside a single deployed gate's risk certificate.** |
| M6 | **Silent degradation of the *reported* risk over a retraining sequence** | Per-checkpoint re-certification measures the checkpoint you have; the *next* fine-tune can move calibration without moving accuracy, so nothing alerts you. | Coverage 0.92 → 0.61 on the worst task in a sequential fine-tuning study while accuracy stayed "within three points of baseline"; coverage loss ≈3.4×±0.5× the accuracy loss across seeds [25] (preprint, 3 model families, 8 task sequences, classification/multiple-choice). | Re-fit the conformal threshold on a small held-out buffer after every update ("calibration replay") | Authors report restoration "to within two points of nominal at buffer size m=200", with <1% of the memory of ordinary experience replay and no extra gradient cost [25]. **Single-preprint evidence; not independently replicated in anything I found.** |
| M7 | **Detecting any of the above in production** | By construction the failure is silent: model confidence is not a drift detector. | "ML systems … tend to fail silently" [19]. Measured detector power: with a *universal* perturbation the detection accuracy is **0.26 at 10 test samples, 0.84 at 1,000, 0.86 at 10,000** when 100% of the test data is perturbed — but only **0.11 / 0.54 / 0.66** at the same sample counts when just **10%** of the data is perturbed; small shifts are "hard to detect even with many samples" [19]. | Anytime-valid sequential monitoring of the risk itself; conformal test martingales; backtests on coverage | Podkopaev & Ramdas give time-uniform confidence sequences for "is the target risk materially above the source risk", valid under continuous monitoring without inflating false alarms, and applicable when labels arrive late or in batches [21]. WATCH (weighted conformal test martingales) reports average detection delay **>3× faster** than the sequential-testing baseline and an unnecessary-alarm rate **≈⅓** of the standard conformal test martingale, at an anytime false-alarm rate of 0.01 [22]. |

---

## 2. Guarantee ledger — every guarantee, and the assumption it stands on

The T3 rule was: *a guarantee without an assumption is a red flag.* Here is the ledger.

| Guarantee (as published) | Assumption it needs | What it does **not** cover | Source |
|---|---|---|---|
| P{Y ∈ Ĉ(X)} ≥ 1−α | Exchangeability of the n+1 points; symmetric fitting algorithm | Anything conditional; anything after the calibration distribution changes | [1][2] |
| P{Y ∈ Ĉ(X)} ≥ 1−α−Σwᵢd_TV(R(Z),R(Zᵢ)) | **None on the distribution** — but the TV term is unobservable | Per-input validity; a usable numeric bound | [2] |
| E[L(λ̂)] ≤ α | Lᵢ non-increasing in λ, right-continuous, Lᵢ(λ_max) ≤ α, sup Lᵢ ≤ B < ∞; exchangeability used in the proof | Per-decision risk; any guarantee after shift; expects over both calibration and test draw | [6] |
| E[L] ≤ α + B·ΣᵢTV(Zᵢ,Z_{n+1}) | As above | A small, checkable bound — B can be 1, i.e. 100% slack | [6] |
| Weighted CP under covariate shift | q(x)≠p(x) but **q(y\|x)=p(y\|x)**; likelihood ratio known or estimable from unlabelled target data | Label/concept shift; miscalibrated ratios | [4] |
| Weighted CP / calibration under label shift | q(y)≠p(y) but **q(x\|y)=p(x\|y)**; unlabelled target data | Concept shift; poor weight estimation (they measure a coverage penalty with estimated weights) | [5] |
| Long-run coverage ≈ 1−α at any interval | {A_t} stationary with non-zero absolute spectral gap; small, smooth shift | Coverage at any single time step — stated explicitly | [3] |
| Coverage over every group in a finite collection | Target family of covariate shifts must be **finite-dimensional**; exact only there. Infinite-dimensional families get a *quantified error*, not a guarantee | Arbitrary conditioning; per-input validity | [8] |
| Exact conditional coverage, distribution-free | **Impossible.** Theorem 2 lower-bounds interval width for any (1−α,δ)-conditionally-covering method at the trivial width of marginal CP run at 1−αδ | — | [7] |
| Hierarchical / second-moment coverage | Hierarchical exchangeability; independent repeated measurements for the stronger property | Arbitrary cluster dependence | [10] |
| SCRC-T exact finite-sample coverage+risk on the accepted subset | Conditional exchangeability between calibration and test **within the selected set** | Shift; the selection stage's own instability | [13] |
| SCRC-I PAC-style risk control | Exchangeability, δ budget | Same | [13] |
| Anytime-valid "is the risk higher than source?" test | Labels (possibly delayed/batched) from deployment; bounded loss for the confidence sequence | Shift that does not yet show in observed labels | [21] |
| False-alarm-controlled changepoint detection | Within-segment exchangeability for the betting martingale | That the change is *harmful* to the gate (they add a second martingale to diagnose type) | [22] |
| Near-nominal coverage **conditional on the validation draw** (not just marginal) | Exchangeability; if you want approximate conditional coverage, quantile regression on a held-out set is minimax optimal | Any shift; any per-input reading | [12] |

**Reading of the ledger.** Only two entries are assumption-free in form — [2] and [6, Prop. 3] — and both
pay for it by putting an unmeasurable distance term into the guarantee. Every operative remedy purchases
its guarantee with either target-side data or an invariance/stationarity assumption. Per the T3 rules, I
flag: *there is no published distribution-free guarantee that survives an unanticipated population or
checkpoint change.*

---

## 3. Detail on the two failure modes most likely to bite this program

### 3.1 Checkpoint/population change (M1 + M6)

The program already has a measured non-transfer (17.3% → 23.9% across checkpoints; master plan §Q5).
The literature supplies three things around that number:

- **A reason it is expected, not surprising.** The conformal guarantee is an average over the joint draw of
  calibration and test points [1][6]. A new checkpoint is a new calibration problem; nothing transfers by
  construction.
- **A stronger, scarier version of the phenomenon.** Shihab et al. find coverage can collapse *before*
  accuracy does: 0.92 → 0.61 with accuracy within three points, coverage loss ≈3.4× accuracy loss [25]
  (preprint). If that generalises, "accuracy unchanged, so the gate is unchanged" is a false inference —
  and it is exactly the inference a per-checkpoint recertification process will make if it only re-checks
  accuracy.
- **A cheap mitigation with a measured, but unreplicated, effect.** Refit the threshold on a small
  per-task buffer after each update; reported to restore coverage to within 2 points at m=200, <1% of
  experience-replay memory [25]. Treat as a hypothesis worth testing on our own artifacts, not a
  established result.

### 3.2 Clustered evaluation units (M2)

This is the failure mode most likely to invalidate *our own* certificate arithmetic. If the evaluation
screens/episodes come in clusters (same app, same user, same task template), the effective sample size is
the number of clusters, not the number of rows. The transported-beta preprint makes the arithmetic
explicit: with *b* clusters contributing *m* near-identical scores, realised coverage follows
Beta(⌈k/m⌉, b+1−⌈k/m⌉) [11]. For a 20%-risk certificate at 40.7% coverage, the relevant question is not
"how many screens did we score" but "how many *independent* screens did we score" — and I found no
published method that recovers exact marginal coverage from clustered calibration alone; the fixes
([9][10]) change the estimand (second-moment coverage, hierarchical coverage) or require repeated
subsampling [9].

Its companion quantitative result is also directly usable: in a half-normal scale-shift model, a **10%
increase in test-score scale corresponds to ≈3.2 percentage points of coverage loss at γ=0.9** [11]
(preprint). That is the right order of magnitude to compare against our 17.3→23.9 shift (a 6.6-point
move), i.e. our observed non-transfer is consistent with a modest residual-scale change, not necessarily
a catastrophic one — *this is my inference, not a claim from the source*.

---

## 4. Remedy ledger — proposed vs. measured

| Remedy | Proposed in | Measured? | What the measurement actually shows |
|---|---|---|---|
| Weighted CP (covariate shift) | [4] | Yes, simulations + real | Works when ratio is known; requires P(Y\|X) invariance |
| Label-shift reweighting | [5] | Yes, toy + wine-quality | Nominal coverage restored with oracle weights; **slightly degraded with estimated weights** (BBSE) |
| Adaptive Conformal Inference | [3] | Yes, market volatility + election night | Non-adaptive CP shows "large troughs" at time-zone change; ACI holds ≈90%. γ=0.005 chosen by hand; authors list γ selection as open [3 §7] |
| Non-exchangeable / weighted-quantile CP | [2] | Yes, electricity + election forecasting | "substantially less loss of coverage" under drift; theoretical ceiling still the TV gap |
| Multi-valid conditional CP | [8] | Yes | Exact for finite shift classes; quantified (not zero) error for infinite classes |
| Group/hierarchical CP | [9][10] | Yes, simulations (+ medical-style two-layer data) | Coverage/width trade-off reported; second-moment coverage at mild width cost when the model is good |
| Distributionally-robust training + selection | [18] | Yes, 5 vision/NLP datasets | Selection then uniformly improves each group **after** equalising full-coverage accuracy — a precondition we do not currently have |
| SCRC (select-then-risk-control) | [13] | Yes, 2 public datasets | Both variants hit target coverage and risk; SCRC-T exact, SCRC-I slightly conservative |
| Sequential risk tracking | [21] | Yes, simulated + real | Time-uniform validity under continuous monitoring; needs labels |
| Conformal test martingales (WATCH) | [22] | Yes, MNIST-C/CIFAR-10-C/MEPS/tabular | >3× faster detection than baseline; ⅓ the unnecessary alarms; diagnoses concept vs covariate shift |
| Backtesting CP coverage (VaR-style) | [23] | Yes, synthetic + electricity + financial series | Exposes that "nominal empirical coverage" hides conditional-coverage failure; adaptive methods can overreact and fail independence tests |
| ERT conditional-coverage metric | [24] | Yes, benchmark of CP methods | Higher statistical power than CovGap; diagnostic only — no guarantee |
| Calibration replay across fine-tunes | [25] | Yes (single preprint) | Coverage within 2 points of nominal at m=200 |
| Label-free performance estimation (AutoEval) | [26] | Yes (CVPR), regression on distribution-distance features | "Reasonable and promising prediction of the model accuracy"; **no error figures were available from the page I could read** — treat as unquantified |
| Label-free OOD accuracy prediction from ID/OOD agreement | [20] | Yes, 6 benchmarks | Agreement between model pairs is near-linear ID↔OOD with R² 0.974–0.999, and the slope/bias matches the accuracy line — so OOD accuracy is *predictable without labels*, but only as a fitted linear relation across a family of models, not as a per-gate guarantee |

**Remedies proposed but not measured (as far as I could verify):** weighted conformal *risk* control with
an *estimated* covariate shift — the CRC paper explicitly leaves this "for future research" [6]; and
"per-checkpoint certification as a product property", which no paper I found even formulates.

---

## 5. What a deployer must monitor — derived only from cited work

Each item names the work it comes from. Nothing here is invented.

1. **Coverage / risk on a rolling window of *labelled* deployment data, with a time-uniform test** — not a
   one-shot empirical-coverage check. Rationale: the guarantee is one-shot; repeated naive checks inflate
   false alarms [21], and there are anytime-valid tools for exactly this [21][22].
2. **Changepoint alarms with a controlled anytime false-alarm rate**, so alarms do not decay into noise —
   at 0.01 in [22]; alarm rate matters as much as delay (⅓ of baseline unnecessary alarms [22]).
3. **Conditional-coverage diagnostics, not just marginal coverage.** Marginal empirical coverage "hides"
   violations; formal backtests detected near-total failure of standard CP under changepoints, and
   independence/conditional failures in adaptive variants whose marginal coverage looked fine [23]. ERT
   gives a higher-power diagnostic than CovGap [24].
4. **Effective sample size of the monitoring stream** — count independent units (users, episodes, tasks),
   not rows [11][9].
5. **Prevalence per segment**, tracked as a first-class quantity, because PPV/FPR are mechanically tied to
   it by FPR = (p/(1−p))·((1−PPV)/PPV)·(1−FNR) [17].
6. **Worst-group metrics under the *current* abstention rate**, because abstention can *lower* worst-group
   accuracy even as average accuracy rises [18].
7. **A gold-label audit cadence independent of the shipping metric** — label errors are 3.3–6% of standard
   test sets and flip model rankings at +6% mislabelled prevalence [14]; a one-pixel matching tolerance
   change moves mAP 8.4% [16]. The metric must be audited as carefully as the model.
8. **Calibration of the *pipeline*, not just the weights** — the score depends on prompt/decoding/deployment
   configuration, so a config change is a distribution change (this is asserted in a preprint I did not
   independently verify [14a] — flagging as low confidence, see §7).
9. **Re-fit cadence tied to model updates, not to calendar time** — coverage can move on an update with no
   accuracy signal [25].
10. **At least one label-free tripwire between label batches** — ID↔OOD agreement gives a fitted linear
    predictor of OOD accuracy with R² 0.974–0.999 across the benchmarks tested [20]; cheap, but it predicts
    a *ranking* relationship across models, so it can flag "something moved" without certifying risk.

---

## 6. Is a per-checkpoint certificate defensible as a product property?

**No — not as a statistical guarantee, and it should not be marketed as one.**

Here is the blunt version. A certificate produced by conformal risk control is a statement about
`E[L(λ̂)]` where the expectation runs over the calibration draw *and* a test point drawn from the same
distribution [6]. When we ship checkpoint *k* and certify it, we have certified one number about one
distribution at one instant. The instant is gone the moment traffic composition moves, the moment
prevalence moves, the moment the next fine-tune lands, and — most relevant to this program — the moment
the evaluation clusters stop looking like the deployment clusters. The two theorems that speak to
"what if the distribution changes" do not rescue us; they convert the guarantee into
"1−α minus an unmeasurable total-variation term" [2][6]. And the one guarantee a product marketer would
actually want to advertise — "when it says ask, it is right X% of the time" — is a *conditional*
guarantee, which is provably impossible distribution-free at non-trivial width [7].

So a per-checkpoint certificate is defensible only as a **documented, dated, falsifiable measurement with a
stated scope**: "on calibration set C drawn from distribution D on date T, at abstention rate 40.7%, the
estimated selective risk was 20% with the following interval and the following audit trail." The strongest
statistical form that claim can honestly take is the *sample-conditional* one: split conformal gives
near-nominal coverage with high probability **conditional on the validation data**, not merely on average
over it [12]. That is a real improvement in honesty — it says the number we measured is close to the number
we would get on a fresh draw from D — and it changes nothing about what happens when D changes. It becomes
*indefensible* the moment it is restated as
"this gate operates at 20% risk" without D and T attached, because the claim then asserts exactly the
transfer property that the literature says does not hold and that our own checkpoint measurement
(17.3% → 23.9%) already contradicts.

The nearest thing to a defensible product property in the literature is not a certificate at all — it is a
*contract plus an instrument*: weighted/conditional calibration under a **named, checked** invariance
assumption [4][5][8], plus a live anytime-valid monitor with a stated false-alarm rate [21][22], plus a
published re-fit trigger [25]. Even then, note that no paper I found demonstrates that combination
end-to-end on a shipped act/ask/refuse gate. That absence is itself a finding ([13] is the closest
published analogue of the gate, and it stops at two public datasets).

The honest product sentence is therefore: **"Certified per checkpoint, monitored continuously, invalidated
by any distributional change we cannot verify we did not have."**

---

## 7. Dead ends, failures, and things I could not verify

| Item | What happened |
|---|---|
| Semantic Scholar API | Rate-limited (HTTP 429) on both attempts; tool silently fell back to OpenAlex. No `SEMANTIC_SCHOLAR_API_KEY`/`OPENALEX_API_KEY` configured, so OpenAlex ran on the anonymous budget. Citation-sorted search was therefore **not** available. |
| `ar5iv.labs.arxiv.org` answer-mode fetches | Two calls (`2202.13415`, `2106.00170`) returned "Page answer model returned an empty response". Worked around by downloading PDFs and extracting text with `pypdf`. |
| `proceedings.mlr.press/v267/prinster25a/prinster25a.pdf` (and v266, v306) | HTTP 404. Real PDFs are mirrored at `raw.githubusercontent.com/mlresearch/v.../assets/...`. Recorded so the verifier does not repeat the failure. |
| `pypdf` text extraction | "Exceeded 5000 form XObject invocations" warning on `2512.12844` and `braun26c`; text captured but some figure content skipped. Numbers inside figures are therefore **not** quoted in this note. |
| Exact numeric coverage deltas for label-shift CP | The deltas are in figures (Fig. 1c/1d); not extractable from text. I quote the authors' textual statement ("slightly downgraded performance" with estimated weights) rather than inventing numbers [5]. |
| Exact numeric deltas for SCRC | Abstract/text report "achieve the target coverage and risk levels … nearly identical performance" on two datasets; per-dataset deltas sit in tables I could not reliably parse [13]. |
| Gao et al. reward-model overoptimization | Full paper read; the result is reported as fitted curves (gold score peaks then declines) rather than a single scalar. I therefore cite the phenomenon, not a number [28]. |
| `[14a]` "Conformalized Large Language Models under Configuration Shift" (arXiv 2608.01460) | Surfaced in search as claiming nonconformity scores depend on prompt/decoding/config. **I did not fetch or read it.** Used only as a flagged, low-confidence pointer in §5 item 8, not as evidence. |
| AutoEval quantification | CVPR page read; it reports "a reasonable and promising prediction" with **no error figures on the page**. Recorded as unquantified [26]. |
| Grey literature deliberately excluded | A blog post on anytime-valid coverage monitoring (`burning-cost.github.io`, 2026-04-04) was surfaced and its argument is consistent with [21][22], but it is undated-author grey literature with no primary citation; excluded. |
| **Not found (searched, no result)** | Any published work that (a) formulates "per-checkpoint certification" as a product property, (b) measures coverage transfer *across checkpoints* in a shipped system, or (c) demonstrates a shift-robust gate end-to-end with an external audit. Absence of these in my search is reported as a gap, not as proof of nonexistence. |

---

## 8. Exact search terms used

**`web_search` (Exa provider behind the `web_search` tool, run in parallel pairs):**
1. `SCRC conformal risk control sample-conditional`
2. `conformal prediction beyond exchangeability Barber Candes Ramdas Tibshirani clustered data`
3. `Selective Classification Can Magnify Disparities Across Groups Jones Sagawa Koh arXiv`
4. `conformal prediction clustered data coverage fails naive exchangeability effective sample size group`
5. `monitoring conformal prediction coverage in deployment sequential test audit miscoverage detection`
6. `AutoEval predicting model accuracy without labels distribution shift Deng Zheng`
7. `conformal prediction recalibration after model update retraining coverage degrades checkpoint`
8. `object detection mAP IoU threshold effect on reported accuracy metric sensitivity study`
9. `Dunn Wasserman Ramdas "two-layer hierarchical" distribution-free prediction sets JASA`
10. `Vovk Mondrian conformal prediction taxonomy conditional coverage book Algorithmic Learning in a Random World`

**`feynman_science_database_search`:**
- `source=semanticscholar`, `"conformal prediction under distribution shift coverage guarantee"` → fell back to OpenAlex
- `source=semanticscholar`, `"adaptive conformal inference online coverage distribution shift Gibbs Candes"` → fell back to OpenAlex
- `source=arxiv`, `arxiv_get_papers:2107.07511,2202.13415,2106.00170,1904.06019`
- `source=arxiv`, `arxiv_get_papers:2512.12844,2503.00220,2306.06342,2103.14749`
- `source=arxiv`, `arxiv_get_papers:2110.06177,2406.17819`

**Direct PDF fetches (`https://arxiv.org/pdf/<id>`, plus PMLR mirrors)** for: 2202.13415, 2106.00170,
2103.03323, 2208.02814, 2305.12616, 1903.04684, 2512.12844, 1810.11953, 2206.13089, 2010.14134,
2210.10760, 1701.08230, 1703.00056, 2006.07159, 2306.06342, 1809.07441, 2605.19024, 2604.23987,
2206.10107, and PMLR v267/v266/v306 (retrieved via the `raw.githubusercontent.com/mlresearch/...`
mirror). `1701.08230` (Corbett-Davies, KDD'17) was fetched and read but is **not cited** in the final
text — the base-rate claim is carried by Chouldechova [17], which states it as an exact identity.

**Recorded mistake (kept for the verifier):** I initially trusted the search engine's association of
`arXiv:2010.15778` with "Selective Classification Can Magnify Disparities Across Groups". The PDF turned out
to be *"Contextual BERT: Conditioning the Language Model Using a Global State"*. The correct ID is
**2010.14134**. Any note downstream that cites 2010.15778 for selective classification is wrong.

---

## 9. Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction…*, arXiv:2107.07511 | https://arxiv.org/abs/2107.07511 | Coverage guarantees are marginal, distribution-free, non-asymptotic; require exchangeability | primary tutorial/review; I read the arXiv version only, published version not checked | high for content |
| 2 | Barber, Candès, Ramdas, Tibshirani, *Conformal prediction beyond exchangeability*, arXiv:2202.13415 (search metadata reports Annals of Statistics 2023, DOI 10.1214/23-aos2276; I read the arXiv PDF) | https://arxiv.org/abs/2202.13415 | Coverage ≥ 1−α−Σwᵢd_TV(R(Z),R(Zᵢ)); upper bound 1−α+w̃_{n+1}+Σ…; assumption-free but unobservable gap | primary | high (theorem quoted verbatim from PDF p.15) |
| 3 | Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift*, arXiv:2106.00170 | https://arxiv.org/abs/2106.00170 | Long-run average coverage under stationary spectrally-gapped state chain; "provides no information about the marginal coverage frequency at a single time step"; γ=0.005; open problem = γ selection | primary | high (theorems + §7 quoted) |
| 4 | Tibshirani, Barber, Candès, Ramdas, *Conformal Prediction Under Covariate Shift*, arXiv:1904.06019 | https://arxiv.org/abs/1904.06019 | Weighted CP needs known/estimable likelihood ratio; requires covariate-shift form | primary | high |
| 5 | Podkopaev & Ramdas, *Distribution-free UQ for classification under label shift*, arXiv:2103.03323 | https://arxiv.org/abs/2103.03323 | Label shift degrades coverage/calibration; reweighting needs q(x\|y)=p(x\|y) + unlabelled target data; estimated weights slightly degrade coverage | primary | high (assumptions quoted, numbers in figures) |
| 6 | Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, arXiv:2208.02814 | https://arxiv.org/abs/2208.02814 | Thm 1: E[L]≤α under monotone/right-continuous/bounded loss; Prop 3: ≤α+B·ΣTV; tight up to 2B/(n+1); estimated-covariate-shift CRC left to future work | primary | high (theorems quoted verbatim) |
| 7 | Barber, Candès, Ramdas, Tibshirani, *The limits of distribution-free conditional predictive inference*, arXiv:1903.04684 | https://arxiv.org/abs/1903.04684 | Thm 2: (1−α,δ)-conditional coverage forces width ≥ inf_c (1−α)/(1−cα)·L_P(1−cαδ) — i.e. only the trivial marginal solution | primary | high |
| 8 | Gibbs, Cherian, Candès, *Conformal Prediction with Conditional Guarantees*, arXiv:2305.12616 | https://arxiv.org/abs/2305.12616 | Exact finite-sample coverage over all shifts only for finite-dimensional shift classes; infinite classes get quantified error only | primary | high |
| 9 | Dunn, Wasserman, Ramdas, *Distribution-Free Prediction Sets for Two-Layer Hierarchical Models*, JASA 118(544) 2023, arXiv:1809.07441 | https://arxiv.org/abs/1809.07441 | Exchangeability fails for grouped observations; CDF pooling vs repeated subsampling trade-off; recommend pooling for asymptotic, subsampling for guarantee | primary, peer-reviewed | high |
| 10 | Lee, Barber, Willett, *Distribution-free inference with hierarchical data*, arXiv:2306.06342 | https://arxiv.org/abs/2306.06342 | Hierarchical exchangeability; "second-moment coverage" gives uniformly small conditional miscoverage at mild width cost when the model is accurate | primary | high |
| 11 | *Conformal Prediction via Transported Beta Laws*, arXiv:2605.19024 | https://arxiv.org/abs/2605.19024 | Clustered calibration: n=mb near-identical scores ⇒ realised coverage ~Beta(⌈k/m⌉, b+1−⌈k/m⌉), effective units = b; 10% scale increase ⇒ ≈3.2pp coverage loss at γ=0.9 | preprint | medium (single very recent preprint; the Beta identity is elementary and I checked the derivation in-text) |
| 12 | Duchi, *A Few Observations on Sample-Conditional Coverage in Conformal Prediction*, arXiv:2503.00220 | https://arxiv.org/abs/2503.00220 | Split conformal attains near-nominal coverage conditional on the validation draw w.h.p.; quantile regression on held-out data is minimax optimal for approximate conditional coverage | primary | high |
| 13 | Xu, Guo, Wei, *Selective Conformal Risk Control*, arXiv:2512.12844 | https://arxiv.org/abs/2512.12844 | Two-stage select-then-risk-control; SCRC-T exact finite-sample via joint thresholds, SCRC-I PAC-style; both hit targets on two datasets | preprint | medium-high (closest published analogue of the gate) |
| 14 | Northcutt, Athalye, Mueller, *Pervasive Label Errors in Test Sets…*, NeurIPS 2021 D&B, arXiv:2103.14749 | https://arxiv.org/abs/2103.14749 | ≥3.3% mean test-set errors; ≥6% ImageNet val; 51% of flagged candidates truly mislabelled; ResNet-18 > ResNet-50 at +6% mislabelled prevalence | primary, peer-reviewed | high |
| 15 | Beyer et al., *Are we done with ImageNet?*, arXiv:2006.07159 | https://arxiv.org/abs/2006.07159 | 24,889/50,000 val images needed re-annotation; ~29% multi-object/synonym; random-correct-label baseline 29.9% vs NoisyStudent 71.1% on *desktop computer* | preprint ("under review" on the PDF) | high for the quoted numbers |
| 16 | Borji, *Sensitivity of Average Precision to Bounding Box Perturbations*, arXiv:2206.10107 | https://arxiv.org/abs/2206.10107 | One-pixel shift ⇒ −8.4% mAP (−23.1% small objects); GT-boxes-as-predictions −23%/−41.7% | preprint (single author) | medium (plausible, unreplicated) |
| 17 | Chouldechova, *Fair prediction with disparate impact*, arXiv:1703.00056 (Big Data 2017) | https://arxiv.org/abs/1703.00056 | FPR = (p/(1−p))·((1−PPV)/PPV)·(1−FNR); equal PPV + unequal prevalence ⇒ unequal error rates; Broward prevalence 51% vs 39% | primary, peer-reviewed | high (equation quoted verbatim) |
| 18 | Jones, Sagawa, Koh, Kumar, Liang, *Selective Classification Can Magnify Disparities Across Groups*, ICLR 2021, arXiv:2010.14134 | https://arxiv.org/abs/2010.14134 | Worst-group accuracy can fall as coverage falls (CelebA); MultiNLI 65%→75% vs 77%→95% at 50% coverage; DR training fixes it | primary, peer-reviewed (header states ICLR 2021) | high |
| 19 | Rabanser, Günnemann, Lipton, *Failing Loudly*, arXiv:1810.11953 | https://arxiv.org/abs/1810.11953 | ML systems "fail silently"; detection accuracy 0.26→0.84→0.86 at 10/1,000/10,000 samples for a universal shift affecting 100% of test data, but 0.11→0.54→0.66 when only 10% is affected; small shifts hard to detect | primary | high (Table 2b quoted row by row) |
| 20 | Baek, Jiang, Raghunathan, Kolter, *Agreement-on-the-Line*, arXiv:2206.13089 | https://arxiv.org/abs/2206.13089 | ID↔OOD agreement is near-linear; R² 0.974–0.999 across CIFAR-10.1/10.2, ImageNetv2, CIFAR-10C-Fog/Snow, fMoW-WILDS — label-free OOD accuracy *estimation* | primary | high (table quoted); applicability to a risk gate unproven |
| 21 | Podkopaev & Ramdas, *Tracking the risk of a deployed model and detecting harmful distribution shifts*, ICLR 2022, arXiv:2110.06177 | https://arxiv.org/abs/2110.06177 | Sequential, time-uniform tests for "did target risk materially increase"; continuous monitoring without false-alarm inflation; handles delayed/batched labels | primary, peer-reviewed (arXiv comment field) | high |
| 22 | Prinster, Han, Liu, Saria, *WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales*, ICML 2025, PMLR v267 | https://proceedings.mlr.press/v267/prinster25a.html | Anytime-false-alarm-controlled changepoint monitoring; >3× faster ADD than sequential-testing baseline; ≈⅓ unnecessary alarms vs CTM; diagnoses concept vs extreme covariate shift | primary, peer-reviewed | high (numbers quoted from PDF text) |
| 23 | Retzlaff, Schlembach, Bams, Dreesen, *Testing Marginal and Conditional Coverage… via VaR Backtesting*, CP 2025, PMLR v266 | https://proceedings.mlr.press/v266/retzlaff25a.html | Under changepoints standard CP fails nearly all coverage tests; NexCP+WLS passes marginal but fails independence & conditional coverage completely (overreaction) | primary, peer-reviewed proceedings | high |
| 24 | Braun, Holzmüller, Jordan, Bach, *Conditional Coverage Diagnostics for Conformal Prediction*, PMLR v306 | https://proceedings.mlr.press/v306/braun26c.html | ERT: conditional-coverage violation ⇔ some classifier beats target risk; higher power than CovGap; diagnostic only | primary, proceedings | medium-high (venue numbering recent; peer-review status assumed from PMLR) |
| 25 | Shihab, Akter, Sharma, *Continual Calibration: Coverage Can Collapse Before Accuracy in Lifelong LLM Fine-Tuning*, arXiv:2604.23987 | https://arxiv.org/abs/2604.23987 | Coverage 0.92→0.61 with accuracy within 3 points; coverage loss ≈3.4×±0.5× accuracy loss; calibration replay restores within 2 points at m=200 | preprint | medium (single preprint, small author team; strongest direct evidence for M6 but unreplicated) |
| 26 | Deng & Zheng, *Are Labels Always Necessary for Classifier Accuracy Evaluation?*, CVPR 2021 | https://openaccess.thecvf.com/content/CVPR2021/html/Deng_Are_Labels_Always_Necessary_for_Classifier_Accuracy_Evaluation_CVPR_2021_paper.html | AutoEval: estimate accuracy on unlabelled test sets by regressing on distribution-distance features from a synthetic meta-dataset | primary, peer-reviewed | medium — **no error figures available on the page read** |
| 27 | Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World* (2nd ed., Springer) | https://link.springer.com/book/10.1007/978-3-031-06649-8 | Mondrian / category-wise conformal prediction: per-category validity by construction | primary (monograph) | high for existence and the method; **I did not re-read the Mondrian chapter** — used only for the standard attribution |
| 28 | Gao, Schulman, Hilton, *Scaling Laws for Reward Model Overoptimization*, arXiv:2210.10760 | https://arxiv.org/abs/2210.10760 | Optimizing against a proxy scorer raises the proxy while the gold score peaks and then declines (Goodhart); fitted curve forms, no single scalar reported | primary | high for the phenomenon |

---

## 10. Sources

1. Angelopoulos & Bates — https://arxiv.org/abs/2107.07511
2. Barber, Candès, Ramdas, Tibshirani — https://arxiv.org/abs/2202.13415
3. Gibbs & Candès — https://arxiv.org/abs/2106.00170
4. Tibshirani, Barber, Candès, Ramdas — https://arxiv.org/abs/1904.06019
5. Podkopaev & Ramdas (label shift) — https://arxiv.org/abs/2103.03323
6. Angelopoulos, Bates, Fisch, Lei, Schuster (Conformal Risk Control) — https://arxiv.org/abs/2208.02814
7. Barber, Candès, Ramdas, Tibshirani (limits of conditional inference) — https://arxiv.org/abs/1903.04684
8. Gibbs, Cherian, Candès (conditional guarantees) — https://arxiv.org/abs/2305.12616
9. Dunn, Wasserman, Ramdas (two-layer hierarchical, JASA) — https://arxiv.org/abs/1809.07441 · https://doi.org/10.1080/01621459.2022.2060112
10. Lee, Barber, Willett (hierarchical data) — https://arxiv.org/abs/2306.06342
11. Conformal Prediction via Transported Beta Laws — https://arxiv.org/abs/2605.19024
12. Duchi (sample-conditional coverage) — https://arxiv.org/abs/2503.00220
13. Xu, Guo, Wei (SCRC) — https://arxiv.org/abs/2512.12844
14. Northcutt, Athalye, Mueller (label errors) — https://arxiv.org/abs/2103.14749
15. Beyer et al. (ReaL ImageNet) — https://arxiv.org/abs/2006.07159
16. Borji (AP sensitivity) — https://arxiv.org/abs/2206.10107
17. Chouldechova (disparate impact / base rates) — https://arxiv.org/abs/1703.00056
18. Jones, Sagawa, Koh, Kumar, Liang (selective classification disparities) — https://arxiv.org/abs/2010.14134
19. Rabanser, Günnemann, Lipton (Failing Loudly) — https://arxiv.org/abs/1810.11953
20. Baek, Jiang, Raghunathan, Kolter (Agreement-on-the-Line) — https://arxiv.org/abs/2206.13089
21. Podkopaev & Ramdas (tracking deployed risk) — https://arxiv.org/abs/2110.06177
22. Prinster, Han, Liu, Saria (WATCH, ICML 2025) — https://proceedings.mlr.press/v267/prinster25a.html
23. Retzlaff et al. (VaR backtesting, CP 2025) — https://proceedings.mlr.press/v266/retzlaff25a.html
24. Braun, Holzmüller, Jordan, Bach (ERT diagnostics) — https://proceedings.mlr.press/v306/braun26c.html
25. Shihab, Akter, Sharma (continual calibration) — https://arxiv.org/abs/2604.23987
26. Deng & Zheng (AutoEval, CVPR 2021) — https://openaccess.thecvf.com/content/CVPR2021/html/Deng_Are_Labels_Always_Necessary_for_Classifier_Accuracy_Evaluation_CVPR_2021_paper.html
27. Vovk, Gammerman, Shafer (Algorithmic Learning in a Random World) — https://link.springer.com/book/10.1007/978-3-031-06649-8
28. Gao, Schulman, Hilton (reward overoptimization) — https://arxiv.org/abs/2210.10760

---

## 11. Coverage status

**Question 1 — shift & threshold non-transfer: `done`.** Covariate shift [4], label shift [5], arbitrary
drift [2], online/adaptive [3], risk-control version [6], checkpoint-specific [25]. Guarantee/assumption
pairs extracted verbatim where the theorem text was machine-readable.

**Question 2 — correlated / clustered errors: `done`, with one caveat.** Effective-sample-size arithmetic
[11], hierarchical alternatives [9][10], second-moment coverage [10]. *Caveat:* the cleanest
closed-form result [11] is a preprint; the peer-reviewed sources ([9][10]) state the failure qualitatively
and fix it by changing the estimand. I did **not** find a peer-reviewed paper reporting a measured coverage
shortfall from naive conformal on clustered real data — the closest is [11]'s synthetic cluster model.

**Question 3 — base-rate change & selective risk: `done`.** Exact identity and a real base-rate pair [17];
measured subgroup divergence under abstention [18].

**Question 4 — measurement artefacts: `done`.** Label ambiguity [14][15], matching tolerance [16], scorer
gaming [28]. The synthesis "how much of our reported error rate is the metric" is **my inference** and is
labelled as such: at the 20%-risk scale, label ambiguity (3.3–6% [14]; ~29% multi-valid-label on ImageNet
[15]; 8.4% mAP from one pixel [16]) is the same order of magnitude as the effect being certified. No
source states this for *our* setting.

**Question 5 — monitoring / re-certification: `done`.** Sequential risk tracking [21], conformal test
martingales [22], coverage backtests [23], conditional-coverage diagnostics [24], label-free estimation
[26], re-fit protocol [25]. Evidence that monitoring works: yes, for *detection* ([21][22] report
detection delay and alarm-rate numbers). Evidence that monitoring *preserves validity*: none found —
monitoring tells you something changed, not that you are still certified.

**Not completed / residual uncertainty:**
- Citation-ranked search was unavailable (Semantic Scholar 429, no API keys). Seminal-work discovery relied
  on OpenAlex relevance + PDF chasing, so a relevant paper may exist that I did not surface.
- Mondrian conformal is cited from the monograph [27] but I did not re-read the chapter; the attribution is
  standard, not freshly verified.
- "Essential-sample-size" and "checkpoint transfer" have very little peer-reviewed literature; the two most
  on-point sources ([11], [25]) are both preprints and both load-bearing for §3. Flagged.
- No source was found that measures a shipped act/ask/refuse gate's risk transfer end-to-end.
