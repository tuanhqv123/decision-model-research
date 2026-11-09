# Benchmarks & calibration for decision models (System One / typed-decider class)

Scope: what "decision models" are benchmarked on, how they are scored, the standard text
benchmarks and what each measures, calibration metrics, independent evals for the category,
whether public open benchmarks exist, and a proposed mobile/edge benchmark. All claims cite a
numbered source (see Sources); every source has a direct URL. Vendor marketing claims are
labelled "claim, not evidence" and/or flagged as self-reported.

> Timeline note: the sources in this area are dated 2026 (this is the live state of the field).
> Production dates below are as stated by each source.

---

## 1. Decision Index — what it is, who publishes it, how it's scored

**What it is.** The Decision Index is an *independent, community-maintained* leaderboard for
typed-decision models (the class of model TypeSafe calls "System One"). It gives each model one
number for how well it makes typed decisions — pick an option, a label, a tool, a ranking, a
probability — over a frozen suite of requests, with the reference model being TypeSafe's hosted
`Jev 1.13.0`. It is explicitly **not affiliated with TypeSafe** [3][4].

**Who publishes it.** The board lives on a Hugging Face Space by the account **multimodalart**
(`multimodalart/jev-decision-index`) with a Metodology page and a News/artifacts tracker [3].
The reproduction kit and scorer live in the open at **github.com/apolinario/decision-index** [5].
The board is mirrored/re-catalogued by third parties such as BenchLM (which re-keys the aggregate
scores as reference data and excludes them from its own ranking formula) [1]. So: it is an
independent, third-party effort, not a vendor benchmark.

**The public suite (v0.3).** The frozen public corpus holds **110,201 requests** across
**37 static benchmarks**, from **110,313 source cases**; 442 questions are excluded for every
engine alike, leaving **109,759 scoreable rows** [2][4]. Selection is a deterministic hash sample
without replacement (ESCI proportional locale × relevance; request-budget datasets retain
complete linked groups; no outcome-based selection) [2].

**Scoring.** Each benchmark keeps its own primary metric (accuracy, macro-F1, nDCG@10,
case-exact accuracy, per-review F1 …). Scores are **chance-corrected**: `k_b = clip((s_b − r_b) /
(1 − r_b))`, where `r_b` is what random guessing scores on that benchmark, so **0 = random
guessing and 100 = perfect** [2][4]. An unanswered or unsupported request counts as **wrong** [1][4].
The Index is then an area-weighted average of chance-corrected benchmark scores across five
weighted capability areas; two additional variants (a raw-accuracy index and a "breadth" index)
are computed but kept hidden [2][4][6].

**The five weighted areas and their panel (v0.3).** Weights are fixed at v0.2.1 and do not move
between editions [2][4]:

| Area | Weight | Panel benchmarks (id → name, native metric, Jev score) |
|---|---|---|
| Knowledge & Reasoning | 25.8% | GPQA Diamond (acc, Jev .783), GSM8K (.799), ChessBench (.172), MuSR (.661), SATA-Bench (.264), CRUXEval (.730), CLadder (.726), HLE (.204), MMLU-Pro, BBH, WinoGrande (.920) |
| Language Understanding | 25.8% | ContractNLI (F1, .717), ANLI (.748), HellaSwag (.945), ACOS (.295), FinEntity (.870), iSarcasmEval (.505), VAST (.646), NLI4CT (.841), RAGTruth |
| Retrieval & Classification | 20.0% | BANKING77 (F1, .797), CLINC150+OOS (.893), BRIGHT (nDCG@10, .475), Amazon ESCI (.552), PhishNChips (.611), HoVer |
| Tools & Automation | 18.3% | BFCL (acc, .958), ToolRet (nDCG@10, .653), API-Bank (.882), Home appliance simulator (.523), When2Call |
| Arts & Human Taste | 10.0% | BPoMP (.906), Humicroedit (.619), POP909-CL (.181), cfcolor (.647), Habermas Machine (.459), New Yorker |
| Games & Strategy | (Jev-only, weight 0) | folded into Knowledge |

*(Jev scores from the v0.3 panel data [2]; "gold" benchmarks marked ★ weigh 1.2 inside their
area [4][6].)*

**Important nuance — the "standard text benchmarks" are NOT the Decision Index suite.** The seven
text benchmarks the parent listed (SQuAD2.0, Civil Comments, MASSIVE, PubMedQA, BoolQ, XNLI,
PAWS-X) do **not** appear in the Decision Index panel. They are Liquid AI's own internal
"Benchmarks as decisions" evaluation for d1-3B, run "besides the Decision Index" [6][7]. The
Decision Index suite instead leans on knowledge/MMLU/BBH/HLE, NLI/F1-style language tasks, and
retrieval/tools/games/arts tasks. So a model tuned to the "standard seven" is being measured on
*a different* benchmark than the Decision Index leaderboard. **This matters for the plan:** to
rank on the Decision Index you need to score the DI panel from its frozen public sources (via the
reproduction kit), not the seven-text set.

### Version history (how v0.2.1 and v0.3 differ)

- **v0.1 (2026-09-22)** — first public edition: 31 open reproductions plus Jev on a frozen
  **132,422-request** suite; panel "core25" with six interactive environments left out (19 static
  benchmarks, five equal-weight areas); ChessBench folded into Knowledge; Language & Retrieval
  split; only the balanced *raw* index shown [2].
- **v0.2 (2026-09-27)** — scores now **chance-corrected** (0 = random, 100 = perfect), so yes/no
  tasks no longer give free points. Index grows from 19 to **40** benchmarks (adds MMLU-Pro, BBH,
  When2Call, RAGTruth, HoVer, New Yorker, PhishNChips; MMLU retires in favour of MMLU-Pro; ARC-Easy/
  ARC-Challenge shown but not counted) [2].
- **v0.2.1 (2026-09-28)** — new area weights (Arts fixed at 10%, rest share remaining by √n of
  benchmark count): Tools 18.3%, Retrieval 20.0%, Knowledge 25.8%, Language 25.8%. Gold ★
  benchmarks weigh 1.2. ACOS rescored (F1-per-review). ToolRet/BRIGHT count only queries with ≥1
  relevant candidate; RAGTruth chance baseline set to "always hallucinated" (F1 0.518) [2][4].
  On **v0.2.1 public split**, Liquid reports **d1-3B = 48.57** and **d1-omni-600M = 15.95** [6][7][29].
- **v0.3 (2026-10-06)** — introduces **private tests**. The headline is now a **Full score**:
  `Full = 0.20 × public + 0.50 × private-tests-of-same-skills + 0.30 × private-new-domain-tasks`,
  each part mapped onto the public scale using untuned stock models, so **most of the score comes
  from tests nobody can train on** and a public-only advantage counts for little. Public index
  rebuilt: GSM8K rebuilt, ForecastBench retired, WinoGrande moved to Knowledge. A **Vision board**
  was added (0.5 × public vision + 0.5 × private same-skill). Ties within 0.9 points share a rank
  [2][4][6][1].

**v0.3 board snapshot (112 systems).** Hosted **Jev is ranked #3 (60.1)**; **#1 Perplexity
Decider v1.1 (27B) = 62.8**, **#2 Fastino GLiDE no-thinking (28B) = 60.2** [1]. (Third-party
coverage reports Perplexity's own claim of 61.56 overall and "beats Jev by 3.6 points" — a
self-reported comparison from a vendor [30]; the BenchLM snapshot reproduces the board's 62.8/60.1.)

---

## 2. The standard text benchmarks — what each measures

These are the seven in the parent's list. Primary sources:

| Benchmark | What it measures | Source notes |
|---|---|---|
| **SQuAD 2.0** | Extractive reading comprehension + **unanswerability**: 100k answerable questions from SQuAD 1.1 plus >50k adversarially-written unanswerable ones (151k total) that *look* answerable. A system that guesses on unanswerable questions is penalised. Metric: exact-match / F1 | [18] |
| **Civil Comments** | **Toxicity / content-moderation** classification. ~2M public comments from the Civil Comments plugin (2015–2017); crowd raters give each comment a 0–1 fraction for 7 labels (toxicity, severe_toxicity, identity_attack, insult, threat, sexual_explicit, obscene). Multi-label, heavily skewed; Jigsaw ran a bias-detection competition on it | [19] |
| **MASSIVE** | **Multilingual NLU — intent classification + slot-filling** for virtual assistants. 1M+ parallel utterances, 51→52 languages, 18 domains, 60 intents, 55 slots; created by localising SLURP. Metric: intent accuracy / F1 and slot F1 | [20] |
| **PubMedQA** | **Biomedical yes/no/maybe question answering** over PubMed abstracts (1k expert-labeled QA pairs; leaderboard uses a reasoning-required setting). Metric: accuracy / macro-F1 | [21] |
| **BoolQ** | **Natural yes/no question answering** — 15,942 naturally-occurring yes/no questions with a context passage, from unprompted/unconstrained sources. Metric: accuracy (text-pair classification) | [22] |
| **XNLI** | **Cross-lingual natural-language inference**. ~5,000 MNLI test + 2,500 dev pairs translated into 14 languages (112.5k annotated pairs) → entailment/neutral/contradiction. Tests cross-lingual transfer | [23] |
| **PAWS-X** | **Cross-lingual adversarial paraphrase identification**. 23,659 human-translated PAWS evaluation pairs across six typologically-distinct languages (de, es, fr, zh, ja, ko). Metric: accuracy | [24] |

These are all *classification/decision*-shaped tasks (yes/no, NLI, intent, toxicity label, span
or bounded choice), which is why they suit a non-generative decider. The exact scores Liquid
reports on them for d1-3B are in §4.

---

## 3. Calibration metrics — ECE, Brier, reliability diagrams, "higher confidence = higher accuracy"

See the companion note `calibration.md` for the deeper treatment. Summary of the canonical metrics
and definitions:

- **Reliability diagram (calibration plot)** — binned predicted-probability vs. observed
  frequency; a perfectly calibrated model lies on the diagonal. Visualises miscalibration [16][26].
- **Expected Calibration Error (ECE)** — the average |gap| across bins of a reliability diagram,
  weighted by the number of instances per bin (`binary-ECE = Σ_i (|B_i|/N) |ybar(B_i) − pbar(B_i)|`).
  The most popular scalar; has documented flaws (binning choices, it ignores resolution/variance) [16][17][26].
- **Brier score** — the proper scoring rule `(predicted − outcome)²` averaged. It admits a unique
  decomposition `Score = Uncertainty − Resolution + Reliability`, which separates miscalibration
  from discrimination [17][25].
- **"Higher confidence ⇒ higher accuracy"** — this is the calibration property: a model whose
  stated 0.8 is correct ~80% of the time. It is what makes threshold-based routing ("escalate if
  p > 0.8") possible, and it is what TypeSafe claims its RLCD training optimises ("epistemically
  honest probabilities"; "higher confidence means higher accuracy") [12]. **The vendor names RLCD
  but has not published architecture, training data, or reward, so the mechanism is unverifiable**
  (claim, not evidence) [12][27].
- **Post-hoc calibration methods** (temperature scaling, Platt, isotonic, Beta) are the standard
  fixes when a model is miscalibrated, from Guo et al. 2017 (temperature scaling) onward [15][17].

**Two important caveats from the independent evals (see §4):** (1) Calibration is a property of
*model + data*, so probabilities calibrated on one distribution need not be calibrated on another
[12]; (2) the vendor-published "confidence" on Jev's `choice` answers is almost exactly a rescale
of the chosen option's probability (correlation 0.9997) and adds no information beyond the raw
distribution [12].

---

## 4. Independent eval results relevant to calibrated decisions

These are the strongest independent (non-TypeSafe) empirical results for the category. Vendor
claims are flagged.

### (a) DecisionEval — an independent, reproducible Jev run
Jev 1.13.0 on the frozen test split of `LocalLLaMA/typed-decisions` (400 cases, 2,000 decisions),
run by a third party, 2026-09-20 [8]:
- **Accuracy 0.740** (95% CI [0.721, 0.759]), **Brier 0.148** [0.139, 0.156], **ECE (10 bins)
  0.045** [0.037, 0.066], Score MAE 0.389; latency P50 687 ms / P95 777 ms.
- **Calibration is the product:** ECE 0.045 ⇒ you can route on confidence. Threshold ≥0.7 gives
  57.6% coverage at 0.864 accuracy; ≥0.9 gives 24.5% coverage at 0.937 accuracy.
- **Important catch:** the gold labels are a *teacher-ensemble distribution, not human ground
  truth*; 40.6% of decisions had teachers disagree. On agreed items accuracy = 0.841, on disagreed
  = 0.591. So a headline accuracy here averages a defensible task with an ambiguous one.
- **A calibration measurement gap:** a cited Jev figure on the `laya-typed-decisions` card is
  ECE 0.144, which this run could **not reproduce** (their ECE = 0.045, in a different binning/
  definition). Accuracy and Brier matched to 3 decimals. **This is the key warning: ECE numbers are
  not comparable across unknown binning rules.**

### (b) Jev vs. LLMs as rubric judges (Rao & Callison-Burch)
Compared Jev to three flash-tier LLM judges (GPT-5.6 Luna, Gemini 3.8 Flash, DeepSeek V4.1 Flash)
on 9 panels from 7 human-judged benchmarks (5,003 pairs) [13]:
- Jev can **often stand in for LLM judges**. The LLMs cost **16–325× more** and take **28–350×
  longer**, yet Jev's accuracy differs significantly from the best in at most **8 of 27 paired
  comparisons** — ahead mostly on binary checklist criteria, behind only on ordinal ones.
- **They are wrong in the same places.** On Jev's most confident errors, ~96% of LLM verdicts
  repeat Jev's wrong answer (independent errors would be ~50%). So a cascade that defers
  uncertain Jev verdicts to an LLM judge **only lowers cost and adds little accuracy — even with
  oracle thresholds, none beats the best single judge by more than 2.7 points.**
- Implication: a calibrated confidence signal is not sufficient for a good cascade; the fallback
  must err *elsewhere*. This is a directly usable design lesson.

### (c) JEV vs. LLMs on seven political-science replications (Denney & DiGiuseppe)
Replicated seven human-coded studies; benchmarked Jev 1.13 against GPT-6 Luna and Qwen3.8-27B [12]:
- JEV **matches, or comes close to, both LLMs** on the tasks, measured against human coding.
- **No clear cost advantage** vs. GPT-6 Luna at batch prices or vs. open-weight local models;
  Jev bills 1.2–3.3× more *input* tokens per decision than Luna for short labels.
- **Clear speed advantage.**
- **Calibration: better than GPT-6 Luna, not consistently better than Qwen3.8-27B.** When each
  question is asked once, Jev's probabilities beat Luna's token probabilities on all 8 compared
  tasks; but Luna is extremely overconfident (74% of its probabilities < 0.01 or > 0.99, and it
  returns no probabilities when reasoning), while Qwen3.8-27B's calibration is similar to Jev's.
- **Context caveat:** calibration is data-dependent; RLCD-calibrated on TypeSafe's training data
  need not transfer.

### (d) Jev in medicine (preliminary, arXiv 2609.34024)
Jev 1.13 vs GPT-6 Sol (medium reasoning) on four medical benchmarks [11]:
- PubMedQA 78.4% vs 78.2% (similar); but **lower** on MetaMedQA 74.8% vs 82.7%,
  DiagnosisArena-MCQ 59.8% vs 82.4%, and NEJM Case Challenges 61.8% vs 82.4%.
- **Better calibrated on MetaMedQA: ECE 0.063 vs 0.146** (and fast / inexpensive).
- Note the pattern: Jev is *competitive* on extractive/QA, *weaker* on hard multi-step diagnostic
  reasoning, and **its key differentiator is calibration**, not raw accuracy.

### (e) SearchJev (arXiv 2610.05107) — a calibrated System-1 model for search agents
Introduces **SearchJev**, a non-generative decider for search decisions (relevance, evidence
sufficiency, next query), trained with **Soft-Label Learning for Calibrated Decisions (SLCD)**, and
**SearchDecision-Bench**, a benchmark unifying six types of search decision [10]:
- On SearchDecision-Bench: improves decision quality over same-size Qwen3.5 autoregressive models,
  is **5.2–5.3× faster**, and **reduces average ECE by 41–74%**.
- In a dual-system agent, on BrowseComp-Plus: **3.7–4.7× speedup** in active search time while
  improving answer accuracy from 45% to up to 54%.
- This is the clearest evidence that a *dedicated calibrated decider + System-2 fallback* can beat
  pure autoregressive systems on both speed and calibration (contrast with (b), where the fallback
  was an LLM judge that erred in the same places).

### (f) Vendor-reported performance of the open blueprint (d1-3B) — self-reported
Liquid's d1-3B HF card (the open model the parent plans to fine-tune) [6][7][29]:
- **Decision Index 0.2.1 public split: 48.57** (d1-omni-600M: 15.95). Area breakdown: Knowledge
  23.8, Language 56.4, Retrieval 52.8, Tools 74.5, Arts 36.3. Claimed "best under 10B" (ahead of
  Decider 35B-A3B 47.11). **Vendor claim (self-scored with the official scorer), not an
  independent run.**
- "Benchmarks as decisions" (internal): SQuAD2.0 85.3, Civil Comments 93.0, MASSIVE intent 87.3,
  HelpSteer2 36.7, PubMedQA 66.0, BoolQ 86.7, XNLI 85.0, PAWS-X 76.9 → mean 77.1 (over 8); the
  blog's mean of 82.9 is over 7 (excluding HelpSteer2). **Same numbers, different included set —
  an illustration of how benchmark-inclusion changes the "headline".** Vendor-reported.
- Also reports **71.8 on DecisionBench** (Hanno-Labs/decision-bench, en v1, 23,900 rows) and
  **69.3 on Fast Decisions** (fastino/fast-decisions dev split); ImajevBench 64.0 (vision).
- **Latency (vendor-measured, edge/GPU):** single question 8 ms (RTX 4090), 9 ms (MI325X),
  30 ms (Apple M5 Pro), 16 ms (Jetson AGX Thor), 50 ms (Jetson Orin Nano); throughput up to
  262–475 decisions/s (packing 64 states) [6][7].
- **Architecture:** 3.12B params (SigLIP2 NaFlex vision 400M, 32,768 context, 128k vocab), built
  on LFM2.5-VL-3B; pushed NVFP4 + llama.cpp support; an INT8-w8a8 release is aimed at Jetson-type
  tensor cores [7][28]. **License** is stated by third-party catalogs as "LFM license conditions
  apply" — treat license terms as an open question to verify before fine-tuning/distribution [29].

---

## 5. Do public open benchmarks exist for this category? — Yes, but young & partly unaudited

- **Decision Index** — the most serious open one: frozen public suite, MIT-style reproduction kit
  (github.com/apolinario/decision-index), public + private scoring, pinned sources, chance
  correction [2][3][5]. This is the benchmark to target for a publicly comparable score.
- **JevBench family (independent, third-party)** — several boards exist with *different* task
  sets and self-published runs [9]:
  - **Benchmark Heaven JevBench** (benchmarkheaven.com/jev-models) — typed-decision accuracy on a
    fixed rubric set; reports Cygnet ~73.7, Jev 1.13.0 ~72.1 [9].
  - **jevbench.dev** — head-to-head games (StarCraft II, Minecraft); TypeSafe Jev 1.13 ~84% win
    rate (their numbers) [9].
  - **fstandhartinger/jevbench** (GitHub) — an open benchmark for Jev-class typed decision models [9].
- **Decision-model-benchmark (DMB, nibzard)** — an independent, reproducible benchmark comparing a
  decider to constrained LLMs and deterministic baselines on typed decisions, measuring accuracy,
  latency, cost, failures, and confidence-vs-correctness [14].
- **SearchDecision-Bench** (SearchJev) — six types of search decision, calibrated [10].
- **DecisionBench** (Hanno-Labs) and **Fast Decisions** (fastino) — additional public decision
  datasets referenced by the d1-3B card [7].
- **Caveat:** none are audited; several are self-published; "the JevBench score" depends on which
  board you mean [9]. Non-vendor ECE/calibration numbers are not comparable across boards because
  binning/definitions are unpublished [8].

**Bottom line for the plan:** a public, reproducible, category-specific benchmark *does* exist
(Decision Index public split + reproduction kit) — so we do **not** need to build the leaderboard
from scratch. What does not exist and we should (partly) build ourselves is a *mobile/edge-native,
latency- and calibration-weighted* benchmark (§6).

---

## 6. Proposed mobile/edge benchmark criteria for our own model

Goal: measure a decision model as it will actually be used on-device — single forward pass,
zero output tokens, bounded answer + probabilities — where the differentiators are **calibration,
latency, footprint, and power**, not just accuracy.

1. **Coverage & refusal honesty** — measure the share of requests answered (don't let "unanswered
   counts as wrong" inflate apparent precision; adopt DI's rule that an unanswered request counts
   as wrong, but report coverage separately [2]).
2. **Calibration, reported with definition** — ECE (fixed bin count, *state it*), Brier score with
   its reliability/resolution decomposition (Brier decomposition gives reliability + resolution
   separately [17][25]), and reliability-diagram AUC / area between curve and diagonal [16][26].
   Also report a **selective-prediction (coverage-vs-accuracy) curve** rather than a single
   accuracy point, since threshold routing is the actual use case [8].
3. **"Higher confidence ⇒ higher accuracy" slope** — verify the property directly: group outputs
   by stated confidence, show observed accuracy per bin (reliability diagram), and report the
   slope / whether the vendor's "an 0.8 means 80%" claim holds [8][12].
4. **Accuracy per decision type** — report separately for `choice` (closed set), `score` (bounded
   ordinal), and `noul` (yes/no), since the parent's model must serve all three; Jev shows these
   differ (choice 0.738 / score 0.705 / noul 0.787 [8]).
5. **Robustness / OOS** — include out-of-domain and adversarial/unanswerable items. This is where
   SQuAD2.0's design is directly reusable (adversarial unanswerable questions) [18].
6. **Latency, end-to-end on real edge hardware** — single-question P50/P95 and throughput (packed
   states), on the actual target (Apple M5 Pro / iPhone, Jetson, Snapdragon). Use d1-3B's published
   method as a template (8–50 ms single question; 475/s packed on RTX; 262/s Jetson Thor) [6][7].
   **Important:** decision models emit no output tokens, so *measure end-to-end input→output, not
   tokens/s* (Liquid explicitly measures end-to-end for this reason) [6].
7. **Footprint & memory** — parameters, quantized size (INT8/NVFP4), peak memory, and whether it
   fits alongside the app. Reference the d1-3B INT8-w8a8 release targeting INT8 tensor cores [28].
8. **Change-pinned / version-stable** — a decision model whose confidence drifts between versions
   breaks threshold-based routing; pin and diff calibration between releases [9].
9. **Cost & energy** — battery/energy per decision and (if a hosted fallback is used) cost per
   decision; decode that a "192× faster / 40× cheaper" headline is vendor claim [27].
10. **Train/unseen separation** — mirror the Decision Index v0.3 design: private same-skill tests
    + private new-domain tasks, so a public-only advantage counts for little and overfitting to a
    public board is penalised [2][4].

**Ranked recommendations for what to actually build first** (most feasible / most informative):
(1) a calibrated-accuracy + selective-prediction harness on the seven public text benchmarks plus
SQuAD2.0-style unanswerables (cheap, uses existing data); (2) score the **Decision Index public
split** via the apolinario reproduction kit to get a comparable, non-self-reported number; (3) add
a real-device latency/power harness on Apple Silicon + Jetson; (4) publish the binning/ECE
definition so your numbers are reproducible and comparable (the Jev ECE 0.045-vs-0.144 mismatch
shows exactly why this matters [8]).

---

## Coverage Status

**Checked directly (read the source):** Decision Index v0.3 methodology JSON and formula (areas,
weights, panel, chance-correction, version changelog) [2][4]; Decision Index board snapshot [1];
Liquid d1-3B HF card and blog (benchmarks, latency, architecture) [6][7]; SearchJev abstract
[10]; the two Jev-vs-LLM papers (abstract + methods/results) [12][13]; DecisionEval Jev run [8];
medical Jev results via secondary summaries [11]; standard-benchmark primary pages [18][19][20][21][22][23][24].

**Inferred (not directly stated by a single source):** that the DI suite and the "standard seven"
are different sets — confirmed structurally ([2] panel vs [7] card table), not by any single
prose statement; the Jump from v0.2.1 48.57 to the Full-score scale is inferred from [6][7][29].

**Uncertain / blocked / needs follow-up:**
- **d1-3B license terms** — the HF model page could not be read via the Hub API (401) and the
  card I fetched did not state the license; third-party catalogs say "LFM license conditions apply"
  [29]. **Verify before fine-tuning/distributing.**
- **Decision Index v0.3 private-task contents and item-level answers are not public** by design;
  only aggregate private results are shown [1][2]. We can score the *public* split ourselves but
  cannot reconstruct the private Full score independently.
- The independent Jev-vs-LLM and medical arXiv IDs (2610.06625, 2609.29769, 2609.34024, 2610.05107)
  appear in a 2026 timeline; I read their abstracts/posters, not full appendices.
- Vendor latency/accuracy figures for d1-3B and TypeSafe's speed/cost claims are **self-reported**;
  no independent replication of d1-3B was found.
