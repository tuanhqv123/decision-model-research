# Calibration metrics for decision models — companion note

Focus: the metrics that decide whether a typed-decider's confidence is trustworthy enough to route
on ("escalate if p > 0.8"). For the broader benchmark landscape see `benchmark.md`. All claims cite
a numbered source (Sources section); vendor marketing claims are labelled.

---

## What calibration is, and why it is the whole value of this class

A decision model returns a bounded answer plus a probability. A model is **calibrated** when its
stated confidence matches true outcome frequency: "an 0.8 is correct about 80% of the time"
[12]. This is the property that makes threshold-based automation possible. TypeSafe names its
post-training method RLCD (reinforcement learning for calibrated decisions) and says it optimises
for "epistemically honest probabilities" so that "higher confidence means higher accuracy" [12]
— but has **not published the architecture, training data, or reward function**, so the mechanism
itself is unverifiable (claim, not evidence) [12][27]. Calibration is a property of *model + data*:
probabilities calibrated on one distribution are not guaranteed to be calibrated on another [12].

## The canonical metrics

| Metric | Definition / how it works | Notes & caveats |
|---|---|---|
| **Reliability diagram** (calibration plot) | Bin predicted probabilities on the x-axis; plot observed outcome frequency per bin (y). Perfect = the diagonal. Used to *visualise* miscalibration | The canonical diagnostic [16][26]; non-binned smoothing variants exist [16] |
| **Expected Calibration Error (ECE)** | Weighted average of the |observed − predicted| gaps over bins: `ECE = Σ_i (|B_i|/N) · |ybar(B_i) − pbar(B_i)|` | Most-used scalar [16][17][26]; **flawed**: sensitive to binning, ignores resolution/discrimination; different definitions give different numbers [16][8] |
| **Brier score** | Mean squared error `(predicted − outcome)²`; a proper scoring rule | Admits a unique decomposition `Score = Uncertainty − Resolution + Reliability`, separating miscalibration from discrimination [17][25] |
| **Higher-confidence ⇒ higher-accuracy** | The property itself; measured by a reliability diagram (or the slope of a confidence-vs-accuracy curve) | The precondition for routing; must be verified per task and per data distribution [8][12] |
| **Selective prediction** (coverage vs accuracy) | Accuracy on the subset kept after abstaining below a confidence threshold | The actual production metric for a decision model, not a single accuracy point [8] |

## Post-hoc repair

If a model is miscalibrated, standard fixes are **temperature scaling** (Guo et al. 2017), Platt,
isotonic regression, and Beta calibration [15][17]. These do not change accuracy — only the
probability scale.

---

## What independent measurements show

1. **DecisionEval's independent Jev run (no vendor):** accuracy 0.740, Brier 0.148, **ECE (10 bins)
   0.045** on a 2,000-decision frozen test split [8]. Confidence routing works: threshold ≥0.7 →
   57.6% coverage at 0.864 accuracy; ≥0.9 → 24.5% coverage at 0.937 accuracy.
   **Key warning:** they could **not reproduce** the ECE 0.144 cited on the `laya-typed-decisions`
   card (their number 0.045; accuracy & Brier matched to 3 decimals). Likely a different binning/
   definition. **ECE is not comparable unless the definition is published.** [8]
2. **Jev's "confidence" field adds nothing:** on Jev `choice` answers, the vendor's confidence is
   almost exactly a rescale of the chosen option's probability (correlation 0.9997) [12].
3. **Vendor calibration usually beats LLM token probabilities — but not open-weight models.** The
   political-science replication found Jev's probabilities better calibrated than GPT-6 Luna's token
   probabilities on all 8 compared tasks, but **not consistently better than Qwen3.8-27B's** [12].
   GPT-6 Luna is extremely overconfident (74% of its probs < 0.01 or > 0.99) and returns no
   probabilities when it reasons [12].
4. **Calibration is one of the few places a decider wins on accuracy too:** in medicine, Jev scored
   *lower* accuracy than GPT-6 Sol on MetaMedQA (74.8 vs 82.7%) yet was better calibrated there
   (**ECE 0.063 vs 0.146**) [11]. In search, SearchJev cut **average ECE by 41–74%** while being
   5.2–5.3× faster than same-size autoregressive models and lifted answer accuracy in a dual-system
   agent from 45% → 54% [10].
5. **A calibrated signal is not enough for a good cascade.** In the rubric-judge study, deferring
   Jev's uncertain verdicts to an LLM judge lowered cost but **added little accuracy** — even with
   oracle thresholds, none beat the best single judge by more than 2.7 points, because the LLM
   judges err in the *same places* as Jev (on Jev's most confident errors, ~96% of LLM verdicts
   repeated the wrong answer) [13]. A cascade needs a fallback that errs elsewhere.

---

## Recommendations for our calibration harness

- Report **ECE with the bin count/definition stated**, plus Brier score decomposed into
  reliability + resolution (Brier decomposition [17][25]). Do not publish a bare ECE.
- Report a **reliability diagram** and a **coverage-vs-accuracy (selective prediction) curve**,
  not a single accuracy number [8].
- Verify the **"higher confidence ⇒ higher accuracy"** property directly per task type, and split
  `choice` / `score` / `noul` (Jev differs: 0.738 / 0.705 / 0.787 [8]).
- Check calibration on **out-of-domain** data — it will not transfer automatically [12].
- **Pin calibration between releases**; confidence drift breaks threshold routing [9].

---

## Sources

1. Decision Index 0.3 (BenchLM board snapshot) — https://benchlm.ai/benchmarks/decisionindex
2. Decision Index v0.3 methodology data (primary, HF Space) — https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json
3. Decision Index Space (publisher: multimodalart) — https://huggingface.co/spaces/multimodalart/jev-decision-index
4. Decision Index methodology page — https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/methodology.html
5. apolinario/decision-index reproduction kit — https://github.com/apolinario/decision-index
6. Liquid AI, "Open d1" — https://www.liquid.ai/blog/d1-open
7. LiquidAI/d1-3B model card — https://huggingface.co/LiquidAI/d1-3B
8. DecisionEval: Jev by TypeSafe (independent run) — https://decisioneval.dev/models/typesafe-jev/
9. JevBench overview (independent boards) — https://www.jevtypesafeai.com/jevbench
10. SearchJev: A Fast and Calibrated System-1 Model for Search Agents — https://arxiv.org/abs/2610.05107
11. Jev in Medicine: A Benchmark Evaluation (preliminary) — https://arxiv.org/html/2609.34024
12. JEV versus LLMs: Accuracy, Cost and Calibration on Seven Political Science Replications — https://arxiv.org/html/2610.06625
13. Jev vs. LLMs as Rubric Judges — https://arxiv.org/html/2609.29769
14. nibzard/decision-model-benchmark — https://github.com/nibzard/decision-model-benchmark
15. Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks" (2017) — https://arxiv.org/abs/1706.04599
16. "Metrics of Calibration for Probabilistic Predictions" (JMLR 23) — https://jmlr.org/papers/volume23/22-0658/22-0658.pdf
17. "Classifier calibration: a survey" (Machine Learning) — https://link.springer.com/article/10.1007/s10994-023-06336-7
18. SQuAD 2.0: "Know What You Don't Know" — https://arxiv.org/abs/1806.03822
19. google/civil_comments dataset — https://huggingface.co/datasets/google/civil_comments
20. MASSIVE (ACL 2023) — https://aclanthology.org/2023.acl-long.235/
21. PubMedQA — https://pubmedqa.github.io/index.html
22. BoolQ (NAACL 2019) — https://aclanthology.org/N19-1300/
23. XNLI (EMNLP 2018) — https://aclanthology.org/D18-1269/
24. PAWS-X (EMNLP 2019) — https://aclanthology.org/D19-1382/
25. TensorFlow Probability, `calibration.py` (Brier decomposition) — https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/stats/calibration.py
26. "Understanding Model Calibration" (arXiv 2501.19047) — https://arxiv.org/html/2501.19047v2
27. TypeSafe, "Introducing System One Models & Jev" (vendor) — https://typesafe.ai/blog/introducing-system-one-models-and-jev
28. LiquidAI/d1-3B-w8a8 (INT8 edge release) — https://huggingface.co/LiquidAI/d1-3B-w8a8
29. Decision Index 0.2.1 (Liquid report, BenchLM) — https://benchlm.ai/benchmarks/decision-index-0-2-1-liquid
