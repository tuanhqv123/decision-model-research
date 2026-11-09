# Why the action head collapsed to the majority prior — and the correct calibration-training approach

**Scope.** Diagnose why a linear head (and a tiny MLP) on frozen embeddinggemma-2 features
produces test accuracy EXACTLY equal to the majority-class prior (click = 0.5828), and give a
concrete, citable recipe for a calibrated, on-device, single-pass decision head whose confidence
is usable for risk-gating.

**Setup (from parent).** EmbeddingGemma-2 → 768-d multimodal embedding; small linear head + tiny
MLP; CE over K=7 action types; train 6,805 / test 1,630; majority 'click' = 0.5828 on test.
Trained head test acc = 0.5828 (= prior). Multimodal CE ECE = 0.029, text-only CE ECE = 0.093
(image helped calibration but not accuracy). "Proper-scoring" training collapsed max-softmax
confidence to [0.602, 0.632] (unusable for risk-gating) whereas CE preserved spread.

---

## TL;DR (the four answers in brief)

1. **Root cause is feature non-separability, not the probe or the loss.** The signature "accuracy
   = prior, argmax ≡ click for every input" is the *degenerate constant-predictor* optimum. A
   linear head, a tiny MLP, and your observed low ECE (high calibration but no resolution) together
   point to the frozen EmbeddingGemma-2 embeddings not carrying class-discriminative signal for the
   seven action types that a small decoder can exploit. The compounding factor is the 58%
   majority prior + plain CE, which makes "always predict click" the cheap optimum. The remedy is
   adapting the backbone (LoRA / LP-FT), and/or a stronger decoder, and rebalancing — not loss
   tuning alone. **[oracle: inference, see Q1]**
2. **CE and "proper scoring" (Brier/quadratic) are both proper scoring rules, so neither is
   intrinsically "more calibrated."** The real distinction is calibration vs *sharpness*
   (resolution). A model that predicts the marginal (≈ click prior) is perfectly calibrated but
   has zero resolution. That is precisely your [0.602, 0.632] collapse; it is the textbook
   *calibrated-but-not-sharp* failure, not an "accuracy" failure. **[verified: Kuleshov & Deshpande]**
3. **softmax-max is the wrong P(success) signal.** `argmax-softmax = p(y=k|x)` (class membership)
   is not `p(ŷ=y|x)` (the probability that the *chosen* action is correct). Under class imbalance
   it is contaminated by the label prior (high for click even when wrong, low for minority even
   when right). Use a **separate binary success/compatibility head** trained on the
   `[predicted == ground-truth]` indicator. **[oracle: inference + selective-prediction lit]**
4. **Recipe:** keep a 7-way action head (CE/focal + rebalance + flat-minima regularization), and
   add a separate scalar **success head** (binary CE) sharing the frozen embedding; regularize with
   label smoothing / confidence penalty / logit-norm, and KL to a temperature-scaled **reference
   head**; evaluate with **AURC / risk-coverage / failure-prediction AUROC**, not ECE alone.
   **[Q4]**

---

## Q1. Why does a linear probe on frozen features fail here?

### What the literature says about linear probing

- A linear probe is a *separability diagnostic*: it measures how linearly the target property is
  carried in a fixed representation. Low probe accuracy ⇒ the property is not (linearly) present in
  a usable way. **[1]**
- Probing has known pitfalls: a probe with enough capacity can **memorize** (control tasks show
  high probe accuracy even when the representation does not encode the property), so a *high* probe
  score must be read with control-task selectivity; conversely, a *low* score can reflect either a
  genuinely non-separable representation **or** an under-expressive probe. **[2][3]**
- For feature transfer, **linear probing ≠ fine-tuning**. Fine-tuning can *distort* pretrained
  features and underperform OOD; the two-step "linear-probe-then-fine-tune" (LP-FT) is the
  recommended way to adapt a frozen backbone. **[4]**

### Applying this to your result

Your evidence is strong and self-consistent:

- **Accuracy exactly equals the prior** (0.5828) with **argmax always = click**. This is the exact
  signature of the degenerate solution: the decoder ignores the input and emits the majority class.
  It is the global optimum of CE when the frozen features carry **no usable class-discriminative
  gradient** for any action other than "inherit the prior."
- **A tiny MLP also collapsed.** Non-linear capacity on the frozen embedding did not help. This is
  evidence *against* hypothesis (c) "need a non-linear decoder" being the binding constraint —
  unless the MLP was too small / under-trained (must be ruled out with a control-capacity check,
  see below).
- **Image improved ECE (0.093 → 0.029) but not accuracy.** Adding the image made the *confidence*
  better track *correctness*. That is a **resolution/sharpness** signal (the features carry
  information about *difficulty/certainty*), not a **class-separation** signal. So the features are
  not empty — they carry "how hard is this" but not "which of the 7 actions."

**Likely hypotheses, ranked:**

| Hypothesis | Assessment |
|---|---|
| (a) EmbeddingGemma-2 features do not separate the 7 action semantics | **Most likely.** Aligns with argmax≡prior, MLP also failing, and the model being a *retrieval/semantic-similarity* encoder (docs describe RAG / semantic search / similarity-based zero-shot classification), where fine-grained UI-action semantics are not an aligned axis. **[19]** |
| (b) Imbalanced labels (58% click) make predict-click the cheap optimum | **Compounding factor, likely not sufficient alone.** Imbalance strongly biases classifiers toward the majority class, but with any real feature signal you would expect to beat the prior. **[11]** |
| (c) Need a non-linear decoder | **Less likely** given the MLP also collapsed, but not excluded until you verify the MLP had enough capacity (see diagnostic). |
| (d) Need backbone fine-tune/adaptation | **Likely necessary as the real fix.** A frozen retrieval embedding that doesn't separate the task needs to be *adapted* (LoRA / LP-FT). **[4]** |

**Diagnostics to disambiguate (cheap, decisive):**

1. **kNN on frozen embeddings.** A k-nearest-neighbour classifier (cosine) on the frozen features has
   no training bias toward the prior and directly tests whether *nearest neighbours share the
   action*. If kNN ≈ prior ⇒ features lack class structure ⇒ (a). If kNN ≫ prior ⇒ the head was
   the problem ⇒ (b/c).
2. **Balanced per-class linear SVM + control task.** Train a linear SVM to separate each action from
   all others on a *class-balanced* subset. Also add a **control task** (random labels on the same
   features, Hewitt & Liang) to confirm the probe is expressive enough to overfit random labels; if
   it can't, your probe was the limiter, not the features. **[2]**
3. **Separability metric on class centroids.** Compare between-class versus within-class cosine
   distances in the frozen space. If centroids aren't separated, features are the issue.
4. **LoRA / LP-FT.** Fine-tune the last block(s) of EmbeddingGemma-2 (or a LoRA adapter) on the
   action task; if accuracy jumps, the features needed adaptation. **[4]**

**Honesty note.** I have not run these. The "features are non-separable" conclusion is a *strong
inference* from your numbers + the retrieval-model nature of EmbeddingGemma-2, not a measured
fact. Run diagnostics 1–4 to confirm.

---

## Q2. Calibration training: proper-scoring / Brier / log-loss vs CE

### Proper scoring basics

- A **proper scoring rule** is one whose expected value is minimized by the true conditional
  distribution p(y|x). Both **cross-entropy (log loss)** and **Brier score (quadratic/squared
  error)** are strictly proper. So in the infinite-data, well-specified limit, neither is "better"
  for calibration. **[8][7]**
- **Brier score decomposes** into **Reliability (calibration) + Resolution (sharpness) −
  Uncertainty**, the Murphy/calibration-refinement decomposition. **[10][17]** This is the key
  formal tool: it separates "does the forecast match observed frequency" from "does the forecast
  distinguish outcomes." **ECE measures only the reliability/calibration term; it says nothing about
  resolution.**

### The mechanism behind your [0.602, 0.632] collapse

- Kuleshov & Deshpande state it exactly:
  > "Calibration by itself is not sufficient to produce a useful forecast. For example, a binary
  > classifier that always predicts the marginal data probability P(Y=1) is calibrated; however it
  > does not even use the features X and thus cannot be accurate. In order to be useful, forecasts
  > must also be **sharp**." **[7]**

  Your proper-scoring head found the *flat basin* at the class marginal (click ≈ 0.58), giving low
  ECE and no separation. The [0.602, 0.632] band is near-zero **resolution**: the model learned the
  (small) residual drift around the prior but no real per-example certainty. ECE stays low because
  predictions match frequencies in bins; the score is simply uninformative.

- **Why CE "preserved spread":** CE is *unbounded* and steep near p=0 for the true class, so it
  pushes logits apart (sharpens) and, with any capacity, produces a wider, sometimes
  over-confident, confidence distribution. Empirical calibration work shows modern CE-trained nets
  are **over-confident** and poorly calibrated. **[5][6]** Brier is *bounded and quadratic*, so it
  shrinks predictions toward the base rate → flatter, low-ECE, low-sharpness. **[17]**
- **Caveat / honest nuance:** CE's "spread" is largely *noise and over-confidence*, not genuine
  resolution — it did not help accuracy (still the prior). **Neither the proper-scoring collapse
  nor the CE spread is a usable risk signal**, because the features don't separate the task. The
  loss choice only changed *how* the head failed, not *whether*.
- **Calibration-focused objectives can actively hurt risk-gating.** Zhu et al. find that "most
  confidence calibration methods are useless or harmful for failure prediction" and temperature
  scaling "can hardly improve failure prediction," because calibration *aligns accuracy with mean
  confidence and compresses the score*, erasing sample-hardness information. **[9]** This is the
  same mechanism as your collapsed spread.

**Temperature scaling (Guo et al. 2017)** is post-hoc, single-parameter, and only rescales; it
**cannot create resolution** — if the score is already flat, temperature makes it flatter (T>1) or
fatter (T<1) but never more discriminative. It fixes *miscalibration of the mean*, not *sharpness*.
**[5]**

**Bottom line for Q2.** You did not "hurt accuracy with proper scoring" — accuracy was already the
prior because the features don't separate. What you did is **trade sharpness for reliability**:
proper scoring removed the noisy/large CE confidences and landed on a calibrated-but-flat
prediction. That is mathematically expected, and it is *the wrong objective* if your downstream need
is risk-gating (which requires resolution). Use ECE **and** a resolution/risk metric.

---

## Q3. Why argmax-softmax is a bad P(success) signal, and the correct head design

### Two different probabilities

- **p(y = k | x)** — the model's probability that the *truth* is class k. This is what the softmax
  gives, and `max_k p(y=k|x)` is the confidence you were reading.
- **p(ŷ = y | x)** — the probability that the *chosen* label (argmax) is correct. This is the
  quantity you actually want for risk-gating.

These coincide only in a perfectly calibrated, well-behaved classifier with balanced classes. Under
class imbalance they diverge:

- For the **majority class** (click), softmax-max is inflated by the class prior and the model's
  majority-bias; it reports "high" confidence even when the prediction is frequently wrong.
- For **minority classes**, softmax-max is compressed even when the prediction is right.

So `argmax-softmax ≈ prior-contaminated class-membership confidence`, not *per-example correctness*.
The literature is explicit that **confidence (max-softmax) ≠ correctness/risk**; treating them as
equal is a known fallacy. **[21]**

Even with perfect calibration, an unsharp confidence is useless for gating (Q2). Calibrated
selective-classification work notes calibrated models still "allow wrong predictions that have high
confidence, or skip correct predictions that have low confidence." **[16]**

### Correct design: a separate success/compatibility head

Model `p(correct | x, ŷ)` directly as a **binary classification** problem:

- Target: `s = 1[ŷ == y]` (or soft labels from a teacher / self-consistency on held-out data).
- Head: a single scalar on top of the same frozen embedding (or the fused multimodal embedding,
  since image improved calibration). Loss: **binary cross-entropy** with logit-norm / temperature or
  a confidence-penalty term to keep it sharp and bounded.
- Because the target is a *correctness indicator* rather than the 7-way label, this head is **not
  constrained by the class prior**. It only needs the features to carry *difficulty* information —
  which your image signal already demonstrates (ECE 0.093 → 0.029).

This is precisely the **failure-prediction / selective-classification / reject-option** setup:
- **SelectiveNet** trains an auxiliary selection head alongside the classifier. **[15]**
- **Calibrated Selective Classification** trains a *separate selector network* to control the
  calibration of accepted examples. **[16]**
- **Zhu et al.** show risk-gating depends on a *gap* between correct and incorrect confidences and
  that flat-minima (SWA/SAM) enlarges that gap. **[9]**

> **Why this matters for your framing.** You derived p_success from the 7-way softmax. That is the
> wrong place to read success. Read it from a dedicated head that was *trained* on correctness, and
> that head can be both calibrated *and* sharp if the features carry difficulty signal (yours do).

---

## Q4. Recommended concrete training approach

Goal: a single-pass, on-device decision head whose **action choice is accurate** *and* whose
**confidence is usable for risk-gating**.

### Architecture (two heads, one shared frozen embedding)

```
input (text + image) → EmbeddingGemma-2 (frozen, or LoRA-adapted) → shared 768-d vector v
  ├─ Head A: action head  → logits z ∈ R^7 → softmax  (the action decision)
  └─ Head B: success head → scalar s ∈ (0,1)  s = σ(w·v + b)  (P(chosen action is correct))
```

Both heads are tiny (on-device friendly). Head B is trained on `1[argmax_HardA(x) == y]`.

### Losses

- **Head A (action):** CE or **focal loss** (class imbalance). Add a calibration term:
  - **Label smoothing** (softens the target, limits the largest logit — improves calibration). **[12]**
  - **Confidence penalty / entropy regularization** to discourage over-confident peaks. **[13]**
  - **Logit normalization (logit-norm)** to bound logit scale and reduce over-confidence. **[14]**
  - **KL to a reference head** (distillation): train against the soft targets of a
    temperature-scaled **reference action head** (or a fixed teacher). Temperature in the KL
    controls sharpness: a higher teacher temperature yields a softer, more calibrated student
    target and **keeps the confidence distribution well-formed and bounded** instead of collapsing
    to 0/1 or to the prior. This is the cleanest mechanism for "KL regularization to a reference
    head" and directly addresses keeping spread usable.
  - **Flat-minima search (SWA and/or SAM)** to widen the correct/incorrect confidence gap — the
    finding that best improves failure prediction. **[9]**
- **Head B (success):** binary CE on the correctness indicator, with **logit-norm/temperature** and
  possibly a confidence-penalty/gradient term. To stabilize the moving target (Head A's argmax
  changes during training), use **stop-gradient** or derive success labels from a *fixed reference /
  distilled* head.

**Composite objective (single-pass variant):**
```
L = CE_action(z, y; T) + λ_kl * KL(softmax(z/T) || softmax(z_ref/T))
  + λ_s * BCE(s, 1[argmax(z_ref) == y])
  + λ_reg * (label_smoothing / confidence_penalty / logit-norm)
```
Train Head A with the action + KL + regularizers; train/refit Head B on held-out correctness so it
doesn't overfit the training argmax.

### Data reweighting

- **Inside Head A:** class-balanced loss weighting, focal loss, or oversampling minority actions to
  stop the predict-click degeneracy; at inference, **threshold/adjust the decision boundary to
  compensate for the prior** (budgeted coverage). **[11]**
- **For Head B:** ensure the success head sees a **class-balanced, easy/hard-mixed** sample so it
  learns difficulty rather than the majority prior. Use the held-out set with balancing.
- **Calibration data:** always calibrate/verify on a **held-out** split; never on the train split.

### Keeping confidence spread usable for risk-gating

- **Optimize and evaluate sharpness, not just calibration.** Report **ECE** (calibration) *and*
  **resolution / selective-risk**: AURC, risk-coverage curve, or failure-prediction AUROC (the
  ability of your confidence score to rank correct above incorrect). **[9][15][16]** A low ECE with
  flat coverage (Band [0.602, 0.632]) is a red flag for zero resolution.
- **Do not use temperature scaling alone as the fix** — it rescales but cannot add resolution, and
  may harm failure prediction. **[5][9]**
- **If confidence still collapses,** the problem is the **features**, not the objective. The
  highest-leverage action is to **adapt the backbone**: LoRA / LP-FT on EmbeddingGemma-2 for the
  action task. **[4]** No loss engineering creates discrimination from features that don't separate
  the classes.
- On-device consideration: use the **multimodal** embedding (image + text) for Head B, since your
  data show images improve calibration; optionally quantize the frozen encoder.

### Evaluation checklist (concrete)

| Metric | What it tells you | Source |
|---|---|---|
| Test accuracy vs prior | Did the decoder beat the majority baseline? | — |
| ECE | Average calibration (reliability) | **[5]** |
| **AURC / risk-coverage / failure-prediction AUROC** | **Is confidence usable for gating (resolution)?** | **[9][15][16]** |
| Confidence spread / histogram | Is the score sharp (resolution) or flat (collapse)? | **[7]** |
| Per-class accuracy | Did minority classes get learned? | **[11]** |

---

## Coverage status

**Verified by direct read/confirmed citation:**
- Kuleshov & Deshpande (calibrated-but-not-sharp; the marginal-predictor example). **[7]**
- Zhu et al. (calibration methods useless/harmful for failure prediction; temperature scaling can
  hardly improve it; flat minima enlarges the gap; FMFP=SAM+SWA). **[9]**
- Guo et al. 2017 (modern nets over-confident; temperature scaling). **[5]**
- Gneiting & Raftery 2007 (proper scoring rules). **[8]**
- Murphy / calibration-refinement decomposition (reliability = calibration, resolution = sharpness).
  **[10][17]**
- "Bag of Tricks" (Brier loss reduces ECE; relates via calibration-refinement decomposition). **[17]**
- Buda et al. (class imbalance; majority bias; thresholding/oversampling). **[11]**
- Szegedy (label smoothing), Pereyra (confidence penalty), Wei et al. (logit norm). **[12][13][14]**
- Geifman & El-Yaniv (SelectiveNet separate selection head), Calibrated Selective Classification
  (separate selector network). **[15][16]**
- Alain & Bengio (linear probe as separability diagnostic), Belinkov (probing limits),
  Hewitt & Liang (control tasks / probe capacity), Kumar et al. (LP-FT, feature distortion). **[1][2][3][4]**
- EmbeddingGemma-2 existence and spec (768-d, 740M, Gemma 4, retrieval/semantic-similarity). **[19]**

**Inference (my synthesis, not a measured claim):**
- The precise claim that your [0.602, 0.632] collapse is the *calibrated-but-unsharp / zero-resolution*
  regime (Kuleshov's example). Strong inference, well-supported.
- The claim that the root cause is **feature non-separability** rather than the loss — strong
  inference from accuracy=prior + MLP collapse + retrieval-encoder nature, but **not yet measured**.
  Run the Q1 diagnostics (kNN, balanced SVM + control task, centroid distance, LoRA) to confirm.

**Not completed / open questions:**
- Could not read the Hugging Face model card directly (401 Unauthored); used Google developer docs +
  HF page snippet + transformers model_doc instead. **[19]**
- Did not get exact per-class accuracy or the confidence histograms from your runs; those would
  confirm (vs the prior alone) whether minority classes were learned at all.
- Did not verify the specific proper-scoring loss you used (Brier vs other); the analysis holds for
  any bounded, non-log proper scoring rule, but your exact numbers depend on the implementation.
- The tiny-MLP "did not help" observation is taken at face value; an under-capacity MLP could change
  this conclusion.

---

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | Alain & Bengio (2016) — Understanding intermediate layers using linear classifier probes | https://arxiv.org/abs/1610.01644 | Linear probe = measure of how linearly a property is encoded; used to diagnose representations | primary | high |
| 2 | Hewitt & Liang (2019) — Designing and Interpreting Probes with Control Tasks | https://aclanthology.org/D19-1275/ | Probes can memorize; use control tasks to tell apart representation content from probe capacity | primary | high |
| 3 | Belinkov (2022) — Probing Classifiers: Promises, Shortcomings, and Advances | https://aclanthology.org/2022.cl-1.7/ | Survey of probing limits / capacity / interpretability of linear probes | secondary (survey) | high |
| 4 | Kumar et al. (2022) — Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution | https://arxiv.org/abs/2202.10054 | Full fine-tuning can distort features; linear-probe-then-fine-tune (LP-FT) recommended for adapting a frozen backbone | primary | high |
| 5 | Guo, Pleiss, Sun, Weinberger (2017) — On Calibration of Modern Neural Networks | https://proceedings.mlr.press/v70/guo17a.html | Modern nets are over-confident; temperature scaling is a post-hoc calibration method | primary | high |
| 6 | Minderer et al. (2021) — Revisiting the Calibration of Modern Neural Networks | https://arxiv.org/abs/2106.07998 | Reassessment of calibration across architectures/pre-training | primary | high |
| 7 | Kuleshov & Deshpande (2022) — Calibrated and Sharp Uncertainties in Deep Learning via Density Estimation | https://proceedings.mlr.press/v162/kuleshov22a/kuleshov22a.pdf | "A binary classifier that always predicts the marginal data probability is calibrated; however it does not even use features X and thus cannot be accurate. In order to be useful, forecasts must also be sharp." | primary | high |
| 8 | Gneiting & Raftery (2007) — Strictly Proper Scoring Rules, Prediction, and Estimation | https://doi.org/10.1198/016214506000001437 | CE/log loss and Brier are (strictly) proper scoring rules; minimized by true conditional distribution | primary | high |
| 9 | Zhu et al. (2022) — Rethinking Confidence Calibration for Failure Prediction | https://arxiv.org/abs/2303.02970 | Calibration methods are "useless or harmful" for failure prediction; temperature scaling "can hardly improve" it; flat minima (SWA+SAM → FMFP) enlarge the correct/incorrect confidence gap | primary | high |
| 10 | Murphy Decomposition / Calibration-Resolution Principle | https://ar5iv.labs.arxiv.org/html/2005.01835 | Forecast loss decomposes into uncertainty + resolution + miscalibration; resolution = ability to distinguish outcomes | primary (theory) | high |
| 11 | Buda, Maki, Mazurowski (2018) — A systematic study of class imbalance in CNNs | https://arxiv.org/abs/1710.05381 | Class imbalance biases models toward the majority class; thresholding to compensate prior; oversampling | primary | high |
| 12 | Szegedy et al. (2016) — Rethinking the Inception Architecture for Computer Vision | https://arxiv.org/abs/1512.00567 | Label smoothing prevents the largest logit from dominating → better calibration | primary | high |
| 13 | Pereyra et al. (2017) — Regularizing Neural Networks by Penalizing Confident Output Distributions | https://arxiv.org/abs/1701.06548 | Confidence penalty / entropy regularizer discourages over-confident outputs | primary | high |
| 14 | Wei et al. (2022) — Mitigating Neural Network Overconfidence with Logit Normalization | https://arxiv.org/abs/2205.09310 | Logit normalization bounds logits, reduces over-confidence / improves calibration | primary | high |
| 15 | Geifman & El-Yaniv (2019) — SelectiveNet: A Deep Neural Network with an Integrated Reject Option | https://proceedings.mlr.press/v97/geifman19a.html | Train an integrated selection (reject) head alongside the classifier for coverage/risk control | primary | high |
| 16 | Calibrated Selective Classification (2022) | https://arxiv.org/abs/2208.12084 | Calibrated models still allow high-conf wrong and low-conf correct predictions; learn a separate selector for selective calibration | primary | high |
| 17 | Bag of Tricks for In-Distribution Calibration of Pretrained Transformers | https://aclanthology.org/2023.findings-eacl.40.pdf | Brier loss reduces ECE; relates to ECE via calibration-refinement decomposition (Bröcker 2009) | primary | high |
| 18 | Calibration in Deep Learning: A Survey of the State-of-the-Art | https://arxiv.org/abs/2308.01222 | Survey of calibration methods, regularization for calibration | secondary (survey) | medium |
| 19 | EmbeddingGemma-2 — Google AI for Developers / Hugging Face | https://ai.google.dev/gemma/docs/embeddinggemma | 740M-param, 768-d unified multimodal encoder on Gemma 4; for retrieval, semantic search, similarity-based zero-shot classification | primary (vendor docs) | high |
| 20 | A Formalism and Approach for Improving Robustness of LLMs Using Risk-Adjusted Confidence Scores | https://arxiv.org/abs/2310.03283 | Confidence (max-softmax) is commonly equated to correctness but is not; that equation is unreliable in deep learning | primary | medium |
| 21 | Patra et al. (2023) — Calibrating Deep Neural Networks Using Explicit Regularisation and Dynamic Data | https://openaccess.thecvf.com/content/WACV2023/papers/Patra_Calibrating_Deep_Neural_Networks_Using_Explicit_Regularisation_and_Dynamic_Data_WACV_2023_paper.pdf | Combines implicit (focal loss) + explicit (auxiliary calibration) regularizers for calibration | primary | medium |

---

## Sources

1. Alain, G., & Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes.* ICLR. https://arxiv.org/abs/1610.01644
2. Hewitt, J., & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks.* ACL 2019. https://aclanthology.org/D19-1275/ (arXiv:1909.03368)
3. Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics 48(1):207–219. https://aclanthology.org/2022.cl-1.7/ (arXiv:2102.12452, doi:10.1162/coli_a_00422)
4. Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR 2022. https://arxiv.org/abs/2202.10054
5. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML 2017, PMLR 70:1321–1330. https://proceedings.mlr.press/v70/guo17a.html (arXiv:1706.04599)
6. Minderer, M., et al. (2021). *Revisiting the Calibration of Modern Neural Networks.* https://arxiv.org/abs/2106.07998
7. Kuleshov, V., & Deshpande, S. (2022). *Calibrated and Sharp Uncertainties in Deep Learning via Density Estimation.* ICML 2022, PMLR 162. https://proceedings.mlr.press/v162/kuleshov22a/kuleshov22a.pdf
8. Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102(477):359–378. https://doi.org/10.1198/016214506000001437
9. Zhu, F., Cheng, Z., Zhang, X.-Y., & Liu, C.-L. (2022). *Rethinking Confidence Calibration for Failure Prediction.* ECCV 2022. https://arxiv.org/abs/2303.02970
10. *The Murphy Decomposition and the Calibration-Resolution Principle.* https://ar5iv.labs.arxiv.org/html/2005.01835
11. Buda, M., Maki, A., & Mazurowski, M. A. (2018). *A systematic study of the class imbalance problem in convolutional neural networks.* Neural Networks 106:249–259. https://arxiv.org/abs/1710.05381
12. Szegedy, C., et al. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. https://arxiv.org/abs/1512.00567
13. Pereyra, G., Tucker, G., Chorowski, J., Kaiser, Ł., & Hinton, G. (2017). *Regularizing Neural Networks by Penalizing Confident Output Distributions.* https://arxiv.org/abs/1701.06548
14. Wei, H., Xie, R., Cheng, H., Feng, L., An, B., & Li, Y. (2022). *Mitigating Neural Network Overconfidence with Logit Normalization.* https://arxiv.org/abs/2205.09310
15. Geifman, Y., & El-Yaniv, R. (2019). *SelectiveNet: A Deep Neural Network with an Integrated Reject Option.* ICML 2019, PMLR 97. https://proceedings.mlr.press/v97/geifman19a.html (arXiv:1906.00129)
16. *Calibrated Selective Classification.* (2022). https://arxiv.org/abs/2208.12084
17. *Bag of Tricks for In-Distribution Calibration of Pretrained Transformers.* EACL Findings 2023. https://aclanthology.org/2023.findings-eacl.40.pdf
18. *Calibration in Deep Learning: A Survey of the State-of-the-Art.* (2023). https://arxiv.org/abs/2308.01222
19. Google (2025). *EmbeddingGemma 2* — model docs / model card. https://ai.google.dev/gemma/docs/embeddinggemma ; https://huggingface.co/google/embeddinggemma-2 ; transformers model_doc: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/embedding_gemma2.md
20. *A Formalism and Approach for Improving Robustness of Large Language Models Using Risk-Adjusted Confidence Scores.* https://arxiv.org/abs/2310.03283
21. Patra, A., et al. (2023). *Calibrating Deep Neural Networks Using Explicit Regularisation and Dynamic Data.* WACV 2023. https://openaccess.thecvf.com/content/WACV2023/papers/Patra_Calibrating_Deep_Neural_Networks_Using_Explicit_Regularisation_and_Dynamic_Data_WACV_2023_paper.pdf
