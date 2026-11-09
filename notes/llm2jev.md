# How to Build an Open Decision Model (Jev-style "System One")

Goal of this note: a working recipe to post-train / fine-tune an existing open model (~1–4B) into a
calibrated, typed **decision model** — it returns a `choice` / `score` / yes-no (`noul`) answer
together with a probability distribution in **ONE forward pass and ZERO output tokens** (no text
generation). Later steps: benchmark on the Decision Index, deploy to mobile.

Anchors referenced throughout: **Jev** (TypeSafe AI, proprietary) and **d1 / d1-3B** (Liquid AI, open).

> Naming note (inconsistency to be aware of): the arXiv paper **2610.02076** is *titled*
> "LLM2Jev: LLMs Are Already Jev-Style Decision Models" but in the body calls its framework
> **LLM-as-Jev** (joint scoring of all options in one prompt). The open-source repo
> `Yinsongxu/LLM2Jev` is a *frozen, per-option* binary-judging approach (one yes/no prompt per
> option, normalized across options) that the paper cites separately by the same name. The two are
> different methods. See §1.

---

## Coverage Status

**Directly read (primary):**
- Full arXiv text of `2610.02076` (abstract, intro, methods §3, experiments §4, practical recipe §5, references). [1][2]
- Liquid AI "Open d1" blog — architecture & training recipe, benchmark tables, latency. [5]
- Liquid AI `d1-3B` Hugging Face model card — specs, API, Decision Index table, benchmark-as-decisions, vision. [6]
- TypeSafe "Introducing System One Models & Jev" — RLCD framing, System One, cost/speed comparators (heavily self-reported). [11]
- OpenJev-RLCD arXiv `2609.38850` — RLCD algorithm, variance identity, failure modes, results. [16]
- "Is RLCD Just a Regularised RLVR?" — RLCD = RLVR + Tsallis regularization, uniqueness theorems, eve-rlcd repro numbers. [19]
- `Yinsongxu/LLM2Jev` README — backends, prefill-only scoring, latency, JevBench score. [4]

**Read but not fully verified / secondary:**
- Jevstiller (distill-on-the-fly), kev-finetune (Modal fine-tuning harness) — descriptions read from landing/README, not executed. [20][21]
- Academic distillation calibration papers — abstracts/landing pages read; results not independently re-derived. [27][28]
- Selection-bias / option-order papers — abstracts read. [22][23][24]
- SmolVLM, basal-1.5-mini — model cards read. [25][26]

**Vendor marketing — treat as CLAIM, NOT EVIDENCE:**
- TypeSafe's "two orders of magnitude faster", "can't hallucinate", "no type errors is mathematically impossible", "193.6x faster / 444.6x cheaper". [11]
- Liquid AI's "best decision model under 10B", latency/throughput figures. [5][6]
- These are self-reported; no independent public replication found for Jev (proprietary) and no third-party rerun for d1-3B found yet. Liquid's Decision Index / public-benchmark numbers use public data but are vendor-run.

**Could not complete / open:**
- No public technical report for Jev's RLCD (TypeSafe only names it; algorithm is unpublished). [11]
- d1-3B recipe is qualitative ("averaged weights", "shuffling options", "fixing shortcuts") — no hyperparameter table published. [5]
- No found public replication of d1-3B's 48.57 Decision Index score outside Liquid AI.

---

## Executive summary (key findings)

1. **You likely do NOT need to fine-tune first.** A strong open causal LLM is already a good
   decision model out of the box when fed a numbered, prefix-free prompt and scored by summing
   next-token log-probs over bracketed numeric suffixes. Zero-shot Qwen3.5-4B hits 81.4% on public
   JevBench (ECE 0.057), matching/beating community fine-tunes on the same backbone. [1]
2. **Fine-tuning gives targeted, not universal, gains.** It helps weak backbones (Qwen3-0.6B) and
   narrow domains (77-way Banking77), and *diminishing returns / negative transfer* on capable
   backbones on broad OOD benchmarks. Best capable-backbone result is **LoRA + KL anchors**
   (Qwen3.5-4B → 84.0%, ECE 0.050). [1]
3. **The d1 recipe is simple and empirically dominant**: weight-averaging two backbones, then
   fine-tuning with multiple seeds/mixtures and merging; **option shuffling + shortcut removal +
   long-input training beat fancy techniques**. [5]
4. **Calibration is the whole point and it collapses easily.** RL with a verification reward (RLVR)
   makes models sharply overconfident (mean confidence ~0.99, Brier 0.43). RLCD scores the *actual
   answer distribution* with a **strictly proper score** (Brier/log), and its reward is provably
   "RLVR + exactly one entropy regularizer" (sparse Tsallis entropy, α=1). [16][17][19]
5. **Distillation from a reference decision model is the cheapest path to a deployable student**:
   collect the reference's full probability vectors on your own questions, shadow-test, promote.
   Teacher calibration drives student usefulness. [20][27]
6. **What usually fails**: label leakage (train/test n-gram contamination), option-order/position
   bias, calibration collapse (RLVR overconfidence / narrow-data distributional collapse), the
   "System-One collapse" (RLCD switches off reasoning), and generated-text runaway after the answer
   token (prevented by KL anchors). [1][16][22][23]

---

## §1. LLM2Jev / LLM-as-Jev: extraction + when fine-tuning is actually needed

**Source:** arXiv `2610.02076` (body calls the method *LLM-as-Jev*). [1][2]

### The prompt / answer interface
Query the model with its default chat template, **thinking mode disabled**:
```
State: {s}
Options:
[1] {o_1}
...
[K] {o_K}
Select one option. Answer only with its bracketed numeric identifier.
```
Condition the assistant turn on the fixed prefix **`Best answer: [`**, so choosing option k means
generating the suffix `k]`. [1]

### Training-free scoring (no params, no labels, no vocab change)
Score each candidate by accumulating log-probs over its suffix tokens, then softmax-normalize:
```
s_k = Σ_t log p_θ(c_k,t | x, c_k,<t)
P(k|x) = exp(s_k) / Σ_j exp(s_j)
```
Key design advantages:
- **Prefix-free suffix.** Every suffix ends in `]`, so no complete suffix is a prefix of another
  (`1]` is not a prefix of `12]`). This supports arbitrarily large K, unlike letter-logit readouts
  (bounded to ~16/26). [1]
- **Joint (listwise) conditioning.** All options are scored after conditioning on the full prompt,
  comparing them relative to the whole slate. This beats isolated per-option scoring: frozen
  `jev-local` (per-option log-probs, Qwen3.5-9B) = 74.9%, vs listwise on smaller Qwen3.5-4B = 81.4%. [1]
- **Shallow suffix trie.** Digit-level tokenizers (Qwen) give trie depth ⌊log₁₀K⌋+2; multi-digit
  tokenizers (Phi-4) give depth 2 → a 151-way task needs one prefill + ≤3 short forward passes
  (digit-level) or **one forward pass** (multi-digit). [1]
- **Parallel suffix scoring.** Cache KV for the prompt prefix, then teacher-force all candidate
  branches in parallel (mirrors Jev's parallel sampler). [1]

### When fine-tuning is (actually) NOT needed
- **Zero-shot Qwen3.5-4B = 81.4% (188/231) public JevBench**, ECE 0.057 — matches/beats SemIf
  (186 correct, ECE 0.061) and fine-tuned reflex-4B (184 correct). Only Winnow-12B (86.6%) and
  proprietary GPT-5.6 Sol (94.4%) are higher. [1]
- Beats SemIf by +3.2 pp macro across external benchmarks (MMLU +123/−56, MMLU-Pro +96/−41,
  ARC-C +28/−8). [1]
- **Banking77 (77-way intent) = 69.0%** in one pass; letter-logit readouts cannot do K=77. [1]
- **Consistency with generation**: greedy text-mode gives identical 81.4% with zero parse failures,
  same candidate as parallel scoring across all 231 items. [1]
- **Multimodal emerges zero-shot**: prepend image tokens → Qwen3.5-4B 82.4% MMBench (CircularEval,
  permutation-consistent) and 64.6% MMStar with no multimodal tuning; removing images collapses to
  15.2% / 28.8%. [1]

### When fine-tuning IS needed
- **Qwen3-0.6B fails out of the box**: 56.7% JevBench (above 48.1% raw-logit baseline) but
  **ECE = 0.278** (poor calibration), Banking77 **22.2%**, and CLINC150 (151-way) cross-entropy
  reaches **6.9 nats — worse than uniform** (ln 151 ≈ 5.0). [1]
- **Targeted domains** (e.g. high-cardinality intent routing) where base model is weak and annotated
  data exists. [1]
- General rule (§4.3.4): **model scale governs adaptation utility** — compact backbones get
  widespread structural gains; capable backbones only benefit on targeted axes. [1]

### The fine-tuning objective (when you do train)
- **Tree-factorized listwise loss.** At each trie node, renormalize probabilities over *legal*
  continuations only; the loss is the NLL of the gold path:
  `q_θ(k|x) = Π_t p(c_k,t | x, c_k,<t) / m_θ(c_k,<t)`, `ℒ_tree = -log q_θ(y|x)`. [1]
- **Three KL anchors** against a frozen reference model `p0`, to stop probability drift outside the
  candidate suffix:
  - `ℒ_mass`: Bernoulli KL on legal-vs-illegal probability mass at trie nodes.
  - `ℒ_out`: KL on the distribution renormalized over *illegal* tokens at trie nodes.
  - `ℒ_pos`: KL on the next-token distribution at context positions (reply start + after `]`).
  - Total: `ℒ = ℒ_tree + λ(ℒ_mass + ℒ_out + ℒ_pos)`, default **λ = 1**.
  - Tail approximation: keep top-64 tokens per anchored distribution (captures >94.5% of the mass)
    and put the rest in one tail bucket → rigorous lower bound on full KL. [1]
- **Training config**: AdamW under FSDP, conservative LR, linear warmup + cosine decay,
  **shuffled candidate ordering**, single epoch, and an automated validation check that halts if
  legal-token mass drops too far. Supports full-parameter or **LoRA** (all linear projections,
  merged into base after training → identical latency). [1]

### Fine-tuning results / failure modes (the empirical meat)
- **Intent-only data → heuristic shortcuts.** After 1 epoch, 4B shows affirmative confirmation bias
  on yes/no, suppresses "none of the above", relies on lexical matches, and gets overconfident.
  JevBench drops 81.4% → 77.5% (100 steps) → 74.9% (full epoch) even while in-distribution loss
  keeps dropping. **KL anchors cannot prevent this** because they allow redistribution among valid
  candidates. [1]
- **Format diversity fixes it.** Replacing narrow commonsense sets with six balanced reasoning
  benchmarks (Reason-9k) restores 81.4% JevBench, kills affirmative bias, restores calibration. [1]
- **The short-context trap.** Scaling short-context data (Reason-21k/30k) *degrades* long-context
  JevBench (p≤0.03); adding 3 long-context corpora (Reason+Long-12k) halts the regression at 4B and
  doubles long-context accuracy at 0.6B. [1]
- **KL anchors are mandatory.** Without anchors, standard metrics don't move but **generate-mode
  behavior collapses**: the 4B model keeps emitting redundant text after the identifier in 69/231
  queries; 0.6B has severe chat termination failures. Even λ≥0.01 removes almost all of it. [1]
- **λ = 1 is the default** across scales; lighter anchors trade conversational stability for slight
  in-domain accuracy. [1]
- **LoRA > full fine-tune on capable backbones**: 4B + LoRA λ=1 → **84.0% JevBench, ECE 0.050**
  (best overall), significantly beating full-parameter tuning (p=0.01). On 0.6B the two are close.
  LoRA alone cannot substitute for KL anchors. [1]

### Practical recipe (§5, verbatim guidance)
Default to a **training-free baseline**; reserve fine-tuning for verified deficiencies; align your
training mixture to the *test* distribution (binary verification, fallback "none of the above",
realistic lengths, diverse phrasings, prune keyword-solvable instances); keep KL anchors (λ=1);
use **LoRA** on capable backbones; **audit behavior beyond in-domain metrics** — select checkpoints
on OOD transfer probes, and inspect open-ended autoregressive completions (likelihood metrics miss
runaway). [1]

---

## §2. d1-style post-training recipe (the open template)

**Sources:** Liquid AI "Open d1" blog [5]; `d1-3B` model card [6]; d1-omni-600M [7].

### d1-3B (3.12B) — recipe
- Backbone: **LFM2.5-VL-3B** (3.1B decoder-only VLM, text+image). [6]
- **Weight averaging to make a better base**: "We averaged the weights of LFM2.5-2.6B and the text
  backbone of LFM2.5-VL-3B to create a better base model." [5]
- **Seed/model averaging on top**: "We then fine-tuned checkpoints with different random seeds and
  data mixtures before merging them again." [5]
- **The big lever**: "Training on long inputs, **shuffling answer options**, and **fixing shortcuts
  in the data** made a bigger difference than more advanced techniques." [5] ← This is the
  single most actionable sentence for our post-training recipe.
- **Model card specs**: 3.12B total; SigLIP2 NaFlex shape-optimized 400M vision encoder; context
  32,768 tokens; vocab 128,000. Requires `transformers>=5.14`, load with `trust_remote_code=True`. [6]
- **API shape**: `model.system_one(state, questions, images=None)` and
  `model.system_one_batch([(state, questions[, images])...])`. Questions use the Decision Index
  schema (`type` = `noul`|`choice`|`score`, `instructions`, `criteria`). Returns
  `{"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}}`. [6]
- Not a chat model; `output_tokens` is always 0. [6]

### d1-omni-600M (experimental) — the staged recipe
- Backbone: **LFM2.5-Encoder-350M** — a *bidirectional* encoder (not decoder-only). [5]
- Fine-tune backbone on decision tasks first; then add audio (FastConformer + adapter, frozen text
  backbone); then add vision (encoder taken from LFM2.5-VL-450M, adapter + **LoRA that is active
  only when images are present**, vision encoder frozen); then full fine-tune, **merge LoRA**, and
  **average weights with the previous checkpoint** to regularize. [5]
- **Takeaway**: for a decision model you can start from either a decoder-only VLM (d1-3B) or a
  bidirectional encoder with modality adapters (d1-omni-600M). The encoder-based route is a promising
  architecture for a "no generation" model — it never needs to decode.

### Reported results (vendor-run on public benchmarks — treat as self-reported)
| Benchmark | d1-omni-600M | d1-3B | Decider 2B | Decider 4B |
|---|---|---|---|---|
| SQuAD 2.0 | 74.0 | **85.3** | 67.7 | 76.0 |
| Civil Comments | **95.8** | 93.0 | 93.6 | 92.8 |
| MASSIVE intent | 86.1 | 87.3 | 78.1 | **88.3** |
| PubMedQA | 48.3 | **66.0** | 22.5 | 24.9 |
| BoolQ | 75.7 | 80.4 | 100 | 100 |
| XNLI | 93.7 | 94.2 | 92.2 | 94.2 |
| PAWS-X | — | — | — | — |
| **Mean** | **78.4** | **82.9** | 71.5 | 76.2 |
*(Note: the blog prints one table [5] and the model card prints another [6] with slightly different
values for the smaller models; both are vendor-run on public data. Use [6] for the Decision Index
table and [5] for the mixed table.)*
- **Decision Index v0.2.1 (public split)**: d1-3B = **48.57**, ahead of every model under 10B and of
  **Decider 35B-A3B (47.11)**, a model ~12× larger. d1-omni-600M = 15.95. [5][6]
- **d1-3B sub-scores**: Knowledge 23.8, Language 56.4, Retrieval 52.8, **Tools 74.5**, Arts 36.3. [6]
- **Vision**: d1-3B 74.1 mean across 11 public image benchmarks vs base LFM2.5-VL-3B 73.9; with
  images removed the same questions score 45.1 (answers genuinely come from images). [6]
- **DecisionBench**: 71.8 (eng v1, all 23,900 rows); **Fast Decisions** dev: 69.3. [6]

### Latency (vendor-measured; relevant for mobile)
- d1-3B **8 ms / question on RTX 4090**, 9 ms AMD MI325X, **30 ms Apple M5 Pro**, 16 ms Jetson AGX
  Thor, 26 ms AGX Orin 64GB, **50 ms Jetson Orin Nano**. 3 questions ≈ 1.3× one question. 64 states
  packed → 475/s (RTX 4090), 1,106/s (MI325X). [5][6]
- Day-one **llama.cpp** support across Apple/AMD/Qualcomm/NVIDIA with NVFP4. [5]

> **Practical read for our project**: the d1 recipe is *not* "high-tech" — it is weight averaging +
> seed/data-mixing + merge, plus **three data/format interventions** (long inputs, shuffled options,
> de-shortcut data). This is directly transferable and cheap.

---

## §3. RLCD-style RL for calibrated decisions

**Sources:** OpenJev-RLCD arXiv `2609.38850` [16]; repo `ZimmyGao/openjev-rlcd` [17]; analytic
"RLCD as regularised RLVR" [19]; `eve-rlcd` [18].

### The core idea (the thing to copy)
A decision model has no text; what it reports is a **distribution** `u` over options. RLCD trains the
model so that `u` is *calibrated* — **a strictly proper scoring rule of the distribution it actually
commits to** (Brier or log score), not a 1/0 verification reward.
- Readout: sample a rationale `r`, it ends at `Answer:`; then
  `u_θ(x,r) = softmax(logits_θ(x,r)[option tokens])`. [16]
- Score (shifted, negated Brier): `J(u,y) = 2·u_y − ‖u‖² = 1 − ‖u − e_y‖²`. Unique maximizer is the
  true posterior `q(x)`. [16]

### Why naive RL/RLVR breaks calibration (one identity)
`J(E_r u) = E_r J(u) + Var_r(u)`. So **scoring the mixture of several samples pays a bonus for
rationales that disagree**, whether or not the disagreement is grounded. Corollary: **RLVR is the
mixture objective with its diversity (Gini) term removed**; RLVR's optimum is the one-hot policy on
the argmax → wildly overconfident. [16] The analytic post [19] confirms: RLVR's expected reward is
"the outcome-dependent part of the Brier score"; RLCD adds back the missing entropy term. [19]

### Two failure modes (and the antidote)
1. **"System-One collapse"**: a Brier-punished model learns to hedge after an *empty* rationale
   faster than after reasoning, and the policy gradient moves all probability mass to the empty
   rationale → reasoning turns off (`r` becomes just the marker within ~100 steps). **Antidote: a KL
   anchor** to the reasoning model (β = 0.04). [16]
2. **Drowned by the policy gradient**: the score-function term's gradient norm is **7–12×** the
   pathwise term, dominating Adam's second-moment estimate and taking the useful low-variance readout
   signal in steps sized by the noise. [16]

### The recipe: "calibrate, then reinforce"
- **Stage 1 (calibrate the decision)**: train the post-rationale decision with a proper score on the
  model's *own* rationales — pathwise only, rationales sampled on-policy but not reinforced. [16]
- **Stage 2 (reinforce the reasoning)**: from the stage-1 checkpoint, reinforce reasoning with the
  *same* proper score as reward, a down-weighted score-function term (c = 0.3), and a KL anchor at
  the calibrated model (β = 0.04). Uses leave-one-out baseline. [16]
- Hyperparams used: Qwen3-1.7B, AdamW lr 2e-6, 8 questions & M=4 rationales per step, stage1 400
  steps + stage2 400 steps. [16]

### Reported results (Qwen3-1.7B, 3 seeds, paired tests)
- **GSM8K-Verify** (reasoning-essential): SFT+TS = 70.4%; two-stage RLCD +21.8 pts and −0.247 Brier.
  At the same accuracy as GRPO (91.8% vs 92.2%, n.s.), **RLCD's confidence ranking is far better**:
  AURC 0.034 vs 0.061; at 5% error budget RLCD decides **81%** of items vs **19%** for GRPO. GRPO
  needs fitted T≈7, RLCD T=1.4. [16]
- **MMLU-Pro** (knowledge-heavy): RLCD +5.3 pts over SFT+TS, +5.2 over GRPO+TS; at 20% error budget
  decides 28% vs 16% (SFT+TS) and none (reasoning baselines). [16]
- **ChaosNLI** (annotator disagreement / aleatoric): RLCD **cannot** beat cross-entropy (JSD 0.075 vs
  0.074) and correctly discovers it should stop reasoning. Point: prefer a fast direct scorer when
  uncertainty is aleatoric. [16]
- **Ablation**: more policy gradient is monotonically worse; explicit-probability RLCD (no rationale)
  = Brier gradient and never beats cross-entropy (CLINC150 94.6% vs 96.3%); RLVR on the same collapsed
  to the mode. [16]

### "Is RLCD just regularised RLVR?" — the sharpened view [19]
- Maio's `eve-rlcd` reward is `r = c_a − sg(π_a)`; REINFORCE on it follows **−½∇‖π − e_y‖²**, i.e.
  **half the Brier gradient**. [18][19]
- Rewritten: `−½‖π − e_y‖² = π_y (RLVR) + ½(1 − ‖π‖²) (sparse Tsallis entropy W) − ½`. [19]
- **RLVR regularised by αW has optimum `sparsemax(p*/α)`; RLCD is α=1 → `sparsemax(p*) = p*`**, the
  calibrated posterior. [19]
- **Uniqueness theorem**: among all additive reward corrections, RLCD's subtraction is *the only one*
  that calibrates. KL, Shannon-entropy, temperature, and proximal/trust-region regularisers all fail
  to hit the true posterior. [19]
- **Chain-of-thought caveat**: when the action is a reasoning chain τ, `π(τ) ≈ 0`, so
  `c − sg(π(τ)) ≈ c` and the update degenerates to plain RLVR. The regularizer must act on the
  **answer marginal**, not the chain — which is exactly why the OpenJev-RLCD "read the readout after
  `Answer:`" formulation matters. [19]
- `eve-rlcd` repro numbers (Qwen3-0.6B): RLCD acc 0.808 / ECE 0.023 / Brier 0.267 / mean conf 0.830;
  RLVR acc 0.778 / ECE 0.213 / Brier 0.432 / mean conf **0.991**. **Caveats**: RLCD *maintains* but
  does not clearly *improve* calibration over the SFT warmup, and the RLCD arm ran at 5× the LR of
  the RLVR arm. [18][19]

> **Practical read**: RLCD is the one algorithm that provably calibrates. Its essential ingredient is
> a **strictly proper score of the model's own reported distribution** (the "readout"), plus a KL
> anchor to prevent reasoning collapse. You can implement it directly on top of the LLM2Jev/LLM-as-Jev
> readout (which gives you the distribution in one pass).

---

## §4. Distill from a reference decision model

**Sources:** Jevstiller [20]; kev-finetune [21]; academic KD calibration papers [27][28][29][30]. (The
academic ones are read at abstract/landing level.)

### The on-the-fly distillation pattern (Jevstiller)
- Put a proxy in front of a repeated Jev classification call. First every request goes to Jev; from
  Jev's own answers (with **full probability distributions**) it trains a **small local model** on
  your traffic, **shadow-tests** it against Jev, and promotes it when it agrees within a budget you
  set. After promotion the proxy answers what it's sure about locally, and forwards uncertain/novel
  input to Jev. [20]
- This is directly the "distill-from-a-reference decision model" approach: the reference's *probability
  vectors*, not just its labels, are the targets — which preserves calibration.

### The fine-tune-and-serve harness (kev-finetune)
- Fine-tune an open Jev-style decision model on your own questions, get calibrated probabilities,
  serve as a TypeSafe System One endpoint. Runs on Modal from six short scripts; describe the decision
  in `workload.json` (input + questions in System One shape). [21]

### Why distillation for *calibrated* decisions is subtle (academic)
- **Teacher calibration drives student accuracy**: there is a strong correlation between the teacher's
  calibration error and the student's accuracy, so a well-calibrated teacher distills better. [27]
- **Distillation traps**: tail noise, off-policy instability, and the teacher–student gap distort
  training signals and show up as overconfident hallucination, self-correction collapse, and local
  decoding degradation. A calibration knob (temperature guard) helps. [28]
- **Capacity mismatch**: a "better" teacher is not always better for a smaller student. [29]
- **Decision-only KD**: when teacher logits are unavailable, estimate logits from the teacher's
  decision distribution. [30]
- **Student can deviate / disobey** and still outdo the teacher; matching teacher probabilities is not
  always optimal. [31]

> **Practical read**: for our project the lowest-cost path is **distill from a strong reference decision
> model** (e.g. d1-3B or Jev) using its full probability vectors as targets on a small student, but
> expect the teacher–student gap and capacity-mismatch problems to matter, and **re-check the
> student's calibration** after distillation rather than trusting it.

---

## §5. Practical choice of an open base model (~1–4B)

**Sources:** LLM2Jev/LLM-as-Jev [1]; d1 cards [6][7][8][9]; OpenJev-RLCD [16]; SmolVLM [25][26];
basal-1.5-mini [26].

### Concrete models in this class
| Model | Size | Kind | Notes / source |
|---|---|---|---|
| **LFM2.5-VL-3B** | 3.1B | decoder-only VLM (text+image) | Base for d1-3B; SigLIP2 NaFlex 400M vision, 32,768 ctx, 128k vocab. [6][8] |
| **LFM2.5-Encoder-350M** | 350M | bidirectional encoder (+audio/vision adapters) | Base for d1-omni-600M; a "no-decode" architecture. [5][9] |
| **Qwen3.5-4B** | 4B | text LLM (thinking off) | Best zero-shot in LLM-as-Jev (81.4% JevBench). Digit-level tokenizer → shallow trie. [1] |
| **Qwen3-0.6B / Qwen3-1.7B** | 0.6B / 1.7B | compact LLM | Shows the "weak backbone" case needing fine-tuning (0.6B ECE 0.278). [1][16] |
| **SmolVLM-2B / SmolLM** | ~2B | compact multimodal | SmolVLM-2B is strong per memory footprint on image/video; a candidate. [25] |
| **basal-1.5-mini** | 1.5B | open typed-decision model | Already a Jev-inspired decision model; an existence proof / potential reference. [26] |

### What makes a good base model (synthesis, partly inference from [1][5][6][16]])
- **Strong general reasoning/knowledge at its size.** Because decision models *inherit* the base's
  scaling, a capable 3–4B base is already a good decision model zero-shot. [1][5]
- **Decoder-only VLM for multimodal**, or a **bidirectional encoder + adapters** for pure no-decode
  ter (d1-omni-600M). [5][6]
- **Tokenizer matters for cost.** Digit-level tokenizers (Qwen) → trie depth ⌊log₁₀K⌋+2, so K high
  costs a few short forward passes; multi-digit tokenizers (Phi-4) → depth 2 → **one forward pass**.
  This directly drives per-call prefill cost. [1]
- **Open-weight + permissive license + local runtime (llama.cpp) + small enough for edge** if the
  target is mobile. d1-3B runs in 50 ms on Jetson Orin Nano with day-one llama.cpp + NVFP4. [5][6]
- **For our fine-tune target**, the paper stresses that **backbone capacity governs adaptation
  utility**: a "capable" 3–4B tuned with LoRA + KL anchors is the sweet spot — strong enough to be
  good zero-shot, small enough to fine-tune and run on device. [1]

> **Practice note / inference**: choose a small open VLM around 3–4B (e.g. LFM2.5-VL-3B, Qwen3.5-4B,
> SmolVLM-2B). Start training-free (LLM-as-Jev joint bracket scoring). Only fine-tune (LoRA + KL
> anchors, d1-style option-shuffling + shortcuts removal) if your target domain is weak.

---

## §6. Risks / what usually fails

**Sources:** LLM-as-Jev §4 [1]; OpenJev-RLCD [16]; RLCD-regularised-RLVR [19]; d1-open [5]; selection
bias literature [22][23][24].

1. **Label leakage / train-test contamination.** LLM-as-Jev is explicit: it filters every training
   instance sharing a 13-word n-gram (or a ≥6-word identical sentence) with any eval benchmark and
   purges duplicate validation inputs. Without this, reported gains are inflated. [1]
2. **Option-order / position bias.** LLMs are not robust multiple-choice selectors; they exhibit
   "selection bias" preferring option IDs, largely token bias. Mitigation: **shuffle candidate order
   during fine-tuning** (d1 does this; LLM-as-Jev does this so numeric IDs carry no static meaning),
   and optionally cyclic-permutation averaging at inference (AnyJev) — at proportional latency cost.
   [1][5][22][23][24]
3. **Calibration collapse from RL/RLVR.** RLVR rewards only correctness → sharp overconfidence
   (mean conf ~0.99, Brier 0.43). The only additive reward fix is RLCD's subtraction. Temperature
   scaling fixes the *level* but not the *ranking* of confidence (AURC stays flat for GRPO/RFT), so
   it can't restore selective decision-making. [16][19]
4. **Narrow-data distributional collapse.** Intent-only fine-tuning produced affirmative "yes" bias,
   "none of the above" suppression, lexical shortcuts, and overconfidence — and **KL anchors can't
   stop it** because they permit redistribution among valid candidates. Fix by diversifying answer
   formats (binary, fallback, logic) and lengths. [1]
5. **Short-context trap.** Training on only short inputs degrades long-context decision ability, even
   as overall loss drops. Must match the production input-length distribution. [1][5]
6. **"System-One collapse" in RLCD.** A proper score on the decision distribution can make a model
   stop reasoning (empty rationale) to hedge; needs a KL anchor. [16]
7. **Generated-text runaway after the answer.** Without KL anchors the model keeps generating past the
   identifier (69/231 queries at 4B) or fails to terminate in chat (0.6B) — and **standard likelihood
   metrics hide it**. Always audit open-ended completions. [1]
8. **Distillation traps.** Tail noise, off-policy instability, teacher–student gap → overconfident
   hallucination, self-correction collapse, local decoding degradation; a miscalibrated teacher
   distills worse. [27][28]
9. **Per-option scoring underestimates joint tasks.** Scoring each candidate in isolation (separate
   forward passes / per-option binary judging) loses listwise comparison and trails joint scoring on
   the same backbone. [1][4]

---

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|---|---|---|---|---|
| 1 | LLM2Jev / LLM-as-Jev (arXiv 2610.02076, "You" / Yinsongxu) | https://arxiv.org/html/2610.02076 | Architecture-preserving joint bracket-numeric scoring; training-free 4B = 81.4% JevBench, ECE 0.057; fine-tune helps weak backbones/targeted domains; KL anchors + LoRA → 84.0%, ECE 0.050 | primary (academic) | high |
| 2 | Same paper, abstract page | https://arxiv.org/abs/2610.02076 | Canonical abstract/citation | primary | high |
| 3 | HF papers page for the same paper | https://huggingface.co/papers/2610.02076 | Paper landing | primary | high |
| 4 | Yinsongxu/LLM2Jev repo README | https://github.com/Yinsongxu/LLM2Jev/blob/main/README.md | Open-source per-option binary-judging impl; prefill-only scoring; SGLang/Transformers/MLX; Qwen3.5-4B JevBench 76.2%, P50 48ms | primary (code) | high |
| 5 | Liquid AI "Open d1" blog | https://www.liquid.ai/blog/d1-open | d1 recipe: weight avg backbones, seed/mixture merge, **long inputs + option shuffling + shortcut removal**; d1-3B 48.57 DI, d1-omni 15.95 | primary (vendor) | medium (self-reported) |
| 6 | LiquidAI/d1-3B HF card | https://huggingface.co/LiquidAI/d1-3B | Specs (3.12B, SigLIP2 400M, 32k ctx, 128k vocab), API `system_one`, DI table, benchmark-as-decisions, vision, latency | primary (vendor) | medium (self-reported) |
| 7 | LiquidAI/d1-omni-600M HF card | https://huggingface.co/LiquidAI/d1-omni-600M | 600M experimental multimodal decision model | primary (vendor) | medium (self-reported) |
| 8 | LiquidAI/LFM2.5-VL-3B | https://huggingface.co/LiquidAI/LFM2.5-VL-3B | Base VLM for d1-3B | primary (vendor) | medium |
| 9 | LiquidAI/LFM2.5-Encoder-350M | https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M | Bidirectional encoder base for d1-omni | primary (vendor) | medium |
| 10 | Liquid AI "The most capable decision model" | https://www.liquid.ai/blog/d1-decision-model | d1 first model to rival Jev on text, then vision | primary (vendor) | medium (marketing) |
| 11 | TypeSafe "Introducing System One Models & Jev" | https://typesafe.ai/blog/introducing-system-one-models-and-jev | Names RLCD as the method; System One framing; speed/cost; claims are self-reported | primary (vendor) | low (marketing, not evidence) |
| 12 | TypeSafe System One docs | https://docs.typesafe.ai/concepts/system-one | Definition of System One models | primary (vendor) | medium |
| 13 | Jev (AI model), Wikipedia | https://en.wikipedia.org/wiki/Jev_(AI_model) | Release/context; mine for background only | secondary | medium |
| 14 | JevBench | https://benchmarkheaven.com/jev-models/v1.4.2.2 | 231 public items; benchmark for Jev-class models | primary (benchmark) | high |
| 15 | Decision Index HF space | https://huggingface.co/spaces/multimodalart/jev-decision-index | The benchmark d1-3B scores 48.57 on v0.2.1 | primary (benchmark) | high |
| 16 | OpenJev-RLCD (arXiv 2609.38850) | https://arxiv.org/html/2609.38850 | RLCD algorithm (proper score of readout; calibrate-then-reinforce; variance identity; failure modes); Qwen3-1.7B results | primary (academic) | high |
| 17 | ZimmyGao/openjev-rlcd repo | https://github.com/ZimmyGao/openjev-rlcd | Code/run-logs for RLCD paper | primary (code) | high |
| 18 | anthony-maio/eve-rlcd repo | https://github.com/anthony-maio/eve-rlcd | Maio's RLCD repro on Qwen3-0.6B (results cited via [19]) | primary (code) | medium |
| 19 | "Is RLCD Just a Regularised RLVR?" | https://zhenhuan-yang.github.io/rlcd-regularised-rlvr/ | RLCD = RLVR + sparse Tsallis entropy (α=1); uniqueness; eve-rlcd numbers; CoT caveat | secondary (expert analysis) | high |
| 20 | Jevstiller | https://jevstiller.pages.dev/ | Distill Jev on the fly: train small local model from Jev's probability vectors, shadow-test, promote | primary (product) | medium |
| 21 | kev-finetune | https://github.com/jaredpalmer/kev/blob/main/skills/kev-finetune/README.md | Fine-tune open Jev-style decision model, serve as System One endpoint on Modal | primary (code) | medium |
| 22 | Unveiling Selection Biases (ACL Findings 2024) | https://aclanthology.org/2024.findings-acl.333.pdf | LLMs exhibit option-order/token selection biases | primary (academic) | high |
| 23 | LLMs Are Not Robust Multiple Choice Selectors (ICLR 2024) | https://proceedings.iclr.cc/paper_files/paper/2024/file/54dd9e0cff6d9214e20d97eb2a3bae49-Paper-Conference.pdf | Token-bias origin of option-ID preference | primary (academic) | high |
| 24 | AnyJev: LLM Calibration and Option-Order Bias (Wavect) | https://wavect.io/blog/anyjev-calibration-option-order-bias/ | Cyclic permutation + batch calibration to reduce order/prior bias; separates marketing claims from mechanism | secondary (technical blog) | medium |
| 25 | SmolVLM | https://huggingface.co/HuggingFaceTB/SmolVLM-Base | Compact open multimodal model candidate | primary (vendor) | medium |
| 26 | basal-1.5-mini | https://huggingface.co/Remek/basal-1.5-mini | Existing open Jev-inspired typed-decision model (1.5B) | primary (vendor) | medium |
| 27 | The Role of Teacher Calibration in KD | https://arxiv.org/html/2508.20224 | Teacher calibration error correlates with student accuracy | primary (academic) | high |
| 28 | Distillation Traps and Guards (ACL 2026) | https://aclanthology.org/2026.acl-long.908.pdf | Tail noise / off-policy instability / teacher-student gap → overconfident hallucination, collapse | primary (academic) | high |
| 29 | Revisit the Essence of Distilling Knowledge through Calibration | https://proceedings.mlr.press/v235/fan24d.html | Capacity mismatch in distillation | primary (academic) | high |
| 30 | Decision-based KD beyond logits | https://aclanthology.org/2023.acl-long.738.pdf | Estimate logits from teacher decision distribution when logits unavailable | primary (academic) | high |
| 31 | On student-teacher deviations in distillation | https://proceedings.nips.cc/paper_files/paper/2023/file/12d286282e1be5431ea05262a21f415c-Paper-Conference.pdf | Student may deviate from teacher and still improve | primary (academic) | high |

---

## Recommendation (how I'd build it, if asked)

1. **Base**: an open small VLM ~3–4B (LFM2.5-VL-3B or Qwen3.5-4B). Test training-free via the
   LLM-as-Jev joint bracket-numbering readout. [1]
2. **If weak on your domain**: fine-tune with LoRA + KL anchors (λ=1) using LLM-as-Jev's tree-factorized
   listwise loss; use the d1 data interventions (long inputs, option shuffling, shortcut removal) and
   LLM-as-Jev's contamination filter. ~300–1000 gradient steps. [1][5]
3. **Calibration**: either train with the proper-score objective (LLM-as-Jev's tree loss is listwise;
   add a proper-score reward if you do RL), or **distill** from a well-calibrated reference (d1-3B / Jev)
   using its full probability vectors, then re-check calibration. [1][16][20][27]
4. **Serving for mobile**: prefill-only scoring (SGLang/Transformers/MLX; prefix/Radix-cache reuse),
   batch packing, quantization (NVFP4/llama.cpp) — modeled on LLM2Jev [4] and d1-3B [6].

---

## Open questions (need follow-up)
- **Jev's RLCD algorithm is unpublished** — TypeSafe only names it [11]; no public technical report.
- **d1-3B's 48.57 Decision Index** is vendor-run; need an independent replication or third-party
  leaderboard confirmation. No public non-Liquid replication found.
- **d1 hyperparameters** (LR, batch, epochs, LoRA rank, seed count) are not published — only the
  qualitative recipe [5].
- **Which is better for our target — decoder-only VLM vs bidirectional-encoder+adapters?** d1-omni-600M
  argues the encoder route is viable for "never decode" models, but it's experimental.
- **Tokenizer effect** needs a real cost measurement on our chosen base (digit-level vs multi-digit)
  for high-K tasks.
- **Distillation vs direct RLCD** tradeoff not benchmarked side-by-side in a shared setting.

---

*This note was written by the evidence-gathering subagent for the open decision-model project. It does
not modify `references/bib.bib` or `references/ledger.md` (the parent aggregates sources).*
