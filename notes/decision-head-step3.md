# Step 3 — Calibrated Single-Pass Decision Head (design spec)

> **Goal:** the mechanism that makes an on-device decision model *reliable*. This is Q2 from `notes/GOAL.md` (which training objective preserves calibration). Build order per `notes/sota-arch-deep.md` Step 3. **Design now (CPU-free); training blocked on GPU** (see `AGENTS.md` GPU blocker).

## 1. What we build
A **choice-over-K** decision head on top of a frozen/already-SFT'd base decoder (LFM2.5 1.2B→2.6B), producing a **calibrated probability vector over K candidate actions** in **one forward pass, zero output tokens**. Two heads, kept separate:
- **A) Action head** — scores K candidate actions (the next move). Trained CE (task accuracy).
- **B) P(success) head** — a *separate scalar* calibrated head predicting "this choice will be correct." This is the one we calibrate and gate on.

## 2. Inputs / outputs (contract)
```
state  : goal + screenshot/state-embedding (+ interaction context, a11y if available)
K      : candidate actions (from a candidate builder — e.g. AC-v2 choice-over-K)
f(state, [a_1..a_K]) -> p_action ∈ Δ^K, p_success ∈ [0,1]   (single forward pass, zero tokens)
```
- No text generation → no parsing/hallucination, deterministic, low latency.
- `p_action` → argmax + margin. `p_success` → the gate input.

## 3. Training objectives (the Q2 comparison matrix)
We compare exactly three training schemes for the **P(success) head** on a hard/OOD split:

| Scheme | Action head loss | P(success) head loss | Hypothesis |
|---|---|---|---|
| **CE-only** | CE (task) | — (derive P(success) as softmax max / confidence proxy) | Overconfident on hard/OOD (the failure mode we saw: conf 0.885 vs acc 0.25) |
| **CE + proper-scoring** | CE (task) | Brier / log-loss on the *correct/incorrect* label | Calibration preserved where CE degrades |
| **CE + joint-calibration** | CE (task) | Joint: P(success) trained jointly with action head, success-adaptive scaling | Best ECE at fixed coverage (target) |

- **Data:** AC-v2  398-step protocol; build candidates with `benchmarks/build_acv2_candidates.py` (a11y-guided, 14/20 grounded). Use the **AC-v2 train split** (`benchmarks/data/android_control_train.jsonl`) for the memory bank / candidate pool; eval on test.
- **GUI-Libra action-aware reweighting** (from `notes/sota-arch-deep.md`): weight the action head loss by a task difficulty prior so easy steps don't dominate.
- **KL trust region:** constrain the head update so it doesn't wreck the frozen base's representations (small KL penalty vs. a reference head / EMA).
- **Seeds + fixed protocol:** seed 3407 (matches prior 0.7275/ECE 0.0316 run), fixed prompt, fixed candidate order.

## 4. Metrics (must report, per `AGENTS.md` citation rules)
- step accuracy (Pass@1), ECE(15 bins), Brier, reliability diagram, coverage–risk, median/p95 latency.
- **The Step-3 pass condition (P3):** on the hard/OOD split, the proper-scoring scheme keeps ECE low where CE-only collapses, at comparable step accuracy (within ~±2 pts).

## 5. Why separate P(success) head (not softmax-max)
A VLM's `argmax softmax` confidence is not calibrated and is tied to the *action* distribution, not to *correctness*. A dedicated P(success) head is trained directly on the correct/incorrect label → gives us a true calibration signal to gate on (Step 4).

## 6. GPU-block note
Training requires GPU (currently unavailable on both Colab T4 and Kaggle — see `AGENTS.md`). This spec is ready; the moment GPU is available (Colab reset or Kaggle web-UI GPU enabled), train CE-only first (baseline), then proper-scoring, then joint — 3 seeds each.

## 7. Scaffold
`models/decision/` currently holds the prove-of-concept `qwen_lora` (0.7275 / ECE 0.0316) — a **text-only decision model**. Step 3 extends this with (a) the choice-over-K candidate input, (b) the separate P(success) head, (c) the three-way objective comparison.


## Related
- **QUAN-Transcribe lessons (L1-L10)** — see `notes/quan-transcribe-lessons.md` for transferable design lessons from a single-pass non-autoregressive SOTA system (encoder top-layer finetune, light reader, layer-mix, bf16 single-pass, gate): apply to this head (layer-mix latent, freeze encoder, bf16). [quan2026transcribe] [google2026embeddinggemma2]
