# Device research 2026 — data quality, training mechanisms, DeepSeek, competitive reality

Gathered 2026-10-15. Purpose: understand AC-v2 data + the full training-mechanism
landscape + recent 2026 models, to de-risk the decision-model goal and frame it
honestly (avoid "garbage" over-claims). Supersedes naive "CE vs proper-scoring".

## A. AC-v2 data (what we actually have)
- 186 train / 46 test. Fields: `goal` (task, ~unique), `step_instruction` (per-step),
  `image` (screenshot), `candidates[K]` (UI action strings + 'wait'), `gold` (INT index),
  `gt` (execution action {action_type, direction/x/y/text}).
- **`gold` = INDEX (0-5), K≈6** → choice-over-K, single ground truth.
- **`gt` = execution action**, DIFFERENT annotation signal (execution vs grounding).

### 3 data-quality findings (drives both training and the hard/easy split)
1. **Positional shortcut:** gold=5 is 57% of rows (train 105/186, test 26/46).
   Always-pick-last cheats to ~57%. → **MUST shuffle candidate order** in the
   decision head (d1 does this) or we measure position habit, not grounding.
2. **Label noise gold≠gt:** only ~60% of `cand[gold]` loosely contains
   `gt.action_type`; gold and gt differ (e.g. gold='wait' but gt=scroll down).
   → **natural hard vs easy split**: easy = gold clearly matches gt + clean candidates;
   hard = gold≠gt / ambiguous distractors. Perfect raw material for Q1/Q2 breakdown.
3. **Candidate quality mixed:** junk candidates exist ('click: ,', 'click: android.widget...',
   'open_app'), plus plausible-but-wrong ones → can stratify difficulty
   (plausible-wrong = hard distractor; clearly-wrong = easy).

## B. Training-mechanism landscape (much richer than CE vs proper-scoring)
| Family | Technique | Key value |
|---|---|---|
| Proper scoring | CE=log, Brier, spherical, RPS | **CE IS a proper scoring rule (log)**. "CE vs proper-scoring" is a FALSE dichotomy. Real issue = **generalization gap** (proper on train, overconfident on test). |
| Non-proper but calibrating | focal loss | Not proper, but often **better-calibrated on test** (2408.11598) — counterintuitive, worth testing. |
| In-the-loop (train-time) | soft-ECE (2108.00106), MMD distribution-matching (2310.20211), bilevel "Calibration in One Go" (2610.13693), entropy-max bilevel (2608.07419) | Optimize ECE-calibration directly during training, no separate post-hoc stage. |
| Calibration-aware RL | REWARDS (ICLR'26), RLCM (2604.23333), Calib-Aware Policy Opt (ACL'26), RL-ARC (2610.11352), CATTO (2601.23096), Balancing+Calib (ACL Findings) | All show **RLVR/RL collapses calibration** (confirms d1's RLVR finding). |
| Mechanism (Q1) | "Too Sharp, Too Sure" (2604.20614) | **ECE tracks curvature/sharpness through optimization** → calibration tied to optimization geometry (sharpness/margin), not just the loss. |

## C. DeepSeek V4.1-Flash (2026) — architectural validation
- **Causal Encoder-Decoder (CED):** 20-layer causal encoder + 20-layer decoder;
  decoder global KV projected from final encoder hidden states; **8B active prefill /
  16B decode**. → frontier principle: **encode the long input once, then cheaply
  decode/decide** (for input-heavy agentic workloads). Validates our
  "encode screenshot+state once -> rank K candidates cheaply".
- 45T tokens, SFT->RL->on-policy distillation; vision from scratch (DeepSeek-ViT) +
  MLP projector from start of pretraining; MIT license (open, but 552B).

## D. Competitive reality 2026 (honest, crowded)
- **Unsloth "Train your own Decision Model"** (Qwen3.8/Gemma4) — one-click.
- **jevtrainer-qwen35-2b-agent** (2026-10-08): Qwen3.5-2B LoRA, vision frozen,
  150k GUI screenshots, 751,702 records, Jev readout. **≈ our plan, already built.**
- **Microsoft-Decision-1** (2026): proprietary fast decision model.
- **d1 / decider / THX-01**: calibrated single-pass decision models exist.

## E. Remaining genuine gaps (SHARPENED — the honest frame)
"Build a calibrated single-pass decision model" and "train one for GUI" are BOTH
taken (jevtrainer/Unsloth/d1). Framing as such = garbage. Surviving narrow value:
1. **Q1+Q2 on a single-pass GUI decision head**: CE(generalization-gap) vs focal
   (non-proper) vs Brier vs soft-ECE/joint, on easy/hard split (built from gold≠gt
   noise + distractors), explained by curvature/sharpness mechanism.
   **No 2026 paper does this on a GUI decision head** (they use image classifiers or LLM reasoning).
2. **Q3 on-device**: EmbeddingGemma-2 **440M** head (not a 2B LLM) -> genuinely
   smaller than the LLM-LoRA decision models -> on-device reliability-per-cost for
   act/ask/refuse is a distinct path. d1-omni is 587M (no CoreML); decider is MPS but LLM-sized.

## F. Next-step implications (when goal resumed)
- G1: must **shuffle candidates** so embedding alignment is position-independent;
  stratify easy/hard by gold≠gt + distractor semantics.
- G2/G3: split objectives into CE / focal / Brier / soft-ECE(joint), measure ECE+Brier
  on easy AND hard, report curvature/sharpness correlation.
- Do NOT claim "we built a GUI decision model"; claim the calibration mechanism +
  on-device reliability-for-cost.
