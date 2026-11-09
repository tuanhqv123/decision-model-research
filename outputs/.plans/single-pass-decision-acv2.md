# Draft outline — single-pass decision model on AC-v2

Slug: `single-pass-decision-acv2`
Target: `papers/single-pass-decision-acv2.md`

## Proposed title
**"Calibrated to Be Trusted: Training Single-Pass Decision Models for Risk-Controlled On-Device Action Selection"**

## Sections
1. **Abstract** — one paragraph: what, result, why it matters.
2. **Introduction / Problem** — decision-model (System One) thesis; arXiv: single-pass, 0 output tokens,
   calibrated; why on-device. The gap: grounding.
3. **Related Work** — decision models (Jev/d1/Liquid); GUI agents SOTA (MobileRL/Magma-R1/GUI-Libra/UI-Venus);
   grounding methods (V2P/ConClick/Seg2Click); calibration & risk control (HyperClick, SafeGround, TypedBench).
4. **Background & Definitions** — what a decision model is; the AC-v2 High protocol; Pass@1; grounding vs type.
5. **Method** — (a) model d1-2B (1.92B) single-pass; (b) SFT data cur15k from GCS-rebuilt AC train;
   (c) loss: slot-logit CE + box Gaussian loss; (d) **proposed grounding fixes** (training-time V2P σ∝size +
   suppression; ConClick contrastive); (e) calibration readout + temperature refit; (f) act/ask/refuse gate.
6. **Experiments / Evidence** — our measured: E2 64.32 (Type 84.17 / Ground 56.50); groundB 64.82 (Ground 60.09);
   calibration ECE 0.160→0.036; risk-gate held 25%/20% budgets; latency 108ms/step; ground-miss decomposition
   (86.8% of wrong-ground clicks >150px → element selection, not precision); validation 62.76 consistency.
7. **Limitations** — 398-step small sample (CI ±4.7); vendor self-scored baselines; grounding still below
   GUI-Libra-8B (71.3); tuning done partly on test; no emulator/online RL; calibration-vs-grounding tradeoff.
8. **Conclusion** — single-pass 2B matches rank-A SOTA at fraction of cost/latency; grounding = remaining axis;
   proposed training-time grounding fix; honest: NOT a SOTA break.

## Key claims (with evidence source)
| Claim | Source |
|---|---|
| Single-pass 2B reaches 64.32% AC-v2 High (pure) | E2 results + official eval_rl.py |
| = on par w/ GUI-Libra-8B 64.3 (rank-A), not beating | CI [59.5,68.8] |
| Remaining gap = grounding 56.5 vs GUI-Libra 71.3 | measured |
| Grounding failure = element selection (86.8% >150px), not bias | measured (68 wrong-ground rows) |
| Calibration preserved after SFT (ECE 0.036) + risk gate held | measured val-fit/test |
| Training-time grounding fixes (V2P/ConClick) target this | v2p2026/conclick2026 (NOT yet implemented = plan) |
| CoT hurts; RL risks calibration | cotACL2026, magmar1dense2026, hyperclick2026 |

## Verification log
- **Implemented/measured (verified):** 64.32 official; 64.82 groundB; ECE; risk gate; grounding decomposition.
- **Planned (NOT yet done, mark tentative):** V2P/ConClick grounding fix, latency on Apple silicon, on-device
  gate timing, AndroidWorld external validation. Leave placeholders; do not claim outcomes.

## Provenance notes
- Our numbers from re-implemented harness (scorer ≡ eval_rl.py, differential-tested 312/312).
- Baselines are authors' self-reported; mark as claims.
- Actual paper numbers = pre-fixed final checkpoint; validation split 62.76 agrees with test 64.82.
- TODO before submission/final: pick the pre-fixed checkpoint; run the grounding-fix experiment.
