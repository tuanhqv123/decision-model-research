# Plan: paper draft "decision-model-androidworld" (2026-11-12)

**Working title:** *A 2B single-pass decision model as a mobile agent: per-step accuracy, task success, and what the gap is made of.*

**Draft:** `papers/decision-model-androidworld.md`. It supersedes the AC-v2 framing of `papers/single-pass-decision-acv2.md`, which stays as it is because `check_record.py` verifies it.

## Sections
1. Abstract
2. Introduction: the claim and why it matters
3. Related work: generative GUI agents (MobileRL, MAI-UI, UI-TARS, GUI-Owl, AndroidGen, V-Droid, M3A), decision models (TypeSafe Jev, LiquidAI d1), datasets (AndroidControl, AndroidControl-Curated, AMEX), with a comparability note
4. Method: the decision contract (state, typed questions, pointer copy), the harness, the training data and its disclosure (`awgen` uses AndroidWorld goal templates)
5. Experiments
   - 5.1 Setup and metrics
   - 5.2 Main results: AC-High, AC-Low, AndroidWorld (pass@1)
   - 5.3 Per-step vs whole-episode accuracy
   - 5.4 AndroidWorld by difficulty and complexity, with step counts
   - 5.5 Harness and environment effects (paired)
   - 5.6 Image ablation
   - 5.7 Latency
   - 5.8 Failure taxonomy of the 91 failed tasks
   - 5.9 Negative results
   - 5.10 Zero-shot pattern probes
6. Limitations
7. Conclusion
8. Sources

## Key claims, and the artifact behind each
| Claim | Artifact | Status |
|---|---|---|
| AC-High 61.7% per step (5,213/8,444); 344/1,543 episodes all-right | `benchmarks/results/2026-11-12-v4/actest/perstep-v4_s0_ckpt_2482_actest_v4.jsonl` | measured |
| AC-Low 89.3% (7,544/8,444); 932/1,543 | `perstep-v4_s0_ckpt_2482_actest_v4_low.jsonl` | measured |
| AndroidWorld 25/115 = 21.7% (v4 + harness V12) | `2026-11-12-maskenv/androidworld_maskenv.tgz`, `compare_A_AG.out`, `difficulty_report.md` | measured |
| Difficulty table and C1–C4 table | `aw_report.py` on the same traces | measured |
| Harness-only gains: V8→V11 5 wins / 0 losses; V11→V12 5 / 1 | `compare_v4_A_B.out`, `compare_A_AG.out` | measured |
| Image ablation 519 → 337 / 827 | `2026-11-12-imgablate/` | measured |
| Model latency p50 135 ms / p95 181 ms; environment p50 1,624 ms (1,335 steps) | v4g step rows | measured |
| Failure taxonomy (91 tasks) | `2026-11-12-failanalysis/*.tsv` | measured (manual + agent reading, one reader per group) |
| False terminate 1,171/8,444 on a test set with no terminal steps | AC-High per-step file | measured |
| Zero-shot pattern probes (M1 re-rank worse; M3 AUROC 0.76 on 26 negatives) | `2026-11-12-patterns/*.json` | measured, small n for M3 |
| Untrained d1-2B base on validation / AC-High / AC-Low | `base_evals` (running) | **pending** |
| v4 + harness V14 on the 39 affected tasks | `aw116_v4t` runs | **pending** |
| Retrained v6 model | training on the second host | **pending** |

## Verification log (to run before delivery)
- Recompute every number in the draft from its artifact (independent verifier agent).
- Sweep for claims stronger than their support; mark pending items as placeholders.
- No cross-harness comparisons without a "not comparable" note (AGENTS M4).
