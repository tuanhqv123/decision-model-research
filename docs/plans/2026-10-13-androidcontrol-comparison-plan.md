# AndroidControl Comparison Plan — Us vs the Old

*Created 2026-10-13. Status: ACTIVE. Owner: main agent.
Purpose: make our harness **better than the old** on one specific public
benchmark — AndroidControl — and **prove the value** of each component.*

North star: `notes/GOAL.md`. Pass conditions below are fixed; changes need a
decision-log entry in GOAL.md §8.

---

## 1. The benchmark — ground truth

| Item | Fact |
|---|---|
| Dataset | AndroidControl (Google Research), mirror `smolagents/android-control`, Apache-2.0 |
| Scale | 15,283 human demonstrations; 833 apps; 40 app categories; **train 12,232 / test 3,051** steps |
| Fields | `goal`, `step_instructions[]`, `actions[] {action_type, app_name, direction, text, x, y}`, `screenshots_b64` (we do **not** use images — Mac vision constraint, stated openly) |
| Official paper | "On the Effects of Data Scale on UI Control Agents" (arXiv:2406.03679, NeurIPS 2024 D&B) [androidcontrol2024] — key finding: **fine-tuning on this data is what makes small UI agents work; performance scales with data** (our V4 ablation mirrors this theme) |
| Official metrics | Pass@1 **Step Accuracy** (exact action), **Action Type Accuracy**, **Grounding Accuracy** (coords/element), Pass@4 |
| Related "old" work | LiMAC — small on-device VLM + Action Transformer (ICLR 2025) [limac2025]; B-MoCA [bmoca2024]; AndroidControl-Curated (arXiv:2510.18488, "best models capped at ~60%") [androidcontrolcurated2025]; AndroidWorld (interactive, needs emulator+vision) [androidworld2024] |

### The "old" scoreboard (context — their protocol is harder: screenshot + a11y tree + open action generation)
AndroidControl High v2 leaderboard, May 2026 [sota2androidcontrol]:

| Method | Pass@1 Step | Type | Grounding |
|---|---|---|---|
| Mem-W-8B (UI-Venus-1.5-8B + memory) | **68.59** | 81.16 | 71.3 |
| GUI-Libra-8B | 64.3 | — | — |
| Mem-W-4B (Qwen3-VL-4B + memory) | 63.07 | 74.62 | 70.85 |
| GPT-5 + UGround | 61.3 | — | — |
| UI-Venus-1.5-8B | 61.06 | 74.62 | 61.06 |
| Qwen3-VL-32B | 59.05 | 74.62 | 69.96 |
| GPT-4o + UGround | 57.0 | — | — |
| Qwen3-VL-8B | 54.8 | 69.35 | 67.71 |
| Qwen3-VL-4B | 49.3 | 63.57 | 69.06 |
| Qwen3-VL-2B | 45.48 | 64.07 | 59.64 |
| GUI-R1-3B | 40.0 | — | — |

We **do not** claim to beat these head-on (different, harder protocol: they
see the screen and generate coordinates openly). They are the context that
shows the benchmark is hard.

## 2. Our protocol (fixed)

- **Input:** `goal` + `step_instruction` (text only).
- **Task:** K=4 candidates (the correct action + 3 real actions from the same
  split, similar step position) → select the correct one, **one forward pass,
  no text generation**.
- **Splits:** train split → training + 10% calibration slice. Test split →
  evaluation only, never seen by training/selection.
- **Metrics (fixed):**
  - **M1_sel** — correct candidate selected (full action tuple match).
  - **M1_type** — selected candidate has correct `action_type` (mirrors official Type Accuracy; always reported alongside M1_sel).
  - **M2** — coverage–risk from calibrated max-probability; headline operating point: **false-action ≤ 10%** (success among executed actions ≥ 90%) with **coverage ≥ 50%**.
  - **M3** — decision latency p50/p95 (M1 Pro, 4-bit), model size (GB), ask rate.
  - Calibration: ECE (15 bins, matching decider's convention) + Brier + reliability diagram.

## 3. The comparison matrix (identical protocol = fair)

| Row | Method | M1_sel | M1_type | M2 @ risk≤10% | M3 |
|---|---|---|---|---|---|
| B0 | chance (4 candidates) | 25.0% | — | — | — |
| B1 | frozen LFM2.5-1.2B (done) | 57.3% | — | ECE 0.100 | ~18 ms |
| B2 | B1 + post-hoc temperature T=1.75 (done) | 57.3% | — | ECE 0.055 | ~18 ms |
| **B3** | **THE OLD: Qwen3-14B Q4, zero-shot** (llama-server, same candidates) | to run | to run | logprob-threshold sweep | to run |
| B4 | (stretch) THX-01 / decider-2b on same candidates | — | — | — | — |
| H1 | trained head, CE-only (full 12k train) | | | | |
| H2 | trained head, CE + proper-scoring (full 12k train) | | | | |
| **H3** | **H2 + act/ask/refuse gate (headline row)** | | | | |

## 4. "Better than the old" — pass conditions (fixed)

- **P1:** H3 ≥ B3 on M1_sel — a trained 1.2B/2.6B head beats or ties the 14B LLM.
- **P1' (honest fallback if P1 fails):** H3 within 3 points of B3 on M1_sel **and** wins P2 and P3 → headline becomes risk + cost.
- **P2:** at risk ≤ 10% (executed-success ≥ 90%), H3 coverage ≥ 50%; B3 at the same coverage has risk ≥ H3's (if B3 can't hold a risk budget natively, report its best sweep point and say so).
- **P3:** H3 p50 latency ≤ 100 ms and ≥ 10× faster than B3 on M1 Pro.
- **P4:** one command regenerates every number (C3 reproducibility).

## 5. Prove the value (ablations — "what is the value of what")

| Value | Comparison | Proves |
|---|---|---|
| V1: small head ≈ big LLM, 10× cheaper | H3 vs B3 (M1, M2, M3) | trained small head matches zero-shot 14B at a fraction of cost |
| V2: calibration-aware training | H2 vs H1 (ECE, risk point, hard split) | proper-scoring preserves honesty (Q2 mechanism result) |
| V3: the gate | H3 vs H2 (executed-success, false-action rate, ask rate) | act/ask/refuse converts honesty into risk control |
| V4: data scale | 2k vs 12k train rows; 1.2B vs 2.6B | scale curve (mirrors the official paper's theme) |
| V5: train-time beats post-hoc | B2 vs H2 (same model family, ECE + risk point) | the core C1 claim: calibration is better *built in* than patched on |

## 6. Data work (raw only, Apache-2.0)

- test: 3,051 rows, text columns (proven pyarrow path; ~2.5 MB)
- train: 12,232 episodes, text columns (~10 MB)
- distractors: real actions **from the same split only** (never from test).
- no fabricated rows, no synthetic augmentation.

## 7. Execution order

1. **Full text download** (train + test) → `benchmarks/data/`. *(todo #21)*
2. **B3 run**: Qwen3-14B Q4 via `llama-server`, same candidates, full test → the "old" numbers + latency. *(todo #22)*
3. **H1/H2 training**: 1.2B first, then 2.6B; objectives CE / CE+proper / CE+joint; background on Mac. *(todo #14)*
4. **Gate + risk sweep** → H3 operating point at risk ≤ 10%. *(todo #15)*
5. **Ablations V1–V5.** *(todo #23)*
6. **Final table + reliability figures** → `benchmarks/results/` → paper. *(todos #16, #17)*

## 8. Risks

1. 1.2B may not reach 14B on M1_sel → 2.6B + full 12k data is the answer (official paper's data-scale finding supports this); if still short → P1' fallback (risk/cost headline remains valid).
2. Protocol difference vs VLM leaderboard → cited as context only, never as a direct comparison.
3. Coordinate matching inside candidate tuples → exact tuple match for M1_sel; M1_type reported separately; tolerance analysis only if needed for the paper.
4. LFM Open License v1.0 → read full terms before releasing fine-tuned weights.