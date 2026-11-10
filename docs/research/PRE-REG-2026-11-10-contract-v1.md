# PRE-REGISTRATION 2026-11-10 — AndroidWorld baseline B0 and contract-v1 retrain R1

*Written before any result of B0 or R1 exists. Evidence for every premise:
`docs/research/2026-11-10-AUDIT-androidworld-contract.md` (§N) and
`notes/decision-2026-11-10-androidworld-primary.md` (K1–K5).*

## 1. The two runs, and the one question each answers

| Run | What | Question it answers |
|---|---|---|
| **B0** | `ckpt-E2-final`, contract `e2` (8 actions, 5 questions), track V, served exactly as trained; harness on branch `aw-official-harness` (H1–H5 fixed); official seed-30 instances; all 116 tasks | *What does our clean checkpoint score on the official AndroidWorld protocol?* A measurement, not a test. |
| **R1** | E2's recipe from `LiquidAI/d1-2B` (938 steps, effective batch 16, lr 1e-5, warmup 30, frozen embedding, track V, `cur15k`) with ONLY the contract changed: history = gold actions rendered by `render_history_line` (the serve renderer), no app memory, no `is_done`, 9 actions (`navigate_home` added). Seeds 0, 1, 2. | *Does training on the history the agent will actually see change official AndroidWorld success?* |

## 2. Why B0 is E2 and not `appint_full`

`appint_full` was trained on 1,380 rows whose goals are AndroidWorld's own task goals
(`source: aw_registry_app_intent`, e.g. "Turn wifi on."). Its AndroidWorld score would be measured on goals
it was trained on. E2 saw no AndroidWorld data.

## 3. Assumptions — each stated, with how it is checked or why it cannot be

| # | Assumption | Status |
|---|---|---|
| A1 | The harness reproduces upstream `suite_utils` | Checked line by line (audit §8, H1–H9); each fix pinned by a test citing the upstream line. |
| A2 | The tasks run under the `android_world` copy whose source we read (`55ddaf5`) | **UNCHECKED** — the image holds two copies (site-packages and `/app`). Must print `android_world.__file__` and its version inside the container before chunk 1, and record it with the result. |
| A3 | B0 and R1 see identical task instances | True by construction after H2 (sha256 seed), **if** the same image and copy are used for both. Recorded per episode (`instance_params`). |
| A4 | R1's training history matches serve history | **Format**: identical by construction (one renderer, tested). **Distribution**: NOT identical — training history is gold (always correct), serve history contains the model's own mistakes (exposure bias). R1 removes the format/source mismatch, not exposure bias. |
| A5 | R1 differs from E2 only in the contract | Not exactly: also micro 1 × accum 16 instead of 2 × 8 (same effective batch, same mean gradient), app memory OFF (E2 trained with it ON, evaluated OFF), and 35 of 15,000 rows dropped because an earlier step of their episode is missing. All listed; none is believed to dominate, but none is measured. |
| A6 | E2 under contract `e2` is served exactly as trained | Questions/actions: yes (`contract.json`, tested). History: **no** — E2 was trained on human step instructions and is served executed-action history. That mismatch is part of what B0 measures, deliberately. |
| A7 | A task that raised is excluded from the denominator | This is upstream's rule (`NaN` skipped by pandas `mean()`). Risk: exclusion can differ between B0 and R1 (device degradation). Mitigation: also report SR on the tasks scored in BOTH runs. |
| A8 | The validation number of R1 is comparable to E2's 62.76% | **FALSE** — different history input. R1's `validation_v1` accuracy is a sanity check only, never compared to E2's validation number. |

## 4. Outcomes, decision rule and kill conditions (fixed now)

**Primary outcome:** official SR = mean over tasks of per-task float score, raised tasks excluded
(`awenv.metrics.official_sr`). Reported with the excluded list, the TC / answer-only split (25 answer-only tasks
score 0 by construction for both), and a Wilson interval on the number of fully successful tasks.

**Paired comparison R1 vs B0** (same 116 instances): per seed, count tasks where R1 succeeds and B0 fails (*w*)
and the reverse (*l*), on tasks scored in both runs.
- **Claim "R1 changes SR"** only if, pooled over the three seeds, an exact two-sided sign test on *w* vs *l*
  gives p < 0.05 **and** all three seeds' official SR move in the same direction.
- Otherwise report **"no measurable difference at n = 116"** with the counts. Not "no effect".
- Expected scale (stated so the result can surprise us): B0 official SR between 5% and 20% (the unseeded
  2026-11-08 run with a contaminated checkpoint gave 15.0% on 80 scored tasks). A difference under ~10 points is
  below what 116 tasks can resolve.

**Sanity / kill conditions for R1 (offline, before any AndroidWorld time is spent on it):**
- `validation_v1` step accuracy must exceed the always-`click` baseline on the same rows by ≥ 10 points;
  otherwise the run is broken and is not taken online.
- The three seeds' `validation_v1` step accuracies are reported with their spread; if sd ≥ 2.0 points the
  offline number is not used for any ranking (AGENTS M6).
- `contract.json` must read `v1`, tracks `["V"]`, before the checkpoint is served.

## 5. Explicitly NOT tested here

VA / element-index actions, the `answer` action, in-domain AndroidWorld-app training data, app-intent
scaffolding, exposure-bias fixes. Each is a separate pre-registration if pursued.

## 6. Amendment A1 — 2026-11-10 14:30 UTC, before ANY R1 online result exists

Written after reading the evidence in `outputs/aw-state-review.md` (C2: published limits of one-pass decision
models on state tracking, multi-step text and record-vs-observation; R1 targets none of them) and before any R1
checkpoint has been run on AndroidWorld. At the time of writing B0 had 15/116 episodes and no R1 online episode
existed; R1 seed 0 was at training step 480.

- R1 **seed 0** is run online on all 116 tasks, paired with B0.
- R1 seeds 1 and 2 are run online **only if** seed 0's paired difference vs B0 is at least 6 tasks in either
  direction (|w − l| ≥ 6 on tasks scored in both). Otherwise the result is reported as "no measurable difference
  at n = 116 for one seed", the emulator time is spent on the next pre-registered question, and seeds 1–2 are
  still trained and reported offline (`validation_v1`).
- Reason: compute value gate (AGENTS §7 "proof over prose"). A paired difference under 6 tasks cannot reach the
  pre-registered claim threshold with two more seeds of similar size, so their online runs would not change a decision.

## 7. Amendment A2 — 2026-11-10 14:55 UTC (B0 at 48 scored episodes; no R1 online result exists)

**(1) Baseline relabelled.** B0 as launched shows the pointer-location overlay that every official agent hides
(H10, `hide_automation_ui`). It is kept as arm **B0-overlay**. The official-protocol baseline is **B0′**: same
checkpoint and contract, overlay hidden, same instances. R1 is compared with B0′, never with B0-overlay.

**(2) Mechanism metrics for R1 vs B0′ (step-level, primary).** Computed by the frozen scripts in
`benchmarks/results/2026-11-10-b0-diagnosis/`:
- no-effect rate (share of executed steps after which the model saw the same screen);
- fixed-point rate P(same action | same screen);
- typing rate (share of decisions that are `type_text`) and click→type transitions.
Predictions, written before the run: if out-of-distribution history is a cause, R1 lowers the fixed-point rate and
raises the typing rate; if they are unchanged (within 5 points / 1 point), the history contract is not the cause and
that is the reported result. The SR rule of §4 and amendment A1 stand unchanged.

**(3) Out-of-sample confirmation of the diagnosis.** The diagnosis (PLAN-2026-11-10 §1, F2–F6) was read from the
tasks listed in `inspected_tasks.txt`. On the remaining tasks of B0-overlay — not yet looked at — the same scripts
must show: no-effect rate ≥ 40%; P(same action | same screen) ≥ 80%; AUROC(action confidence → effect) < 0.65;
typing ≤ 5% of decisions. Any threshold missed is reported as a failed confirmation of that finding.

## 8. Amendment A2-bis — 2026-11-10 15:20 UTC (no confirmation-set trace has been opened; no R1 online result exists)

A2's thresholds were written on pooled step counts. Re-analysis of the *inspected* tasks showed pooled counts are
dominated by a few long episodes (three episodes carry 50% of the no-effect steps), so the confirmation is restated
on episode-clustered statistics, which are **primary**; A2's pooled thresholds are reported as secondary.

On the confirmation set (tasks not in `inspected_tasks.txt`) the frozen scripts must show:
- per-episode median no-effect rate ≥ 35%;
- first-encounter repeat rate (the first decision on a screen the last action left unchanged repeats that action) ≥ 50%;
- AUROC(action confidence → effect) < 0.65, reported with its episode-bootstrap interval;
- typing ≤ 5% of decisions.
Inspected-set values for reference: 47.2%; 68.2% (CI 55–80); 0.591 (CI 0.48–0.69); 2.0%.
The same clustered statistics are the mechanism metrics for R1 vs B0′ (A2 item 2).
