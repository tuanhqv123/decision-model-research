# A5 Q4 — Statistical power at n=116 AndroidWorld tasks

*Answers question 4 of `docs/research/A5-brief.md`. Arithmetic script: `scripts/power_aw116.py`
(stdlib only, re-runnable). Every formula below is verified by an independent Monte-Carlo
simulation, not asserted.*

---

## The answer, before the arithmetic

**At 116 tasks, the smallest paired improvement this benchmark can certify at 80% power is roughly
14 percentage points. Restricted to the 80 tasks we could actually score, it is 17.** The
programme's open claims are **+4** (stage-2 status down-weighting) and **+6** (the SOTA margin).
**Neither is detectable at this n, and no number of seeds fixes that** — seeds reduce measurement
noise, but they do not add tasks, and task-sampling variance is the binding term.

This is not a pessimistic reading of the numbers. It is what the numbers are.

---

## §4.1 The two designs, computed exactly

`alpha = 0.05` two-sided, `power = 0.80`, baseline SR `p0 = 15/80 = 0.1875` (the measured
scorable success rate). `z_alpha/2 = 1.95996`, `z_beta = 0.84162`.

### Paired design (McNemar, same tasks, two checkpoints)

`psi` is the discordance rate — the share of tasks where the two checkpoints disagree. It is the
single most important unknown, and it is **not yet measured**: our two conditions in the 8-task run
gave 1/8 in both, which tells us almost nothing about psi.

| psi (discordance) | discordant tasks (of 116) | MDE at 80% power |
|---|---|---|
| 0.10 | ~12 | **8.1 pts** |
| 0.15 | ~17 | **10.0 pts** |
| 0.20 | ~23 | **11.5 pts** |
| 0.25 | ~29 | **12.9 pts** |
| 0.30 | ~35 | **14.1 pts** |
| 0.40 | ~46 | **16.3 pts** |

**Realistic psi for two checkpoints of the same base model is 0.20–0.30**, because our measured
per-step accuracy moved 60.80 ↔ 64.07 between arms — a few points, on tasks that are individually
bimodal. So plan on **11–14 points**.

### Tasks required for each claim

At `psi = 0.30`:

| Claim | Tasks needed | Possible at n=116? |
|---|---|---|
| +2 pts | 5,885 | no |
| +4 pts | 1,470 | no |
| +6 pts | 652 | no |
| +10 pts | 234 | no |
| +15 pts | 103 | **yes** |

At the optimistic `psi = 0.15`: +2 needs 2,941, +4 needs 734, +6 needs 325. Still no.

### Unpaired design (two independent task sets, 116 per arm)

**MDE = 16.0 points.** Worse, as expected — pairing is free variance reduction and we should always
take it.

### The denominator that actually applies

The SR claim is computed over **scorable** episodes, which was 80 of 116 in the 2026-11-09 run. The
checker failures are an infrastructure limit, not a model property, so they shrink the effective n
without shrinking the design:

| Effective n | MDE at psi=0.30 | MDE at psi=0.15 |
|---|---|---|
| 116 | 14.1 pts | 10.0 pts |
| 104 (finished) | 14.9 pts | 10.5 pts |
| 92 (attempted) | 15.8 pts | 11.2 pts |
| **80 (scorable — the one we report)** | **16.9 pts** | **12.0 pts** |

---

## §4.2 Verification of the formulas

Analytic power is easy to get subtly wrong, so both were checked against simulation (20,000 trials
each, independent RNG seed):

| Check | Analytic | Monte Carlo |
|---|---|---|
| McNemar power, delta=14 pts, psi=0.30, n=116 | ~0.80 | **0.773** |
| Two-proportion power, delta=16 pts, n=116/arm | ~0.80 | **0.795** |

Both land within Monte-Carlo error of the target. The small McNemar shortfall (0.773 vs 0.80) is the
normal approximation running slightly optimistic against an exact binomial test at these counts —
which makes the real MDE **slightly worse** than the table says, not better. Reported as measured.

---

## §4.3 What this forces

Three consequences, in order of how much they change the plan:

1. **A `+4` stage-2 result can be published, but not as a certified improvement at n=116.** It can be
   published as a point estimate with a wide interval, labelled suggestive, exactly as §5.9's
   duplication result was labelled after the seed-spread withdrawal. The programme has already been
   through this once (M6, M7); the rule now exists — no delta without ≥3 seeds **and** an interval.

2. **The accuracy axis is essentially closed for a two-checkpoint comparison.** To certify +6 points
   paired you need ~650 tasks. AndroidLab-138 does not help (138 < 650). Neither does 40 tasks × 3
   seeds: that is 40 tasks of *sampling* variance, and no count of seeds substitutes for task
   breadth. **Say plainly: no feasible run on one 4090 produces a publishable accuracy delta on
   this benchmark.** That is a property of the instrument, not of the model.

3. **The gate metric is a different instrument and it is not power-limited the same way.** Verified
   from our own artifacts (`benchmarks/results/2026-11-03-stage0/`): the gate is measured over
   **398 test rows / 827 val rows**, with **140 positives** (398 − 258 correct on `E2-938-V.jsonl`).
   That is 3.4× the units of the SR claim, and the estimand is *risk at fixed coverage* — a per-row
   quantity, not a binary task outcome. Measured values: routed AUROC **0.760 val / 0.729 test**,
   learned-fusion risk at 50% coverage **0.191**, AURC 0.201. **This is the strongest argument in the
   programme for making the gate the primary claim**: it is the one measurement the instrument can
   actually certify at these n.

   **But it is no longer a novelty claim.** T4 found CORA (arXiv 2604.09155) already publishes
   conformal risk-controlled execute/abstain on mobile GUI: τ̂=0.952 → 89.95% coverage, 2.42% executed
   harm, 10.05% invocation, on a 1,075-row calibration pool at δ=0.05. So the certificate is prior
   art. What remains ours is the **price of an ask** (T4 Q4: no fetched source prices a human ask;
   CORA prices only intervention latency, 34.13 s/step) and the AC-v2-specific measurement. That
   matches the 2026-11-07 reframe — *Priced Interruptions*, not ensemble accuracy — and it is now
   evidence-backed rather than a judgement call.

---

## §4.4 The gate's own MDE, computed the same way

If the gate becomes the primary claim, it must clear the same bar. Using the Hanley–McNeil standard
error for AUROC on the actual test set (`n1 = 140` positives, `n0 = 258` negatives, `A = 0.729`):

- **SE = 0.0276**, 95% CI = **[0.675, 0.783]**.
- **MDE on AUROC at 80% power = 0.077 (7.7 AUROC points).**

Three measured facts now read differently against that floor:

| Comparison | Measured delta | vs 7.7-point MDE |
|---|---|---|
| routed (0.729) vs global p×box (0.649) | **+0.080** | **just clear** — this is the real result |
| val 0.760 → test 0.729 (same method) | −0.031 | within noise; not a degradation |
| learned fusion vs routed (+0.013, CI [−0.020, 0.043]) | +0.013 | **well below floor** — correctly reported as not significant |

So the programme's one certifiable discrimination result is **type-routed beats a global score by
roughly one MDE** — real, marginal, and about as large as this instrument can resolve. The learned
fusion is not a result at this n and never was. This is consistent with what the record already says
(F6: "report the estimate + interval; claim nothing") and it explains why.

**Honesty note on that margin: it is 1.03× the MDE, not comfortably above it.** The measured delta is
0.0792 against a floor of 0.0772. A reviewer recomputing this will find a result that clears by
about 2%. So the correct sentence in the paper is *"routing improves discrimination by roughly one
standard error's worth of detectable effect"* — not *"routing significantly improves discrimination"*.
The paired episode-clustered interval is the honest instrument here, and it is the one already
recorded for the fusion (+0.013, CI [−0.020, 0.043]).

---

## What I could not verify

- **psi is unmeasured.** Every MDE above is conditional on an assumed discordance rate. The honest
  statement is the range 11–14 points, not a single number. It becomes a measurement the moment two
  checkpoints are run over the same 116 tasks.
- **Whether the scorable set is identical across checkpoints.** Pairing assumes the same tasks are
  scorable in both runs. If the emulator's accessibility forwarder degrades differently across runs,
  the paired denominator shrinks and the MDE grows. Not testable from the current artifacts, which
  hold one run.
- **Whether mAIR / any published AndroidWorld number controls for task-sampling variance at all.**
  That is T2's job (published-numbers corpus), not mine, and it does not change this arithmetic.
