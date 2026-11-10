# PRE-REGISTRATION — episodic corpus rebuild + `is_done` readout (2026-11-09)

*Frozen before the run. Do not move the bars after seeing results.*

## (a) The claim

**Episode-preserving sampling of the AndroidControl corpus, plus an `is_done` noul readout trained on
the free terminal-row label, raises `is_done` AUROC and `target_el` accuracy on held-out episodes
without lowering AC-v2 per-step accuracy.**

Three parts, each separately falsifiable:
1. `is_done` AUROC **conditioned on action ≠ terminate** exceeds 0.50 by more than the seed spread.
2. `target_el` accuracy on held-out episodes exceeds the `cur15k` baseline at the same candidate cap.
3. AC-v2 per-step accuracy (398 steps) does not fall below the 64.07% baseline by more than the seed spread.

## (b) The arms

Sampling is by episode; every taken episode keeps all its steps (`benchmarks/train/rebuild_ac_episodic.py`).
All arms hold out the same 137 validation episodes, by episode id.

| Arm | Rows budget | Episodes | Steps/ep (median) | Purpose |
|---|---|---|---|---|
| **A. baseline** | 15,000 (`cur15k`, existing) | 9,005 | **1** | the current model; median 1 step |
| **B. episodic medium** | 39,000 | ~6,000 | **6** | the primary arm |
| C. episodic full | 88,288 | 13,603 | 6 | reference only, run last |

`is_done` is trained in arms B and C only. Arm A cannot produce it (its rows have no usable
`step == n_steps` distribution — it keeps 1–2 steps of each episode, so the terminal row is frequently
absent).

## (c) Metrics and their units

| metric | unit | denominator | what it answers |
|---|---|---|---|
| `is_done` AUROC (**non-terminal rows only**) | **row** | ~39,000 held-out… no: **827 val rows** | can the model tell "not yet" from "done"? |
| `is_done` risk at 50% coverage | row | 827 val rows | is the confidence usable as a gate? |
| `target_el` accuracy | spatial row | val spatial rows | does it pick the right element? |
| AC-v2 per-step accuracy | **step** | **398 steps** | did the old capability regress? |

**Explicitly NOT a metric of this study: AndroidWorld-116 task success.** Reason: the minimum
detectable paired effect is ~14 percentage points at n=116 (17 at the 80 scorable episodes), and the
arms here predict +2 to +6 (`docs/research/A5-parts/Q4-power-analysis.md`). A task-success number from
this study would be noise read as signal. If one is reported at all it is descriptive, in a separate
table, labelled "not a test".

**Unit discipline:** no per-step number may be printed in the same column as a task-success number.

## (d) Seeds

`seed = 0, 1, 2` for every arm. Report the per-seed table, the mean, and the **sd** — never a single
seed. The withdrawal rule in (e) is applied to the sd, not to a confidence interval computed from one run.

## (e) Withdrawal rule (frozen)

**If the seed spread (sd across the three seeds) of any metric equals or exceeds the effect size on
that metric, the result is reported as INCONCLUSIVE and no claim is made.**

This rule was written because the programme has already withdrawn five grounding deltas after measuring
a seed sd of 2.12 points (`benchmarks/results/2026-11-07-SEED-SPREAD-VERDICT.md`). The same rule applies
here without amendment.

Secondary withdrawal: if `is_done` AUROC computed **including** terminal rows exceeds the non-terminal
AUROC by more than 0.30, the difference is attributed to the `action == terminate` shortcut and the
non-terminal number is the only one reported. (The redundancy is real and measured: `status` rows and
`terminal` rows are the identical 13,740-row set.)

## (f) What this study does not claim

- It does not claim AndroidWorld or AndroidLab task success improves. At n=116 / n=138 the instrument
  cannot resolve the effect sizes involved.
- It does not claim pointer-read can compose text. 2.38% of `type_text` rows need generation and are
  labelled `escalate`, not solved.
- It does not claim the 7.3% duplicate-label rows are solvable from text. They need the image.

## (g) Provenance of every number above

| Number | Script | Command |
|---|---|---|
| 89,144 rows / 13,740 episodes / 0 leakage | `scripts/audit_ac_raw.py` | `--stage 1` |
| 99.9% target-node IoU / 13,740 terminal `status` | `scripts/audit_ac_raw2.py` | `--stage 2` |
| 99.30% gold coverage, 37 median candidates | `scripts/audit_ac_raw3.py` + production `pick_candidates` | `--stage 3` |
| 96.7% / 2.38% text copyability | `scripts/audit_ac_raw.py` §4 table | see `DATA-SPEC-2026-11-09.md` §4 |
| 6.9-point AndroidWorld 3-seed spread | arXiv 2405.14573v5 §5 | `DATA-SPEC-2026-11-09.md` §9 |
| 14-point paired MDE at n=116 | `scripts/power_aw116.py` | — |
