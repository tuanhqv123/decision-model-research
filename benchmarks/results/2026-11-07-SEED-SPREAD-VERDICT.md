# The seed spread verdict — everything grounding-related from today is withdrawn

*Pre-registered in `PRE-REG-2026-11-07-seed-spread.md` before the runs. Rule: if the seed standard deviation in
grounding is ≥ 2.0 points, every grounding delta measured today is withdrawn.*

## The measurement

Three runs, same 398 eval rows, same official scorer. **`ctrl seed 0` and `ctrl seed 2` are the identical recipe
and differ only in `--seed`.**

| run | step acc | grounding (223) | grounded\|type |
|---|---|---|---|
| E2 — seed 0, `cur15k` | 64.07% | **56.05%** (125) | 63.8% |
| ctrl — seed 0, `cur15k`×1.5 | 66.33% | **60.09%** (134) | 70.2% |
| ctrl — seed 2, **same recipe as previous row** | 64.07% | **59.19%** (132) | 70.6% |

**Grounding across the three: 56.05, 60.09, 59.19. Range 4.04 points. Standard deviation 2.12 points.**

## The verdict, per the pre-committed rule

**sd = 2.12 ≥ 2.0 → WITHDRAW.**

Withdrawn, as of tonight:

| claim | measured | why it dies |
|---|---|---|
| "duplication to 22.5k gives +4.0 grounding" | 60.09 vs 56.05 | the same recipe's two seeds span 59.19–60.09, and E2 alone is 56.05 — a 4-pt range from seed alone |
| "`aux7k` gives +2.0 grounding" | 57.4 vs 56.1 | inside 2σ of the seed spread |
| "858 px gives +2.79 grounding" | 58.7 vs 56.1 | inside 2σ |
| "`aux7k` replicates across seeds (+2.23, +1.68)" | — | the two "seeds" shared the same 15,008-sample budget; the control's two seeds span 0.90 pts on the *identical* recipe, which is the right comparator, and it is comparable in size to the effect |
| "the recipe is decomposed into +2/+3/+1" | — | arithmetic on deltas that are inside noise |

**What the three runs DO show, and it survives:** `grounded|type` is **63.8 / 70.2 / 70.6** — the two control seeds
agree at 70.2 and 70.6 while E2 is 8 points below both. That is the one comparison where the two independent seeds
of the same recipe land *together*, and they are both far from the baseline. It is suggestive that the duplicated
recipe genuinely improves **placement**, but with n = 1 baseline seed it is not yet a measured effect either.

## What survives, and why it is not affected

| finding | why it stands |
|---|---|
| **The box loss has zero gradient beyond ~400 bins** (CE constant 690.78 nats) | arithmetic on the loss definition; no training run involved |
| **Base d1-2B vs fine-tuned: 17.4% → 64.07% accuracy, 14.4% → 56.1% grounding** | a 40-point gap, twenty times the seed spread |
| **Candidate lists above 40 collapse grounding** (−17.3 val, −23.8 test) | replicates on both splits at 8–12σ |
| **The fine-tune's grounding gain is 2/3 placement** (base grounds 22.7% given the right verb; ours 63.8%) | derived from the base measurement, not a delta between arms |
| **The decision-level cost window** (gate beats always-act/always-ask at `C_err/C_ask` 2.11–5.79) | computed from saved risk/coverage artifacts, unaffected by which checkpoint produced them |

## What this costs the programme, stated plainly

A full day of arm training produced **no claimable accuracy or grounding improvement**, because the pipeline's seed
spread (±2, range 4) is as large as every effect it was chasing. The correct response is not another arm — it is
either

1. **train 3+ seeds per configuration and report means with intervals** (≈5 h per config), or
2. **stop chasing ±2-point deltas on 398 rows** and go back to the judgements that are far outside noise: the
   loss-gradient mechanism, the base-vs-fine-tune gap, and the calibration/cost-window result.

This is the honest end state of the day, and it is worth more than the nine retracted numbers: **the reason this
programme keeps producing results that do not replicate is now measured.**


---

## UPDATE — the fourth run lands, and the picture is finer than "withdraw everything"

`seed1` finished. Four runs, same 398 rows, official scorer:

| run | recipe | step acc | grounding (223) | grounded\|type |
|---|---|---|---|---|
| E2 | 15k, seed 0 | 64.07% | **56.05%** | 63.8% |
| `seed1` | 15k, seed 1 | 63.57% | **58.30%** | 66.3% |
| ctrl | 22.5k, seed 0 | 66.33% | **60.09%** | 70.2% |
| ctrl_s2 | 22.5k, seed 2 | 64.07% | **59.19%** | 70.6% |

### The right comparison is WITHIN a recipe, not across all four

| grouping | grounding | spread |
|---|---|---|
| **same recipe, 15k** (2 seeds) | 56.05, 58.30 | **2.25** |
| **same recipe, 22.5k** (2 seeds) | 60.09, 59.19 | **0.90** |
| **between recipes** (mean of 2 vs mean of 2) | 57.17 vs **59.64** | **+2.47** |

Pooled within-recipe **sd = 1.73**. Two-sample t on the recipe difference = **1.42** (df = 2, critical 4.30) →
**not significant**.

### So the honest verdict is narrower than "everything is noise"

1. **Grounding is genuinely noisy run-to-run**: the same 15k recipe produced 56.05 and 58.30 — **2.25 points
   apart**. Any single-seed comparison of ±2 points on this benchmark is uninterpretable. That stands.
2. **The 22.5k recipe is ahead by +2.47 on the mean of two seeds each**, and its own two seeds agree much more
   closely (0.90) than the 15k pair (2.25). That is *consistent* with a real effect, but **t = 1.42 with n = 2 is
   not evidence**. It is a hypothesis worth 2–3 more seeds, not a result.
3. **`grounded|type`** separates the recipes cleanly: 63.8 / 66.3 (15k) vs **70.2 / 70.6** (22.5k). Both 22.5k
   seeds land within 0.4 of each other and 4–7 points above both 15k seeds. **This is the most promising signal in
   the dataset** — the placement measure is more stable than the overall grounding number, and it separates the
   recipes consistently.
4. **Accuracy** is also noisy: 64.07 / 63.57 (15k) vs 66.33 / 64.07 (22.5k) — range 2.76, sd 1.24.

### Revised action

Not "withdraw and stop", and not "ship the new recipe". The measured position is:

> The pipeline's within-recipe grounding spread is **~2 points** (sd 1.73). The 22.5k-vs-15k difference is
> **+2.47 on 2+2 seeds, t = 1.42** — inside what noise can produce at this n, but with a placement measure
> (`grounded|type`, 70.4 vs 65.0 mean) that separates the recipes consistently across seeds. **Decide it with 3
> more seeds per recipe (~5 GPU-hours), not with another single arm.**
