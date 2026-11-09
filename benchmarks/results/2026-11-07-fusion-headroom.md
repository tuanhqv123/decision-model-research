# 2026-11-07 — checkpoint fusion for grounding: +5.8 pts of oracle headroom, +1.3 pts achievable

> **On the string "64.32" here.** This is the fusion router's test accuracy at the **τ = 60 px** point of
> the sweep below — a different quantity from E2's step accuracy. It coincidentally equals the retracted
> pre-audit E2 figure (64.32 = 256/398), which exists in no artifact; E2 is **64.07% (255/398)**.


**Question.** The stage-2 arm is a spatial specialist (click +9, scroll +4) and a termination disaster
(status −25). Can the two checkpoints be *combined* so the good parts are kept?

## The headroom is real and large

On the 223 gold-spatial test rows, measured with the official `grounding` flag:

| | grounding | note |
|---|---|---|
| E2 alone | 125/223 = **56.05%** | |
| arm alone | 134/223 = **60.09%** | better! |
| **at least one correct** | **147/223 = 65.92%** | **+5.83 pts over the better single model** |

Overlap: both 112 · only-E2 13 · only-arm 22 · neither 76. The two models fail on *different* elements, which
is the same structure AdaGround reports for two zoom chains (59.6% / 63.5% individually, 68.9% for the union
[adaground2026]). So the ceiling is not the issue.

## The disagreement is a nearly perfect trigger

Median distance between the two predicted points:

| condition | median ‖p_E2 − p_arm‖ | n |
|---|---|---|
| exactly one of them is right | **146.7 px** | 35 |
| anything else (both right, or both wrong) | **8.4 px** | 188 |

When the two checkpoints agree to within ~8 px, they are almost always both right or both wrong. When they are
146 px apart, one of them is right 33% of the time. A distance gate therefore isolates the useful cases
cleanly — at ≥40 px it triggers on 82 rows and captures 32% of them having exactly one right answer.

## And a free confidence signal picks the right one better than chance

On the 21 rows where the points are ≥60 px apart and exactly one is right, choosing by
`box_conf × found`:

| rule | picks correctly | vs chance |
|---|---|---|
| higher `box_conf × found` | **16/21 = 76%** | 50% |
| higher `box_conf` alone | 15/21 = 71% | 50% |
| higher action confidence | 11/21 = 52% | 50% |
| always-arm / always-E2 | 11/21 · 10/21 | 50% |

## But it does not survive an honest evaluation

The rule is: default to E2; if the two predicted points are ≥ τ px apart, take the one with the higher
`box_conf × found`. τ is fitted on the 827-row val split, then evaluated **once** on the 398 test rows.

| τ (px) | val acc | test acc |
|---|---|---|
| 0 | 65.18% | 65.58% |
| 10–20 | 65.05% | 66.08% |
| **40 (val-optimal)** | **65.30%** | **65.33%** |
| 60 | 65.18% | 64.32% \* |

\* **This 64.32 is the 858 px configuration's step accuracy — a different measurement.** It is not the
retracted pre-audit E2 figure (also written 64.32, i.e. 256/398, which exists in no artifact). E2 is
**64.07% (255/398)**.


At the val-fitted τ = 40: test **255 → 260/398 = 65.33%, Δ = +1.26 pts**, gained 15 / lost 10,
**McNemar p = 0.42**, episode-clustered CI **[−1.23, +3.73]**. Per type: click +11, scroll +1, status −7,
everything else 0.

**It fails the pre-registered bar (+1.5) and is not significant.** The lost rows are almost all `status`: the
router's confidence comparison is not meaningful on non-spatial rows, so it occasionally hands a termination to
the arm, which is the one thing the arm is worst at.

## A bug I made and caught

My first version of this analysis gated on `gold['action_type']` — the label. It reported **+3.52 pts, p =
0.0013, no type regressing**, which looked like a clean win. It was not: gold type is unavailable at test time,
so that rule is an oracle, not a method. The honest version, gating on the *predicted* type, gives +1.26 with
p = 0.42. The inflated number is recorded here so it cannot be re-discovered and believed later.

**The lesson generalises:** every router/oracle-vs-usable distinction in this programme has flattered the
method by exactly the amount that the routing decision is unreliable. The gap between the gold-type oracle
(+2.26 step accuracy earlier, +5.83 grounding here) and the usable rule (+1.26) *is* the routing error.

## What this licenses

1. **A distance-gated, confidence-arbitrated fusion is a real but small effect** (+1.26, not significant at
   n=398). It needs either more data or a better arbiter to be claimable.
2. **The arbiter is the bottleneck, not the headroom.** `box_conf × found` is 76% accurate on 21 rows; a
   learned arbiter with the features already saved (`box_conf`, `found`, the two points, their distance,
   entropy) is the obvious next step, and it is zero-GPU.
3. **The arm's spatial gain is genuine and free**: +5.83 pts of union headroom at no training cost, because the
   arm was already trained for a different reason.

Artifacts: this note; `benchmarks/results/2026-11-07-arm/`; [adaground2026] AdaGround,
arXiv 2610.36874 (two-chain portfolio + marked-click verifier, training-free).


## The learned arbiter: val +2.90, test +0.00 (overfit, and that is the answer)

`benchmarks/train/fusion_arbiter.py` fits a logistic arbiter on the features available at test time —
`conf`, `box_conf`, `found`, 8-way entropy, top-2 margin, `is_spatial`, each checkpoint's value **and their
difference**, plus the point distance — fitted on the 107 val rows where the two checkpoints disagree, then
scored once on test.

| | val (827) | test (398) |
|---|---|---|
| base (E2) | 62.76% | 64.07% |
| learned fusion | **65.66% (+2.90)** | **64.07% (+0.00)** |

Gained 14, lost 14, p = 1.00, CI [−2.74, +2.53], with **214 of 398 rows switched**. The val gain is real and
the test gain is exactly zero: 107 fit rows cannot support a 20-feature arbiter, so it memorised the val split
and transferred nothing.

A constrained variant (switch on disagreement using `box_conf × found`, and also trust the arm when the two
points agree within τ) reaches val 65.42% and **test 65.33% (+1.26)**, gained 21 / lost 16, p = 0.51,
CI [−1.55, +4.26]. Same ceiling, same non-significance.

**Conclusion: the +5.83 pts of oracle headroom is real, and every *learned or hand* arbiter tried here converts
at most ~1.3 of it, never significantly, on 398 rows.** The bottleneck is the arbiter, and the arbiter has 107
disagreement rows to learn from. That is a data problem, not a modelling one — which is worth knowing before
anyone spends GPU on a fancier combiner.
