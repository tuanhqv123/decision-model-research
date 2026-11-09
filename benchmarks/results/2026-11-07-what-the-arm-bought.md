# 2026-11-07 — what the arm actually bought: two checkpoints that are good at different things

**Short answer to "is it better": no — not by the bar we set.** But the arm is not noise, and the reason it
failed is now measured rather than guessed.

## The numbers

| Split | E2 | arm | Δ | McNemar | cluster CI | verdict |
|---|---|---|---|---|---|---|
| val (827) | 62.76% | 63.85% | **+1.09** | p = 0.44 | [−1.48, +3.85] | fails +1.5 |
| test (398) | 64.07% | 60.80% | **−3.27** | p = 0.15 | [−7.34, +0.76] | fails, sign flips |

Per gold type on test:

| type | n | E2 | arm | Δ |
|---|---|---|---|---|
| click | 212 | 118 | **127** | **+9** |
| scroll | 37 | 18 | **22** | **+4** |
| status | 87 | 78 | **53** | **−25** |
| navigate_back | 20 | 13 | 12 | −1 |
| open_app | 27 | 18 | 18 | 0 |
| type_text | 11 | 7 | 7 | 0 |
| wait | 4 | 3 | 3 | 0 |

So the arm is a **specialist**: better at spatial actions (+13 rows), much worse at deciding the task is done
(−25 rows). The two checkpoints fail in opposite directions, which is the only genuinely new information here.

## Can that be exploited? Measured, and no — not yet

A router that takes the arm on spatial decisions and E2 on the rest:

| Rule | Test acc | Δ vs E2 | Note |
|---|---|---|---|
| E2 alone | 64.07% | — | baseline |
| arm alone | 60.80% | −3.27 | fails |
| **oracle router on GOLD type** | **66.33%** | **+2.26** | **upper bound only** — gold type is unavailable at test time |
| usable: route on **E2's predicted** type | 64.82% | +0.76 | +3 rows |
| usable: arm only when both agree on type | **65.08%** | **+1.01** | +4 rows; **McNemar p = 0.56**, CI [−1.51, +3.51] |

The gap between the oracle (+2.26) and the best *usable* rule (+1.01) is the interesting part: **the router's
own type decision is the bottleneck**, exactly the same failure the earlier audit found. And +1.01 does not clear
the pre-registered +1.5, with p = 0.56 on 398 rows.

**Verdict: a checkpoint ensemble is not a free win here.** The oracle bound says the ceiling is real (+2.26),
but the router cannot reach it because routing requires the very type prediction that is unreliable.

## What is genuinely new this session (beyond the negative)

1. **The mechanism of the failure, measured**: −27 premature terminations bought −26 correct ones (51→24
   premature, 120→94 true `status` on val). A class weight is one knob on `p(terminate)`.
2. **Grounding improved without being trained**: +2.67 pts val (285/422 → 304/433), +2.26 pts test
   (125/223 → 134/223). Fewer early exits mean more steps spent acting.
3. **An oracle ceiling for the whole line**: a *perfect* progress signal is worth only **+1.26 pts** here,
   because the model's own next-best action rescues just 5 of the 21 premature rows. So no amount of work on
   the status bucket alone can reach +1.5.
4. **The decision-level claim (independent of all this)**: the gate beats always-act and always-ask for
   `C_err/C_ask` in **2.11–5.79**, and 2.08–4.19 on the next checkpoint. That is still the strongest
   contribution in the programme, because it does not depend on beating the accuracy bar.

## What to do next, if continued

The oracle bound points at one thing: **the model needs a progress signal it does not have**, not a better
threshold or a better ensemble. The concrete next arm is an explicit `task_complete` readout feeding the type
decision. Everything else (routing, weighting, gating on `p(status)`) has now been tried and measured as
insufficient.

Artifacts: `benchmarks/results/2026-11-07-status-downweight-arm.md`,
`benchmarks/results/2026-11-07-arm/paired_arm_vs_E2_{val,test}.txt`, paper §5.10.
