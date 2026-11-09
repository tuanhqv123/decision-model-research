# The arithmetic that makes our 64% step accuracy worth ~1% per task

*2026-11-08. This is the finding that reframes the whole programme. It is not a retraction of a number;
it is the reason every previous "why is online SR so low" investigation was asking the wrong question.*

## The product rule

A multi-step task succeeds only if **every** step succeeds. With per-step accuracy `p` over `n` steps:

```
P(task) = p^n
```

| per-step p | 5 steps | 8 steps | 10 steps | 15 steps | 20 steps |
|---|---|---|---|---|---|
| 64.1% (our AC-v2) | 10.8% | **2.8%** | **1.2%** | 0.1% | 0.0% |
| 76% | 25.4% | 11.1% | 6.4% | 1.6% | 0.4% |
| 85% | 44.4% | 27.2% | 19.7% | 8.7% | 3.9% |
| 95% | 77.4% | 66.3% | 59.9% | 46.3% | 35.8% |
| 99% | 95.1% | 92.3% | 90.4% | 86.0% | 81.8% |

## What the state of the art implies about itself

The published AndroidWorld numbers are **end-to-end** success rates. Inverting the product rule gives the
per-step accuracy they must have:

| model | reported end-to-end | implied per-step (at 10 steps) |
|---|---|---|
| MAI-UI-2B | 49.1% | **93.1%** |
| MAI-UI-8B | 70.7% | **96.6%** |
| MAI-UI-235B-A22B | 76.7% | **97.4%** |
| M3A (AndroidWorld's own, GPT-4 Turbo + a11y tree) | 30.6% | 88.8% |

**Every competitive agent runs at 93–97% per-step accuracy.** That is the actual frontier, and it is not
the axis our paper has been optimising.

## The two conclusions, both uncomfortable

**1. Our own online number was never evidence of a harness defect.**

Our model is 64.07% per-step on AC-v2. The product rule predicts **2.8% over 8 steps** and **1.2% over
10**. We measured **1/8 = 12.5%** with a Wilson interval of **2.2–47.1%**. The measurement is *consistent*
with the model's step accuracy. The episode did not fail because `app_name` was empty; it would have
failed anyway with a perfect read slot, because 0.64⁸ ≈ 3%.

This reframes the `app_name` fix (2026-11-08): it is a real bug and a real fix (1/8 → 8/8 on the intent
probe), but it **cannot move task success much** while per-step accuracy stays at 64%. Fixing it raises
the ceiling from ~3% to maybe ~4%. That is the honest statement.

**2. A step-accuracy paper cannot claim task success, and a task-success paper needs a different model.**

* AC-v2 measures per-step accuracy. 64.07% is a real, reproducible number *on that axis*.
* AndroidWorld measures end-to-end. Competing there requires ~93% per-step.
* The gap between 64% and 93% is **29 points of per-step accuracy**, not a fix, not a patch, not more
  data on the same recipe. It is a different training regime.

## Why MAI-UI gets to 93% and we do not

From the MAI-UI report (arXiv:2512.22047), the mechanisms that plausibly buy the last 30 points:

* **Online RL on a live emulator**, GRPO, group size 16, 512 concurrent environments, 50-step horizons.
  Their own ablation: 32 → 512 parallel envs is **+5.2 points**; step budget 15 → 50 is **+4.3 points**.
* **Self-evolving data**: train → rollout → MLLM judge → rejection sample → retrain. Failed trajectories
  reused as their longest correct prefix.
* **Multi-step trajectory data** (GUI-Odyssey: 8,334 cross-app episodes) rather than single-step decisions.
* **12-action space** including `ask_user` and `mcp_call`, so the agent can stop and ask instead of
  guessing when uncertain.

We have: 15k single-step AC decisions, SFT only, one 4090, no emulator farm, 8 actions. The 29-point gap
is structural.

## What this means for the plan, stated plainly

* **Do not promise an AndroidWorld number competitive with 49.1%.** At 64% per-step the ceiling is a few
  percent end-to-end. Reporting 3% is honest; reporting it as a failure of engineering is not.
* **The defensible contribution is not accuracy.** It is: (a) the measured cost window for act/ask/refuse,
  (b) the observation that a *calibrated* gate can substitute for the 29 points of accuracy — if you cannot
  be right 93% of the time, you can at least know when you are wrong 93% of the time — and (c) the
  arithmetic above, which every GUI paper reports end-to-end but none reports as a step-accuracy budget.
* **`ask_user` is the honest research direction**, and MAI-UI already measured its value: **+32.1 absolute**
  on their interaction subset (37.5% vs 5.4%). Our gate is a `continuous` version of the same idea. The
  opening is that no GUI paper publishes an **outcome-level abstention result with a risk certificate**.

## The one-sentence version (Feynman test)

> A model that is right 64 times in 100 on a single screen is right about once in 30 on a ten-screen task,
> so 64% step accuracy and 3% task success are the same fact; the frontier's 76.7% on AndroidWorld means
> their per-step accuracy is 97%, and no amount of fixing one read slot closes that distance.

If that sentence is wrong, this note is wrong. I believe it is right, and it explains why the last two days
of searching for a bug kept coming up empty.
