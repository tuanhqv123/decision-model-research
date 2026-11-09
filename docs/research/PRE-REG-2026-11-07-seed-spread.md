# PRE-REGISTRATION — the seed spread of the baseline (2026-11-07)

*This is the measurement that decides whether every delta claimed today is real.*

## Why

Every arm today differs from E2 by +2 to +4 points of grounding. But the arms differ from each other by **a
different random shuffle of essentially the same data**: with `--steps 938` and batch 16 the trainer draws 15,008
samples and never completes a pass, and every run on disk logs `epoch: 0`. The "duplication control" and E2 differ
by **8 samples out of 15,008 (0.05%)** yet differ by +4.0 grounding.

**The seed spread of this pipeline has never been measured.** Every delta of this size is therefore currently
indistinguishable from noise.

## The run

E2's exact recipe, seeds 1 and 2 (E2 itself is seed 0):

```
data compact_cur15k.jsonl | init LiquidAI/d1-2B | track V | steps 938
lr 1e-5 | warmup 30 | micro 1 | accum 16 | grad-ckpt | freeze-embed | 720 px
--seed 1   and   --seed 2
```

Each is evaluated on the **same 398 rows** with the official scorer.

## What the number decides (pre-committed interpretation)

Let `s` = the standard deviation of grounding across seeds {0, 1, 2}.

| if | then |
|---|---|
| **`s ≥ 2.0 pts`** | **every grounding delta measured today is inside seed noise.** The day's grounding "findings" are withdrawn wholesale; only the loss-gradient arithmetic and the base-vs-fine-tune gap (both ≫ 2σ) survive. |
| `1.0 ≤ s < 2.0` | deltas of +2 are suggestive, +4 is real. Report as such with the seed band attached. |
| `s < 1.0` | the +2 to +4 deltas are real effects and the arms deserve follow-up. |

No bar is moved after seeing the result. If `s ≥ 2.0`, the paper says so plainly.

## Cost

Two runs, ~1.6 h each, both GPUs free. This takes priority over every other experiment, because no other
experiment's result can be interpreted until it exists.
