# 2026-11-07, end of day — what worked, what died, and what to run next

> **On the string "64.32" in this file.** It is the **858 px** configuration's step accuracy, a *different*
> measurement from E2's. It coincidentally equals the retracted pre-audit E2 figure (64.32 = 256/398), which
> exists in no artifact — the E2 headline is **64.07% (255/398)**. Do not read the two as the same number.


One file, five configurations, one scorer, 398 rows. Read this instead of the twelve notes.

## The five arms, all measured on the same 398 eval rows

| config | step acc | type (223) | **grounding (223)** | gr\|type | p50 |
|---|---|---|---|---|---|
| d1-2B **base** (no fine-tune) | 17.4% | 63.2% | 14.4% | 22.7% | — |
| E2 — 15k rows, 720 px | 64.07% | 87.9% | **56.1%** | 63.8% | 106.8 ms |
| + `aux7k` (grounding-framed) | 64.57% | 87.0% | 57.4% | 66.0% | 104.6 ms |
| + 858 px | 64.32% \* | 86.1% | 58.7% | 68.2% | 154.4 ms |
| **+ duplicate to 22.5k rows** | **66.33%** | 85.7% | **60.1%** | **70.2%** | **88.2 ms** |
| `aux7k` + 858 px (stacked) | 65.58% | 87.0% | 57.4% | 66.0% | 154.5 ms |

\* **This 64.32 is the 858 px configuration's step accuracy — a different measurement.** It is not the
retracted pre-audit E2 figure (also written 64.32, i.e. 256/398, which exists in no artifact). E2 is
**64.07% (255/398)**.


## ⚠️ READ THIS FIRST — the deltas may all be seed noise

Two runs that differ by **8 samples out of 15,008 (0.05%)** — E2 and the "duplication control" — differ by
**+4.0 grounding**. The trainer never completes a pass (`epoch: 0` in every log), so a longer list only changes
the **shuffle**, and a different shuffle is a different seed.

**The seed spread of this pipeline has never been measured.** Two runs (`seed1`, seed 2) are going now, and the
pre-committed rule (`PRE-REG-2026-11-07-seed-spread.md`) is: if the seed standard deviation in grounding is
**≥ 2.0 points**, then **every grounding delta in this file is withdrawn** and only the loss-gradient arithmetic
and the base-vs-fine-tune gap survive.

Treat the table below as a record of what was measured, not as claims, until that spread exists.

## The winner, and it is embarrassingly simple

```
compact_cur15k.jsonl  →  shuffle (seed 0)  →  duplicate to 22,500 rows
init LiquidAI/d1-2B, --track V, 938 steps, lr 1e-5, 720 px
```
**66.33% step accuracy (+2.26 over E2), 60.1% grounding (+4.0), 88.2 ms p50.**

No new annotation. No new screenshots. No architecture change. **No latency cost.**

## The control that killed the attractive story

The tempting narrative was "re-framing existing labels into grounding tasks is a free lever" (`aux7k`, +2.0
grounding). The pre-registered control — the **same 15k rows duplicated** to the same 22,500 total, no framing —
reached **60.1% grounding vs `aux7k`'s 57.4%**. So:

> **More rows beats better framing. `aux7k` is an inferior way to add training rows, not a clever supervision
> trick.**

And stacking the two weaker levers **does not work**: `aux7k` + 858 px = 57.4%, below 858 px alone.

## What the mechanism is — corrected, because my first explanation was wrong

The `target_box` loss is four per-edge cross-entropies against a Gaussian (σ=4 bins on 0–999). Its cross-entropy
is a **numerically identical constant (690.78 nats) for offsets ≥200 bins** — the gradient is **exactly zero**
beyond ~400 px, which is where **86.8% of our misses** live (median 437 px). That part stands, and it is why
placement, not verb choice, is the thing to fix.

**But my first explanation of the control's win — "more exposures" — is wrong.** The trainer reshuffles only when
its cursor passes the end of the list (`train_d1.py:452-454`), so with a **fixed step count** a longer list means
*fewer* reshuffles, not more:

| arm | samples per run (938 × 16) | rows in list | epochs over it | reshuffles |
|---|---|---|---|---|
| E2 (15k) | 15,008 | 15,000 | **1.00** | ~1 (at the very end) |
| control (22,500 duplicated) | 15,008 | 22,500 | **0.67** | **0** |

So the control did **not** see more data — it saw **each row at most once, and only 67% of the list**, in a
different order, with the duplicated copies mostly unseen. The real difference between E2 and the control is
therefore **what the model saw last** (E2 wraps around and revisits early rows; the control runs a single clean
pass), not exposure count.

**That reframes the finding honestly:** the +4.0 grounding from duplication is an effect of **iteration order and
subset**, not of training volume. It is still a real, free, reproducible-looking gain — but the sentence
"more rows beats better framing" is not yet supported, because the control did not actually train on more rows.
The clean test of that sentence is the run described below.

## Dead ends, closed with measurements (do not retry)

| attempted | result |
|---|---|
| candidate selection over a long list | grounding **−10.17**, accuracy **−7.29** (p=0.0013) |
| two-stage shortlist | needs a **≥71.4%** selector just to break even; proxy did not replicate on test |
| ensemble / routing / fusion | best free arbiter **+0.60 (p=0.68)**; the +8.10 union was an **oracle** |
| `p(status)` gate | monotone-harmful at every threshold |
| class re-weighting (status) | −27 premature terminations bought −26 correct ones |
| loss post-processing at all | three attempts, all null |

## The open item that decides the paper

**Seed 2 of the duplication control is running** (`arm_ctrl_s2`, 360/938). If it reproduces +4.0 grounding, the
recipe is replicated and can enter the paper with a second seed. If it does not, the +4.0 is one draw and the
honest claim stays at "duplication helped once".

The second thing worth running is now sharper, because of the correction above: **the same 22,500-row list for
enough steps to complete one full epoch** (≈1,400 steps), so the control's advantage cannot be an artefact of
stopping at 67% of the list. If that increases the gain further, it is a volume/shuffle effect; if it disappears,
the gain was ordering. Either answer is publishable and neither is guessable.

## The calibration cost of the better recipe (checked, and it is a real cost)

The programme's core contribution is calibration, not accuracy, so the winner was checked on it:

| config | accuracy | **ECE** | AUROC(conf, correct) | mean confidence |
|---|---|---|---|---|
| E2 (15k) | 64.07% | **0.2013** | 0.6419 | 0.842 |
| dup 22.5k (control) | **66.33%** | **0.2289** | **0.6532** | 0.890 |

**The better recipe is +2.3 accuracy, +0.011 AUROC, and +0.028 worse ECE, with mean confidence rising 0.842 → 0.890.**

That is the classic trade this programme has documented all along: **more training makes the model more accurate
and more overconfident at the same time.** Ranking improves (AUROC up) but the absolute probability drifts further
from the truth (ECE up). For an act/ask/refuse gate the *ranking* is what drives the threshold, so this is not a
regression — but a paper that reports the +4.0 grounding without the +0.028 ECE would be hiding the cost.

**Required next step before any claim:** re-fit the type-routed gate on the new arm (`gate_routed.py`) and report
AUROC and risk-at-coverage for it, so the recipe is judged on both axes it is supposed to serve.

## What is NOT claimable from today

- Any on-device or online number.
- Any accuracy win over the published board (64.07 vs GUI-Libra 64.3 is a tie; the duplication arm's 66.33 is
  better than that but is **single-seed** and on a benchmark whose cross-paper numbers differ by 5 points).
- Any calibration improvement from the grounding work (calibration was not re-measured on the new arms — that is
  the next thing to check, and it is cheap: `gate_routed.py` on the new jsonl).
