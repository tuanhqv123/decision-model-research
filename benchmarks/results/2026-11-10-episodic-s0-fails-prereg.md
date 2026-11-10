# VERDICT — `episodic-s0` fails the pre-registered bars (2026-11-10)

*Every number here is read from an artifact on disk. Nothing is re-derived by hand.*

## TL;DR

`episodic-s0` **cannot be used for a benchmark number**, and the `is_done` gate built on it must be
removed rather than repaired. Two of the three pre-registered conditions fail; the third was never
measurable in this corpus.

The reason is a single false assumption, measured here for the first time: **`is_done` is not a
learnable question in AndroidControl.** It is a constant function of the action label.

## The measurement that ends the question

```
824 val rows
137 positives            (step == n_steps)
137 / 137 of them have  action.action_type == "status"
687 non-status rows  ->   0 positives
```

The label `step == n_steps` and the label `action == status` are the **same set**. There is no row
where the task is finished and the action is not `status`, and no row where `status` is emitted
before the end. So "has the task been completed on this screen?" is answered perfectly by "is this
action `status`?" — which the action head already predicts.

Consequence, from `runs/isdone.json` (produced 2026-11-09 22:58, unread until now):

| metric | value | meaning |
|---|---|---|
| `auroc` | **0.960** | the shortcut. Model learned "this is `status`". |
| `auroc_nonterm` | **null** | the honest number — **not computable**, 0 positives among 687 non-status rows |
| `risk` @ 50% coverage | **0.670** | at the confidence the gate uses, two thirds of selected rows are wrong |

`PRE-REG-2026-11-09-episodic-isdone.md` §Secondary withdrawal already specified this: if the
terminal-inclusive AUROC exceeds the non-terminal one by >0.30, the difference is the
`action == terminate` shortcut and *the non-terminal number is the only one reported*. The
non-terminal number does not exist. **The gate was switched on using the number the pre-registration
forbids.**

## The three pre-registered conditions

| # | Condition | Bar | Measured | Verdict |
|---|---|---|---|---|
| 1 | `is_done` AUROC, non-terminal rows only | > 0.50 + seed spread | **null** (0 positives / 687 rows) | **FAIL — not measurable** |
| 2 | `target_el` accuracy on held-out episodes | > `cur15k` baseline | **not measured** — `el_idx` present 0/398; the run is track V, which has no candidate list | **FAIL — never run** |
| 3 | AC-v2 per-step accuracy | ≥ 64.07% − seed spread | **55.78%** (222/398), **−8.29 pts** | **FAIL** |

Sources: `runs/isdone.json`, `runs/eval/eval-episodic-s0-V.summary.json`, artifact row counts above.

## Condition 3 in detail

`eval-episodic-s0-V.summary.json` (track V, 398 steps):

```
step_acc      0.5578      baseline 0.6407   delta -0.0829
type_acc      0.7764      baseline 0.8417   delta -0.0653
grounding     0.6000      baseline 0.5605   delta +0.0395
ece_action    0.2423
always_status 0.2186      (the degenerate "always emit status" baseline)
```

Per type: `click 108/212`, `status 64/87`, `open_app 18/27`, `navigate_back 11/20`, `scroll 12/37`,
`type_text 6/11`, `wait 3/4`.

Two things are worth separating, because they point in opposite directions:

- **Grounding improved** (+3.95 on all rows; on the 218 spatial rows it is 114/218 = 0.5229). The
  episodic corpus did what it was supposed to do for the spatial head.
- **`status` accuracy collapsed**: 64/87 = 0.736 vs the 76/87 = 0.874 of the baseline. The arm was
  trained with an `is_done` slot on a label that is a copy of `status`, so the two heads compete for
  the same rows, and the action head lost.

`step_acc` is 8.3 points below baseline, so the condition fails on its own terms. It also fails for
a reason that matters more: **the drop is concentrated in the class the `is_done` label was built
from.** Training the redundant readout did not add information; it consumed capacity.

## What this does NOT say

- It does not say the episodic rebuild was wasted. Grounding moved in the right direction and the
  corpus is still the only one with median 6 steps/episode. The corpus is reusable; the `is_done`
  slot is not.
- It does not say the model is bad. `type_acc` 0.776 and grounding 0.523 are ordinary for a 2B on
  this benchmark. What fails is the specific claim that a redundant terminal readout helps.
- It does not license an AndroidWorld number. `PRE-REG` §(c) says task success at n=116 has an MDE
  of ~14 points and *"a task-success number from this study would be noise read as signal."* Any
  AndroidWorld run from this checkpoint is descriptive, and must be labelled "not a test".

## Consequences, in order

1. **Remove the `is_done` gate from the runner.** It is a protocol deviation (upstream lets the agent
   decide `done`) resting on a null number, and it generated three defects of its own on
   2026-11-10: three refused terminations produced identical `wait` signatures, tripped `stuck_k`,
   and the third refusal was overwritten with `navigate_back` — which is what destroyed
   `MarkorCreateNote` with the note correctly typed on screen.
2. **Do not retrain to rescue `is_done`.** The label does not exist in this corpus. A task-completion
   signal has to come from a corpus with mid-episode completion states, or from the live environment.
3. **Roll the checkpoint decision back to the base for any next step**, or record `episodic-s0` as a
   negative result on AC-v2 and keep it only for a grounding-focused experiment.
4. **Re-scope the paper claim.** "Calibrated act/ask/refuse" cannot be supported by a signal that is
   a copy of the action label. This is the same failure mode the programme already recorded for
   `p(status)` gating on 2026-11-07 ("monotone-harmful at every τ").

## Why this took until now to surface

The eval was run on 2026-11-09 22:58 and written to disk. Nobody read it. Between then and
2026-11-10 the work went into the AndroidWorld loop instead — the one measurement the
pre-registration explicitly excludes. The order was: benchmark first, certify second. The correct
order is the reverse, and it costs one `cat` to check.
