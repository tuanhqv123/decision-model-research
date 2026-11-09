# D04 — coordinator labels: what is real, what is a tautology, and what must change

## What was built

A coordinator target set from 88,288 AndroidControl training steps (episodes split 70/15/15 by episode id, no
episode crossing partitions):

| rule | count | label | learnable? |
|---|---|---|---|
| `no_plan` (no history) | 13,575 | CREATE_OR_UPDATE_PLAN | **NO** — the label equals the feature `hist_len == 0` |
| `repeat_subgoal` | 9,963 | CREATE_OR_UPDATE_PLAN | **NO** — the label IS the definition of the feature |
| `recovery_back` | 289 | CREATE_OR_UPDATE_PLAN | partly (action_type is a legal input, but n is tiny) |
| `app_switch` | 66 | CREATE_OR_UPDATE_PLAN | partly, same caveat |
| `plan_valid` | 64,395 | CONTINUE_PLAN | yes |

After excluding `no_plan` (deterministic): **74,713 rows, 86.19% CONTINUE / 13.81% UPDATE (6.2:1)**.

## The two tautologies, stated plainly

1. **`no_plan` vs `hist_len == 0`.** 13,575 of 23,893 UPDATE labels (56.8%) are given away by a feature. A
   model that learns only this scores 57% of the UPDATE class for free.
2. **`repeat_subgoal` vs `active_subgoal == prev_subgoal`.** In the test split, **90.2%** of UPDATE rows have an
   identical active and previous subgoal, versus **6.3%** of CONTINUE rows. My feature `j(active, prev)` therefore
   separates the classes almost perfectly — and my *label rule* is what put them apart.

**A classifier reaching 98.1% accuracy, AUROC 0.982, ECE 0.018 on 11,200 held-out rows (lift +11.2 over
always-CONTINUE) is not evidence of learning plan staleness. It is evidence that the label is a function of the
features.** Reporting it as a coordinator result would be the exact error this project has now made eleven times
today.

## What the guide actually asks for, and what this data cannot give

Guide §35–§39 requires **branch comparison**: run the Continue branch and the Update branch from a documented
equivalent state, and prefer the successful branch. That requires an environment that can be reset, a planner,
and an evaluator — i.e. a live AndroidWorld loop with plan execution. AndroidControl is offline step data with no
planner and no resettable state, so **the guide's label rule cannot be instantiated here.**

What this data *can* support, and what is therefore the honest next target, is a **different and weaker question**:

> Given (goal, active subgoal, previous subgoal, history, last action), does the agent's *next instruction differ
> from the current one*? That is a **subgoal-advance predictor**, not a plan coordinator: it says "the plan's
> pointer moves" (very learnable, and what the 98% is actually measuring), not "the plan is stale" (which needs
> branch evidence).

## Recommendation

1. **Do not report the 98.1%.** Relabel it accurately as a *subgoal-advance predictor* if it is used at all, and
   say it is near-tautological by construction.
2. **The coordinator labels the guide asks for require the online loop** — planner + reset + evaluator. That is
   the D02/D06 work (AndroidWorld adapter), not a data-processing step.
3. **The one coordinator question AndroidControl can answer honestly** is exactly the one the existing
   `status` readout already answers: *is the task finished?* That is measured (84.17% type accuracy) and is not
   the same thing as *should I replan?*

## Files

`coordinator/D04_coordinator_labels.py`, `coordinator/labels_train.jsonl`, `coordinator/D04_audit.json`,
`coordinator/split.json`, `coordinator/D05_train_coordinator.py`, `coordinator/D05_metrics.json`.
