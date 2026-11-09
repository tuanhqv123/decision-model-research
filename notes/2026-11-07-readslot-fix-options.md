# The empty `app_name` read slot — what it is, and the two honest fixes

*2026-11-07. Follows `notes/2026-11-06-online-interface-defect.md` and §5.7 of the paper.*

## What is measured

On AndroidWorld, the model emits `open_app` 27 times in 60 steps and **24 of those carry `app_name = ""`** at
action confidence 0.92. The serving API does return names when they are stated (probes return `clock`,
`Markor` on AC-v2 screenshots), so this is not a serving bug. The name slot copies an app name that the goal
states verbatim; when the goal only *implies* an app ("turn on wifi"), the slot has nothing to copy and emits
the empty string. The action slot and the name slot disagree, and the adapter correctly refuses to invent a
name, so the action is marked invalid and replaced by `wait`.

This is an inference the AC-v2 training set never taught: its `open_app` labels carry the app name as a field,
but the goal text and the label do not require the model to *infer* an app from an implied goal.

## Two fixes, and what each costs

**Fix A — declared fallback (measured, already implemented).** Recover the app name from the goal when the
read slot is empty, stamp every affected step `scaffolding: ["app_name_fallback"]`, and mark the run
`suite.diagnostic`. Effect on the 8-task ablation: invalid actions **20/71 (28.2%) → 10/68 (14.7%)**, point
estimate roughly halved.

Two honest caveats. The Wilson intervals **overlap on 19.0–25.0** and Fisher's exact p = **0.065**. And the two
conditions were run on **different randomised task instances** (5 of 8 differ), so the comparison is unmatched
and the effect is a hypothesis. The replay machinery works (`gb_run1`/`gb_run2` are 5/5 identical); it simply
was not used for that ablation.

**Fix B — teach the slot (proposed, not run).** Synthesize training rows whose goal implies an app and whose
label is `open_app` with that app's name, using the training set's own `open_app` rows to map goal phrasing →
app name. This changes the model rather than the harness, so it composes with Fix A and does not need a
declaration.

## What the next measurement must do (pre-registration sketch)

1. **Paired instances, replayed.** Run the arm and the baseline on the **same** task parameters using the
   `gb_run1`-style replay, or the invalid-action comparison is confounded again.
2. **Primary:** invalid-action rate on emitted actions, with the per-action-type breakdown, plus
   `SR_all` vs `SR_reachable` and the checker-exception count. n = 8 per condition is underpowered for SR; the
   invalid-action rate is the claim, and SR is reported as context.
3. **Pass:** invalid actions fall by ≥ 10 pp with no increase in task-checker exceptions and no loss of the one
   task the model can complete.
4. **Kill:** if the synthesized rows make `open_app` over-fire on goals with no app, the arm trades one invalid
   action for another and is rejected.

## Why this is written down rather than run

Fix B is a data change, not a hyper-parameter, so it can be evaluated cheaply on the existing 8-task suite
once the harness can run paired instances. Doing it before that would reproduce the exact confound the audit
just found.

Artifacts: `backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/`, `backup-2026-11-06/reports/`,
paper §5.7, `notes/2026-11-06-online-interface-defect.md`.
