# THESIS PROPOSAL 2026-11-10 (v2) — one decision model, no planner, no external model

> **SUPERSEDED 2026-11-10 15:20 UTC by `docs/research/PLAN-2026-11-10-question-tree.md`.** The pooled figures below (57%, 83%) are dominated by a few long episodes; the plan quotes the episode-clustered ones (no-effect 53.9%, CI 35–66%; first-encounter repeat 68.2%, CI 55–80%). Repetition is a symptom shared by several causes, and prior art shows generative agents loop too, so C1–C3 here are hypotheses to be tested in the plan's order, not a thesis.

*Operator constraint (2026-11-10): a single decision model only — no planner, no external/API model.
Status: PROPOSAL. Supersedes `thesis-2026-11-10-priced-handoff.md` (rejected).
(File name kept from a first draft about termination; the measurement below moved the thesis.)*

## 0. The measurement that set the thesis (B0 partial: 34 scored episodes, 602 executed steps, official protocol)

| Episode outcome | n |
|---|---|
| lost — never declared done, world not in goal state | 21 / 33 (64%) |
| premature done | 6 |
| success | 4 |
| world right but never declared done | 2 |

A first guess ("termination is the main failure") was **wrong**: termination explains ~24%.
What the steps show instead:
- **57.0% of executed steps change nothing in the UI** (343/602; click 60%, scroll 84%).
- **83% of those no-effect steps directly repeat a no-effect action of the same type** (286/343).
- The first action is usually right (`open_app` of the correct app); episodes then die in runs of 9–77 clicks.
Caveat: "UI changed" is measured on the accessibility structure (class, text, bbox); canvas-only changes (drawing,
camera preview) are invisible to it. Numbers are partial until B0 finishes all 116 tasks.

Mechanism, two parts, both checkable:
1. **Acting on guesses.** The V track emits a pixel point from a box head (median miss 437 px on AC-v2); a click on
   a non-interactive spot does nothing.
2. **A record that hides the outcome.** The history says `click` whether or not anything happened. One-pass decision
   models weigh the action record more than the observation (arXiv 2610.25947: record effect > observation effect
   in 12/12 settings; models still stop or advance after a claimed-but-failed step in 53–100% of cases). Our agent's
   record never says "nothing happened", so it repeats.

## 1. Claims (each with what refutes it)

**C1 — Diagnosis.** On the official AndroidWorld protocol, the dominant failure of a single-pass GUI decision model is
*repeated no-effect action*, not termination or reasoning. *Refuted if*, at 116 tasks, no-effect repetition accounts
for a minority of failed episodes' steps, or lost episodes are mostly driven by something else (wrong app, typing).

**C2 — Act on what is there.** Choosing among elements actually present (candidates extracted exactly as AndroidWorld
does, `forest_to_ui_elements`; executed at the element's centre) instead of guessing a pixel removes
click-on-nothing by construction and lowers the no-effect rate. *Refuted if* the no-effect click rate does not fall,
or task success does not rise, against the pixel-box model trained on the same data with the same recipe.
Prior evidence: on AC-v2 with matched weights, candidates beat boxes by +6.03 step accuracy (p = 1.5e-3).

**C3 — Write what happened into the record.** Rendering whether each action changed the UI into the history —
program-verified from the next observation, identical at train and serve — cuts repeated no-effect actions and raises
success. Training signal from existing data: consecutive AndroidControl accessibility trees give the true effect of
every demonstrated action; recovery examples are synthesized by inserting a no-effect action on a distractor element
into the history, with the demonstrated action as the target. *Refuted if* repeat-after-no-effect does not fall
relative to the same model trained with the action-only record (R1). This is, to our knowledge, the first test of the
record-over-observation effect on a live agent benchmark rather than in controlled probes.

**C4 — The ceiling, stated honestly.** What stays out of reach of one pass: computed answers (10 of the 25
answer-only tasks), multi-update state tracking, text absent from screen and goal. Reported as a measured ceiling,
not hidden.

## 2. Experiments (one variable each, 3 seeds for any offline delta, paired per task online)

| Run | Changes vs previous | Tests |
|---|---|---|
| B0 (running) | — (E2) | C1 |
| R1 (training) | history = executed actions (contract v1) | baseline for C3 |
| R2 | + element candidates, AndroidWorld extraction | C2 |
| R3 | + effect in the record (+ synthesized recovery rows) | C3 |

## 3. What it needs

- Raw AndroidControl accessibility forests (for the AndroidWorld extraction and the per-step effect labels): the
  compact corpora on disk do not keep tree structure; rebuild from the public TFRecords (done once before).
- No external model, no planner, no AndroidWorld-template training data.
