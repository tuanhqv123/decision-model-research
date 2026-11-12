# Error analysis, step 6: what to do about each failure (v3h, seed 30: 96 scored failed episodes; 3 unscored excluded)

Method: skill `error-analysis`. For each episode the FIRST wrong step was read and noted (`notes_v3h.md`), the notes
were grouped (`labels_v3h.py`, `categories_v3h.out`), and for annotated first-error steps the rank of the correct
option was read from the model's own recorded readouts (`rank_correct.py`, `rank_correct.out`; no model call).

## 1. Engineering bugs: fix directly

| Failure | Evidence | Fix | Status |
|---|---|---|---|
| Every `answer` scored 0 (25 answer-only tasks = 22% of AW) | v4 answered "100 Main St Chicago", "Trivia Night", "Grocery Shopping" verbatim; upstream checker on those strings = 1.0; live run = 0.0. On device: inner env holds the answer, the wrapper reads "" | `upstream task_eval.py:145` writes `env.interaction_cache = ""` on our wrapper, shadowing `__getattr__`. Fix: `interaction_cache` property passes reads and writes through | **fixed**: test RED→GREEN, 270 pass; verified on device (right 1.0, wrong 0.0); harness V10 |
| Chrome first-run screens block 3 Browser tasks | traces #1–#3: "Use without an account" / "Add account" | upstream `ChromeApp.setup` clicks "Accept & continue", "No thanks"; this Chrome build shows "Use without an account", so setup leaves the first-run flow open | environment; not fixed (3 hard tasks: draw, maze, multiply) |
| SMS tasks: "Not sent" | 6 tasks; modem simulator unresponsive | none in time | reported separately |

## 2. Decision from the model's own ranking: fix at inference (one model, no planner)

**Scroll chosen while click is ranked 2nd** (`scroll_vs_click.out`):
- v3h has 247 `scroll` steps, and 245 of them are in failed episodes.
- 146 of the 247 changed nothing.
- On 238 of the 247, `click` is the model's 2nd choice.

At the annotated first-error steps (`rank_correct.out`, exact label match):

| Task(s) | Right element | Right action | Model chose |
|---|---|---|---|
| 5 WiFi tasks | "Internet", rank **1** | click, rank 2 (P 0.15–0.39) | scroll |
| RecipeDeleteMultipleRecipesWithNoise | "Grilled Cheese with Tomato and Basil", rank 1 | click, rank 2 (P 0.04) | scroll |
| SimpleCalendarAddOneEvent | "OK", rank **2** | click, rank 2 (P 0.48 vs scroll 0.50) | scroll |

Cases where this fix cannot apply:
- **3 Brightness tasks:** "Display" is not on screen or in the list at that step, and P(click) is 0.03–0.04. The first scroll is reasonable there; the later end-of-list scrolls are the problem.
- **FilesDeleteFile:** the target file is NOT on screen. The model clicks the distractor `edited_q2a8_fancy_banana.mp3`.

**Prior measurement.** Verified no-effect masking (`policy.loop_detection.mask`) on v3, 112 paired tasks (`2026-11-11-v3/postmortem/mask_paired_full.out`):
- 19/112 vs 16/112: **4 wins, 1 loss, sign test p = 0.375** (not significant).
- Wins: SystemWifiTurnOn, SystemWifiTurnOnVerify, TurnOnWifiAndOpenApp, RecipeDeleteMultipleRecipesWithConstraint. Loss: OpenAppTaskEval.
- Declared-done failures rose 49 → 56.

**Run plan** (`docs/research/2026-11-12-PLAN-FIX-vi.md` §4, revised 14:05): arm A = v4 + harness V11 (paired with v4/V8: harness fixes); arm B = merge 0.3 + V11 (paired with A: model). Masking deferred.

## 3. Representation: needs a retrain to change

| Failure | Evidence | Candidate fix |
|---|---|---|
| Toggle state misread (SystemBluetoothTurnOffVerify) | the switch was already off (`is_checked` False, rendered only as "checkable"); the model clicked it on, then stopped; P(status) 0.03 | render explicit "on"/"off" for checkable elements in train and serve |
| Unlabelled row buttons (Retro playlists) | the song's "(no label)" menu button carries no song name | label an unlabelled control with its row's text |

## 4. Knowledge: no inference fix

| Failure | Evidence |
|---|---|
| App procedures (32/96 = 33% of scored first errors) | e.g. Markor delete: long_press ranked 2nd but P 0.04–0.05; the date picker overshoots; creating a folder instead of a copy |
| Picking among similar items (6/96) | distractor files, duplicate detection, keypad digit "6" ranked 6th |

These need data that shows the procedure. AndroidControl has no Markor, Simple Calendar, Broccoli, Retro, Joplin or OpenTracks goals (13/20 AW apps have none).

## Expected effect of sections 1 and 2 (no training)

v4 baseline (`2026-11-12-v4/compare_v3h_v4.out`): **score sum 15.5/115 = 13.48%** official (15 full successes + RetroSavePlaylist 0.5).

| Fix | Expected | Basis |
|---|---|---|
| answer fix | +4 tasks (offline rescoring with the upstream checker: 19.5/115 = 16.96%) | `ir_rescore_v4.out`; a counterfactual, not a run |
| masking | +0 to +3 | prior result 4/1, p = 0.375; it could also be zero |

These are predictions. Arms A and B measure them.
