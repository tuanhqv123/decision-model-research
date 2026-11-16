# Which patterns model v3 learned wrongly, and what extra data is needed (2026-11-12)

Every number in this document can be recomputed from the scripts and result files in `benchmarks/results/2026-11-12-patterns/`.
- **Training data** is `compact_awlike_full_v3.jsonl`: 27,454 rows, 4,197 episodes.
- **AndroidWorld data** is the v3 ckpt_1824 runs on seed 30, masking off: 1,463 scored steps.

This document replaces `2026-11-12-PLAN-passk-gate.md`. That version optimised how to explore before we knew where the model was wrong, so it is
postponed.

## 1. How to find patterns (so we do not pick by hand)
Each step is tagged with a **situation**, made of three features that both the training data and the AndroidWorld logs have:
1. the action immediately before it;
2. whether that action changed the screen, exactly as the history line the model reads says ("→ screen changed" or "→ no change");
3. whether that action was a click on a text input field.

For each situation, I count four things:
- how often it appears in the training data and on AndroidWorld;
- the rate of failed steps on AndroidWorld, that is, an action with no effect or a wrong "done" report;
- what the data teaches to do in that situation;
- what v3 actually does.

Situations are ranked by **number of failed steps**: that is where fixing gains the most. No threshold is set by me.

## 2. Results (`situation_gap.out`)
On AndroidWorld there are **493 failed steps**: 444 no-effect action steps and 49 premature "done" reports.

| Situation (previous action, effect, is input field) | Training rows | AW steps | AW/training | Failed steps on AW | % of all failed steps | What the data teaches | What v3 does on AW |
|---|---|---|---|---|---|---|---|
| click, **no change**, not an input field | 268 | 201 | 14× | 161 (80%) | **33%** | click 62%, status 16%, wait 13% | click 87% |
| scroll, **no change** | 90 | 143 | 30× | 143 (100%) | **29%** | scroll 46%, status 33% | scroll 99% |
| click, changed, not an input field | 15,050 | 450 | 0.6× | 71 (16%) | 14% | click 61%, status 22% | click 78%, status 10% |
| click on input field, **no change** | 25 | 68 | 51× | 57 (84%) | **12%** | type text 60%, click 36% | click 84%, type text 16% |
| type text, changed | 1,664 | 59 | 0.7× | 21 (36%) | 4% | click 90% | click 92% |

The column "AW/training" compares the share a situation has among AW steps with the share it has among training rows.

## 3. Pattern 1: "the last action had no effect"
**This is the biggest failure: 73% of failed steps.**

| | Training data | AndroidWorld |
|---|---|---|
| Step right after a no-effect action | **387 rows = 1.4%** | **416 steps = 28%** |
| After a no-effect click, click **that same element** again | 45/539 = 8% (`train_patterns.out`) | 540/625 = 86% (`aw_patterns.out`) |
| After a no-effect scroll, keep scrolling in the **same direction** | 34/41 (`detail_checks.out`) | 142/142 |
| After a no-effect click on an input field, click that same field again | type text 15 / click 8 (only 23 rows) | 120/132 = 91% |

**Where the model learns wrongly:**
- **Click:** the data does not teach repeating (only 8%), but this situation is almost absent from the data. The model never learned
  that the line "→ no change" must change the decision.
  - When it repeats, the screen is identical and the history gains only one line, so the input is almost the same as at the previous step. The model outputs exactly
    the old decision. This is reasoning from the mechanism, not measured directly.
- **Scroll:** the data itself teaches wrongly. 34 of 41 rows keep scrolling in the same direction after the screen did not change. These 41 rows
  need to be read by hand: it may be a slow-loading list, or the "no change" label may be wrong.
- **Input field:** in the data, clicking a field almost always changes the screen (674/699). The reason is that the keyboard pops up: 637/699 screens
  after the click have a keyboard element (`focus_skew.out`).
  - On AW, the input field is often already selected and the keyboard was already there, so one more click changes nothing: 12/37 cases in
    self-play, and all 12 had the keyboard from before (`aw_patterns.out`).
  - In addition, AW **never** reports `is_focused` (0/18,470 elements), while the data has 584/629 screens that report a field as being
    selected. This is a mismatch between training time and run time.

**Where the model clicks without effect (`detail_checks.out`):**
- 330/353 such clicks land on an element that **can be clicked**. At the top are:
  - an input field that already holds the text just typed: 65 + 25 + 15 times. After typing, the model keeps tapping the field instead of moving on;
  - "Save" tapped with no effect: 44 times;
  - an unlabelled calendar cell: 86 times.
- So the added data must simulate **those situations**, not tapping on an element that cannot be tapped.

## 4. Pattern 2: reporting done too early (10% of failed steps)
- 49 wrong "done" reports. The median step is 5. 47/49 came right after a click, and for 37/49 that click changed the screen
  (`aw_patterns.out`).
- Data: `status` is always the last row, 4,197 rows like this, and 80% come right after a click that changed the screen.
  - After a click that changed the screen, 22% of training rows are `status`.
  - Episodes are short: median 6 rows, p90 is 11 rows, only 3% are longer than 15 rows (`train_patterns.out`).
- **The pattern the model learned:** "a few steps, one click succeeds, so it is done". On AW, 33/49 wrong "done" reports fall in tasks with a
  budget above 10 steps.

## 5. Data to add, ordered by number of failed steps

Every way of creating data below uses only the existing AndroidControl data and verifiable rules, with no external
model. Each must keep the physical truth: the inserted action must really have no effect on that screen.

| # | Fixes | How to create the new row | Why it is physically correct | Available source (needs an exact count) |
|---|---|---|---|---|
| D1 | Typing, then tapping the field again (65+25+15 times) | Take the row right after a text-typing action; insert into the history "click [the field just typed in] → no change"; keep the label as is (data: click 90%, usually Save or the next button) | The field is already selected and the keyboard is already there, so a click changes nothing | about 1,664 rows of the situation "type text, changed" |
| D2 | Tapping a field that is already selected (12% of failed steps) | Take a row whose label is typing, where the field is already selected and the keyboard is already there; insert "click [that field] → no change"; the label stays typing | Same as above | about 194 rows (179 + 15 in `ime_before.out`) |
| D3 | Scrolling the same direction forever (29%) | Take a list screen right after opening the app; insert "scroll up → no change" (we are at the top of the list); keep the label. In parallel, read by hand the 41 rows that teach scrolling in the same direction, and fix the labels if they are wrong | At the top of a list, scrolling up changes nothing | Needs counting |
| D4 | Clicking again on the element that just had no effect (33%) | Use the 268 real rows. Add a variant: insert "click [the currently selected element, `selected`] → no change" before the original row | Tapping a tab or item that is already selected usually changes nothing. **Needs verification**, because not every app behaves this way | Needs counting |
| D5 | Reporting done early (10%) | Concatenate two episodes in the same app ("A, then B"); at the last screen of A the label is to go on (`navigate_home` or `open_app`), not `status` | The "A is done" screen is real, it is just that the goal is not done | Needs a count of same-app pairs |
| D6 | Focus mismatch between training time and run time | Remove the `focused` flag from the input and from the "did it change" comparison at training time, or fix the harness so it can read focus. One of the two must be chosen | Makes training time and run time see the same kind of thing | The whole data |

**Checks before retraining (no long GPU run, no emulator needed):**
- Build an evaluation set **AW-NC**. Each sample is an input the model really saw in self-play seed 1001, at the step right after
  a no-effect action. `state_in` was recorded, so it can be replayed exactly as it was.
- Measure the rate at which the model picks again the very action that just failed:
  - current v3: expected about 86%;
  - v4 (with D1–D4 added): must drop clearly, and AC-High must not drop by more than 1 point.
- Early "done": measure P(status) on the self-play states that the scorer says are **not done**. This rate must drop, and P(status)
  on states that are **done** must not drop.

**What is not known, stated plainly:**
- A failed step is not a failed task.
- Masking blocked repeating mechanically: the repeat rate fell from 86.5% to 62.2%, but SR only went from 16 to 19/112 (p = 0.375), and
  early "done" reports went up from 49 to 56.
- Blocking repeats alone is not enough. The model must **know what to do instead**. D1–D5 teach exactly that, because the label is the correct action, not
  the model's second-best choice. How much SR will rise is not known; the check above answers it before we spend the effort to train.

## 6. Patterns that are real but are not solved by adding data of the kind above
- **Cutting to 40 elements:** 454/1,541 steps have more than 40 candidates. This is an input fix, with its own plan.
- **The `answer` action:** 25 tasks. A new action must be added to the contract.
- **19/33 apps have no data:** the test "clicks in unseen apps fail more" was **refuted** (36.3% against 43.3%).
  Do not rank it as a priority without a new measurement.
