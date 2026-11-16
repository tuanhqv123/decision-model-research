# Audit of the v3 training data: what the model learned, why, and the consequences on AndroidWorld (2026-11-12)

Every number can be recomputed from the scripts in `benchmarks/results/2026-11-12-data-audit/`; each script writes its result into the
`.out` file of the same name. This document goes deeper than `2026-11-12-PATTERNS-data.md`. That one shows **where** the model breaks; this one answers
**why** the data makes the model break there.

- **Training data:** `compact_awlike_full_v3.jsonl`, with 27,454 rows and 4,197 AndroidControl episodes. The element list
  was extracted with AndroidWorld's own `forest_to_ui_elements`.
- **Model:** v3, checkpoint ckpt_1824.

## Summary in five sentences
1. The model **does not read** the text "→ no change". Change it to "→ screen changed" on exactly the states v3 met, and 97% of decisions
   stay the same.
2. The model ignoring that text is **reasonable given the data**. In the data, this text carries only 0.09% of the information about the correct label. Where
   there is a signal, the signal even teaches the opposite: after "no change", the correct label **repeats** 6 times more often (14.7% versus 2.3%).
3. The cause is the data itself:
   - the "no change" label is both rare (1.9%) and often wrong (for scroll, in 46% of cases the image actually changed by ≥ 1%);
   - behind that label, the annotator often redoes the very action that just failed, and 52/79 of those redos also have no effect.
4. Early done-reports follow **step order**. The share of `status` in the data rises from 0% at step 0 to about 28% at steps 5–8. The
   done-probability of v3 in **not-yet-done** states on AW has the same shape, peaking at step 5.
5. Adding rows to the data is not enough. The "no change" text must be made **correct** and **useful** in the data, so that the model has a
   reason to learn to read it.

## L1. What information does the model use? (counterfactual experiment, `nochange_sensitivity.out`)
- **States used for the measurement:** 124 self-play states from seed 1001, each right after an action (not `wait`)
  that was logged as "→ no change".
- **Method:** send exactly the same input (image, goal, history, element list) to the same server again.
- **Check of the measurement:** resending unchanged gives the logged action in 124/124 cases. The model is deterministic, so the measurement is reliable.

| Variant of the last history line | Repeats the action that just failed | Reports done |
|---|---|---|
| A. As it really was: "… → no change" | 40/124 (32%) | 9 |
| B. Changed to "… → screen changed" | 38/124 (31%) | 10 |
| C. Line removed entirely | 63/124 (51%) | 1 |

- A and B give the same action 98% of the time, the same action and the same score 97%. The model **ignores** whether the action had an effect.
- C differs clearly from A: the repeat rate rises from 32% to 51%. The model **does read** "what I just did". The habit it learned is "usually do not
  redo exactly what was just done", and that habit does not depend on the outcome of what was just done.

**Gaps still open:**
- Most states are clicks (119/124). The conclusion for scroll and typing has too few samples.
- Self-play has masking switched on. So these are first encounters; the long loops (86% on seed 30) were not measured again here.

## L2. Why the model learned to ignore it: the "no change" text in the data (`effect_information.out`, `nochange_truth.out`)
**Rare:** 412/22,026 rows (1.9%). Rows after `wait` are not counted.

**It carries no information, and even teaches the opposite:**

| Previous action, effect | n | Correct label repeats it | Correct label is `status` | Correct label is `wait` |
|---|---|---|---|---|
| click, changed | 15,724 | 2.3% | 21.2% | 5.4% |
| click, **no change** | 293 | **14.7%** | 14.7% | 12.3% |
| scroll, changed | 1,733 | 46.6% | 13.2% | 1.2% |
| scroll, **no change** | 90 | 37.8% | **33.3%** | 7.8% |

- I(effect text ; correct action type) = **0.0015 bit**, that is, 0.09% of the uncertainty of the label.
- A model optimised on this data has a reason to ignore that text. If it learned anything from it, it would be "no change, so do it again",
  which is exactly what breaks on AW.

**Often wrong, judged by the image:**

| Action that the history marks "no change" | Image changed under 1% | 1–5% | 5–20% | 20% or more |
|---|---|---|---|---|
| click (293) | 75% | 8% | 12% | 5% |
| scroll (90) | 54% | 13% | 23% | 9% |

Reading 41 scroll rows and 25 click rows by hand (`effect_vs_pixels.out`, `click_nochange.out`) shows three sources of error:
- **The element list does not see the change:** swiping an image carousel (image changes 5–50%), the `RatingBar` stars, choosing AM/PM, the numeric keyboard
  for entering money. The action **did have an effect** but is logged "no change".
- **The action completed the job but the screen does not change:** tapping "SAVE" then `status`. 43/293 "no change" clicks are followed
  immediately by `status`.
- **The screen is loading:** "no change" then `wait` with the same step description, 36/293.

**Annotators repeat themselves** (`retry_outcome.out`):
- There are 79 cases where the correct label redoes exactly the action that just showed "no change". Afterwards, in 27 cases the screen changed and **in 52 it still did not change**.
- Of those 79 cases, in 51 the image actually did change (≥ 1%). Most are carousels, so the label is wrong rather than the annotator repeating uselessly.

## L3. Consequences on AndroidWorld (`../2026-11-12-patterns/situation_gap.out`)
- The step right after an action with no effect makes up **28% of AW steps**, but only 1.4% of the training data. These steps
  account for **73% of the failed steps**.
- On AW, "no change" is almost always literally true, that is, the action really had no effect. Common cases:
  - tapping an item that is already selected;
  - tapping "Save" with no effect (44 times);
  - scrolling to the end of a list.

  In contrast, in the AndroidControl data, "no change" often means "it did take effect but the element list does not see it". The same
  text carries two meanings in two environments.

## L4. Early done-reports are a pattern of step position (`status_by_step_train.out`, `status_by_step_aw.out`)
| Step | Data: P(status) | AW, not-yet-done states: mean P(status) of v3 | AW, not yet done: v3 chooses status |
|---|---|---|---|
| 1 | 6.2% | 0.5% | 0% |
| 3 | 16.2% | 6.0% | 2.0% |
| 4 | 22.0% | 11.0% | 8.2% |
| 5 | 25.6% | **17.8%** | **20.5%** |
| 6–8 | 28–29% | 8–13% | 3–13% |
| 15+ | 22.1% | 6.5% | 0.9% |

- In the 15 **already-done** states, the mean P(status) from step 3 onward is 52–99%. The model **does** tell done from not-done to some extent;
  it is not just counting steps.
- But in not-yet-done states, the probability of reporting done rises strongly around steps 3–5, exactly the zone where the data teaches that an episode usually
  ends (the median episode is 6 rows long; only 3% are longer than 15 rows).
- The data has no example of the kind "the screen looks done, but the goal has another part". `status` is always the last row
  (4,197/4,197).

**Gap:** there are only 15 already-done states, so an AUROC between done and not-done cannot be computed reliably.

## L5. Mismatch between learning time and run time at the input field (`../2026-11-12-patterns/focus_skew.out`, `ime_before.out`)
- **Field being focused:** AW never reports `is_focused` (0/18,470 elements). The data reports the field as focused on 584/629 screens
  after a click on the field.
- **Click on a field in the data:** 674/699 times the screen changed, mainly because the keyboard pops up.
- **Click on a field on AW:** the field is often already focused and the keyboard is already there, so another click changes nothing.

## Consequences for the data (replaces section 5 of the PATTERNS version)
Goal: make the effect line **correct** and **useful**, and only then add volume.

1. **Fix the effect labels.** Use the image as a second witness: if the element list does not change but the image changes by ≥ 5%, log it as
   "changed" (carousel, stars, picker).
   - At run time, the harness must use the **same rule**.
   - On AW, the image almost always differs because of the blinking cursor. The threshold must first be measured on AW images; this is work still to be done, there is no number yet.
2. **Remove or flag the annotator's useless repeats:** 52 rows whose correct label repeats an action that just had no effect, and the repeat
   also has no effect. If they are not removed, the data teaches the loop.
3. **Add "no change, then do something else" rows** with the inserted action following the physical truth (D1–D4 of the PATTERNS version): typing, then
   tapping the field again, tapping an item that is already selected, scrolling up when already at the top of the list, tapping the item that is currently selected.
   - **Criterion measurable on the data right away, before training:** after the fix, P(repeat | no change) must be **lower** than P(repeat | changed),
     and I(effect text ; label) must rise clearly from 0.0015 bit.
4. **Done-reports:** add "partly done, goal not done" situations (D5, joining two episodes of the same app), so that the `status` rate by
   step is no longer a curve of position.
5. **Focus:** choose one of two options: remove the `focused` flag from the training data, or fix the harness so that it can read focus.

**Check after training:**
- Rerun exactly the L1 experiment on v4.
- Pass if: changing "no change" to "screen changed" changes the decision in **most** states (no longer 97% identical), and the repeat
  rate in variant A falls clearly below 32%.
- At the same time AC-High must not drop by more than 1 point.
