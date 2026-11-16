# Plan v4: fix all the errors in one training run, aiming at about 40% on AndroidWorld (2026-11-12)

This version replaces the one of the same name written earlier the same day (the version that only fixed the "no change" signal).

**Goal set by the operator:** get close to existing 2B–4B models, about 40% on AndroidWorld (about 46 of 114 scored
tasks). v3 currently scores 16/114 = 14%.
- Comparison with numbers in papers is for orientation only. They use a different harness, so a direct ranking is not possible (rule M4).
- The nearest reference point is AppVLM 3B: trained on AndroidControl only it gets 11.4%; adding AW runs confirmed by the checker raises it to 37.8%
  (their numbers, Tab. 4).

Evidence: `docs/research/2026-11-12-DATA-AUDIT.md` and `benchmarks/results/2026-11-12-*`.


## 0. Update 2026-11-12: v4 uses only existing data plus augmentation, no emulator runs
The operator decided: this training run must fix every error, using the data we have plus augmentation, without waiting for new AW runs.

| Branch | How it is done with augmentation | Can it be fully fixed? |
|---|---|---|
| W0 harness | The history line is written against the screen of the next step; has a test | Yes |
| W1 loop | Remove the 52 rows that teach repetition; add 10% rows with one hypothetical failure and 3% rows with 2–3 consecutive failures. The effect label is still "did the element list change", identical to run time | Yes in mechanism: measured on the data, the repeat rate falls from 19.2% to 1.6% |
| W2 early done | Join two AndroidControl episodes of the same app into a two-part goal ("A. Then B."), connected by `navigate_home`. On the "part A done" screen the label is to continue; `status` only at the end. **No self-play** (the operator dropped it) | Yes in mechanism (P3, P4); only for the error type "stops after the first part" (see 0b) |
| W3 `answer` | Add an `answer` action and an `answer_el` slot (choose the element whose text is the answer). The data are template-generated questions over AndroidControl screens; the answer is the text of exactly one element (for example "What is the title at the top?", "What is the time of the first alarm?") | Teaches answering by copying. Counting or summing questions (about 8/26 answer types) are **not** covered |
| W4 drop the 40 cap | New rule `nocap_reading_noime`, applied at both training and run time | Yes |
| Focus | Set `is_focused = False` in the training data, because AW never reports this flag | Yes |
| W5 workflows in the 19 apps with no data | **Not done in v4.** There are no AW data to learn from, and self-play was dropped | No. This is the part that limits the ceiling |

**Checks before training, no GPU cost:**
- the repeat rate after "no change" is lower than after "changed";
- the per-step `status` rate no longer peaks at steps 5–8;
- "New Event" is in the list;
- `answer` rows have the answer inside the element list;
- the test suites of the harness, the training code and the server are green.

**Checks after training:** probes P1–P5, AC-High, then AW 116 tasks.

## 0b. Re-reviewing the plan from another angle (2026-11-12)
Each point is a question "if the plan is wrong, where is it wrong", with a measurement.

1. **"Early done" is really three different errors** (`2026-11-12-expect/premature_goals.out`).
   - There are 49 wrong done declarations. **14 are tasks that must be answered in text**: the model reached the right screen, but
     had no `answer` action, so it declared done. For example SimpleCalendarLocationOfEvent clicks exactly "100 Main St Chicago", which is the answer.
     This group is fixed by W3, not W2.
   - Of the remaining 35, 18 have a multi-part goal **and** a final action of the Save/Delete/Create kind. This is stopping after
     the first part is done (adding 1 of 3 recipes, deleting 1 of several events). The episode joining in W2 targets exactly this type.
   - The rest are **other** errors, not termination errors: pressing the wrong send button ("SMS"), typing the wrong content ("clipboard content"
     instead of the clipboard's content). Termination augmentation cannot fix these. They are recorded so that expectations are not set wrongly.
   - The counts of "multi-part" and "Save kind" use a crude regex, so they are estimates.
2. **The project record notes "collapses with more than 40 candidates"** (−17.3 / −23.8, date 2026-11-07).
   - That was the old design, whose output is coordinates: a long list damaged the coordinate output. The v3 contract chooses the element by index,
     which is a different mechanism.
   - But the risk "a long list makes selection worse" is still real. **Must be checked before training:** run v3 without the cap on
     the 656 AC-High rows with more than 40 candidates (a few minutes of GPU) and compare with the cap at 40. If it collapses, keep a higher cap
     with a priority order, instead of removing it entirely.
3. **`answer` may be used too early.** If every `answer` row is at step 0 (a question about the screen currently shown), the model will
   learn "question → answer immediately". The AW tasks need navigation to the right screen first.
   - Fix: place the `answer` row at the **end of a real AndroidControl episode** ("<goal A>, then tell me <question about the
     final screen>"). The earlier steps keep their real labels; only the final screen gets the `answer` label.
4. **The hypothetical element in W1 is currently chosen at random**, so it is an easy negative. On AW, the model repeats a **plausible-looking**
   element.
   - Fix: prefer elements of the same type as the correct answer or close to it, to get harder negatives. The two metrics are still re-measured
     on the data.
5. **Mixing ratio.** Augmented rows must not overwhelm the original data. Set caps: W1 about 13%, W2 about 15%, W3 about 5% of
   the rows.
   - Reason: the 2026-11-07 lesson showed that changing only the `status` weight failed (−27 wrong done declarations traded for −26 correct done
     declarations). We must add **information**, not adjust weights.
6. **Leakage into the evaluation tasks:** none, because v4 uses no AndroidWorld rows other than the 1,736 app-choice rows already
   in v3 (19 templates held out).

## 1. Error map of v3 (seed 30, 114 tasks, `expected_gain.out`)
| Error group | Tasks | Measured cause | Fix direction |
|---|---|---|---|
| Declared done too early | **35** | P(status) by step order: in the data it rises from 0 to 28% at steps 5–8; v3 in the not-done state peaks at step 5. The model reads the remaining part of the goal very weakly (P4: −8 points). The data have no example of "partly done, goal has another part" | W2 |
| Stuck in a loop | **26** (17 tasks ≤ 20 steps) | The model ignores "no change" (P1: −0.3 points), copies the failed attempt (P2: +1.3). The data teach repetition: 19.2% versus 5.4% | W1 |
| Needs a text answer (`answer`) | **25** | The contract has no `answer` action, so the ceiling of this group is 0 | W3 |
| Wanders and runs out of steps | 12 | 19/33 AW apps have no training row; the model does not know the workflow inside the app | W5 |
| List cut at 40 elements | overlaps the groups above | 454/1,541 AW steps have more than 40 candidates; the "New Event" button is cut 155/155 times; Simple Calendar 0/17 | W4 |
| Harness records the wrong effect label | overlaps the groups above | 9% of "no change" clicks are read too early | W0 |

## 2. The fix branches, all in the same v4
Each branch has **its own measurement** (a probe or a specific metric), so we still know which branch has an effect without
retraining many times.

### W0: fix the harness (no retraining needed)
- The history line of step k is fixed at the start of step k+1, against the element list of step k+1. This is exactly the definition used when generating
  the data (`add_history_v3.py`). Has a test.
- **Measure:** the rate of "no change" where the image still changes by 1% or more, on AW.

### W1: the "no change" signal (loops)
- Remove the 52 target rows that teach useless repetition.
- Add about 10% rows whose history has "click a different element → no change", with the label unchanged.
- Add about 3% rows with 2–3 consecutive failures.
- **Already tried on the data:** repetition after "no change" falls from 19.2% to 1.6%; the information carried by the words "no change" rises ×20.
- **Probe gate:** P1 ≤ −5 points; P2 must fall (in v3 it rises).

### W2: early done
This is the largest group, 35 tasks.

**(a) Multi-part goals, joined from two AndroidControl episodes of the same app.**
- New goal: "A. Then B."
- On the final screen of A, the label is to **continue** (the first step of B), not `status`. `status` only at the end of B.
- The source is sufficient: 615 episodes that start from the home screen and 400 apps (`v4_sizing.out`). A joined episode is about 13 rows long, so the
  `status` rate will no longer concentrate at steps 5–8.
- B will be chosen to start from the home screen and joined with `navigate_home`, so the screen sequence stays real.

**(b) `status` labels taken from AndroidWorld itself, using the checker queried after every step.**
- Run on training instances (a seed other than 30), with no external model.
- Checker state says **done** → positive `status` row.
- State **not done** while the model chooses `status` → **negative row for `status`**. This needs an added unlikelihood loss component
  in `train_d1.py`, −log(1 − p_status), applied only to these rows. Has a test.

**(c) Probe gates:**
- P3 (add 10 `wait` rows to the start of the history) is almost unchanged.
- P4 (add a clause for unfinished work to the goal) must reduce P(status) by at least 30 points (v3: −8).
- On AW not-done states at steps 3–5, mean P(status) must fall by half.

### W3: the `answer` action (25 tasks)
- **v4 contract:** add `answer` to the action choice, and an answer slot that points at the **text of one element** on the screen
  (copying, the same way `type_text` points into the goal). The harness sends `answer` using the exact AW protocol (`interaction_cache`).
- **Data:** training instances of the 25 answer templates, with the answer taken from the task's own parameters (no external model needed).
  On screens where the answer appears verbatim in the text of an element, create a row "answer → that element".
- **Must be measured first:** how many of the 25 templates have the answer appearing verbatim on the screen. Counting or summing
  questions cannot be copied. Already known: 16 string answers, 8 numbers, 1 date, 1 time (the total of 26 versus 25 is still not
  resolved).
- **Gate:** `answer` accuracy on training instances of the held-out templates.

### W4: remove the 40-element cap
- Train and serve with the full list under the v3 rule.
- 10.8% of rows have more than 40 candidates. The p99 token count is 2,659, the maximum 6,456, still very far from the 128k limit (`cap_tokens.out`).
- **Gate:** "New Event" is always present in the list. Accuracy on the 656 AC-High rows with more than 40 candidates does not fall.

### W5: workflow data inside AW apps (RFT), after v4 passes the gates
- Run v4 on training instances of the templates that are **not held out**. One third of the templates are held out and never
  trained on.
- Take the runs the checker confirmed as successes as imitation rows, take the branching steps between a failure and a success of the same
  task, and retrain as v4.1, v4.2.
- This is the strongest lever by the AppVLM reference point (11.4 → 37.8), and the only source that uses no external model for the 19 apps with no
  data.
- The exploration randomness level is chosen by measurement on AW (the new success rate per run), not set by hand.

## 3. Expectations
These are estimates, not results. Each number is a range, with a reason.

| Branch | Task group | Expected extra | Reason / risk |
|---|---|---|---|
| W1 + W0 | 17 loop-stuck tasks (≤ 20 steps) | +2 to +5 | Masking, which breaks the loop mechanically, really saves only about 1 task; v4 learns the correct action, so more is expected. Risk: shifts to the early-done error |
| W2 | 35 early-done tasks | +4 to +10 | We do not know how many tasks were on the right track when they declared done; the checker labels will tell |
| W3 | 25 answer tasks | +3 to +8 | Depends on the rate of copyable answers; measure first |
| W4 | Simple Calendar and long screens | +1 to +4 | Add-event tasks also need the text typed correctly |
| **v4 total** | | **+10 to +27 tasks, i.e. about 23–37%** | The branches interact, so they cannot be added directly |
| W5, RFT rounds | 12 wandering tasks and the tasks in apps with no data | the rest, up to about 40% | By the AppVLM reference point; different harness so not guaranteed |

## 4. Training and checking
1. **v4 data** = v3 + W1 + W2a + W3 + W4, with a run card. Check on the data before training:
   - the repeat rate after "no change" is lower than after "changed", and the information is ≥ 10 times v3;
   - the `status` distribution by step is flatter;
   - "New Event" is in the list.
2. **Train v4:** same recipe as v3 (lr 1e-5, frozen embedding, 1 epoch), plus the unlikelihood loss for the negative rows of W2b.
   - Order: v4 first, then one training AW round to collect the W2b and W5 rows, then v4.1.
3. **Gates in order:**
   - probes P1–P4;
   - AC-High no more than 1 point below v3;
   - `answer` on the held-out templates;
   - AW 116 tasks, seed 30, W0 harness, with v3 and v4.
   - Report by error group in section 1. The final SR runs with 3 seeds.
