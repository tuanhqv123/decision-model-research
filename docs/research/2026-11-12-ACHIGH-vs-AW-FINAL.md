# High AC-High score but low AndroidWorld score: why, and the final plan (2026-11-12)

Every number below is recomputed from files. An independent agent, reading the files without knowing the conclusion in advance, re-checked all of it (read-only, no edits).

**Data sources**
- AC-High: `benchmarks/results/2026-11-12-v4/actest/perstep-v4_s0_ckpt_2482_actest_v4.jsonl` (model v4 ckpt_2482, scored with the InfiGUI scorer).
- AndroidWorld: `benchmarks/results/2026-11-12-maskenv/aw116_v4g_s0{,_b,_c}` (v4 + harness V12, seed 30) and `compare_A_AG.out`.

## 1. The two benchmarks do not measure the same thing

| | AndroidControl-High (offline) | AndroidWorld (online) |
|---|---|---|
| What the model sees at step t | History and screen **of the human demonstrator** (teacher-forced): `eval_d1.py:42,48-49,220-225`; history built from the gold actions: `add_history_v3.py:32-39` | The screen produced by **the model's own earlier steps** |
| What is scored | **Each step**, compared with exactly the action the human took (`vendor/infigui_eval.py:83-122`). **No step checks whether the goal has been reached** | The **final state** of the device, decided by the task's checker |
| Stopping / answering | There is no stop step: the `status` row is dropped at `make_actest_v3.py:5`. There is no answer | The model must declare done by itself. 25/116 tasks are scored only on the answer. If it does not declare done, the score is 0, even when the device is already in the right state (upstream rule) |
| One step wrong | The next step still starts from the correct screen (errors do not accumulate) | Errors accumulate; the model must notice and go back |
| Path | Must match the human's path. Only click and text are scored leniently: a click that falls inside the gold box enlarged 1.2× counts, text only needs to be a substring | Any path is accepted |

"High" only means that the instruction given to the model is the goal, not each step. It does not mean scoring by goal.

## 2. On the same "whole task correct" measure the two sides are level

- **AC-High:** per-step correct **61.7%** (5,213/8,444), but correct at **every step** of a task only **344/1,543 = 22.3%**, even though the model is led by the hand at every step.
- **AndroidWorld:** **25/115 = 21.74%** (24 full-score tasks + 2 half-score tasks; OsmAndMarker had a setup error and was excluded).

The whole-task-correct rate on AC-High falls with task length:

| Steps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | ≥12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Whole task correct | 69/116 | 37/122 | 70/200 | 60/221 | 49/256 | 19/173 | 18/138 | 11/102 | 4/65 | 1/42 | 2/30 | 4/78 |
| % | 59 | 30 | 35 | 27 | 19 | 11.0 | 13.0 | 10.8 | 6.2 | 2.4 | 6.7 | 5.1 |

Pooling the tasks of 6 steps or more: 59/628 = 9.4%.

AndroidWorld falls with complexity in exactly the same way: 22/67 tasks when complexity < 1.5; 2.5/25 when 1.5–2.5; 0/12 when 2.5–4.5; 0.5/11 when higher. AC-Low (step-by-step instructions) reaches 89.3% per step but also only 932/1,543 = 60.4% whole-task.

**Conclusion:** the two benchmarks do not contradict each other. The "good AC-High" number is a per-step number. Measured by whole task, the model's ability is about 22% in both places.

The two numbers are not the same quantity, and the differences between them push in opposite directions:
- AW is **harder** in that errors accumulate, the model must stop by itself, must answer, and meets unfamiliar apps;
- AW is **easier** in that any path is accepted.

The two effects happen to offset each other. Use 22.3% as an indicator, not as a prediction.

## 3. The same error in both places: stopping at the wrong time

- **The AC test has no correct stop step.** Yet the model still chooses `terminate` at **1,171/8,444 steps (13.9%)**, that is, it wants to stop when the task is not finished.
- **AndroidWorld:**
  - **49 tasks declare done when not done**, plus 2 half-score tasks.
  - Conversely, **2 tasks had the device in the right state but the model did not stop**, so they score 0: `SystemBrightnessMinVerify` and `RecipeDeleteMultipleRecipesWithConstraint` (`task_successful=True`, `score=0`).
  - 40 tasks ran out of step budget.
- **First error on AC-High** (1,199 wrong tasks):
  - wrong action type chosen: 684;
  - right type but wrong target or value: 515, of which click on the wrong target 435;
  - stopped early: 154.

So stopping at the right time is the most lacking skill. Per-step AC-High cannot see it, but the number of `terminate` predictions on the AC test measures it without an emulator.

## 4. AMEX (arXiv 2407.17490): read the paper closely, checked against the file

1. **AMEX reports results per step only** (ScreenSpot, AitW, AC). The paper itself admits it could not run AndroidWorld because "SPHINX is not deployable", and writes that online evaluation is more accurate. So there is no evidence that AMEX raises the task completion rate.
2. **The useful part is the screen-understanding data (Level 1/2:** element boxes and functions), not the action sequences. Level 3 transfers poorly between the two collection methods: training on AMEX and testing on AitW reaches only 56.2, against 76.3 when training on AitW itself. The gain on AC is not consistent between the two versions:
   - paper: 49.8 → 61.7, but it needs 4× the data;
   - at equal sample count: 49.8 → 52.8;
   - figure in the README: 60.2 → 63.7.
3. **The paper says something different from the file.** The paper (§A.1.5) writes that annotators were allowed to make mistakes, so PRESS_BACK is frequent. The file: 135/38,709 steps = 0.35%, about 12 times less than the AC test (4.06%). 2,527/3,046 episodes open with a swipe at the launcher; there is no open-app action. AMEX cannot teach going back or stopping at the right time, and those are exactly our two main errors.

Single images can be downloaded by HTTP range (tried successfully), so there is no need to download all 93.5 GB. But for the reasons above, **AMEX is not used for this training round**.

## 5. Final plan (registered before the v5 result)

**Best measured so far:** v4 + harness V12 = **25/115 = 21.74%**. Against A (V11): 5 wins, 1 loss, p = 0.22.

1. **v5:** one combined training run, data as in the RUNCARD: chains of 4 episodes to learn stopping at the right time, back rows with their control rows, on/off display and `[id: …]`.
   - The first run ran out of memory (OOM) at step ~40. The cause was our own 0.42 limit hitting the ceiling on one long chain row, not another process.
   - Rerun with a 0.48 limit, saving every 250 steps.
2. **Gate 1 (offline, against v4, same file and same scorer).** If it does not pass, the final version uses v4.
   - (a) AC-High whole-task correct ≥ 344/1,543, and McNemar by episode is not worse.
   - (b) The number of `terminate` predictions on the AC test drops clearly from 1,171/8,444.
   - (c) Validation: the number of tasks right at every step does not fall.
   - (d) P(back | gold ≠ back) ≤ that of v4 + 0.02.
   - The AC test is used only as a diagnostic; the main benchmark remains AndroidWorld.
3. **Gate 2 (13 targeted AndroidWorld tasks, run in parallel):**
   - stopped early: NotesIsTodo, MarkorDeleteNewestNote, ExpenseDeleteMultiple, MarkorDeleteNote, RecipeDeleteMultipleRecipes, SimpleCalendarDeleteOneEvent;
   - did not stop or ran out of steps: SystemBrightnessMinVerify, RecipeDeleteMultipleRecipesWithConstraint, MarkorCreateNote;
   - keep the tasks that are currently right: SystemWifiTurnOn, RecipeDeleteSingleRecipe, ExpenseDeleteSingle, SimpleCalendarEventsOnDate.
4. **Final number:** run all 116 tasks with harness **V13** (V12 + `answer_format`, packaged and tested, md5 e4fb3991), no mask, on 6 emulators, paired with v4 + V12. Report the official SR in the points/N convention, together with wins/losses and the error groups.

No number is promised in advance. The number reported is the number measured.
