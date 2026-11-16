# Re-analysis after the v3 round (2026-11-11): where it went wrong, why, how to fix it. Every claim has a number and a file

**Reasoning and counter-arguments for each conclusion:** `2026-11-11-REASONING-v3.md`. It contains counter-arguments that rejected part of a conclusion below.

**Rule of this document:** every "measured" line has a file in `benchmarks/results/2026-11-11-v3/postmortem/`
(script + output), and rerunning it gives exactly that number. A line that is knowledge from a paper names the source and says clearly that it is the number **of
theirs, on their benchmark**. A line that is a prediction is marked **hypothesis**, with how to measure it and the condition to drop it.
A number that could not be recomputed has been withdrawn (see §6).

## 1. Summary (measured numbers only)
d1-mobile v3 chooses **step by step**: AC-High 70.6% (without `status`) / 67.1% (standard), 8,444 steps. **Completing a whole task by itself**:
AndroidWorld **14.0%** (16/114, `aw_facts.out`). The five causes below all have measured numbers. Causes 1 and 3 have been
tested statistically or are exactly true on the data (4,197/4,197). The other causes are only observations, not
controlled experiments; that is work still to be done.

## 2. Measured facts

| # | Measurement | Number | File |
|---|---|---|---|
| F1 | Official AndroidWorld SR (the repo's `official_sr` function; MarkorMergeNotes excluded because the checker errors, SimpleSmsReplyMostRecent because of an initialisation error) | **16/114 = 14.0%**; 16/116 Wilson 95% 8.7–21.2 | `aw_facts.out` |
| F2 | Success by task length (step budget) | ≤10: 11/48 · 11–20: 5/39 · 21–30: 0/8 · >30: 0/21 | `aw_facts.out` |
| F3 | Failed tasks: reported done when not done / did not report, ran out of steps | 49 / 51 (of 100) | `aw_facts.out` |
| F4 | Steps where the history line (which the model sees) says "no change" | 663/1,476 = 44.9%; **includes `wait`**. In scored tasks: 595 = wait 151 + scroll 155 + click 285 + open_app 4 | `aw_facts.out`, `challenge_aw.out` |
| F5 | Right after a useless step, the model does exactly the same again | 534/609 = 87.7%. Repeats that are **certainly wasted** (the next time also has no change): click 188 + scroll 142 = **≥330**; `wait` repeated 131, of which 37 are reasonable. After a click on an input field with no change: click again 57, type 11 | `aw_facts.out`, `challenge_aw.out` |
| F6 | Failed tasks with one action repeated ≥5 times in a row | 46/100 | `aw_facts.out` |
| F7 | Tasks scored only through `answer` | 0/25 | `aw_facts.out` |
| F8 | AndroidWorld apps with **0 training rows** | **19/33**; training rows in apps outside AW: 18,788/27,454 | `train_facts.out` |
| F9 | Success by app coverage (whether the task's main app has training rows) | yes: **8/25 = 32.0%** · no: **8/89 = 9.0%** · one-sided Fisher **p = 0.0072** | `app_coverage.out` |
| F9b | The same split, within each length group | ≤10 steps: 7/16 vs 4/32 (p = 0.021) · 11–20: 1/7 vs 4/31 · >20: 0/2 vs 0/26 | `app_coverage.out` |
| F10 | Training rows right after a useless step | 539/27,454 = 2.0% (answers in those rows: click 261, status 122, wait 62, scroll 54, …) | `train_facts.out` |
| F11 | The `status` row sits at the **last step** of the record | **4,197/4,197**; rows that are not the last step but carry the label `status`: **0** | `train_facts.out` |
| F12 | After tapping an input field, the next answer is `type_text` | 577/699 = 82.5% | `train_facts.out` |
| F13 | AW steps with >40 candidates (cut) | **454/1,541 = 29.5%**; `New Event` on a 155-step screen, cut **155/155**, in all 17 Simple Calendar tasks (v3: 0/17) | `aw_cut.out` (the rebuilt list matches the real choice 899/903) |
| F14 | Training rows with >40 candidates | 740/6,863 samples = 10.8% | `cap_tokens.out` |
| F15 | Text length with **no cut** (state + options) / model limit | median 775, p99 2,659, max 6,456 tokens / **128,000** | `cap_tokens.out` |
| F16 | Keeping the answer on screens with >40 candidates: cut in reading order (R0) / R1 / R2 | train 1,285 / 1,427 / **1,465** of 1,697 · val 59 / 67 / 68 of 80 · AC-High 487 / **547** / 543 of 656 | `rule_retention.out` |
| F17 | Forbid repeats at run time (mask), 112 paired tasks | v3 16/112 · mask 19/112 (4 wins, 1 loss, p = 0.375; 3 of the 4 wins are "free" tasks); identical repeats 86.5% → 62.2%; wrong done-reports 49 → 56 | `mask_paired_full.out` |
| F18 | Self-play: the checker answers after **every** step | 395/395 steps, 0 errors; 8 tasks reached the done state (reported correctly at the next step: 5, reported late: 2, not reported: 1); **18** tasks reported done when the checker said not done | `selfplay_facts.out` (37 tasks, still running) |
| F19 | Model speed per decision | p50 139.5 ms, p95 174.6 ms | `aw_facts.out` |

## 3. Causes: each with numbers, and the level of certainty

### N1. Learning on apps other than the test apps (most certain: it has a test)
19/33 AW apps have no training row (F8). Tasks in an app that has data succeed 32% versus 9% (p = 0.007, F9).
**Counter-argument raised by ourselves:** familiar apps (Settings, Clock, Contacts) have shorter tasks, so the difference may come from task length. In the group of
tasks ≤10 steps, the difference remains: 7/16 (44%) versus 4/32 (12.5%), p = 0.021 (F9b). In the 11–20 step group there is **no visible
difference** (14% versus 13%). Conclusion that is defended: missing app data breaks even short tasks. Conclusion that is **not** defended:
that adding app data alone is enough for long tasks.

### N2. Imitation learning only with always-correct history, so it does not know how to get out when it errs itself (observation, no control yet)
45% of steps do not change the screen (F4). After each such step, in 88% of cases the model does exactly the same again (F5). Meanwhile, only 2% of training
rows have a useless previous step (F10). Tasks longer than 20 steps: 0/29 (F2).
Theory (Ross & Bagnell, DAgger, 2011): imitation learning only sees the demonstrator's states, so errors compound with task
length. The measurements **fit** this theory, but we have **not** run an experiment that isolates it.
One shortcut has been ruled out with numbers: forbidding repeats at run time does not raise SR meaningfully: 16 → 19/112, p = 0.375, 3 of the 4 wins are "free" tasks (F17).

### N3. The "report done" label is tied to position (certain: exactly true on the data)
4,197/4,197 `status` rows are the last step of the record, 0 rows elsewhere (F11). The original AndroidControl has no terminating action,
we added it ourselves (`benchmarks/train/preprocess_ac.py:5` and `:134`: one `status` step per record, placed on the screen after the last action). So the "done" label carries only one signal: position in the demonstration record. **Counter-argument (see REASONING L3):** the model does **not** rely only on position. The step at which it reports done rises with task length (Spearman 0.46), but on long tasks it still reports too early (median step 8 for tasks with a budget >20 steps).
Measured consequence: 49 tasks reported done when not done (F3); on AC-High the model predicts `status` 539 times, although the benchmark has no
terminating step (`RESULTS.md` §3). The experiment of 2026-11-10 confirmed the same mechanism: the `is_done` label coincides 137/137 with
`action == status` (`benchmarks/results/2026-11-10-episodic-s0-fails-prereg.md`).
A replacement label already exists on the machine (F18): the checker answers at 395/395 steps.

### N4. Cutting the list at 40: not a limit of the model, and never tested on v3
The model accepts 128,000 tokens; with no cut, p99 is only 2,659 tokens (F15). The only basis for the number 40 is an **observational comparison**
on 2026-11-07 on a different model version (`benchmarks/results/2026-11-07-list-length-is-causal.md`). That is not an
experiment: screens with many elements are harder to begin with.
The measured cost: 29.5% of AW steps are cut, `New Event` is lost 155/155 times, 0/17 Simple Calendar tasks (F13); 10.8% of training rows
are partly hidden (F14).
**Limit of the conclusion:** 0/17 Simple Calendar also coincides with N1 (the app has no training rows) and with N5 (8/17 tasks need
`answer`). The cut alone has not been separated out; that is experiment T1 in §4.

### N5. Missing `answer` action
25/116 tasks are scored only through `answer`: 0/25 (F7). The current score ceiling is 91/116 = 78.4%.

## 4. Solutions. The "effectiveness" parts are all **hypotheses**, with a measured ceiling and a way to drop them

| # | Work | Fixes | Measured ceiling (at most how many tasks) | Supporting evidence | Experiment / condition to drop |
|---|---|---|---|---|---|
| T1 | **Remove the cut at 40** (or loosen it), try offline first | N4 | 9 Simple Calendar tasks that do not need `answer` (+7.9 points if all are done) | F13, F15 | (a) v3, same checkpoint, 656 AC-High rows with >40 candidates: no cut vs cut at 40. (b) If (a) does not drop: run the 17 Simple Calendar tasks. Drop if (a) drops against cut at 40. Fallback: R2 (F16: 1,465 vs 1,285 on train) |
| T2 | **Self-play on exactly the test apps** (seed ≠ 30), keep the successful tasks then retrain, several rounds | N1, N2 | 89 tasks in apps with no training rows | F9; **DigiRL (arXiv 2406.11896), on AitW, not AW:** imitation learning (SFT) 17.7% → retraining on filtered self-collected trajectories 57.8% → RL 67.2% | Drop if the AW-practice dev set does not rise ≥5 points after round 1 |
| T3 | **"Already done" label from the checker** at every step | N3 | 49 tasks reporting wrongly | F11, F18 (18 early reports and 2 late reports could be labelled) | The number of done-reports when not done on dev must fall |
| T4 | **"Do not choose"** for useless steps (unlikelihood loss) | N2 | 46 tasks repeating ≥5 times | F5, F10 | The repeat rate after a useless step (87.7%) must fall on dev |
| T5 | Add the `answer` action | N5 | 25 tasks | F7 | 25 information-retrieval tasks |
| T6 | RL with reward from the checker, after T2–T5 | N2 | — | MAI-UI Table 9 (`references/raw/maiui2026-androidworld-table9.txt`): 2B SFT 45.1 → RL 49.1 (+4.0), their harness | Only do it when SR on dev is ≥ ~30% |

**Measured to give no meaningful help:** forbidding repeats at run time (F17, all 112 tasks). **Not to be done again because they touch none of the causes above:** adding
data or epochs from AndroidControl alone, tuning for AC-High, v3i (AC-Low).

## 5. Order
1. T1(a): offline, no retraining, cheapest.
2. Create the AW-practice dev set (its own seed, never learned from) as the yardstick for every later change.
3. Round 1 consists of T2 + T3 + T4, learn v4, measure on dev, then measure the 116 tasks with ≥3 seeds (one seed on 116 tasks has a standard error of ~3.2
   points, 95% interval ±6.3).
4. T5, then T6.

## 6. Where the previous version (commit `8c0ceeb7`) was wrong or had no evidence, now corrected
| Previous version | Truth | Why it was wrong |
|---|---|---|
| Cut steps: 201/1,541 = 13% | **454/1,541 = 29.5%** | the old way of counting could not be found again; the new number was checked against the model's real choices (899/903) |
| After tapping an input field, typing 536/619 = 87% | **577/699 = 82.5%** | the old definition could not be reproduced; the new definition is in the script |
| "First step opens the right app 98/115" | Only provable: in 115/116 tasks, the first action is `open_app` | there is no rerunnable definition of "right", so it was withdrawn |
| "Try 9 cut tasks" | All **17** Simple Calendar tasks are cut (9 do not need `answer`) | information-retrieval tasks were undercounted |
| "14.0% versus 45–56% of MAI-UI-2B / UI-Venus-1.5-2B" | Their numbers use a different harness and state neither the number of runs nor the step limit (`outputs/2026-11-11-published-ac-aw-metrics.md` §2) | M4: do not compare numbers from different harnesses. Use only as context, do not rank |
| "Successful tasks concentrate in familiar apps" (no number) | 32% vs 9%, p = 0.007; still true for tasks ≤10 steps, not visible for tasks of 11–20 steps | not measured before |
| "Forbidding repeats does not save it" (one half) | True on half A (12/56 = 12/56); half B not finished | the scope has been stated clearly |
| "DigiRL raises it strongly" (no number) | 17.7 → 57.8 (filter then retrain) → 67.2 (RL), on AitW | the numbers had not been quoted before |
