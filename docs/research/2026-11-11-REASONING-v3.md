# Reasoning and counter-arguments for the v3 post-mortem (2026-11-11)

This document accompanies `2026-11-11-POSTMORTEM-v3.md`. It records **how** each conclusion was reached, the counter-questions
we raised ourselves from a different angle, the measurements that answer each one, and which gaps remain open. Every file mentioned is in
`benchmarks/results/2026-11-11-v3/postmortem/`.

## 0. Method
1. **One claim = one number + one file that can be rerun.** Without a file it can only be called a hypothesis.
2. **State the definition and the denominator.** For example, a "useless step" here is the definition that **the model sees**: the history line
   `→ no change`, computed by `ac_common.effect_changed`, identical to training time. It is not the harness's `ui_changed`.
3. **Every claim must survive at least one counter-question** that could refute it: a confounder, a different explanation,
   repetition on another run, a mechanism at a finer level.
4. **Grading after the counter-argument:** STANDS · STANDS WITH CONDITIONS · PARTLY REFUTED · REFUTED · UNTESTED.
5. **Observation is not experiment.** Every comparison on an existing run is an observation. Only a controlled intervention (change one
   thing, keep the rest unchanged) can prove a causal relation. Every open gap comes with an intervention experiment
   to close it.

---

## L0. Can the 14.0% number itself be trusted?
**Claim:** v3 reaches 16/114 = 14.0% on AndroidWorld under the official protocol (`aw_facts.out`, the repo's `official_sr` function).

| Counter-question | Answer in numbers |
|---|---|
| Does the harness have an error that lowers the score? | The five deviations from the original `suite_utils` (H1–H5) were fixed, each with a test quoting the original code line. Gate A passes 41/41, checking the consequence of each action on the emulator (`AGENTS.md`, log of 2026-11-10). |
| Is excluding tasks from the denominator within the rules? | 2 tasks excluded: MarkorMergeNotes (checker adb error) and SimpleSmsReplyMostRecent (initialisation error). Following the `NaN` rule of `suite_utils`, implemented in `awenv.metrics.official_sr`. |
| Is one seed enough? | Not enough to compare small differences. The standard error at 14% over 116 tasks is 3.2 points (95% ±6.3). Rerunning on the same machine: 0/20 results changed (Q10, 2026-11-10), so the noise comes mainly from a different seed or a different instance, not from rerunning. |

**Grade: STANDS** as a one-seed number. **Gap:** there are not yet ≥3 seeds.

---

## L1. "Learning on apps other than the test apps" (N1)
**Original claim:** 19/33 AW apps have no training rows; tasks in apps with training rows succeed more.

| # | Counter-question | Measurement | File |
|---|---|---|---|
| 1 | Do familiar apps just have shorter tasks? | In the group of tasks ≤10 steps it remains: 7/16 vs 4/32, p = 0.021. Group of 11–20 steps: **no difference** (1/7 vs 4/31) | `app_coverage.out` |
| 2 | Do the "free" tasks (the target state is already there from the start) inflate the familiar-app group? | The risk is real: an agent that does nothing (Q11) solves 7 tasks, **6/7 in Settings**. Removing those 7 tasks: v3 **7/19 (36.8%) vs 8/88 (9.1%), p = 0.005**. The effect remains | `floor_tasks.out`, `coverage_replication_v3.out` |
| 3 | Does it replicate in other independent runs? | On all tasks: R1 seed 0 36% vs 5.6% (p = 0.0003); R1 seed 1 40% vs 7.8% (p = 0.0003); B0′ 8% vs 6.7% (p = 0.55; B0′ is only level with the do-nothing agent). Removing the free tasks: R1 s0 15.8% vs 4.5% (p = 0.10), R1 s1 21.1% vs 6.7% (p = 0.07). **Same direction, 3–3.5 times, but not statistically significant** | `coverage_replication.out` |
| 4 | Was the "has training rows" threshold chosen after seeing the results? | The app with the fewest training rows is files (155 rows); every other app has 0. The thresholds >0 and ≥100 give the same split | `train_facts.out` |
| 5 | Mechanism: in unfamiliar apps, are clicks more useless? | **No.** Useless clicks in unlearned apps 240/662 = 36.3%, in familiar apps 45/104 = 43.3% (z = −1.38) | `challenge_aw.out` Q-D |
| 6 | Or are familiar apps simply apps with an easy interface (on/off in Settings)? | **Cannot be separated** with observational data: "app" and "task type" coincide | — |

**Grade: STANDS WITH CONDITIONS.** It is real at task level for v3 (p = 0.005 after removing free tasks), same direction in two
R1 runs. **Refuted:** the mechanism "clicks do not take effect in unfamiliar apps" (question 5).
**Open gap:** confounding between app and task-type difficulty (question 6); the real mechanism is not known.
**Experiment to close the gap (intervention):** add self-play data **only** for apps with 0 rows, keep everything else unchanged, measure on unlearned
seeds. If the 0-row app group does not rise, then N1 is not the cause.

---

## L2. "Does not know how to get out when it errs itself" (N2)
**Original claim:** 45% of steps are useless; 88% exact repeats after a useless step; only 2% of training rows have a useless previous step.

| # | Counter-question | Measurement | File |
|---|---|---|---|
| 1 | Do the two ways of counting "useless" (663 and 573) contradict each other? | No. 663 is the definition the model sees (history line) = 595 steps in scored tasks + 68 steps of MarkorMergeNotes. 573 is `ui_changed`, a different comparison in the harness | `challenge_aw.out` |
| 2 | Is every "no change" step useless? | **No.** The 595 steps consist of: `wait` 151 (standing still is normal), scroll 155, click 285 (interactive element 191, text input field 71, non-interactive element 23), open_app 4. **"45% useless" is exaggerated**, because it counts `wait` | `challenge_aw.out` Q-A |
| 3 | Is repeating sometimes reasonable? | 469 exact repeats (scored tasks): `wait` 131, of which in 37 the next time had an effect (waiting for loading is reasonable). click 196, only 8 had an effect. scroll 142, **0** had an effect. Certainly wasted repeats: **≥ 330** (click 188 + scroll 142) | `challenge_aw.out` Q-E |
| 4 | What is the most specific error? | After a click on a text input field where the screen does not change, the model **clicks again 57 times, types only 11 times**. In the training data, after a click on an input field, the answer is to type 82.5% of the time (577/699) | `challenge_aw.out`, `train_facts.out` |
| 5 | Is missing error-correction data the cause? | Only a **correlation** so far: 2.0% of training rows have a useless previous step. Another explanation: the model decides deterministically, the screen does not change and the history gains only one line, so the decision repeats | `train_facts.out` |
| 6 | Is blocking repeats enough? | Full 112 paired tasks: no block 16/112, block 19/112 (4 wins, 1 loss, sign test p = 0.375). **3 of the 4 wins are "free" tasks** (an agent that does nothing also solves them). Blocking repeats does reduce repeats (86.5% → 62.2%), but tasks with wrong done-reports rise from 49 to 56: the loop turned into early done-reports. Repeating is a symptom | `mask_paired_full.out` |

**Grade:** the symptom **STANDS**, with corrected numbers: ≥330 wasted repeats, the error "click again instead of typing" 57/68. The cause
"missing error-correction data" is **UNTESTED**.
**Experiment to close the gap:** add "do not choose" rows (from self-play) and history rows containing errors, keep the rest unchanged. Measure the
repeat rate after a useless step and the rate of "type after clicking an input field" on the dev set. Drop it if neither changes.

---

## L3. "The done label is tied to position" (N3)
**Original claim:** the `status` label sits only at the last step of the record, so the model learns to report done by position.

| # | Counter-question | Measurement | File |
|---|---|---|---|
| 1 | The truth about the label? | 4,197/4,197 `status` labels at the last step, 0 elsewhere. Added by `preprocess_ac.py:5,134` | `train_facts.out` |
| 2 | Does the model report done at a step like the length of the demonstration record? | Wrong done-reports: median step 6 (IQR 5–9). `status` label in training: median 6 (IQR 4–8). **They coincide** | `challenge_aw.out` Q-B, `train_status_position.out` |
| 3 | If the model relied only on position, the done-report step should **not** change with task length. Does it change? | **It changes:** median 5 (tasks ≤10 steps) → 7 (11–20) → 8 (>20); Spearman 0.46 (n = 49). So the model **does** read the state partly | `challenge_aw.out` Q-C |
| 4 | What about correct done-reports? | Successful tasks also report at median step 6, so the done-report step cannot tell correct from wrong | `challenge_aw.out` Q-B |
| 5 | Is there a better label? | Yes, already collected: the checker answers at 395/395 self-play steps. 18 early done-reports and 2 late ones have been labelled | `selfplay_facts.out` |

**Grade:** the truth about the label **STANDS** (exactly true). "The model reports done only by position": **PARTLY REFUTED**, because the
done-report step does rise with task length. What stands: on long tasks, the model reports done too early (median step 8 for tasks with a
budget >20 steps), and the current labels have no "not yet done" example on screens that look like an ending screen.
**Open gap:** whether the `Progress: step n` line in the input is the position signal the model relies on.
**Experiment to close the gap:** use exactly the logged self-play input (`state_in`), change only the step number in the `Progress` line, and measure
how much the `status` probability changes. No retraining needed.

---

## L4. "Cutting at 40 elements does harm" (N4)
| # | Counter-question | Measurement | File |
|---|---|---|---|
| 1 | Does the model receive the full list? | Limit 128,000 tokens; with no cut, p99 is 2,659, max 6,456 tokens | `cap_tokens.out` |
| 2 | What is the number 40 based on? | An observational comparison on 2026-11-07 on a different model, not an experiment | `2026-11-07-list-length-is-causal.md` |
| 3 | Does cutting really happen on AW? | 454/1,541 steps = 29.5%; `New Event` is cut 155/155 times. The rebuilt list matches the model's real choice 899/903 | `aw_cut.out` |
| 4 | Is 0/17 Simple Calendar due to the cut? | **Cannot be separated:** this app also has 0 training rows (N1) and 8/17 tasks need `answer` (N5) | — |
| 5 | Does removing the cut slow things down? | Latency not yet measured; the token count at p99 rises from 1,706 to 2,659 | — |
| 6 | Can v3 use the 41st option onward? | Not known: it never had them at training time | — |

**Grade:** "40 is not a limit of the model" **STANDS**; "the cut causes 0/17" is **UNTESTED**.
**Experiment:** T1(a): same checkpoint, 656 AC-High rows with >40 candidates, compare no cut with cut at 40, measure accuracy and
latency. Then the 9 Simple Calendar tasks that do not need `answer`.

---

## L5. "Missing `answer`" (N5)
Pure arithmetic: 25/116 tasks are scored only through `answer`, v3 gets 0/25, score ceiling 78.4%. **Counter-argument:** adding `answer` would
not necessarily solve the 25 tasks, because being able to express it does not mean being able to solve it. 25 is a **ceiling**, not an expected gain.
**Grade: STANDS** (as a ceiling).

---

## L6. Solutions
Every gain is a **hypothesis**. A number from a paper is the number **of theirs, on their benchmark**. Example: DigiRL, arXiv 2406.11896,
on AitW: SFT 17.7% → retraining on filtered self-collected trajectories 57.8% → RL 67.2%. That number only shows the direction has succeeded
elsewhere; it does not predict our number. Each solution has a condition to drop it in post-mortem §4.

---

## 7. What the counter-arguments changed compared with the earlier post-mortem
| Before | After the counter-argument |
|---|---|
| "45% of steps are useless" | Exaggerated, because it counts `wait`. More accurate: ≥330 certainly wasted repeats, plus the error "click again instead of typing" 57/68 |
| "The model learns to report done by position" | Partly refuted: the done-report step rises with task length (ρ = 0.46). What stands: it reports too early on long tasks |
| "In unfamiliar apps clicks do not take effect" (implicit) | Refuted at step level (36.3% vs 43.3%) |
| "Learning on other apps is the cause" | Conditional: stands at task level for v3 (p = 0.005 after removing free tasks); on R1 same direction but not statistically significant; not yet separated from task-type difficulty |

## 8. Open gaps, ordered by cost of closing
1. **Is the `Progress` line the position signal?** Intervention on the logged input. No retraining, a few minutes of GPU.
2. **Remove the cut at 40 on v3** (T1a). No retraining, about 1 GPU hour.
3. **One seed:** run v3 for 2 more seeds (2 emulators, about 3 hours per seed).
4. **App or task-type difficulty?** Intervention with self-play data for the 0-row apps only. Needs retraining.
5. **Is missing error-correction data the cause of repeating?** Intervention with "do not choose" rows and history containing errors. Needs retraining.
