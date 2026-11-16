# Plan: a pass@k gate for top-k exploration (2026-11-12)

> **DROPPED (2026-11-12).** Replaced by `2026-11-12-PLAN-v4-retrain.md`: optimising exploration before we know where the model is wrong is the wrong order.

This document makes §5 of `2026-11-11-PLAN-after-selfplay.md` concrete. Every figure has an accompanying file in
`benchmarks/results/2026-11-12-passk/` or `benchmarks/results/2026-11-11-v3/postmortem/`. The number in parentheses is the denominator.

## 0. The question this gate answers, and the decision that follows

**Question:** with the same v3 model (ckpt_1824), on the same task instance, if the model is allowed to try its own rank-2 and rank-3 choices, does it
find **additional** paths to the goal that choosing rank 1 does not find?

**Why ask before retraining:**
- Retraining of the RFT kind (AppVLM, UI-Voyager) has something to learn from only when exploration produces **new** successful paths.
- Rank-1 self-play gave only 1 new task out of 9 successful tasks (`selfplay_value.out`).
- If exploration also finds nothing more, then all later retraining effort is wasted. This gate costs only about 3 hours
  of emulator time and needs no retraining.

**Decision that follows the result:**
- **PASS:** collect full data on the practice templates, then train v4. This needs its own pre-registration.
- **FAIL:** drop the RFT direction. Switch to fixing the input: remove the 40-cut, the Progress row, learning to type text after clicking an input field.

## 1. Design (every choice has a measurable reason)

### 1.1 Task set: 30 tasks, seed 1001, no task where rank 1 ever touched the goal
The source is 54 self-play episodes on seed 1001 (`sp_tasks.txt`): 45 scored failed episodes.
- Drop 4 tasks that are scored only through `answer`. The model does not have this action, so the ceiling for those tasks is 0.
- 41 tasks remain. Keep the tasks with a budget ≤ 24 steps, giving 30 tasks.
- Drop **CameraTakePhoto** and **OpenAppTaskEval**. The scorer queried after each step shows that rank-1 **already touched the goal**
  and then spoiled it (probe = 1 at step 2 and at steps 1–3). These two tasks are no longer "not yet found".
- Replace them with two tasks that have a 34-step budget: **ExpenseDeleteMultiple2** and **RecipeDeleteDuplicateRecipes3**.

Result: **30 tasks, all 30 had not touched the goal with rank 1** on this instance.
- Budgets: 10 (×5), 12 (×6), 14 (×3), 16, 18 (×3), 20 (×5), 22 (×2), 24 (×3), 34 (×2). 7 tasks are above 20 steps.
- 7 tasks are in the held-out template list (`awenv/heldout_templates.json`): ClockTimerEntry, ExpenseDeleteDuplicates,
  MarkorDeleteNewestNote, RecipeDeleteDuplicateRecipes, BrowserMultiply, RecipeDeleteMultipleRecipes,
  ExpenseDeleteMultiple2. This gate only measures and does not train, so they may be used; every result will be reported split into the two groups.
- Seed 30 (the evaluation seed) is not touched.

### 1.2 Sampling method: top-3, two temperatures, never sample INTO `status`/`wait`
- **Action-type head:** sample among the 3 types with the highest probability, probabilities renormalised at temperature T.
- **Element:** if the chosen type is click / long_press / type_text, sample the element from the top-3 of `target_el`, with the same T.
- **Other arguments stay as rank 1:** the text to type, scroll direction, app name. This is a limit (§4, H11–H12).
- **Temperature:** attempts a1 and a2 use T = 0.3; a3 and a4 use T = 0.7.

The measurements this is based on: `sampling_policy.out` measured on 8,444 AC-High rows; `status_on_aw.out` measured on 730 AW self-play steps.

| T | Type accuracy, AC-High | Element accuracy, AC-High | AW: fraction of steps with a different type | AW: fraction of steps with a different element |
|---|---|---|---|---|
| 0 (rank 1) | 81.0% | 60.9% | 0% | 0% |
| 0.3 | 81.2% | 59.9% | 6.4% | 10.6% |
| 0.5 | 80.6% | 58.8% | 10.4% | 16.1% |
| 0.7 | 79.5% | 57.5% | 14.1% | 20.8% |
| 1.0 | 77.3% | 55.5% | 19.0% | 26.8% |

The accuracy columns are computed under the policy "never sample into status". The reasons for choosing as above:

- **Top-3, not free sampling.**
  - The correct answer is in the top-3: 97.8% for the action type, 73.9% for the element (`topk_actest.out`).
  - Free sampling at T = 1 reaches only 51.2% against 60.4% for rank 1.
- **Two values of T, not one.**
  - T = 0.3 loses almost no per-step accuracy (type 81.2%, element 59.9%). It still departs from rank 1 in about 6% of
    steps by type and 11% by element.
  - T = 0.7 departs about 2 times as often, at the cost of about 3 points of element accuracy.
  - We do not know which is better on AW. Running both gives more information for choosing T in the next data collection, with no
    extra run. Descriptive only; 30 tasks are not enough to conclude.
- **Do not sample INTO `status` (the selector calls it `terminate`), unless `status` is already rank 1.** Measured on AW:
  - At steps where the task is **not** done, if sampling into `status` is allowed the probability of a wrong finish rises from 3.92% to 4.71% (T 0.3)
    and 6.54% (T 0.7) **per step**. Blocking it leaves 3.51% and 3.07%.
  - Reporting done too early was already the largest error of v3.
  - The cost: at 15 steps where the task **is** done, the probability of reporting done falls from 60.0% to 57.6% (T 0.3) and 52.9% (T 0.7).
  - This damage is negligible: "reached the goal" is counted by the scorer queried after each step (§1.4), and does not need the model to
    report done itself.
- **Do not sample into `wait`.** `wait` does not change the screen so it explores nothing, it only spends step budget.
- **Sample first, mask after.** No-effect masking stays on as it was in the self-play baseline run. If the sampled draw falls on a pair
  (screen, action) verified to have no effect, masking replaces it as before.
  - Among the 100 steps masked during self-play there were **0** mismatches in the element list (`candidates_mismatch`). The way the element
    list is rebuilt is trustworthy.
- **Reproducible:** random numbers come from `sha256(seed | episode_id | step)`. Each step records the rank taken, the probability and the reason for
  falling back to rank 1 if any.

### 1.3 Five runs, 30 tasks each
| Run | Policy | Purpose |
|---|---|---|
| a0 | rank 1 (T = 0), re-run | **Noise control:** when tasks are chosen by "already failed", a re-run can also succeed because the environment is non-deterministic |
| a1, a2 | top-3, T = 0.3, exploration seeds 1 and 2 | light exploration |
| a3, a4 | top-3, T = 0.7, exploration seeds 3 and 4 | strong exploration |

- All 5 runs keep the following unchanged: checkpoint `v3_s0_ckpt1824`, `cap40_reading_noime`, contract v3, track VA,
  history v3, masking on, `record.state`, `probe_success_each_step`, `done_threshold` 0.
- Only the sampled path changes.

### 1.4 Metrics and gate rule (pre-registered, fixed before running)
**Reached the goal (reached):** at some step, the scorer queried after that step returns `probe_score ≥ 1`, or the episode ends in
success. This is the metric related to the data: the step that reaches the goal gives a successful path, and the step right after it gives the correct
`status` label.

**Strict success (success):** the model reports done by itself and the scorer confirms. Always reported alongside, but not used as the gate.

**Credit for exploration:** a run among a1–a4 that reaches the goal is counted only if **before the goal-reaching step there is at least one step**
whose executed action differs from rank 1 because of sampling. If it reaches the goal without deviating at any step, that is environment noise, not
thanks to exploration.

The quantities:
- `N_explore`: the number of tasks (out of 30) with at least 1 run among a1–a4 that reached the goal **and** earned credit.
- `N_noise`: the number of tasks where a0 reached the goal.
- `N_new`: the number of tasks counted in `N_explore` where a0 did **not** reach the goal.

**Gate rule:**
- **PASS** if `N_new ≥ 5`.
- **FAIL** if `N_new < 5`.

The threshold 5 comes from the 2026-11-11 plan and is kept unchanged, not adjusted after seeing the numbers. Reason: 5/30 = 17% of the currently failing tasks.
AppVLM's first RFT round raised 11.4 → 17.9 points, about 7.5 tasks out of 116. The figure 7.5 tasks is **our own conversion** of their number,
not a number they report. If exploration on 30 tasks cannot produce even 5 new tasks, a full data-collection round is unlikely to produce a gain of
that size.

**Validity of a run:**
- A run is not valid (the missing part must be re-run) if `setup_error` + `unscored` make up more than 20% of the episodes.
- It is also not valid if `candidates_mismatch` makes up more than 2% of the sampled spatial steps.
- Episodes set aside because an emulator died are re-run by task name, exactly as in earlier runs.

**Prediction made in advance (so that it can be wrong):**
- P1: the 7 tasks with a budget above 20 steps still **0/7 reach the goal** in every run.
- P2: `N_noise ≤ 1`. An earlier re-run (Q10) had 0/20 outcomes flipped.

**Always reported:**
- `N_new`, `N_explore`, `N_noise`, pass@1, pass@2 and pass@4 with Wilson intervals;
- split by level of T, by held-out template or not, and by budget;
- the number of strict successes;
- the actual deviation rate per step;
- for each new task, the first branching step compared with rank 1, since that is the input of branch-point learning (GRSD).

## 2. Implementation (TDD, one commit per task)

**Files:**
- `benchmarks/aw/awenv/explore.py` (new);
- `benchmarks/aw/awenv/agent.py`, `benchmarks/aw/run_bench.py` (modified);
- `benchmarks/aw/passk_gate.py` (new);
- `benchmarks/aw/tests/test_explore.py`, `test_passk_gate.py` (new), additions to `test_agent.py`, `test_cli.py`;
- 5 configs `benchmarks/aw/configs/aw30_passk_s1001_a{0..4}.yaml` and the task list `benchmarks/aw/configs/passk30_tasks.txt`.

### Task 1: `awenv/explore.py`
**Interface produced:**
- `explore_rng(seed: int, episode_id: str, step: int) -> random.Random`
- `sample_decision(dec, sent, width, height, *, top_k=3, temperature, rng, never_into=("status", "wait")) -> (dict, dict)`.
  Returns (the decision to be executed, the record
  `{"T", "k", "type_rank", "element_rank", "p_type", "p_element", "fallback"}`).

**Reuse** `masking.action_ranking`, `masking.element_points` and `masking._argument_decision`, so that exploration and masking build the
element list with exactly one function.

**Tests (written first, watch them go red):**
- `test_temperature_zero_returns_the_argmax_decision_unchanged`
- `test_type_is_sampled_only_from_the_top_k`: 5 types with probabilities, 3,000 draws, the set of results ⊆ top-3.
- `test_frequencies_follow_the_renormalised_tempered_probabilities`: probabilities 0.6/0.3/0.1, T = 1, 5,000 draws, tolerance
  ±0.03; at T = 0.5 this corresponds to squaring then normalising.
- `test_never_samples_into_status_or_wait_unless_it_is_the_argmax`: when rank 1 is `status` it is still kept or left out of
  `status`.
- `test_spatial_sample_takes_an_element_from_the_top_k_of_target_el_at_its_centre`: uses the sample screen of
  `test_mask_v2.py` under contract v3.
- `test_candidates_mismatch_falls_back_to_the_argmax_element_and_says_so`
- `test_type_whose_argument_is_missing_falls_back_and_says_so`: gets `open_app` with an empty app name.
- `test_same_seed_episode_and_step_give_the_same_draw`

### Task 2: wire into `AgentCore`
**Interface:** new parameter `explore: Optional[dict] = None`, containing `{"top_k", "temperature", "seed"}`.
- When exploration is on, always request readouts.
- Order: sample → masking → execute.
- `history_line` is generated from the **executed** action, as masking already does.
- `record["explore"]` stores the sampling record. `action_raw` is still what the model said.
- `episode_id` is taken from the runner. The runner must pass it in; if missing, use goal + step index, and say so in the record.

**Tests:**
- `test_explore_is_off_by_default_and_changes_nothing`: the record and the decision are identical to before.
- `test_explore_executes_and_records_the_sampled_action_and_history_names_it`
- `test_mask_still_applies_to_a_sampled_action`

### Task 3: config
**Interface:** key `policy.explore: {top_k, temperature, seed}`.
- `ConfigError` if `temperature < 0`, `top_k < 1`, or `suite.diagnostic: true` is missing. Purpose: an exploration run
  can never be misread as a benchmark number.
- The plan printout has an `explore` line.

**Tests:**
- `test_explore_requires_a_diagnostic_suite`
- `test_plan_carries_the_explore_settings`
- `test_explore_rejects_negative_temperature`

### Task 4: `passk_gate.py`, the pre-registered analysis
**Interface:** `python passk_gate.py --tasks passk30_tasks.txt --budgets aw116_budgets.json --heldout heldout_templates.json --a0 DIR --explore DIR... > gate.out`.
Prints every quantity in §1.4 and a line `VERDICT: PASS|FAIL`.

**Tests (synthetic episode data):**
- `test_reached_counts_a_probe_of_one_at_any_step`
- `test_a_success_with_no_sampled_deviation_before_the_goal_gets_no_credit`
- `test_n_new_excludes_tasks_the_argmax_rerun_also_reached`
- `test_verdict_is_pass_at_five_new_tasks_and_fail_at_four`
- `test_setup_errors_and_unscored_episodes_are_excluded_and_counted`
- `test_budget_prediction_is_reported_for_tasks_over_twenty_steps`

### Task 5: run
1. Run the whole test suite `pytest benchmarks/aw/tests -q`. The current baseline is 249 passed / 43 skipped; the new count must equal the baseline plus
   the number of new tests.
2. Package the harness as `awhV7`, the same way as awhV5/V6, and record the sha256.
3. Write 5 configs: copies of `aw116_selfplay_s1001.yaml` with `tasks` set to the 30 tasks, `diagnostic: true`, and `policy.explore`.
4. Run a trial of 1 task × 2 steps with T = 0.7. Check the following before the long run:
   - the record has `explore`;
   - there is at least one deviating step;
   - `/health` reports `v3_s0_ckpt1824` correctly.
5. Run `stage2_chain_par2.sh` with TARGET = 30:
   - emulator 1 runs a0 → a1 → a3;
   - emulator 2 runs a2 → a4.
   Each chain has its own watchdog, a sweep after each batch, and checks `/health` itself.
6. Analyse with `passk_gate.py`, then commit the result into `benchmarks/results/2026-11-12-passk/`. Tell you **only when
   done or when broken**.

**Estimate:**
- Each run is at most 518 steps; the actual number of steps is estimated at about 400 (failed tasks usually run to the end of the budget).
- 5 runs, about 2,000 steps.
- Self-play ran 730 steps in 1.22 hours. With 2 emulators, estimated **2–3.5 hours**, not counting emulator incidents.

**Not touched:** the customer's containers and processes, the customer's vLLM, files on the server other than `~/aw`, `~/awruns` and
`~/pm_tmp`.

## 3. Objections and how the design blocks them

| # | Objection | Answer / how it is blocked |
|---|---|---|
| H1 | Choosing tasks that "already failed" and re-running them will always give a few successes (regression to the mean) | a0 measures exactly that part; `N_new` removes the tasks a0 also solved |
| H2 | Reached the goal by luck, not by sampling | Credit is given only when there is a step that deviates from rank 1 **before** the goal-reaching step |
| H3 | "Reached the goal" is easier than "success" | True, and on purpose: the unit of learning data is the step. Strict success is always reported alongside |
| H4 | The T numbers come from AC-High, not AW | The deviation rate was also measured on AW (`status_on_aw.out`); AW has element deviations of 10.6% and 20.8% |
| H5 | Masking confuses the comparison | Masking is on in the self-play baseline and in all 5 runs; the only difference is sampling |
| H6 | Leakage into the evaluation tasks | Only seed 1001 is used; seed 30 is untouched; the gate learns nothing. The 7 held-out templates are reported separately |
| H7 | 30 tasks are too few to test | No claim of statistical significance. This is a decision rule fixed in advance, with Wilson intervals |
| H8 | The environment is non-deterministic so "same instance" is no longer the same | Q10: 6/20 paths differ but 0/20 outcomes flipped; a0 re-measures this on exactly these 30 tasks |
| H9 | Blocking sampling into `status` loses the correct done reports | It loses 2.4–7.1 points of probability of reporting done at done steps, in exchange for fewer wrong finishes at not-done steps. Reaching the goal is still recorded by the scorer |
| H10 | Dropping the 2 tasks that touched the goal is choosing the sample as we like | The dropping rule is written down ("rank 1 already touched the goal") and applied equally to all 41 tasks; only 2 tasks match |
| H11 | The text to type is not sampled | A real limit. Tasks that fail because of typing the wrong text will not be explored. If the gate fails, the "learn to type text" direction handles this part |
| H12 | The scroll direction is not sampled | A limit. On AC, scroll errors are mostly the wrong action type (16/19), not the wrong direction (3/19), per the 2026-11-04 record |
| H13 | A wrong sampling record gives wrong credit | Task 1 and Task 4 have tests for exactly this case; the trial run in Task 5 step 4 checks it before the long run |
| H14 | The two emulators differ | pass@k pools the runs so a difference between machines only dilutes the result; a0 runs on machine 1 |

## 4. Outside the scope of this gate
- No retraining. Seed 30 is not touched. `cap40` is not changed.
- Two cheap experiments (the Progress row, removing the 40-cut) will have their own plan. They need inference GPU on the training machine, not an
  emulator, so they can run in parallel and do not affect this gate.
- If PASS, the full data-collection plan needs its own pre-registration. It must state: the share of held-out templates (one
  third), T, k, and how to merge the data with the v3 data.
