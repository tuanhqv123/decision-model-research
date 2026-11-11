# R3 unified plan — element choice, full trajectories, AndroidWorld-app data, answer, termination (one plan, no phases)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One d1-2B decision model (imitation learning, zero generated tokens) that, at every step, reads the current
screenshot + the previous trajectory (as text) + the current element tree, and chooses action / element / app /
direction / text span / answer — trained on all AndroidControl trajectories plus trajectories collected in the
AndroidWorld apps, and measured on the 116-task official protocol.

**Architecture:** The existing v2 pipeline (`aw_extract.py` → `rebuild_ac_v2.py` → `ac_common.py` contract →
`train_d1.py` / `eval_d1.py` → `serving/d1_api.py` → `benchmarks/aw/awenv/agent.py`) extended to contract v3
(`docs/research/CONTRACT-v3-io.md`), plus a rollout collector that runs the model on AndroidWorld instances whose
parameters differ from the evaluation instances and keeps checker-verified trajectories.

**Tech Stack:** Python 3.9 locally; torch on the training host (one 24 GB 4090); AndroidWorld @55ddaf5 emulator
containers; AndroidControl TFRecords.

**Spec:** `docs/superpowers/specs/2026-11-11-r3-elements-trajectories-design.md`; I/O: `docs/research/CONTRACT-v3-io.md`;
questions that test the reasoning: `docs/research/2026-11-11-R3-QUESTIONS.md`.

## Global Constraints

- Contract `v3`; `element_rule = cap40_reading`; image width 858; history = last 12 lines + count line.
- Per step the model receives exactly: goal, current screenshot, previous trajectory as text lines, current element
  list, launchable-app list. Never task metadata (no `complexity`, no step budget, no task name).
- Imitation only, no generated text. One model at inference.
- AndroidControl test episodes never enter training (leak-check against `splits.json`).
- **Test set** = the 116 official task instances (`run.py` default `task_random_seed = 30`) — never in training (test).
  **Practice instances** = the same 116 task templates filled with other random parameters by AndroidWorld's own
  generator, scored by its own checkers. **Decided 2026-11-11: allowed** (README: "an environment for building and
  benchmarking … millions of unique task variations"; paper §3: supports online learning and train/test generation;
  no rule forbids it) — disclosed in every report, and **one third of the templates (per app, fixed list committed
  before training) are never practised**, so a second number measures unseen templates.
- One renderer each for element list, history line, app options, answer options — shared by train, eval and serve.
- No online number before `docs/research/PRE-REG-2026-11-11-r3.md` is committed; claims need 3 seeds + paired test.
- Training-host disk: ≤ 10 GB of new files; raw extraction runs niced on the emulator host.

## Review Focus

1. Empty element list → spatial actions masked; still a valid action.
2. Same (action, element, effect) → identical history line from training rows, from rollouts, and at serving.
3. Every app option and every emitted action passes AndroidWorld's own `JSONAction` and `launch_app`.
4. A step with a missing screenshot or forest is dropped and counted, never shifted.
5. A rollout instance whose parameters equal an evaluation instance is refused.

---

### Task 1: All-steps AndroidControl v3 corpus
**Files:** `benchmarks/train/rebuild_ac_v2.py` (`extract-all`, `merge-all`); test `benchmarks/train/test_rebuild_v3.py`.
**Produces:** `compact_full_v3.jsonl` (every step of every train episode incl. terminal `status`), `compact_val_v3.jsonl`
(validation episodes, = the 827 v2 rows); v2 row schema + `split`.
- [ ] Failing tests: `test_all_steps_rows_match_v2_on_the_v2_rows`, `test_test_split_episodes_are_refused`,
  `test_missing_image_or_forest_drops_and_counts_the_step`. Run → FAIL.
- [ ] Implement; run on the emulator host (`nice … --workers 6`). Expected: row count reported (≈ 88k incl. status),
  val 827, leak-check line, 0 differential mismatches on the 14,965 v2 rows. Copy to training host, sha256 in EXEC. Commit.

### Task 1b: AndroidControl reduction (operator 2026-11-11: training data = AndroidControl + AndroidWorld practice only)
**Files:** `benchmarks/train/rebuild_ac_v2.py` (`--ac-reduce`); test in `benchmarks/train/test_rebuild_v3.py`.
- [ ] Rule: keep every episode in AndroidWorld-like apps and every non-click row; subsample click-only episodes to
  ~12k rows; deterministic; counts reported. Tests: `test_reduce_keeps_all_non_click_rows`, `test_reduce_is_deterministic`.

### Task 1c: Typed-text label = a span of the goal
**Files:** `ac_common.py` (`goal_span(goal, text)`); test `benchmarks/train/test_goal_span.py`.
- [ ] Rule: exact substring of the goal; else case-insensitive substring (label = the goal's own characters); else no
  text supervision for that row. Measured on AC cur15k: 358 exact + 84 case-insensitive = **442/449 (98.4%)**; 2 share
  only some words, 5 none. AndroidWorld: the text is in the goal for its typing templates except ~10 that need text from
  the screen or clipboard (transcribe, clipboard, received address, merge, from-image/from-file).
- [ ] Failing tests: `test_exact_span`, `test_case_insensitive_span_returns_goal_characters`, `test_no_span_no_label`.

### Task 2: History v3
**Files:** `ac_common.py` (`render_history_line_v3`, `render_history_v3(lines, k=12)`), `agent.py`; tests
`benchmarks/train/test_history_v3.py`, `benchmarks/aw/tests/test_agent.py`.
- [ ] Failing tests: `test_history_capped_to_last_k_with_count_line`, `test_same_line_from_gold_and_from_executed`,
  `test_no_coordinates_in_history`. → FAIL → implement → full suites green → commit.

### Task 3: App choice
**Files:** `ac_common.py` (`app_options`), `serving/d1_api.py`, `agent.py`, `benchmarks/aw/awenv/aw_apps.json`;
test `benchmarks/train/test_app_choice.py`.
- [ ] Build `aw_apps.json` inside the container: installed packages ∩ `_PATTERN_TO_ACTIVITY` (first alias + aliases).
- [ ] Failing tests: `test_gold_app_always_in_options`, `test_options_deterministic_per_row`,
  `test_every_option_resolves_in_androidworld` (CONTRACT §6.2), `test_intent_goal_gets_app_supervision`,
  `test_ac_app_names_canonicalised_by_aw_aliases`. → FAIL → implement (24 distractors in training) → commit.
- [ ] Done 2026-11-11: of 1,042 AC `open_app` rows, ~37 truly do not name the app (read row by row, data recipe §7.1).
- [ ] App labels for practice instances come from the template's own `app_names` (91/116 templates): first step of an
  intent-only goal → `open_app(<app>)`; any step whose foreground app is not in `app_names` → recovery row
  `open_app(<app>)`, capped at 2.5% of the mix. Test `test_recovery_label_uses_template_app_names`.

### Task 4: Answer head
**Files:** `ac_common.py` (action option `answer`; questions `answer_span` = span copy (source per spec D3: goal only, or goal + element labels), `answer_number` = choice `0..60`, `answer_kind` = choice `span|number`), `serving/d1_api.py`, `adapter.py`
(→ `{"action_type": "answer", "text": …}` followed by `status`); test `benchmarks/train/test_answer_head.py`.
- [ ] Failing tests: `test_answer_maps_to_aw_answer_then_status`, `test_number_answer_renders_as_integer_text`, and
  (only if spec D3 = yes) `test_span_answer_can_come_from_an_element_label`. → FAIL → implement → commit.
- [ ] Coverage report from `information_retrieval/proto/tasks.textproto`: answer types are STRING 16, NUMBER 8,
  DATE 1, TIME 1 (2026-11-11); list which can be expressed by span/number and which cannot.

### Task 5: Contract v3 end to end + conformance
**Files:** `ac_common.py` (`CONTRACTS["v3"]`, `gold_answers_v3`), `train_d1.py`, `eval_d1.py`, `serving/d1_api.py`,
`agent.py`, `adapter.py`, Gate A; tests extend `test_parity.py`, `benchmarks/aw/tests/test_mask_v2.py`.
- [ ] Failing tests: `test_serving_image_gets_the_same_jpeg_generations_as_training`,
  `test_every_v3_output_is_a_valid_aw_jsonaction`, `test_spatial_actions_masked_when_no_elements`,
  `test_v3_questions_have_choice_app_and_target_el_no_box`, `test_v3_serving_refuses_request_without_elements_or_apps`,
  `test_no_task_metadata_reaches_the_state`, `test_eval_reports_episode_all_right`. → FAIL → implement → green → commit.
- [ ] On the device: CONTRACT §6 checks 1–7 (incl. live element parity). All pass before any rollout or online run.

### Task 6: Rollout collector (AndroidWorld apps)
**Files:** create `benchmarks/aw/rollouts.py`, `benchmarks/aw/rollout_seeds.json`; test
`benchmarks/aw/tests/test_rollouts.py`.
**Produces:** `rollouts_v3.jsonl` in the v3 row schema (screenshot path, elements, executed action as gold, effect,
history, goal, `source=aw_rollout`, `instance_seed`, `success`, `premature_done_step`).
- [ ] Failing tests: `test_eval_instances_are_refused` (seed 30 and its 116 instance params),
  `test_rollout_rows_have_the_v3_schema`, `test_failed_rollout_marks_the_premature_status_step`. → FAIL → implement.
- [ ] Sources by operator decision D1: MiniWoB (92 tasks, own family) always; AW template instances with seeds from
  `rollout_seeds.json` only if D1 = yes; exploration of the 20 apps with goals from our own templates if D1 = no.
- [ ] Runs on both emulator hosts in parallel with `sweep_infra.py`; one emulator per host.

### Task 7: Termination signal and pairwise element loss
**Files:** `train_d1.py` (`--status-negatives`): for each failed rollout, an unlikelihood term on p(`status`) at the
step where it declared done; test `benchmarks/train/test_status_negatives.py`.
- [ ] Failing tests: `test_unlikelihood_only_on_marked_steps`, `test_no_effect_on_rows_without_mark`. → FAIL → implement → commit.
- [ ] `--el-pairwise`: V-Droid-style pairwise term on `target_el` (gold vs each other kept element). Tests:
  `test_pairwise_term_zero_when_gold_dominates`, `test_pairwise_off_by_default`. Recovery rows capped at 2.5% of the mix.

### Task 8: Sizing and smoke (training host)
- [ ] (Measured on v2 states: ~2,800 tokens, 11.3 GB, 1.6–1.8 samples/s — CONTRACT §8.) Token histogram of v3 states (p50/p95) at 858 px; 30-step overfit smoke on 64 rows (`target_el`, app, answer
  train acc > 0.9); throughput → hours per epoch; fall back to 720 only if 858 does not fit.

### Task 9: Offline AndroidWorld-format validation + data-mix decision
- [ ] Hold out rollouts by template (or by app for exploration data) → `aw_val_v3.jsonl`.
- [ ] One seed each, same steps: AndroidControl-reduced only, AndroidWorld practice only, AndroidControl-reduced + practice, and the best of these with `--el-pairwise`. Report step acc, element acc, app acc,
  premature-status rate, episodes-all-right on `aw_val_v3` and `compact_val_v3`. Pick the mix by `aw_val_v3`;
  record the answer to "does AndroidControl help or hurt?" (Questions doc, D-questions).

### Task 10: Self-training rounds (rejection fine-tuning)
- [ ] Round r: train on (chosen mix + successes so far) → roll out (Task 6) → add checker-verified successes and
  premature-status marks → retrain. Stop when `aw_val_v3` step acc gains < 1 point between rounds or after 3 rounds.

### Task 11: Pre-registration, 3 seeds, online, analysis
- [ ] `docs/research/PRE-REG-2026-11-11-r3.md` before any online run: R3 vs R1 paired on the 116 (seed 30), primary
  official SR on 116 and 104 with the 25 IR tasks reported separately, pooled sign test over 3 seeds, all seeds same
  direction; secondary: non-interactive clicks, unresolvable `open_app`, first-encounter repeat, premature done, IR
  answers right; disclosure of every AndroidWorld-app data source.
- [ ] Seeds 0–2 trained with the final recipe; online 116 per seed (chain with `sweep_infra.py`); frozen paired
  analysis; README, EXEC, AGENTS; push redacted to `tuan`.

### Task 12: Two past screenshots (operator, 2026-11-11 — after the first R3 model exists)
- [ ] Same recipe as the first R3 seed, input = current screenshot + the 2 previous screenshots (same width), history
  text unchanged. Rationale: HiconAgent (2512.01763, Table 6, Qwen2.5-VL-3B, AndroidControl) +8.96 step SR with 2 past
  observations at ~2.7x FLOPs; a single forward pass keeps the extra cost to image-encoding only (no decoding), so the
  latency stays far below an autoregressive agent (INFERRED; measured in this task). Compare on `aw_val_v3` first;
  online + 3 seeds only if offline gains ≥ the seed noise (~2 points).

### Data (operator, 2026-11-11, final)
- Training data = AndroidControl (reduced) + AndroidWorld practice rollouts. AMEX, AndroidLab, OS-Atlas were examined
  and dropped (data recipe §8–§9).

## Named risks

1. Attribution: R3 changes data, interface, history, app choice, answer, image size together — a win credits the package.
2. Template contamination if D1 = yes: report the number with and without AW-template data if both exist.
3. Rollout yield is low at today's ~15% success; MiniWoB and many instances are needed to get enough successes.
4. Compute (measured 2026-11-11): ~2,800 tokens/sample, 11.3 GB, 1.6–1.8 samples/s at 858 px → a 20–25k-row run
   ≈ 4–5 h; plus emulator hours for practice rollouts (~1–9 emulator-days for 0.5–3k trajectories).
5. Canvas/map targets have no element; ~10 AW templates need text that is not in the goal.
