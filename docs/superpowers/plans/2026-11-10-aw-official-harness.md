# AndroidWorld Official-Protocol Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (operator chose inline execution, no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make our AndroidWorld harness reproduce upstream `suite_utils` exactly (H1–H5 of the audit), add the two measurements the audit says are missing, and produce the first AndroidWorld number on the official seed-30 instances.

**Architecture:** Fix the deviations where they live (`runner.py`, `run_bench.py`, `awenv_bridge.py`, `metrics.py`), each pinned by a test that encodes the upstream behaviour with a file:line reference. No model, training, or data change in this plan — those wait for the baseline it produces (decision K4).

**Tech Stack:** Python 3, pytest, existing `benchmarks/aw/awenv` package; upstream reference `google-research/android_world@55ddaf5` (local clone in the session scratchpad `dl/aw`).

**Spec:** `notes/decision-2026-11-10-androidworld-primary.md` + `docs/research/2026-11-10-AUDIT-androidworld-contract.md` (§8 H1–H9).

## Global Constraints

- Upstream reference: `android_world@55ddaf5`. Quote `file:line` in every comment that encodes an upstream rule.
- Suite seed `30`, `n_task_combinations = 1`, instance index `0`.
- Instance seed = `int(hashlib.sha256(f"{seed}_{name}_{i}".encode()).hexdigest(), 16) % 2**32` (`suite_utils.py:149-153`).
- Score is a float; per-episode official score = `task_score if declared_done else 0.0` (`suite_utils.py:261`).
- Official SR = mean over tasks of per-task mean score, skipping tasks that raised (`NaN`, `suite_utils.py:544-558, 684-716`).
- Budget = `int(10 * task.complexity)` read from the **instance** (`suite_utils.py:459, 491-502`; `information_retrieval.py:70`).
- Order: `initialize_task(env)` then reset with `go_home=True` (`suite_utils.py:246-247`, `episode_runner.py:79`, `task_eval.py:40`).
- No change to the model, the served checkpoint, the training code, or `ac_common.py` in this plan.
- Tests run with: `cd benchmarks/aw && python -m pytest tests -q`. The full suite must stay green (baseline: 148 passed, 43 skipped per AGENTS §3.5).

## Review Focus

1. A checker returning `None` or a non-number → must be `unscored` with a `checker_error`, never `score 0.0` (pinned in Task 1).
2. A `--replay-params` run must keep replaying recorded params and must NOT re-seed over them (pinned in Task 2).
3. A task object without a numeric `complexity` (our fakes, MiniWoB) → falls back to the configured table/flat budget (pinned in Task 2).
4. `elements()` called before any `screenshot()` in an episode → must still return the elements of one stabilised state, not raise (pinned in Task 3).
5. Old episode files with only `success` (no `score`) → `official_sr` must treat them as `score = 1.0 if success else 0.0` so `refresh`/reports on old runs do not crash (pinned in Task 4).

---

### Task 1: Runner — float score and initialise-then-home order (H1, H3)

**Files:**
- Modify: `benchmarks/aw/awenv/runner.py:182-252` (`_run_episode`)
- Test: `benchmarks/aw/tests/test_runner.py`

**Interfaces:**
- Produces: result record gains `"score": float` (official per-episode score) and `"task_score": float | None` (raw checker value). `"success"` becomes `score > 0.5` (upstream's success print, `suite_utils.py:263`). `"task_successful"` stays, = `task_score is not None and task_score > 0.5`.

- [ ] **Step 1: Write the failing tests** in `test_runner.py`:
  - `test_partial_credit_is_a_float_not_a_success`: FakeTask `is_successful` returns `0.5`, agent declares done → `score == 0.5`, `success is False`, `task_score == 0.5`.
  - `test_correct_world_without_declared_done_scores_zero`: checker `1.0`, agent never declares done within budget → `score == 0.0`, `task_score == 1.0`.
  - `test_non_numeric_checker_is_unscored`: checker returns `None` → `unscored is True`, `checker_error` mentions `None`, `score` is `None`.
  - `test_initialize_happens_before_go_home`: FakeEnv and FakeTask append to a shared `calls` list → `calls[:2] == ["initialize", "reset"]`.
- [ ] **Step 2: Run** `cd benchmarks/aw && python -m pytest tests/test_runner.py -q -k "partial or declared_done_scores or non_numeric or before_go_home"` → 4 FAIL.
- [ ] **Step 3: Implement.** In `_run_episode`: call `_init_task` first (same setup-error handling), then the existing 3-attempt `env.reset()` loop; read `raw = task.is_successful(env)`, accept `int|float` (not `bool` subclass issues: `float(raw)`), else record `checker_error = "non-numeric checker result: %r" % raw`; `score = task_score if declared else 0.0`, `None` when unscored/setup_error/env_crash. Update the `timeout` expression to use `score` (timeout only when `score` is not None and `score == 0` and not declared).
- [ ] **Step 4: Run** the full `python -m pytest tests -q` → all pass; fix any existing test that asserted the old order or `bool` semantics by updating it to the upstream rule (cite the line in the test docstring).
- [ ] **Step 5: Commit** `git add benchmarks/aw/awenv/runner.py benchmarks/aw/tests/test_runner.py && git commit -m "aw harness: float score + initialise-then-home, as suite_utils (H1, H3)"`

### Task 2: Seeded instances and instance budget (H2, H4)

**Files:**
- Modify: `benchmarks/aw/run_bench.py:248-257` (`task_factory`)
- Modify: `benchmarks/aw/awenv/runner.py:128-131` (budget resolution)
- Test: `benchmarks/aw/tests/test_cli.py`, `benchmarks/aw/tests/test_runner.py`

**Interfaces:**
- Produces (in `run_bench.py`): `instance_seed(seed: int, name: str, i: int) -> int` and `instantiate_task(task_class, name: str, seed: int, i: int = 0, replay: dict | None = None) -> Any` (sets `task.aw_params`; params carry `"seed": instance_seed(...)` like `suite_utils.py:102`).
- Produces (in `runner.py`): `resolve_budget(task, fallback: int) -> int` = `int(10 * task.complexity)` when `complexity` is an `int`/`float` on the instance, else `fallback`.

- [ ] **Step 1: Write the failing tests:**
  - `test_instance_seed_matches_upstream_formula`: `instance_seed(30, "ContactsAddContact", 0) == int(hashlib.sha256(b"30_ContactsAddContact_0").hexdigest(), 16) % 2**32`.
  - `test_instantiate_is_deterministic`: a fake class whose `generate_random_params` returns `{"x": random.random()}` → two `instantiate_task(cls, "T", 30)` calls give equal `aw_params["x"]`; `aw_params["seed"] == instance_seed(30, "T", 0)`.
  - `test_replay_params_are_not_reseeded`: `replay={"x": 7}` → `aw_params["x"] == 7` and `generate_random_params` is not called.
  - `test_budget_comes_from_the_instance`: FakeTask with `complexity = 2.0` → episode header `step_budget == 20`; FakeTask without `complexity` → header uses the configured fallback.
- [ ] **Step 2: Run** `python -m pytest tests -q -k "instance_seed or deterministic or reseeded or budget_comes"` → FAIL.
- [ ] **Step 3: Implement** `instance_seed`, `instantiate_task` (`random.seed(instance_seed(...))` immediately before `generate_random_params()`), call it from `task_factory` with the episode's `spec.seed` and `spec.task_idx`; delete the "takes no seed in this version" comment and replace it with the upstream reference. In `Runner`, create the task before choosing the budget and use `resolve_budget(task, budget_of(name) if budget_of else step_budget)`; write the resolved value into the header.
- [ ] **Step 4: Run** full suite → pass.
- [ ] **Step 5: Commit** `git commit -am "aw harness: official seed-30 instances and instance complexity budget (H2, H4)"`

### Task 3: One observation per step (H5)

**Files:**
- Modify: `benchmarks/aw/awenv/awenv_bridge.py:113-124`
- Test: `benchmarks/aw/tests/test_bridge_observation.py` (new)

**Interfaces:**
- Produces: `AndroidWorldEnv.screenshot()` takes ONE stabilised `get_state(wait_to_stabilize=True)` and caches it as `self._obs`; `AndroidWorldEnv.elements()` returns `_element_dicts(self._obs)`, taking a stabilised state first only if `self._obs is None`. `execute`/`reset` set `self._obs = None`.

- [ ] **Step 1: Write the failing tests** with a fake inner env whose `get_state` returns a new state object (distinct `ui_elements`) on every call and counts calls:
  - `test_screenshot_and_elements_describe_the_same_state`: after `screenshot()`, `elements()` equals `_element_dicts(<that same state>)` and `get_state` was called exactly once.
  - `test_elements_before_screenshot_takes_one_stable_state`: fresh bridge, `elements()` → one call with `wait_to_stabilize=True`.
  - `test_action_invalidates_the_cached_observation`: `screenshot()`, `execute(...)`, `screenshot()` → two distinct states.
- [ ] **Step 2: Run** `python -m pytest tests/test_bridge_observation.py -q` → FAIL.
- [ ] **Step 3: Implement** the cache as specified; keep `_last_state` behaviour for `__getattr__` consumers.
- [ ] **Step 4: Run** full suite → pass.
- [ ] **Step 5: Commit** `git commit -am "aw bridge: screenshot and element list from one State (H5)"`

### Task 4: Official success rate in metrics and report

**Files:**
- Modify: `benchmarks/aw/awenv/metrics.py` (new `official_sr`), `benchmarks/aw/awenv/report.py` (print it first)
- Test: `benchmarks/aw/tests/test_metrics.py`

**Interfaces:**
- Produces: `official_sr(episodes) -> {"sr": float | None, "n_tasks_scored": int, "n_tasks_excluded": int, "excluded": [task_name], "per_task": {task_name: float}}`. An episode is excluded when `setup_error` or `unscored` or `env_crash`; `score` falls back to `1.0 if success else 0.0` when absent.

- [ ] **Step 1: Write the failing test** `test_official_sr_skips_raised_tasks_and_keeps_partial_credit`: episodes `A: score 1.0`, `B: score 0.5`, `C: setup_error`, `D: unscored`, `E: score 0.0` → `sr == 0.5`, `n_tasks_scored == 3`, `excluded == ["C", "D"]`; plus `test_official_sr_reads_old_success_only_records`: `{"task_name": "A", "success": True}` → `per_task["A"] == 1.0`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `official_sr`; in `report.py` print `Official SR (suite_utils): x.x% over N tasks; M excluded (raised)` above the existing table, with the excluded names.
- [ ] **Step 4: Run** full suite → pass.
- [ ] **Step 5: Commit** `git commit -am "aw metrics: official SR = mean of per-task float scores, raised tasks excluded"`

### Task 5: Record whether typed text landed

**Files:**
- Modify: `benchmarks/aw/awenv/agent.py` (step record after an `input_text`)
- Test: `benchmarks/aw/tests/test_agent.py`

**Interfaces:**
- Consumes: Task 3's `elements()` (post-action observation).
- Produces: step record field `"typed_landed": bool | None` — `None` for non-typing steps; for `input_text`, `True` iff the typed string (case-insensitive, stripped) is a substring of some element's `text` in the post-action observation.

- [ ] **Step 1: Write the failing tests** `test_typed_text_landed_is_recorded` (post-action elements contain the text → `True`) and `test_typed_text_missing_is_recorded` (→ `False`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** after execution of a mapped `input_text`.
- [ ] **Step 4: Run** full suite → pass.
- [ ] **Step 5: Commit** `git commit -am "aw agent: record whether typed text landed (measures the double-tap hypothesis)"`

### Task 6: Gates, record, and the baseline run

**Files:**
- Modify: `AGENTS.md` (program log), `docs/research/2026-11-10-AUDIT-androidworld-contract.md` (§8 status column)
- Create: `benchmarks/aw/configs/aw116_official_baseline.yaml`

- [ ] **Step 1:** Run `bash benchmarks/aw/gates.sh` (or `python -m pytest benchmarks/aw/tests -q` + `python3 scripts/check_record.py`) → green; record counts.
- [ ] **Step 2:** Create the config: same model/checkpoint/track as the aw116 run (`appint_full@938`, `track: V`), `seed: 30`, all 116 tasks, `done_threshold: 0.0`, `step_budgets_file` only as fallback.
- [ ] **Step 3: Operator gate.** The run uses the shared GPU host and the emulator host. Ask before starting; then follow `docs/HANDOFF-2026-11-10.md` §8 (preflight, one chunk, confirm the episode count rose, continue).
- [ ] **Step 4:** Report `official_sr` with the excluded list, the per-family split (TC vs the 25 answer-only tasks), and `typed_landed` counts. Label it the first number on official instances.
- [ ] **Step 5:** Update AGENTS.md + audit §8 status, commit.
