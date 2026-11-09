"""The runner is where a benchmark either becomes resumable and paired or becomes noise.

These tests use fake tasks and a fake environment so they run without AndroidWorld; the
point is the bookkeeping a 116-task run depends on — skip-if-done, reset-before-every-episode,
a pairing key that survives an agent switch, and a crash that is recorded instead of
ending the run.
"""
import pytest

from awenv.recorder import Recorder
from awenv.runner import EpisodeSpec, Runner, plan_episodes


class FakeTask:
    def __init__(self, name, idx, succeed=True, coverage=None):
        self.name = name
        self.idx = idx
        self._succeed = succeed
        self.coverage = coverage
        self.initialized = 0
        self.torn_down = 0
        self.checked = 0

    def initialize(self, env):
        self.initialized += 1

    def tear_down(self):
        self.torn_down += 1

    def is_successful(self, env):
        self.checked += 1
        return self._succeed

    def gold_action_types(self):
        if self.coverage is None:
            raise AttributeError("this task does not report its gold action types")
        return ["click", "input_text"] if self.coverage == "reachable" else ["swipe"]


class FakeAgent:
    """Emits a fixed sequence, then declares done."""

    def __init__(self, n_steps=2, done=True):
        self.n_steps = n_steps
        self.done = done
        self.steps = 0

    def step(self, goal, i):
        self.steps += 1
        terminal = self.done and self.steps >= self.n_steps
        return {
            "i": i,
            "action_raw": {"action": "status" if terminal else "click", "confidence": 0.8},
            "action_executed": None if terminal else {"action_type": "click", "x": 1, "y": 1},
            "confidence": 0.8,
            "invalid_action": False,
            "action_ok": None if terminal else True,
            "screen_changed": False if terminal else True,
            "emitted": not terminal,
            "done": terminal,
            "model_ms": 90.0,
            "env_ms": 10.0,
        }


class FakeEnv:
    def __init__(self):
        self.resets = 0
        self.alive = True

    def reset(self):
        self.resets += 1

    def screenshot(self):
        return None

    def execute(self, action):
        return {"ok": True}

    def elements(self):
        return []

    def is_alive(self):
        return self.alive


def _spec(task_name="ClockStopWatchRunning", task_idx=0, seed=30, agent="d1_sft"):
    return EpisodeSpec(
        task_name=task_name,
        task_idx=task_idx,
        seed=seed,
        agent=agent,
        episode_id="p1_t%04d_s%d_%s" % (task_idx, seed, agent),
        instance_params={"task_idx": task_idx},
    )


def _runner(tmp_path, task=None, agent=None, env=None):
    rec = Recorder(str(tmp_path), "run1", config={"protocol": "aw-main-v1"})
    task = task or FakeTask("ClockStopWatchRunning", 0)
    env = env or FakeEnv()
    runner = Runner(
        recorder=rec,
        env_factory=lambda: env,
        agent_factory=lambda e: agent or FakeAgent(),
        task_factory=lambda spec: task,
    )
    return runner, rec, task, env


def test_plan_episodes_is_deterministic_and_agent_scoped(tmp_path):
    tasks = [("ClockStopWatchRunning", 0), ("ContactsAddContact", 1)]

    a = plan_episodes(tasks, seed=30, agent="d1_sft")
    b = plan_episodes(tasks, seed=30, agent="d1_sft")
    other = plan_episodes(tasks, seed=30, agent="wait")

    assert [s.episode_id for s in a] == [s.episode_id for s in b]
    assert [s.episode_id for s in a] != [s.episode_id for s in other]
    assert a[0].episode_id == "p1_t0000_clockstopwatchrunning_s30_d1_sft"


def test_episode_ids_are_unique_across_tasks():
    """Two tasks sharing an episode id would overwrite each other's trajectory."""
    tasks = [("ClockStopWatchRunning", 0), ("ContactsAddContact", 0), ("MarkorCreateNote", 0)]
    specs = plan_episodes(tasks, seed=30, agent="d1_sft")
    ids = [s.episode_id for s in specs]
    assert len(set(ids)) == len(ids)


def test_finished_episode_is_skipped_on_rerun(tmp_path):
    runner, rec, task, env = _runner(tmp_path)
    specs = [_spec()]

    first = runner.run(specs, step_budget=10)
    second = runner.run(specs, step_budget=10)

    assert first["attempted"] == 1
    assert second["attempted"] == 0
    assert second["skipped"] == 1
    assert task.initialized == 1


def test_reset_and_teardown_bracket_every_episode(tmp_path):
    runner, rec, task, env = _runner(tmp_path)

    runner.run([_spec(task_idx=0), _spec(task_idx=1)], step_budget=10)

    assert env.resets == 2
    assert task.initialized == 2
    assert task.torn_down == 2


def test_pairing_key_is_written_into_the_header(tmp_path):
    runner, rec, task, env = _runner(tmp_path)

    runner.run([_spec()], step_budget=10)

    header = rec.episode_header(rec.episodes()[0]["episode_id"])
    assert header["pairing_key"] == "ClockStopWatchRunning|0|30"
    assert header["step_budget"] == 10


def test_step_budget_produces_a_timeout_not_a_silent_success(tmp_path):
    runner, rec, task, env = _runner(
        tmp_path, task=FakeTask("ClockStopWatchRunning", 0, succeed=False), agent=FakeAgent(n_steps=99)
    )

    runner.run([_spec()], step_budget=4)

    ep = rec.episodes()[0]
    assert ep["timeout"] is True
    assert ep["success"] is False
    assert ep["declared_done"] is False
    assert ep["n_steps"] == 4


def test_declared_done_uses_the_official_checker(tmp_path):
    runner, rec, task, env = _runner(tmp_path)

    runner.run([_spec()], step_budget=10)

    ep = rec.episodes()[0]
    assert ep["declared_done"] is True
    assert task.checked == 1
    assert ep["success"] is True


class FlakyAgent(FakeAgent):
    """Raises once, then behaves: an execution error, not a dead environment."""

    def __init__(self):
        super().__init__(n_steps=2)
        self.raised = False

    def step(self, goal, i):
        if not self.raised:
            self.raised = True
            raise RuntimeError("element vanished")
        return super().step(goal, i)


class DeadEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.alive = False


def test_action_error_is_not_an_environment_crash(tmp_path):
    runner, rec, task, env = _runner(tmp_path, task=FakeTask("ClockStopWatchRunning", 0, succeed=False))
    agent = FlakyAgent()
    runner.agent_factory = lambda e: agent

    out = runner.run([_spec()], step_budget=4)

    assert ep_crash_any(rec.episodes()) is False
    assert out["n_action_errors"] == 1
    assert rec.episodes()[0]["n_action_errors"] == 1


class FlakyThenAliveEnv(FakeEnv):
    """One failed probe, then healthy: a hiccup, not a crash."""

    def __init__(self):
        super().__init__()
        self.probes = 0

    def is_alive(self):
        self.probes += 1
        return self.probes > 1


def test_one_failed_probe_is_not_a_crash(tmp_path):
    runner, rec, task, env = _runner(tmp_path, env=FlakyThenAliveEnv())
    agent = FlakyAgent()
    runner.agent_factory = lambda e: agent

    runner.run([_spec()], step_budget=4)

    assert ep_crash_any(rec.episodes()) is False


def test_dead_environment_is_recorded_as_a_crash_and_the_run_continues(tmp_path):
    runner, rec, task, env = _runner(tmp_path, env=DeadEnv())
    agent = FlakyAgent()
    runner.agent_factory = lambda e: agent

    out = runner.run([_spec(task_idx=0), _spec(task_idx=1)], step_budget=4)

    assert out["attempted"] == 2
    assert ep_crash_any(rec.episodes()) is True


def test_a_crashed_episode_can_never_be_scored_as_a_success(tmp_path):
    runner, rec, task, env = _runner(tmp_path, env=DeadEnv(), task=FakeTask("ClockStopWatchRunning", 0, succeed=True))
    agent = FlakyAgent()
    runner.agent_factory = lambda e: agent

    runner.run([_spec()], step_budget=4)

    ep = rec.episodes()[0]
    assert ep["env_crash"] is True
    assert ep["success"] is False
    assert ep["checker_unavailable"] is True


class RaisingCheckerTask(FakeTask):
    """The DeskClock case: the a11y dump fails, the checker raises, the device lives."""

    def is_successful(self, env):
        self.checked += 1
        raise RuntimeError("Could not get a11y tree")


def test_a_raising_checker_on_a_live_device_is_not_an_environment_crash(tmp_path):
    runner, rec, task, env = _runner(tmp_path, task=RaisingCheckerTask("ClockTimerEntry", 0))
    runner.agent_factory = lambda e: FlakyAgent()

    runner.run([_spec(task_name="ClockTimerEntry")], step_budget=4)

    ep = rec.episodes()[0]
    assert ep["env_crash"] is False
    assert ep["success"] is False
    assert ep["checker_unavailable"] is True
    assert "Could not get a11y tree" in ep["checker_error"]


def test_a_raising_checker_on_a_dead_device_is_still_a_crash(tmp_path):
    runner, rec, task, env = _runner(
        tmp_path, env=DeadEnv(), task=RaisingCheckerTask("ClockTimerEntry", 0)
    )
    runner.agent_factory = lambda e: FlakyAgent()

    runner.run([_spec(task_name="ClockTimerEntry")], step_budget=4)

    ep = rec.episodes()[0]
    assert ep["env_crash"] is True
    assert ep["success"] is False


def ep_crash_any(episodes):
    return any(e.get("env_crash") for e in episodes)


def test_unknown_coverage_is_null_not_zero(tmp_path):
    runner, rec, task, env = _runner(tmp_path, task=FakeTask("ClockStopWatchRunning", 0, coverage=None))

    runner.run([_spec()], step_budget=10)

    assert rec.episodes()[0]["coverage"] is None


def test_latency_is_split_between_model_and_environment(tmp_path):
    runner, rec, task, env = _runner(tmp_path)

    runner.run([_spec()], step_budget=10)

    lat = rec.episodes()[0]["latency"]
    assert lat["model_p50"] == pytest.approx(90.0)
    assert lat["env_total_ms"] > 0


def test_task_params_with_exotic_types_do_not_break_the_trajectory(tmp_path):
    """A CalendarEvent in params killed a real run after the episode had finished."""

    class Event:
        def __str__(self):
            return "event:standup"

    class WeirdTask(FakeTask):
        @property
        def aw_params(self):
            return {"title": "standup", "when": Event(), "tags": ("a", "b")}

    runner, rec, task, env = _runner(tmp_path, task=WeirdTask("SimpleCalendarAddOneEvent", 0))

    runner.run([_spec(task_name="SimpleCalendarAddOneEvent")], step_budget=4)

    params = rec.episode_header(rec.episodes()[0]["episode_id"])["instance_params"]
    assert params["when"] == "event:standup"
    assert params["tags"] == ["a", "b"]


def test_wall_clock_budget_is_per_episode_not_for_the_whole_run(tmp_path):
    """A run-wide cap silently turned a 116-task suite into a 10-task one (2026-11-08).

    The suite printed "SR (all) 10.0% (1/10)" and nothing else, which reads as a finished
    benchmark. The cap must bind inside an episode, never across the suite.
    """
    runner, rec, task, env = _runner(tmp_path)

    out = runner.run([_spec(task_idx=i) for i in range(4)], step_budget=10, wall_clock_budget=60.0)

    assert out["planned"] == 4
    assert out["completed"] == 4, "a per-episode budget must not end the suite early"
    assert out["attempted"] == 4


def test_wall_clock_budget_records_a_timeout_inside_the_episode(tmp_path):
    """The per-episode cap must mark the episode a timeout, not a failure or a success."""
    runner, rec, task, env = _runner(
        tmp_path, task=FakeTask("ClockStopWatchRunning", 0, succeed=False), agent=FakeAgent(n_steps=999)
    )

    runner.run([_spec()], step_budget=999, wall_clock_budget=0.001)

    ep = rec.episodes()[0]
    assert ep["wall_clock_exceeded"] is True
    assert ep["timeout"] is True
    assert ep["success"] is False


def test_a_task_that_cannot_initialize_does_not_end_the_suite(tmp_path):
    """`ExpenseDeleteDuplicates` raised `no such table: map_markers` in initialize_task and killed
    a 116-task run at episode 42 on 2026-11-08, losing 74 tasks of coverage to one bad schema."""

    class BrokenInitTask(FakeTask):
        def initialize_task(self, env):
            raise RuntimeError("no such table: map_markers")

    rec = Recorder(str(tmp_path), "run1", config={"protocol": "aw-main-v1"})
    task = BrokenInitTask("ExpenseDeleteDuplicates", 0)
    runner = Runner(
        recorder=rec,
        env_factory=lambda: FakeEnv(),
        agent_factory=lambda e: FakeAgent(),
        task_factory=lambda spec: task,
    )

    out = runner.run([_spec(task_idx=0), _spec(task_idx=1)], step_budget=4)

    assert out["attempted"] == 2, "a setup failure must not abort the remaining tasks"
    eps = rec.episodes()
    assert len(eps) == 2
    assert eps[0]["setup_error"].startswith("RuntimeError: no such table")
    assert eps[0]["success"] is False
    assert eps[0]["timeout"] is False, "a task that never started is not a timeout"
    assert eps[0]["n_steps"] == 0


def test_a_failing_reset_does_not_end_the_suite(tmp_path):
    """`env.reset()` raised on a live device at episode 93 of the 116-task run (2026-11-08).

    `adb devices` still showed `device` and /health returned success, so this is the same
    transient a11y-dump failure AndroidWorld retries internally — it must not cost the run
    its remaining tasks.
    """

    class FlakyEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.fail_next = 2

        def reset(self):
            if self.fail_next > 0:
                self.fail_next -= 1
                raise RuntimeError("Could not get a11y tree.")
            super().reset()

    rec = Recorder(str(tmp_path), "run1", config={"protocol": "aw-main-v1"})
    env = FlakyEnv()
    runner = Runner(
        recorder=rec,
        env_factory=lambda: env,
        agent_factory=lambda e: FakeAgent(),
        task_factory=lambda spec: FakeTask("ClockStopWatchRunning", spec.task_idx),
    )

    out = runner.run([_spec(task_idx=0), _spec(task_idx=1)], step_budget=4)

    assert out["attempted"] == 2, "a transient reset failure must not abort the suite"
    assert env.fail_next == 0, "the runner should have retried and succeeded"
    assert len(rec.episodes()) == 2
    assert all(e["setup_error"] is None for e in rec.episodes())


def test_a_persistently_failing_reset_is_recorded_not_raised(tmp_path):
    """If reset never succeeds the episode is a setup failure, but the rest of the suite runs."""

    class DeadResetEnv(FakeEnv):
        def reset(self):
            raise RuntimeError("Could not get a11y tree.")

    rec = Recorder(str(tmp_path), "run1", config={"protocol": "aw-main-v1"})
    runner = Runner(
        recorder=rec,
        env_factory=lambda: DeadResetEnv(),
        agent_factory=lambda e: FakeAgent(),
        task_factory=lambda spec: FakeTask("ClockStopWatchRunning", spec.task_idx),
    )

    out = runner.run([_spec(task_idx=0), _spec(task_idx=1)], step_budget=4)

    assert out["attempted"] == 2
    eps = rec.episodes()
    assert len(eps) == 2
    assert eps[0]["setup_error"].startswith("reset: RuntimeError")
    assert eps[0]["success"] is False
    assert eps[0]["timeout"] is False


def test_a_raising_checker_marks_the_episode_unscored_not_a_timeout(tmp_path):
    """12 episodes ran to completion and were never judged; they entered the model's buckets.

    On 2026-11-08 `task.is_successful(env)` raised AttributeError for 12 episodes that each ran
    4-25 steps. The runner recorded success=False, and because `timeout` was computed from
    `declared_done` alone, 6 became "timeout" and 6 became "premature status" — model failures
    that no checker ever assessed.
    """

    class RaisingCheckerTask(FakeTask):
        def is_successful(self, env):
            raise AttributeError("'AndroidWorldEnv' object has no attribute 'get_state'")

    rec = Recorder(str(tmp_path), "run1", config={"protocol": "aw-main-v1"})
    runner = Runner(
        recorder=rec,
        env_factory=lambda: FakeEnv(),
        agent_factory=lambda e: FakeAgent(n_steps=99),
        task_factory=lambda spec: RaisingCheckerTask("ContactsAddContact", 0, succeed=False),
    )

    runner.run([_spec()], step_budget=4)

    ep = rec.episodes()[0]
    assert ep["unscored"] is True, "a raising checker means the episode was never judged"
    assert ep["checker_unavailable"] is True
    assert ep["timeout"] is False, "an unscored episode is not a model timeout"
    assert ep["success"] is False
    assert "no attribute 'get_state'" in ep["checker_error"]


def test_a_declared_done_episode_with_a_raising_checker_is_still_unscored(tmp_path):
    """The other half: the agent declared done, so it could not be labelled a timeout either."""

    class RaisingCheckerTask(FakeTask):
        def is_successful(self, env):
            raise RuntimeError("checker exploded")

    rec = Recorder(str(tmp_path), "run1", config={"protocol": "aw-main-v1"})
    runner = Runner(
        recorder=rec,
        env_factory=lambda: FakeEnv(),
        agent_factory=lambda e: FakeAgent(n_steps=2),
        task_factory=lambda spec: RaisingCheckerTask("BrowserDraw", 0, succeed=False),
    )

    runner.run([_spec()], step_budget=10)

    ep = rec.episodes()[0]
    assert ep["declared_done"] is True
    assert ep["unscored"] is True
    assert ep["timeout"] is False


def test_the_bridge_forwards_get_state_to_the_wrapped_env():
    """The root cause: AndroidWorld's checkers call `env.get_state()`, the wrapper did not have it.

    `AndroidWorldEnv` defined reset/close/controller/screenshot/elements/foreground/screen_size/
    execute/is_alive and no `__getattr__`, so `task.is_successful(env)` raised AttributeError for
    every task whose checker reads state — 12 episodes in the 116-task run (2026-11-08).
    """
    from awenv.awenv_bridge import AndroidWorldEnv

    class Inner:
        def __init__(self):
            self.calls = 0

        def get_state(self, wait_to_stabilize=False):
            self.calls += 1
            return "STATE"

    inner = Inner()
    bridge = AndroidWorldEnv(inner)

    assert bridge.get_state() == "STATE", "the wrapper must expose the wrapped env's get_state"
    assert inner.calls == 1
    # and an undefined member still raises AttributeError rather than returning something odd
    try:
        bridge.definitely_not_a_real_attribute
    except AttributeError:
        pass
    else:
        raise AssertionError("unknown attributes must raise AttributeError")
