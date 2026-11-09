"""The agent loop must attribute every failure to the model or the environment.

The 2026-11-02 smoke run could not answer "was that a model error or a harness error",
so these tests pin the four places that ambiguity came from: silent `done`, silent
`wait`, swallowed execution exceptions, and history rebuilt from model output.
"""
import numpy as np
import pytest

from awenv.agent import AgentCore


class FakeModel:
    """Returns the queued decisions in order, and records what it was asked."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, goal, history=None, pixels=None, **kw):
        self.calls.append({"goal": goal, "history": list(history or [])})
        if not self.decisions:
            raise AssertionError("model called more times than the test queued decisions")
        dec = dict(self.decisions.pop(0))
        dec.setdefault("confidence", 0.7)
        dec.setdefault("latency_ms", {"model": 100.0})
        return dec


class FakeEnv:
    """Minimal environment: fixed frames, scripted execution outcomes, no AndroidWorld."""

    def __init__(self, frames=None, outcomes=None):
        self.frames = list(frames or [np.zeros((8, 8, 3), dtype=np.uint8)])
        self.outcomes = list(outcomes or [])
        self.executed = []
        self._n = 0

    def screenshot(self):
        idx = min(self._n, len(self.frames) - 1)
        return self.frames[idx]

    def advance(self):
        self._n += 1

    def execute(self, action):
        self.executed.append(action)
        self.advance()
        if self.outcomes:
            out = self.outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            return out
        return {"ok": True, "error": None}

    def elements(self):
        return []


def _frame(value):
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_status_terminates_without_executing_anything():
    env = FakeEnv()
    agent = AgentCore(FakeModel([{"action": "status"}]), env)

    rec = agent.step("do it", i=0)

    assert rec["done"] is True
    assert env.executed == []
    assert rec["action_executed"] is None


def test_unmappable_action_is_invalid_and_not_a_clean_wait():
    env = FakeEnv()
    agent = AgentCore(FakeModel([{"action": "answer", "text": "42"}]), env)

    rec = agent.step("do it", i=0)

    assert rec["invalid_action"] is True
    assert rec["action_executed"]["action_type"] == "wait"
    assert rec["error"] is None
    assert "unmappable" in rec["reason"]


def test_type_text_executes_click_then_input_text():
    env = FakeEnv()
    agent = AgentCore(
        FakeModel([{"action": "type_text", "point": [4, 5], "text": "hello"}]),
        env,
    )

    rec = agent.step("type hello", i=0)

    assert [a["action_type"] for a in env.executed] == ["click", "input_text"]
    assert env.executed[1]["text"] == "hello"
    assert rec["invalid_action"] is False


def test_execution_exception_is_recorded_not_raised():
    env = FakeEnv(outcomes=[RuntimeError("element vanished")])
    agent = AgentCore(FakeModel([{"action": "click", "point": [1, 2]}]), env)

    rec = agent.step("click it", i=0)

    assert rec["action_ok"] is False
    assert "element vanished" in rec["error"]
    assert rec["done"] is False


def test_executor_rejection_is_an_env_reject_not_an_invalid_action():
    env = FakeEnv(outcomes=[{"ok": False, "error": "app not found"}])
    agent = AgentCore(FakeModel([{"action": "click", "point": [1, 2]}]), env)

    rec = agent.step("click it", i=0)

    assert rec["invalid_action"] is False
    assert rec["action_ok"] is False
    assert rec["env_reject"] is True


def test_loop_detection_fires_once_and_is_logged():
    env = FakeEnv(frames=[_frame(1)])
    agent = AgentCore(
        FakeModel([{"action": "click", "point": [3, 3]}] * 5),
        env,
        stuck_k=3,
    )

    records = [agent.step("stuck task", i=i) for i in range(5)]

    assert records[0]["stuck_event"] is False
    assert records[1]["stuck_event"] is False
    assert records[2]["stuck_event"] is True
    assert "loop_break" in records[2]["scaffolding"]
    assert records[2]["action_executed"]["action_type"] == "navigate_back"


def test_history_is_built_from_what_executed_not_from_the_decision():
    env = FakeEnv()
    model = FakeModel([{"action": "click", "point": [7, 9]}] * 2)
    agent = AgentCore(model, env)

    agent.step("do it", i=0)
    agent.step("do it", i=1)

    history = model.calls[1]["history"]
    assert history == ["click (7,9) -> ok"]


def test_done_is_never_inferred_from_the_step_counter():
    env = FakeEnv()
    model = FakeModel([{"action": "wait"}] * 4)
    agent = AgentCore(model, env)

    records = [agent.step("do it", i=i) for i in range(4)]

    assert all(r["done"] is False for r in records)


def test_the_frame_the_model_saw_is_attached_for_replay():
    env = FakeEnv(frames=[_frame(7)])
    agent = AgentCore(FakeModel([{"action": "click", "point": [1, 2]}] * 2), env)

    rec = agent.step("do it", i=0)

    assert rec["screenshot_pixels"] is not None
    assert int(rec["screenshot_pixels"][0][0][0]) == 7


def test_model_ms_is_read_from_the_latency_dict_not_assumed_a_float():
    env = FakeEnv()
    agent = AgentCore(
        FakeModel([{"action": "wait", "latency_ms": {"model": 87.5, "total": 99.0}}]),
        env,
    )

    rec = agent.step("do it", i=0)

    assert rec["model_ms"] == pytest.approx(87.5)
