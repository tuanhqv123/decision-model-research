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
    # `answer` became emittable with contract v4 (2026-11-12); `double_tap` is still outside the model's action set
    agent = AgentCore(FakeModel([{"action": "double_tap", "text": "42"}]), env)

    rec = agent.step("do it", i=0)

    assert rec["invalid_action"] is True
    assert rec["action_executed"]["action_type"] == "wait"
    assert rec["error"] is None
    assert "unmappable" in rec["reason"]


def test_type_text_executes_single_input_text():
    env = FakeEnv()
    agent = AgentCore(
        FakeModel([{"action": "type_text", "point": [4, 5], "text": "hello"}]),
        env,
    )

    rec = agent.step("type hello", i=0)

    assert [a["action_type"] for a in env.executed] == ["input_text"]
    assert env.executed[0]["text"] == "hello"
    assert env.executed[0]["clear_text"] is True
    assert rec["invalid_action"] is False
    # the whole list must reach the trace, not just actions[0]
    assert rec["actions_executed"] == env.executed


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
        loop_break=True,
    )

    records = [agent.step("stuck task", i=i) for i in range(5)]

    assert records[0]["stuck_event"] is False
    assert records[1]["stuck_event"] is False
    assert records[2]["stuck_event"] is True
    assert "loop_break" in records[2]["scaffolding"]
    assert records[2]["action_executed"]["action_type"] == "navigate_back"


def test_blocked_status_never_trips_the_loop_breaker():
    """Three refused terminations must not be rewritten into a back-press.

    On 2026-11-10 MarkorCreateNote showed a correctly typed note, the model declared done
    three times, the is_done gate refused all three, and each refusal mapped to `wait` — the
    same signature. stuck_k then fired and REPLACED the third decision with
    LOOP_BREAK_ACTION (`navigate_back`), throwing away the screen the task needed.

    A refused decision was never executed, so it carries no evidence about the screen and
    must not count toward the loop counter.
    """
    env = FakeEnv(frames=[_frame(1)])
    agent = AgentCore(
        FakeModel([{"action": "status", "is_done": 0.2}] * 3),
        env,
        stuck_k=3,
        done_threshold=0.5,
    )

    records = [agent.step("save the note", i=i) for i in range(3)]

    assert all(r["stuck_event"] is False for r in records), (
        "a gate refusal was counted as a screen-level loop")
    assert all(r["reason"] == "is_done_below_threshold" for r in records)
    assert [r["action_executed"]["action_type"] for r in records] == ["wait"] * 3
    assert not any("loop_break" in r["scaffolding"] for r in records)


def test_loop_signature_ignores_nothing_that_can_differ():
    """Two different typings, or two opposite scrolls, are not the same action."""
    from awenv.agent import _signature
    from awenv.adapter import map_decision

    a = _signature(map_decision({"action": "type_text", "point": [1, 2], "text": "alpha"}))
    b = _signature(map_decision({"action": "type_text", "point": [1, 2], "text": "beta"}))
    assert a != b, "different typed text collapsed into one signature"

    up = _signature(map_decision({"action": "scroll", "direction": "up"}))
    down = _signature(map_decision({"action": "scroll", "direction": "down"}))
    assert up != down, "opposite scroll directions collapsed into one signature"


def test_history_never_claims_a_refused_termination_happened():
    """A refused `status` must not be reported to the model as a completed declaration.

    The model's history is its memory of the episode. Saying "status -> declare done" when the
    gate refused and a `wait` ran makes every later decision be taken by a model that believes
    the task already ended — the exact state MarkorCreateNote was left in on 2026-11-10.
    """
    env = FakeEnv()
    model = FakeModel([{"action": "status", "is_done": 0.2},
                       {"action": "click", "point": [4, 4]}])
    agent = AgentCore(model, env, done_threshold=0.5)

    agent.step("save the note", i=0)
    agent.step("save the note", i=1)

    first_line = model.calls[1]["history"][0]
    assert "declare done" not in first_line, (
        "a refused termination was reported as completed: %r" % first_line)
    # What EXECUTED was a wait, and the history says exactly that, in the trained vocabulary.
    assert first_line == "wait", first_line


def test_history_reports_a_rejected_decision_as_rejected():
    """An unmappable decision must not be reported as if the named action ran.

    A `click` with no point maps to `wait` and sets `invalid`. Reporting "click -> ok" told the
    model its coordinate-less click had executed, so it had no way to learn the point was
    missing — and `invalid_action` never appears in the history the model reads.
    """
    env = FakeEnv()
    model = FakeModel([{"action": "click", "point": None},
                       {"action": "click", "point": [4, 4]}])
    agent = AgentCore(model, env)

    agent.step("do it", i=0)
    agent.step("do it", i=1)

    line = model.calls[1]["history"][0]
    # The adapter executed a `wait`; the history records what executed, rendered by the same
    # function that renders training history (decision K3) — not a serve-only "(rejected: ...)".
    assert line == "wait", line
    # no coordinate pair, because the point was missing and nothing was clicked
    import re
    assert not re.search(r"\(\s*\d+\s*,", line), (
        "a rejected decision was reported with coordinates it never had: %r" % line)


def test_history_reports_a_real_termination_as_one():
    env = FakeEnv()
    model = FakeModel([{"action": "status", "is_done": 0.9}])
    agent = AgentCore(model, env, done_threshold=0.5)

    rec = agent.step("done already", i=0)

    assert rec["done"] is True
    assert agent.history[0] == "status"


def test_history_is_built_from_what_executed_not_from_the_decision():
    env = FakeEnv()
    model = FakeModel([{"action": "click", "point": [7, 9]}] * 2)
    agent = AgentCore(model, env)

    agent.step("do it", i=0)
    agent.step("do it", i=1)

    history = model.calls[1]["history"]
    # ONE line per executed step, in the trained vocabulary and nothing else. A serve-only
    # sentence such as "Note: the last action did not change the interface." is an input the
    # model never saw in training (decision 2026-11-10, K3); `ui_changed` stays in the trace.
    assert history == ["click"], history


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


# ---------------------------------------------------------------- did the typed text land?
# Measures the open hypothesis "typing failed because of the harness" (audit §7): the pointer
# read produced the right string 24/24 times, but no trace recorded whether it reached the field.
class TypingEnv(FakeEnv):
    def __init__(self, after_texts):
        super().__init__()
        self.after_texts = after_texts

    def elements(self):
        # before any execution the field is empty; after it, show what the device holds
        if not self.executed:
            return [{"text": ""}]
        return [{"text": t} for t in self.after_texts]


def test_typed_text_landed_is_recorded():
    env = TypingEnv(["Emily Mohammed", "Save"])
    agent = AgentCore(FakeModel([{"action": "type_text", "point": [4, 5], "text": "emily mohammed "}]), env)
    rec = agent.step("add Emily", i=0)
    assert rec["typed_landed"] is True


def test_typed_text_missing_is_recorded():
    env = TypingEnv(["", "Save"])
    agent = AgentCore(FakeModel([{"action": "type_text", "point": [4, 5], "text": "Emily"}]), env)
    rec = agent.step("add Emily", i=0)
    assert rec["typed_landed"] is False


def test_typed_landed_is_none_for_other_actions():
    env = TypingEnv(["x"])
    agent = AgentCore(FakeModel([{"action": "click", "point": [1, 2]}]), env)
    rec = agent.step("click", i=0)
    assert rec["typed_landed"] is None


def test_history_uses_the_training_renderer():
    """Serve-time history lines are produced by `ac_common.render_history_line`, the training renderer."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "train"))
    from ac_common import render_history_line
    env = FakeEnv()
    decs = [{"action": "type_text", "point": [4, 5], "text": "Floor lamp"},
            {"action": "scroll", "direction": "down"},
            {"action": "open_app", "app_name": "Pro Expense"},
            {"action": "click", "point": [1, 2]}]
    model = FakeModel(decs + [{"action": "status"}])
    agent = AgentCore(model, env)
    for i in range(5):
        agent.step("g", i=i)
    want = [render_history_line({"action_type": "type_text", "text": "Floor lamp"}),
            render_history_line({"action_type": "scroll", "direction": "down"}),
            render_history_line({"action_type": "open_app", "app_name": "Pro Expense"}),
            render_history_line({"action_type": "click"})]
    assert model.calls[4]["history"] == want


def test_loop_is_recorded_but_not_overridden_by_default():
    """Detection is a measurement; overriding the model is a harness policy, off unless configured.

    The override replaced a correct screen with a back-press on MarkorCreateNote (2026-11-10) and no
    measurement shows it helping; upstream agents have no such rule.
    """
    env = FakeEnv(frames=[_frame(1)])
    agent = AgentCore(FakeModel([{"action": "click", "point": [3, 3]}] * 3), env, stuck_k=3)
    records = [agent.step("stuck task", i=i) for i in range(3)]
    assert records[2]["stuck_event"] is True
    assert records[2]["action_executed"]["action_type"] == "click"
    assert "loop_break" not in records[2]["scaffolding"]


# ---------------------------------------------------------------- generative control arm (Q1) trace fields
def test_generative_parse_failure_keeps_its_raw_text_and_token_count():
    """A parse failure must be auditable: the raw generated text, the parse error and the number of
    generated tokens reach the trace (they were dropped by `_raw_of`)."""
    env = FakeEnv()
    model = FakeModel([{"action": "", "raw": "{\"action_type\": \"click\", \"x\": [499,366]}",
                        "parse_error": "missing_point", "tokens_generated": 17}])
    rec = AgentCore(model, env).step("g", i=0)
    assert rec["invalid_action"] is True
    assert rec["action_raw"]["raw"].startswith("{\"action_type\"")
    assert rec["action_raw"]["parse_error"] == "missing_point"
    assert rec["action_raw"]["tokens_generated"] == 17
