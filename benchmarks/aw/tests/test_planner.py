"""LLM planner + executor harness (2026-11-16): one runner step = at most one device action; the executor's `status`
ends a local goal and never enters the model's history; the LLM finishes or answers the task."""
import numpy as np
import pytest

from awenv.agent import AgentCore
from awenv.planner import PlannerAgent, parse_reply, render_elements


class FakeModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, goal, history=None, pixels=None, instruction=None, **kw):
        self.calls.append({"history": list(history or []), "instruction": instruction})
        dec = dict(self.decisions.pop(0))
        dec.setdefault("confidence", 0.9)
        dec.setdefault("latency_ms", {"model": 10.0})
        return dec


class FakeEnv:
    def __init__(self):
        self.executed = []
        self.n = 0

    def screenshot(self):
        return np.full((8, 8, 3), self.n, dtype=np.uint8)

    def execute(self, action):
        self.executed.append(action)
        self.n += 1
        return {"ok": True}

    def elements(self):
        return [{"text": "Save %d" % self.n, "class_name": "android.widget.Button", "is_clickable": True,
                 "bbox_pixels": {"x_min": 0, "x_max": 4, "y_min": 0, "y_max": 4}, "package_name": "com.app"}]


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def ask(self, messages):
        self.seen.append(messages[-1]["content"])
        return {"text": self.replies.pop(0), "ms": 1.0}


def make(decisions, replies, **kw):
    env = FakeEnv()
    model = FakeModel(decisions)
    core = AgentCore(model, env, executor=True)
    return PlannerAgent(core, FakeLLM(replies), **kw), model, env


GOAL1 = '{"next": "local_goal", "instruction": "Click on Save"}'


def test_the_local_goal_reaches_the_model_as_its_instruction():
    agent, model, env = make([{"action": "click", "point": [1, 1]}], [GOAL1])
    rec = agent.step("Save the note", 0)
    assert model.calls[0]["instruction"] == "Click on Save"
    assert len(env.executed) == 1 and rec["done"] is False
    assert rec["planner"]["local_goal"] == "Click on Save"


def test_executor_done_asks_the_llm_again_in_the_same_step_and_stays_out_of_history():
    agent, model, env = make([{"action": "click", "point": [1, 1]}, {"action": "status"},
                              {"action": "click", "point": [2, 2]}],
                             [GOAL1, '{"next": "local_goal", "instruction": "Click on OK"}'])
    agent.step("t", 0)
    rec = agent.step("t", 1)
    assert len(env.executed) == 2                      # one device action per runner step
    assert model.calls[2]["instruction"] == "Click on OK"
    assert not any("status" in h for h in model.calls[2]["history"])   # training never shows a status line
    assert agent.log[0].startswith('Local goal "Click on Save": DONE; actions: ')


def test_llm_finish_ends_the_episode_without_a_device_action():
    agent, model, env = make([], ['{"next": "finish"}'])
    rec = agent.step("t", 0)
    assert rec["done"] is True and env.executed == []


def test_llm_answer_is_sent_to_the_environment_then_the_episode_ends():
    agent, model, env = make([], ['{"thought": "x", "next": "answer", "text": "42"}'])
    rec = agent.step("How many?", 0)
    assert env.executed == [{"action_type": "answer", "text": "42"}] and rec["done"] is True


def test_too_many_steps_on_one_local_goal_reports_blocked_to_the_llm():
    agent, model, env = make([{"action": "click", "point": [1, 1]}] * 2, [GOAL1], max_goal_steps=2)
    agent.step("t", 0)
    agent.step("t", 1)
    assert agent.local_goal is None and "BLOCKED" in agent.log[-1]


def test_an_unusable_llm_reply_costs_no_device_action():
    agent, model, env = make([], ["no json here", "{}", '{"next": "local_goal"}', "?"])
    rec = agent.step("t", 0)
    assert env.executed == [] and rec["done"] is False and rec["reason"] == "planner_no_action"


def test_the_planner_requires_an_executor_core():
    with pytest.raises(ValueError):
        PlannerAgent(AgentCore(FakeModel([]), FakeEnv()), FakeLLM([]))


def test_parse_reply_accepts_json_inside_text_and_rejects_empty_goals():
    assert parse_reply('ok: {"next": "local_goal", "instruction": " Open the Clock app "}')["instruction"] == \
        "Open the Clock app"
    assert parse_reply('{"next": "local_goal", "instruction": ""}')["next"] == "invalid"
    assert parse_reply('{"next": "finish"}')["next"] == "finish"


def test_render_elements_shows_labels_and_flags_and_skips_unlabelled():
    s = render_elements([{"text": "Wi-Fi", "class_name": "a.Switch", "is_checked": True, "package_name": "p"},
                         {"class_name": "a.View"}])
    assert "Switch 'Wi-Fi' (checked)" in s and "View" not in s


def test_a_planner_section_builds_the_planner_agent_and_none_builds_the_plain_agent():
    import yaml
    from run_bench import build_agent, plan_from_config
    cfg = yaml.safe_load(open("configs/aw116_v4g_s0_a.yaml"))
    cfg["suite"].pop("step_budgets_file")   # a container path
    plain = build_agent(plan_from_config(cfg), FakeModel([]), FakeEnv())
    assert isinstance(plain, AgentCore) and plain.executor is False
    cfg["planner"] = {"url": "http://x/v1/chat/completions", "model": "m", "max_goal_steps": 5}
    agent = build_agent(plan_from_config(cfg), FakeModel([]), FakeEnv(), llm=FakeLLM([]))
    assert isinstance(agent, PlannerAgent) and agent.core.executor is True and agent.max_goal_steps == 5
    assert agent.core.history_format == "v3" and agent.core.fresh_wait_s == 20.0   # the harness settings still apply


def test_the_llm_is_told_which_actions_a_local_goal_took():
    agent, model, env = make([{"action": "click", "point": [1, 1]}, {"action": "status"}, {"action": "click", "point": [2, 2]}],
                             [GOAL1, '{"next": "local_goal", "instruction": "Click on OK"}'])
    agent.step("t", 0)
    agent.step("t", 1)
    assert agent.log[0].startswith('Local goal "Click on Save": DONE; actions: click')


def test_a_goal_the_executor_calls_done_without_acting_is_not_sent_again_unchanged():
    # BrowserMultiply 2026-11-16: the LLM repeated "Click on Chrome" 4 times; each time the executor said done at once
    agent, model, env = make([{"action": "status"}, {"action": "click", "point": [3, 3]}],
                             [GOAL1, GOAL1, '{"next": "local_goal", "instruction": "Click on Just once"}'])
    rec = agent.step("t", 0)
    assert len(model.calls) == 2                       # the repeated goal never reached the executor
    assert model.calls[1]["instruction"] == "Click on Just once" and len(env.executed) == 1
    assert "did nothing" in agent.log[0] and "DIFFERENT" in agent.log[1]


def test_a_history_line_saying_no_change_counts_as_an_unchanged_screen():
    # SystemBrightnessMax 2026-11-16: the structural fingerprint changed (ui_changed True) while the effect the model sees
    # said "no change" six times in a row; BLOCKED came only from the 6-action cap
    agent, model, env = make([{"action": "click", "point": [1, 1]}] * 2, [GOAL1], stuck_unchanged=2)
    agent.core.step = lambda goal, i, instruction=None: {"done": False, "ui_changed": True,
                                                         "history_line": 'click "Save" → no change'}
    agent.step("t", 0)
    agent.step("t", 1)
    assert agent.local_goal is None and "screen did not change" in agent.log[-1]


def test_the_prompt_names_what_the_helper_can_do_and_asks_to_finish_at_once():
    from awenv.planner import SYSTEM
    assert "cannot drag" in SYSTEM and "finish" in SYSTEM and ", then" in SYSTEM
