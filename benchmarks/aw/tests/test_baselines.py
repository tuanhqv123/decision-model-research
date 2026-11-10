"""Baselines exist so "0/6" can be read as a result instead of a mystery.

They are model stand-ins, not separate agent loops: each implements the same ``decide()``
interface the d1 server client does, so they go through the identical adapter, the
identical recorder and the identical metric definitions. A baseline that took a
shortcut through its own code path would prove nothing.
"""
import numpy as np
import pytest

from awenv.adapter import EMITTABLE
from awenv.agent import AgentCore
from awenv.baselines import RandomPolicy, WaitPolicy, policy_names, build_policy


def test_wait_policy_only_waits():
    policy = WaitPolicy()
    decisions = [policy.decide("anything") for _ in range(5)]
    assert {d["action"] for d in decisions} == {"wait"}


def test_random_policy_is_seeded_and_reproducible():
    policy_a, policy_b = RandomPolicy(seed=11), RandomPolicy(seed=11)
    seq_a = [policy_a.decide("g") for _ in range(6)]
    seq_b = [policy_b.decide("g") for _ in range(6)]
    other = [RandomPolicy(seed=12).decide("g") for _ in range(6)]

    assert seq_a == seq_b
    assert seq_a != other


def test_random_policy_never_declares_done():
    """A random control that quits is indistinguishable from one that gave up."""
    policy = RandomPolicy(seed=3)
    for _ in range(200):
        assert policy.decide("g")["action"] != "status"


def test_random_policy_only_emits_representable_actions():
    policy = RandomPolicy(seed=5)
    for _ in range(200):
        assert policy.decide("g")["action"] in EMITTABLE


def test_random_policy_supplies_the_fields_its_action_needs():
    policy = RandomPolicy(seed=5)
    seen = set()
    for _ in range(400):
        dec = policy.decide("g")
        action = dec["action"]
        seen.add(action)
        if action in ("click", "long_press"):
            assert dec["point"][0] is not None and dec["point"][1] is not None
        if action == "scroll":
            assert dec["direction"] in ("up", "down", "left", "right")
        if action == "open_app":
            assert dec["app_name"]
        if action == "type_text":
            assert dec["text"]
    assert {"click", "scroll"} <= seen


def test_baselines_plug_into_the_real_agent_core():
    class Env:
        def __init__(self):
            self.frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
            self.executed = []
            self.n = 0

        def screenshot(self):
            return self.frames[0]

        def execute(self, action):
            self.executed.append(action)
            return {"ok": True}

    env = Env()
    agent = AgentCore(WaitPolicy(), env)

    rec = agent.step("do nothing", i=0)

    assert rec["invalid_action"] is False
    assert env.executed == [{"action_type": "wait"}]


def test_policy_registry_names_are_stable():
    assert policy_names() == ("wait", "random", "status_now")
    assert isinstance(build_policy("wait"), WaitPolicy)
    with pytest.raises(KeyError):
        build_policy("m3a")


def test_m3a_is_declared_outside_the_policy_registry():
    """M3A is a full agent with its own loop; pretending it fits decide() would be a lie."""
    from awenv import baselines

    assert "m3a" not in baselines.policy_names()
    assert callable(baselines.build_m3a)


def test_status_now_policy_declares_done_at_once_through_the_real_agent_core():
    """Q11 floor: an agent that declares `status` at step 0. Any task it 'solves' is satisfied by
    the initial state, and that share of every arm's score is free."""
    from awenv.agent import AgentCore
    from awenv.baselines import StatusNowPolicy

    class Env:
        def screenshot(self):
            return None

        def elements(self):
            return []

        def execute(self, action):
            raise AssertionError("status must not execute anything")

    rec = AgentCore(build_policy("status_now"), Env()).step("any goal", i=0)
    assert rec["done"] is True
    assert isinstance(build_policy("status_now"), StatusNowPolicy)
    assert "status_now" in policy_names()
