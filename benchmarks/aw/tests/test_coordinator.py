"""Tests for the plan coordinator — the integration checks the guide's Step 6 demands.

Guide §Step 6 completion check: "An integration test proves that the no-plan route invokes the planner, a continue
decision preserves the plan, a plan update precedes action selection, and the final evaluator result is traceable
to every submitted action."

These tests use a fake planner and no AndroidWorld, so they run anywhere.
"""
import pytest

from awenv.coordinator import (CONTINUE, UPDATE, Coordinator, Plan)


def planner_ok(goal, obs):
    return ["open the clock app", "start the timer", "finish"]


def planner_says(seq):
    return lambda goal, obs: seq


def test_missing_plan_always_forces_update_for_every_policy():
    for policy in ("B1", "B2", "B3", "P1"):
        c = Coordinator(policy=policy, planner=planner_ok)
        d = c.decide("g", "screen", "obs0", last_action_ok=True)
        assert d["choice"] == UPDATE, policy
        assert d["reason"] == "no_plan", policy
        assert d["forced"] is True, policy


def test_invalid_plan_forces_update():
    c = Coordinator(policy="B1", planner=planner_ok)
    c.apply_update("g", "s")
    c.plan.subgoals = []                       # now invalid
    d = c.decide("g", "screen", "obs1", last_action_ok=True)
    assert d["choice"] == UPDATE
    assert d["reason"] == "invalid_plan"


def test_b1_continues_forever_and_preserves_version():
    c = Coordinator(policy="B1", planner=planner_ok)
    c.apply_update("g", "s")
    v0 = c.plan.version
    for i in range(5):
        d = c.decide("g", "screen", "obs%d" % i, last_action_ok=True)
        assert d["choice"] == CONTINUE
        assert c.plan.version == v0, "B1 must never revise the plan"
        c.observe("obs%d" % i)


def test_b2_updates_every_valid_observation():
    c = Coordinator(policy="B2", planner=planner_ok)
    c.apply_update("g", "s")
    for i in range(3):
        d = c.decide("g", "screen", "obs%d" % i, last_action_ok=True)
        assert d["choice"] == UPDATE
        assert d["reason"] == "B2_always_replan"
        c.observe("obs%d" % i)
        c.apply_update("g", "screen")


def test_b3_fires_on_failed_action_and_on_stuck():
    c = Coordinator(policy="B3", planner=planner_ok)
    c.apply_update("g", "s")
    d = c.decide("g", "screen", "obs1", last_action_ok=False)
    assert d["choice"] == UPDATE and "action_failed" in d["reason"]
    c.observe("obs1")
    c.apply_update("g", "s")
    d = c.decide("g", "screen", "obs2", last_action_ok=True, stuck=True)
    assert d["choice"] == UPDATE and "blocked" in d["reason"]


def test_b3_continues_when_plan_words_are_on_screen():
    c = Coordinator(policy="B3", planner=lambda g, o: ["start the timer"])
    c.apply_update("g", "s")
    d = c.decide("g", "Timer 20h 00m 85s start", "obs1", last_action_ok=True)
    assert d["choice"] == CONTINUE, d["reason"]


def test_stale_observation_does_not_decide_again():
    c = Coordinator(policy="B2", planner=planner_ok)
    c.apply_update("g", "s")
    c.observe("obs1")
    d = c.decide("g", "screen", "obs1", last_action_ok=True)
    assert d["reason"] == "stale_observation_no_decision"
    assert d["forced"] is False


def test_update_rejects_empty_or_invalid_planner_output():
    for bad in ([], ["  "], None):
        c = Coordinator(policy="B1", planner=planner_says(bad))
        c.decide("g", "s", "obs0", True)                 # forces UPDATE
        plan = c.apply_update("g", "s")
        assert plan is None
        assert c.plan is None
        assert "invalid" in c.decisions[-1]["planner_result"] or \
               "empty" in c.decisions[-1]["planner_result"]


def test_plan_revision_preserves_completed_subgoals():
    c = Coordinator(policy="B1", planner=planner_ok)
    c.apply_update("g", "s")
    c.plan.advance("timer started")                       # subgoal 0 -> COMPLETED
    assert c.plan.status[0] == "COMPLETED"
    new = c.apply_update("g", "s2")
    assert new.version == 2
    assert new.status[0] == "COMPLETED", "completed work must survive a revision"
    assert new.cursor == 1


def test_p1_uses_the_learned_score_and_reports_it():
    c = Coordinator(policy="P1", planner=planner_ok, learned_model=lambda f: 0.9)
    c.apply_update("g", "s")
    d = c.decide("g", "screen", "obs1", last_action_ok=True)
    assert d["choice"] == CONTINUE and "P1_p=0.900" in d["reason"]

    c2 = Coordinator(policy="P1", planner=planner_ok, learned_model=lambda f: 0.1)
    c2.apply_update("g", "s")
    d2 = c2.decide("g", "screen", "obs1", last_action_ok=True)
    assert d2["choice"] == UPDATE and "P1_p=0.100" in d2["reason"]


def test_coordinator_never_executes_an_action():
    """The guide forbids the decision layer from acting (§20)."""
    c = Coordinator(policy="B2", planner=planner_ok)
    c.apply_update("g", "s")
    d = c.decide("g", "s", "obs1", True)
    assert set(d) <= {"choice", "reason", "policy", "observation_id", "plan_version", "forced",
                      "planner_result"}
    for k in ("action", "action_type", "submitted_action", "point"):
        assert k not in d


def test_features_exclude_future_and_evaluator_state():
    c = Coordinator(policy="P1", planner=planner_ok)
    c.apply_update("g", "s")
    f = c.features("goal", "obs", "o1", True, False)
    forbidden = {"success", "is_successful", "terminal", "future", "evaluator", "gold", "label"}
    assert not (set(f) & forbidden), set(f) & forbidden
