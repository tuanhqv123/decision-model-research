"""`record_state` (2026-11-11): a step record keeps exactly what the model was given and what it answered, so a
self-played episode can be turned back into training rows (learning from the model's own mistakes):
the element dicts sent, the history lines at decision time, the goal, and every readout of the pass.
Off by default: records are unchanged unless a config asks for it."""
from awenv.agent import AgentCore

from test_mask_v2 import SCREEN, FakeModel, ScreenEnv, v2_click


def test_full_state_is_recorded_when_asked():
    env = ScreenEnv(SCREEN)
    agent = AgentCore(FakeModel([v2_click(), v2_click()]), env, track="VA", record_state=True)
    agent.step("save it", i=0)
    rec = agent.step("save it", i=1)
    st = rec["state_in"]
    assert st["goal"] == "save it"
    assert st["elements"] == [dict(e) for e in SCREEN]          # exactly the dicts sent to the model
    assert st["history"] == [agent.history[0]]                    # the history BEFORE this decision
    assert st["readouts"]["target_el"]["probabilities"] == {"1": 0.7, "2": 0.2}


def test_records_are_unchanged_by_default():
    env = ScreenEnv(SCREEN)
    agent = AgentCore(FakeModel([v2_click()]), env, track="VA")
    rec = agent.step("save it", i=0)
    assert "state_in" not in rec
