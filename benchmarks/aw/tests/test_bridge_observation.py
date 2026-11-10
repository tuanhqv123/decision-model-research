"""The screenshot and the element list the model sees must describe the same moment (H5).

Upstream agents read ONE `State` per step (`env.get_state()` -> pixels AND ui_elements). Until
2026-11-10 `screenshot()` and `elements()` each called `get_state()` — one stabilised, one not —
so the model could be shown the pixels of one screen and the elements of another.
"""
from types import SimpleNamespace

import awenv.awenv_bridge as bridge
from awenv.awenv_bridge import AndroidWorldEnv, _element_dicts


class FakeInner:
    """Every get_state() returns a NEW state whose one element is labelled with the call number."""

    def __init__(self):
        self.calls = []

    def get_state(self, wait_to_stabilize=False):
        self.calls.append(wait_to_stabilize)
        n = len(self.calls)
        return SimpleNamespace(pixels=SimpleNamespace(size=1, shape=(2400, 1080, 3)),
                               ui_elements=[SimpleNamespace(text="state-%d" % n)])

    def execute_action(self, action):
        pass

    def reset(self, go_home=False):
        return None


def test_screenshot_and_elements_describe_the_same_state():
    inner = FakeInner()
    env = AndroidWorldEnv(inner)
    env.screenshot()
    els = env.elements()
    assert [e["text"] for e in els] == ["state-1"]
    assert inner.calls == [True]


def test_elements_before_screenshot_takes_one_stable_state():
    inner = FakeInner()
    env = AndroidWorldEnv(inner)
    els = env.elements()
    assert [e["text"] for e in els] == ["state-1"]
    assert inner.calls == [True]
    env.screenshot()  # a screenshot always STARTS a new observation (the step's first read)...
    assert [e["text"] for e in env.elements()] == ["state-2"]  # ...and elements() follow it
    assert inner.calls == [True, True]


def test_action_invalidates_the_cached_observation(monkeypatch):
    monkeypatch.setattr(bridge, "_to_json_action", lambda a: a)
    inner = FakeInner()
    env = AndroidWorldEnv(inner)
    env.screenshot()
    env.execute({"action_type": "click", "x": 1, "y": 1})
    env.screenshot()
    assert [e["text"] for e in env.elements()] == ["state-2"]
    assert inner.calls == [True, True]


def test_element_dicts_of_the_cached_state_match_a_direct_conversion():
    inner = FakeInner()
    env = AndroidWorldEnv(inner)
    env.screenshot()
    assert env.elements() == _element_dicts(env._obs)


def test_reset_hides_the_automation_overlay_like_every_official_agent():
    """m3a.py:365, t3a.py:295, seeact.py:132 all call `env.hide_automation_ui()` in reset.

    Without it the pointer-location bar and touch trails stay in every screenshot (seen in the B0
    screenshots of 2026-11-10) — pixels no official agent observes and no training image contains.
    """
    class Inner(FakeInner):
        def __init__(self):
            super().__init__()
            self.order = []

        def reset(self, go_home=False):
            self.order.append("reset")

        def hide_automation_ui(self):
            self.order.append("hide")

    inner = Inner()
    AndroidWorldEnv(inner).reset()
    assert inner.order == ["reset", "hide"]


def test_reset_tolerates_an_env_without_the_overlay_switch():
    AndroidWorldEnv(FakeInner()).reset()  # FakeInner has no hide_automation_ui: must not raise
