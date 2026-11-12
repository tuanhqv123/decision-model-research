"""Harness changes for contract v4 and the 2026-11-12 harness review (docs/research/2026-11-12-V4-WORKLOG-vi.md):
* deferred effect: a step's "→ screen changed / no change" is decided against the NEXT step's observation (the training
  definition), not an immediate re-read (9% of v3's "no change" clicks were reads taken before the screen updated);
* effect "v4": focus flag and keyboard elements do not count;
* answer: AndroidWorld keeps only the LAST `answer` text (interface.py interaction_cache), so successive element answers
  are sent accumulated, comma-joined (proto_utils.check_agent_answer splits on ','); numbers / True / False replace;
* scroll: inside the smallest scrollable element that contains the chosen element, when that element is small (a
  picker); a full-screen list keeps the plain screen scroll (upstream actuation.py: index -> swipe inside its box)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
sys.path.insert(0, os.path.join(HERE, ".."))

from awenv.agent import AgentCore  # noqa: E402

W, H = 1080, 2400


def el(text, box, cls="android.widget.TextView", pkg="com.app", **flags):
    d = {"text": text, "content_description": "", "hint_text": "", "class": cls, "package_name": pkg,
         "bbox": list(box), "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box)), "is_visible": True,
         "clickable": bool(flags.get("clickable")), "is_clickable": bool(flags.get("clickable"))}
    d.update({"is_%s" % k: v for k, v in flags.items()})
    return d


class FakeModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.histories = []

    def decide(self, goal, history=None, pixels=None, **kw):
        self.histories.append(list(history or []))
        dec = dict(self.decisions.pop(0))
        dec.setdefault("confidence", 0.7)
        dec.setdefault("latency_ms", {"model": 100.0})
        return dec


class LagEnv:
    """screens[k] after k executed actions; with lag, the FIRST element read after an action still sees the old screen."""

    def __init__(self, screens, lag=False):
        self.screens, self.lag, self.executed, self._stale = [list(s) for s in screens], lag, [], False

    def _k(self):
        return min(len(self.executed), len(self.screens) - 1)

    def screenshot(self):
        return np.full((H, W, 3), self._k(), dtype=np.uint8)

    def elements(self):
        if self._stale:
            self._stale = False
            return [dict(e) for e in self.screens[max(0, self._k() - 1)]]
        return [dict(e) for e in self.screens[self._k()]]

    def execute(self, action):
        self.executed.append(action)
        self._stale = self.lag
        return {"ok": True, "error": None}


S0 = [el("Save", (800, 140, 1040, 260), cls="android.widget.Button", clickable=True)]
S1 = S0 + [el("Saved", (300, 2200, 800, 2300))]


def click_save():
    return {"action": "click", "point": [920, 200], "element": {"idx": 1, "history_label": "Save", "aw_index": 0}}


def test_immediate_effect_is_fooled_by_a_late_screen_but_deferred_effect_is_not():
    for deferred, want in ((False, 'click "Save" → no change'), (True, 'click "Save" → screen changed')):
        m = FakeModel([click_save(), {"action": "wait"}])
        agent = AgentCore(m, LagEnv([S0, S1], lag=True), track="VA", history_format="v3", deferred_effect=deferred)
        agent.step("g", 0)
        agent.step("g", 1)
        assert m.histories[1] == [want]


def test_effect_v4_ignores_the_keyboard():
    kb = S0 + [el("q", (0, 1600, 100, 1700), pkg="com.google.android.inputmethod.latin", clickable=True)]
    for effect, want in (("v3", "→ screen changed"), ("v4", "→ no change")):
        m = FakeModel([click_save(), {"action": "wait"}])
        agent = AgentCore(m, LagEnv([S0, kb]), track="VA", history_format="v3", deferred_effect=True, effect=effect)
        agent.step("g", 0)
        agent.step("g", 1)
        assert m.histories[1][0].endswith(want)


def test_answers_accumulate_comma_joined_and_numbers_replace():
    screen = [el("Meeting", (0, 500, 1080, 600)), el("Lunch", (0, 700, 1080, 800))]
    decs = [{"action": "answer", "answer_value": "element", "answer_text": "Meeting",
             "element": {"idx": 1, "history_label": "Meeting", "aw_index": 0}},
            {"action": "answer", "answer_value": "element", "answer_text": "Lunch",
             "element": {"idx": 2, "history_label": "Lunch", "aw_index": 1}},
            {"action": "answer", "answer_value": "3", "answer_text": "3", "element": {"idx": 1}}]
    env = LagEnv([screen])
    m = FakeModel(decs + [{"action": "wait"}])
    agent = AgentCore(m, env, track="VA", history_format="v3", deferred_effect=True)
    for i in range(4):
        agent.step("g", i)
    assert [a["text"] for a in env.executed[:3]] == ["Meeting", "Meeting, Lunch", "3"]
    assert all(a["action_type"] == "answer" for a in env.executed[:3])
    assert m.histories[3][:3] == ['answer "Meeting"', 'answer "Lunch"', 'answer "3"']


def test_scroll_goes_inside_a_small_picker_but_not_a_full_screen_list():
    screen = [el("", (0, 0, 1080, 2400), cls="androidx.recyclerview.widget.RecyclerView", scrollable=True),
              el("", (400, 900, 680, 1300), cls="android.widget.NumberPicker", scrollable=True),
              el("15", (400, 1050, 680, 1150), cls="android.widget.EditText"),
              el("Row", (0, 1500, 1080, 1600))]
    for aw_index, want in ((2, 1), (3, None)):
        env = LagEnv([screen])
        dec = {"action": "scroll", "direction": "down", "element": {"idx": 1, "aw_index": aw_index}}
        AgentCore(FakeModel([dec]), env, track="VA", history_format="v3", scroll_index=True).step("g", 0)
        assert env.executed[0].get("index") == want and env.executed[0]["direction"] == "down"


def test_post_action_wait(monkeypatch):
    import awenv.agent as ag
    slept = []
    monkeypatch.setattr(ag.time, "sleep", lambda s: slept.append(s))
    AgentCore(FakeModel([click_save()]), LagEnv([S0, S1]), track="VA", history_format="v3",
              post_action_wait_s=1.5).step("g", 0)
    assert slept == [1.5]


def test_config_reaches_the_agent_and_the_effect_guard():
    from run_bench import agent_kwargs, model_format_error, plan_from_config
    cfg = {"protocol": "aw-main-v1", "agent": "x", "environment": {}, "suite": {"tasks": ["ClockStopWatchRunning"], "seed": 30},
           "model": {"track": "VA", "history_format": "v3", "effect": "v4"},
           "harness": {"deferred_effect": True, "scroll_index": True, "post_action_wait_s": 1.5}}
    kw = agent_kwargs(plan_from_config(cfg))
    assert (kw["effect"], kw["deferred_effect"], kw["scroll_index"], kw["post_action_wait_s"]) == ("v4", True, True, 1.5)
    assert model_format_error({"contract": "v4", "img_width": 720}, "v3", effect="v4") is None
    assert "effect" in model_format_error({"contract": "v4", "img_width": 720}, "v3", effect="v3")
    assert "effect" in model_format_error({"contract": "v3", "img_width": 720}, "v3", effect="v4")


# --- answer reaches the checker (2026-11-12, measured on the device) -----------------------------------------------
# upstream `task_eval.py:145` does `env.interaction_cache = ""` inside initialize_task. On our wrapper that assignment
# created the wrapper's OWN attribute, which then shadowed `__getattr__` forwarding for the rest of the episode:
# `execute_action(answer)` wrote the answer on the inner env, the checker read "" on the wrapper, and every
# information-retrieval answer scored 0 (live: inner 'Zoom meeting', wrapper '', is_successful 0.0).

class _InnerEnv:
    interaction_cache = ""

    def execute_action(self, action):      # what upstream interface.py:298 does for an answer
        self.interaction_cache = action


def test_answer_survives_the_initialize_task_reset_of_interaction_cache():
    from awenv.awenv_bridge import AndroidWorldEnv
    inner = _InnerEnv()
    w = AndroidWorldEnv(inner)
    w.interaction_cache = ""                 # task_eval.initialize_task(env)
    inner.execute_action("Zoom meeting")    # the answer lands on the inner env
    assert w.interaction_cache == "Zoom meeting"
    w.interaction_cache = "reset"            # a write through the wrapper reaches the inner env too
    assert inner.interaction_cache == "reset"


def test_agent_policy_names_every_setting_that_changes_behaviour():
    agent = AgentCore(FakeModel([]), LagEnv([S0]), track="VA", history_format="v3", effect="v4", mask=True,
                      deferred_effect=True, scroll_index=True, post_action_wait_s=1.5)
    p = agent.policy()
    assert p == {"track": "VA", "history_format": "v3", "effect": "v4", "mask": True, "loop_break": False,
                 "stuck_k": 3, "done_threshold": 0.0, "allow_app_fallback": False, "deferred_effect": True,
                 "scroll_index": True, "post_action_wait_s": 1.5, "fresh_wait_s": 0.0}
    json.dumps(p)                                            # must be serialisable into the trace header


# --- fresh observation (2026-11-12, reproduced on the device) ------------------------------------------------------
# After clicking "Internet" in Settings the screenshot shows the new page at once, but the accessibility tree still
# lists the OLD page (0.3 s), then only the status bar (1.3-4.5 s), and the new page only at ~5.6 s. The model then
# acted on a list of elements that were no longer on screen and every such step read "no change".

class StaleTreeEnv(LagEnv):
    """Pixels update at once; the element list lags `lag` reads behind after each action."""

    def __init__(self, screens, lag):
        super().__init__(screens)
        self.lag, self._left = lag, 0

    def elements(self):
        if self._left > 0:
            self._left -= 1
            return [dict(e) for e in self.screens[max(0, self._k() - 1)]]
        return [dict(e) for e in self.screens[self._k()]]

    def execute(self, action):
        self.executed.append(action)
        self._left = self.lag
        return {"ok": True, "error": None}


class SeeingModel(FakeModel):
    def __init__(self, decisions):
        super().__init__(decisions)
        self.seen = []

    def decide(self, goal, history=None, pixels=None, **kw):
        self.seen.append([e.get("text") for e in (kw.get("elements") or [])])
        return super().decide(goal, history=history, pixels=pixels, **kw)


def test_fresh_wait_rereads_a_tree_that_lags_the_pixels(monkeypatch):
    import awenv.agent as ag
    monkeypatch.setattr(ag.time, "sleep", lambda s: None)
    for wait, want_seen, want_hist in ((0.0, "Save", "→ no change"), (5.0, "Saved", "→ screen changed")):
        m = SeeingModel([click_save(), {"action": "wait"}])
        agent = AgentCore(m, StaleTreeEnv([S0, S1], lag=6), track="VA", history_format="v3", deferred_effect=True,
                          fresh_wait_s=wait)
        agent.step("g", 0)
        agent.step("g", 1)
        assert (want_seen in m.seen[1]) and ("Saved" in m.seen[1]) == (wait > 0)
        assert m.histories[1][0].endswith(want_hist)


def test_fresh_wait_gives_up_when_the_tree_never_changes(monkeypatch):
    import awenv.agent as ag
    slept = []
    monkeypatch.setattr(ag.time, "sleep", lambda s: slept.append(s))
    agent = AgentCore(FakeModel([click_save()]), StaleTreeEnv([S0, S0], lag=0), track="VA", history_format="v3",
                      fresh_wait_s=2.0)
    rec = agent.step("g", 0)                      # a click that truly changes nothing must not wait forever
    assert sum(slept) <= 2.0 + 1e-9 and rec["fresh_wait_polls"] >= 1


def test_fresh_wait_is_off_by_default_and_in_the_policy():
    agent = AgentCore(FakeModel([]), LagEnv([S0]), track="VA", history_format="v3")
    assert agent.fresh_wait_s == 0.0 and agent.policy()["fresh_wait_s"] == 0.0
