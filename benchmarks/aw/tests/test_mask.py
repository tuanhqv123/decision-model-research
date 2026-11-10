"""Q2-ii verified-failure masking (PLAN-2026-11-10 §3, Q2-ii), behind `mask=True`, default OFF.

Within an episode, a (screen, action) pair the environment verified to change nothing is not issued
again; the model's own next-best option FROM THE SAME FORWARD PASS is taken instead. These tests pin
the exact semantics: what a signature is, the order alternatives are tried in, what is never masked,
what happens when nothing is left, and that the flag off changes nothing.
"""
import numpy as np

from awenv.agent import AgentCore


def el(text="", cls="android.widget.Button", bbox=(0, 0, 10, 10), **flags):
    x0, y0, x1, y1 = bbox
    d = {
        "text": text, "content_description": "", "hint_text": "", "class": cls,
        "package_name": "com.example", "bbox": [x0, y0, x1, y1],
        "bbox_pixels": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
                        "width": x1 - x0, "height": y1 - y0,
                        "center": [(x0 + x1) / 2.0, (y0 + y1) / 2.0]},
        "is_visible": True, "clickable": False, "is_clickable": False, "is_editable": False,
        "is_scrollable": False, "is_checked": False, "is_checkable": False, "is_selected": False,
        "is_focused": False, "is_long_clickable": False, "is_enabled": True, "resource_name": "",
    }
    d.update(flags)
    if "is_clickable" in flags:
        d["clickable"] = flags["is_clickable"]
    return d


SAVE = el("Save", bbox=(10, 10, 30, 20), is_clickable=True)      # area 200 -> candidate 1
CANCEL = el("Cancel", bbox=(40, 10, 80, 20), is_clickable=True)  # area 400 -> candidate 2
FRAME = el("", cls="android.widget.FrameLayout", bbox=(0, 0, 100, 100))  # unlabelled, inert
SCREEN_A = [FRAME, SAVE, CANCEL]
SCREEN_B = [FRAME, el("Saved", cls="android.widget.TextView", bbox=(10, 40, 90, 60))]


class FakeModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, goal, history=None, pixels=None, **kw):
        self.calls.append({"history": list(history or []), **kw})
        if not self.decisions:
            raise AssertionError("masking must never make an extra model call")
        dec = dict(self.decisions.pop(0))
        dec.setdefault("confidence", 0.7)
        dec.setdefault("latency_ms", {"model": 100.0})
        return dec


class ScreenEnv:
    """screens[k] is the screen after k executed actions (the last one repeats)."""

    def __init__(self, screens):
        self.screens = [list(s) for s in screens]
        self.executed = []

    def _k(self):
        return min(len(self.executed), len(self.screens) - 1)

    def screenshot(self):
        return np.full((120, 100, 3), self._k(), dtype=np.uint8)

    def elements(self):
        return [dict(e) for e in self.screens[self._k()]]

    def execute(self, action):
        self.executed.append(action)
        return {"ok": True, "error": None}


def readouts(action_probs, scroll="down", app="", text="", target_el=None):
    """The server's `readouts`: option-named action probabilities plus the argument slots."""
    from ac_common import TO_OPTION

    r = {
        "action": {"choice": TO_OPTION[max(action_probs, key=action_probs.get)],
                   "probabilities": {TO_OPTION[k]: v for k, v in action_probs.items()}},
        "scroll_dir": {"choice": scroll, "probabilities": {scroll: 1.0}},
        "app_name": {"text": app},
        "type_text": {"text": text},
    }
    if target_el is not None:
        r["target_el"] = {"choice": max(target_el, key=target_el.get), "probabilities": target_el}
    return r


def v_click(point=(20, 15), probs=None, **kw):
    probs = probs or {"click": 0.6, "navigate_back": 0.3, "scroll": 0.1}
    return {"action": "click", "point": list(point), "track": "V",
            "probabilities": dict(probs), "readouts": readouts(probs, **kw)}


# --------------------------------------------------------------------------- track V
def test_masked_click_on_track_v_falls_back_to_the_next_action_type():
    env = ScreenEnv([SCREEN_A])  # nothing ever changes
    model = FakeModel([v_click(), v_click()])
    agent = AgentCore(model, env, mask=True)

    first = agent.step("save", i=0)
    second = agent.step("save", i=1)

    assert first["masked_from"] is None and first["mask_choice"] is None
    assert env.executed[0] == {"action_type": "click", "x": 20, "y": 15}
    # verified no effect -> the same click on the same screen is replaced, from the same pass
    assert env.executed[1] == {"action_type": "navigate_back"}
    assert "click" in second["masked_from"]
    assert second["mask_choice"]["via"] == "action"
    assert second["mask_choice"]["rank"] == 1  # rank 0 = click itself (spatial, no new point on V)
    assert second["action_raw"]["action"] == "click"  # the trace keeps what the model said
    assert second["action_executed"] == {"action_type": "navigate_back"}
    assert len(model.calls) == 2


def test_history_records_what_executed_after_masking():
    env = ScreenEnv([SCREEN_A])
    model = FakeModel([v_click(), v_click(), {"action": "wait"}])
    agent = AgentCore(model, env, mask=True)
    for i in range(3):
        agent.step("save", i=i)
    assert model.calls[2]["history"] == ["click", "navigate_back"]


def test_scroll_alternative_takes_its_direction_from_the_readout():
    probs = {"click": 0.6, "scroll": 0.3, "navigate_back": 0.1}
    env = ScreenEnv([SCREEN_A])
    agent = AgentCore(FakeModel([v_click(probs=probs, scroll="up")] * 2), env, mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    assert env.executed[1] == {"action_type": "scroll", "direction": "up"}
    assert agent.history[-1] == "scroll up"
    assert rec["mask_choice"]["rank"] == 1


def test_open_app_alternative_needs_a_name_in_the_readout():
    probs = {"click": 0.6, "open_app": 0.3, "navigate_home": 0.1}
    env = ScreenEnv([SCREEN_A])
    agent = AgentCore(FakeModel([v_click(probs=probs, app="")] * 2), env, mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    # no app name was read -> open_app's argument is unavailable -> next type
    assert env.executed[1] == {"action_type": "navigate_home"}
    assert rec["mask_choice"]["rank"] == 2


def test_a_click_that_changed_the_screen_is_never_masked():
    env = ScreenEnv([SCREEN_A, SCREEN_B, SCREEN_A])  # click works, back returns to A
    model = FakeModel([v_click(), {"action": "navigate_back"}, v_click()])
    agent = AgentCore(model, env, mask=True)
    recs = [agent.step("g", i=i) for i in range(3)]
    assert env.executed[2] == {"action_type": "click", "x": 20, "y": 15}
    assert all(r["masked_from"] is None for r in recs)


def test_masking_is_per_screen():
    other = [FRAME, SAVE, CANCEL, el("Banner", cls="android.widget.TextView", bbox=(0, 90, 100, 99))]
    env = ScreenEnv([SCREEN_A, SCREEN_A, other])
    # step 0 no effect on A; step 1 masked on A (-> navigate_back, which "changes" A into other)
    model = FakeModel([v_click(), v_click(), v_click()])
    agent = AgentCore(model, env, mask=True)
    recs = [agent.step("g", i=i) for i in range(3)]
    assert recs[1]["masked_from"] is not None
    assert recs[2]["masked_from"] is None, "a pair verified on screen A masked a different screen"
    assert env.executed[2]["action_type"] == "click"


def test_a_click_elsewhere_on_the_same_element_is_still_the_same_action():
    env = ScreenEnv([SCREEN_A])
    model = FakeModel([v_click(point=(12, 12)), v_click(point=(28, 18))])
    agent = AgentCore(model, env, mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    assert rec["masked_from"] is not None  # both points are inside `Save`


def test_without_an_element_the_point_is_quantised_to_a_40px_grid():
    # Below FRAME (y > 100) no element contains the point, so the point itself is the argument.
    env = ScreenEnv([SCREEN_A])
    model = FakeModel([v_click(point=(85, 105)), v_click(point=(95, 115)), v_click(point=(85, 125))])
    agent = AgentCore(model, env, mask=True)
    first = agent.step("g", i=0)
    assert first["click_target"] is None
    rec = agent.step("g", i=1)  # (95, 115) is in the same 40-px cell (2, 2) as (85, 105)
    assert rec["masked_from"] is not None
    rec = agent.step("g", i=2)  # (85, 125) is in cell (2, 3)
    assert rec["masked_from"] is None


def test_typing_a_different_text_into_the_same_field_is_a_different_action():
    field = el("", cls="android.widget.EditText", bbox=(0, 30, 100, 50), is_editable=True)
    env = ScreenEnv([[FRAME, field]])
    probs = {"type_text": 0.7, "navigate_back": 0.3}
    model = FakeModel([
        {"action": "type_text", "point": [50, 40], "text": "alpha", "track": "V",
         "readouts": readouts(probs, text="alpha")},
        {"action": "type_text", "point": [50, 40], "text": "beta", "track": "V",
         "readouts": readouts(probs, text="beta")},
    ])
    agent = AgentCore(model, env, mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    assert rec["masked_from"] is None
    assert env.executed[1]["text"] == "beta"


# --------------------------------------------------------------------------- track VA
def va_click(element_idx=1, point=(20.0, 15.0), target=None, probs=None):
    probs = probs or {"click": 0.8, "navigate_back": 0.2}
    target = target or {"1": 0.7, "2": 0.2}
    labels = {1: "Save", 2: "Cancel"}
    return {"action": "click", "point": list(point), "track": "VA",
            "element": {"idx": element_idx, "label": labels.get(element_idx), "confidence": 0.7},
            "element_point": list(point), "probabilities": dict(probs),
            "readouts": readouts(probs, target_el=target)}


def test_masked_element_on_track_va_falls_back_to_the_next_element():
    env = ScreenEnv([SCREEN_A])
    model = FakeModel([va_click(), va_click()])
    agent = AgentCore(model, env, track="VA", mask=True)

    agent.step("cancel", i=0)
    rec = agent.step("cancel", i=1)

    assert env.executed[0] == {"action_type": "click", "x": 20, "y": 15}  # Save
    assert env.executed[1] == {"action_type": "click", "x": 60, "y": 15}  # Cancel: next element
    assert rec["mask_choice"]["via"] == "target_el"
    assert rec["mask_choice"]["rank"] == 1
    assert rec["click_target"]["label"] == "Cancel"
    assert agent.history == ["click", "click"]


def test_va_falls_back_to_the_next_type_when_every_element_is_masked():
    env = ScreenEnv([SCREEN_A])
    model = FakeModel([va_click(), va_click(element_idx=2, point=(60.0, 15.0),
                                            target={"2": 0.7, "1": 0.2}), va_click()])
    agent = AgentCore(model, env, track="VA", mask=True)
    agent.step("g", i=0)   # Save: no effect
    agent.step("g", i=1)   # Cancel: no effect
    rec = agent.step("g", i=2)
    assert env.executed[2] == {"action_type": "navigate_back"}
    assert rec["mask_choice"]["via"] == "action"


def test_va_element_alternatives_are_refused_when_the_candidate_list_disagrees_with_the_server():
    """The harness rebuilds the server's numbered list; if the model's own pick does not match it,
    no element number can be trusted, so element alternatives are skipped (never guessed)."""
    env = ScreenEnv([SCREEN_A])
    wrong = va_click()
    wrong["element"] = {"idx": 1, "label": "Something else", "confidence": 0.7}
    model = FakeModel([va_click(), wrong])
    agent = AgentCore(model, env, track="VA", mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    assert env.executed[1] == {"action_type": "navigate_back"}
    assert rec["mask_choice"]["via"] == "action"
    assert rec["mask_choice"].get("note") == "candidates_mismatch"


# --------------------------------------------------------------------------- never masked / none left
def test_wait_is_never_masked():
    env = ScreenEnv([SCREEN_A])
    probs = {"wait": 0.6, "navigate_back": 0.4}
    dec = {"action": "wait", "track": "V", "readouts": readouts(probs)}
    agent = AgentCore(FakeModel([dec] * 3), env, mask=True)
    recs = [agent.step("g", i=i) for i in range(3)]
    assert [a["action_type"] for a in env.executed] == ["wait"] * 3
    assert all(r["masked_from"] is None for r in recs)


def test_wait_is_not_an_alternative():
    """The plan lists the alternative types and their argument sources; `wait` is not among them,
    and replacing a verified no-op by a no-op would only spend the step."""
    env = ScreenEnv([SCREEN_A])
    probs = {"click": 0.6, "wait": 0.3, "navigate_back": 0.1}
    agent = AgentCore(FakeModel([v_click(probs=probs)] * 2), env, mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    assert env.executed[1] == {"action_type": "navigate_back"}
    assert rec["mask_choice"]["rank"] == 2


def test_an_invalid_decision_that_ran_as_wait_is_never_masked():
    env = ScreenEnv([SCREEN_A])
    dec = {"action": "open_app", "app_name": "", "track": "V",
           "readouts": readouts({"open_app": 0.9, "navigate_back": 0.1})}
    agent = AgentCore(FakeModel([dec] * 2), env, mask=True)
    recs = [agent.step("g", i=i) for i in range(2)]
    assert [a["action_type"] for a in env.executed] == ["wait", "wait"]
    assert all(r["masked_from"] is None for r in recs)


def test_nothing_left_executes_the_original_and_says_so():
    env = ScreenEnv([SCREEN_A])
    # only spatial types ranked, on track V: no alternative has a new point
    probs = {"click": 0.7, "type_text": 0.3}
    agent = AgentCore(FakeModel([v_click(probs=probs)] * 2), env, mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    assert env.executed[1] == {"action_type": "click", "x": 20, "y": 15}
    assert rec["masked_from"] is not None
    assert rec["mask_choice"]["via"] == "none_left"
    assert rec["mask_choice"]["rank"] is None


def test_nothing_left_when_the_server_sent_no_readouts():
    env = ScreenEnv([SCREEN_A])
    dec = {"action": "click", "point": [20, 15], "track": "V"}
    agent = AgentCore(FakeModel([dec] * 2), env, mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    assert env.executed[1]["action_type"] == "click"
    assert rec["mask_choice"]["via"] == "none_left"


def test_a_masked_action_can_fall_back_to_termination():
    env = ScreenEnv([SCREEN_A])
    probs = {"click": 0.6, "status": 0.4}
    agent = AgentCore(FakeModel([v_click(probs=probs)] * 2), env, mask=True)
    agent.step("g", i=0)
    rec = agent.step("g", i=1)
    assert rec["done"] is True
    assert rec["masked_from"] is not None
    assert agent.history[-1] == "status"
    assert len(env.executed) == 1


# --------------------------------------------------------------------------- flag off
def _script():
    return [v_click(), v_click(), v_click()]


def test_flag_off_is_identical_to_today():
    env_off, env_default = ScreenEnv([SCREEN_A]), ScreenEnv([SCREEN_A])
    m_off, m_default = FakeModel(_script()), FakeModel(_script())
    off = AgentCore(m_off, env_off, mask=False)
    default = AgentCore(m_default, env_default)
    recs = [off.step("g", i=i) for i in range(3)]
    for i in range(3):
        default.step("g", i=i)
    assert env_off.executed == env_default.executed == [{"action_type": "click", "x": 20, "y": 15}] * 3
    assert off.history == default.history == ["click"] * 3
    assert all(r["masked_from"] is None and r["mask_choice"] is None for r in recs)
    # flag off must not change the request either
    assert [c["return_readouts"] for c in m_off.calls] == [False] * 3


def test_mask_on_requests_the_readouts_it_needs():
    env = ScreenEnv([SCREEN_A])
    model = FakeModel([v_click()])
    AgentCore(model, env, mask=True).step("g", i=0)
    assert model.calls[0]["return_readouts"] is True
