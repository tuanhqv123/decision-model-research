"""Q3 measurement fields (PLAN-2026-11-10-question-tree §3, Q3).

Of the no-effect clicks, how many land on NO interactive element (grounding) versus on a present
but wrong or unresponsive element (semantics)? Answering that needs, per step, the element under
the click and the pre-action element list with its state flags. These fields are MEASUREMENT ONLY:
every test that adds one also checks that the executed action and the history are unchanged.
"""
from types import SimpleNamespace

import numpy as np

import awenv.awenv_bridge as bridge
from awenv.agent import AgentCore, compact_element, element_under
from awenv.awenv_bridge import AndroidWorldEnv, _element_dicts

STATE_FLAGS = ("is_checked", "is_checkable", "is_selected", "is_focused", "is_long_clickable",
               "is_enabled")


# --------------------------------------------------------------------------- fakes
class FakeModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, goal, history=None, pixels=None, **kw):
        self.calls.append({"goal": goal, "history": list(history or []), **kw})
        dec = dict(self.decisions.pop(0))
        dec.setdefault("confidence", 0.7)
        dec.setdefault("latency_ms", {"model": 100.0})
        return dec


def el(text="", cls="android.widget.Button", bbox=(0, 0, 10, 10), **flags):
    """An element in the flat form `awenv_bridge._element_dicts` produces."""
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
    for k, v in flags.items():
        d[k] = v
    if "is_clickable" in flags:
        d["clickable"] = flags["is_clickable"]
    return d


class ScreenEnv:
    """Element lists per observation: screens[k] is shown after k executed actions."""

    def __init__(self, screens):
        self.screens = [list(s) for s in screens]
        self.executed = []

    def _k(self):
        return min(len(self.executed), len(self.screens) - 1)

    def screenshot(self):
        return np.full((40, 30, 3), self._k(), dtype=np.uint8)

    def elements(self):
        return [dict(e) for e in self.screens[self._k()]]

    def execute(self, action):
        self.executed.append(action)
        return {"ok": True, "error": None}


# --------------------------------------------------------------------------- bridge
def _aw_element(**over):
    base = dict(
        text="Wi-Fi", content_description="", class_name="android.widget.Switch",
        package_name="com.android.settings", hint_text=None,
        bbox_pixels=SimpleNamespace(x_min=10, y_min=20, x_max=110, y_max=60),
        is_visible=True, is_clickable=True, is_editable=False, is_scrollable=False,
        is_checked=True, is_checkable=True, is_selected=False, is_focused=True,
        is_long_clickable=False, is_enabled=True, resource_name="android:id/switch_widget",
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_bridge_carries_the_state_flags_it_used_to_drop():
    out = _element_dicts(SimpleNamespace(ui_elements=[_aw_element()]))[0]
    assert out["is_checked"] is True
    assert out["is_checkable"] is True
    assert out["is_selected"] is False
    assert out["is_focused"] is True
    assert out["is_long_clickable"] is False
    assert out["is_enabled"] is True
    assert out["resource_name"] == "android:id/switch_widget"


def test_bridge_keeps_every_existing_key():
    out = _element_dicts(SimpleNamespace(ui_elements=[_aw_element()]))[0]
    for key in ("id", "text", "content_description", "class", "package_name", "clickable", "bbox",
                "bbox_pixels", "is_visible", "is_clickable", "is_editable", "is_scrollable",
                "hint_text"):
        assert key in out, key
    assert out["bbox"] == [10, 20, 110, 60]
    assert out["bbox_pixels"]["width"] == 100


def test_bridge_flags_are_plain_bools_when_androidworld_leaves_them_unset():
    """UIElement fields default to None upstream; the trace must stay JSON-clean and typed."""
    bare = SimpleNamespace(text="x", bbox_pixels=None)
    out = _element_dicts(SimpleNamespace(ui_elements=[bare]))[0]
    for key in STATE_FLAGS:
        assert isinstance(out[key], bool), key
    assert out["resource_name"] == ""


# --------------------------------------------------------------------------- compact form
def test_compact_element_label_class_bbox_flags():
    e = el("  A very long label that goes on and on past forty characters  ",
           cls="android.widget.CheckBox", bbox=(1, 2, 3, 4),
           is_clickable=True, is_checkable=True, is_checked=True, is_focused=True)
    label, cls, bbox, flags = compact_element(e)
    assert len(label) <= 40 and label.startswith("A very long label")
    assert cls == "CheckBox"
    assert bbox == [1, 2, 3, 4]
    assert flags == "ckxf"


def test_compact_label_falls_back_to_description_then_hint():
    e = el("", content_description="Navigate up")
    assert compact_element(e)[0] == "Navigate up"
    e = el("", hint_text="Search")
    assert compact_element(e)[0] == "Search"


def test_compact_flags_use_the_documented_letters_in_order():
    e = el(is_clickable=True, is_editable=True, is_scrollable=True, is_checkable=True,
           is_checked=True, is_long_clickable=True, is_focused=True, is_selected=True)
    assert compact_element(e)[3] == "ceskxlfn"


# --------------------------------------------------------------------------- element under a point
def test_element_under_point_is_the_smallest_container():
    els = [el("screen", cls="android.widget.FrameLayout", bbox=(0, 0, 100, 100)),
           el("Save", bbox=(10, 10, 30, 20), is_clickable=True),
           el("row", cls="android.widget.LinearLayout", bbox=(0, 5, 100, 25))]
    target = element_under(els, 15, 15)
    assert target["idx"] == 1
    assert target["label"] == "Save"
    assert target["class_short"] == "Button"
    assert target["flags"] == "c"
    assert target["bbox"] == [10, 10, 30, 20]


def test_element_under_point_is_none_when_nothing_contains_it():
    els = [el("Save", bbox=(10, 10, 30, 20))]
    assert element_under(els, 200, 200) is None


# --------------------------------------------------------------------------- step record
def test_step_records_the_pre_action_element_list():
    before = [el("Save", bbox=(10, 10, 30, 20), is_clickable=True)]
    after = [el("Saved!", cls="android.widget.TextView", bbox=(0, 0, 30, 10))]
    env = ScreenEnv([before, after])
    rec = AgentCore(FakeModel([{"action": "click", "point": [15, 15]}]), env).step("save", i=0)
    assert rec["elements_compact"] == [["Save", "Button", [10, 10, 30, 20], "c"]]


def test_click_target_and_interactive_flag_for_a_click_on_a_button():
    els = [el("screen", cls="android.widget.FrameLayout", bbox=(0, 0, 100, 100)),
           el("Save", bbox=(10, 10, 30, 20), is_clickable=True)]
    env = ScreenEnv([els])
    rec = AgentCore(FakeModel([{"action": "click", "point": [15, 15]}]), env).step("save", i=0)
    assert rec["click_target"] == {"idx": 1, "label": "Save", "class_short": "Button",
                                   "flags": "c", "bbox": [10, 10, 30, 20]}
    assert rec["click_on_interactive"] is True


def test_click_on_static_text_is_not_interactive():
    els = [el("Title", cls="android.widget.TextView", bbox=(0, 0, 100, 20))]
    env = ScreenEnv([els])
    rec = AgentCore(FakeModel([{"action": "click", "point": [50, 10]}]), env).step("g", i=0)
    assert rec["click_target"]["label"] == "Title"
    assert rec["click_on_interactive"] is False


def test_interactive_counts_any_container_not_only_the_smallest():
    """A tap on a label inside a clickable row reaches the row's handler."""
    els = [el("row", cls="android.widget.LinearLayout", bbox=(0, 0, 100, 40), is_clickable=True),
           el("Wi-Fi", cls="android.widget.TextView", bbox=(5, 5, 50, 30))]
    env = ScreenEnv([els])
    rec = AgentCore(FakeModel([{"action": "click", "point": [10, 10]}]), env).step("g", i=0)
    assert rec["click_target"]["label"] == "Wi-Fi"
    assert rec["click_on_interactive"] is True


def test_click_on_empty_space_has_no_target():
    els = [el("Save", bbox=(10, 10, 30, 20), is_clickable=True)]
    env = ScreenEnv([els])
    rec = AgentCore(FakeModel([{"action": "click", "point": [90, 90]}]), env).step("g", i=0)
    assert rec["click_target"] is None
    assert rec["click_on_interactive"] is False


def test_input_text_and_long_press_carry_a_target_too():
    field = el("", cls="android.widget.EditText", bbox=(0, 0, 100, 20), is_editable=True)
    env = ScreenEnv([[field]])
    agent = AgentCore(FakeModel([{"action": "type_text", "point": [5, 5], "text": "hi"},
                                 {"action": "long_press", "point": [5, 5]}]), env)
    rec = agent.step("g", i=0)
    assert rec["click_target"]["class_short"] == "EditText"
    assert rec["click_on_interactive"] is True
    rec = agent.step("g", i=1)
    assert rec["click_target"]["class_short"] == "EditText"


def test_actions_without_a_point_have_null_target_fields():
    env = ScreenEnv([[el("Save", bbox=(10, 10, 30, 20), is_clickable=True)]])
    agent = AgentCore(FakeModel([{"action": "scroll", "direction": "down"},
                                 {"action": "navigate_back"},
                                 {"action": "status"}]), env)
    for i in range(3):
        rec = agent.step("g", i=i)
        assert rec["click_target"] is None
        assert rec["click_on_interactive"] is None
        assert rec["elements_compact"] is not None


def test_measurement_fields_do_not_change_what_executes_or_the_history():
    els = [el("Save", bbox=(10, 10, 30, 20), is_clickable=True)]
    env = ScreenEnv([els])
    model = FakeModel([{"action": "click", "point": [15, 15]}, {"action": "click", "point": [15, 15]}])
    agent = AgentCore(model, env)
    agent.step("g", i=0)
    agent.step("g", i=1)
    assert env.executed == [{"action_type": "click", "x": 15, "y": 15}] * 2
    assert model.calls[1]["history"] == ["click"]
    assert model.calls[0]["elements"] is None  # V track still sends no elements


def test_elements_compact_comes_from_the_same_state_as_the_screenshot(monkeypatch):
    """One `State` per observation (H5): no extra `get_state` and no post-action leak."""
    monkeypatch.setattr(bridge, "_to_json_action", lambda a: a)

    class Inner:
        def __init__(self):
            self.calls = []

        def get_state(self, wait_to_stabilize=False):
            self.calls.append(wait_to_stabilize)
            n = len(self.calls)
            return SimpleNamespace(
                pixels=np.zeros((20, 20, 3), dtype=np.uint8),
                ui_elements=[_aw_element(text="state-%d" % n)])

        def execute_action(self, action):
            pass

    inner = Inner()
    env = AndroidWorldEnv(inner)
    rec = AgentCore(FakeModel([{"action": "click", "point": [50, 40]}]), env).step("g", i=0)
    assert [e[0] for e in rec["elements_compact"]] == ["state-1"]
    assert rec["click_target"]["label"] == "state-1"
    assert inner.calls == [True, True], "the measurement added a get_state call"


# --------------------------------------------------------------------------- toggle blindness (F2)
def test_a_pure_toggle_now_registers_as_a_ui_change():
    """Flipping a switch changes `is_checked` and nothing else; it is an effect, not a no-op."""
    off = [el("Wi-Fi", cls="android.widget.Switch", bbox=(0, 0, 100, 20),
              is_clickable=True, is_checkable=True, is_checked=False)]
    on = [el("Wi-Fi", cls="android.widget.Switch", bbox=(0, 0, 100, 20),
             is_clickable=True, is_checkable=True, is_checked=True)]
    env = ScreenEnv([off, on])
    rec = AgentCore(FakeModel([{"action": "click", "point": [50, 10]}]), env).step("wifi on", i=0)
    assert rec["ui_changed"] is True


def test_an_unchanged_screen_is_still_no_change():
    same = [el("Wi-Fi", cls="android.widget.Switch", bbox=(0, 0, 100, 20), is_checkable=True)]
    env = ScreenEnv([same, same])
    rec = AgentCore(FakeModel([{"action": "click", "point": [50, 10]}]), env).step("g", i=0)
    assert rec["ui_changed"] is False


def test_step_record_with_measurement_fields_is_json_serialisable():
    import json

    els = [el("Save", bbox=(10, 10, 30, 20), is_clickable=True)]
    env = ScreenEnv([els])
    rec = AgentCore(FakeModel([{"action": "click", "point": [15, 15]}]), env).step("g", i=0)
    rec.pop("screenshot_pixels")
    json.dumps(rec)
