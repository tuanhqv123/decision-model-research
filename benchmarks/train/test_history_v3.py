"""Contract v3 state: the model is told where it is in the task — progress, current app, and every previous action
with the element it acted on and whether the screen changed. One renderer for training rows and the live agent.

Why (2026-11-11): the v1/v2 history was `click, click, click`; the model could not tell start, middle and end apart.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common as A  # noqa: E402


def _el(text, cls="android.widget.Button", box=(0, 0, 100, 100), pkg="com.app", **flags):
    d = {"text": text, "class_name": cls, "package_name": pkg,
         "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box)), "is_clickable": True}
    d.update({f"is_{k}": v for k, v in flags.items()})
    return d


def test_entry_click_with_label_and_change():
    e = A.canonical_element(_el("Send"))
    assert A.history_entry_v3({"action_type": "click"}, e, True) == 'click "Send" → screen changed'


def test_entry_icon_without_label_uses_its_type():
    e = A.canonical_element(_el("", cls="android.widget.ImageButton"))
    assert A.history_entry_v3({"action_type": "click"}, e, False) == "click [ImageButton] → no change"


def test_entry_type_text_names_the_field():
    e = A.canonical_element(_el("To", cls="android.widget.EditText"))
    line = A.history_entry_v3({"action_type": "type_text", "text": "abc@gmail.com"}, e, True)
    assert line == 'type_text "abc@gmail.com" into "To" → screen changed'


def test_entry_input_text_is_type_text_and_unknown_effect_has_no_suffix():
    assert A.history_entry_v3({"action_type": "input_text", "text": "hi"}, None, None) == 'type_text "hi"'


def test_entry_open_app_scroll_back_wait():
    assert A.history_entry_v3({"action_type": "open_app", "app_name": "clock"}, None, True) == \
        "open_app clock → screen changed"
    assert A.history_entry_v3({"action_type": "scroll", "direction": "down"}, None, False) == "scroll down → no change"
    assert A.history_entry_v3({"action_type": "navigate_back"}, None, True) == "navigate_back → screen changed"
    assert A.history_entry_v3({"action_type": "wait"}, None, None) == "wait"


def test_effect_ignores_system_ui_and_sees_a_toggle():
    a = [_el("Wi-Fi", checked=False), _el("12:00", pkg=A.SYSTEM_UI)]
    b = [_el("Wi-Fi", checked=False), _el("12:01", pkg=A.SYSTEM_UI)]
    c = [_el("Wi-Fi", checked=True), _el("12:01", pkg=A.SYSTEM_UI)]
    canon = lambda xs: [A.canonical_element(x) for x in xs]  # noqa: E731
    assert A.effect_changed(canon(a), canon(b)) is False
    assert A.effect_changed(canon(a), canon(c)) is True


def test_current_app_is_the_main_package_and_launcher_is_home():
    els = [A.canonical_element(_el("x", pkg="com.simplemobiletools.notes")) for _ in range(3)]
    els.append(A.canonical_element(_el("t", pkg=A.SYSTEM_UI)))
    assert A.current_app(els) == "com.simplemobiletools.notes"
    home = [A.canonical_element(_el("Phone", pkg="com.google.android.apps.nexuslauncher"))]
    assert A.current_app(home) == "home screen"


def test_state_v3_has_progress_app_numbered_history_and_cap():
    entries = ["click \"%d\" → screen changed" % i for i in range(15)]
    els = [_el("OK", pkg="com.app")]
    state, kept, _ = A.element_inputs_v3("Do it", entries, els, 1080, 2400, "cap40_reading")
    lines = state.splitlines()
    assert lines[0] == "Goal: Do it"
    assert lines[1] == "Progress: step 16, 15 actions done"
    assert lines[2] == "Current app: com.app"
    assert "(3 earlier actions not shown)" in state
    assert 'Step 4: click "3" → screen changed' in state and 'Step 3: click "2"' not in state
    assert state.rstrip().endswith(A.render_elements(kept, 1080, 2400)[-1])


def test_state_v3_first_step():
    state, _, _ = A.element_inputs_v3("Turn wifi on.", [], [_el("Phone", pkg="com.google.android.apps.nexuslauncher")],
                                      1080, 2400, "cap40_reading")
    assert "Progress: step 1, 0 actions done" in state and "Current app: home screen" in state
    assert "Previous actions: none" in state


def test_episode_entries_from_full_flow_rows():
    from add_history_v3 import episode_entries
    s0 = [_el("Notes app", cls="android.widget.TextView", box=(0, 0, 200, 200), pkg="com.google.android.apps.nexuslauncher")]
    s1 = [_el("Send", box=(0, 0, 200, 200), pkg="com.notes")]
    s2 = [_el("Sent!", cls="android.widget.TextView", box=(0, 300, 200, 400), pkg="com.notes")]
    rows = [
        {"ep": 1, "step": 0, "n_steps": 2, "W": 1080, "H": 2400, "elements": s0, "target": {"x_min": 0, "y_min": 0, "x_max": 200, "y_max": 200}, "target_el_idx": 0,
         "action": {"action_type": "click", "x": 100, "y": 100}},
        {"ep": 1, "step": 1, "n_steps": 2, "W": 1080, "H": 2400, "elements": s1, "target": {"x_min": 0, "y_min": 0, "x_max": 200, "y_max": 200}, "target_el_idx": 0,
         "action": {"action_type": "click", "x": 100, "y": 100}},
        {"ep": 1, "step": 2, "n_steps": 2, "W": 1080, "H": 2400, "elements": s2, "target": None, "target_el_idx": None,
         "action": {"action_type": "status", "goal_status": "successful"}},
    ]
    h = episode_entries(rows)
    assert h[0] == [] and h[1] == ['click "Notes app" → screen changed']
    assert h[2] == ['click "Notes app" → screen changed', 'click "Send" → screen changed']


def test_keyboard_is_not_the_current_app_and_not_in_the_list():
    ime = "com.google.android.inputmethod.latin"
    els = [_el("Send", pkg="com.google.android.gm")] + [_el(k, cls="android.widget.FrameLayout", box=(i * 10, 2000, i * 10 + 9, 2100), pkg=ime) for i, k in enumerate("qwertyuiop")]
    canon = [A.canonical_element(e) for e in els]
    assert A.current_app(canon) == "com.google.android.gm"
    kept, _ = A.select_elements(els, 1080, 2400, "cap40_reading_noime")
    assert [e["label"] for e in kept] == ["Send"]
    assert A.CONTRACTS["v3"]["element_rule"] == "cap40_reading_noime"
