"""Contract v4 (2026-11-12, docs/research/2026-11-12-V4-WORKLOG-vi.md):
* element rule `all_reading_noime`: contract v3's list without the 40 cap (New Event is never cut);
* `effect_changed_v4`: the "→ screen changed / no change" label ignores the focused flag and keyboard elements, because
  AndroidWorld never reports focus (0/18,470 elements) and a click on an already-focused field leaves the app unchanged;
* an `answer` action whose text is the chosen element's text, plus `answer_value` (element / a count 0-20 / True / False);
* `avoid_action`: rows that only penalise one action (unlikelihood), e.g. `status` after "click Save → no change"."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common as A  # noqa: E402


def _el(text, box, cls="android.view.View", pkg="com.example.app", **flags):
    d = {"text": text, "class_name": cls, "package_name": pkg, "is_visible": True,
         "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box))}
    d.update({"is_%s" % k: v for k, v in flags.items()})
    return d


def _month():
    cells = [_el("", (c * 154, 500 + r * 300, c * 154 + 150, 790 + r * 300), clickable=True)
             for r in range(6) for c in range(7)]
    return [_el("Search", (147, 149, 626, 275), cls="android.widget.EditText", editable=True)] + cells + [
        _el("New Event", (891, 2148, 1038, 2295), cls="android.widget.ImageButton", clickable=True)]


def test_v4_rule_keeps_every_candidate_including_new_event():
    kept, _ = A.select_elements(_month(), 1080, 2400, "all_reading_noime")
    assert len(kept) == 44 and "New Event" in [e["label"] for e in kept]
    ys = [(e["bbox"]["y_min"], e["bbox"]["x_min"]) for e in kept]
    assert ys == sorted(ys)


def test_v4_rule_equals_v3_rule_under_the_cap_and_drops_keyboard_keys():
    els = _month()[:20] + [_el("q", (0, 1600, 100, 1700), pkg="com.google.android.inputmethod.latin", clickable=True)]
    assert A.select_elements(els, 1080, 2400, "all_reading_noime")[1] == \
        A.select_elements(els, 1080, 2400, "cap40_reading_noime")[1]


def test_effect_v4_ignores_focus_and_keyboard_but_not_real_changes():
    field = _el("Title", (0, 300, 1080, 400), cls="android.widget.EditText", editable=True)
    focused = dict(field, is_focused=True)
    key = _el("q", (0, 1600, 100, 1700), pkg="com.google.android.inputmethod.latin", clickable=True)
    c = A.canonical_element
    before, after_focus, after_kb = [c(field)], [c(focused)], [c(field), c(key)]
    assert A.effect_changed(before, after_focus) and A.effect_changed(before, after_kb)      # v3 says "changed"
    assert not A.effect_changed_v4(before, after_focus) and not A.effect_changed_v4(before, after_kb)
    assert A.effect_changed_v4(before, [c(dict(field, text="Groceries"))])


def test_v4_actions_and_answer_questions():
    kept, _ = A.select_elements(_month()[:5], 1080, 2400, "all_reading_noime")
    qs = A.build_questions("VA", kept, contract="v4", apps=["clock", "markor"])
    assert "answer" in qs["action"]["criteria"] and "terminate" in qs["action"]["criteria"]
    opts = list(qs["answer_value"]["criteria"])
    assert opts[0] == "element" and opts[1:22] == [str(i) for i in range(21)] and opts[-2:] == ["True", "False"]
    # v3 is unchanged: no answer option, no answer_value question
    q3 = A.build_questions("VA", kept, contract="v3", apps=["clock", "markor"])
    assert "answer" not in q3["action"]["criteria"] and "answer_value" not in q3


def test_history_line_for_an_answer():
    assert A.history_entry_with_label({"action_type": "answer", "text": "100 Main St"}, None, None) == 'answer "100 Main St"'


def _row(action, target=None, **kw):
    els = [_el("Location", (40, 600, 400, 680)), _el("100 Main St Chicago", (420, 600, 1040, 680)),
           _el("Save", (800, 140, 1040, 260), cls="android.widget.Button", clickable=True)]
    r = {"ep": 1, "step": 3, "goal": "g", "W": 1080, "H": 2400, "elements": els, "history_v3": [],
         "action": action, "target": target, "target_el_idx": None}
    r.update(kw)
    return r


def test_gold_for_an_answer_row_points_at_the_answer_element():
    rec = _row({"action_type": "answer", "text": "100 Main St Chicago"},
               target={"x_min": 420, "y_min": 600, "x_max": 1040, "y_max": 680}, answer_value="element")
    state, kept, idx = A.element_inputs_v3("g", [], rec["elements"], 1080, 2400, "all_reading_noime")
    gold, _ = A.gold_answers_v2(rec, kept, idx, state, apps=["clock"], contract="v4")
    assert gold["action"] == "answer" and gold["answer_value"] == "element"
    assert kept[int(gold["target_el"]) - 1]["label"] == "100 Main St Chicago"


def test_gold_for_a_count_answer_and_an_avoid_row():
    rec = _row({"action_type": "answer", "text": "3"}, answer_value="3")
    state, kept, idx = A.element_inputs_v3("g", [], rec["elements"], 1080, 2400, "all_reading_noime")
    gold, _ = A.gold_answers_v2(rec, kept, idx, state, apps=["clock"], contract="v4")
    assert gold["action"] == "answer" and gold["answer_value"] == "3" and "target_el" not in gold
    av = _row(None, avoid_action="status")
    gold, _ = A.gold_answers_v2(av, kept, idx, state, apps=["clock"], contract="v4")
    assert gold == {"avoid_action": "terminate"}


def test_answer_text_is_the_full_text_of_the_chosen_element_or_the_value():
    long = "A delicious and healthy choice for any time of the day, with extra words"
    nodes = [_el("Description", (40, 600, 400, 680)), _el(long, (420, 600, 1040, 680))]
    kept, idx = A.select_elements(nodes, 1080, 2400, "all_reading_noime")
    ans = {"answer_value": {"choice": "element"}, "target_el": {"choice": "2"}}
    assert A.answer_text(ans, nodes, kept, idx) == ("element", long)        # not the 40-char label
    assert A.answer_text({"answer_value": {"choice": "7"}, "target_el": {"choice": "2"}}, nodes, kept, idx) == ("7", "7")
    assert A.answer_text({"answer_value": {"choice": "element"}}, nodes, kept, idx) == ("element", "")


# --- contract v5 (2026-11-12, plan fix-all Task 5): checkable elements say "on" / "off" explicitly ---------------------
# SystemBluetoothTurnOffVerify: the switch was already off, rendered only as "checkable" (no "checked"), and the model
# clicked it on. v3/v4 rendering must not change; v5 renders the state as a word.

def test_checkable_renders_on_and_off_only_under_v5():
    sw_off = _el("", (839, 675, 985, 801), cls="android.widget.Switch", checkable=True, checked=False, clickable=True)
    sw_on = dict(sw_off, is_checked=True)
    kept, _ = A.select_elements([sw_off], 1080, 2400, "all_reading_noime")
    v4_line = A.render_elements(kept, 1080, 2400)[0]
    assert "| click,checkable |" in v4_line and " off" not in v4_line                     # unchanged for v3/v4
    assert "| click,checkable,off |" in A.render_elements(kept, 1080, 2400, on_off=True)[0]
    kept_on, _ = A.select_elements([sw_on], 1080, 2400, "all_reading_noime")
    assert "| click,checkable,on |" in A.render_elements(kept_on, 1080, 2400, on_off=True)[0]
    assert A.CONTRACTS["v5"]["on_off"] is True and not A.CONTRACTS["v4"].get("on_off")
    s5, _, _ = A.element_inputs_v3("g", [], [sw_off], 1080, 2400, "all_reading_noime", on_off=True)
    s4, _, _ = A.element_inputs_v3("g", [], [sw_off], 1080, 2400, "all_reading_noime")
    assert "checkable,off" in s5 and "checkable,off" not in s4


# --- contract v5: unlabelled INTERACTIVE elements are named by their resource id (2026-11-12) --------------------------
# On arm A, 15% of interactive elements had no text and 92% of those had a resource id ('menu', 'overflow_menu_icon',
# 'completeBox', ...); the model saw "(no label)". Non-interactive containers keep no label (the list must not grow).

def test_rid_rule_names_unlabelled_interactive_elements_only():
    btn = dict(_el("", (900, 100, 1000, 200), cls="android.widget.ImageButton", clickable=True),
               resource_name="org.fossify.calendar:id/overflow_menu_icon")
    box = dict(_el("", (0, 300, 1080, 900), cls="android.widget.FrameLayout"), resource_name="app:id/main_container")
    txt = dict(_el("Save", (0, 1000, 300, 1100), cls="android.widget.Button", clickable=True), resource_name="app:id/save_btn")
    kept, _ = A.select_elements([btn, box, txt], 1080, 2400, "all_reading_noime_rid")
    labels = [e["label"] for e in kept]
    assert "[id: overflow menu icon]" in labels and "Save" in labels and len(kept) == 2
    kept4, _ = A.select_elements([btn, box, txt], 1080, 2400, "all_reading_noime")
    assert "" in [e["label"] for e in kept4]                                   # v4 unchanged
    assert A.CONTRACTS["v5"]["element_rule"] == "all_reading_noime_rid"


def test_rid_label_is_humanised_from_camel_and_snake_case():
    assert A.rid_label("com.app:id/miniPlayerPlayPauseButton") == "[id: mini player play pause button]"
    assert A.rid_label("completeBox") == "[id: complete box]"
    assert A.rid_label("") == ""


# --- contract v6: an unlabelled interactive element also shows the text on its own row (2026-11-12) -------------------
# Trace analysis of the failed AndroidWorld tasks: Settings switches are "[id: switch widget]" with the setting's name
# in a separate TextView on the same row (TurnOffWifiAndTurnOnBluetooth re-enabled Wi-Fi); Retro Music's per-song menu
# button has no label. The row text is shown in the element list and option texts ONLY: history labels stay v5's,
# because the training rows' history lines were rendered with v5 labels.

def test_v6_shows_row_text_for_unlabelled_interactive_elements_but_keeps_history_labels():
    sw = dict(_el("", (900, 1230, 1040, 1330), cls="android.widget.Switch", clickable=True, checkable=True),
              resource_name="android:id/switch_widget")
    name = _el("Airplane mode", (190, 1250, 520, 1310), cls="android.widget.TextView")
    far = _el("Internet", (190, 650, 400, 700), cls="android.widget.TextView")
    kept, _ = A.select_elements([far, name, sw], 1080, 2400, A.CONTRACTS["v6"]["element_rule"])
    s = next(e for e in kept if e["class_short"] == "Switch")
    line = [l for l in A.render_elements(kept, 1080, 2400, on_off=True) if "Switch" in l][0]
    assert "[id: switch widget] · Airplane mode | Switch" in line
    assert A.history_label(s) == "[id: switch widget]"                      # history unchanged from v5
    assert any("Airplane mode" in t for t in A.option_texts(kept).values() if "switch" in t)
    kept5, _ = A.select_elements([far, name, sw], 1080, 2400, A.CONTRACTS["v5"]["element_rule"])
    assert all("·" not in l for l in A.render_elements(kept5, 1080, 2400, on_off=True))     # v5 unchanged
    assert A.CONTRACTS["v6"] == dict(A.CONTRACTS["v5"], element_rule=A.CONTRACTS["v6"]["element_rule"])


def test_v6_row_text_ignores_labelled_elements_and_other_rows():
    btn = _el("Save", (800, 100, 1000, 200), cls="android.widget.Button", clickable=True)
    menu = dict(_el("", (980, 900, 1060, 980), cls="android.widget.ImageView", clickable=True), resource_name="app:id/menu")
    other = _el("Golden Days", (150, 300, 700, 360), cls="android.widget.TextView")
    kept, _ = A.select_elements([btn, menu, other], 1080, 2400, A.CONTRACTS["v6"]["element_rule"])
    lines = A.render_elements(kept, 1080, 2400)
    assert any(l.startswith("[1] Save |") or "] Save |" in l for l in lines)
    assert any("[id: menu] |" in l for l in lines)                          # no text on its row: nothing appended


def test_v6_row_text_is_not_given_to_large_scroll_containers():
    lst = dict(_el("", (0, 300, 1080, 1500), cls="android.widget.ScrollView", scrollable=True), resource_name="app:id/list")
    title = _el("Save to", (100, 320, 500, 380), cls="android.widget.TextView")
    kept, _ = A.select_elements([lst, title], 1080, 2400, A.CONTRACTS["v6"]["element_rule"])
    assert all("·" not in l for l in A.render_elements(kept, 1080, 2400))
