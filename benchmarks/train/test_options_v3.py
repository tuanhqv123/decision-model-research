"""Contract v3 option texts: what d1 actually shows the model for each `choice`.

d1-2B renders a choice in its default `desc` style as `<code> <description>` (prompt.py `_option_line`): the option NAME
is never shown. So the description must carry the meaning. Found 2026-11-11:
  * the scroll descriptions said the opposite of the label (AC/AW `down` reveals content below: content moves UP,
    measured on 158 rows, CONTRACT-v3-io.md §5);
  * element options dropped the `[i]` id of the state list, and unlabelled/duplicate ones carried pixel centres while the
    state shows percentages;
  * action options did not use the verbs the history lines use (`click`, `type_text`, `status`).
v1/v2/v2a keep their texts (their checkpoints were trained on them).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common as A  # noqa: E402


def _kept():
    raw = [{"text": t, "class_name": c, "package_name": "com.app", "is_clickable": True,
            "bbox_pixels": {"x_min": 10 * i, "y_min": 100 * i, "x_max": 10 * i + 50, "y_max": 100 * i + 60}}
           for i, (t, c) in enumerate([("Send", "android.widget.Button"), ("", "android.widget.ImageButton"),
                                       ("", "android.widget.ImageButton"), ("OK", "android.widget.Button")])]
    return [A.canonical_element(e) for e in raw]


def _q(contract):
    return A.build_questions("VA", _kept(), contract, apps=["clock", "settings"])


def test_v3_scroll_texts_say_what_androidworld_does():
    s = _q("v3")["scroll_dir"]["criteria"]
    assert set(s) == {"up", "down", "left", "right"}
    assert "reveal content below" in s["down"] and "content moves up" in s["down"]
    assert "reveal content above" in s["up"] and "content moves down" in s["up"]
    assert "on the right" in s["right"] and "content moves left" in s["right"]
    assert "on the left" in s["left"] and "content moves right" in s["left"]


def test_v3_element_options_are_the_state_line_heads():
    kept = _kept()
    opts = _q("v3")["target_el"]["criteria"]
    lines = A.render_elements(kept, 1080, 2400)
    assert list(opts) == ["1", "2", "3", "4"]
    for i, text in opts.items():
        assert text.startswith(f"[{i}] ")
        assert lines[int(i) - 1].startswith(text)  # the option is the start of the same element's state line
    assert len(set(opts.values())) == len(opts)  # unlabelled twins are told apart by their id


def test_v3_action_texts_use_the_history_verbs():
    a = _q("v3")["action"]["criteria"]
    assert list(a) == list(A.ACTIONS)  # names (the gold keys) unchanged
    assert a["click"].startswith("click:") and a["write"].startswith("type_text:")
    assert a["terminate"].startswith("status:") and a["open_app"].startswith("open_app:")
    assert a["navigate_back"].startswith("navigate_back:") and a["scroll"].startswith("scroll:")


def test_older_contracts_keep_their_texts():
    q = _q("v2a")
    assert q["scroll_dir"]["criteria"] == A.SCROLL
    assert q["action"]["criteria"] == {k: A.ACTIONS[k] for k in A.CONTRACTS["v2a"]["actions"]}
    assert q["target_el"]["criteria"] == A.option_texts(_kept())


def test_current_app_names_an_androidworld_app_like_the_app_list():
    gm = [A.canonical_element({"text": "Send", "class_name": "android.widget.Button", "package_name": "com.google.android.gm",
                               "bbox_pixels": {"x_min": 0, "y_min": 0, "x_max": 9, "y_max": 9}})]
    assert A.current_app(gm) == "gmail (com.google.android.gm)"
    other = [dict(gm[0], package_name="com.ticktick.task")]
    assert A.current_app(other) == "com.ticktick.task"


def test_androidworld_package_map_is_the_emulator_app_list():
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aw", "awenv", "aw_apps.json")
    assert A.AW_APP_BY_PACKAGE == {x["package"]: x["name"] for x in json.load(open(path))}


def test_v3_status_text_says_only_when_the_goal_is_achieved():
    t = A.ACTIONS_V3["terminate"]
    assert t.startswith("status:") and "fully achieved" in t and "on this screen" in t


def test_v3_target_el_instruction_points_at_the_element_list():
    assert "[id]" in _q("v3")["target_el"]["instructions"]
    assert _q("v2a")["target_el"]["instructions"] == "The element the next action acts on."
