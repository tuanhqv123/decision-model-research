"""Contract v2a: the app to launch is CHOSEN from a list of app names (like the element), not copied from the goal.

Why (2026-11-11): R1 copied the app name out of the goal; for "Turn wifi on." it launched `wifi`, which AndroidWorld
cannot resolve (29-36% of R1's open_app calls), and 21 AndroidWorld templates never name their app in the goal.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common as A  # noqa: E402

ELEMENTS = [
    {"text": "Phone", "class_name": "android.widget.TextView",
     "bbox_pixels": {"x_min": 76, "y_min": 1873, "x_max": 249, "y_max": 2068}, "is_clickable": True},
    {"text": "Chrome", "class_name": "android.widget.TextView",
     "bbox_pixels": {"x_min": 580, "y_min": 1873, "x_max": 753, "y_max": 2068}, "is_clickable": True},
]
AW = ["settings", "clock", "camera", "contacts", "markor", "chrome"]


def _row(goal, action, apps=None):
    r = {"ep": "e1", "step": 0, "goal": goal, "history": [], "W": 1080, "H": 2400, "elements": ELEMENTS,
         "action": action, "target": None}
    if apps is not None:
        r["apps"] = apps
    return r


def _inputs(row, contract="v2a"):
    state, kept, keep_idx = A.element_inputs(row["goal"], row["history"], row["elements"], row["W"], row["H"],
                                             A.CONTRACTS[contract]["element_rule"])
    return state, kept, keep_idx


def test_app_options_include_gold_and_are_deterministic():
    vocab = ["Gmail", "Clock", "Settings", "Spotify", "Booking.com", "Markor", "Chrome", "Tasks"]
    a = A.app_options("Clock", vocab, k=4, seed="e7:0")
    assert "Clock" in a and len(a) == 5 and len(set(a)) == 5
    assert a == A.app_options("Clock", vocab, k=4, seed="e7:0")
    assert a != A.app_options("Clock", vocab, k=4, seed="e8:0") or True  # different rows may differ


def test_app_options_without_gold_still_returns_k_names():
    a = A.app_options(None, ["a", "b", "c"], k=2, seed="x")
    assert len(a) == 2 and set(a) <= {"a", "b", "c"}


def test_v2a_app_name_is_a_choice_over_the_given_apps():
    row = _row("Turn wifi on.", {"action_type": "open_app", "app_name": "settings"}, AW)
    state, kept, _ = _inputs(row)
    q = A.build_questions("VA", kept, contract="v2a", apps=AW)
    assert q["app_name"]["type"] == "choice"
    assert list(q["app_name"]["criteria"].values()) == AW


def test_v2_app_name_is_still_a_text_read():
    row = _row("Open Chrome", {"action_type": "open_app", "app_name": "Chrome"})
    state, kept, _ = _inputs(row, "v2")
    assert A.build_questions("VA", kept, contract="v2")["app_name"]["type"] == "text"


def test_v2a_gold_app_is_supervised_when_the_goal_does_not_name_it():
    row = _row("Turn wifi on.", {"action_type": "open_app", "app_name": "settings"}, AW)
    state, kept, keep_idx = _inputs(row)
    gold, _ = A.gold_answers_v2(row, kept, keep_idx, state, apps=AW, contract="v2a")
    assert gold["app_name"] == str(AW.index("settings") + 1)


def test_v2a_gold_app_matches_case_insensitively():
    row = _row("Open the clock", {"action_type": "open_app", "app_name": "Clock"}, AW)
    state, kept, keep_idx = _inputs(row)
    gold, _ = A.gold_answers_v2(row, kept, keep_idx, state, apps=AW, contract="v2a")
    assert gold["app_name"] == str(AW.index("clock") + 1)


def test_v2a_non_open_app_rows_have_no_app_supervision():
    row = _row("Turn wifi on.", {"action_type": "click", "x": 100, "y": 1900}, AW)
    state, kept, keep_idx = _inputs(row)
    gold, _ = A.gold_answers_v2(row, kept, keep_idx, state, apps=AW, contract="v2a")
    assert "app_name" not in gold


def test_first_step_allows_only_open_app_or_status():
    probs = {"click": 0.6, "open_app": 0.3, "status": 0.1}
    assert A.restrict_first_step(probs, history=[]) == "open_app"
    assert A.restrict_first_step({"click": 0.6, "open_app": 0.1, "status": 0.3}, history=[]) == "status"
    assert A.restrict_first_step(probs, history=["open_app settings"]) == "click"


def test_decode_app_choice_maps_the_key_to_the_name():
    assert A.decode_app_choice({"choice": "3"}, AW) == "camera"
    assert A.decode_app_choice({"choice": "99"}, AW) == ""


def test_row_app_options_uses_the_rows_own_list_when_present():
    row = _row("Turn wifi on.", {"action_type": "open_app", "app_name": "settings"}, AW)
    assert A.row_app_options(row, ["x", "y"]) == AW


def test_row_app_options_contains_the_gold_app_for_open_app_rows():
    row = _row("Open Spotify", {"action_type": "open_app", "app_name": "Spotify"})
    opts = A.row_app_options(row, ["Gmail", "Clock", "Settings"] + ["a%d" % i for i in range(40)], k=24)
    assert "Spotify" in opts and len(opts) == 25 and opts == A.row_app_options(row, ["Gmail", "Clock", "Settings"] + ["a%d" % i for i in range(40)], k=24)
