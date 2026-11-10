# tests/test_no_app_leak.py
"""No-answer-leak guarantees: APP_INTENT off by default, and launcher icon selection works without it."""
import pytest

from benchmarks.train import ac_common as A


def _cand(idx, label, x0=10, y0=10):
    return {"idx": idx, "label": label, "pos": "x:0-1% y:0-1%",
            "bbox": {"x_min": x0, "x_max": x0 + 40, "y_min": y0, "y_max": y0 + 40,
                     "width": 40, "height": 40, "area": 1600,
                     "center": [x0 + 20, y0 + 20]}}


def test_app_intent_is_off_by_default():
    """The hand-written keyword table must never inject an answer unless explicitly asked."""
    assert A.ALLOW_APP_INTENT is False


def test_build_state_never_adds_application_line_by_default():
    """build_state only adds the line when handed an app_intent — and the trainer no longer hands one."""
    st = A.build_state("Turn wifi on", "V", None, None, [])
    assert "Application for this task" not in st


def test_build_state_adds_line_only_when_explicitly_given():
    st = A.build_state("Turn wifi on", "V", None, None, [], app_intent="settings")
    assert "Application for this task: settings" in st


def test_gold_app_name_has_no_span_when_not_in_state():
    """With the leak off, an app name absent from the state gets NO read supervision."""
    rec = {"action": {"action_type": "open_app", "app_name": "Booking.com"},
           "W": 1080, "H": 2400, "step": 0, "n_steps": 3, "nodes": []}
    g = A.gold_answers(rec, "V", None, "Goal: find a flight to Hawaii")
    assert g.get("app_name") is None
    assert "_app_hint" not in rec or rec.get("_app_hint") is None


def test_gold_app_name_still_works_when_named_in_goal():
    rec = {"action": {"action_type": "open_app", "app_name": "Markor"},
           "W": 1080, "H": 2400, "step": 0, "n_steps": 3, "nodes": []}
    state = "Goal: Create a note in Markor"
    g = A.gold_answers(rec, "V", None, state)
    s, e = g["app_name"]
    assert state[s:e] == "Markor"


def test_launcher_detection_needs_a_launcher_package():
    assert A.is_launcher_screen({"nodes": [{"package_name": "com.google.android.apps.nexuslauncher"}]})
    assert not A.is_launcher_screen({"nodes": [{"package_name": "com.android.settings"}]})


def test_launcher_icon_target_finds_the_labeled_icon():
    rec = {"target": None, "action": {"action_type": "open_app", "app_name": "YouTube"},
           "nodes": [{"package_name": "com.google.android.apps.nexuslauncher"}]}
    cands = [_cand(1, "Gmail"), _cand(2, "YouTube"), _cand(3, "Photos")]
    got = A.launcher_icon_target(rec, cands)
    assert got is not None and got["idx"] == 2


def test_launcher_icon_target_refuses_when_not_launcher():
    rec = {"target": None, "action": {"action_type": "open_app", "app_name": "YouTube"},
           "nodes": [{"package_name": "com.google.android.youtube"}]}
    cands = [_cand(2, "YouTube")]
    assert A.launcher_icon_target(rec, cands) is None


def test_launcher_icon_target_refuses_when_absent_from_candidates():
    rec = {"target": None, "action": {"action_type": "open_app", "app_name": "Booking.com"},
           "nodes": [{"package_name": "com.google.android.apps.nexuslauncher"}]}
    cands = [_cand(1, "Gmail"), _cand(2, "YouTube")]
    assert A.launcher_icon_target(rec, cands) is None


def test_launcher_icon_supervises_target_el_and_box():
    """The icon becomes a real element label, not a name copied out of the state."""
    rec = {"target": None, "action": {"action_type": "open_app", "app_name": "YouTube"},
           "W": 1080, "H": 2400, "step": 0, "n_steps": 3,
           "nodes": [{"package_name": "com.google.android.apps.nexuslauncher"}]}
    cands = [_cand(1, "Gmail"), _cand(2, "YouTube", x0=200, y0=300)]
    g = A.gold_answers(rec, "VA", cands, "Goal: watch a video")
    assert g["target_el"] == "2"
    top, bottom, left, right = g["target_box"]
    assert abs(left - 200 / 1080) < 1e-9
    assert abs(top - 300 / 2400) < 1e-9
