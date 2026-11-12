"""Self-play episodes -> training rows (2026-11-11). Three kinds of rows, all labelled without any external model:
* done:      the task checker says the task was already complete before this step -> gold `status`;
* imitate:   a step of an episode that SUCCEEDED, before the task became complete -> gold = the executed action;
* avoid:     the chosen element changed nothing on this screen, or `status` was declared while the checker said
             not done -> that option is pushed down (no positive gold)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selfplay_rows import element_at, episode_rows  # noqa: E402


def _el(i, text, box):
    x0, y0, x1, y1 = box
    return {"id": i, "text": text, "class": "android.widget.Button", "package_name": "com.app", "bbox": list(box),
            "bbox_pixels": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1, "center": [(x0 + x1) / 2, (y0 + y1) / 2]},
            "is_clickable": True, "is_visible": True}


ROOT = _el(0, "", (0, 0, 1080, 2400))
OK = _el(1, "OK", (100, 200, 300, 300))
NO = _el(2, "No", (500, 200, 700, 300))
ELS = [ROOT, OK, NO]


def _step(i, executed, probe, ui_changed=True, done=False, raw_action=None, hist=()):
    return {"type": "step", "i": i, "screenshot": f"/out/screens/ep1_{i:04d}.png", "probe_score": probe,
            "ui_changed": ui_changed, "done": done, "action_executed": executed,
            "action_raw": {"action": raw_action or (executed or {}).get("action_type")},
            "state_in": {"goal": "press OK", "history": list(hist), "elements": [dict(e) for e in ELS], "readouts": {}}}


def _episode(steps, success):
    return [{"type": "episode", "episode_id": "ep1", "task_name": "T"}] + steps + [
        {"type": "result", "episode_id": "ep1", "task_name": "T", "success": success, "score": 1.0 if success else 0.0}]


def test_element_at_finds_the_tapped_element_by_its_centre():
    assert element_at(ELS, 200, 250) == 1 and element_at(ELS, 600, 250) == 2


def test_success_episode_gives_imitation_rows_then_a_done_row():
    lines = _episode([_step(0, {"action_type": "click", "x": 200, "y": 250}, 1.0),
                      _step(1, None, 1.0, done=True, raw_action="status", hist=['click "OK" → screen changed'])], True)
    rows = episode_rows(lines, run="r", apps=["clock"])
    assert [r["label"] for r in rows] == ["imitate", "done"]
    assert rows[0]["action"] == {"action_type": "click", "x": 200, "y": 250} and rows[0]["target_el_idx"] == 1
    assert rows[0]["target"] == OK["bbox_pixels"] and rows[0]["img"] == "selfplay/r/screens/ep1_0000.png"
    assert rows[1]["action"]["action_type"] == "status" and rows[1]["history_v3"] == ['click "OK" → screen changed']
    assert rows[0]["W"] == 1080 and rows[0]["H"] == 2400 and rows[0]["apps"] == ["clock"]


def test_failed_episode_gives_avoid_rows_for_no_effect_clicks_and_a_premature_done():
    lines = _episode([_step(0, {"action_type": "click", "x": 600, "y": 250}, 0.0, ui_changed=False),
                      _step(1, None, 0.0, done=True, raw_action="status")], False)
    rows = episode_rows(lines, run="r", apps=["clock"])
    assert [r["label"] for r in rows] == ["avoid", "avoid"]
    assert rows[0]["avoid"] == {"target_el_idx": 2} and rows[1]["avoid"] == {"action": "status"}


def test_a_step_after_the_task_was_done_is_labelled_status_even_in_a_failed_episode():
    lines = _episode([_step(0, {"action_type": "click", "x": 200, "y": 250}, 1.0),
                      _step(1, {"action_type": "click", "x": 600, "y": 250}, 0.0)], False)
    rows = episode_rows(lines, run="r", apps=["clock"])
    assert [r["label"] for r in rows] == ["done"] and rows[0]["step"] == 1
