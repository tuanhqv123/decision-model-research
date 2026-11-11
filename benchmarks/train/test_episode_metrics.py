"""Whole-task view of a per-step validation file (2026-11-11): a goal counts as solved offline only if EVERY one of its
steps is right (each step still sees the demonstrator's correct history), and the first wrong step is placed at the
start, middle or end of the task."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episode_metrics import episode_metrics  # noqa: E402


def _row(ep, step, correct, gold="click", pred="click"):
    return {"episode_id": ep, "step": step, "correct": correct, "gold": {"action_type": gold}, "pred": pred}


def test_goal_solved_only_when_every_step_is_right_and_first_error_is_placed():
    rows = [_row(1, 0, True, "open_app", "open_app"), _row(1, 1, True), _row(1, 2, True, "status", "terminate"),
            _row(2, 0, False, "open_app", "click"), _row(2, 1, True), _row(2, 2, True, "status", "terminate"),
            _row(3, 0, True, "open_app", "open_app"), _row(3, 1, False), _row(3, 2, True), _row(3, 3, True, "status", "terminate"),
            _row(4, 0, True, "open_app", "open_app"), _row(4, 1, True), _row(4, 2, False, "status", "click")]
    m = episode_metrics(rows)
    assert m["steps"] == 13 and m["step_correct"] == 10
    assert m["goals"] == 4 and m["goals_all_correct"] == 1
    assert m["first_error_at"] == {"start": 1, "middle": 1, "end": 1}


def test_premature_status_is_counted_on_rows_whose_gold_is_not_status():
    rows = [_row(1, 0, False, "click", "terminate"), _row(1, 1, True, "status", "terminate")]
    assert episode_metrics(rows)["premature_status"] == 1


def test_premature_status_counts_the_eval_files_label_status():
    """eval_d1 writes the action as `status` (the AndroidWorld name), not the option name `terminate`."""
    rows = [_row(1, 0, False, "click", "status"), _row(1, 1, True, "status", "status")]
    assert episode_metrics(rows)["premature_status"] == 1
