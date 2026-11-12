"""aw_report (2026-11-12): AndroidWorld results by official difficulty, with average steps, like published tables."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import aw_report as R  # noqa: E402

META = [{"task_name": "A", "difficulty": "easy", "optimal_steps": "3", "tags": ["x"]},
        {"task_name": "B", "difficulty": "easy", "optimal_steps": "5", "tags": []},
        {"task_name": "C", "difficulty": "hard", "optimal_steps": "12", "tags": ["x"]},
        {"task_name": "D", "difficulty": "medium", "optimal_steps": "8", "tags": []}]


def test_by_difficulty_uses_score_sum_over_scored_tasks_and_reports_steps():
    res = {"A": {"score": 1.0, "n_steps": 4}, "B": {"score": 0.0, "n_steps": 10},
           "C": {"score": 0.5, "n_steps": 20}, "D": {"score": None, "n_steps": 0, "unscored": True}}
    t = R.table(res, META)
    assert t["easy"]["n"] == 2 and t["easy"]["score"] == 1.0 and abs(t["easy"]["sr"] - 50.0) < 1e-9
    assert t["easy"]["avg_steps"] == 7.0 and t["easy"]["avg_steps_success"] == 4.0 and t["easy"]["avg_optimal"] == 4.0
    assert t["hard"]["score"] == 0.5 and t["medium"]["n"] == 0 and t["medium"]["excluded"] == 1
    assert t["all"]["n"] == 3 and t["all"]["score"] == 1.5


BUD = {"A": {"complexity": 1.0}, "B": {"complexity": 1.2}, "C": {"complexity": 3.4}, "D": {"complexity": 6.0}}


def test_complexity_levels_are_ceil_of_complexity_capped_at_4():
    assert [R.level(BUD[t]["complexity"]) for t in "ABCD"] == [1, 2, 4, 4]


def test_step_win_rate_compares_two_runs_like_mobilerl_fig5():
    """A win = solving the task in fewer steps; categories All, C1-C4, BC (both correct), BW (both wrong), Others."""
    a = {"A": {"score": 1.0, "n_steps": 4}, "B": {"score": 1.0, "n_steps": 6}, "C": {"score": 0.0, "n_steps": 30},
         "D": {"score": 1.0, "n_steps": 9}}
    b = {"A": {"score": 1.0, "n_steps": 5}, "B": {"score": 1.0, "n_steps": 6}, "C": {"score": 0.0, "n_steps": 34},
         "D": {"score": 0.0, "n_steps": 20}}
    w = R.step_wins(a, b, BUD)
    assert w["BC"] == {"n": 2, "a_win": 1, "tie": 1, "b_win": 0}
    assert w["BW"] == {"n": 1, "a_win": 0, "tie": 1, "b_win": 0}       # both wrong: a tie (no completion)
    assert w["Others"] == {"n": 1, "a_win": 1, "tie": 0, "b_win": 0}   # only a solved D
    assert w["All"]["n"] == 4 and w["C1"]["n"] == 1 and w["C4"]["n"] == 2
