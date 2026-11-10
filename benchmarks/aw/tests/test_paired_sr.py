"""The pre-registered R1-vs-B0' comparison (PRE-REG-2026-11-10-contract-v1 §4, A1, A3), on synthetic results."""
import pytest

from paired_sr import a1_runs_more_seeds, paired_counts, sign_test_two_sided


def _r(task, score, **kw):
    e = {"task_name": task, "score": score, "task_score": score, "declared_done": score is not None,
         "setup_error": None, "unscored": False, "env_crash": False}
    e.update(kw)
    return e


def test_sign_test_matches_hand_computed_binomial_values():
    assert sign_test_two_sided(0, 0) == 1.0
    assert sign_test_two_sided(3, 3) == 1.0
    assert sign_test_two_sided(6, 0) == pytest.approx(2 * 0.5 ** 6)              # 0.03125
    assert sign_test_two_sided(10, 2) == pytest.approx(2 * (1 + 12 + 66) / 4096)  # 0.0386
    assert sign_test_two_sided(2, 10) == sign_test_two_sided(10, 2)


def test_w_and_l_count_only_tasks_scored_in_both_arms():
    a = {"T1": _r("T1", 1.0), "T2": _r("T2", 0.0), "T3": _r("T3", 1.0), "T4": _r("T4", 1.0, setup_error="boom")}
    b = {"T1": _r("T1", 0.0), "T2": _r("T2", 1.0), "T3": _r("T3", 1.0), "T4": _r("T4", 0.0)}
    out = paired_counts(a, b, ["T1", "T2", "T3", "T4"])
    assert (out["w"], out["l"], out["both_success"], out["both_fail"]) == (1, 1, 1, 0)
    assert out["n_both_scored"] == 3
    assert out["excluded"] == ["T4"]


def test_partial_credit_is_not_a_success():
    """upstream prints 'Task Successful' iff agent_successful > 0.5 (suite_utils.py:263)"""
    a = {"T": _r("T", 0.5)}
    b = {"T": _r("T", 0.0)}
    assert paired_counts(a, b, ["T"])["w"] == 0


def test_a_task_missing_from_one_arm_is_reported_not_silently_dropped():
    out = paired_counts({"T1": _r("T1", 1.0)}, {}, ["T1"])
    assert out["missing"] == ["T1"]
    assert out["n_both_scored"] == 0


def test_a1_gate_is_six_tasks_in_either_direction():
    assert not a1_runs_more_seeds(5, 0)
    assert a1_runs_more_seeds(6, 0)
    assert a1_runs_more_seeds(0, 6)
    assert not a1_runs_more_seeds(8, 3)
