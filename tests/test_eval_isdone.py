# tests/test_eval_isdone.py
"""Task 4 — is_done AUROC and risk-coverage. Tests first, per the plan."""
import pytest

from benchmarks.train.eval_isdone import auroc, risk_coverage


def test_auroc_perfect_separation():
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0


def test_auroc_chance_is_half():
    assert abs(auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) - 0.5) < 1e-9


def test_auroc_inverted_is_zero():
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0


def test_auroc_one_positive():
    assert auroc([0.9, 0.1, 0.1], [1, 0, 0]) == 1.0


def test_auroc_all_ties_with_mixed_labels():
    # 2 pos, 2 neg, all tied -> 0.5 by the Mann-Whitney convention
    assert abs(auroc([0.3] * 4, [1, 1, 0, 0]) - 0.5) < 1e-12


def test_auroc_raises_on_single_class():
    with pytest.raises(ValueError):
        auroc([0.1, 0.9], [1, 1])


def test_risk_coverage_at_full_coverage_is_base_rate():
    labels = [1, 0, 1, 0]
    assert risk_coverage([0.9, 0.4, 0.3, 0.1], labels, 1.0) == pytest.approx(0.5)


def test_risk_coverage_at_half_keeps_the_half_most_confident():
    labels = [1, 1, 0, 0]
    assert risk_coverage([0.9, 0.8, 0.2, 0.1], labels, 0.5) == 0.0


def test_risk_coverage_worst_half():
    labels = [0, 0, 1, 1]
    assert risk_coverage([0.9, 0.8, 0.2, 0.1], labels, 0.5) == 1.0


def test_risk_coverage_single_row():
    assert risk_coverage([0.9], [1], 1.0) == 0.0
