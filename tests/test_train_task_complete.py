# tests/test_train_task_complete.py
"""Task 3 — is_done loss wiring and the candidate-cap guard. Tests first, per the plan."""
import os

import pytest

from benchmarks.train.train_d1 import W_LOSS, check_cand_cap


def test_cap_below_median_candidates_is_refused(monkeypatch):
    """Review Focus 4: the exact defect that produced the false-negative VA result."""
    monkeypatch.setenv("D1_MAX_CANDS", "14")
    with pytest.raises(SystemExit, match="40"):
        check_cand_cap(track="VA")


def test_cap_at_or_above_40_is_allowed(monkeypatch):
    monkeypatch.setenv("D1_MAX_CANDS", "40")
    check_cand_cap(track="VA")


def test_cap_of_64_is_allowed(monkeypatch):
    monkeypatch.setenv("D1_MAX_CANDS", "64")
    check_cand_cap(track="mix")


def test_v_track_ignores_cap(monkeypatch):
    monkeypatch.setenv("D1_MAX_CANDS", "14")
    check_cand_cap(track="V")


def test_cap_of_zero_means_unlimited_and_is_allowed(monkeypatch):
    """D1_MAX_CANDS=0 is the 'no cap' default in ac_common; it must not be refused."""
    monkeypatch.setenv("D1_MAX_CANDS", "0")
    check_cand_cap(track="VA")


def test_loss_table_has_is_done():
    assert "is_done" in W_LOSS
    assert 0.0 < W_LOSS["is_done"] < 1.0
