"""eval_d1.pick_action (2026-11-14): the harness rule that turns the action question's probabilities into the action
taken. `status_threshold`: the executor reports "local goal done" only when p(status) reaches the threshold; otherwise
the best other option of the same forward pass is taken (its readouts come from the same pass)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_d1 as E  # noqa: E402

P = {"click": 0.30, "terminate": 0.60, "wait": 0.10}


def test_without_rules_the_model_choice_stands():
    assert E.pick_action(P, "terminate") == ("terminate", 0.60)


def test_status_below_the_threshold_falls_back_to_the_best_other_option():
    assert E.pick_action(P, "terminate", status_threshold=0.7) == ("click", 0.30)


def test_status_at_or_above_the_threshold_is_kept():
    assert E.pick_action(P, "terminate", status_threshold=0.6) == ("terminate", 0.60)


def test_the_threshold_never_creates_a_status():
    assert E.pick_action(dict(P, click=0.65, terminate=0.25), "click", status_threshold=0.1) == ("click", 0.65)


def test_excluded_options_are_never_taken():
    assert E.pick_action(P, "terminate", exclude_actions=("terminate",)) == ("click", 0.30)
