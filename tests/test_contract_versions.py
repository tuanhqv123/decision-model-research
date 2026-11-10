"""A checkpoint is only valid with the questions and state it was trained on (decision K3).

Measured 2026-11-10: `ckpt-E2-final` was trained with 8 action options and 5 questions; the serving
code then sent 9 options (`navigate_home` added) and 6 questions (`is_done` added) to every
checkpoint, E2 included. The fix is a named contract that travels with the checkpoint
(`contract.json`) and that train, eval and serve all build their questions from.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks", "train"))

from ac_common import (CONTRACTS, build_questions, load_contract, render_history_line,  # noqa: E402
                       write_contract)

E2_ACTIONS = ["click", "write", "long_press", "scroll", "wait", "navigate_back", "open_app", "terminate"]
E2_QUESTIONS = ["action", "target_box", "scroll_dir", "app_name", "type_text"]


def test_e2_contract_reproduces_the_questions_e2_was_trained_with():
    q = build_questions("V", contract="e2")
    assert list(q) == E2_QUESTIONS
    assert list(q["action"]["criteria"]) == E2_ACTIONS


def test_v1_contract_has_navigate_home_and_no_is_done():
    q = build_questions("V", contract="v1")
    assert list(q) == E2_QUESTIONS
    assert "navigate_home" in q["action"]["criteria"]
    assert "is_done" not in q


def test_unknown_contract_is_refused():
    with pytest.raises(KeyError):
        build_questions("V", contract="nope")


def test_contract_round_trips_through_the_checkpoint_dir(tmp_path):
    write_contract(str(tmp_path), "v1")
    assert json.load(open(tmp_path / "contract.json"))["name"] == "v1"
    assert load_contract(str(tmp_path)) == "v1"


def test_a_checkpoint_without_a_contract_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_contract(str(tmp_path))


@pytest.mark.parametrize("action,line", [
    ({"action_type": "click", "x": 3, "y": 4}, "click"),
    ({"action_type": "long_press"}, "long_press"),
    ({"action_type": "type_text", "text": "Floor lamp"}, "type_text Floor lamp"),
    ({"action_type": "input_text", "text": "Floor lamp"}, "type_text Floor lamp"),
    ({"action_type": "scroll", "direction": "down"}, "scroll down"),
    ({"action_type": "open_app", "app_name": "YouTube Music"}, "open_app YouTube Music"),
    ({"action_type": "navigate_home"}, "navigate_home"),
    ({"action_type": "navigate_back"}, "navigate_back"),
    ({"action_type": "wait"}, "wait"),
    ({"action_type": "status", "goal_status": "successful"}, "status"),
])
def test_one_renderer_for_train_and_serve_history(action, line):
    assert render_history_line(action) == line
    assert not line.lower().startswith("step ")


from ac_common import questions_for, read_contract  # noqa: E402


def test_contract_records_the_tracks_the_checkpoint_was_trained_on(tmp_path):
    write_contract(str(tmp_path), "v1", tracks=["V"])
    doc = read_contract(str(tmp_path))
    assert doc["name"] == "v1" and doc["tracks"] == ["V"]


def test_an_untrained_track_is_refused(tmp_path):
    """A V-only checkpoint never learned `target_el`; serving VA to it would read an untrained head."""
    write_contract(str(tmp_path), "e2", tracks=["V"])
    doc = read_contract(str(tmp_path))
    assert list(questions_for(doc, "V")) == E2_QUESTIONS
    with pytest.raises(ValueError):
        questions_for(doc, "VA", cands=[{"idx": 1, "label": "OK", "pos": "x:1-2% y:1-2%"}])
