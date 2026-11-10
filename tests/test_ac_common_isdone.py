# tests/test_ac_common_isdone.py
"""Task 2 — is_done noul slot and the escalate label. Tests first, per the plan."""
import json
import os
from pathlib import Path

import pytest

from benchmarks.train.ac_common import (
    ACTION_VOCAB,
    FROM_OPTION,
    TO_OPTION,
    build_questions,
    build_state,
    gold_answers,
    is_done_gold,
)


def test_action_vocab_includes_navigate_home():
    """Review Focus 2: navigate_home exists in raw (29 rows) and vanished from every compact_*."""
    assert "navigate_home" in ACTION_VOCAB
    assert len(ACTION_VOCAB) == 9


def test_action_vocab_maps_to_options():
    for name in ACTION_VOCAB:
        assert name in TO_OPTION, f"{name} has no option mapping"
        assert TO_OPTION[name] in FROM_OPTION
    # and the option set has no orphan the raw vocabulary cannot produce
    assert set(TO_OPTION.values()) == set(FROM_OPTION)


def test_is_done_gold_is_binary():
    assert is_done_gold(True) == "yes"
    assert is_done_gold(False) == "no"


def test_questions_include_is_done():
    # is_done belongs to the `episodic` contract only (withdrawn 2026-11-10; v1 drops it)
    q = build_questions("V", contract="episodic")
    assert "is_done" in q
    assert q["is_done"]["type"] == "noul"


def test_questions_include_navigate_home_in_action_criteria():
    q = build_questions("V")
    assert "navigate_home" in q["action"]["criteria"]


def test_is_done_is_not_redundant_with_action():
    """Review Focus 3: status rows and terminal rows are the SAME set — record it, do not hide it."""
    root = Path("backup-2026-10-29/GPU_HOST_C/acwork/acdata/records")
    if not root.exists():
        pytest.skip("raw corpus not present")
    files = sorted(root.glob("*.jsonl"))[:1]
    n_status = n_term = n_both = 0
    for f in files:
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                st = (r.get("action") or {}).get("action_type") == "status"
                term = r["step"] == r["n_steps"]
                n_status += st
                n_term += term
                n_both += (st and term)
    assert n_status == n_term, "status rows must be exactly the terminal rows"
    assert n_both == n_status
    # Redundancy is real: is_done is 100% predictable from `action == terminate`.
    # Task 4 therefore must report is_done AUROC **conditioned on action != terminate**.
    assert n_status > 0


def _cand(idx=1):
    return {"idx": idx, "label": "box", "pos": "x:0-1% y:0-1%",
            "bbox": {"x_min": 0, "x_max": 9, "y_min": 0, "y_max": 9,
                     "width": 9, "height": 9, "area": 81, "center": [4, 4]}}


def test_non_copyable_text_is_labelled_escalate_not_dropped():
    """Review Focus 5: current code does del g['type_text']; that teaches nothing."""
    state = "Goal: write something original"
    rec = {"action": {"action_type": "type_text", "text": "a sentence not in state"},
           "W": 1080, "H": 2400, "step": 0, "n_steps": 3}
    g = gold_answers(rec, "V", [_cand()], state)
    assert g.get("type_text") == "escalate"


def test_copyable_text_stays_a_span():
    state = "Goal: type 98104 please"
    rec = {"action": {"action_type": "type_text", "text": "98104"},
           "W": 1080, "H": 2400, "step": 0, "n_steps": 3}
    g = gold_answers(rec, "V", [_cand()], state)
    assert isinstance(g.get("type_text"), tuple)
    s, e = g["type_text"]
    assert state[s:e] == "98104"


def test_is_done_true_on_terminal_row():
    state = "Goal: do a thing"
    rec = {"step": 3, "n_steps": 3, "W": 1080, "H": 2400, "action": {"action_type": "status"}}
    assert gold_answers(rec, "V", [_cand()], state)["is_done"] == "yes"


def test_is_done_false_on_intermediate_row():
    state = "Goal: do a thing"
    rec = {"step": 1, "n_steps": 3, "W": 1080, "H": 2400, "action": {"action_type": "click"}}
    assert gold_answers(rec, "V", [_cand()], state)["is_done"] == "no"


def test_build_state_has_no_is_clickable():
    """Global constraint: never publish is_clickable to the model."""
    c = _cand()
    c["is_clickable"] = True
    st = build_state("g", "VA", cands=[c])
    assert "clickable" not in st.lower()
    assert "box" in st
