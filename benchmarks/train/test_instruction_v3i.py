"""Contract v3i (2026-11-11): v3 plus an optional `Step instruction:` line (AndroidControl's low-level step instruction,
`local_goal`). Trained on about half the rows (fixed per (episode, step)), so one checkpoint serves AC-Low (instruction
given), AC-High and AndroidWorld (no instruction)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common as A  # noqa: E402


def _el(text, box=(0, 0, 100, 100), pkg="com.app"):
    return {"text": text, "class_name": "android.widget.Button", "package_name": pkg, "is_clickable": True,
            "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box))}


def test_state_has_the_instruction_line_right_after_the_goal_only_when_given():
    s, _, _ = A.element_inputs_v3("Send hi", [], [_el("Send")], 1080, 2400, "cap40_reading_noime",
                                  instruction="Click on the Send button")
    lines = s.splitlines()
    assert lines[0] == "Goal: Send hi" and lines[1] == "Step instruction: Click on the Send button"
    assert lines[2].startswith("Progress:")
    s2, _, _ = A.element_inputs_v3("Send hi", [], [_el("Send")], 1080, 2400, "cap40_reading_noime")
    assert "Step instruction" not in s2 and s2.splitlines()[1].startswith("Progress:")


def test_instruction_rows_are_a_fixed_half_of_the_rows():
    picks = [A.instruction_included({"ep": e, "step": t}) for e in range(400) for t in range(5)]
    assert 0.45 < sum(picks) / len(picks) < 0.55
    assert A.instruction_included({"ep": 7, "step": 2}) == A.instruction_included({"ep": 7, "step": 2})


def test_contract_v3i_is_v3_plus_instruction():
    v3, v3i = A.CONTRACTS["v3"], A.CONTRACTS["v3i"]
    assert v3i.get("instruction") is True and not v3.get("instruction")
    assert {k: v for k, v in v3i.items() if k != "instruction"} == v3
    assert A.CONTRACT_TRACKS["v3i"] == ("VA",)


def test_training_sample_carries_the_instruction_only_on_selected_rows_of_v3i():
    import train_d1 as T
    T.load_image = lambda path, width=None: None
    T.RUN["app_vocab"] = ["Mail", "Clock"]
    base = {"step": 0, "n_steps": 2, "goal": "Send hi", "history": [], "history_v3": [], "img": "x.jpg",
            "W": 1080, "H": 2400, "action": {"action_type": "click", "x": 50, "y": 50},
            "target": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}, "elements": [_el("Send")],
            "target_el_idx": 0, "local_goal": "Click on the Send button"}
    import json
    on = next(e for e in range(100) if A.instruction_included({"ep": e, "step": 0}))
    off = next(e for e in range(100) if not A.instruction_included({"ep": e, "step": 0}))
    T.RUN["contract"] = "v3i"
    s_on = T.make_sample(json.dumps(dict(base, ep=on)), "/x", "VA", None, None)[0]
    s_off = T.make_sample(json.dumps(dict(base, ep=off)), "/x", "VA", None, None)[0]
    assert "Step instruction: Click on the Send button" in s_on and "Step instruction" not in s_off
    T.RUN["contract"] = "v3"
    assert "Step instruction" not in T.make_sample(json.dumps(dict(base, ep=on)), "/x", "VA", None, None)[0]
