import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_ac_full_v2 import _need, assemble_rows  # noqa: E402


def test_every_built_step_becomes_a_row_with_image_path_and_oracle_fields():
    oracle = {0: {"action": {"action_type": "click", "x": 5, "y": 6}, "history": [], "target": {"x_min": 0}},
              1: {"action": {"action_type": "status", "goal_status": "successful"}, "history": ["click"], "target": None}}
    built = {0: {"v2": {"elements": [1], "target_el_idx": 0}}, 1: {"v2": {"elements": [], "target_el_idx": None}}}
    rows = assemble_rows(42, "train", "do it", [1080, 1080], [2400, 2400], oracle, built)
    assert [r["step"] for r in rows] == [0, 1]
    assert rows[0]["img"] == "imgs/42/0.jpg" and rows[1]["history"] == ["click"] and rows[0]["split"] == "train"
    assert rows[0]["target_el_idx"] == 0 and rows[1]["action"]["action_type"] == "status"


def test_need_carries_point_only_for_actions_with_coordinates():
    n = _need({0: {"action": {"action_type": "click", "x": 1, "y": 2}, "target": None},
               1: {"action": {"action_type": "scroll", "direction": "down"}, "target": None}})
    assert n[0]["pt"] == [1, 2] and n[1]["pt"] is None
