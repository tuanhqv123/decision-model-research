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


def test_screenshot_saver_is_the_training_one_720_wide(tmp_path):
    """Test-set images must be made by the SAME code as the training images (720 px wide, bicubic, quality 90)."""
    import io
    from PIL import Image
    from preprocess_ac import IMG_W, save_screenshot
    buf = io.BytesIO()
    Image.new("RGB", (1080, 2400), (10, 200, 30)).save(buf, format="PNG")
    p = tmp_path / "0.jpg"
    save_screenshot(buf.getvalue(), str(p))
    im = Image.open(p)
    assert IMG_W == 720 and im.size == (720, 1600)


def test_test_split_only_in_eval_mode_and_only_into_an_actest_file():
    import pytest
    from build_ac_full_v2 import check_split_request
    with pytest.raises(SystemExit):
        check_split_request({"test"}, eval_test=False, out="x/compact_awlike_full_v3.jsonl")
    with pytest.raises(SystemExit):
        check_split_request({"test"}, eval_test=True, out="x/compact_awlike_full_v3.jsonl")
    with pytest.raises(SystemExit):
        check_split_request({"train", "test"}, eval_test=True, out="x/actest_raw.jsonl")
    check_split_request({"test"}, eval_test=True, out="x/actest_raw.jsonl")
    check_split_request({"train"}, eval_test=False, out="x/full.jsonl")
