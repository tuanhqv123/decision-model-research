"""AC-High scoring of v3 predictions with InfiGUI's own scorer (vendored verbatim, benchmarks/train/vendor/infigui_eval.py).

Each v3 decision is mapped to InfiGUI's action dict; the scorer then compares it with their `step_check_pams`, which
were verified step by step against Google's raw actions (8,444/8,444 equal, 2026-11-11)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_actest import score, to_infigui_pred  # noqa: E402
from vendor.infigui_eval import evaluate_android_control_action, predict_direction  # noqa: E402

W, H = 1080, 2400


def _ok(pred, gt):
    return evaluate_android_control_action(pred, gt, W, H, W, H, pred_type="abs_resized", gt_type="abs_resized")


def test_click_inside_the_gold_box_is_an_exact_match_and_a_miss_is_not():
    gt = {"action": "click", "coordinate": [540, 1200], "candidate_bbox": [[500, 1150, 600, 1250]]}
    assert _ok(to_infigui_pred("click", None, [550, 1210]), gt) == (True, True)
    assert _ok(to_infigui_pred("click", None, [100, 100]), gt) == (True, False)
    assert _ok(to_infigui_pred("click", None, None), gt) == (True, False)  # no element chosen: never a hit


def test_each_scroll_label_is_read_back_by_their_direction_rule():
    for d in ("up", "down", "left", "right"):
        p = to_infigui_pred("scroll", d, None)
        assert p["action"] == "swipe" and predict_direction(p["coordinate"], p["coordinate2"]) == d
        assert _ok(p, {"action": "swipe", "direction": d}) == (True, True)
    assert to_infigui_pred("scroll", None, None)["action"] != "swipe"  # no direction: not a swipe at all


def test_type_open_back_home_wait_and_status():
    assert _ok(to_infigui_pred("type_text", "Lucas Miller", None), {"action": "type", "text": "lucas miller"}) == (True, True)
    assert _ok(to_infigui_pred("open_app", "Gmail", None), {"action": "open", "text": "Gmail", "candidate_bbox": []}) == (True, True)
    assert _ok(to_infigui_pred("navigate_back", None, None), {"action": "system_button", "button": "Back"}) == (True, True)
    assert to_infigui_pred("navigate_home", None, None) == {"action": "system_button", "button": "Home"}
    assert _ok(to_infigui_pred("wait", None, None), {"action": "wait"}) == (True, True)
    for gt in ({"action": "wait"}, {"action": "type", "text": "x"}, {"action": "swipe", "direction": "up"}):
        assert _ok(to_infigui_pred("status", None, None), gt) == (False, False)


def test_score_counts_every_test_step_and_a_missing_prediction_is_wrong():
    test = [{"episode": {"episode_id": 7}, "width": W, "height": H,
             "step_check_pams": [{"action": "wait"}, {"action": "system_button", "button": "Back"}]}]
    preds = {(7, 0): to_infigui_pred("wait", None, None)}
    s = score(preds, test)
    assert s["steps"] == 2 and s["exact"] == 1 and s["type"] == 1 and s["missing"] == 1
    assert abs(s["exact_match_acc"] - 50.0) < 1e-9


def test_an_empty_text_or_app_name_is_never_credited():
    """InfiGUI's check_text credits `pred in gt`, and '' is in every string: an empty typed text or app name would score
    as correct. v3 answers 'nothing' when it cannot copy the text, so empty must be sent as a non-matching value."""
    assert _ok(to_infigui_pred("type_text", "", None), {"action": "type", "text": "hello"}) == (True, False)
    assert _ok(to_infigui_pred("type_text", None, None), {"action": "type", "text": "hello"}) == (True, False)
    assert _ok(to_infigui_pred("open_app", "  ", None), {"action": "open", "text": "Gmail", "candidate_bbox": []}) == (True, False)
