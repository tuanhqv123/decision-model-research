"""score_curated (2026-11-16): the AndroidControl-Curated rules, re-implemented from the authors' src/eval/utils.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_curated as S  # noqa: E402


def row(gt, box=None, ep=1, step=0, **kw):
    r = {"image": "android_control_images/%d/step_%d.png" % (ep, step), "gt_action": gt, "gt_max_bbox": box}
    r.update(kw)
    return r


def pred(a, value=None, pt=None, ep=1, step=0):
    return {(ep, step): {"pred": a, "value": value, "pt": pt}}


def test_a_click_is_right_when_its_point_is_inside_the_gold_box():
    rows = [row("click", [100, 100, 200, 200])]
    assert S.score(rows, pred("click", pt=[150, 150]))[0]["sr"] == 1
    assert S.score(rows, pred("click", pt=[250, 150]))[0]["sr"] == 0
    assert S.score(rows, pred("long_press", pt=[150, 150]))[0]["sr"] == 0


def test_our_scroll_direction_maps_to_their_opposite_swipe():
    # content scrolls down = the finger swipes up (their correct_type)
    assert S.score([row("swipe:up")], pred("scroll", "down"))[0]["sr"] == 1
    assert S.score([row("swipe:up")], pred("scroll", "up"))[0]["sr"] == 0
    assert S.score([row("swipe:left")], pred("scroll", "right"))[0]["sr"] == 1


def test_text_and_app_names_match_by_containment_or_token_f1():
    assert S.score([row("input_text:flora")], pred("type_text", "Flora"))[0]["sr"] == 1
    assert S.score([row("open_app:Cx File Explorer")], pred("open_app", "Cx File Explorer"))[0]["sr"] == 1
    assert S.score([row("input_text:red shoes size 9")], pred("type_text", "blue hat"))[0]["sr"] == 0


def test_a_candidate_action_counts_only_on_the_improved_file():
    r = [row("click", [0, 0, 10, 10], candidate_actions=[{"action_type": "long_press", "action_bounds": [0, 0, 10, 10]}])]
    p = pred("long_press", pt=[5, 5])
    assert S.score(r, p, improved=True)[0]["sr"] == 1
    assert S.score(r, p, improved=False)[0]["sr"] == 0


def test_a_revised_row_is_scored_with_the_rerun_prediction():
    r = [row("click", [0, 0, 10, 10], revised_task="a clearer task")]
    old, new = pred("click", pt=[50, 50]), pred("click", pt=[5, 5])
    assert S.score(r, old, improved=True, revised=new)[0]["sr"] == 1
    assert S.score(r, old, improved=True, revised={})[0]["missing"] == 1
