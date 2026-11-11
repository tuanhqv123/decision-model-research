"""`eval_d1 --oracle`: the gold answers sent down the SAME decoding path as the model's (state, kept elements, app list,
FROM_OPTION, decode_app_choice, chosen_element -> element_centre). Its score is the ceiling of the v3 interface on a
benchmark, and a check that rows, element ids and the mapping to the benchmark line up before any model is scored."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_d1 as E  # noqa: E402


def _el(text, box, cls="android.widget.Button", pkg="com.app"):
    return {"text": text, "content_description": None, "hint_text": None, "class_name": cls, "package_name": pkg,
            "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box)), "is_clickable": True,
            "is_visible": True}


def _rec(action, target_el_idx=None, step=0):
    els = [_el("Cancel", (0, 0, 200, 100)), _el("Send", (800, 0, 1000, 100))]
    target = els[target_el_idx]["bbox_pixels"] if target_el_idx is not None else None  # real rows carry it
    return {"ep": 9, "step": step, "n_steps": 3, "goal": 'send "hi" with Mail', "history": [], "history_v3": [],
            "img": "imgs/9/0.jpg", "W": 1080, "H": 2400, "action": action, "target": target,
            "elements": els, "target_el_idx": target_el_idx, "apps": ["Mail", "Clock"]}


def _oracle(rec):
    rows = [E.val_row(rec, "/nonexistent")]
    out, _ = E._run_eval_elements(None, rows, 8, False, "v3", "cap40_reading_noime", lambda p: None, [])
    return out[0]


def test_oracle_click_lands_on_the_gold_element_centre():
    o = _oracle(_rec({"action_type": "click", "x": 900, "y": 50}, target_el_idx=1))
    assert o["pred"] == "click" and o["pt"] == [900.0, 50.0]


def test_oracle_open_app_names_the_gold_app_from_the_row_list():
    o = _oracle(_rec({"action_type": "open_app", "app_name": "Mail"}))
    assert o["pred"] == "open_app" and o["value"] == "Mail"


def test_oracle_scroll_and_type_text():
    assert _oracle(_rec({"action_type": "scroll", "direction": "down"}))["value"] == "down"
    o = _oracle(_rec({"action_type": "type_text", "text": "hi", "x": 900, "y": 50}, target_el_idx=1))
    assert o["pred"] == "type_text" and o["value"] == "hi"


def test_a_screen_with_an_empty_element_list_is_evaluated_like_in_training():
    """Training accepts `elements: []` (a screen with no a11y nodes); evaluation must too (AC test ep 1959 step 1)."""
    rec = dict(_rec({"action_type": "wait"}), elements=[])
    o = _oracle(rec)
    assert o["pred"] == "wait" and o["n_elements"] == 0


class _FakeModel:
    """Answers every row with the same readout: `status` most likely, `scroll` second."""
    def system_one_batch(self, reqs):
        ans = {"action": {"choice": "terminate", "confidence": 0.6,
                          "probabilities": {"terminate": 0.6, "scroll": 0.3, "click": 0.1}},
               "target_el": {"choice": "2"}, "scroll_dir": {"choice": "down"}, "type_text": {"text": ""}}
        return [{"answers": ans} for _ in reqs]


def test_excluded_action_is_replaced_by_the_most_likely_remaining_one():
    """AC-High has no termination step: `--exclude-actions terminate` takes the best other action of the same pass."""
    rows = [E.val_row(_rec({"action_type": "scroll", "direction": "down"}), "/nonexistent")]
    out, _ = E._run_eval_elements(_FakeModel(), rows, 8, False, "v3", "cap40_reading_noime", lambda p: None, [],
                                  exclude_actions=("terminate",))
    assert out[0]["pred"] == "scroll" and out[0]["value"] == "down" and abs(out[0]["conf"] - 0.3) < 1e-9
    out, _ = E._run_eval_elements(_FakeModel(), rows, 8, False, "v3", "cap40_reading_noime", lambda p: None, [])
    assert out[0]["pred"] == "status"  # default unchanged
