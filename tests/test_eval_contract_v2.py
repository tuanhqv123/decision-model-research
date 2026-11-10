"""eval_d1 under contract v2: the same state and questions as training, the point is the chosen element's
centre, scored by the existing `score_row`; rows without an element list are refused, not approximated."""
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "train"))

import ac_common as AC  # noqa: E402
import eval_d1  # noqa: E402
from run_acv2 import score_row  # noqa: E402

W, H = 1080, 2400
RULE = AC.CONTRACTS["v2"]["element_rule"]


def el(label="", box=(0, 0, 10, 10), pkg="com.example", **flags):
    x0, y0, x1, y1 = box
    d = {"text": label or None, "content_description": None, "hint_text": None,
         "class_name": "android.widget.Button", "package_name": pkg,
         "bbox_pixels": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}}
    for f in ("clickable", "editable", "scrollable", "checkable", "checked", "selected", "focused",
              "long_clickable"):
        d[f"is_{f}"] = bool(flags.get(f, False))
    return d


ELEMENTS = [el("9:06", (40, 0, 140, 60), pkg="com.android.systemui"),
            el("Cancel", (40, 2000, 500, 2100), clickable=True),
            el("Save", (580, 2000, 1040, 2100), clickable=True)]
TARGET = {"x_min": 580, "y_min": 2000, "x_max": 1040, "y_max": 2100}


def v2_rec():
    return {"ep": 5, "step": 2, "goal": "Save the note", "history": ["open_app Markor", "click"], "img": "s.png",
            "W": W, "H": H, "action": {"action_type": "click", "x": 800, "y": 2050}, "target": TARGET,
            "cands": [], "elements": ELEMENTS, "target_el_idx": 2, "target_el_centre_idx": 2}


class FakeModel:
    """`system_one_batch` with canned answers; records exactly what it was asked."""

    def __init__(self, choice="2", action="click"):
        self.choice, self.action, self.requests = choice, action, []

    def system_one_batch(self, reqs):
        self.requests.extend(reqs)
        out = []
        for state, qs, imgs in reqs:
            ans = {"action": {"choice": self.action, "probabilities": {self.action: 0.9}, "confidence": 0.9},
                   "scroll_dir": {"choice": "down"}, "app_name": {"text": ""}, "type_text": {"text": ""}}
            if "target_el" in qs:
                ans["target_el"] = {"choice": self.choice, "confidence": 0.8,
                                    "probabilities": {k: (0.8 if k == self.choice else 0.1) for k in
                                                      qs["target_el"]["criteria"]}}
            out.append({"answers": ans})
        return out


@pytest.fixture
def data(tmp_path):
    Image.new("RGB", (108, 240), (255, 255, 255)).save(tmp_path / "s.png")
    return str(tmp_path)


def test_val_row_carries_the_element_list(data):
    row = eval_d1.val_row(v2_rec(), data)
    assert row["elements"] == ELEMENTS
    assert "elements" not in eval_d1.val_row({k: v for k, v in v2_rec().items() if k != "elements"}, data)


def test_v2_eval_asks_the_training_questions_and_scores_the_chosen_elements_centre(data):
    rec = v2_rec()
    row = eval_d1.val_row(rec, data)
    model = FakeModel(choice="2")
    out, summ = eval_d1.run_eval(model, [row], "VA", None, contract="v2", element_rule=RULE)
    state, qs, _ = model.requests[0]
    want_state, kept, keep_idx = AC.element_inputs(rec["goal"], rec["history"], ELEMENTS, W, H, RULE)
    assert state == want_state
    assert qs == AC.build_questions("VA", kept, contract="v2")
    o = out[0]
    assert kept[1]["label"] == "Save"
    assert o["el_idx"] == 2 and o["aw_index"] == keep_idx[1] == 2
    assert o["pt"] == o["el_pt"] == [810.0, 2050.0]
    assert o["box_pt"] is None and "correct_box_only" not in o
    assert (o["correct"], o["grounding"]) == score_row(row, "click", None, [810.0, 2050.0])
    assert o["correct"] is True and summ["grounding"] == 1.0


def test_v2_eval_wrong_element_is_not_grounded(data):
    out, _ = eval_d1.run_eval(FakeModel(choice="1"), [eval_d1.val_row(v2_rec(), data)], "VA", None,
                              contract="v2", element_rule=RULE)
    assert out[0]["correct"] is False and out[0]["grounding"] is False


def test_v2_eval_non_spatial_prediction_reads_no_point(data):
    out, _ = eval_d1.run_eval(FakeModel(action="navigate_back"), [eval_d1.val_row(v2_rec(), data)], "VA", None,
                              contract="v2", element_rule=RULE)
    assert out[0]["pt"] is None


def test_v2_eval_refuses_rows_without_elements_and_the_v_track(data):
    row = eval_d1.val_row({k: v for k, v in v2_rec().items() if k != "elements"}, data)
    with pytest.raises(ValueError):
        eval_d1.run_eval(FakeModel(), [row], "VA", None, contract="v2", element_rule=RULE)
    with pytest.raises(ValueError):
        eval_d1.run_eval(FakeModel(), [eval_d1.val_row(v2_rec(), data)], "V", None, contract="v2",
                         element_rule=RULE)


def test_eval_track_defaults():
    assert eval_d1.resolve_track("v2", None) == "VA"
    assert eval_d1.resolve_track("v1", None) == "V"
    assert eval_d1.resolve_track("v1", "VA") == "VA"
    with pytest.raises(SystemExit):
        eval_d1.resolve_track("v2", "V")


def test_eval_rule_comes_from_the_checkpoint(tmp_path):
    AC.write_contract(str(tmp_path), "v2", tracks=["VA"])
    assert eval_d1.element_rule_for(str(tmp_path), "v2") == RULE
    assert eval_d1.element_rule_for(str(tmp_path), "v1") is None
    with pytest.raises(SystemExit):
        eval_d1.element_rule_for(str(tmp_path / "missing"), "v2")
