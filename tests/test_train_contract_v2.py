"""The trainer under contract v2 builds its state, questions and gold from the row's AndroidWorld element list
with the same functions the server calls, refuses every input v2 does not define, and stamps the rule."""
import argparse
import json
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "train"))

import ac_common as AC  # noqa: E402
from benchmarks.train import train_d1  # noqa: E402
from benchmarks.train.train_d1 import check_args, make_sample, stamp_contract  # noqa: E402

W, H = 1080, 2400


def el(label="", box=(0, 0, 10, 10), pkg="com.example", cls="android.widget.TextView", **flags):
    x0, y0, x1, y1 = box
    d = {"text": label or None, "content_description": None, "hint_text": None, "class_name": cls,
         "bbox_pixels": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}, "package_name": pkg,
         "is_visible": True, "is_enabled": True, "resource_name": None}
    for f in ("clickable", "editable", "scrollable", "checkable", "checked", "selected", "focused",
              "long_clickable"):
        d[f"is_{f}"] = bool(flags.get(f, False))
    return d


def v2_rec(tmp_path):
    Image.new("RGB", (108, 240), (255, 255, 255)).save(tmp_path / "s.jpg")
    elements = [el("9:06", (40, 0, 140, 60), pkg="com.android.systemui"),
                el("", (0, 300, 1080, 2200), cls="androidx.recyclerview.widget.RecyclerView", scrollable=True),
                el("Network & internet", (40, 400, 1040, 500), clickable=True),
                el("", (40, 400, 140, 500), cls="android.widget.ImageView")]
    return {"ep": 7, "step": 1, "goal": "Open network settings", "history": ["open_app Settings"],
            "img": "s.jpg", "W": W, "H": H, "cands": [],
            "action": {"action_type": "click", "x": 90, "y": 450},
            "target": {"x_min": 40, "y_min": 400, "x_max": 1040, "y_max": 500},
            "elements": elements, "target_el_idx": 3, "target_el_centre_idx": 3}


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setitem(train_d1.RUN, "contract", "v2")


def test_v2_sample_is_built_from_the_element_list(tmp_path, v2):
    rec = v2_rec(tmp_path)
    state, qs, gold, img = make_sample(rec, str(tmp_path), "VA", None, None)
    rule = AC.CONTRACTS["v2"]["element_rule"]
    want_state, kept, keep_idx = AC.element_inputs(rec["goal"], rec["history"], rec["elements"], W, H, rule)
    assert state == want_state
    assert qs == AC.build_questions("VA", kept, contract="v2")
    assert list(qs) == ["action", "target_el", "scroll_dir", "app_name", "type_text"]
    assert gold == AC.gold_answers_v2(rec, kept, keep_idx, state)[0]
    assert "target_box" not in gold and img.size == (720, 1600)


def test_v2_sample_accepts_the_jsonl_line(tmp_path, v2):
    rec = v2_rec(tmp_path)
    assert make_sample(json.dumps(rec), str(tmp_path), "VA", None, None)[0] == \
        make_sample(rec, str(tmp_path), "VA", None, None)[0]


def test_v2_state_never_carries_memory_lines(tmp_path, v2):
    class Memory:
        def query(self, goal, exclude_ep=None):
            return ["Settings"]
    state, _, _, _ = make_sample(v2_rec(tmp_path), str(tmp_path), "VA", Memory(), None)
    assert "Apps used for similar tasks" not in state


def test_v2_refuses_the_v_and_mix_tracks(tmp_path, v2):
    for track in ("V", "mix"):
        with pytest.raises(ValueError):
            make_sample(v2_rec(tmp_path), str(tmp_path), track, None, None)


def test_v2_refuses_rows_without_elements(tmp_path, v2):
    rec = v2_rec(tmp_path)
    del rec["elements"]
    with pytest.raises(ValueError):
        make_sample(rec, str(tmp_path), "VA", None, None)


def test_v2_refuses_auxiliary_box_samples(tmp_path, v2):
    aux = {"kind": "aux_label", "text": "OK", "img": "s.jpg", "W": W, "H": H,
           "target": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}}
    with pytest.raises(ValueError):
        make_sample(aux, str(tmp_path), "VA", None, None)


def test_v1_va_sample_is_unchanged(tmp_path, monkeypatch):
    """The v1 VA path still renders the old candidate list (no element list involved)."""
    monkeypatch.setitem(train_d1.RUN, "contract", "v1")
    rec = v2_rec(tmp_path)
    rec["cands"] = [{"idx": 1, "label": "OK", "pos": "x:1-2% y:3-4%",
                     "bbox": {"x_min": 40, "y_min": 400, "x_max": 1040, "y_max": 500, "center": [540, 450]}}]
    state, qs, gold, _ = make_sample(rec, str(tmp_path), "VA", None, None)
    assert "Elements on screen (id: label, position):" in state
    assert list(qs) == ["action", "target_box", "target_el", "scroll_dir", "app_name", "type_text"]
    assert gold["target_el"] == "1"


def test_v2_checkpoint_is_stamped_with_its_rule(tmp_path, v2):
    stamp_contract(str(tmp_path), "VA")
    doc = json.load(open(tmp_path / "contract.json"))
    assert doc["name"] == "v2" and doc["tracks"] == ["VA"]
    assert doc["element_rule"] == AC.CONTRACTS["v2"]["element_rule"]
    with pytest.raises(ValueError):
        stamp_contract(str(tmp_path), "mix")


def args(**kw):
    base = dict(contract="v2", track="VA", eval_every=0, traj_k=0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_v2_run_arguments_are_checked_before_any_model_is_loaded():
    check_args(args())
    for bad in (dict(track="V"), dict(track="mix"), dict(eval_every=50), dict(traj_k=2), dict(contract="nope")):
        with pytest.raises(SystemExit):
            check_args(args(**bad))


def test_v1_run_arguments_are_unchanged():
    check_args(args(contract="v1", track="mix", eval_every=50, traj_k=2))
