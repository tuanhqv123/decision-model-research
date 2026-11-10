"""The served v2 request equals the trained v2 sample, byte for byte (the most important v2 test).

One AndroidWorld screen is built once (a synthetic accessibility forest) and expressed twice:
  * as the R2 corpus stores it — `aw_extract.extract_elements`, the function that wrote compact_*_v2.jsonl —
    and fed to the TRAINING path, `train_d1.make_sample`;
  * as AndroidWorld serves it to the harness — the vendored `forest_to_ui_elements` -> `awenv_bridge.
    _element_dicts` -> JSON over HTTP — and fed to the SERVING path, `serving/d1_api.py`'s real `/v1/decide`
    handler, with a stand-in model that records exactly what it is asked.
The state string and the questions must be identical. No GPU and no weights: `transformers` is replaced by a
stub before `d1_api` is imported, so the model the handler calls is the recorder.
"""
import base64
import importlib
import io
import json
import os
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "benchmarks", "train")
SERVING = os.path.join(ROOT, "serving")
sys.path.insert(0, TRAIN)

import ac_common as AC  # noqa: E402
import aw_extract as AX  # noqa: E402
from benchmarks.aw.awenv.awenv_bridge import _element_dicts  # noqa: E402
from benchmarks.train import train_d1  # noqa: E402

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

W, H = 1080, 2400


# ----------------------------------------------------------------------------- the stand-in model
class Recorder:
    """What `AutoModel.from_pretrained(...).cuda()` returns in these tests: records every request."""

    def __init__(self):
        self.requests = []
        self.engine = NS(calibration="untouched")
        self.action, self.choice = "click", "1"

    def cuda(self):
        return self

    def system_one_batch(self, reqs):
        self.requests.extend(reqs)
        out = []
        for _, qs, _ in reqs:
            ans = {"action": {"choice": self.action, "probabilities": {self.action: 0.9, "wait": 0.1},
                              "confidence": 0.9},
                   "scroll_dir": {"choice": "down", "confidence": 0.7}, "app_name": {"text": ""},
                   "type_text": {"text": ""}}
            if "target_box" in qs:
                ans["target_box"] = {"box": {"left": 0.1, "right": 0.2, "top": 0.3, "bottom": 0.4},
                                     "confidence": 0.5, "found": True}
            if "target_el" in qs:
                keys = list(qs["target_el"]["criteria"])
                ans["target_el"] = {"choice": self.choice, "confidence": 0.8,
                                    "probabilities": {k: (0.8 if k == self.choice else 0.2 / max(1, len(keys) - 1))
                                                      for k in keys}}
            out.append({"answers": ans})
        return out


def _load_api(monkeypatch, tmp_path, contract, tracks):
    ckpt = tmp_path / f"ckpt-{contract}"
    ckpt.mkdir()
    AC.write_contract(str(ckpt), contract, tracks=tracks)
    model = Recorder()
    stub = ModuleType("transformers")
    stub.AutoModel = NS(from_pretrained=lambda *a, **k: model)
    monkeypatch.setitem(sys.modules, "transformers", stub)
    monkeypatch.setenv("D1_MODEL", str(ckpt))
    monkeypatch.delenv("D1_MEM_FRAC", raising=False)
    monkeypatch.delenv("D1_API_TOKEN", raising=False)
    monkeypatch.syspath_prepend(SERVING)
    sys.modules.pop("d1_api", None)
    api = importlib.import_module("d1_api")
    return api, model


@pytest.fixture
def v2_api(monkeypatch, tmp_path):
    api, model = _load_api(monkeypatch, tmp_path, "v2", ["VA"])
    yield api, model
    sys.modules.pop("d1_api", None)


@pytest.fixture
def e2_api(monkeypatch, tmp_path):
    api, model = _load_api(monkeypatch, tmp_path, "e2", ["V"])
    yield api, model
    sys.modules.pop("d1_api", None)


def png_b64(w=W, h=H):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- one screen
def node(uid, box, child_ids=(), text="", cd="", hint="", cls="android.widget.TextView", pkg="com.example",
         **flags):
    l, t, r, b = box
    base = dict(unique_id=uid, bounds_in_screen=NS(left=l, top=t, right=r, bottom=b), child_ids=list(child_ids),
                text=text, content_description=cd, class_name=cls, hint_text=hint,
                is_checked=False, is_checkable=False, is_clickable=False, is_editable=False, is_enabled=True,
                is_focused=False, is_focusable=False, is_long_clickable=False, is_scrollable=False,
                is_selected=False, is_visible_to_user=True, package_name=pkg, view_id_resource_name="")
    base.update(flags)
    return NS(**base)


def screen():
    """A Markor-like editor screen: status bar, toolbar, a title field, a checkbox, 45 list rows (so the cap and
    the reading order matter), an invisible node, and a clickable container AndroidWorld drops."""
    nodes = [node(0, (0, 0, 1080, 2400), child_ids=list(range(1, 60))),
             node(1, (40, 0, 140, 60), text="9:06", pkg="com.android.systemui"),
             node(2, (900, 0, 1040, 60), cd="Battery 80 percent.", pkg="com.android.systemui"),
             node(3, (0, 80, 140, 220), cd="Navigate up", cls="android.widget.ImageButton", is_clickable=True),
             node(4, (160, 100, 900, 200), text="  New   note\n", cls="android.widget.EditText", is_editable=True,
                  is_focused=True, is_clickable=True, is_long_clickable=True),
             node(5, (920, 80, 1060, 220), cd="Save", cls="android.widget.ImageButton", is_clickable=True),
             node(6, (40, 240, 1040, 330), text="Pin to top", cls="android.widget.CheckBox", is_checkable=True,
                  is_checked=True, is_clickable=True),
             node(7, (0, 2200, 1080, 2400), child_ids=[8], is_clickable=True, cls="android.widget.LinearLayout"),
             node(8, (40, 2250, 1040, 2350), text="Delete", cls="android.widget.TextView"),
             node(9, (0, 0, 50, 50), text="ghost", is_visible_to_user=False)]
    for i in range(45):
        y = 360 + 40 * i
        nodes.append(node(100 + i, (40, y, 1040, y + 36), text=f"line {i}", cls="android.widget.TextView"))
    return NS(windows=[NS(tree=NS(nodes=nodes))])


GOAL = "Create a note titled shopping list in Markor and save it"
HISTORY = ["open_app Markor", "click", "type_text shopping list"]


def corpus_row(tmp_path, elements, point):
    Image.new("RGB", (W, H), (255, 255, 255)).save(tmp_path / "s.png")
    target = {"x_min": 920, "y_min": 80, "x_max": 1060, "y_max": 220}
    return {"ep": 1, "step": 3, "goal": GOAL, "history": list(HISTORY), "img": "s.png", "W": W, "H": H,
            "action": {"action_type": "click", "x": point[0], "y": point[1]}, "target": target, "cands": [],
            "elements": elements, "target_el_idx": AX.smallest_containing(elements, *point),
            "target_el_centre_idx": AX.smallest_containing(elements, *point)}


def served_elements(f):
    ui = AX.forest_to_ui_elements(f, exclude_invisible_elements=True)
    return json.loads(json.dumps(_element_dicts(NS(ui_elements=ui))))   # the HTTP body


# ----------------------------------------------------------------------------- THE parity test
def test_trained_and_served_v2_inputs_are_byte_identical(v2_api, tmp_path, monkeypatch):
    api, model = v2_api
    f = screen()
    rec = corpus_row(tmp_path, AX.extract_elements(f), (990, 150))
    monkeypatch.setitem(train_d1.RUN, "contract", "v2")
    t_state, t_qs, t_gold, _ = train_d1.make_sample(json.dumps(rec), str(tmp_path), "VA", None, None)

    with TestClient(api.app) as client:          # runs the startup warm-up too
        model.requests.clear()
        r = client.post("/v1/decide", json={"goal": GOAL, "history": HISTORY, "image_b64": png_b64(),
                                            "elements": served_elements(f), "track": "VA"})
    assert r.status_code == 200, r.text
    s_state, s_qs, _ = model.requests[-1]
    assert s_state == t_state
    assert json.dumps(s_qs) == json.dumps(t_qs)
    # and the screen really exercised the rule: system UI, the invisible node and the cap all bit
    kept_lines = [l for l in s_state.split("\n") if l.startswith("[")]
    assert len(kept_lines) == AC.ELEMENT_CAP
    assert "9:06" not in s_state and "ghost" not in s_state
    assert "New note | EditText | click,edit,long_click,focused" in s_state
    assert t_gold["target_el"] == next(k for k, v in t_qs["target_el"]["criteria"].items() if v == "Save")


def test_parity_holds_for_the_old_flat_request_shape_too(v2_api, tmp_path, monkeypatch):
    """An older client sends only `bbox` lists and one `clickable` boolean; flags it does not send are false on
    both sides only if the corpus has them false — so compare against a corpus row with the same facts."""
    api, model = v2_api
    f = NS(windows=[NS(tree=NS(nodes=[node(1, (920, 80, 1060, 220), cd="Save", cls="android.widget.ImageButton",
                                           is_clickable=True)]))])
    rec = corpus_row(tmp_path, AX.extract_elements(f), (990, 150))
    monkeypatch.setitem(train_d1.RUN, "contract", "v2")
    t_state, t_qs, _, _ = train_d1.make_sample(rec, str(tmp_path), "VA", None, None)
    flat = [{"text": "", "content_description": "Save", "class": "android.widget.ImageButton",
             "package_name": "com.example", "clickable": True, "bbox": [920, 80, 1060, 220]}]
    with TestClient(api.app) as client:
        model.requests.clear()
        r = client.post("/v1/decide", json={"goal": GOAL, "history": HISTORY, "image_b64": png_b64(),
                                            "nodes": flat, "track": "VA"})
    assert r.status_code == 200, r.text
    assert model.requests[-1][0] == t_state and model.requests[-1][1] == t_qs


# ----------------------------------------------------------------------------- the v2 response
def test_v2_response_point_is_the_chosen_elements_centre(v2_api):
    api, model = v2_api
    f = screen()
    with TestClient(api.app) as client:
        model.choice = "3"
        r = client.post("/v1/decide", json={"goal": GOAL, "history": HISTORY, "image_b64": png_b64(),
                                            "elements": served_elements(f), "track": "VA",
                                            "return_readouts": True})
    out = r.json()
    kept, keep_idx = AC.select_elements(served_elements(f), W, H, AC.CONTRACTS["v2"]["element_rule"])
    assert out["action"] == "click" and out["track"] == "VA"
    assert out["element"]["idx"] == 3 and out["element"]["label"] == AC.element_label(kept[2])
    assert out["element"]["confidence"] == 0.8 and out["element"]["aw_index"] == keep_idx[2]
    assert out["point"] == out["element_point"] == AC.element_centre(kept[2])
    assert out["contract"] == "v2" and out["element_rule"] == AC.CONTRACTS["v2"]["element_rule"]
    assert set(out["readouts"]["target_el"]["probabilities"]) == {str(i) for i in range(1, len(kept) + 1)}
    assert "box" not in out and "box_point" not in out


def test_v2_non_spatial_action_has_no_point(v2_api):
    api, model = v2_api
    with TestClient(api.app) as client:
        model.action = "navigate_back"
        out = client.post("/v1/decide", json={"goal": GOAL, "image_b64": png_b64(),
                                              "elements": served_elements(screen()), "track": "VA"}).json()
    assert out["action"] == "navigate_back" and out["point"] is None


def test_v2_request_without_elements_is_refused_with_a_clear_400(v2_api):
    api, model = v2_api
    with TestClient(api.app) as client:
        model.requests.clear()
        for body in ({"goal": GOAL, "image_b64": png_b64(), "track": "VA"},
                     {"goal": GOAL, "image_b64": png_b64(), "track": "VA", "elements": []},
                     {"goal": GOAL, "image_b64": png_b64()}):
            r = client.post("/v1/decide", json=body)
            assert r.status_code == 400
            assert "elements" in r.json()["detail"]
        assert model.requests == []   # refused before the model was asked anything


def test_v2_request_on_the_v_track_is_refused(v2_api):
    api, _ = v2_api
    with TestClient(api.app) as client:
        r = client.post("/v1/decide", json={"goal": GOAL, "image_b64": png_b64(), "track": "V",
                                            "elements": served_elements(screen())})
    assert r.status_code == 400 and "VA" in r.json()["detail"]


def test_v2_health_reports_the_rule(v2_api):
    api, _ = v2_api
    with TestClient(api.app) as client:
        h = client.get("/health").json()
    assert h["contract"] == "v2" and h["element_rule"] == AC.CONTRACTS["v2"]["element_rule"]


# ----------------------------------------------------------------------------- v1-family behaviour unchanged
def test_e2_checkpoint_is_served_exactly_as_before(e2_api):
    api, model = e2_api
    with TestClient(api.app) as client:
        model.requests.clear()
        out = client.post("/v1/decide", json={"goal": "Open the clock", "history": ["open_app Clock"],
                                              "image_b64": png_b64(108, 240)}).json()
    state, qs, _ = model.requests[-1]
    assert state == AC.build_state("Open the clock", "V", None, None, ["open_app Clock"])
    assert qs == AC.build_questions("V", None, contract="e2")
    assert out["box_point"] == [(0.1 + 0.2) / 2 * 108, (0.3 + 0.4) / 2 * 240]
    assert out["point"] == out["box_point"] and "element" not in out and "contract" not in out
