"""serving/gen_api.py with the engine mocked: the response carries every field d1_api always returns, the harness
client (awenv.model_client) reads it unchanged, and awenv.adapter maps it to the intended AndroidWorld action —
including a parse failure, which must surface as an INVALID action with the raw text, never a silent wait."""
import base64
import importlib.util
import io
import os
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "aw"))

from awenv.adapter import map_decision  # noqa: E402
from awenv.model_client import ModelClient  # noqa: E402

# Keys d1_api's /v1/decide returns on EVERY request (serving/d1_api.py, `out = {...}` and the unconditional
# assignments after it). The generative server must return all of them.
D1_ALWAYS = {"action", "confidence", "probabilities", "track", "screen", "box_confidence", "point", "gate_score",
             "latency_ms"}


@pytest.fixture(scope="module")
def api():
    spec = importlib.util.spec_from_file_location("gen_api_under_test", os.path.join(ROOT, "serving", "gen_api.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeEngine:
    max_new_tokens = 64

    def __init__(self):
        self.text, self.calls = "", []

    def run(self, goal, history, image):
        self.calls.append((goal, list(history), image.size))
        pieces = list(self.text) + ["<|im_end|>"]
        return {"text": self.text, "pieces": pieces, "probs": [0.8] * len(pieces), "ids": list(range(len(pieces))),
                "tokens_generated": len(pieces), "hit_eos": True, "greedy_consistent": True, "model_ms": 42.0}


@pytest.fixture()
def served(api, monkeypatch):
    from fastapi.testclient import TestClient
    eng = FakeEngine()
    monkeypatch.setitem(api.STATE, "engine", eng)
    monkeypatch.setitem(api.STATE, "contract", {"name": "gen_v1", "tracks": ["V"]})
    monkeypatch.setattr(api, "TOKEN", "t0k")
    return TestClient(api.app), eng


def _png(w=1080, h=2400):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decide(client, eng, text, w=1080, h=2400, **extra):
    eng.text = text
    r = client.post("/v1/decide", json={"goal": "Set a 5 minute timer", "history": ["open_app Clock"],
                                        "image_b64": _png(w, h), "track": "V", **extra},
                    headers={"x-api-key": "t0k"})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_reports_the_contract(served):
    client, _ = served
    h = client.get("/health").json()
    assert h["ok"] and h["contract"] == "gen_v1" and h["tracks"] == ["V"] and h["decoding"] == "greedy"
    assert h["img_width"] == 720 and h["max_new_tokens"] == 64


def test_click_response_has_every_d1_field_and_a_pixel_point(served):
    client, eng = served
    out = _decide(client, eng, '{"action_type":"click","x":500,"y":250}')
    assert D1_ALWAYS <= set(out)
    assert out["action"] == "click" and out["point"] == [540.0, 600.0] and out["screen"] == [1080, 2400]
    assert out["confidence"] == pytest.approx(0.8 ** 5) and out["gate_score"] == out["confidence"]
    assert out["confidence_source"] == "action_type_token_probability_product"
    assert out["probabilities"] is None and out["box_confidence"] is None
    assert out["tokens_generated"] == len('{"action_type":"click","x":500,"y":250}') + 1
    assert out["latency_ms"]["model"] == 42.0 and out["latency_ms"]["total"] >= 0
    assert out["parse_error"] is None and out["raw"] == '{"action_type":"click","x":500,"y":250}'
    # the engine saw the request's goal + history and train_d1.load_image's 720-px picture
    assert eng.calls[-1] == ("Set a 5 minute timer", ["open_app Clock"], (720, 1600))


def test_point_is_in_pixels_of_the_screenshot_actually_sent(served):
    client, eng = served
    out = _decide(client, eng, '{"action_type":"long_press","x":500,"y":250}', w=1440, h=3120)
    assert out["point"] == [720.0, 780.0] and out["screen"] == [1440, 3120]


@pytest.mark.parametrize("text,expected", [
    ('{"action_type":"click","x":500,"y":250}', [{"action_type": "click", "x": 540, "y": 600}]),
    ('{"action_type":"long_press","x":0,"y":999}', [{"action_type": "long_press", "x": 0, "y": 2398}]),
    ('{"action_type":"type_text","x":100,"y":100,"text":"5"}',
     [{"action_type": "input_text", "x": 108, "y": 240, "text": "5", "clear_text": True}]),
    ('{"action_type":"scroll","direction":"down"}', [{"action_type": "scroll", "direction": "down"}]),
    ('{"action_type":"open_app","app_name":"Clock"}', [{"action_type": "open_app", "app_name": "clock"}]),
    ('{"action_type":"navigate_back"}', [{"action_type": "navigate_back"}]),
    ('{"action_type":"navigate_home"}', [{"action_type": "navigate_home"}]),
    ('{"action_type":"wait"}', [{"action_type": "wait"}]),
])
def test_every_action_maps_through_the_harness_adapter(served, text, expected):
    client, eng = served
    m = map_decision(_decide(client, eng, text))
    assert not m.invalid and m.actions == expected


def test_status_ends_the_episode(served):
    client, eng = served
    m = map_decision(_decide(client, eng, '{"action_type":"status"}'))
    assert m.done and m.actions == [] and not m.invalid


@pytest.mark.parametrize("text,error", [
    ("Tap the timer tab.", "no_json_object"),
    ('{"action_type":"click","x":500', "invalid_json"),
    ('{"action_type":"swipe","direction":"up"}', "unknown_action_type"),
])
def test_a_parse_failure_is_an_invalid_action_with_the_raw_text(served, text, error):
    client, eng = served
    out = _decide(client, eng, text)
    assert out["action"] == "" and out["raw"] == text and out["parse_error"] == error
    assert out["confidence"] is None and out["point"] is None
    m = map_decision(out)
    assert m.invalid and m.reason == "unmappable_action:empty" and m.actions == [{"action_type": "wait"}]


def test_an_empty_type_text_is_flagged_by_the_adapter(served):
    client, eng = served
    m = map_decision(_decide(client, eng, '{"action_type":"type_text","x":1,"y":1,"text":""}'))
    assert m.invalid and m.reason == "text_missing"


def test_va_track_is_refused_and_a_bad_token_rejected(served):
    client, _ = served
    r = client.post("/v1/decide", json={"goal": "g", "image_b64": _png(), "track": "VA"}, headers={"x-api-key": "t0k"})
    assert r.status_code == 400 and "VA" in r.json()["detail"]
    r = client.post("/v1/decide", json={"goal": "g", "image_b64": _png()}, headers={"x-api-key": "nope"})
    assert r.status_code == 401


def test_readouts_on_request(served):
    client, eng = served
    out = _decide(client, eng, '{"action_type":"wait"}', return_readouts=True)
    assert set(out["readouts"]) == {"ids", "pieces", "probs"}


def test_the_harness_client_reads_the_response_unchanged(served, monkeypatch):
    client, eng = served
    import requests

    def post(url, data=None, headers=None, timeout=None):
        return client.post(url, content=data, headers=headers)

    monkeypatch.setattr(requests, "post", post)
    eng.text = '{"action_type":"type_text","x":500,"y":50,"text":"Floor lamp"}'
    dec = ModelClient(base_url="http://testserver", token="t0k").decide(
        goal="Search for a floor lamp", history=["open_app IKEA", "click"], pixels=__import__("numpy").zeros(
            (2400, 1080, 3), dtype="uint8"), elements=[{"bbox": [0, 0, 10, 10]}], track="V")
    assert dec["action"] == "type_text" and dec["text"] == "Floor lamp" and dec["point"] == [540.0, 120.0]
    assert dec["latency_ms"]["model"] == 42.0 and dec["wall_ms"] > 0
    assert eng.calls[-1][1] == ["open_app IKEA", "click"]
    m = map_decision(dec)
    assert m.actions == [{"action_type": "input_text", "x": 540, "y": 120, "text": "Floor lamp", "clear_text": True}]
