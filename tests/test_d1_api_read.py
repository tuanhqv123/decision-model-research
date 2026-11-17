"""A typed text is a read: DMA-2B points at a start and an end token of its own input (or at nothing). The API
returns the line of that input the span lies in, where the span sits in it, and the probability it was found, so the
page can show what was copied from where."""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_d1_api_v2 import AC, GOAL, HISTORY, W, H, png_b64, screen, served_elements, v2_api  # noqa: E402,F401


def _state_text(api, f):
    built = api.build_request(api.CONTRACT, GOAL, HISTORY, served_elements(f), W, H, "VA")
    st = built["state"]
    return st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, indent=2)


def test_a_typed_text_comes_with_its_source_line_span_and_found_probability(v2_api):
    api, model = v2_api
    f = screen()
    text = _state_text(api, f)
    s = text.index("shopping list")
    e = s + len("shopping list")
    inner = model.system_one_batch

    def with_read(reqs):
        out = inner(reqs)
        for o in out:
            o["answers"]["type_text"] = {"text": "shopping list", "span": [s, e], "found": 0.97}
        return out
    model.system_one_batch = with_read
    model.action = "write"
    with TestClient(api.app) as client:
        out = client.post("/v1/decide", json={"goal": GOAL, "history": HISTORY, "image_b64": png_b64(),
                                              "elements": served_elements(f), "track": "VA"}).json()
    rd = out["read"]
    assert out["text"] == "shopping list" and rd["found"] == 0.97 and rd["span"] == [s, e]
    a, b = rd["at"]
    assert rd["context"][a:b] == "shopping list" and "\n" not in rd["context"] and rd["clipped"] == [False, False]
    line_start = text.rfind("\n", 0, s) + 1
    assert rd["context"] == text[line_start:text.find("\n", e) if "\n" in text[e:] else len(text)][:len(rd["context"])]


def test_no_read_is_reported_when_the_model_pointed_at_nothing(v2_api):
    api, model = v2_api
    model.action = "write"                     # the fake's type_text has no span
    with TestClient(api.app) as client:
        out = client.post("/v1/decide", json={"goal": GOAL, "history": HISTORY, "image_b64": png_b64(),
                                              "elements": served_elements(screen()), "track": "VA"}).json()
    assert "read" not in out
