"""Executor: DMA-2B, one forward pass per decision, over HTTP (`serving/d1_api.py` of the DMA-2B project).

`decide` sends the goal, the local goal (as the step instruction), the model's own previous actions, the screenshot and
the element list, and returns the decision with its probability. `to_device_action` turns it into a `Device.act` call.
`history_line` writes the previous-action line exactly as in training (action, element, whether the screen changed).
"""
import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from mobile_use.agents.dma import ac_common as v


class Executor:
    def __init__(self, url: str, token: str = "tok", timeout: float = 60.0, post=None):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        self._post = post        # tests inject a fake
        self.ms: List[float] = []

    def decide(self, goal: str, instruction: Optional[str], history: List[str], image,
               elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        req = {"goal": goal, "history": history, "image_b64": base64.b64encode(buf.getvalue()).decode(),
               "elements": elements, "track": "VA", "instruction": instruction}
        t0 = time.time()
        if self._post is not None:
            dec = self._post(req)
        else:
            r = requests.post(self.url + "/v1/decide", json=req, headers={"x-api-key": self.token}, timeout=self.timeout)
            r.raise_for_status()
            dec = r.json()
        self.ms.append((dec.get("latency_ms") or {}).get("model") or (time.time() - t0) * 1000)
        return dec


def probability(dec: Dict[str, Any]) -> float:
    return float((dec.get("probabilities") or {}).get(dec["action"], dec.get("confidence") or 0.0))


def element_label(dec: Dict[str, Any]) -> Optional[str]:
    el = dec.get("element") or {}
    return el.get("history_label") or el.get("label")


def to_device_action(dec: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    """(Device.act kind, parameters, short description) for one decision (not `status`)."""
    act, label = dec["action"], element_label(dec)
    pt = dec.get("element_point") or dec.get("point")
    if act in ("click", "long_press"):
        return act, ({"x": pt[0], "y": pt[1]} if pt else {}), "%s %s" % (act, label)
    if act == "type_text":
        p = {"text": dec.get("text") or ""}
        if pt:
            p.update(x=pt[0], y=pt[1])
        return "type", p, 'type "%s" into %s' % (p["text"], label)
    if act == "scroll":
        d = dec.get("direction") or "down"
        return "scroll", {"direction": d}, "scroll %s" % d
    if act == "open_app":
        return "open_app", {"app": dec.get("app_name") or ""}, "open_app %s" % dec.get("app_name")
    if act == "navigate_back":
        return "back", {}, "navigate_back"
    if act == "navigate_home":
        return "home", {}, "navigate_home"
    return "wait", {"seconds": 2.0}, act


MULTILINE_PX = 250     # an editable field at least this tall is a text body (a note), not a one-line input


def text_to_type(field: Optional[Dict[str, Any]], typed: str) -> str:
    """The text the field should hold after typing. DMA-2B gives the words to type; how they combine with what is
    already there depends on the field, as for a person: a one-line input (name, phone, search) is replaced, while a
    text body that already has content gets the words on a new line (otherwise "add a line" would erase the note).
    The device then sets the field to exactly this text."""
    if not field:
        return typed
    current = str(field.get("text") or "")
    hint = str(field.get("hint_text") or "")
    b = field.get("bbox_pixels") or {}
    tall = (b.get("y_max", 0) - b.get("y_min", 0)) >= MULTILINE_PX
    if not current.strip() or current == hint or not tall:
        return typed
    if typed.strip() and typed.strip() in current:
        return current
    return current.rstrip("\n") + "\n" + typed


def history_line(dec: Dict[str, Any], changed: Optional[bool]) -> str:
    return v.history_entry_with_label({"action_type": dec["action"], "direction": dec.get("direction"),
                                       "app_name": dec.get("app_name"), "text": dec.get("text")},
                                      element_label(dec), changed)


def screen_changed(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> bool:
    return v.effect_changed_v4([v.canonical_element(e) for e in before], [v.canonical_element(e) for e in after])
