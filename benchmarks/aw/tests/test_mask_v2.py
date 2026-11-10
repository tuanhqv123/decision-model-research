"""Masking under contract v2: the element alternatives come from the v2 kept list (the server's
`element_rule`), not from the v1 candidate list. The screen is built so the two numberings differ
(v1 numbers candidates smallest-first, v2 in reading order), so using the wrong one is detectable."""
import numpy as np

from awenv.agent import AgentCore
from awenv.masking import element_points

from ac_common import CONTRACTS, TO_OPTION, element_centre, select_elements

RULE = CONTRACTS["v2"]["element_rule"]


def el(text="", cls="android.widget.Button", bbox=(0, 0, 10, 10), **flags):
    x0, y0, x1, y1 = bbox
    d = {"text": text, "content_description": "", "hint_text": "", "class": cls,
         "package_name": "com.example", "bbox": [x0, y0, x1, y1],
         "bbox_pixels": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1, "width": x1 - x0,
                         "height": y1 - y0, "center": [(x0 + x1) / 2.0, (y0 + y1) / 2.0]},
         "is_visible": True, "clickable": False, "is_clickable": False, "is_editable": False,
         "is_scrollable": False, "is_checked": False, "is_checkable": False, "is_selected": False,
         "is_focused": False, "is_long_clickable": False, "is_enabled": True, "resource_name": ""}
    d.update(flags)
    if "is_clickable" in flags:
        d["clickable"] = flags["is_clickable"]
    return d


FRAME = el("", cls="android.widget.FrameLayout", bbox=(0, 0, 100, 100))
SAVE = el("Save", bbox=(10, 10, 50, 20), is_clickable=True)      # v2 id 1 (left); v1 id 2 (larger)
CANCEL = el("Cancel", bbox=(60, 10, 80, 20), is_clickable=True)  # v2 id 2; v1 id 1 (smaller)
SCREEN = [FRAME, SAVE, CANCEL]
W, H = 100, 120   # ScreenEnv's screenshot is 120 x 100


def readouts(action_probs, target_el):
    return {"action": {"choice": TO_OPTION[max(action_probs, key=action_probs.get)],
                       "probabilities": {TO_OPTION[k]: v for k, v in action_probs.items()}},
            "scroll_dir": {"choice": "down", "probabilities": {"down": 1.0}},
            "app_name": {"text": ""}, "type_text": {"text": ""},
            "target_el": {"choice": max(target_el, key=target_el.get), "probabilities": target_el}}


def v2_click(idx=1, target=None, rule=RULE):
    kept, keep_idx = select_elements(SCREEN, W, H, RULE)
    e = kept[idx - 1]
    probs = {"click": 0.8, "navigate_back": 0.2}
    return {"action": "click", "point": element_centre(e), "track": "VA", "contract": "v2", "element_rule": rule,
            "element": {"idx": idx, "label": e["label"], "confidence": 0.7, "aw_index": keep_idx[idx - 1]},
            "element_point": element_centre(e), "probabilities": dict(probs),
            "readouts": readouts(probs, target or {"1": 0.7, "2": 0.2})}


def test_the_screen_numbers_elements_differently_in_v1_and_v2():
    from awenv.masking import server_candidates
    assert [c["label"] for c in server_candidates(SCREEN, W, H)] == ["Cancel", "Save"]
    assert [e["label"] for e in select_elements(SCREEN, W, H, RULE)[0]] == ["Save", "Cancel"]


def test_v2_element_points_follow_the_v2_numbering():
    points, note = element_points(v2_click(target={"1": 0.2, "2": 0.7}), SCREEN, W, H)
    assert note is None
    assert points == [(0, [70.0, 15.0]), (1, [30.0, 15.0])]   # Cancel first (0.7), then Save


def test_v2_with_an_unknown_rule_is_never_guessed():
    points, note = element_points(v2_click(rule="no_such_rule"), SCREEN, W, H)
    assert points == [] and note == "candidates_mismatch"


def test_v2_label_disagreement_is_refused():
    dec = v2_click()
    dec["element"]["label"] = "Something else"
    points, note = element_points(dec, SCREEN, W, H)
    assert points == [] and note == "candidates_mismatch"


class FakeModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, goal, history=None, pixels=None, **kw):
        dec = dict(self.decisions.pop(0))
        dec.setdefault("confidence", 0.7)
        dec.setdefault("latency_ms", {"model": 100.0})
        return dec


class ScreenEnv:
    def __init__(self, screen):
        self.screen, self.executed = list(screen), []

    def screenshot(self):
        return np.zeros((H, W, 3), dtype=np.uint8)

    def elements(self):
        return [dict(e) for e in self.screen]

    def execute(self, action):
        self.executed.append(action)
        return {"ok": True, "error": None}


def test_a_masked_v2_click_falls_back_to_the_next_v2_element():
    env = ScreenEnv(SCREEN)   # nothing ever changes
    agent = AgentCore(FakeModel([v2_click(), v2_click()]), env, track="VA", mask=True)
    agent.step("g", i=0)                                         # Save: no effect -> masked
    rec = agent.step("g", i=1)
    assert env.executed[0] == {"action_type": "click", "x": 30, "y": 15}   # Save
    assert env.executed[1] == {"action_type": "click", "x": 70, "y": 15}   # Cancel, from target_el rank 1
    assert rec["mask_choice"]["via"] == "target_el"
