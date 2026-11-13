"""Contract v3: the live agent renders EXACTLY the history line training renders for the same step (2026-11-11).
Training: `add_history_v3.episode_entries` over full-flow rows (AndroidWorld element records). Serving: the agent
from the executed decision, the server's `history_label` of the chosen element, and the element lists (harness
dicts) before and after the action."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
sys.path.insert(0, os.path.join(HERE, ".."))

import ac_common as A  # noqa: E402
from add_history_v3 import episode_entries  # noqa: E402
from awenv.adapter import map_decision  # noqa: E402
from awenv.agent import history_entry_v3_for_step  # noqa: E402


def _aw(text, box, pkg="com.notes", cls="android.widget.Button", **flags):
    """An AndroidWorld element record (training rows)."""
    d = {"text": text, "content_description": None, "hint_text": None, "class_name": cls, "package_name": pkg,
         "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box)), "is_clickable": True,
         "is_visible": True}
    d.update({f"is_{k}": v for k, v in flags.items()})
    return d


def _harness(d):
    """The same element as `awenv_bridge._element_dicts` sends it (serving)."""
    b = d["bbox_pixels"]
    out = {"text": d["text"] or "", "content_description": d["content_description"] or "", "class": d["class_name"],
           "package_name": d["package_name"], "clickable": d["is_clickable"],
           "bbox": [b["x_min"], b["y_min"], b["x_max"], b["y_max"]], "bbox_pixels": dict(b),
           "hint_text": d["hint_text"] or ""}
    out.update({k: v for k, v in d.items() if k.startswith("is_")})
    return out


def test_agent_line_equals_training_line_for_click_and_type():
    s0 = [_aw("Notes", (0, 0, 200, 200), pkg="com.google.android.apps.nexuslauncher")]
    s1 = [_aw("", (0, 300, 500, 400), cls="android.widget.EditText", editable=True)]
    s2 = [_aw("hello", (0, 300, 500, 400), cls="android.widget.EditText", editable=True)]
    rows = [
        {"ep": 1, "step": 0, "n_steps": 2, "W": 1080, "H": 2400, "elements": s0, "target_el_idx": 0,
         "target": {"x_min": 0, "y_min": 0, "x_max": 200, "y_max": 200}, "action": {"action_type": "click", "x": 100, "y": 100}},
        {"ep": 1, "step": 1, "n_steps": 2, "W": 1080, "H": 2400, "elements": s1, "target_el_idx": 0,
         "target": {"x_min": 0, "y_min": 300, "x_max": 500, "y_max": 400},
         "action": {"action_type": "type_text", "text": "hello", "x": 250, "y": 350}},
        {"ep": 1, "step": 2, "n_steps": 2, "W": 1080, "H": 2400, "elements": s2, "target_el_idx": None, "target": None,
         "action": {"action_type": "status", "goal_status": "successful"}},
    ]
    train = episode_entries(rows)[2]
    served = []
    for before, after, dec in (
        (s0, s1, {"action": "click", "point": [100, 100]}),
        (s1, s2, {"action": "type_text", "text": "hello", "point": [250, 350]}),
    ):
        kept, _ = A.select_elements([_harness(e) for e in before], 1080, 2400, A.CONTRACTS["v3"]["element_rule"])
        dec["element"] = {"history_label": A.history_label(kept[0])}  # what d1_api returns for the chosen element
        served.append(history_entry_v3_for_step(dec, map_decision(dec), [_harness(e) for e in before],
                                                [_harness(e) for e in after]))
    assert served == train == ['click "Notes" → screen changed', 'type_text "hello" into [EditText] → screen changed']


def test_substituted_action_renders_what_ran():
    dec = {"action": "click"}  # no point -> invalid -> the adapter runs `wait`
    line = history_entry_v3_for_step(dec, map_decision(dec), [], [])
    assert line == "wait → no change"
