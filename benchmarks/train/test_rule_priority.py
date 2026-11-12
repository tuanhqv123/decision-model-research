"""Element rule `cap40_priority_noime` (2026-11-11). `cap40_reading_noime` keeps the first 40 candidates in reading
order, so on a screen with more than 40 the bottom of the screen is cut: on the AndroidWorld run, Simple Calendar's
month grid (~42 unlabelled day cells) pushed `New Event` out of the list on every add-event task (155 times).
The new rule keeps, when over the cap: labelled interactive elements first, then labelled ones, then the rest; the kept
elements are still numbered in reading order. Under the cap it is identical to `cap40_reading_noime`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common as A  # noqa: E402


def _el(text, box, cls="android.view.View", clickable=True, pkg="com.simplemobiletools.calendar.pro"):
    return {"text": text, "class_name": cls, "package_name": pkg, "is_clickable": clickable, "is_visible": True,
            "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box))}


def _calendar_month():
    cells = [_el("", (c * 154, 500 + r * 300, c * 154 + 150, 790 + r * 300)) for r in range(6) for c in range(7)]
    top = [_el("Search", (147, 149, 626, 275), cls="android.widget.EditText"), _el("October", (147, 296, 933, 438))]
    fab = _el("New Event", (891, 2148, 1038, 2295), cls="android.widget.ImageButton")
    return top + cells + [fab]


def test_the_old_rule_drops_new_event_and_the_new_rule_keeps_it():
    els = _calendar_month()
    old, _ = A.select_elements(els, 1080, 2400, "cap40_reading_noime")
    new, idx = A.select_elements(els, 1080, 2400, "cap40_priority_noime")
    assert "New Event" not in [e["label"] for e in old]
    assert "New Event" in [e["label"] for e in new] and len(new) == 40
    ys = [(e["bbox"]["y_min"], e["bbox"]["x_min"]) for e in new]
    assert ys == sorted(ys)  # still numbered in reading order


def test_under_the_cap_the_new_rule_equals_the_old_one():
    els = _calendar_month()[:30]
    assert A.select_elements(els, 1080, 2400, "cap40_priority_noime")[1] == \
        A.select_elements(els, 1080, 2400, "cap40_reading_noime")[1]


def test_keyboard_keys_are_still_excluded():
    els = _calendar_month() + [_el("q", (0, 1600, 100, 1700), pkg="com.google.android.inputmethod.latin")]
    kept, _ = A.select_elements(els, 1080, 2400, "cap40_priority_noime")
    assert "q" not in [e["label"] for e in kept]
