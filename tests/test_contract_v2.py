"""Contract v2 (R2): the decision model chooses among AndroidWorld-extracted elements.

One canonical element, one gold-independent filter, one renderer — shared by training (`make_sample`),
evaluation (`eval_d1.run_eval`) and serving (`d1_api`). Every test here pins a piece of the string the model
sees, because a train/serve skew in this string has happened three times already (test_state_contract.py).
"""
import inspect
import json
import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "train"))

import ac_common as AC  # noqa: E402
import aw_extract as AX  # noqa: E402
from benchmarks.aw.awenv.awenv_bridge import _element_dicts  # noqa: E402

W, H = 1080, 2400


# ----------------------------------------------------------------------------- one AndroidWorld node, two shapes
def node(uid, box, child_ids=(), text="", cd="", hint="", cls="android.widget.TextView", pkg="com.example",
         **flags):
    """A node with the attribute surface of android_env's AndroidAccessibilityNodeInfo (proto defaults)."""
    l, t, r, b = box
    base = dict(unique_id=uid, bounds_in_screen=NS(left=l, top=t, right=r, bottom=b), child_ids=list(child_ids),
                text=text, content_description=cd, class_name=cls, hint_text=hint,
                is_checked=False, is_checkable=False, is_clickable=False, is_editable=False, is_enabled=True,
                is_focused=False, is_focusable=False, is_long_clickable=False, is_scrollable=False,
                is_selected=False, is_visible_to_user=True, package_name=pkg, view_id_resource_name="")
    base.update(flags)
    return NS(**base)


def forest(*nodes):
    return NS(windows=[NS(tree=NS(nodes=list(nodes)))])


def both_shapes(f):
    """(corpus element dicts, bridge element dicts after a JSON round trip) for one AndroidWorld forest.

    Corpus: `aw_extract.extract_elements` (how `compact_*_v2.jsonl` was written). Serve: the vendored
    `forest_to_ui_elements` -> the harness's `_element_dicts` -> JSON (the HTTP body)."""
    corpus = AX.extract_elements(f)
    ui = AX.forest_to_ui_elements(f, exclude_invisible_elements=True)
    served = json.loads(json.dumps(_element_dicts(NS(ui_elements=ui))))
    return corpus, served


SWITCH = node(1, (40, 600, 1040, 700), text="  Wi-Fi\n  calling  ", cls="android.widget.Switch",
              is_clickable=True, is_checkable=True, is_checked=True, is_focused=True)
FIELD = node(2, (40, 800, 1040, 900), hint="Search settings", cls="android.widget.EditText",
             is_editable=True, is_clickable=True, is_long_clickable=True, is_selected=True)
ICON = node(3, (10, 150, 110, 250), cd="Navigate up", cls="android.widget.ImageButton", is_clickable=True)
LIST = node(4, (0, 300, 1080, 2200), child_ids=[9], cls="androidx.recyclerview.widget.RecyclerView",
            is_scrollable=True)
CLOCK = node(5, (40, 0, 140, 60), text="9:06", pkg="com.android.systemui")
INERT = node(6, (0, 1000, 1080, 1100), cls="android.view.View")  # leaf, no label, no flag
LONG = node(7, (0, 1200, 1080, 1300), text="x" * 55)
SCREEN = forest(SWITCH, FIELD, ICON, LIST, CLOCK, INERT, LONG)


def test_one_node_canonicalizes_identically_from_the_corpus_and_the_bridge_shape():
    corpus, served = both_shapes(SCREEN)
    assert len(corpus) == len(served) == 7
    for c, s in zip(corpus, served):
        assert AC.canonical_element(c) == AC.canonical_element(s)


def test_canonical_element_fields():
    corpus, _ = both_shapes(SCREEN)
    sw = AC.canonical_element(corpus[0])
    assert sw == {"label": "Wi-Fi calling", "class_short": "Switch",
                  "bbox": {"x_min": 40, "y_min": 600, "x_max": 1040, "y_max": 700},
                  "flags": {"clickable": True, "editable": False, "scrollable": False, "checkable": True,
                            "checked": True, "selected": False, "long_clickable": False, "focused": True},
                  "package_name": "com.example"}
    assert AC.canonical_element(corpus[1])["label"] == "Search settings"   # hint_text when no text / desc
    assert AC.canonical_element(corpus[2])["label"] == "Navigate up"       # content_description
    assert AC.canonical_element(corpus[3])["label"] == ""                  # nothing to show
    assert AC.canonical_element(corpus[6])["label"] == "x" * 40            # <= 40 chars
    assert AC.canonical_element(corpus[4])["package_name"] == "com.android.systemui"


def test_label_takes_the_first_non_empty_of_text_desc_hint():
    d = {"text": "   ", "content_description": "Back", "hint_text": "ignored", "class_name": "a.B",
         "bbox_pixels": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}}
    assert AC.canonical_element(d)["label"] == "Back"


def test_the_api_request_shapes_canonicalize_like_the_bridge():
    """`d1_api` used to accept `bbox` as a flat list and `clickable` as one boolean; both still canonicalize."""
    _, served = both_shapes(forest(ICON))
    flat = {"text": "", "content_description": "Navigate up", "class": "android.widget.ImageButton",
            "package_name": "com.example", "clickable": True, "bbox": [10, 150, 110, 250]}
    assert AC.canonical_element(flat) == AC.canonical_element(served[0])


def test_float_bboxes_become_ints():
    d = {"text": "a", "bbox_pixels": {"x_min": 1.0, "y_min": 2.0, "x_max": 3.0, "y_max": 4.0}}
    assert AC.canonical_element(d)["bbox"] == {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}


# ----------------------------------------------------------------------------- the filter rules
def canon(*dicts):
    return [AC.canonical_element(d) for d in dicts]


def el(label="", box=(0, 0, 10, 10), pkg="com.example", **flags):
    x0, y0, x1, y1 = box
    d = {"text": label, "class_name": "android.widget.TextView", "package_name": pkg,
         "bbox_pixels": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}}
    d.update({f"is_{k}": v for k, v in flags.items()})
    return d


def test_rules_are_pure_functions_of_the_list_and_the_screen_never_of_the_gold():
    assert set(AC.ELEMENT_RULES) == {"all_app", "interactive_or_labelled", "cap40_reading",
                                     "cap40_interactive_first"}
    for fn in AC.ELEMENT_RULES.values():
        assert list(inspect.signature(fn).parameters) == ["elements", "W", "H"]


def test_all_app_drops_system_ui_zero_area_and_fully_off_screen():
    els = canon(el("a", (0, 0, 10, 10)),
                el("clock", (0, 0, 10, 10), pkg="com.android.systemui"),
                el("flat", (5, 5, 5, 40)),                     # zero width
                el("below", (0, 2400, 100, 2500)),             # starts at the bottom edge
                el("left", (-50, 10, 0, 20)),                  # ends at the left edge
                el("partly", (-50, 10, 50, 20)),               # partly visible: kept
                el("", (0, 100, 10, 110)))                      # unlabelled, inert: kept by all_app
    assert AC.ELEMENT_RULES["all_app"](els, W, H) == [0, 5, 6]


def test_interactive_or_labelled_drops_unlabelled_inert_elements():
    els = canon(el("", (0, 0, 10, 10)),                        # dropped
                el("", (0, 10, 10, 20), clickable=True),
                el("", (0, 20, 10, 30), editable=True),
                el("", (0, 30, 10, 40), scrollable=True),
                el("", (0, 40, 10, 50), checkable=True),
                el("", (0, 50, 10, 60), long_clickable=True),
                el("", (0, 60, 10, 70), checked=True, selected=True, focused=True),  # state only: dropped
                el("Title", (0, 70, 10, 80)))
    assert AC.ELEMENT_RULES["interactive_or_labelled"](els, W, H) == [1, 2, 3, 4, 5, 7]


def test_cap40_reading_sorts_top_to_bottom_then_left_to_right_and_keeps_40():
    els = canon(*[el(f"e{i}", (100 * (i % 3), 50 * (60 - i), 100 * (i % 3) + 50, 50 * (60 - i) + 20))
                  for i in range(60)])
    kept = AC.ELEMENT_RULES["cap40_reading"](els, W, H)
    assert len(kept) == 40
    keys = [(els[i]["bbox"]["y_min"], els[i]["bbox"]["x_min"]) for i in kept]
    assert keys == sorted(keys)
    assert kept[0] == 59  # highest on screen first


def test_reading_order_ties_keep_the_androidworld_order():
    els = canon(el("b", (0, 0, 10, 10)), el("a", (0, 0, 10, 10)))
    assert AC.ELEMENT_RULES["cap40_reading"](els, W, H) == [0, 1]


def test_cap40_interactive_first_puts_interactive_elements_first_then_labelled():
    els = canon(el("label top", (0, 0, 10, 10)),
                el("", (0, 500, 10, 510), clickable=True),
                el("button", (0, 100, 10, 110), clickable=True),
                el("label low", (0, 900, 10, 910)))
    assert AC.ELEMENT_RULES["cap40_interactive_first"](els, W, H) == [2, 1, 0, 3]
    many = canon(*[el(f"t{i}", (0, i, 10, i + 5)) for i in range(45)],
                 el("", (0, 2000, 10, 2010), editable=True))
    kept = AC.ELEMENT_RULES["cap40_interactive_first"](many, W, H)
    assert len(kept) == 40 and kept[0] == 45  # the one interactive element survives the cap


# ----------------------------------------------------------------------------- rendering + state
def test_render_elements_format():
    corpus, _ = both_shapes(SCREEN)
    kept = [AC.canonical_element(corpus[i]) for i in (0, 1, 3)]
    assert AC.render_elements(kept, W, H) == [
        "[1] Wi-Fi calling | Switch | click,checkable,checked,focused | x:4-96% y:25-29%",
        "[2] Search settings | EditText | click,edit,selected,long_click | x:4-96% y:33-38%",
        "[3] (no label) | RecyclerView | scroll | x:0-100% y:12-92%",
    ]


def test_render_marks_no_flags_and_clamps_positions_to_the_screen():
    kept = canon(el("Title", (-50, -10, 2000, 30)))
    assert AC.render_elements(kept, W, H) == ["[1] Title | TextView | - | x:0-100% y:0-1%"]


def test_v2_state_layout():
    corpus, _ = both_shapes(SCREEN)
    state, kept, idx = AC.element_inputs("Turn on Wi-Fi calling", ["open_app Settings", "click"], corpus, W, H,
                                         "interactive_or_labelled")
    assert idx == [0, 1, 2, 3, 6]
    lines = state.split("\n")
    assert lines[:4] == ["Goal: Turn on Wi-Fi calling", "Previous actions:", "Step 1: open_app Settings",
                         "Step 2: click"]
    assert lines[4] == "Elements on screen (id: label | type | state | position):"
    assert lines[5:] == AC.render_elements(kept, W, H)
    assert "Apps used for similar tasks" not in state and "Application for this task" not in state


def test_v2_state_with_no_history_uses_the_trained_sentinel():
    state, _, _ = AC.element_inputs("g", [], [el("a")], W, H, "all_app")
    assert state.split("\n")[1] == "Previous actions: none"


def test_v1_state_is_unchanged_by_the_refactor():
    """build_state (V and VA) must render exactly what it rendered before v2 existed."""
    cands = [{"idx": 1, "label": "OK", "pos": "x:1-2% y:3-4%"}, {"idx": 2, "label": "", "pos": "x:5-6% y:7-8%"}]
    assert AC.build_state("g", "VA", cands, None, ["click"]) == (
        "Goal: g\nPrevious actions:\nStep 1: click\nElements on screen (id: label, position):\n"
        "[1] OK  (x:1-2% y:3-4%)\n[2] (no label)  (x:5-6% y:7-8%)")
    assert AC.build_state("g", "V", history=None, apps=["Clock"]) == (
        "Goal: g\nPrevious actions: none\nApps used for similar tasks: Clock")


# ----------------------------------------------------------------------------- contract + questions
V2_QUESTIONS = ["action", "target_el", "scroll_dir", "app_name", "type_text"]


def test_v2_contract_shape():
    spec = AC.CONTRACTS["v2"]
    assert spec["actions"] == AC.CONTRACTS["v1"]["actions"] and len(spec["actions"]) == 9
    assert spec["is_done"] is False
    assert spec["element_rule"] in AC.ELEMENT_RULES


def test_v2_questions_have_target_el_and_no_target_box():
    kept = canon(el("Save", clickable=True), el(""))
    q = AC.build_questions("VA", kept, contract="v2")
    assert list(q) == V2_QUESTIONS
    assert q["target_el"]["type"] == "choice"
    assert q["target_el"]["criteria"] == {"1": "Save", "2": "(no label) at (5,5)"}  # unlabelled -> position (option_texts)
    assert list(q["action"]["criteria"]) == list(AC.CONTRACTS["v1"]["actions"])


def test_v2_questions_without_any_kept_element_have_no_target_el():
    assert list(AC.build_questions("VA", [], contract="v2")) == ["action", "scroll_dir", "app_name", "type_text"]


def test_v2_refuses_the_v_track():
    with pytest.raises(ValueError):
        AC.build_questions("V", None, contract="v2")


def test_v1_questions_are_unchanged():
    assert list(AC.build_questions("V", contract="v1")) == ["action", "target_box", "scroll_dir", "app_name",
                                                           "type_text"]
    cands = [{"idx": 1, "label": "OK", "pos": "x:1-2% y:3-4%"}]
    assert list(AC.build_questions("VA", cands, contract="v1")) == ["action", "target_box", "target_el",
                                                                   "scroll_dir", "app_name", "type_text"]


def test_v2_contract_records_the_element_rule(tmp_path):
    AC.write_contract(str(tmp_path), "v2", tracks=["VA"])
    raw = json.load(open(tmp_path / "contract.json"))
    assert raw["element_rule"] == AC.CONTRACTS["v2"]["element_rule"]
    doc = AC.read_contract(str(tmp_path))
    assert doc["name"] == "v2" and doc["tracks"] == ["VA"]
    assert doc["element_rule"] == AC.CONTRACTS["v2"]["element_rule"]
    assert list(AC.questions_for(doc, "VA", canon(el("a")))) == V2_QUESTIONS


def test_v2_contract_only_for_the_va_track(tmp_path):
    with pytest.raises(ValueError):
        AC.write_contract(str(tmp_path), "v2", tracks=["V"])
    with pytest.raises(ValueError):
        AC.write_contract(str(tmp_path), "v2", tracks=["V", "VA"])


def test_a_v2_contract_without_a_known_rule_is_refused(tmp_path):
    AC.write_contract(str(tmp_path), "v2", tracks=["VA"])
    doc = json.load(open(tmp_path / "contract.json"))
    doc.pop("element_rule")
    json.dump(doc, open(tmp_path / "contract.json", "w"))
    with pytest.raises(KeyError):
        AC.read_contract(str(tmp_path))
    doc["element_rule"] = "no_such_rule"
    json.dump(doc, open(tmp_path / "contract.json", "w"))
    with pytest.raises(KeyError):
        AC.read_contract(str(tmp_path))


def test_a_stamped_rule_wins_over_the_current_default(tmp_path):
    """A checkpoint keeps the rule it was trained with even if CONTRACTS['v2'] later changes."""
    AC.write_contract(str(tmp_path), "v2", tracks=["VA"])
    doc = json.load(open(tmp_path / "contract.json"))
    other = next(r for r in AC.ELEMENT_RULES if r != AC.CONTRACTS["v2"]["element_rule"])
    doc["element_rule"] = other
    json.dump(doc, open(tmp_path / "contract.json", "w"))
    assert AC.read_contract(str(tmp_path))["element_rule"] == other


# ----------------------------------------------------------------------------- gold
def rec_with(elements, action, target=None, idx=None, centre=None, history=()):
    return {"ep": 1, "step": 0, "goal": "Turn on Wi-Fi calling", "history": list(history), "W": W, "H": H,
            "img": "x.jpg", "action": action, "target": target, "elements": elements,
            "target_el_idx": idx, "target_el_centre_idx": centre}


def test_gold_target_el_prefers_the_smallest_containing_element():
    els = [el("", (0, 0, 10, 10)), el("Save", (0, 100, 500, 200), clickable=True)]
    rec = rec_with(els, {"action_type": "click", "x": 50, "y": 150},
                   target={"x_min": 0, "y_min": 100, "x_max": 500, "y_max": 200}, idx=1, centre=1)
    state, kept, keep_idx = AC.element_inputs(rec["goal"], rec["history"], els, W, H, "interactive_or_labelled")
    gold, case = AC.gold_answers_v2(rec, kept, keep_idx, state)
    assert case == "idx" and gold["target_el"] == "1"   # kept list = [Save]: id 1
    assert gold["action"] == "click" and "target_box" not in gold and "is_done" not in gold


def test_gold_target_el_falls_back_to_the_centre_element_when_the_containing_one_is_filtered():
    els = [el("", (0, 100, 500, 200)),                              # containing, inert: filtered
           el("Save", (0, 120, 400, 180), clickable=True)]
    rec = rec_with(els, {"action_type": "click", "x": 450, "y": 150},
                   target={"x_min": 0, "y_min": 100, "x_max": 500, "y_max": 200}, idx=0, centre=1)
    state, kept, keep_idx = AC.element_inputs(rec["goal"], [], els, W, H, "interactive_or_labelled")
    gold, case = AC.gold_answers_v2(rec, kept, keep_idx, state)
    assert case == "creditable" and gold["target_el"] == "1"


def test_gold_without_a_surviving_element_still_supervises_the_action():
    els = [el("", (0, 100, 500, 200))]
    rec = rec_with(els, {"action_type": "click", "x": 450, "y": 150},
                   target={"x_min": 0, "y_min": 100, "x_max": 500, "y_max": 200}, idx=0, centre=0)
    state, kept, keep_idx = AC.element_inputs(rec["goal"], [], els, W, H, "interactive_or_labelled")
    gold, case = AC.gold_answers_v2(rec, kept, keep_idx, state)
    assert case == "none" and "target_el" not in gold and gold["action"] == "click"


def test_gold_for_non_spatial_rows_has_no_target_el():
    els = [el("Save", clickable=True)]
    rec = rec_with(els, {"action_type": "scroll", "direction": "down"})
    state, kept, keep_idx = AC.element_inputs(rec["goal"], [], els, W, H, "all_app")
    gold, case = AC.gold_answers_v2(rec, kept, keep_idx, state)
    assert case is None and gold == {"action": "scroll", "scroll_dir": "down", "app_name": "nothing",
                                     "type_text": "nothing"}


def test_gold_type_text_reads_the_text_from_the_goal():
    els = [el("", (0, 100, 500, 200), editable=True)]
    rec = rec_with(els, {"action_type": "type_text", "text": "Wi-Fi", "x": 250, "y": 150},
                   target={"x_min": 0, "y_min": 100, "x_max": 500, "y_max": 200}, idx=0, centre=0)
    state, kept, keep_idx = AC.element_inputs(rec["goal"], [], els, W, H, "all_app")
    gold, case = AC.gold_answers_v2(rec, kept, keep_idx, state)
    assert gold["target_el"] == "1"
    a, b = gold["type_text"]
    assert state[a:b].lower() == "wi-fi"


# ----------------------------------------------------------------------------- answer -> point
def test_chosen_element_maps_the_answer_to_the_kept_element_and_its_centre():
    kept = canon(el("a", (0, 0, 10, 10)), el("b", (100, 200, 301, 400)))
    i, e = AC.chosen_element({"target_el": {"choice": "2", "confidence": 0.9}}, kept)
    assert i == 2 and e is kept[1]
    assert AC.element_centre(e) == [200.5, 300.0]
    assert AC.chosen_element({"target_el": {"choice": "9"}}, kept) == (None, None)
    assert AC.chosen_element({}, kept) == (None, None)


# ----------------------------------------------------------------------------- the rule choice is the measured one
def _measure_rules():
    import importlib.util
    path = os.path.join(ROOT, "benchmarks", "results", "2026-11-10-contract-v2", "measure_rules.py")
    spec = importlib.util.spec_from_file_location("measure_rules", path)
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    return mr


def test_the_contract_rule_is_the_one_the_pre_declared_selection_picked_on_train():
    """CONTRACTS['v2'] must carry the rule `measure_rules.choose_rule` picks from the stored TRAIN table
    (p90 <= 40, then highest gold coverage, ties -> shorter median), so the choice cannot drift silently."""
    mr = _measure_rules()
    summary = json.load(open(os.path.join(ROOT, "benchmarks", "results", "2026-11-10-contract-v2", "summary.json")))
    assert summary["train"]["rows"] == 14965 and summary["validation"]["rows"] == 827
    assert summary["train"]["sha256"] == "3642d03b217dc0fe97549945d50e64d3482e4bf73c6a4a1fb5d63d40cf9a914b"
    assert summary["validation"]["sha256"] == "6f15fc7e85a0c0365aa0e8a43e6869890ac098b939f209187bfba5b6cf6cfa55"
    assert mr.choose_rule(summary["train"]["rules"]) == summary["chosen_rule"] == AC.CONTRACTS["v2"]["element_rule"]


def test_choose_rule_applies_the_length_gate_then_coverage_then_median():
    mr = _measure_rules()

    def row(p90, p50, n):
        return {"len": {"p90": p90, "p50": p50}, "coverage": {"n": n}}
    table = {"all_app": row(118, 43, 99), "interactive_or_labelled": row(67, 29, 95),
             "cap40_reading": row(40, 29, 86), "cap40_interactive_first": row(40, 28, 86)}
    assert mr.choose_rule(table) == "cap40_interactive_first"   # tie on coverage -> shorter median
    table["cap40_reading"] = row(40, 29, 87)
    assert mr.choose_rule(table) == "cap40_reading"
    table["all_app"] = row(41, 10, 100)
    assert mr.choose_rule(table) == "cap40_reading"             # p90 41 > 40: excluded


# ---------------------------------------------------------------- gold must be creditable (R4 class, 2026-11-10 fix)
def _kept(*specs):
    """specs: (label, box) -> canonical kept list."""
    return canon(*[el(label, box, clickable=True) for label, box in specs])


def test_gold_v2_never_supervises_an_element_the_scorer_cannot_credit():
    # target = (100,100)-(200,200). Element 1 is the "smallest containing the click" but its centre is outside
    # the target (a wide row whose centre lies right of the target); element 2's centre is inside.
    kept = _kept(("row", (90, 90, 600, 210)), ("icon", (120, 120, 180, 180)))
    rec = {"action": {"action_type": "click", "x": 150, "y": 150},
           "target": {"x_min": 100, "x_max": 200, "y_min": 100, "y_max": 200},
           "target_el_idx": 0, "target_el_centre_idx": None}
    gold, case = AC.gold_target_el_v2(rec, [0, 1], kept)
    assert gold == "2" and case == "creditable"


def test_gold_v2_finds_a_creditable_element_when_the_named_ones_were_cut():
    kept = _kept(("other", (500, 500, 600, 600)), ("inside", (130, 130, 170, 170)))
    rec = {"action": {"action_type": "click", "x": 150, "y": 150},
           "target": {"x_min": 100, "x_max": 200, "y_min": 100, "y_max": 200},
           "target_el_idx": 7, "target_el_centre_idx": 8}      # neither survived the filter
    gold, case = AC.gold_target_el_v2(rec, [3, 4], kept)
    assert gold == "2" and case == "creditable"


def test_gold_v2_prefers_the_named_element_when_it_is_creditable():
    kept = _kept(("a", (110, 110, 190, 190)), ("b", (120, 120, 180, 180)))
    rec = {"action": {"action_type": "click", "x": 150, "y": 150},
           "target": {"x_min": 100, "x_max": 200, "y_min": 100, "y_max": 200},
           "target_el_idx": 0, "target_el_centre_idx": 1}
    assert AC.gold_target_el_v2(rec, [0, 1], kept) == ("1", "idx")


def test_gold_v2_none_when_nothing_kept_is_creditable():
    kept = _kept(("far", (500, 500, 600, 600)))
    rec = {"action": {"action_type": "click", "x": 150, "y": 150},
           "target": {"x_min": 100, "x_max": 200, "y_min": 100, "y_max": 200},
           "target_el_idx": None, "target_el_centre_idx": None}
    assert AC.gold_target_el_v2(rec, [0], kept) == (None, "none")


def test_v2_option_texts_are_distinct_for_unlabelled_and_duplicate_labels():
    kept = _kept(("", (0, 0, 100, 100)), ("", (0, 200, 100, 300)), ("OK", (0, 400, 100, 500)),
                 ("OK", (200, 400, 300, 500)), ("Save", (0, 600, 100, 700)))
    crit = AC.build_questions("VA", kept, contract="v2")["target_el"]["criteria"]
    texts = list(crit.values())
    assert len(set(texts)) == len(texts), texts
    assert crit["5"] == "Save"                       # a unique label stays as it is
    assert "at (" in crit["1"] and "at (" in crit["3"]
