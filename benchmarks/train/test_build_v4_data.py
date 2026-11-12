"""Invariants of build_v4_data (2026-11-12): augmentation never changes a row's screen, keeps the gold (except where the
construction says otherwise), and every history line it adds is in the contract's own format."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common  # noqa: E402
import build_v4_data as B  # noqa: E402

B.A = ac_common
RULE = ac_common.CONTRACTS["v4"]["element_rule"]
HOME = "com.google.android.apps.nexuslauncher"


def el(text, box, cls="android.widget.TextView", pkg="com.example.notes", **flags):
    d = {"text": text, "class_name": cls, "package_name": pkg, "is_visible": True, "is_focused": True,
         "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box))}
    d.update({"is_%s" % k: v for k, v in flags.items()})
    return d


def app_screen():
    return [el("Title", (40, 300, 300, 380)), el("Groceries", (320, 300, 1000, 380)),
            el("Name", (0, 500, 1080, 600), cls="android.widget.EditText", editable=True, clickable=True),
            el("Save", (800, 140, 1040, 260), cls="android.widget.Button", clickable=True),
            el("Back", (0, 140, 200, 260), cls="android.widget.ImageButton", clickable=True),
            el("", (0, 700, 1080, 2000), cls="androidx.recyclerview.widget.RecyclerView", scrollable=True),
            el("Dark mode", (40, 2100, 600, 2200), cls="android.widget.Switch", checkable=True, checked=True,
               clickable=True)]


def home_screen():
    return [el("Notes", (100, 1000, 300, 1200), pkg=HOME, clickable=True)]


def row(ep, step, n, action, els, target=None, goal="Add a note", local_goal="x"):
    return {"ep": ep, "step": step, "n_steps": n, "goal": goal, "W": 1080, "H": 2400, "img": "i.jpg",
            "elements": els, "action": action, "target": target, "target_el_idx": None, "local_goal": local_goal,
            "history_v3": [], "source": "androidcontrol"}


def episode(ep, start_home=True):
    s0 = home_screen() if start_home else app_screen()
    return [row(ep, 0, 3, {"action_type": "open_app", "app_name": "Notes"}, s0),
            row(ep, 1, 3, {"action_type": "type_text", "text": "Milk"}, app_screen(),
                target={"x_min": 0, "y_min": 500, "x_max": 1080, "y_max": 600}),
            row(ep, 2, 3, {"action_type": "click"}, app_screen(),
                target={"x_min": 800, "y_min": 140, "x_max": 1040, "y_max": 260}),
            row(ep, 3, 3, {"action_type": "status", "goal_status": "successful"},
                app_screen() + [el("Note saved", (300, 2250, 700, 2300))])]


def test_build_unfocuses_and_keeps_screens_and_golds():
    rows, stats = B.build(episode("a") + episode("b"), RULE, seed=0)
    assert all(not e["is_focused"] for r in rows for e in r["elements"])
    base = [r for r in rows if r["aug"] == "base"]
    assert len(base) == 8 and stats["base rows"] == 8
    for r in rows:
        if r["aug"].startswith("nc_"):
            assert r["history_v3"][-1].endswith("→ no change")
            orig = next(b for b in base if b["ep"] == r["ep"] and b["step"] == r["step"])
            assert r["action"] == orig["action"] and r["elements"] == orig["elements"]


def test_field_rows_teach_type_after_a_no_change_click_on_that_field():
    B.RATES["nc_field"] = 1.0
    rows, _ = B.build(episode("a"), RULE, seed=0)
    f = [r for r in rows if r["aug"] == "nc_field"]
    assert f and f[0]["action"]["action_type"] == "type_text" and f[0]["history_v3"][-1] == 'click "Name" → no change'


def test_chain_goes_home_at_the_end_of_part_one_and_stops_only_at_the_end():
    B.CAPS["chain"] = 5
    chains = B.chain_rows({"a": B.build(episode("a"), RULE)[0][:4], "b": B.build(episode("b"), RULE)[0][:4]},
                          RULE, random.Random(0))
    assert len(chains) == 1
    ch = chains[0]
    acts = [r["action"]["action_type"] for r in ch]
    assert acts == ["open_app", "type_text", "click", "navigate_home", "open_app", "type_text", "click", "status"]
    assert ch[4]["history_v3"][-1] == "navigate_home → screen changed"
    assert all(len(ch[i + 1]["history_v3"]) == len(ch[i]["history_v3"]) + 1 for i in range(len(ch) - 1))
    assert len({r["goal"] for r in ch}) == 1 and "add a note" in ch[0]["goal"].lower()


def test_answer_rows_point_at_the_value_and_end_with_status():
    B.CAPS["answer"] = 50
    rows = B.build(episode("a"), RULE)[0][:4]
    for seed in range(40):
        ans = B.answer_rows({"a": rows}, RULE, random.Random(seed))
        if ans and ans[0][-2]["answer_value"] == "element":
            break
    a, s = ans[0][-2], ans[0][-1]
    assert a["action"] == {"action_type": "answer", "text": "Groceries"} and "'Title'" in a["goal"] or "Title" in a["goal"]
    state, kept, idx = ac_common.element_inputs_v3(a["goal"], a["history_v3"], a["elements"], 1080, 2400, RULE)
    gold, _ = ac_common.gold_answers_v2(a, kept, idx, state, apps=["notes"], contract="v4")
    assert kept[int(gold["target_el"]) - 1]["label"] == "Groceries" and gold["answer_value"] == "element"
    assert s["action"]["action_type"] == "status" and s["history_v3"][-1] == 'answer "Groceries"'


def test_avoid_rows_penalise_status_before_a_save_that_did_not_happen():
    rows = B.build(episode("a"), RULE)[0][:4]
    av = B.save_avoid_rows({"a": rows}, RULE, random.Random(0))
    assert len(av) == 1 and av[0]["avoid_action"] == "status" and av[0]["action"] is None
    assert av[0]["history_v3"][-1] == 'click "Save" → no change'


def test_futile_repeat_targets_are_dropped():
    ep = episode("a", start_home=False)
    # step 2 clicks Save, nothing changes; step 3 clicks Save again (gold), nothing changes again; then status
    rep = [dict(ep[2], step=2), dict(ep[2], step=3), dict(ep[3], step=4, elements=app_screen())]
    rows, stats = B.build(ep[:2] + rep, RULE)
    assert stats["drop"] == 1
