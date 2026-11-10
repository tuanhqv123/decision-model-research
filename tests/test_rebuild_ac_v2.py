"""R2 corpus rebuild: one synthetic AndroidControl episode through the per-episode builder, the v1 oracle
(the ORIGINAL preprocess/compact code paths), and the merge that writes v2 rows.

Episode: click a clickable container whose own node has no text (AndroidWorld drops it: audit D2), type
into the focused field, terminal status.
"""
import json
import os
from types import SimpleNamespace as NS

from benchmarks.train import rebuild_ac_v2 as RB
from benchmarks.train.ac_common import pick_candidates


def node(uid, box, child_ids=(), text="", cd="", **flags):
    l, t, r, b = box
    base = dict(unique_id=uid, bounds_in_screen=NS(left=l, top=t, right=r, bottom=b), child_ids=list(child_ids),
                text=text, content_description=cd, class_name="android.widget.TextView", hint_text="",
                is_checked=False, is_checkable=False, is_clickable=False, is_editable=False, is_enabled=True,
                is_focused=False, is_focusable=False, is_long_clickable=False, is_scrollable=False,
                is_selected=False, is_visible_to_user=True, package_name="com.example",
                view_id_resource_name="")
    base.update(flags)
    return NS(**base)


def flat(forest):
    """preprocess_ac.flat_nodes, applied to an already-parsed (synthetic) forest."""
    out = []
    for w in forest.windows:
        for n in w.tree.nodes:
            b = n.bounds_in_screen
            x0, y0, x1, y1 = b.left, b.top, b.right, b.bottom
            out.append({"text": n.text or None, "content_description": n.content_description or None,
                        "hint_text": n.hint_text or None, "class_name": n.class_name or None,
                        "bbox_pixels": {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1,
                                        "center": [(x0 + x1) / 2, (y0 + y1) / 2],
                                        "width": x1 - x0, "height": y1 - y0, "area": (x1 - x0) * (y1 - y0)},
                        "is_clickable": n.is_clickable, "is_editable": n.is_editable,
                        "is_scrollable": n.is_scrollable, "is_focused": n.is_focused,
                        "is_long_clickable": n.is_long_clickable, "is_visible": n.is_visible_to_user,
                        "package_name": n.package_name or None})
    return out


def F(*nodes):
    return NS(windows=[NS(tree=NS(nodes=list(nodes)))])


ROOT = node(0, (0, 0, 1080, 2400), child_ids=[1, 2, 3])
TITLE = node(1, (10, 10, 500, 80), text="Notes")
BUTTON = node(2, (300, 2100, 700, 2300), child_ids=[4], is_clickable=True)
LABEL = node(4, (320, 2150, 680, 2250), text="New")
EDIT = node(3, (40, 400, 1040, 520), is_editable=True, is_focused=True)
EDIT_FILLED = node(3, (40, 400, 1040, 520), text="hi", is_editable=True, is_focused=True)

FORESTS = [F(ROOT, TITLE, BUTTON, LABEL), F(ROOT, TITLE, EDIT), F(ROOT, TITLE, EDIT_FILLED)]
ACTS = [{"action_type": "click", "x": 310, "y": 2110}, {"action_type": "input_text", "text": "hi"}]
INSTRS = ["Tap New", "Type hi"]
W, H = [1080] * 3, [2400] * 3


def keep(b):
    return {k: b[k] for k in ("x_min", "x_max", "y_min", "y_max", "center")}


def v1_rows():
    rows = []
    edit_box = flat(FORESTS[1])[2]["bbox_pixels"]
    specs = [
        (0, {"action_type": "click", "x": 310, "y": 2110}, keep(flat(FORESTS[0])[2]["bbox_pixels"]), []),
        (1, {"action_type": "type_text", "text": "hi", "x": 540.0, "y": 460.0}, keep(edit_box), ["click"]),
        (2, {"action_type": "status", "goal_status": "successful"}, None, ["click", "type_text hi"]),
    ]
    for step, action, target, hist in specs:
        cands = pick_candidates(flat(FORESTS[step]), 1080, 2400)
        for c in cands:
            c["bbox"] = keep(c["bbox"])
        rows.append(json.loads(json.dumps({
            "ep": 42, "step": step, "goal": "Make a note saying hi", "history": hist, "img": f"imgs/42/{step}.jpg",
            "W": 1080, "H": 2400, "action": action, "target": target, "cands": cands,
            "history_source": "rendered_gold_actions_v1"})))
    return rows


def build():
    rows = v1_rows()
    need = RB.needed_from_rows(rows)[42]
    old = {i: flat(FORESTS[i]) for i in need}
    return rows, RB.build_episode(42, "Make a note saying hi", ACTS, INSTRS, FORESTS, W, H, need, old_nodes=old)


def test_needed_rows_carry_the_v1_gold_point_and_target():
    need = RB.needed_from_rows(v1_rows())
    assert set(need) == {42} and set(need[42]) == {0, 1, 2}
    assert need[42][0]["pt"] == [310, 2110] and need[42][2]["pt"] is None


def test_the_v1_oracle_reproduces_every_v1_field_of_the_synthetic_rows():
    rows, res = build()
    for r in rows:
        d = res[r["step"]]["diff"]
        assert d["action"] == r["action"]
        assert d["target"] == r["target"]
        assert d["history"] == r["history"]
        assert d["cands_sha"] == RB.cands_sha(r["cands"])
        assert d["n_steps"] == 2


def test_dropped_container_target_and_its_reason():
    _, res = build()
    v, d = res[0]["v2"], res[0]["diff"]
    names = [e["text"] for e in v["elements"]]
    assert names == ["Notes", "New"]                      # ROOT and BUTTON are containers: dropped
    assert v["target_el_idx"] is None                      # nothing AW serves contains (310, 2110)
    assert v["target_el_centre_idx"] == 1                  # but tapping the "New" label reaches the button
    assert d["d2_reason"] == "container"


def test_type_text_target_is_the_field_and_exact():
    _, res = build()
    v = res[1]["v2"]
    assert v["elements"][v["target_el_idx"]]["is_editable"] is True
    assert v["target_el_exact"] is True and v["target_el_hits_gold"] is True


def test_effect_local_done_and_the_terminal_row():
    _, res = build()
    assert res[0]["v2"]["effect"] is True                  # new screen
    assert res[1]["v2"]["effect"] is True                  # same field, text landed
    assert [res[t]["v2"]["local_done"] for t in (0, 1)] == [True, True]
    assert [res[t]["v2"]["local_goal"] for t in (0, 1)] == INSTRS
    assert res[0]["v2"]["effect_app"] is True
    last = res[2]["v2"]
    assert (last["effect"], last["effect_app"], last["pixel_change"], last["local_goal"],
            last["local_done"]) == (None,) * 5
    assert last["n_steps"] == 2


def test_no_effect_when_the_screen_does_not_change():
    need = {0: {"pt": None, "tgt": None}}
    res = RB.build_episode(1, "g", [{"action_type": "wait"}], ["Wait"], [FORESTS[0], FORESTS[0]], W, H, need)
    assert res[0]["v2"]["effect"] is False and res[0]["v2"]["effect_app"] is False


def test_a_clock_tick_is_an_effect_but_not_an_app_effect():
    clock = node(9, (40, 0, 150, 130), text="11:38", package_name="com.android.systemui")
    tick = node(9, (40, 0, 150, 130), text="11:39", package_name="com.android.systemui")
    need = {0: {"pt": None, "tgt": None}}
    res = RB.build_episode(1, "g", [{"action_type": "wait"}], ["Wait"],
                           [F(ROOT, TITLE, clock), F(ROOT, TITLE, tick)], W, H, need)
    assert res[0]["v2"]["effect"] is True and res[0]["v2"]["effect_app"] is False


def test_merge_keeps_every_v1_field_and_order_and_reports_zero_mismatches(tmp_path):
    rows, res = build()
    src = tmp_path / "v1.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in reversed(rows)))   # order must be preserved
    work = tmp_path / "work"
    (work / "shards").mkdir(parents=True)
    shard = work / "shards" / "s0.jsonl"
    shard.write_text("".join(json.dumps({"ep": 42, "step": s, **r}) + "\n" for s, r in res.items()))
    open(str(shard) + ".done", "w").close()
    dst = tmp_path / "v2.jsonl"
    assert RB.main(["merge", "--work", str(work), "--rows", str(src), "--outs", str(dst)]) == 0
    out = [json.loads(line) for line in dst.read_text().splitlines()]
    assert [r["step"] for r in out] == [2, 1, 0]
    for o, r in zip(out, reversed(rows)):
        assert {k: o[k] for k in r} == r
        assert {"elements", "target_el_idx", "effect", "pixel_change", "local_goal", "local_done"} <= set(o)
    summary = json.load(open(work / "summary.json"))
    f = summary["files"]["v2.jsonl"]
    assert all(v == 0 for v in f["differential"].values()), f["differential"]
    assert f["validation"]["d2"]["aw_drop_reasons"] == {"container": 1}
