"""Invariants of build_exec_data (2026-11-13): the executor data for the LLM + decision-model harness.
Every action row keeps its screen and gold; an instructed episode's local goals are runs of consecutive AndroidControl
step instructions, each followed by exactly one `status` row on the screen after its last action ("local goal done");
a goal-only episode has no `status` row at all (finishing the whole task belongs to the LLM)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common  # noqa: E402
import build_exec_data as X  # noqa: E402
import build_v4_data as B  # noqa: E402

B.A = ac_common
X.A = ac_common
RULE = ac_common.CONTRACTS["v4"]["element_rule"]


def el(text, box, cls="android.widget.Button"):
    return {"text": text, "class_name": cls, "package_name": "com.example", "is_visible": True, "is_clickable": True,
            "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box))}


def screen(i):
    return [el("More", (900, 100, 1080, 250)), el("Sort", (0, 600, 1080, 760)), el("Date", (0, 1900, 1080, 2100)),
            el("screen %d" % i, (0, 300, 500, 400), cls="android.widget.TextView")]


def episode(ep=16, instrs=(("Click on the more options icon", True), ("Click on sort.", True),
                           ("Click on date", False), ("Click on date", True))):
    acts = [{"action_type": "click", "x": 990, "y": 175}, {"action_type": "click", "x": 540, "y": 680},
            {"action_type": "click", "x": 540, "y": 2000}, {"action_type": "navigate_back"}][:len(instrs)]
    n = len(instrs)
    rows = []
    for t, ((lg, done), a) in enumerate(zip(instrs, acts)):
        rows.append({"ep": ep, "step": t, "n_steps": n, "goal": "Sort by date", "W": 1080, "H": 2400,
                     "img": "imgs/%d/%d.jpg" % (ep, t), "elements": screen(t), "action": a, "target": None,
                     "target_el_idx": None, "local_goal": lg, "local_done": done, "split": "train"})
    rows.append({"ep": ep, "step": n, "n_steps": n, "goal": "Sort by date", "W": 1080, "H": 2400,
                 "img": "imgs/%d/%d.jpg" % (ep, n), "elements": screen(n),
                 "action": {"action_type": "status", "goal_status": "successful"}, "target": None,
                 "target_el_idx": None, "local_goal": None, "local_done": None, "split": "train"})
    return rows


def prepared(rows):
    """What `build` hands to `episode_rows`: unfocused elements and the v4 history."""
    rows = [dict(r, elements=B.unfocus(r["elements"])) for r in rows]
    for r, h in zip(rows, B.entries_v4(rows, RULE)):
        r["history_v3"] = h
    return rows


def test_units_group_consecutive_steps_of_one_instruction_until_it_is_done():
    assert [[r["step"] for r in u] for u in X.units(episode())] == [[0], [1], [2, 3]]


def test_join_instructions_reads_as_one_local_goal():
    assert X.join_instructions(["Open the Dailymotion app", "Click on the Search icon.", "Type pizza "]) == \
        "Open the Dailymotion app, then click on the Search icon, then type pizza"
    assert X.join_instructions(["Click on sort."]) == "Click on sort"


def test_draw_spans_partitions_the_units_into_runs_of_one_to_four():
    rng = random.Random(0)
    for n in range(1, 30):
        spans = X.draw_spans(n, rng)
        assert sum(spans) == n and all(1 <= s <= X.MAX_SPAN for s in spans)


def test_goal_mode_keeps_every_action_row_hides_the_instruction_and_has_no_status():
    rows = prepared(episode())
    out = X.episode_rows(rows, "goal", [])
    assert [r["step"] for r in out] == [0, 1, 2, 3]
    assert [r["action"] for r in out] == [r["action"] for r in rows[:-1]]
    assert all(r["show_instruction"] is False and r["aug"] == "goal" for r in out)


def test_each_span_carries_its_text_and_ends_with_one_done_row_on_the_next_screen():
    rows = prepared(episode())
    out = X.episode_rows(rows, "span", [2, 1])
    acts = [r for r in out if r["aug"] == "span"]
    dones = [r for r in out if r["aug"] == "span_done"]
    first = "Click on the more options icon, then click on sort"
    assert [(r["step"], r["local_goal"]) for r in acts] == [(0, first), (1, first), (2, "Click on date"),
                                                            (3, "Click on date")]
    assert [r["action"] for r in acts] == [r["action"] for r in rows[:-1]]  # gold unchanged
    assert [(r["step"], r["local_goal"]) for r in dones] == [(2, first), (4, "Click on date")]
    for d in dones:
        src = rows[d["step"]]
        assert d["action"]["action_type"] == "status" and d["target"] is None and d["target_el_idx"] is None
        assert d["img"] == src["img"] and d["elements"] == src["elements"] and d["history_v3"] == src["history_v3"]
    assert all(r["show_instruction"] is True for r in out)
    assert len({(r["ep"], r["step"]) for r in out}) == len(out)  # a done row never shares a key with an action row


def test_an_episode_with_an_empty_instruction_is_goal_only():
    rows = episode(instrs=(("Click on the more options icon", True), ("  ", True)))
    assert X.choose_mode(rows, random.Random(0), p_goal=0.0) == "goal"


def test_build_counts_rows_and_keeps_every_action_row_once():
    eps = episode(16) + episode(17) + episode(18)
    out, stats = X.build(eps, RULE, seed=0, p_goal=0.5)
    acts = [r for r in out if r["aug"] in ("goal", "span")]
    assert sorted((r["ep"], r["step"]) for r in acts) == sorted((r["ep"], r["step"]) for r in eps
                                                                if r["action"]["action_type"] != "status")
    assert stats["rows"] == len(out)
    assert stats["status rows"] == sum(r["action"]["action_type"] == "status" for r in out)
    assert X.build(eps, RULE, seed=0, p_goal=0.5)[0] == out  # deterministic


def test_show_instruction_overrides_the_hash_choice():
    rec = {"ep": 1, "step": 0}
    assert ac_common.instruction_included(dict(rec, show_instruction=True)) is True
    assert ac_common.instruction_included(dict(rec, show_instruction=False)) is False
    assert ac_common.instruction_included(rec) in (True, False)


def test_eval_shows_the_instruction_per_row_when_asked():
    import eval_d1 as E
    rec = {"ep": 1, "step": 0, "local_goal": "Click on sort"}
    assert E.instruction_for(dict(rec, show_instruction=False), True) == "Click on sort"  # AC-Low: always
    assert E.instruction_for(rec, False) is None  # AC-High: never
    assert E.instruction_for(dict(rec, show_instruction=True), "row") == "Click on sort"
    assert E.instruction_for(dict(rec, show_instruction=False), "row") is None


def test_open_ended_eval_episodes_get_done_rows_only_where_the_next_screen_exists():
    rows = episode()[:-1]  # an AndroidControl test file: action rows only, no final screen
    out, stats = X.build(rows, RULE, seed=0, p_goal=0.0, open_end=True)
    assert stats["episodes span"] == 1 and stats["episodes skipped (incomplete)"] == 0
    dones = [r for r in out if r["aug"] == "span_done"]
    assert all(d["step"] < 4 for d in dones) and stats["done rows without a next screen"] == 1
    assert X.build(rows, RULE, seed=0, p_goal=0.0)[1]["episodes skipped (incomplete)"] == 1  # training: refused


def test_an_episode_split_across_the_input_is_refused():
    import pytest
    rows = episode(16)[:2] + episode(17) + episode(16)[2:]
    with pytest.raises(ValueError):
        X.build(rows, RULE)
