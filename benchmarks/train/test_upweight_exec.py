"""upweight_exec (2026-11-14): extra copies of the executor rows that continue a multi-step step instruction (the row
before, in the same local goal, has local_done False). exec1 says "done" on 34% of such AndroidControl-Low steps."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upweight_exec as U  # noqa: E402


def rows():
    return [{"ep": 1, "step": 0, "aug": "span", "local_done": False}, {"ep": 1, "step": 1, "aug": "span", "local_done": True},
            {"ep": "done:1", "step": 2, "aug": "span_done", "local_done": None},
            {"ep": 1, "step": 2, "aug": "span", "local_done": False}, {"ep": 2, "step": 0, "aug": "span", "local_done": True},
            {"ep": 3, "step": 0, "aug": "goal", "local_done": False}, {"ep": 3, "step": 1, "aug": "goal", "local_done": True}]


def test_only_continuation_rows_are_copied():
    out = list(U.upweight(rows(), k=3))
    keys = [(r["ep"], r["step"]) for r in out]
    assert keys.count((1, 1)) == 3  # continues the instruction of step 0
    assert all(keys.count(x) == 1 for x in [(1, 0), ("done:1", 2), (1, 2), (2, 0), (3, 0), (3, 1)])
    assert [r for r in out if (r["ep"], r["step"]) == (1, 1)][1]["aug"] == "span_cont"


def test_k1_is_the_identity():
    assert list(U.upweight(rows(), k=1)) == rows()


def test_a_done_row_or_a_new_episode_breaks_the_chain():
    rs = [{"ep": 1, "step": 0, "aug": "span", "local_done": False}, {"ep": "done:1", "step": 1, "aug": "span_done"},
          {"ep": 1, "step": 1, "aug": "span", "local_done": False}, {"ep": 5, "step": 0, "aug": "span", "local_done": True}]
    assert len(list(U.upweight(rs, k=3))) == 4


def test_extras_are_exactly_the_added_copies():
    # the trainer reads several files (--train-set exec_v1,execcont): original file + extras = upweighted multiset
    ex = list(U.extras(rows(), k=3))
    assert [(r["ep"], r["step"], r["aug"]) for r in ex] == [(1, 1, "span_cont")] * 2
    full = sorted(json_key(r) for r in U.upweight(rows(), k=3))
    assert full == sorted(json_key(r) for r in rows() + ex)


def json_key(r):
    import json
    return json.dumps(r, sort_keys=True)
