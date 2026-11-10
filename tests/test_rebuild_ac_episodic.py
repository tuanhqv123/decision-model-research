# tests/test_rebuild_ac_episodic.py
"""Task 1 — episode-preserving corpus rebuild. Tests first, per the plan."""
import json

import pytest

from benchmarks.train.rebuild_ac_episodic import (
    build_compact,
    is_dropped,
    load_episodes,
    sample_episodes,
)


def _rec(**kw):
    base = dict(
        ep=1, step=0, n_steps=3, split="train", goal="g",
        step_instruction="s", img="imgs/1/0.jpg", W=1080, H=2400,
        action={"action_type": "wait"}, target=None, nodes=[],
    )
    base.update(kw)
    return base


def test_drop_inverted_bbox():
    b = dict(x_min=10, x_max=5, y_min=0, y_max=10, area=0, center=[7, 5])
    assert is_dropped(_rec(target=b)) == "inverted_or_zero_area"


def test_drop_zero_area_bbox():
    b = dict(x_min=0, x_max=10, y_min=0, y_max=10, area=0, center=[5, 5])
    assert is_dropped(_rec(target=b)) == "inverted_or_zero_area"


def test_drop_out_of_bounds():
    b = dict(x_min=0, x_max=2000, y_min=0, y_max=100, area=200000, center=[1000, 50])
    assert is_dropped(_rec(target=b)) == "out_of_bounds"


def test_drop_negative_coord_is_out_of_bounds():
    b = dict(x_min=-5, x_max=10, y_min=0, y_max=10, area=150, center=[2, 5])
    assert is_dropped(_rec(target=b)) == "out_of_bounds"


def test_drop_click_without_target():
    assert is_dropped(_rec(action={"action_type": "click"})) == "click_without_target"


def test_drop_type_text_without_target():
    assert is_dropped(_rec(action={"action_type": "type_text"})) == "type_text_without_target"


def test_keep_valid_click():
    b = dict(x_min=0, x_max=10, y_min=0, y_max=10, area=100, center=[5, 5])
    assert is_dropped(_rec(action={"action_type": "click"}, target=b)) is None


def test_keep_navigate_home():
    """Review Focus 2: navigate_home must survive a rebuild."""
    assert is_dropped(_rec(action={"action_type": "navigate_home"})) is None


def test_keep_status():
    assert is_dropped(_rec(action={"action_type": "status", "goal_status": "successful"})) is None


def test_sampler_never_truncates_an_episode():
    eps = {("train", i): [_rec(ep=i, step=s, n_steps=4) for s in range(5)]
           for i in range(3)}
    picked = sample_episodes(eps, rows_budget=7, seed=0)
    assert sum(len(eps[k]) for k in picked) <= 7
    assert all(len(eps[k]) == 5 for k in picked)


def test_sampler_is_deterministic():
    eps = {("train", i): [_rec(ep=i)] * 3 for i in range(20)}
    a = sample_episodes(eps, rows_budget=30, seed=7)
    b = sample_episodes(eps, rows_budget=30, seed=7)
    assert a == b


def test_sampler_uses_full_budget_when_it_can():
    eps = {("train", i): [_rec(ep=i)] * 5 for i in range(10)}
    picked = sample_episodes(eps, rows_budget=50, seed=0)
    assert sum(len(eps[k]) for k in picked) == 50


def test_sampler_respects_budget_exactly():
    eps = {("train", i): [_rec(ep=i)] * 6 for i in range(20)}
    picked = sample_episodes(eps, rows_budget=40, seed=0)
    total = sum(len(eps[k]) for k in picked)
    assert total <= 40
    assert total >= 36, "should keep filling after a skip, not stop at the first overflow"


def test_build_compact_emits_history_as_text():
    rec = _rec(step=2, n_steps=4)
    out = build_compact(rec, prev=[_rec(step=0), _rec(step=1)])
    assert isinstance(out["history"], list)
    assert out["history"] and all(isinstance(h, str) for h in out["history"])
    # BARE entries: `build_state` owns the "Step N: " prefix. Writing it here too produced
    # `Step 1: Step 1: open_app IKEA` for all 159,954 entries of the episodic corpus, while the
    # cur15k corpus E2 was trained on had 0 prefixed entries (measured 2026-11-10).
    assert not out["history"][0].startswith("Step "), out["history"]


def test_build_compact_history_carries_action_and_label():
    prev = [
        _rec(step=0, action={"action_type": "open_app", "app_name": "Markor"}),
        _rec(step=1, action={"action_type": "click"}),
    ]
    out = build_compact(_rec(step=2), prev=prev)
    assert "open_app" in out["history"][0]
    assert "Markor" in out["history"][0]
    assert out["history"][1] == "click", out["history"]


def test_build_compact_preserves_navigate_home():
    out = build_compact(_rec(action={"action_type": "navigate_home"}))
    assert out["action"]["action_type"] == "navigate_home"


def test_build_compact_keeps_required_keys():
    out = build_compact(_rec(step=0))
    for k in ("ep", "step", "n_steps", "split", "goal", "history", "img", "W", "H",
              "action", "target"):
        assert k in out, f"missing {k}"


def test_build_compact_without_prev_has_empty_history():
    out = build_compact(_rec(step=0))
    assert out["history"] == []


def test_load_episodes_groups_and_sorts(tmp_path):
    rows = [
        _rec(ep=1, step=1), _rec(ep=1, step=0), _rec(ep=2, step=0, split="validation"),
    ]
    p = tmp_path / "records"
    p.mkdir()
    with (p / "shard.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    eps = load_episodes(str(tmp_path))
    assert set(eps) == {("train", 1), ("validation", 2)}
    assert [r["step"] for r in eps[("train", 1)]] == [0, 1]


def test_compact_always_emits_cands_key():
    """A missing key crashed the trainer at step 40 (KeyError: 'cands'). Never omit it."""
    out = build_compact(_rec(step=0, nodes=[]))
    assert "cands" in out, "cands must always be present, even when empty"
    assert out["cands"] == []


def test_compact_with_tree_produces_candidates():
    node = {"text": "Save", "is_visible": True,
            "bbox_pixels": {"x_min": 10, "x_max": 50, "y_min": 10, "y_max": 50,
                            "width": 40, "height": 40, "area": 1600, "center": [30, 30]}}
    out = build_compact(_rec(step=0, nodes=[node], W=1080, H=2400))
    assert out["cands"], "a labelled visible node must become a candidate"
    assert out["cands"][0]["label"] == "Save"
