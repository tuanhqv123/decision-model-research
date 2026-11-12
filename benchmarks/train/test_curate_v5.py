"""curate_v5 (2026-11-12, operator: small, deduplicated, balanced beats large and redundant)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common  # noqa: E402
import curate_v5 as C  # noqa: E402
import test_build_recovery_data as T  # noqa: E402

C.A = ac_common


def _rows():
    e = T.mail_eps()["E"]
    base = [dict(e[0], ep="e%d" % k, goal="Write to person %d" % k, aug="base") for k in range(6)]   # one screen+action x6
    rare = [dict(e[1], ep="r", aug="base", action={"action_type": "long_press"})]
    keep = [dict(e[0], ep="c", aug="chain"), dict(e[0], ep="b", aug="recover_back")]
    return base + rare + keep


def test_duplicates_are_capped_and_targeted_rows_are_kept():
    out = C.curate(_rows(), random.Random(0), per_key=2, click_share=1.0)
    base = [r for r in out if r["aug"] == "base" and r["action"].get("action_type") == "click"]
    assert len(base) == 2 and len({r["goal"] for r in base}) == 2                 # 6 identical (screen, click) -> 2
    assert {r["aug"] for r in out} >= {"chain", "recover_back"}                    # targeted rows never dropped
    assert any(r["action"].get("action_type") == "long_press" for r in out)        # rare actions kept


def test_click_share_cap():
    rows = _rows()
    out = C.curate(rows, random.Random(0), per_key=10, click_share=0.5)
    base_click = [r for r in out if r["aug"] == "base" and r["action"].get("action_type") == "click"]
    base = [r for r in out if r["aug"] == "base"]
    assert len(base_click) <= 0.5 * len(base) + 1


def test_targets_with_list_values_are_keyed():
    """Real rows carry target["center"] = [x, y] (a list): the key must still be hashable."""
    rows = [dict(r, target=dict(r["target"] or {}, center=[1.0, 2.0])) for r in _rows()]
    assert C.curate(rows, random.Random(0), per_key=2, click_share=1.0)


def test_stop_pairs_survive_dedupe_and_click_cap():
    """2026-11-12: curation kept both the last action and the `status` row in only 1,918 of 4,191 episodes; that pair
    is the only evidence separating 'one step from done' from 'done' (the top AndroidWorld failure). It is never cut."""
    e = T.mail_eps()["E"]
    rows = []
    for k in range(6):                                   # 6 episodes whose last action is the same (screen, click)
        last = dict(e[0], ep="s%d" % k, step=1, n_steps=2, goal="Write to person %d" % k, aug="base")
        done = dict(e[1], ep="s%d" % k, step=2, n_steps=2, goal="Write to person %d" % k, aug="base",
                    action={"action_type": "status", "goal_status": "successful"}, target=None)
        rows += [last, done]
    out = C.curate(rows, random.Random(0), per_key=1, click_share=0.0)
    for k in range(6):
        steps = sorted(r["step"] for r in out if r["ep"] == "s%d" % k)
        assert steps == [1, 2]
