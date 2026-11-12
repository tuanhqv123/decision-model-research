"""build_v5_data (2026-11-12): the v4 build with more chained (longer) episodes, plus recovery rows each paired with
its twin (the same screen under the goal it IS on the path of, with that episode's own gold)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common  # noqa: E402
import build_recovery_data as R  # noqa: E402
import build_v4_data as B  # noqa: E402
import build_v5_data as V  # noqa: E402
import test_build_recovery_data as T  # noqa: E402

B.A = R.A = V.A = ac_common
RULE = ac_common.CONTRACTS["v5"]["element_rule"]


def test_every_recovery_row_comes_with_its_twin():
    eps = T.mail_eps()
    rec, twins = V.recovery_with_twins(eps, RULE, random.Random(0), cap=10)
    assert len(rec) == len(twins) >= 1
    for r, t in zip(rec, twins):
        f_ep, step = r["branch_from"]
        assert t["ep"] == f_ep and int(t["step"]) == step and t["img"] == r["img"]        # same screen
        assert t["action"]["action_type"] != "navigate_back" and t["goal"] != r["goal"]   # other goal, its own gold
        assert t["aug"] == "recover_twin"


def test_chain_cap_is_raised_for_v5():
    assert V.CHAIN_CAP > 300
