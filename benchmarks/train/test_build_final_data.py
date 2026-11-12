"""build_final_data (2026-11-12): the consolidated final mix, from the v5 curated rows plus the base episodes.

Each test pins one decision taken from the 2026-11-12 trace analysis of the 91 failed AndroidWorld tasks and the
ckpt_250 early warning (docs/research/2026-11-12-FINAL-PLAN-vi.md)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common  # noqa: E402
import build_final_data as F  # noqa: E402
import test_build_recovery_data as T  # noqa: E402

F.A = ac_common


def _rec(k):
    back = dict(T.row("E%d" % k, 1, 2, {"action_type": "navigate_back"}, T.SETTINGS), aug="recover_back",
                branch_from=["F%d" % k, 1])
    twin = dict(T.row("F%d" % k, 1, 2, {"action_type": "click"}, T.SETTINGS), aug="recover_twin")
    return back, twin


def test_recovery_is_cut_to_the_cap_and_every_kept_back_row_keeps_its_twin():
    """ckpt_250 misused back 44/798 (v4 16/798): the recovery dose is cut, pairs kept whole."""
    rows = []
    for k in range(10):
        rows += list(_rec(k))
    base = [dict(T.row("B", 0, 1, {"action_type": "click"}, T.INBOX, "Compose"), aug="base")]
    out = F.cut_recovery(rows + base, random.Random(0), cap=3)
    backs = [r for r in out if r["aug"] == "recover_back"]
    twins = {(r["ep"], r["step"]) for r in out if r["aug"] == "recover_twin"}
    assert len(backs) == 3 and len(twins) == 3
    assert all(tuple(b["branch_from"]) in twins for b in backs)
    assert any(r["aug"] == "base" for r in out)


def test_long_press_and_the_step_after_it_are_repeated():
    """Markor delete/move, Files, SMS resend all start with a long press (P(long_press) 0.0002-0.04 on AW)."""
    ep = [T.row("L", 0, 3, {"action_type": "click"}, T.INBOX, "Compose"),
          T.row("L", 1, 3, {"action_type": "long_press"}, T.INBOX, "Settings"),
          T.row("L", 2, 3, {"action_type": "click"}, T.SETTINGS),
          T.row("L", 3, 3, {"action_type": "status"}, T.SETTINGS)]
    out = F.long_press_repeats({"L": ep}, k=4)
    steps = sorted(r["step"] for r in out)
    assert steps == [1] * 4 + [2] * 4 and all(r["aug"] == "lp_rep" for r in out)


def test_a_screen_already_in_the_goal_state_is_done_without_the_flip():
    """Verify tasks start satisfied (Wi-Fi already off); the model flipped it back. From an AC episode whose last action
    flips a switch: the status screen with the history BEFORE the flip is labelled status."""
    sw = T.el("Wi-Fi", (900, 500, 1040, 560), cls="android.widget.Switch", clickable=True, checkable=True, checked=False)
    flip = T.row("W", 1, 2, {"action_type": "click"}, [sw], hist=["open_app Settings → screen changed"])
    flip["target_el_idx"] = 0
    done = T.row("W", 2, 2, {"action_type": "status", "goal_status": "successful"}, [dict(sw, is_checked=True)],
                 hist=["open_app Settings → screen changed", 'click "Wi-Fi" → screen changed'])
    other = T.row("O", 1, 2, {"action_type": "click"}, T.INBOX, "Compose")
    out = F.satisfied_rows({"W": [flip, done], "O": [other, dict(other, step=2, action={"action_type": "status"})]})
    assert len(out) == 1
    r = out[0]
    assert r["action"]["action_type"] == "status" and r["aug"] == "satisfied"
    assert r["history_v3"] == ["open_app Settings → screen changed"] and r["img"] == done["img"]


def test_a_back_row_whose_twin_was_curated_away_is_not_chosen():
    rows = []
    for k in range(6):
        back, twin = _rec(k)
        rows += [back] + ([twin] if k % 2 == 0 else [])         # twins of k = 1, 3, 5 were removed by curation
    out = F.cut_recovery(rows, random.Random(0), cap=10)
    backs = [r for r in out if r["aug"] == "recover_back"]
    assert sorted(b["branch_from"][0] for b in backs) == ["F0", "F2", "F4"]
    assert len([r for r in out if r["aug"] == "recover_twin"]) == 3


def test_answer_episodes_asking_about_an_id_label_are_dropped():
    good = dict(T.row("answer:1", 0, 1, {"action_type": "answer"}, T.INBOX), aug="answer",
                goal="Do X. Then answer: what is the Theme?")
    bad = dict(T.row("answer:2", 0, 1, {"action_type": "answer"}, T.INBOX), aug="answer",
               goal="Do Y. Then read the value of '[id: alarm recycler view]' and answer with it only.")
    out = F.drop_id_keyed_answers([good, bad, dict(bad, step=1)])
    assert [r["ep"] for r in out] == ["answer:1"]
