"""Recovery rows from AndroidControl alone (2026-11-12): a wrong step's REAL consequence is taken from another episode
that acted differently on the same screen; the row teaches going back. No emulator, no self-play."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ac_common  # noqa: E402
import build_v4_data as B  # noqa: E402
import build_recovery_data as R  # noqa: E402

B.A = ac_common
R.A = ac_common
RULE = ac_common.CONTRACTS["v4"]["element_rule"]


def el(text, box, cls="android.widget.TextView", pkg="com.example.mail", **flags):
    d = {"text": text, "class_name": cls, "package_name": pkg, "is_visible": True,
         "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box))}
    d.update({"is_%s" % k: v for k, v in flags.items()})
    return d


INBOX = [el("Compose", (800, 2100, 1040, 2250), cls="android.widget.Button", clickable=True),
         el("Settings", (40, 140, 300, 260), cls="android.widget.Button", clickable=True)]
COMPOSE = [el("To", (0, 300, 1080, 400), cls="android.widget.EditText", editable=True, clickable=True)]
SETTINGS = [el("Notifications", (0, 300, 1080, 400), clickable=True), el("Theme", (0, 420, 1080, 520), clickable=True)]
BOX = {"Compose": (800, 2100, 1040, 2250), "Settings": (40, 140, 300, 260)}


def row(ep, step, n, action, els, target=None, goal="g", hist=None):
    t = dict(zip(("x_min", "y_min", "x_max", "y_max"), BOX[target])) if target else None
    return {"ep": ep, "step": step, "n_steps": n, "goal": goal, "W": 1080, "H": 2400, "img": "%s_%d.jpg" % (ep, step),
            "elements": els, "action": action, "target": t, "target_el_idx": None, "history_v3": list(hist or []),
            "history": [], "split": "train", "aug": "base"}


def mail_eps():
    e = [row("E", 0, 2, {"action_type": "click"}, INBOX, "Compose", goal="Write to Ann"),
         row("E", 1, 2, {"action_type": "type_text", "text": "Ann"}, COMPOSE, goal="Write to Ann")]
    f = [row("F", 0, 2, {"action_type": "click"}, INBOX, "Settings", goal="Turn off notifications"),
         row("F", 1, 2, {"action_type": "click"}, SETTINGS, goal="Turn off notifications")]
    return {"E": e, "F": f}


def test_wrong_step_takes_the_real_next_screen_from_another_episode_and_teaches_back():
    rows = R.recover_back_rows(mail_eps(), RULE, random.Random(0), cap=10)
    mine = [r for r in rows if r["goal"] == "Write to Ann"]
    assert len(mine) == 1
    r = mine[0]
    assert r["img"] == "F_1.jpg" and r["elements"] == SETTINGS          # the real screen the wrong click led to
    assert r["action"] == {"action_type": "navigate_back"} and r["aug"] == "recover_back"
    assert r["history_v3"][-1] == 'click "Settings" → screen changed'     # the contract's own history format
    assert r["target"] is None and r["ep"] == "E"
    assert r["branch_from"] == ["F", 1]          # the twin: F's own row on that screen (goal F, F's gold)


def test_no_row_when_the_other_screen_is_on_this_episodes_own_path():
    eps = mail_eps()
    eps["E"].insert(1, row("E", 1, 3, {"action_type": "navigate_back"}, SETTINGS, goal="Write to Ann"))
    rows = R.recover_back_rows(eps, RULE, random.Random(0), cap=10)
    assert not [r for r in rows if r["goal"] == "Write to Ann"]


def test_system_ui_and_keyboard_screens_are_never_used_as_the_branch_point():
    eps = mail_eps()
    for r in eps["E"][:1] + eps["F"][:1]:
        r["elements"] = [dict(e, package_name="com.android.systemui") for e in r["elements"]]
    assert R.recover_back_rows(eps, RULE, random.Random(0), cap=10) == []


def test_cap_bounds_the_number_of_rows():
    eps = {}
    for k in range(6):
        for src in ("E", "F"):
            for r in mail_eps()[src]:
                eps.setdefault(src + str(k), []).append(dict(r, ep=src + str(k), img="%s%d_%d.jpg" % (src, k, r["step"])))
    assert len(R.recover_back_rows(eps, RULE, random.Random(0), cap=3)) == 3


def test_an_action_that_stays_on_the_same_page_is_not_a_wrong_path():
    """'Pause' on a timer or a tab switch keeps most of the screen: going back is not the recovery, so no row."""
    eps = mail_eps()
    near = INBOX + [el("Sent", (0, 600, 1080, 700), clickable=True)]          # the 'Settings' click barely changed the page
    eps["F"][1] = dict(eps["F"][1], elements=near)
    assert not [r for r in R.recover_back_rows(eps, RULE, random.Random(0), cap=10) if r["goal"] == "Write to Ann"]


def test_an_action_named_in_the_goal_is_not_treated_as_wrong():
    eps = mail_eps()
    for r in eps["E"]:
        r["goal"] = "Open Settings and write to Ann"       # 'Settings' may well be on this goal's path
    assert not [r for r in R.recover_back_rows(eps, RULE, random.Random(0), cap=10) if r["ep"] == "E"]


def test_per_state_gives_several_different_wrong_steps_for_one_state():
    eps = mail_eps()
    THEME = [el("Light", (0, 300, 1080, 400), clickable=True), el("Dark", (0, 420, 1080, 520), clickable=True)]
    BOX["Theme"] = (40, 300, 300, 380)
    g = [row("G", 0, 2, {"action_type": "click"}, INBOX + [el("Theme", (40, 300, 300, 380), cls="android.widget.Button",
                                                                   clickable=True)], "Theme", goal="Use dark theme"),
         row("G", 1, 2, {"action_type": "click"}, THEME, goal="Use dark theme")]
    eps["E"][0]["elements"] = g[0]["elements"]; eps["F"][0]["elements"] = g[0]["elements"]
    eps["G"] = g
    one = [r for r in R.recover_back_rows(eps, RULE, random.Random(0), cap=50) if r["goal"] == "Write to Ann"]
    two = [r for r in R.recover_back_rows(eps, RULE, random.Random(0), cap=50, per_state=3) if r["goal"] == "Write to Ann"]
    assert len(one) == 1 and len(two) == 2 and len({r["img"] for r in two}) == 2
