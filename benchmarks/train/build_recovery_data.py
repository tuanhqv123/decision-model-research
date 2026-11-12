#!/usr/bin/env python3
"""Recovery rows from AndroidControl alone (2026-11-12). No emulator, no self-play, no external model.

Why: on AndroidWorld neither v3 nor v4 ever chose `navigate_back` (0 of 2,835 steps; mean P 0.008), while the
human AndroidControl data uses it on 3.2% of steps. After one wrong step the agent never recovers. The first wrong
step of 33% of scored failures is a wrong path inside the app (benchmarks/results/2026-11-12-erroranalysis/).

How: AndroidControl episodes share screens (2,136 screens of the AW-like set appear in >= 2 episodes with different
actions). For episode E at screen s, another episode F that acted differently on the same screen gives the REAL
next screen of that action. If E never visits that screen, it is off E's path, so the row is:
    goal of E, history of E up to s + '<F's action> → screen changed', screen = F's next screen, gold = navigate_back.

    python3 build_recovery_data.py CODE_DIR IN_ROWS.jsonl OUT.jsonl [--cap N] [--seed 0]
"""
import collections
import json
import random
import re
import sys

import build_v4_data as B

A = None  # ac_common, set in main() or by the tests

NO_BRANCH = ("com.android.systemui", "com.google.android.inputmethod.latin", "com.google.android.apps.nexuslauncher",
             "com.android.launcher3", "com.sec.android.app.launcher")
NAV_ACTIONS = ("click", "long_press")
SAME_PAGE = 0.5   # Jaccard of interactive elements at or above this: the action stayed on the page (tab, pause, toggle)


def named_in_goal(element, goal):
    """The 'wrong' element's label shares a word (4+ letters) with the goal: it may be on the goal's path."""
    words = {w for w in re.findall(r"[a-z]{4,}", (element.get("label") or "").lower())}
    return bool(words & set(re.findall(r"[a-z]{4,}", (goal or "").lower())))


def same_page(a, b):
    sa, sb = set(a[1]), set(b[1])
    return bool(sa | sb) and len(sa & sb) / len(sa | sb) >= SAME_PAGE


def screen_key(row, rule):
    """Same app + same set of interactive elements (class, label with digits masked): 'the same screen'."""
    kept, _ = A.select_elements(row["elements"], row["W"], row["H"], rule)
    pk = collections.Counter(e.get("package_name") for e in row["elements"]).most_common(1)
    pkg = pk[0][0] if pk else ""
    keys = tuple(sorted({(e["class_short"], re.sub(r"\d", "#", e["label"] or "")[:20]) for e in kept
                         if e["flags"]["clickable"] or e["flags"]["editable"]}))
    return pkg, keys


def recover_back_rows(rows_by_ep, rule, rng, cap, per_state=1):
    B.A = A
    keys = {ep: [screen_key(r, rule) for r in rows] for ep, rows in rows_by_ep.items()}
    index = collections.defaultdict(list)                      # screen -> [(ep, t)] where a nav action was taken
    for ep, rows in rows_by_ep.items():
        for t, r in enumerate(rows[:-1]):
            k = keys[ep][t]
            if k[1] and k[0] not in NO_BRANCH and r["action"]["action_type"] in NAV_ACTIONS:
                index[k].append((ep, t))
    out = []
    for ep, rows in rows_by_ep.items():
        own = set(keys[ep])
        for t, r in enumerate(rows):
            k = keys[ep][t]
            if not k[1] or k[0] in NO_BRANCH:
                continue
            mine = B.acted(r, rule)
            mine_key = A._effect_key_v4(mine) if mine is not None else None
            cands = []
            for f_ep, j in index.get(k, ()):
                if f_ep == ep:
                    continue
                other = B.acted(rows_by_ep[f_ep][j], rule)
                if other is None or (mine_key is not None and A._effect_key_v4(other) == mine_key):
                    continue                                    # the same action is not a wrong step
                if named_in_goal(other, r["goal"]):
                    continue
                nxt = keys[f_ep][j + 1]
                if nxt in own or nxt[0] != k[0] or not nxt[1] or same_page(k, nxt):
                    continue                                    # on E's own path, left the app, empty, or the same page
                cands.append((f_ep, j, other))
            if not cands:
                continue
            seen_next = set()
            rng.shuffle(cands)
            for f_ep, j, other in cands:
                nxt = keys[f_ep][j + 1]
                if nxt in seen_next:
                    continue                                    # one row per distinct wrong destination
                seen_next.add(nxt)
                src, nxt_row = rows_by_ep[f_ep][j], rows_by_ep[f_ep][j + 1]
                line = A.history_entry_v3(src["action"], other, True)
                out.append(dict(nxt_row, ep=r["ep"], step=r["step"], goal=r["goal"], split=r.get("split", "train"),
                                history_v3=list(r["history_v3"]) + [line], history=list(r.get("history") or []) + [line],
                                action={"action_type": "navigate_back"}, target=None, target_el_idx=None,
                                target_el_exact=None, target_el_hits_gold=None, target_el_centre_idx=None,
                                local_goal="Go back", local_done=None, aug="recover_back",
                                branch_from=[f_ep, int(nxt_row["step"])]))
                if len(seen_next) >= per_state:
                    break
    rng.shuffle(out)
    return out[:cap]


def main(code_dir, in_rows, out_path, cap=3000, seed=0, per_state=1):
    global A
    sys.path.insert(0, code_dir)
    import ac_common as _A
    A = _A
    rule = A.CONTRACTS["v4"]["element_rule"]
    by_ep = collections.defaultdict(list)
    for l in open(in_rows):
        r = json.loads(l)
        if (r.get("aug") or "base") == "base" and r.get("split", "train") == "train":
            by_ep[r["ep"]].append(r)
    for rows in by_ep.values():
        rows.sort(key=lambda r: int(r["step"]))
    rows = recover_back_rows(by_ep, rule, random.Random(seed), cap, per_state)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    apps = collections.Counter(collections.Counter(e.get("package_name") for e in r["elements"]).most_common(1)[0][0]
                               for r in rows if r["elements"])
    print("recover_back rows %d from %d episodes; top apps %s" % (len(rows), len(by_ep), apps.most_common(8)))


if __name__ == "__main__":
    a = sys.argv[1:]
    cap = int(a[a.index("--cap") + 1]) if "--cap" in a else 3000
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 0
    per = int(a[a.index("--per-state") + 1]) if "--per-state" in a else 1
    main(a[0], a[1], a[2], cap, seed, per)
