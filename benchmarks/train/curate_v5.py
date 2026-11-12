#!/usr/bin/env python3
"""Curate the v5 training rows (2026-11-12): deduplicate and rebalance instead of training on a large redundant set.

Measured on compact_awlike_full_v4: 37,977 rows but only 17,052 distinct (screen, action) pairs; 74% of rows sit on a
pair seen more than once; click is 58% of base rows.
  * base rows: at most PER_KEY rows per (screen, action, target label), preferring distinct goals;
  * then click capped at CLICK_SHARE of the kept base rows (long_press, navigate_back, type_text, scroll, open_app,
    status, wait all kept);
  * targeted rows (chain, recover_back, recover_twin, answer, nc_*, avoid, drop-free) are kept as built;
  * stop pairs are kept as built: every episode's last action (step n_steps-1) and its `status` row (step n_steps).
    Without this, curation kept both in only 1,918 of 4,191 episodes (2026-11-12), and the pair is the only evidence
    separating "one step from done" from "done" (49 premature `status` failures on AndroidWorld; 460 of 1,171 wrong
    `terminate` predictions on the AndroidControl test fall on an episode's last real step).

    python3 curate_v5.py CODE_DIR IN.jsonl OUT.jsonl [--per-key 2] [--click-share 0.45] [--seed 0]
"""
import collections
import json
import random
import re
import sys

A = None
TARGETED = ("chain", "recover_back", "recover_twin", "answer", "avoid", "nc_field", "nc_repeat", "nc_other",
            "nc_scroll", "nc_status")


def row_key(r):
    rule = A.CONTRACTS["v5"]["element_rule"]
    kept, _ = A.select_elements(r["elements"], r["W"], r["H"], rule)
    pk = collections.Counter(e.get("package_name") for e in r["elements"]).most_common(1)
    screen = (pk[0][0] if pk else "", tuple(sorted({(e["class_short"], re.sub(r"\d", "#", e["label"] or "")[:20])
                                                    for e in kept if e["flags"]["clickable"] or e["flags"]["editable"]})))
    a = r.get("action") or {}
    t = r.get("target") or {}
    return (screen, a.get("action_type"), a.get("app_name"), a.get("direction"),
            json.dumps(t, sort_keys=True) if t else None)


def is_stop_pair(r):
    """A base row that is its episode's `status` row or the action just before it."""
    return r.get("aug") in (None, "base") and r.get("n_steps") is not None and r["step"] >= r["n_steps"] - 1


def curate(rows, rng, per_key=2, click_share=0.45):
    targeted = [r for r in rows if r.get("aug") in TARGETED or is_stop_pair(r)]
    base = [r for r in rows if not (r.get("aug") in TARGETED or is_stop_pair(r))]
    groups = collections.defaultdict(list)
    for r in base:
        groups[row_key(r)].append(r)
    kept = []
    for g in groups.values():
        rng.shuffle(g)
        seen_goals, pick = set(), []
        for r in g:                                   # distinct goals first
            if r["goal"] not in seen_goals:
                pick.append(r); seen_goals.add(r["goal"])
            if len(pick) >= per_key:
                break
        kept += pick
    clicks = [r for r in kept if (r.get("action") or {}).get("action_type") == "click"]
    others = [r for r in kept if (r.get("action") or {}).get("action_type") != "click"]
    max_click = int(click_share / (1 - click_share) * len(others)) if click_share < 1 else len(clicks)
    rng.shuffle(clicks)
    out = others + clicks[:max_click] + targeted
    rng.shuffle(out)
    return out


def main(code_dir, inp, outp, per_key=2, click_share=0.45, seed=0):
    global A
    sys.path.insert(0, code_dir)
    import ac_common as _A
    A = _A
    rows = [json.loads(l) for l in open(inp)]
    out = curate(rows, random.Random(seed), per_key, click_share)
    with open(outp, "w") as f:
        for r in out:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    def dist(rs):
        c = collections.Counter((r.get("action") or {}).get("action_type", "avoid") for r in rs)
        n = len(rs)
        return {k: "%d (%.1f%%)" % (v, 100 * v / n) for k, v in c.most_common()}
    print("IN ", len(rows), dist(rows))
    print("OUT", len(out), dist(out))
    print("OUT aug", dict(collections.Counter(r.get("aug") for r in out).most_common()))


if __name__ == "__main__":
    a = sys.argv[1:]
    get = lambda k, d, f: f(a[a.index(k) + 1]) if k in a else d
    main(a[0], a[1], a[2], get("--per-key", 2, int), get("--click-share", 0.45, float), get("--seed", 0, int))
