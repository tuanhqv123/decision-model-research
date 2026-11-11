#!/usr/bin/env python3
"""Add `history_v3` (contract v3 previous-action entries) to full-flow rows: for step t, one entry per earlier step j
of the SAME episode — its gold action, the element it acted on (the supervised kept element, as the live agent names
the element the model chose) and whether the screen changed (elements j vs j+1, `effect_changed`).

    python3 add_history_v3.py IN.jsonl OUT.jsonl     (episodes must be complete: every step 0..n_steps present)
Generator rows (`source == aw_generator`, step 0) get an empty history.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ac_common as A  # noqa: E402

RULE = A.CONTRACTS["v3"]["element_rule"]  # the rule the v3 checkpoint is served with


def acted_element(row):
    """The canonical element the gold action acted on, or None (non-spatial, or no element)."""
    if row["action"]["action_type"] not in A.SPATIAL:
        return None
    kept, idx = A.select_elements(row["elements"], row["W"], row["H"], RULE)
    el, _ = A.gold_target_el_v2(row, idx, kept)
    if el is not None:
        return kept[int(el) - 1]
    i = row.get("target_el_idx")
    return A.canonical_element(row["elements"][i]) if i is not None else None


def episode_entries(rows):
    """rows: one episode, steps 0..T in order. Returns the entry list for every step (entries[t] = history of t)."""
    canon = [[A.canonical_element(e) for e in r["elements"]] for r in rows]
    lines, out = [], []
    for t, r in enumerate(rows):
        out.append(list(lines))
        if t + 1 < len(rows):
            lines.append(A.history_entry_v3(r["action"], acted_element(r), A.effect_changed(canon[t], canon[t + 1])))
    return out


def main(src, dst):
    eps = collections.OrderedDict()
    for line in open(src):
        r = json.loads(line)
        eps.setdefault(r["ep"], []).append(r)
    n = 0
    with open(dst, "w") as fo:
        for ep, rows in eps.items():
            rows.sort(key=lambda r: r["step"])
            if rows[0].get("source") == "aw_generator":
                hist = [[] for _ in rows]
            else:
                if [r["step"] for r in rows] != list(range(rows[0]["n_steps"] + 1)):
                    raise SystemExit(f"episode {ep} is not complete: steps {[r['step'] for r in rows]}")
                hist = episode_entries(rows)
            for r, h in zip(rows, hist):
                r["history_v3"] = h
                fo.write(json.dumps(r, separators=(",", ":")) + "\n")
                n += 1
    print(f"rows={n} episodes={len(eps)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
