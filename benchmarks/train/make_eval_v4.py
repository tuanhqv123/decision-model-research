#!/usr/bin/env python3
"""Evaluation rows for contract v4 (2026-11-12): the v3 evaluation files (AC-High test `compact_actest_v3.jsonl`,
`compact_validation_v3.jsonl`) with the v4 serving conditions: `is_focused` False (AndroidWorld never reports it) and the
history recomputed with `effect_changed_v4` over each episode's own element lists (build_v4_data.entries_v4). Rows,
golds and images are unchanged. Rows of an episode with missing steps keep their v3 history (counted and printed).
    python3 make_eval_v4.py CODE_DIR IN.jsonl OUT.jsonl"""
import collections
import json
import sys


def main(code_dir, src, dst):
    sys.path.insert(0, code_dir)
    import ac_common as A
    import build_v4_data as B
    B.A = A
    rule = A.CONTRACTS["v4"]["element_rule"]
    rows = [json.loads(l) for l in open(src)]
    for r in rows:
        r["elements"] = B.unfocus(r.get("elements"))
    eps = collections.defaultdict(list)
    for r in rows:
        eps[r.get("ep")].append(r)
    full = partial = 0
    for ep, rs in eps.items():
        rs.sort(key=lambda r: r["step"])
        if [r["step"] for r in rs] == list(range(len(rs))):
            for r, h in zip(rs, B.entries_v4(rs, rule)):
                r["history_v3"] = h
            full += 1
        else:
            partial += 1
    with open(dst, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, separators=(",", ":")) + "\n")
    print("rows %d, episodes recomputed %d, episodes kept with v3 history (incomplete) %d" % (len(rows), full, partial))


if __name__ == "__main__":
    main(*sys.argv[1:4])
