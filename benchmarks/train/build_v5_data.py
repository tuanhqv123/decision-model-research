#!/usr/bin/env python3
"""Build the contract-v5 training data (2026-11-12, one consolidated retrain). No emulator, no external model.

Evidence (docs/research/2026-11-12-V4-WORKLOG-vi.md, benchmarks/results/2026-11-12-confidence/):
  * premature `status` is the largest AW failure (49/95): in AndroidControl the gold is `status` on 21% of rows right
    after a click and 37% after a wait, and the status share reaches 29% by step 7-8 (episodes are short). Chained
    two-goal episodes (v4) cut P(stop | second task undone) from 0.92 to 0.01 -> far more chains (longer episodes).
  * the agent never goes back (0/1,307 AW steps): recovery rows (build_recovery_data), each with its TWIN — the same
    screen under the goal it is on the path of, with that episode's gold — so the goal, not the screen, decides
    (pilot 1 without twins: back misuse 1-2% -> 10%).
  * rendering (contract v5: resource-id labels, on/off) is applied at train time from the raw elements.

    python3 build_v5_data.py CODE_DIR IN_AC.jsonl OUT_DIR [--seed 0] [--recovery 3500]
"""
import collections
import json
import os
import random
import sys

import build_recovery_data as R
import build_v4_data as B

A = None
CHAIN_CAP = 1200


def recovery_with_twins(rows_by_ep, rule, rng, cap):
    R.A = A
    rec = R.recover_back_rows(rows_by_ep, rule, rng, cap, per_state=1)
    index = {(str(ep), int(r["step"])): r for ep, rows in rows_by_ep.items() for r in rows}
    out_rec, out_twin = [], []
    for r in rec:
        t = index.get((str(r["branch_from"][0]), int(r["branch_from"][1])))
        if t is None or t["action"]["action_type"] == "navigate_back":
            continue
        out_rec.append(r)
        out_twin.append(dict(t, aug="recover_twin"))
    return out_rec, out_twin


def main(code_dir, in_ac, out_dir, seed=0, n_rec=3500):
    global A
    sys.path.insert(0, code_dir)
    import ac_common as _A
    A = B.A = R.A = _A
    rule = A.CONTRACTS["v5"]["element_rule"]
    B.CAPS["chain"] = CHAIN_CAP
    ac = [json.loads(l) for l in open(in_ac)]
    rows, stats = B.build(ac, rule, seed)
    by_ep = collections.OrderedDict()
    for r in rows:
        if r["aug"] == "base":
            by_ep.setdefault(r["ep"], []).append(r)
    for v in by_ep.values():
        v.sort(key=lambda r: int(r["step"]))
    rec, twins = recovery_with_twins(by_ep, rule, random.Random(seed), n_rec)
    rows = rows + rec + twins
    stats["recover_back"], stats["recover_twin"], stats["total rows"] = len(rec), len(twins), len(rows)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "compact_awlike_full_v5.jsonl"), "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, separators=(",", ":")) + "\n")
    json.dump(stats, open(os.path.join(out_dir, "build_v5.out"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    a = sys.argv[1:]
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 0
    n_rec = int(a[a.index("--recovery") + 1]) if "--recovery" in a else 3500
    main(a[0], a[1], a[2], seed, n_rec)
