"""records/*.jsonl (full a11y nodes, ~8GB) -> compact_<split>.jsonl for training.

Per record keeps only what training reads: goal, history (earlier steps' instructions), image, size, action,
target bbox, and the precomputed a11y candidate list (ac_common.pick_candidates, the same function eval uses).
Streams shard by shard (an episode's records are contiguous within its shard) so RAM stays small.
"""
import glob
import json
import sys

from ac_common import pick_candidates

data = sys.argv[1]
outs = {s: open(f"{data}/compact_{s}.jsonl", "w") for s in ("train", "validation")}
n = {"train": 0, "validation": 0}


def keep_bbox(b):
    return {k: b[k] for k in ("x_min", "x_max", "y_min", "y_max", "center")}


for p in sorted(glob.glob(f"{data}/records/*.jsonl")):
    eps = {}
    for line in open(p):
        r = json.loads(line)
        eps.setdefault(r["ep"], []).append(r)
    for ep, rs in eps.items():
        instr = {r["step"]: r["step_instruction"] for r in rs if r["action"]["action_type"] != "status"}
        for r in rs:
            if r["action"]["action_type"] == "navigate_home":
                continue  # not an eval action; still part of later steps' history
            cands = pick_candidates(r["nodes"], r["W"], r["H"])
            for c in cands:
                c["bbox"] = keep_bbox(c["bbox"])
            rec = {"ep": ep, "step": r["step"], "goal": r["goal"],
                   "history": [instr.get(j, "") for j in range(r["step"])],
                   "img": r["img"], "W": r["W"], "H": r["H"], "action": r["action"],
                   "target": keep_bbox(r["target"]) if r["target"] else None, "cands": cands}
            outs[r["split"]].write(json.dumps(rec) + "\n")
            n[r["split"]] += 1
    print(p, n, flush=True)
print("COMPACT_DONE", n)
