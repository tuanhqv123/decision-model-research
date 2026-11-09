"""Curate compact_train.jsonl into small, clean, eval-shaped training sets (quality over quantity).

Rule filters (each counted and reported):
  click/long_press : the gold point must fall inside some labeled-or-actionable element (else a tap on a dead
                     zone: likely a bad label), the target must not be a >20%-of-screen container, and the step instruction must not describe
                     another action (swipe/scroll/type/back/open app/wait).
  scroll           : instruction must mention scroll/swipe (if it names another action, drop).
  type_text        : a target field exists and the text is copyable from the state (goal or history), since
                     the decision model answers it by pointing into the state.
  open_app         : app name non-empty.
  wait             : mostly noise in AndroidControl and 1% of the eval -> capped.
Then sample to the AC-v2 eval's action mix (status 22%, click 53%, scroll 9%, open_app 7%, back 5%, type 3%,
wait 1%), stratified across episodes.

Usage: python curate_ac.py acdata <name> <n_total> [seed] [exclude_name,...]
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict

data, name, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
MIX = {"status": .22, "click": .53, "scroll": .09, "open_app": .07, "navigate_back": .05, "type_text": .03,
       "wait": .01}

recs = [json.loads(l) for l in open(f"{data}/compact_train.jsonl")]
excl = set()
for name_x in (sys.argv[5].split(",") if len(sys.argv) > 5 else []):  # e.g. cur15k: never re-use those rows
    for l in open(f"{data}/compact_{name_x}.jsonl"):
        r = json.loads(l)
        excl.add((r["ep"], r["step"]))
print(f"excluding {len(excl)} rows already used")
# the current step's own instruction = the next decision's history entry
instr = {}
for r in recs:
    for j, h in enumerate(r["history"]):
        instr[(r["ep"], j)] = h

OTHER = {
    "click": r"\b(swipe|scroll|type|enter the|input|go back|navigate back|press back|wait)\b",
    "scroll": r"\b(click|tap|select|type|enter|go back|open the)\b",
}
SCROLLY = r"\b(scroll|swipe)\b"

drop, keep = Counter(), defaultdict(list)
for r in recs:
    if (r["ep"], r["step"]) in excl:
        drop["already_used"] += 1; continue
    a, at = r["action"], r["action"]["action_type"]
    if at == "long_press":
        at_k = "click"  # rare (157); grouped with clicks for the mix, label kept as long_press
    else:
        at_k = at
    ins = instr.get((r["ep"], r["step"]), "").lower()
    W, H = r["W"], r["H"]
    if at in ("click", "long_press"):
        t = r["target"]
        if t is None:
            drop["click:no_target"] += 1; continue
        if (t["x_max"] - t["x_min"]) * (t["y_max"] - t["y_min"]) > 0.2 * W * H:
            drop["click:container"] += 1; continue
        gx, gy = a["x"], a["y"]  # the smallest node may be an unlabeled icon INSIDE a button: that is valid;
        if not any(c["bbox"]["x_min"] <= gx <= c["bbox"]["x_max"] and c["bbox"]["y_min"] <= gy <= c["bbox"]["y_max"]
                   for c in r["cands"]):  # a tap on no labeled/actionable element at all is likely a bad label
            drop["click:dead_zone"] += 1; continue
        if ins and re.search(OTHER["click"], ins):
            drop["click:instruction_says_other"] += 1; continue
    elif at == "scroll":
        if ins and not re.search(SCROLLY, ins) and re.search(OTHER["scroll"], ins):
            drop["scroll:instruction_says_other"] += 1; continue
    elif at == "type_text":
        text = (a.get("text") or "").strip().lower()
        hay = (r["goal"] + " " + " ".join(r["history"])).lower()
        if r["target"] is None or not text:
            drop["type:no_field"] += 1; continue
        if text not in hay:
            drop["type:not_copyable"] += 1; continue
    elif at == "open_app":
        if not (a.get("app_name") or "").strip():
            drop["open_app:empty"] += 1; continue
    keep[at_k].append(r)

rng = random.Random(seed)
out = []
for k, frac in MIX.items():
    pool = keep.get(k, [])
    rng.shuffle(pool)
    want = round(frac * total)
    # stratify: at most 3 decisions of one type per episode before taking more
    per_ep, picked, rest = Counter(), [], []
    for r in pool:
        (picked if per_ep[r["ep"]] < 3 else rest).append(r)
        per_ep[r["ep"]] += 1
    sel = (picked + rest)[:want]
    out += sel
    print(f"  {k:14s} pool={len(pool):6d} want={want:6d} got={len(sel):6d}")
rng.shuffle(out)
with open(f"{data}/compact_{name}.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("dropped:", dict(drop))
print(f"kept pool: { {k: len(v) for k, v in keep.items()} }")
print(f"wrote {data}/compact_{name}.jsonl: {len(out)} decisions, {len(set(r['ep'] for r in out))} episodes")
