"""Validation metrics for the early warning / gate: step acc, episodes all right, false status, back misuse."""
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]
st = lambda a: a in ("status", "terminate")
ns = [r for r in R if r["gold"]["action_type"] != "status"]
nb = [r for r in R if r["gold"]["action_type"] != "navigate_back"]
eps = collections.defaultdict(list)
for r in R:
    eps[r["episode_id"]].append(bool(r["correct"]))
print("rows %d step_acc %d/%d | episodes all right %d/%d | false status %d/%d | back misuse %d/%d | mean P(back|gold!=back) %.4f"
      % (len(R), sum(r["correct"] for r in R), len(R), sum(all(v) for v in eps.values()), len(eps),
         sum(st(r["pred"]) for r in ns), len(ns), sum(r["pred"] == "navigate_back" for r in nb), len(nb),
         sum(r["probs"].get("navigate_back", 0) for r in nb) / len(nb)))
