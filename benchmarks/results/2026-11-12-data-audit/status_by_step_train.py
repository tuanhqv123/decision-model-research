"""Training: P(gold = status | step index), and among episodes still running at that step. (2026-11-12)
    python status_by_step_train.py DATA.jsonl"""
import collections as C, json, sys
c = C.defaultdict(C.Counter)
for l in open(sys.argv[1]):
    r = json.loads(l); b = min(r["step"], 15); c[b]["n"] += 1; c[b]["status"] += r["action"]["action_type"] == "status"
for b in sorted(c): print("step %s%-3s rows %5d  P(status) %.1f%%" % (b, "+" if b == 15 else "", c[b]["n"], 100 * c[b]["status"] / c[b]["n"]))
