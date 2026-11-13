"""AndroidWorld self-play (v3 ckpt_1824, seed 1001): mean model P(terminate) and argmax-terminate rate by step, on states the
checker says are NOT done, and the same on states that ARE done. From the recorded readouts. (2026-11-12)
    python3 status_by_step_aw.py EPISODE_DIR"""
import collections as C, glob, json, os, sys
c = C.defaultdict(C.Counter)
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    if ".abandoned." in f: continue
    st = sorted([json.loads(l) for l in open(f) if '"type": "step"' in l], key=lambda s: s["i"])
    for k, s in enumerate(st):
        p = ((s.get("state_in") or {}).get("readouts") or {}).get("action", {}).get("probabilities")
        if not p: continue
        done = k > 0 and (st[k - 1].get("probe_score") or 0) >= 1
        b = (min(s["i"], 15), done); c[b]["n"] += 1; c[b]["p"] += p.get("terminate", 0); c[b]["argmax"] += max(p, key=p.get) == "terminate"
for b in sorted(c): print("step %s%-3s %-9s n %4d  mean P(terminate) %5.1f%%  argmax terminate %5.1f%%" % (b[0], "+" if b[0] == 15 else "", "DONE" if b[1] else "not done", c[b]["n"], 100 * c[b]["p"] / c[b]["n"], 100 * c[b]["argmax"] / c[b]["n"]))
