"""Rebalance a running arm onto one more emulator (2026-11-12). For the two half-configs of an arm: tasks not yet
scored, in config order; keep the next KEEP for the running chain (it may be mid-chunk), move a tail to a new
config <arm>_c so the three chains end together. Rewrites the half configs in place (run_arm re-reads them every
round) and writes the new one. Prints the task counts each chain must reach."""
import glob, json, os, sys, yaml
arm, keep = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
A = os.path.expanduser("~/aw"); R = os.path.expanduser("~/awruns")
def scored(run):
    done = set()
    for f in glob.glob(f"{R}/{run}/episodes/*.jsonl"):
        if ".abandoned." in f: continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if any(r.get("type") == "result" for r in rows): done.add(rows[0]["task_name"])
    return done
halves = [arm, arm + "_b"]; cfgs = {h: yaml.safe_load(open(f"{A}/{h}.yaml")) for h in halves}
rem = {h: [t for t in cfgs[h]["suite"]["tasks"] if t not in scored(h)] for h in halves}
total = sum(len(v) for v in rem.values()); per = total // 3
tails = {}
for h in halves:
    movable = max(0, len(rem[h]) - keep)
    tails[h] = rem[h][len(rem[h]) - min(movable, max(0, len(rem[h]) - per)):] if movable else []
new = tails[halves[0]] + tails[halves[1]]
for h in halves:
    c = cfgs[h]; c["suite"]["tasks"] = [t for t in c["suite"]["tasks"] if t not in tails[h]]
    src = open(f"{A}/{h}.yaml").read(); head = "".join(l for l in src.splitlines(True) if l.startswith("#"))
    open(f"{A}/{h}.yaml", "w").write(head + "# REBALANCED 2026-11-12: %d tail tasks moved to %s_c\n" % (len(tails[h]), arm) + yaml.safe_dump(c, sort_keys=False))
c = yaml.safe_load(open(f"{A}/{halves[0]}.yaml")); c["suite"]["tasks"] = new
open(f"{A}/{arm}_c.yaml", "w").write("# third chain of %s (rebalanced 2026-11-12): tail tasks of both halves\n" % arm + yaml.safe_dump(c, sort_keys=False))
for h in halves: print(h, "tasks", len(cfgs[h]["suite"]["tasks"]), "remaining", len(rem[h]) - len(tails[h]), "moved", len(tails[h]))
print(arm + "_c", "tasks", len(new))
