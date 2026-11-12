"""Replication of F9 on independent AndroidWorld runs of OTHER checkpoints (all trained on AndroidControl screens only,
so the same AW apps have no training screens). Same task->app map and same covered-app set as app_coverage.py.
    python coverage_replication.py RUN_DIR [RUN_DIR ...]"""
import glob, json, math, os, sys
sys.argv, runs = sys.argv[:1], sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "app_coverage.py")).read()
PREFIX = eval(src.split("PREFIX = ", 1)[1].split("\nper =")[0])
import ast
rows = ast.literal_eval(next(l for l in open(os.path.join(here, "train_facts.out")) if l.startswith("rows per AndroidWorld app")).split("app", 1)[1].strip())
def fisher(a, n1, b, n2):
    K = a + b; N = n1 + n2
    return sum(math.comb(n1, x) * math.comb(n2, K - x) for x in range(a, min(n1, K) + 1)) / math.comb(N, K)
FLOOR = os.environ.get("FLOOR_RUN"); floor = set(x for x in os.environ.get("FLOOR_TASKS", "").split(",") if x)
if FLOOR:
    for f in glob.glob(os.path.join(FLOOR, "episodes", "*.jsonl")):
        r = [json.loads(l) for l in open(f) if '"type": "result"' in l]
        if r and r[-1].get("success"): floor.add(r[-1]["task_name"])
if floor:
    print("excluding %d tasks a do-nothing agent (Q11, status at step 0) already solves: %s" % (len(floor), sorted(floor)))
for run in runs:
    res = {}
    for f in [g for d in run.split("+") for g in glob.glob(os.path.join(d, "episodes", "*.jsonl"))]:
        if ".abandoned." in f: continue
        r = [json.loads(l) for l in open(f) if '"type": "result"' in l]
        if r and not (r[-1].get("setup_error") or r[-1].get("unscored")): res[r[-1]["task_name"]] = bool(r[-1].get("success"))
    g = {"rows": [0, 0], "zero": [0, 0]}
    for t, ok in res.items():
        if t in floor: continue
        app = next(a for p, a in PREFIX if t.startswith(p)); k = "rows" if rows.get(app, 0) > 0 else "zero"
        g[k][0] += ok; g[k][1] += 1
    print("%-24s app has rows %2d/%2d = %4.1f%% | zero rows %2d/%2d = %4.1f%% | one-sided Fisher p = %.4f" % (
        "+".join(os.path.basename(d.rstrip("/")) for d in run.split("+")), *g["rows"], 100 * g["rows"][0] / max(1, g["rows"][1]),
        *g["zero"], 100 * g["zero"][0] / max(1, g["zero"][1]), fisher(*g["rows"], *g["zero"])))

