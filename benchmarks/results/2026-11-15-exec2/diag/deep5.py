import json
def clean(s): return " ".join((s or "").split()).rstrip(" .")
def said(p): return p["pred"] in ("status","terminate")
low=sorted(({"ep":str(r["ep"]),"step":r["step"],"lg":r["local_goal"],"ld":r["local_done"]} for r in map(json.loads,open("actest/data/compact_actest_v4.jsonl"))),key=lambda r:(r["ep"],r["step"]))
by={(r["ep"],r["step"]):r for r in low}
def first(r):
    pr=by.get((r["ep"],r["step"]-1))
    return not (pr and clean(pr["lg"])==clean(r["lg"]) and pr["ld"] is False)
L=lambda f:{(str(p["episode_id"]),p["step"]):p for p in map(json.loads,open(f))}
# InfiGUI per-step correctness (the official scorer) lives in perstep files; use the eval's `correct` and check totals
for m,g in (("exec1","gate_exec1/eval-exec1_actest_low-VA-val.jsonl"),("exec2","gate_exec2/eval-exec2_actest_low-VA-val.jsonl")):
    A=L(g); N=L(g.replace("_low-","_low_noterm-"))
    base=sum(bool(A[k]["correct"]) for k in A)
    rule=sum(bool((N if (first(by[k]) and said(A[k])) else A)[k]["correct"]) for k in A)
    nf=sum(1 for k in A if first(by[k]) and said(A[k]))
    print(m,"eval-correct",base,"->",rule,"(%.2f -> %.2f)"%(100*base/8444,100*rule/8444),"| done blocked at first step:",nf)
