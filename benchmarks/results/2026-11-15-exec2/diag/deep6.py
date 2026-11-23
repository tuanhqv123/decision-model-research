import json, collections
def clean(s): return " ".join((s or "").split()).rstrip(" .")
def said(p): return p["pred"] in ("status","terminate")
low=sorted(({"ep":str(r["ep"]),"step":r["step"],"lg":r["local_goal"],"ld":r["local_done"],"a":r["action"]["action_type"],"h":(r.get("history_v3") or [""])[-1] if r.get("history_v3") else ""} for r in map(json.loads,open("actest/data/compact_actest_v4.jsonl"))),key=lambda r:(r["ep"],r["step"]))
by={(r["ep"],r["step"]):r for r in low}
E2={(str(p["episode_id"]),p["step"]):p for p in map(json.loads,open("gate_exec2/eval-exec2_actest_low-VA-val.jsonl"))}
c=collections.defaultdict(lambda:[0,0])
for r in low:
    pr=by.get((r["ep"],r["step"]-1))
    if not (pr and clean(pr["lg"])==clean(r["lg"]) and pr["ld"] is False): continue
    eff="changed" if "screen changed" in r["h"] else "no change" if "no change" in r["h"] else "other:"+r["h"][-20:]
    k="gold %s | last history %s → %s"%(r["a"], r["h"].split(" ")[0], eff)
    c[k][0]+=1; c[k][1]+=said(E2[(r["ep"],r["step"])])
for k,v in sorted(c.items(),key=lambda x:-x[1][0])[:14]: print(v,k)
