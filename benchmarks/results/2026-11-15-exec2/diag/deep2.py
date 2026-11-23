import json, collections, random
def clean(s): return " ".join((s or "").split()).rstrip(" .")
def said(p): return p["pred"] in ("status","terminate")
low=[{k:r.get(k) for k in ("ep","step","local_goal","local_done","action","history_v3","goal")} for r in map(json.loads,open("actest/data/compact_actest_v4.jsonl"))]
low.sort(key=lambda r:(str(r["ep"]),r["step"]))
P={(str(p["episode_id"]),p["step"]):p for p in map(json.loads,open("gate_exec2/eval-exec2_actest_low-VA-val.jsonl"))}
by={(str(r["ep"]),r["step"]):r for r in low}
cat=collections.Counter(); ex=collections.defaultdict(list)
for r in low:
    k=(str(r["ep"]),r["step"]); p=P[k]
    if not said(p): continue
    pr=by.get((k[0],k[1]-1)); nx=by.get((k[0],k[1]+1))
    same_prev = pr is not None and clean(pr["local_goal"])==clean(r["local_goal"])
    same_prev_ci = pr is not None and clean(pr["local_goal"]).lower()==clean(r["local_goal"]).lower()
    last = r.get("local_done")
    c=("prev same text" if same_prev else "prev same text (case-insens.)" if same_prev_ci else "prev different" if pr else "episode start")+" | "+("this step closes the instruction" if last else "instruction continues")
    cat[c]+=1
    ex[c].append({"k":k,"goal":r["goal"][:90],"prev":pr and pr["local_goal"],"prev_done":pr and pr.get("local_done"),"instr":r["local_goal"],"gold":r["action"]["action_type"],"next":nx and nx["local_goal"],"hist_last":(r["history_v3"] or [None])[-1],"p":round(p["probs"]["terminate"],2)})
print(json.dumps(cat.most_common(),indent=0))
random.seed(1)
for c,v in ex.items():
    print("\n=====",c,len(v))
    for e in random.sample(v,min(7,len(v))): print(json.dumps(e))
