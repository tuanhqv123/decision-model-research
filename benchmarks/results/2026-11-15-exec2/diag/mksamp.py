import json, random, collections, os
eps=collections.defaultdict(list)
for l in open("acdata_exec/compact_exec_v1.jsonl"):
    r=json.loads(l)
    if r["aug"] in ("span","span_done"):
        eps[r["ep"][5:] if str(r["ep"]).startswith("done:") else str(r["ep"])].append(l)
def has_cont(lines):
    rs=[json.loads(l) for l in lines if json.loads(l)["aug"]=="span"]
    by={r["step"]:r for r in rs}
    return any(by.get(r["step"]-1) and by[r["step"]-1]["local_goal"]==r["local_goal"] and by[r["step"]-1].get("local_done") is False for r in rs)
keys=sorted(eps); random.Random(7).shuffle(keys)
out=[]; n=0
for k in keys:
    if has_cont(eps[k]):
        out+=eps[k]; n+=1
    if n==150: break
os.makedirs("diag/data",exist_ok=True)
open("diag/data/compact_trainsamp_exec.jsonl","w").writelines(out)
print(n,"episodes",len(out),"rows")
v=json.loads(open("g2val/compact_validation_v4.jsonl").readline()); print("val_v4 local_goal:",v.get("local_goal"),"| local_done:",v.get("local_done"))
