import json, sys, collections, os
sys.path.insert(0, os.path.expanduser("~/d1pilot/code_exec/benchmarks/train"))
import score_exec
def clean(s): return " ".join((s or "").split()).rstrip(" .")
def said(p): return p["pred"] in ("status","terminate")
def unit_pos(rows, key_ep="ep"):
    rows=sorted(rows,key=lambda r:(str(r[key_ep]),r["step"])); pos={}; cur=[]; prev=None
    def close(u):
        for j,r in enumerate(u): pos[(str(r[key_ep]),r["step"])]=(j,len(u))
    for r in rows:
        if prev is None or str(prev[key_ep])!=str(r[key_ep]) or clean(prev["local_goal"])!=clean(r["local_goal"]):
            if cur: close(cur)
            cur=[]
        cur.append(r)
        if r.get("local_done"): close(cur); cur=[]
        prev=r
    if cur: close(cur)
    return pos
V=[{k:r.get(k) for k in ("ep","step","local_goal","local_done","action")} for r in map(json.loads,open(os.path.expanduser("~/d1pilot/g2val/compact_validation_v4.jsonl")))]
V=[r for r in V if r["action"]["action_type"]!="status"]
vpos=unit_pos(V)
T=[{k:r.get(k) for k in ("ep","step","local_goal","local_done","action","aug")} for r in map(json.loads,open(os.path.expanduser("~/d1pilot/diag/data/compact_trainsamp_exec.jsonl")))]
TA=[r for r in T if r["aug"]=="span"]
tby={(str(r["ep"]),r["step"]):r for r in TA}
def tcont(r):
    p=tby.get((str(r["ep"]),r["step"]-1)); return bool(p and p["local_goal"]==r["local_goal"] and p.get("local_done") is False)
out={}
for tag in sys.argv[1:]:
    d=os.path.expanduser("~/d1pilot/diag/out/")
    res={}
    f=d+"eval-%s_vallow-VA-val.jsonl"%tag
    if os.path.exists(f):
        P={(str(p["episode_id"]),p["step"]):p for p in map(json.loads,open(f))}
        c=collections.Counter()
        for r in V:
            k=(str(r["ep"]),r["step"]); p=P[k]; j,n=vpos[k]
            cat="continuation" if j>0 else "first"
            c["n"]+=1; c["correct"]+=bool(p["correct"]); c["said done"]+=said(p)
            c["n "+cat]+=1; c["said done "+cat]+=said(p)
            c["said done, last step of unit"]+=said(p) and j==n-1
        res["val AC-Low style (690 action rows)"]=dict(c)
    f=d+"eval-%s_train-VA-val.jsonl"%tag
    if os.path.exists(f):
        preds=[json.loads(l) for l in open(f)]
        res["train sample, executor scorer"]=score_exec.score(preds,T)
        P={(str(p["episode_id"]),p["step"]):p for p in preds}
        c=collections.Counter()
        for r in TA:
            p=P.get((str(r["ep"]),r["step"]))
            if p is None: continue
            k="continuation" if tcont(r) else "first of instruction"
            c["n "+k]+=1; c["said done "+k]+=said(p)
        res["train sample by row kind"]=dict(c)
    out[tag]=res
print(json.dumps(out,indent=1))
