import json,glob,collections
rows=[]
for f in sorted(glob.glob("/tmp/awruns/aw116_appint/episodes/*.jsonl")):
    st=[json.loads(l) for l in open(f) if l.strip()]
    steps=[r for r in st if r["type"]=="step"]
    res=[r for r in st if r["type"]=="result"]
    if not res: continue
    r=res[0]
    acts=collections.Counter((s.get("action_raw") or {}).get("action") for s in steps)
    rows.append(dict(task=r.get("task_name"),succ=bool(r.get("success")),tmo=bool(r.get("timeout")),
                     decl=bool(r.get("declared_done")),n=r.get("n_steps") or 0,acts=dict(acts),
                     setup=r.get("setup_error")))
n=len(rows); s=sum(r["succ"] for r in rows)
print("finished=%d success=%d SR=%.1f%%"%(n,s,100*s/max(n,1)))
# buckets
b=collections.Counter()
for r in rows:
    if r["succ"]: b["success"]+=1
    elif r["setup"]: b["setup_error"]+=1
    elif r["tmo"]: b["timeout_no_terminate"]+=1
    elif r["decl"]: b["premature_status"]+=1
    else: b["other_fail"]+=1
print("buckets:",dict(b))
# premature status: declared done with few steps
pre=[r for r in rows if r["decl"] and not r["succ"]]
if pre:
    print("premature: n=%d median steps=%d"%(len(pre),sorted(r["n"] for r in pre)[len(pre)//2]))
tmo=[r for r in rows if r["tmo"]]
if tmo:
    print("timeout:   n=%d median steps=%d"%(len(tmo),sorted(r["n"] for r in tmo)[len(tmo)//2]))
# which tasks succeeded
print("SUCCESSES:", [r["task"] for r in rows if r["succ"]])
