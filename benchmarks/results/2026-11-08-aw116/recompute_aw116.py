import json,glob,collections
rows=[]
for f in sorted(glob.glob("/tmp/awruns/aw116_appint/episodes/*.jsonl")):
    st=[json.loads(l) for l in open(f) if l.strip()]
    steps=[r for r in st if r["type"]=="step"]
    res=[r for r in st if r["type"]=="result"]
    if not res: continue
    r=res[0]
    acts=collections.Counter((s.get("action_raw") or {}).get("action") for s in steps)
    g=[(s.get("action_raw") or {}).get("gate_score") for s in steps if (s.get("action_raw") or {}).get("gate_score") is not None]
    conf=[s.get("confidence") for s in steps if s.get("confidence") is not None]
    rows.append(dict(task=r.get("task_name"),succ=bool(r.get("success")),tmo=bool(r.get("timeout")),
        decl=bool(r.get("declared_done")),n=r.get("n_steps") or 0,acts=dict(acts),setup=r.get("setup_error"),
        g=g,conf=conf))
n=len(rows); s=sum(r["succ"] for r in rows)
# wilson
import math
if n:
    p=s/n; z=1.96
    den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    lo,hi=c-h,c+h
else: lo=hi=0
print("TASKS_FINISHED=%d SUCCESS=%d SR=%.1f%% WILSON=%.1f-%.1f"%(n,s,100*s/max(n,1),100*lo,100*hi))
b=collections.Counter()
for r in rows:
    if r["succ"]: b["success"]+=1
    elif r["setup"]: b["setup_error"]+=1
    elif r["tmo"]: b["timeout_non_terminating"]+=1
    elif r["decl"]: b["premature_status"]+=1
    else: b["other"]+=1
print("BUCKETS", dict(b))
pre=[r for r in rows if r["decl"] and not r["succ"]]
tmo=[r for r in rows if r["tmo"]]
def med(v): 
    v=sorted(v); return v[len(v)//2] if v else None
print("premature median_steps=%s"%med([r["n"] for r in pre]))
print("timeout   median_steps=%s"%med([r["n"] for r in tmo]))
# confidence/gate medians
import statistics
allconf=[x for r in rows for x in r["conf"]]
allgate=[x for r in rows for x in r["g"]]
print("conf p50=%.3f  gate p50=%.3f  gate==0 frac=%.1f%%"%(statistics.median(allconf),statistics.median(allgate),
      100*sum(1 for x in allgate if x==0)/len(allgate)))
# gate drift
def drift(v):
    d=[]
    for r in v:
        if len(r["g"])<3: continue
        d.append(statistics.mean(r["g"][-3:])-statistics.mean(r["g"][:3]))
    return (len(d), statistics.mean(d) if d else None, 100*sum(1 for x in d if x<0)/max(len(d),1))
print("drift success: n=%d mean=%s decaying=%.0f%%"%drift([r for r in rows if r["succ"]]))
print("drift fail   : n=%d mean=%s decaying=%.0f%%"%drift([r for r in rows if not r["succ"]]))
# type_text split
tt=[r for r in rows if r["acts"].get("type_text",0)>0]; no=[r for r in rows if r["acts"].get("type_text",0)==0]
print("type_text emitted: %d/%d success=%d"%(len(tt),n,sum(r["succ"] for r in tt)))
print("no type_text     : %d/%d success=%d"%(len(no),n,sum(r["succ"] for r in no)))
alls=collections.Counter()
for r in rows: alls.update(r["acts"])
print("ACTION_MIX", dict(alls))
print("SUCCESSES", [r["task"] for r in rows if r["succ"]])
