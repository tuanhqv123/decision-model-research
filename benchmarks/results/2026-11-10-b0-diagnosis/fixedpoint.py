import json,glob,ast,os,collections as C
import numpy as np
from PIL import Image
root=os.path.expanduser(os.environ.get("AW_RUN_DIR","~/awruns/aw116_official_b0"))
def lit(v):
    if isinstance(v,str):
        try: return ast.literal_eval(v)
        except Exception: return {}
    return v or {}
def load(p):
    a=np.asarray(Image.open(p.replace("/out",root)).convert("L").resize((270,600)),dtype=np.int16); return a[30:,:]
def same_screen(a,b): return float((np.abs(a-b)>24).mean())<0.002
def same_action(r1,r2):
    if r1.get("action")!=r2.get("action"): return False
    p1,p2=r1.get("point"),r2.get("point")
    if p1 and p2: return abs(p1[0]-p2[0])<=40 and abs(p1[1]-p2[1])<=40
    return (r1.get("app_name"),r1.get("text"),r1.get("direction"))==(r2.get("app_name"),r2.get("text"),r2.get("direction"))
tot=C.Counter(); eps=[]; runlens=[]
for f in sorted(glob.glob(root+"/episodes/*.jsonl")):
    L=[json.loads(l) for l in open(f)]
    r=[l for l in L if l.get("type")=="result"]
    if not r or r[-1].get("setup_error") or r[-1].get("unscored"): continue
    r=r[-1]; S=[l for l in L if l.get("type")=="step"]
    imgs=[load(s["screenshot"]) if s.get("screenshot") else None for s in S]
    raws=[lit(s.get("action_raw")) for s in S]
    absorbed=0; run=0
    for i in range(1,len(S)):
        if imgs[i] is None or imgs[i-1] is None: continue
        ss=same_screen(imgs[i-1],imgs[i]); sa=same_action(raws[i-1],raws[i])
        tot[("same_screen" if ss else "new_screen","same_action" if sa else "new_action")]+=1
        if ss and sa: run+=1
        else:
            if run: runlens.append(run)
            run=0
    if run: runlens.append(run)
    # did the episode END inside a fixed point (last >=3 decisions identical on an identical screen)?
    tail=0
    for i in range(len(S)-1,0,-1):
        if imgs[i] is not None and imgs[i-1] is not None and same_screen(imgs[i-1],imgs[i]) and same_action(raws[i-1],raws[i]): tail+=1
        else: break
    ok=(r.get("score") or 0)>0.5
    eps.append((r["task_name"],ok,bool(r.get("declared_done")),len(S),tail))
ss_sa=tot[("same_screen","same_action")]; ss_na=tot[("same_screen","new_action")]
ns_sa=tot[("new_screen","same_action")]; ns_na=tot[("new_screen","new_action")]
print("episodes",len(eps),"decision pairs",sum(tot.values()))
print("P(same action | SAME screen) = %d/%d = %.1f%%" % (ss_sa,ss_sa+ss_na,100*ss_sa/max(1,ss_sa+ss_na)))
print("P(same action | NEW  screen) = %d/%d = %.1f%%" % (ns_sa,ns_sa+ns_na,100*ns_sa/max(1,ns_sa+ns_na)))
fail=[e for e in eps if not e[1]]
dead=[e for e in fail if e[4]>=3]
print("failed episodes %d; ended inside a fixed point (>=3 identical decisions on an identical screen): %d (%.0f%%)" % (len(fail),len(dead),100*len(dead)/max(1,len(fail))))
print("  of failed-without-done %d, ended in fixed point %d" % (sum(1 for e in fail if not e[2]), sum(1 for e in dead if not e[2])))
if runlens:
    runlens.sort(); print("fixed-point run lengths: n=%d median %d p90 %d max %d; steps inside runs %d" % (len(runlens),runlens[len(runlens)//2],runlens[int(.9*(len(runlens)-1))],runlens[-1],sum(runlens)))
esc=[e for e in eps if e[1]]
print("successes",len(esc),[ (e[0],e[3]) for e in esc])
