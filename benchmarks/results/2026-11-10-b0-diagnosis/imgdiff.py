import json,glob,ast,os,collections as C
import numpy as np
from PIL import Image
root=os.path.expanduser(os.environ.get("AW_RUN_DIR","~/awruns/aw116_official_b0"))
def ex(s):
    a=s.get("action_executed")
    if isinstance(a,str):
        try: a=ast.literal_eval(a)
        except Exception: a=None
    return a or {}
def load(p):
    p=p.replace("/out",root)
    im=Image.open(p).convert("L").resize((270,600))
    a=np.asarray(im,dtype=np.int16)
    return a[30:,:]            # drop the status bar (top 5%)
def frac(a,b): return float((np.abs(a-b)>24).mean())
rows=[]; miss=0
for f in sorted(glob.glob(root+"/episodes/*.jsonl")):
    L=[json.loads(l) for l in open(f)]
    r=[l for l in L if l.get("type")=="result"]
    if not r or r[-1].get("setup_error") or r[-1].get("unscored"): continue
    S=[l for l in L if l.get("type")=="step"]
    for i in range(len(S)-1):
        s,n=S[i],S[i+1]
        if s.get("done") or not s.get("screenshot") or not n.get("screenshot"): miss+=1; continue
        try: d=frac(load(s["screenshot"]),load(n["screenshot"]))
        except Exception as e: miss+=1; continue
        rows.append((ex(s).get("action_type","?"), s.get("ui_changed") is False, d))
print("step pairs",len(rows),"unreadable",miss)
for flag in (True,False):
    ds=sorted(d for a,f,d in rows if f==flag)
    if not ds: continue
    q=lambda p: ds[int(p*(len(ds)-1))]
    print("flag no_ui_change=%s n=%d | model-visible screen change between step i and i+1: median %.4f p75 %.4f p90 %.4f | identical(<0.2%%): %d (%.0f%%) | large(>5%%): %d (%.0f%%)" % (flag,len(ds),q(.5),q(.75),q(.9),sum(d<0.002 for d in ds),100*sum(d<0.002 for d in ds)/len(ds),sum(d>0.05 for d in ds),100*sum(d>0.05 for d in ds)/len(ds)))
allsame=sum(1 for a,f,d in rows if d<0.002); print("ALL: model saw an (almost) identical screen on the next step in %d / %d = %.1f%% of steps" % (allsame,len(rows),100*allsame/len(rows)))
for a in ("click","scroll"):
    r=[d for x,f,d in rows if x==a]; print("  %s: %d steps, next screen identical %.0f%%" % (a,len(r),100*sum(d<0.002 for d in r)/max(1,len(r))))
