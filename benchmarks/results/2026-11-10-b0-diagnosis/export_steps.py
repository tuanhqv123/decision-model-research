import json,glob,ast,os
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
out=[]
for f in sorted(glob.glob(root+"/episodes/*.jsonl")):
    L=[json.loads(l) for l in open(f)]
    r=[l for l in L if l.get("type")=="result"]
    if not r: continue
    r=r[-1]; h=[l for l in L if l.get("type")=="episode"][0]
    S=[l for l in L if l.get("type")=="step"]
    imgs=[]
    for s in S:
        try: imgs.append(load(s["screenshot"]) if s.get("screenshot") else None)
        except Exception: imgs.append(None)
    steps=[]
    for i,s in enumerate(S):
        raw=lit(s.get("action_raw")); exe=lit(s.get("action_executed"))
        nxt=None
        if i+1<len(S) and imgs[i] is not None and imgs[i+1] is not None:
            nxt=float((np.abs(imgs[i]-imgs[i+1])>24).mean())
        steps.append({"i":i,"action":raw.get("action"),"exec":exe.get("action_type"),"point":raw.get("point"),
                      "conf":raw.get("confidence"),"box_conf":raw.get("box_confidence"),"gate":raw.get("gate_score"),
                      "probs":raw.get("probabilities"),"app":raw.get("app_name"),"text":raw.get("text"),
                      "ui_changed":s.get("ui_changed"),"next_diff":nxt,"reason":s.get("reason"),"done":bool(s.get("done")),
                      "typed_landed":s.get("typed_landed"),"n_el":s.get("a11y_elements_n")})
    out.append({"task":r["task_name"],"goal":r.get("goal"),"score":r.get("score"),"task_score":r.get("task_score"),
                "declared":r.get("declared_done"),"setup_error":r.get("setup_error"),"unscored":r.get("unscored"),
                "budget":h.get("step_budget"),"steps":steps})
json.dump(out,open(os.path.expanduser("~/aw/b0_export.json"),"w"))
print("episodes",len(out),"steps",sum(len(e["steps"]) for e in out))
