import json,sys,collections as C,statistics as st
S=sys.argv[1]
eps=[e for e in json.load(open(S+"/b0_export.json")) if not e["setup_error"] and not e["unscored"]]
meta={t["task_name"]:t for t in json.load(open(S+"/dl/aw/android_world/task_metadata.json"))}
def auroc(pos,neg):
    if not pos or not neg: return None
    w=sum((p>n)+0.5*(p==n) for p in pos for n in neg); return w/(len(pos)*len(neg))
print("scored episodes",len(eps),"| success",sum((e["score"] or 0)>0.5 for e in eps))
# ---- (a) does confidence know when the action does nothing?
rows=[s for e in eps for s in e["steps"] if not s["done"] and s["next_diff"] is not None and s["conf"] is not None]
eff=[s for s in rows if s["next_diff"]>=0.002]; noeff=[s for s in rows if s["next_diff"]<0.002]
print("\n(a) executed steps %d: effect %d, no visible effect %d (%.1f%%)" % (len(rows),len(eff),len(noeff),100*len(noeff)/len(rows)))
for key in ("conf","box_conf","gate"):
    p=[float(s[key]) for s in eff if s[key] is not None]; n=[float(s[key]) for s in noeff if s[key] is not None]
    if p and n: print("    %-8s median effect %.3f | no-effect %.3f | AUROC(score -> action has effect) = %.3f" % (key,st.median(p),st.median(n),auroc(p,n)))
cl=[s for s in rows if s["action"]=="click"]
p=[float(s["box_conf"]) for s in cl if s["next_diff"]>=0.002 and s["box_conf"] is not None]; n=[float(s["box_conf"]) for s in cl if s["next_diff"]<0.002 and s["box_conf"] is not None]
print("    clicks only: box_conf AUROC = %.3f (n=%d/%d)" % (auroc(p,n),len(p),len(n)))
hi=[s for s in noeff if float(s["conf"])>=0.9]; print("    no-effect steps taken at action confidence >= 0.90: %d/%d = %.0f%%" % (len(hi),len(noeff),100*len(hi)/len(noeff)))
# ---- (b) by benchmark-author tag and (e) by optimal steps
print("\n(b) by AndroidWorld tag (authors' labels): ran / success")
tg=C.defaultdict(lambda:[0,0])
for e in eps:
    for t in (meta[e["task"]]["tags"] or ["(none)"]):
        tg[t or "(none)"][0]+=1; tg[t or "(none)"][1]+=(e["score"] or 0)>0.5
for t,(n,k) in sorted(tg.items(),key=lambda kv:-kv[1][0]): print("    %-26s %2d / %d" % (t,n,k))
print("\n(e) by optimal_steps: ran / success / mean steps used / budget-hit")
hz=C.defaultdict(lambda:[0,0,0,0])
for e in eps:
    k=int(meta[e["task"]]["optimal_steps"]); b="1-3" if k<=3 else "4-6" if k<=6 else "7-10" if k<=10 else "11+"
    hz[b][0]+=1; hz[b][1]+=(e["score"] or 0)>0.5; hz[b][2]+=len(e["steps"]); hz[b][3]+=(not e["declared"])
for b in ("1-3","4-6","7-10","11+"):
    n,k,s,t=hz[b]
    if n: print("    %-5s %2d / %d / %.1f / %d" % (b,n,k,s/n,t))
# ---- (c) first-action failures
na=[e["task"] for e in eps if e["steps"] and e["steps"][0]["reason"]=="app_name_missing"]
print("\n(c) episodes whose FIRST decision was open_app with an empty app name: %d/%d ->" % (len(na),len(eps)),na)
oa=[(e["task"],e["steps"][0]["app"]) for e in eps if e["steps"] and e["steps"][0]["action"]=="open_app" and e["steps"][0]["reason"] is None]
print("    first-step open_app names emitted:",C.Counter(a for _,a in oa).most_common(14))
print("    first decision types:",C.Counter(e["steps"][0]["action"] for e in eps if e["steps"]).most_common())
# ---- typing
ty=[s for e in eps for s in e["steps"] if s["exec"]=="input_text"]
print("\n(typing) input_text executed %d times; landed %s" % (len(ty),C.Counter(str(s["typed_landed"]) for s in ty)))
tt=[s for e in eps for s in e["steps"] if s["action"]=="type_text"]; print("    type_text decisions %d of %d decisions (%.1f%%); with empty text %d" % (len(tt),sum(len(e["steps"]) for e in eps),100*len(tt)/sum(len(e["steps"]) for e in eps),sum(1 for s in tt if not s["text"])))
