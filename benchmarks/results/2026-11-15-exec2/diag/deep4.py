import json, collections, re
def clean(s): return " ".join((s or "").split()).rstrip(" .")
def said(p): return p["pred"] in ("status","terminate")
def toks(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower())) - {"the","on","to","a","an","of","in","and","at","for","icon","button","option","click","tap","select"}
low=[]
for l in open("actest/data/compact_actest_v4.jsonl"):
    r=json.loads(l)
    els=r.get("elements") or ""
    low.append({"ep":str(r["ep"]),"step":r["step"],"lg":r["local_goal"],"ld":r["local_done"],"a":r["action"]["action_type"],
                "prog":("ProgressBar" in els) , "nel": els.count("class_name")})
low.sort(key=lambda r:(r["ep"],r["step"]))
by={(r["ep"],r["step"]):r for r in low}
L=lambda f:{(str(p["episode_id"]),p["step"]):p for p in map(json.loads,open(f))}
E2=L("gate_exec2/eval-exec2_actest_low-VA-val.jsonl"); DMA=L("actest/eval/eval-v4_s0_ckpt_2482_actest_v4_low-VA-val.jsonl")
N2=L("gate_exec2/eval-exec2_actest_low_noterm-VA-val.jsonl")
out={}
# (a) similarity of a new instruction to the previous one
bins=collections.defaultdict(lambda:[0,0])
for r in low:
    pr=by.get((r["ep"],r["step"]-1))
    if not pr or clean(pr["lg"])==clean(r["lg"]): continue
    a,b=toks(pr["lg"]),toks(r["lg"]); j=len(a&b)/max(1,len(a|b))
    k="jaccard 0" if j==0 else "<0.25" if j<.25 else "0.25-0.5" if j<.5 else ">=0.5"
    bins[k][0]+=1; bins[k][1]+=said(E2[(r["ep"],r["step"])])
out["(a) new instruction: false done by word overlap with the previous instruction [n, exec2 said done]"]=dict(bins)
# (c) ProgressBar in the element list, on continuation rows whose gold is wait vs everything else
c=collections.defaultdict(lambda:[0,0])
for r in low:
    pr=by.get((r["ep"],r["step"]-1))
    cont=bool(pr and clean(pr["lg"])==clean(r["lg"]) and pr["ld"] is False)
    k="%s | gold %s | progressbar %s"%("cont" if cont else "first", "wait" if r["a"]=="wait" else "other", r["prog"])
    c[k][0]+=1; c[k][1]+=said(E2[(r["ep"],r["step"])])
out["(c) [n, exec2 said done]"]=dict(sorted(c.items()))
# (e) DMA-2B and exec2-noterm on exec2's false-done steps
fd=[k for k in E2 if said(E2[k])]
out["(e) on exec2's 449 false-done steps: DMA-2B correct / exec2 noterm correct"]=[sum(bool(DMA[k]["correct"]) for k in fd), sum(bool(N2[k]["correct"]) for k in fd)]
# diagnostic only: AC-Low if 'done' instead of a gold wait counted as correct
w=sum(1 for k in fd if by[k]["a"]=="wait")
out["(d) false done whose gold is wait"]=w
print(json.dumps(out,indent=1))
