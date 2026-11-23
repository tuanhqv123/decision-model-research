import json, collections
def said(p): return p["pred"] in ("status","terminate")
low={(str(r["ep"]),r["step"]):r["action"]["action_type"] for r in map(json.loads,open("actest/data/compact_actest_v4.jsonl"))}
def L(f): return {(str(p["episode_id"]),p["step"]):p for p in map(json.loads,open(f))}
def LP(f): return {(str(p["episode_id"]),p["step"]):p for p in map(json.loads,open(f))}
runs={"DMA-2B":("actest/eval/eval-v4_s0_ckpt_2482_actest_v4_low-VA-val.jsonl","actest/eval/perstep-v4_s0_ckpt_2482_actest_v4_low.jsonl"),
      "exec1":("gate_exec1/eval-exec1_actest_low-VA-val.jsonl","gate_exec1/perstep-exec1_actest_low.jsonl"),
      "exec2":("gate_exec2/eval-exec2_actest_low-VA-val.jsonl","gate_exec2/perstep-exec2_actest_low.jsonl"),
      "exec2 noterm":("gate_exec2/eval-exec2_actest_low_noterm-VA-val.jsonl","gate_exec2/perstep-exec2_actest_low_noterm.jsonl")}
out={}
for m,(e,ps) in runs.items():
    E=L(e); P=LP(ps); c=collections.Counter(); per=collections.defaultdict(lambda:[0,0])
    for k,g in low.items():
        p=P[k]; ok=bool(p["exact_match"])
        per[g][0]+=1; per[g][1]+=ok
        if ok: c["correct"]+=1
        elif said(E[k]): c["said done"]+=1
        elif not p["type_match"]: c["wrong action type"]+=1
        else: c["right type, wrong element/value"]+=1
    out[m]={"errors":dict(c),"per gold type (n, correct)":{g:v for g,v in sorted(per.items(),key=lambda x:-x[1][0])}}
# paired DMA vs exec2-noterm (action skill only)
D=LP(runs["DMA-2B"][1]); N=LP(runs["exec2 noterm"][1])
b=collections.Counter((bool(D[k]["exact_match"]),bool(N[k]["exact_match"])) for k in low)
out["paired DMA-2B vs exec2-noterm (dma_ok, e2_ok)"]={str(k):v for k,v in b.items()}
# diagnostic only: AC-Low without gold 'wait' steps (a harness that waits for a stable screen never shows them)
for m in ("DMA-2B","exec2"):
    P=LP(runs[m][1]); ks=[k for k,g in low.items() if g!="wait"]
    out["no-wait steps: "+m]="%d/%d = %.2f"%(sum(bool(P[k]["exact_match"]) for k in ks),len(ks),100*sum(bool(P[k]["exact_match"]) for k in ks)/len(ks))
print(json.dumps(out,indent=1))
