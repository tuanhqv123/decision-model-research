import json, collections
from scipy.stats import binomtest
L=lambda p:[json.loads(l) for l in open(p)]
V4="benchmarks/results/2026-11-12-v4/actest/"; V6="benchmarks/results/2026-11-13-v6/"
def ac(R):
    n=len(R); t=sum(r["type_match"] for r in R)/n*100
    g=[r for r in R if r["type_match"] and r["gt"]["action"] in ("click","long_press")]
    ep=collections.defaultdict(list)
    for r in R: ep[r["episode_id"]].append(r["exact_match"])
    return dict(type=round(t,1),ground=round(100*sum(r["exact_match"] for r in g)/len(g),1),sr=round(100*sum(r["exact_match"] for r in R)/n,1),ep_all=f'{sum(all(v) for v in ep.values())}/{len(ep)}')
for name,a,b in [("AC-Low",V4+"perstep-v4_s0_ckpt_2482_actest_v4_low.jsonl",V6+"perstep-final_actest_low.jsonl"),("AC-High",V4+"perstep-v4_s0_ckpt_2482_actest_v4.jsonl",V6+"perstep-final_actest.jsonl")]:
    A,B=L(a),L(b); print(name,"v4",ac(A),"v6",ac(B))
    ka={(r["episode_id"],r["step"]):r["exact_match"] for r in A}; w=l=0
    for r in B:
        x=ka[(r["episode_id"],r["step"])]
        if r["exact_match"] and not x: w+=1
        if x and not r["exact_match"]: l+=1
    print("  paired v6 wins",w,"losses",l,"p=%.3g"%binomtest(w,w+l).pvalue)
for name,p in [("v4",V4+"eval-v4_s0_ckpt_2482_actest_v4-VA-val.jsonl"),("v6",V6+"eval-final_actest-VA-val.jsonl")]:
    R=L(p); print(name,"AC-High terminate predicted",sum(r["pred"]=="status" for r in R),"/",len(R),"| back predicted",sum(r["pred"]=="navigate_back" for r in R),"gold back",sum(r["gold"]["action_type"]=="navigate_back" for r in R))
for name,p in [("v4",V4+"eval-v4_s0_ckpt_2482_validation_v4-VA-val.jsonl"),("v6",V6+"eval-final_validation-VA-val.jsonl")]:
    R=L(p); ep=collections.defaultdict(list)
    for r in R: ep[r["episode_id"]].append(r["correct"])
    ns=[r for r in R if r["gold"]["action_type"]!="status"]; nb=[r for r in R if r["gold"]["action_type"]!="navigate_back"]
    st=[r for r in R if r["gold"]["action_type"]=="status"]
    print(name,"val step",sum(r["correct"] for r in R),"/",len(R),"ep_all",sum(all(v) for v in ep.values()),"/",len(ep),
      "false_status",sum(r["pred"]=="status" for r in ns),"/",len(ns),"back_misuse",sum(r["pred"]=="navigate_back" for r in nb),"/",len(nb),
      "true_status",sum(r["pred"]=="status" for r in st),"/",len(st))
