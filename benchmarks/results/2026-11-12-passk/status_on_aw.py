"""Sampling and `status` on AndroidWorld itself (self-play seed 1001, v3 ckpt_1824, readouts recorded per step).
Label per step = the task checker asked after the previous step (`probe_score`): 1 = the task was already done.
Question: if exploration samples the action TYPE from the top-3 at temperature T, how often does it pick `terminate`
when the task is NOT done (adds premature ends) and when it IS done (finds a correct end the argmax missed)?
    python status_on_aw.py EPISODE_DIR"""
import glob, json, math, os, sys, collections as C
def topk(p, T, k=3, never=None):
    o = sorted(p, key=lambda x: -p[x])[:k]
    if never is not None and o[0] != never: o = [x for x in o if x != never]
    if T == 0: return {o[0]: 1.0}
    w = {x: math.exp(math.log(max(p[x], 1e-12)) / T) for x in o}; z = sum(w.values())
    return {x: w[x] / z for x in w}
steps = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    if ".abandoned." in f: continue
    L = [json.loads(l) for l in open(f)]
    S = sorted([x for x in L if x.get("type") == "step"], key=lambda s: s["i"])
    for k, s in enumerate(S):
        ro = ((s.get("state_in") or {}).get("readouts") or {}).get("action", {}).get("probabilities")
        if not ro or k == 0 and False: continue
        done = (S[k - 1].get("probe_score") or 0) >= 1.0 if k > 0 else False
        steps.append((done, ro, ((s.get("state_in") or {}).get("readouts") or {}).get("target_el", {}).get("probabilities")))
n_done = sum(d for d, _, _ in steps); print("steps with readouts: %d | task already done: %d | not done: %d" % (len(steps), n_done, len(steps) - n_done))
rank = C.Counter()
for d, p, _ in steps:
    o = sorted(p, key=lambda x: -p[x]); r = o.index("terminate") + 1
    rank[(d, min(r, 4))] += 1
print("rank of terminate (4 = 4th or lower), done:", {r: rank[(True, r)] for r in (1, 2, 3, 4)}, "| not done:", {r: rank[(False, r)] for r in (1, 2, 3, 4)})
for never in (None, "terminate"):
    print("\n== %s" % ("terminate sampled like any type" if never is None else "never sample INTO terminate"))
    print("  T     P(terminate | not done)  P(terminate | done)  type_dev  el_dev")
    for T in (0, 0.3, 0.5, 0.7, 1.0):
        c = C.Counter()
        for d, p, pe in steps:
            q = topk(p, T, never=never); top = max(p, key=p.get)
            c["nd" if not d else "d"] += 1; c["tnd" if not d else "td"] += q.get("terminate", 0); c["dev"] += 1 - q.get(top, 0)
            if pe: qe = topk(pe, T); c["ne"] += 1; c["edev"] += 1 - qe.get(max(pe, key=pe.get), 0)
        print("  %-4s  %8.2f%%               %8.1f%%          %5.1f%%   %5.1f%%" % (T, 100*c["tnd"]/c["nd"], 100*c["td"]/max(c["d"],1), 100*c["dev"]/len(steps), 100*c["edev"]/max(c["ne"],1)))
