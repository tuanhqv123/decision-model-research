"""Which exploration policy? Top-k-limited sampling at temperature T, measured offline on AC-High (ckpt_1824, the
saved readouts of the v3 eval). For each T: expected per-step accuracy of ONE sample (action type; click element;
both) and how often the sample differs from the argmax. Also the `status` question: how often is gold `status` when
status is the model's 2nd/3rd type (what sampling INTO status would find), vs how often sampling picks status when
gold is not status (premature terminations it would add).
    python sampling_policy.py CODE_DIR TEST_COMPACT EVAL_JSONL"""
import json, math, sys, collections as C
sys.path.insert(0, sys.argv[1]); import ac_common as A
test = {(str(r["ep"]), str(r["step"])): r for r in map(json.loads, open(sys.argv[2]))}
ACT = {"click": "click", "long_press": "long_press", "type_text": "write", "scroll": "scroll", "open_app": "open_app",
       "navigate_back": "navigate_back", "navigate_home": "navigate_home", "wait": "wait", "status": "terminate"}
K = 3
def topk(p, T, k=K, never=None, keep_if_top=None):
    o = sorted(p, key=lambda x: -p[x])[:k]
    if never is not None and o[0] != never: o = [x for x in o if x != never]
    if T == 0: return {o[0]: 1.0}
    w = {x: math.exp(math.log(max(p[x], 1e-12)) / T) for x in o}; z = sum(w.values())
    return {x: w[x] / z for x in w}
rows = []
keys = C.Counter()
for l in open(sys.argv[3]):
    e = json.loads(l); r = test.get((str(e["episode_id"]), str(e["step"])))
    if r is None: continue
    pa = e["readouts"]["action"]["probabilities"]; keys.update(pa.keys())
    g = r["action"]["action_type"]; ok = None
    if g in ("click", "long_press") and r.get("target") and "target_el" in e["readouts"]:
        kept, _ = A.select_elements(r["elements"], int(r["W"]), int(r["H"]), "cap40_reading_noime")
        ok = {str(i + 1) for i, k in enumerate(kept) if A.centre_in_box(k["bbox"], r["target"])}
    rows.append((ACT.get(g, g), pa, e["readouts"].get("target_el", {}).get("probabilities"), ok))
print("action option keys:", sorted(keys)); print("rows:", len(rows))
st = "terminate"
for never in (None, st):
    print("\n== policy: top-%d, %s" % (K, "status sampled like any type" if never is None else "never sample INTO status (only if argmax)"))
    print("  T     type_acc  type_dev  el_acc  el_dev  click_step_acc  P(status|gold!=status)  P(status|gold=status)")
    for T in (0, 0.3, 0.5, 0.7, 1.0):
        c = C.Counter()
        for g, pa, pe, ok in rows:
            q = topk(pa, T, never=never); top = max(pa, key=pa.get)
            c["n"] += 1; c["ta"] += q.get(g, 0); c["td"] += 1 - q.get(top, 0)
            if g != "terminate": c["ns"] += 1; c["s_wrong"] += q.get("terminate", 0)
            else: c["ys"] += 1; c["s_right"] += q.get("terminate", 0)
            if ok is not None and pe:
                qe = topk(pe, T); te = max(pe, key=pe.get)
                c["ne"] += 1; ea = sum(qe.get(x, 0) for x in ok); c["ea"] += ea; c["ed"] += 1 - qe.get(te, 0)
                c["cs"] += q.get(g, 0) * ea
        print("  %-4s  %6.1f%%  %6.1f%%  %5.1f%%  %5.1f%%  %8.1f%%        %6.2f%%               %6.1f%%" % (
            T, 100*c["ta"]/c["n"], 100*c["td"]/c["n"], 100*c["ea"]/c["ne"], 100*c["ed"]/c["ne"], 100*c["cs"]/c["ne"],
            100*c["s_wrong"]/max(c["ns"],1), 100*c["s_right"]/max(c["ys"],1)))
# status as 2nd/3rd type: is it then right?
c = C.Counter()
for g, pa, pe, ok in rows:
    o = sorted(pa, key=lambda x: -pa[x])
    if "terminate" in o[1:3]: c["st23"] += 1; c["st23_gold"] += g == "terminate"
    if o[0] == "terminate": c["st1"] += 1; c["st1_wrong"] += g != "terminate"; c["st1_wrong_gold_in_top3"] += (g != "terminate" and g in o[:3])
    if g == "terminate": c["gst"] += 1
print("\ngold status rows: %d" % c["gst"]); print("status ranked 2nd/3rd: %d rows, gold=status in %d" % (c["st23"], c["st23_gold"]))
print("status ranked 1st: %d rows, gold!=status in %d (premature), of which gold type in top-3: %d" % (c["st1"], c["st1_wrong"], c["st1_wrong_gold_in_top3"]))
