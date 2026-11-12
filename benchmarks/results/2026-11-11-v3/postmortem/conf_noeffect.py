"""AndroidWorld v3: does the model's confidence predict that a click will change nothing? (AUROC; 0.5 = no signal)
Positive = the click changed the screen (history line, the model's own definition). First click per screen only
(repeats are excluded, so one stuck screen does not dominate).
    python conf_noeffect.py ANDROIDWORLD_DIR"""
import glob, json, sys
def auroc(pos, neg):
    s = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg]); r = 0; i = 0; rk = {}
    ranks = [0] * len(s)
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1][0] == s[i][0]: j += 1
        for k in range(i, j + 1): ranks[k] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(ranks[k] for k in range(len(s)) if s[k][1]); n1, n0 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n0)
pos = {"element": [], "action": []}; neg = {"element": [], "action": []}
for f in glob.glob(sys.argv[1] + "/aw116_v3_s0*/episodes/*.jsonl"):
    if ".abandoned." in f: continue
    seen = set()
    for l in open(f):
        s = json.loads(l)
        if s.get("type") != "step" or (s.get("action_executed") or {}).get("action_type") not in ("click", "long_press"): continue
        if s.get("screen_hash") in seen: continue
        seen.add(s.get("screen_hash"))
        hl = s.get("history_line") or ""; el = (s.get("action_raw") or {}).get("element") or {}
        if not hl or el.get("confidence") is None: continue
        d = pos if not hl.endswith("→ no change") else neg
        d["element"].append(el["confidence"]); d["action"].append(s.get("confidence") or 0)
for k in ("element", "action"):
    print("AUROC(%s confidence -> click changed the screen) = %.3f  (changed %d, no change %d)" % (k, auroc(pos[k], neg[k]), len(pos[k]), len(neg[k])))
