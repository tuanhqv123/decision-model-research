"""Why would a model learn to ignore "→ no change"? Because in the training data it barely predicts the gold. (2026-11-12)
For each previous action type, compare the gold right after it when the history says "screen changed" vs "no change":
  P(gold repeats the previous action: same type and same element / same scroll direction)
  P(gold = status), P(gold = wait)
and the mutual information I(effect word ; gold action type) in bits, with the count of rows that carry it.
    python effect_information.py CODE_DIR DATA.jsonl"""
import collections as C, json, math, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]; ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
def tl(r):
    t = r.get("target")
    if not t: return None
    cx, cy = (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2; best = None
    for e in map(A.canonical_element, r.get("elements") or []):
        b = e["bbox"]
        if b and b["x_min"] <= cx <= b["x_max"] and b["y_min"] <= cy <= b["y_max"] and (best is None or (b["x_max"]-b["x_min"])*(b["y_max"]-b["y_min"]) < (best["bbox"]["x_max"]-best["bbox"]["x_min"])*(best["bbox"]["y_max"]-best["bbox"]["y_min"])): best = e
    return best
tab = C.defaultdict(C.Counter); joint = C.Counter()
for e, st in ep.items():
    for s, r in st.items():
        if s - 1 not in st: continue
        h = (r.get("history_v3") or [""])[-1]
        if not (h.endswith("→ no change") or h.endswith("→ screen changed")): continue
        w = "no change" if h.endswith("→ no change") else "changed"
        p = st[s - 1]; pa, g = p["action"]["action_type"], r["action"]["action_type"]
        if pa == "wait": continue
        rep = g == pa and ((pa == "scroll" and p["action"].get("direction") == r["action"].get("direction")) or
                           (pa in ("click", "long_press", "type_text") and tl(p) is not None and tl(r) is not None and A._effect_key(tl(p)) == A._effect_key(tl(r))) or
                           pa in ("navigate_back", "navigate_home", "open_app"))
        k = (pa if pa in ("click", "scroll") else "other", w)
        tab[k]["n"] += 1; tab[k]["repeat"] += rep; tab[k]["status"] += g == "status"; tab[k]["wait"] += g == "wait"
        joint[(w, g)] += 1
print("%-26s %6s %9s %9s %9s" % ("previous action, effect", "n", "repeat", "status", "wait"))
for k in sorted(tab):
    t = tab[k]; print("%-26s %6d %8.1f%% %8.1f%% %8.1f%%" % (", ".join(k), t["n"], 100*t["repeat"]/t["n"], 100*t["status"]/t["n"], 100*t["wait"]/t["n"]))
N = sum(joint.values()); pw = C.Counter(); pg = C.Counter()
for (w, g), v in joint.items(): pw[w] += v; pg[g] += v
mi = sum(v / N * math.log2((v / N) / ((pw[w] / N) * (pg[g] / N))) for (w, g), v in joint.items())
hg = -sum(v / N * math.log2(v / N) for v in pg.values())
print("\nI(effect word ; gold action type) = %.4f bits of H(gold) = %.3f bits (%.2f%%), over %d rows; 'no change' rows: %d (%.1f%%)" % (mi, hg, 100 * mi / hg, N, pw["no change"], 100 * pw["no change"] / N))
