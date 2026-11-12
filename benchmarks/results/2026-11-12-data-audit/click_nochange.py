"""Click rows labelled "→ no change" in the training data, and what the gold does next (2026-11-12).
Also: how often does an annotator repeat the SAME step instruction on consecutive rows (a retry)?
    python click_nochange.py CODE_DIR DATA.jsonl"""
import collections as C, json, random, sys
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
cases = []; same_instr = C.Counter()
for e, st in ep.items():
    for s, r in st.items():
        if s - 1 not in st: continue
        p = st[s - 1]
        same_instr["consecutive rows"] += 1
        if (p.get("local_goal") or "").strip().lower() == (r.get("local_goal") or "").strip().lower():
            same_instr["same step instruction as previous row"] += 1
            same_instr["  ... and previous action had no change" if p.get("effect") is False else "  ... and previous action changed the screen"] += 1
        if p["action"]["action_type"] == "click" and p.get("effect") is False: cases.append((e, s, p, r))
print("retries:", dict(same_instr))
g = C.Counter()
for e, s, p, r in cases:
    pe, re_ = tl(p), tl(r)
    same_el = pe is not None and re_ is not None and A._effect_key(pe) == A._effect_key(re_)
    same_instr_ = (p.get("local_goal") or "").strip().lower() == (r.get("local_goal") or "").strip().lower()
    g[(r["action"]["action_type"] + (" SAME element" if same_el else ""), "same instruction" if same_instr_ else "new instruction")] += 1
print("\nafter a click labelled no change (n=%d): gold, and was the step instruction the same (a retry)?" % len(cases))
for k, v in g.most_common(): print("  ", k, v)
random.seed(0)
print("\n25 random cases:")
for e, s, p, r in random.sample(cases, 25):
    pe, re_ = tl(p), tl(r)
    print("- ep %s s%d px %.3f | prev click %r (%s) [%s]" % (e, s, p.get("pixel_change") or 0, pe and pe["label"], pe and pe["class_short"], (p.get("local_goal") or "")[:70]))
    print("      gold %s %r [%s]" % (r["action"]["action_type"], re_ and re_["label"], (r.get("local_goal") or "")[:70]))
