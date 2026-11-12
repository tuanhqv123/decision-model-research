"""Are the inserted 'no change' actions physically true? Check them on REAL AndroidControl clicks (2026-11-12).
For clicks on an element of a given kind, how often did the element list (effect_changed) AND the pixels (<1%) both stay
the same? Kinds: the focused editable field; any editable when the keyboard is already up; a selected tab/item;
an element that is not clickable and has no clickable element containing it.
    python physical_checks.py CODE_DIR DATA.jsonl"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]; ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
def inside(b, x, y): return b and b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"]
c = C.defaultdict(C.Counter)
for e, st in ep.items():
    for s, r in st.items():
        t = r.get("target")
        if r["action"]["action_type"] != "click" or not t or s + 1 not in st: continue
        x, y = (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2
        els = [A.canonical_element(q) for q in r["elements"]]
        hit = [q for q in els if inside(q["bbox"], x, y)]
        if not hit: continue
        sm = min(hit, key=lambda q: (q["bbox"]["x_max"] - q["bbox"]["x_min"]) * (q["bbox"]["y_max"] - q["bbox"]["y_min"]))
        ime = any("inputmethod" in q["package_name"] for q in els)
        kinds = []
        if sm["flags"]["editable"] and sm["flags"]["focused"]: kinds.append("focused editable field")
        if sm["flags"]["editable"] and ime: kinds.append("editable, keyboard already up")
        if sm["flags"]["editable"] and not ime and not sm["flags"]["focused"]: kinds.append("editable, not focused, no keyboard")
        if any(q["flags"]["selected"] for q in hit): kinds.append("selected item (or inside one)")
        if not any(q["flags"]["clickable"] or q["flags"]["long_clickable"] or q["flags"]["checkable"] or q["flags"]["editable"] for q in hit): kinds.append("nothing clickable at the point")
        kinds.append("ALL clicks")
        same_el = not A.effect_changed(els, [A.canonical_element(q) for q in st[s + 1]["elements"]])
        same_px = (r.get("pixel_change") or 0) < 0.01
        for k in kinds:
            c[k]["n"] += 1; c[k]["elements same"] += same_el; c[k]["elements AND pixels same"] += same_el and same_px
            c[k]["next gold " + st[s + 1]["action"]["action_type"]] += 1
for k, v in c.items():
    n = v["n"]; print("%-40s n=%5d  elements same %5.1f%%  elements+pixels same %5.1f%%   next gold: %s" % (k, n, 100 * v["elements same"] / n, 100 * v["elements AND pixels same"] / n,
          ", ".join("%s %d" % (g[10:], m) for g, m in sorted(((g, m) for g, m in v.items() if g.startswith("next gold")), key=lambda z: -z[1])[:3])))
