"""Training side: was the keyboard already up BEFORE the field click? (2026-11-12)
    python ime_before.py CODE_DIR DATA.jsonl"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]; ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
ime = lambda r: any("inputmethod" in A.canonical_element(x)["package_name"] for x in r.get("elements") or [])
c = C.Counter()
for e, st in ep.items():
    for s, r in st.items():
        t = r.get("target")
        if r["action"]["action_type"] != "click" or not t or s + 1 not in st: continue
        cx, cy = (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2
        if not any(q["flags"]["editable"] and q["bbox"] and q["bbox"]["x_min"] <= cx <= q["bbox"]["x_max"] and q["bbox"]["y_min"] <= cy <= q["bbox"]["y_max"] for q in map(A.canonical_element, r.get("elements") or [])): continue
        n = st[s + 1]; ch = A.effect_changed([A.canonical_element(x) for x in r["elements"]], [A.canonical_element(x) for x in n["elements"]])
        c[("keyboard before", ime(r), "after", ime(n), "changed" if ch else "no change", "next gold", n["action"]["action_type"] if n["action"]["action_type"] in ("type_text", "click") else "other")] += 1
for k, v in sorted(c.items(), key=lambda kv: -kv[1]): print(k, v)
