"""Train/serve skew on a field click (2026-11-12). On AndroidWorld `is_focused` is never true (0/18,470 elements in the
self-play states), while in training 584/629 screens after a field click show a focused editable field. Contract v3's
effect ("→ screen changed / no change") compares elements INCLUDING the focused flag. Question: if training saw the
screen the way AndroidWorld serves it (no focus), how often would a field click read "→ no change"?
    python focus_skew.py CODE_DIR DATA.jsonl"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]
ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
def canon(r, drop_focus, drop_ime):
    out = []
    for x in r.get("elements") or []:
        e = A.canonical_element(x)
        if drop_ime and "inputmethod" in e["package_name"]: continue
        if drop_focus: e = dict(e, flags=dict(e["flags"], focused=False))
        out.append(e)
    return out
def field_click(r):
    t = r.get("target")
    if r["action"]["action_type"] != "click" or not t: return False
    cx, cy = (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2
    return any(e["flags"]["editable"] and e["bbox"] and e["bbox"]["x_min"] <= cx <= e["bbox"]["x_max"] and e["bbox"]["y_min"] <= cy <= e["bbox"]["y_max"]
               for e in map(A.canonical_element, r.get("elements") or []))
c = C.Counter(); ime = C.Counter()
for e, st in ep.items():
    for s, r in st.items():
        if s + 1 not in st or not field_click(r): continue
        n = st[s + 1]; c["n"] += 1
        for name, f, i in (("as trained (focus kept, IME kept)", False, False), ("focus dropped", True, False), ("focus AND keyboard dropped (= AW view)", True, True)):
            c[name + ": no change"] += not A.effect_changed(canon(r, f, i), canon(n, f, i))
        ime["keyboard elements on the next screen"] += any("inputmethod" in A.canonical_element(x)["package_name"] for x in n.get("elements") or [])
print("field clicks with a next row:", c["n"])
for k, v in c.items():
    if k != "n": print("  %-55s %d (%.0f%%)" % (k, v, 100 * v / c["n"]))
print("  %-55s %d" % ("next screen contains keyboard (IME) elements", ime["keyboard elements on the next screen"]))
