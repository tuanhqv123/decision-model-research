"""Situation counts in the v3 training data (2026-11-12). A situation = (previous action, its effect, was it a click on an
editable field, step bucket) — features both the training rows and the AndroidWorld traces carry. Output: JSON counts,
plus the gold action distribution per situation (what the data teaches there).
    python situations_train.py CODE_DIR DATA.jsonl > situations_train.json"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]; ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
def bucket(i): return "0" if i == 0 else "1-5" if i <= 5 else "6-10" if i <= 10 else "11-15" if i <= 15 else ">15"
def eff(h): return "no change" if h.endswith("→ no change") else "changed" if h.endswith("→ screen changed") else "-"
def editable_click(r):
    t = r.get("target")
    if r["action"]["action_type"] != "click" or not t: return False
    cx, cy = (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2
    return any(q["flags"]["editable"] and q["bbox"] and q["bbox"]["x_min"] <= cx <= q["bbox"]["x_max"] and q["bbox"]["y_min"] <= cy <= q["bbox"]["y_max"]
               for q in map(A.canonical_element, r.get("elements") or []))
NAME = {"type_text": "input_text"}
out = C.defaultdict(lambda: {"n": 0, "gold": C.Counter()})
for e, st in ep.items():
    for s, r in st.items():
        if s == 0 or s - 1 not in st: key = ("-", "-", False, bucket(s))
        else:
            p = st[s - 1]; h = (r.get("history_v3") or [""])[-1]
            pa = p["action"]["action_type"]; key = (NAME.get(pa, pa), eff(h) if pa != "wait" else "-", editable_click(p), bucket(s))
        out["|".join(map(str, key))]["n"] += 1; g = r["action"]["action_type"]; out["|".join(map(str, key))]["gold"][NAME.get(g, g)] += 1
print(json.dumps({k: {"n": v["n"], "gold": dict(v["gold"])} for k, v in out.items()}))
