"""Training-data numbers of the v3 post-mortem, recomputed from the v3 training file (2026-11-11).
Run on the training host:  python train_facts.py CODE_DIR/benchmarks/train DATA/compact_awlike_full_v3.jsonl"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]
out = C.OrderedDict(); out["rows (AndroidControl, AW-like full flow)"] = len(rows)
# 1. which app each row's screen is in
apps = C.Counter()
for r in rows:
    a = A.current_app([A.canonical_element(e) for e in r.get("elements") or []])
    apps[a.split(" (")[0] if "(" in a and a.split(" (")[0] in A.AW_APP_BY_PACKAGE.values() else ("home screen" if a == "home screen" else "other app")] += 1
aw = sorted(set(A.AW_APP_BY_PACKAGE.values()))
out["rows per AndroidWorld app"] = {a: apps[a] for a in aw}
out["rows home screen / other (non-AW) app"] = (apps["home screen"], apps["other app"])
out["AndroidWorld apps with ZERO rows"] = "%d/%d: %s" % (sum(apps[a] == 0 for a in aw), len(aw), [a for a in aw if apps[a] == 0])
# 2. history after a step that changed nothing
after = [r for r in rows if (r.get("history_v3") or []) and r["history_v3"][-1].endswith("→ no change")]
out["rows right after a no-change step"] = "%d/%d = %.1f%%" % (len(after), len(rows), 100 * len(after) / len(rows))
out["  gold action on those rows"] = dict(C.Counter(r["action"]["action_type"] for r in after).most_common())
# 3. what follows a click on an editable element (consecutive rows of one episode)
byep = C.defaultdict(dict)
for r in rows: byep[r["ep"]][r["step"]] = r
nxt = C.Counter()
for ep, st in byep.items():
    for s, r in st.items():
        if r["action"]["action_type"] != "click" or (s + 1) not in st or not r.get("target"): continue
        canon = [A.canonical_element(e) for e in r.get("elements") or []]
        hit = [e for e in canon if e["bbox"] and A.centre_in_box(e["bbox"], r["target"]) or (e["bbox"] == r["target"])]
        tgt = r["target"]; ed = [e for e in canon if e["flags"]["editable"] and e["bbox"] and
              e["bbox"]["x_min"] <= (tgt["x_min"] + tgt["x_max"]) / 2 <= e["bbox"]["x_max"] and
              e["bbox"]["y_min"] <= (tgt["y_min"] + tgt["y_max"]) / 2 <= e["bbox"]["y_max"]]
        if ed: nxt[st[s + 1]["action"]["action_type"]] += 1
n = sum(nxt.values())
out["after a click on an editable field, next gold is type_text"] = "%d/%d = %.1f%%" % (nxt["type_text"], n, 100 * nxt["type_text"] / max(1, n))
# 4. where `status` sits in its episode
st_rows = [r for r in rows if r["action"]["action_type"] == "status"]
last = sum(r["step"] == max(byep[r["ep"]]) for r in st_rows)
out["status rows"] = len(st_rows)
out["status rows that are the LAST row of their episode"] = "%d/%d" % (last, len(st_rows))
out["episodes"] = len(byep)
out["non-final rows labelled status"] = sum(r["step"] != max(byep[r["ep"]]) for r in st_rows)
for k, v in out.items(): print("%-60s %s" % (k, v))
