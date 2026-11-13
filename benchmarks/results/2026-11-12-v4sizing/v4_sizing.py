"""Sizing for the all-issues v4 (2026-11-12), from the v3 training file.
  1. chained multi-part goals (premature status): AndroidControl episodes grouped by the app they finish in; how many
     same-app pairs exist, and how long would a chained episode be (rows)?
  2. no element cap: how many rows have more than 40 candidates under the v3 rule without the cap, and how many more?
    python v4_sizing.py CODE_DIR DATA.jsonl"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]; ep = C.defaultdict(dict)
for r in rows:
    if r.get("source") != "aw_generator": ep[r["ep"]][r["step"]] = r
app = C.Counter(); by = C.defaultdict(list); first_home = 0
for e, st in ep.items():
    last = st[max(st)]; a = A.current_app([A.canonical_element(x) for x in last.get("elements") or []]); by[a].append(e); app[a] += 1
    first_home += A.current_app([A.canonical_element(x) for x in st[0].get("elements") or []]) == "home screen"
pairs = sum(n * (n - 1) for n in app.values())
print("1. episodes %d; episodes whose first screen is the home screen: %d; distinct finishing apps %d; same-app ordered pairs %d" % (len(ep), first_home, len(app), pairs))
print("   top apps:", app.most_common(8))
L = sorted(len(st) for st in ep.values()); print("   episode rows median %d -> a chained pair would be ~%d rows" % (L[len(L)//2], 2 * L[len(L)//2] + 1))
over = C.Counter(); extra = []
for r in rows:
    cands = [e for e in map(A.canonical_element, r.get("elements") or []) if e["package_name"] != A.SYSTEM_UI and "inputmethod" not in e["package_name"]]
    n = len(cands)
    over["rows"] += 1
    if n > 40: over["rows with >40 candidates (raw, system UI and keyboard removed)"] += 1; extra.append(n)
print("2.", dict(over), "| among them median %s, p90 %s, max %s" % (sorted(extra)[len(extra)//2] if extra else None, sorted(extra)[int(.9*len(extra))] if extra else None, max(extra) if extra else None))
