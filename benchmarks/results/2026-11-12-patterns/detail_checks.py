"""Two details behind the top situations (2026-11-12). Training side: after a scroll that changed nothing, is the gold
scroll the same direction or another? Run: python detail_checks.py train CODE_DIR DATA.jsonl
AW side: no-change clicks — interactive element or not; no-change scrolls — next scroll same direction?
Run: python detail_checks.py aw V3_EPISODES [...]"""
import collections as C, glob, json, os, sys
c = C.Counter()
if sys.argv[1] == "train":
    sys.path.insert(0, sys.argv[2]); rows = [json.loads(l) for l in open(sys.argv[3])]; ep = C.defaultdict(dict)
    for r in rows: ep[r["ep"]][r["step"]] = r
    for e, st in ep.items():
        for s, r in st.items():
            h = (r.get("history_v3") or [""])[-1]
            if s - 1 in st and st[s - 1]["action"]["action_type"] == "scroll" and h.endswith("→ no change") and r["action"]["action_type"] == "scroll":
                c["same direction" if st[s - 1]["action"].get("direction") == r["action"].get("direction") else "other direction"] += 1
else:
    for d in sys.argv[2:]:
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            if ".abandoned." in f: continue
            st = sorted([json.loads(l) for l in open(f) if '"type": "step"' in l], key=lambda s: s["i"])
            for k, s in enumerate(st):
                a = s.get("action_executed") or {}; h = s.get("history_line") or ""
                if not h.endswith("→ no change"): continue
                if a.get("action_type") == "click":
                    c["no-change click on interactive=%s" % s.get("click_on_interactive")] += 1
                if a.get("action_type") == "scroll" and k + 1 < len(st):
                    b = st[k + 1].get("action_executed") or {}
                    if b.get("action_type") == "scroll": c["after no-change scroll: next scroll %s direction" % ("same" if b.get("direction") == a.get("direction") else "other")] += 1
for k, v in sorted(c.items()): print(k, v)
