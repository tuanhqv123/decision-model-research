"""Compact AndroidWorld trace printer for error analysis: goal, every step (model action, chosen element, effect, conf), result.
Usage: python3 show_trace.py RUN_DIR[,RUN_DIR] [--fail-only] [--start N --n K]"""
import glob, json, sys
dirs = sys.argv[1].split(","); fail_only = "--fail-only" in sys.argv
start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 999
eps = []
for d in dirs:
    for f in sorted(glob.glob(d + "/episodes/*.jsonl")):
        if ".abandoned." in f: continue
        rows = [json.loads(l) for l in open(f)]
        res = [r for r in rows if r.get("type") == "result"]
        if not res: continue
        res = res[-1]
        if fail_only and (res.get("score") or 0) >= 1: continue
        eps.append((rows[0], [r for r in rows if r.get("type") == "step"], res))
eps.sort(key=lambda e: e[0]["task_name"])
for k, (h, st, res) in enumerate(eps[start:start + n], start):
    print(f"#{k} {h['task_name']}  score={res.get('score')} steps={len(st)} budget={h.get('step_budget')}")
    print(f"   GOAL: {' '.join(str(res.get('goal')).split())[:260]}")
    prev=None; rep=0
    for s in st:
        a = s.get("action_raw") or {}
        el = (a.get("element") or {}).get("label") or ""
        extra = a.get("text") or a.get("app_name") or a.get("direction") or ""
        line = s.get("history_line") or ""
        eff = "NC" if "no change" in line else ("ok" if "changed" in line else "")
        key=(a.get('action'), el, str(extra), eff)
        if key==prev:
            rep+=1; continue
        if rep: print(f"      ... same x{rep} more")
        prev, rep = key, 0
        print(f"   {s.get('i'):>2} {a.get('action'):<13} {el[:38]!r:<40} {str(extra)[:30]!r:<20} c={a.get('confidence') or 0:.2f} {eff} n_el={s.get('a11y_elements_n')}")
    if rep: print(f"      ... same x{rep} more")
