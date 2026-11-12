"""Is the harness's "→ no change" label stale on AndroidWorld? (2026-11-12) The label compares the element list read right
after the action with the one before. The NEXT step reads the list again, later. If the history said "no change" but the
next step's list differs from this step's (system UI and keyboard ignored), the immediate read missed a real change.
Pixels (aw_extract.pixel_change between the two screenshots) are the second witness.
    python stale_effect.py TRAIN_CODE_DIR RUN_DIR [RUN_DIR ...]"""
import collections as C, glob, json, os, sys
sys.path.insert(0, sys.argv[1]); import aw_extract as X
def key(els): return C.Counter((e[0], e[1], tuple(e[2]), e[3]) for e in (els or []) if not (e[2][3] <= 130 and not e[3]))
c = C.Counter()
for run in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(run, "episodes", "*.jsonl"))):
        if ".abandoned." in f: continue
        st = sorted([json.loads(l) for l in open(f) if '"type": "step"' in l], key=lambda s: s["i"])
        for k in range(len(st) - 1):
            a, b = st[k], st[k + 1]; ex = a.get("action_executed") or {}
            if ex.get("action_type") not in ("click", "scroll", "input_text") or not (a.get("history_line") or "").endswith("→ no change"): continue
            t = ex["action_type"]; c[(t, "n")] += 1
            later_differs = key(a.get("elements_compact")) != key(b.get("elements_compact"))
            pa, pb = (os.path.join(run, "screens", os.path.basename(x.get("screenshot") or "")) for x in (a, b))
            px = X.pixel_change(X.grey_small(open(pa, "rb").read()), X.grey_small(open(pb, "rb").read())) if os.path.isfile(pa) and os.path.isfile(pb) else None
            c[(t, "next step's list differs")] += later_differs
            c[(t, "pixels >=1%")] += px is not None and px >= .01
            c[(t, "both: list differs AND pixels >=1%")] += later_differs and px is not None and px >= .01
for t in ("click", "scroll", "input_text"):
    n = c[(t, "n")]
    if n: print("%-10s 'no change' n=%4d | %s" % (t, n, " | ".join("%s %d (%.0f%%)" % (k, c[(t, k)], 100 * c[(t, k)] / n) for k in ("next step's list differs", "pixels >=1%", "both: list differs AND pixels >=1%"))))
