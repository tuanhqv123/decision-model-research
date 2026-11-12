"""What would the AndroidWorld history say if the effect were computed the way TRAINING computes it — this step's
element list vs the NEXT step's list (system-UI status bar ignored) — instead of an immediate re-read? (2026-11-12)
Cross-tab immediate label x deferred label x pixels, per action. Also, for 'changed' scrolls with <0.5% pixel change,
what differs in the lists.
    python deferred_effect.py TRAIN_CODE_DIR RUN_DIR [RUN_DIR ...]"""
import collections as C, glob, json, os, sys
sys.path.insert(0, sys.argv[1]); import aw_extract as X
def key(els): return C.Counter((e[0], e[1], tuple(e[2]), e[3]) for e in (els or []) if not (e[2][3] <= 130 and not e[3]))
c = C.Counter(); diffs = []
for run in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(run, "episodes", "*.jsonl"))):
        if ".abandoned." in f: continue
        st = sorted([json.loads(l) for l in open(f) if '"type": "step"' in l], key=lambda s: s["i"])
        for k in range(len(st) - 1):
            a, b = st[k], st[k + 1]; ex = a.get("action_executed") or {}; h = a.get("history_line") or ""
            if not ex or not (h.endswith("→ no change") or h.endswith("→ screen changed")): continue
            t = {"input_text": "type_text"}.get(ex["action_type"], ex["action_type"])
            imm = "changed" if h.endswith("changed") else "no change"
            ka, kb = key(a.get("elements_compact")), key(b.get("elements_compact")); dfr = "changed" if ka != kb else "no change"
            pa, pb = (os.path.join(run, "screens", os.path.basename(x.get("screenshot") or "")) for x in (a, b))
            px = X.pixel_change(X.grey_small(open(pa, "rb").read()), X.grey_small(open(pb, "rb").read())) if os.path.isfile(pa) and os.path.isfile(pb) else -1
            c[(t, imm, dfr, "px>=1%" if px >= .01 else "px<1%")] += 1
            if t == "scroll" and imm == "changed" and px < .005 and len(diffs) < 4:
                diffs.append((os.path.basename(f)[:50], h, sorted((ka - kb).elements())[:3], sorted((kb - ka).elements())[:3]))
print("%-10s %-10s %-10s %-7s %s" % ("action", "immediate", "deferred", "pixels", "n"))
for k in sorted(c): print("%-10s %-10s %-10s %-7s %d" % (*k, c[k]))
print("\nscroll 'changed' with <0.5% pixel change — what differs (before-only | after-only):")
for d in diffs: print(" ", d)
