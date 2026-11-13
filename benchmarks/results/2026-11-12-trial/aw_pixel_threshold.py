"""Pixel change on AndroidWorld with the SAME function the training rows used (aw_extract.pixel_change: grey, 270x600,
top 30 rows cropped, |diff| > 24) between the screenshot before step i and before step i+1 (= after action i). (2026-11-12)
Split by what the history line says (element-list effect) and by action. Question: is there a pixel threshold that
(a) leaves AW's genuine no-change steps as "no change" and (b) catches AC's false "no change" (carousels, rating bars)?
    python aw_pixel_threshold.py TRAIN_CODE_DIR RUN_DIR [RUN_DIR ...]"""
import collections as C, glob, json, os, sys
sys.path.insert(0, sys.argv[1]); import aw_extract as X
vals = C.defaultdict(list)
for run in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(run, "episodes", "*.jsonl"))):
        if ".abandoned." in f: continue
        st = sorted([json.loads(l) for l in open(f) if '"type": "step"' in l], key=lambda s: s["i"])
        for k in range(len(st) - 1):
            a, b = st[k], st[k + 1]; ex = a.get("action_executed") or {}
            pa, pb = (os.path.join(run, "screens", os.path.basename(x.get("screenshot") or "")) for x in (a, b))
            if not ex or not os.path.isfile(pa) or not os.path.isfile(pb): continue
            h = a.get("history_line") or ""; eff = "no change" if h.endswith("→ no change") else "changed" if h.endswith("→ screen changed") else "-"
            p = X.pixel_change(X.grey_small(open(pa, "rb").read()), X.grey_small(open(pb, "rb").read()))
            t = {"input_text": "type_text"}.get(ex["action_type"], ex["action_type"])
            vals[(t if t in ("click", "scroll", "type_text", "wait") else "other", eff)].append(p)
def q(v, x): v = sorted(v); return v[min(len(v) - 1, int(x * len(v)))]
print("%-24s %5s %7s %7s %7s %7s   %s" % ("action, history says", "n", "p50", "p90", "p95", "max", "share above 0.5% / 1% / 2% / 5%"))
for k in sorted(vals):
    v = vals[k]
    print("%-24s %5d %6.2f%% %6.2f%% %6.2f%% %6.2f%%   %s" % (", ".join(k), len(v), 100*q(v,.5), 100*q(v,.9), 100*q(v,.95), 100*max(v),
          " / ".join("%.0f%%" % (100 * sum(x > t for x in v) / len(v)) for t in (.005, .01, .02, .05))))
