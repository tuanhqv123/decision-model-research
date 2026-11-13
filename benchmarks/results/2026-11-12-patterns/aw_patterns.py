"""The same three situations as train_patterns.py, on AndroidWorld with the same model (v3 ckpt_1824). (2026-11-12)
  S1 after a click on an editable field (self-play seed 1001: state_in keeps the full element list incl. focus)
  S2 after a no-change step: is the next decision the same action on the same element? (v3 seed-30 runs, no masking)
  S3 premature `status` (v3 seed-30 runs): what came just before it, at which step, in which budget?
    python aw_patterns.py SELFPLAY_EPISODES V3_EPISODES [V3_EPISODES ...]"""
import collections as C, glob, json, os, statistics as S, sys
def eps(d):
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        if ".abandoned." in f: continue
        L = [json.loads(l) for l in open(f)]
        st = sorted([x for x in L if x.get("type") == "step"], key=lambda s: s["i"]); res = [x for x in L if x.get("type") == "result"]
        if st and res: yield st, res[-1], next((x for x in L if x.get("type") in ("episode", "header", "start")), {})
def eff(line): line = line or ""; return "no change" if line.endswith("→ no change") else "changed" if line.endswith("→ screen changed") else "none"
def flag(e, f): return bool(e.get("is_" + f) or (f == "clickable" and e.get("clickable")))
def editable_hit(s):
    t = s.get("click_target") or {}
    ex = s.get("action_executed") or {}
    if ex.get("action_type") != "click": return False
    return "e" in str(t.get("flags") or "") or bool(t.get("editable") or t.get("is_editable"))
# S1
nxt = C.Counter(); ef = C.Counter(); foc = C.Counter(); n_click_t = 0
for st, r, _ in eps(sys.argv[1]):
    for k, s in enumerate(st[:-1]):
        if not editable_hit(s): continue
        n = st[k + 1]; ex = n.get("action_executed") or {}
        nxt[ex.get("action_type") if ex else ("status" if n.get("done") else "none")] += 1
        ef[eff(s.get("history_line"))] += 1
        els = (n.get("state_in") or {}).get("elements") or []
        foc[any(flag(e, "focused") and flag(e, "editable") for e in els)] += 1
print("S1 AW after a click on an editable field: n=%d" % sum(nxt.values()))
print("  next action:", dict(nxt.most_common())); print("  history line of that click says:", dict(ef))
print("  next screen has a focused editable element:", dict(foc))
print("  (click_target example keys:", next((list((s.get('click_target') or {}).keys()) for st, _, _ in eps(sys.argv[1]) for s in st if s.get('click_target')), None), ")")
# S2 / S3
rep = C.Counter(); pre = C.Counter(); pst = []; pb = C.Counter()
for d in sys.argv[2:]:
    for st, r, h in eps(d):
        for k in range(1, len(st)):
            p, s = st[k - 1], st[k]
            if eff(p.get("history_line")) != "no change": continue
            a, b = p.get("action_executed") or {}, s.get("action_executed") or {}
            rep["n"] += 1
            same = a and b and a.get("action_type") == b.get("action_type") and (a.get("action_type") not in ("click", "long_press", "input_text") or (s.get("click_target") or {}) == (p.get("click_target") or {}))
            rep["exact repeat"] += bool(same)
        last = st[-1]
        if last.get("done") and not r.get("success") and not (r.get("setup_error") or r.get("unscored")):
            pre["n"] += 1; pst.append(last["i"])
            if len(st) > 1:
                q = st[-2]; pre["prev " + str((q.get("action_executed") or {}).get("action_type"))] += 1; pre["prev effect " + eff(q.get("history_line"))] += 1
            b = r.get("step_budget") or h.get("step_budget") or 0
            pb["budget <=10" if b <= 10 else "11-20" if b <= 20 else ">20"] += 1
print("\nS2 AW after a no-change step: n=%d, exact repeat %d (%.0f%%)" % (rep["n"], rep["exact repeat"], 100 * rep["exact repeat"] / max(rep["n"], 1)))
print("\nS3 AW premature status: n=%d, at step median %s (p75 %s)" % (pre["n"], S.median(pst) if pst else None, sorted(pst)[int(.75*len(pst))] if pst else None))
print("  ", {k: v for k, v in pre.items() if k != "n"}); print("   by budget:", dict(pb))
