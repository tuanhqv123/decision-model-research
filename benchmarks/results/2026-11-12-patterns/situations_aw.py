"""Situation counts on AndroidWorld, v3 ckpt_1824, seed-30 runs (no masking) (2026-11-12). Same key as situations_train.py.
Per step also: did it go wrong in a way we can see without a gold label — (a) the action changed nothing (wait excluded),
(b) `status` in an episode the checker scored as failed (premature end).
    python situations_aw.py V3_EPISODES [V3_EPISODES ...] > situations_aw.json"""
import collections as C, glob, json, os, sys
def bucket(i): return "0" if i == 0 else "1-5" if i <= 5 else "6-10" if i <= 10 else "11-15" if i <= 15 else ">15"
def eff(h): h = h or ""; return "no change" if h.endswith("→ no change") else "changed" if h.endswith("→ screen changed") else "-"
out = C.defaultdict(lambda: {"n": 0, "no_effect": 0, "premature_status": 0, "act": C.Counter()})
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        if ".abandoned." in f: continue
        L = [json.loads(l) for l in open(f)]; res = [x for x in L if x.get("type") == "result"]
        if not res or res[-1].get("setup_error") or res[-1].get("unscored"): continue
        ok = bool(res[-1].get("success")); st = sorted([x for x in L if x.get("type") == "step"], key=lambda s: s["i"])
        for k, s in enumerate(st):
            if k == 0: key = ("-", "-", False, bucket(s["i"]))
            else:
                p = st[k - 1]; pa = (p.get("action_executed") or {}).get("action_type") or "-"
                key = (pa, eff(p.get("history_line")) if pa != "wait" else "-", pa == "click" and "e" in str((p.get("click_target") or {}).get("flags") or ""), bucket(s["i"]))
            o = out["|".join(map(str, key))]; o["n"] += 1
            a = (s.get("action_executed") or {}).get("action_type") or ("status" if s.get("done") else "none"); o["act"][a] += 1
            if a not in ("wait", "status", "none") and eff(s.get("history_line")) == "no change": o["no_effect"] += 1
            if s.get("done") and not ok: o["premature_status"] += 1
print(json.dumps({k: dict(v, act=dict(v["act"])) for k, v in out.items()}))
