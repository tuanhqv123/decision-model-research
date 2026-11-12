#!/usr/bin/env python3
"""Pre-training checks of the v4 data (2026-11-12), against the v3 data. Every check prints PASS/FAIL with its numbers.
    python3 check_v4_data.py CODE_DIR V3.jsonl V4.jsonl"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1])
import ac_common as A
import build_v4_data as B
B.A = A
RULE4 = A.CONTRACTS["v4"]["element_rule"]


def load(p):
    rows = [json.loads(l) for l in open(p)]
    ep = C.defaultdict(dict)
    for r in rows:
        if r.get("aug", "base") == "base":
            ep[r["ep"]][r["step"]] = r
    return rows, ep


def after_no_change(rows, ep, rule, real_only):
    t = C.defaultdict(C.Counter)
    for r in rows:
        if real_only and r.get("aug", "base") != "base":
            continue
        if r.get("avoid_action") or not r.get("action"):
            continue
        h = r.get("history_v3") or []
        if not h or not (h[-1].endswith("→ no change") or h[-1].endswith("→ screen changed")):
            continue
        w = "no change" if h[-1].endswith("→ no change") else "changed"
        prev = ep.get(r["ep"], {}).get(r["step"] - 1) if r.get("aug", "base") == "base" else None
        rep = bool(prev) and B.is_repeat(prev, r, rule)
        t[w]["n"] += 1; t[w]["repeat"] += rep; t[w]["status"] += r["action"]["action_type"] == "status"
    return {w: "n=%d repeat %.1f%% status %.1f%%" % (v["n"], 100 * v["repeat"] / v["n"], 100 * v["status"] / v["n"]) for w, v in t.items()}, t


def main(code, p3, p4):
    r3, e3 = load(p3); r4, e4 = load(p4)
    ok = True
    def check(name, cond, info):
        nonlocal ok
        ok &= bool(cond); print("%s  %-62s %s" % ("PASS" if cond else "FAIL", name, info))
    s3, t3 = after_no_change(r3, e3, "cap40_reading_noime", True)
    s4r, t4r = after_no_change(r4, e4, RULE4, True)
    s4, t4 = after_no_change(r4, e4, RULE4, False)
    print("v3 data:", s3); print("v4 real rows:", s4r); print("v4 all rows:", s4)
    rep_nc = t4r["no change"]["repeat"] / max(1, t4r["no change"]["n"]); rep_ch = t4r["changed"]["repeat"] / max(1, t4r["changed"]["n"])
    check("real rows: repeat after no change < v3's 19.2%", rep_nc < 0.192, "%.1f%% (after changed %.1f%%)" % (100 * rep_nc, 100 * rep_ch))
    st_nc3 = t3["no change"]["status"] / t3["no change"]["n"]; st_nc4 = t4["no change"]["status"] / t4["no change"]["n"]
    check("P(status | last line 'no change') not collapsed (>= 1/2 of v3)", st_nc4 >= st_nc3 / 2, "v3 %.1f%% -> v4 %.1f%%" % (100 * st_nc3, 100 * st_nc4))
    def share(rows):
        g = [r for r in rows if r.get("action")]
        return sum(r["action"]["action_type"] == "status" for r in g) / len(g)
    sh3, sh4 = share(r3), share(r4)
    check("status share within 3 points of v3", abs(sh3 - sh4) <= 0.03, "v3 %.1f%% -> v4 %.1f%%" % (100 * sh3, 100 * sh4))
    aug = C.Counter(r.get("aug", "base") for r in r4)
    base = aug["base"]
    check("augmentation share <= 45% of base rows", sum(v for k, v in aug.items() if k != "base") <= 0.45 * base, dict(aug))
    check("no element is focused", not any(e.get("is_focused") for r in r4 for e in r["elements"]), "")
    bad = 0; n_ans = 0; n_av = 0
    for r in r4:
        if r.get("avoid_action") or (r.get("action") or {}).get("action_type") == "answer":
            state, kept, idx = A.element_inputs_v3(r["goal"], r["history_v3"], r["elements"], r["W"], r["H"], RULE4)
            g, _ = A.gold_answers_v2(r, kept, idx, state, apps=["x"], contract="v4")
            if r.get("avoid_action"):
                n_av += 1; bad += g != {"avoid_action": "terminate"}
            else:
                n_ans += 1; bad += (r["answer_value"] == "element") != ("target_el" in g)
    check("every answer/avoid row has a resolvable gold", bad == 0, "answer %d, avoid %d, bad %d" % (n_ans, n_av, bad))
    nk = sorted(len(A.select_elements(r["elements"], r["W"], r["H"], RULE4)[0]) for r in r4[::25])
    check("uncapped list size sane (p99 <= 400)", nk[int(.99 * len(nk))] <= 400, "median %d p90 %d p99 %d max %d" % (nk[len(nk)//2], nk[int(.9*len(nk))], nk[int(.99*len(nk))], nk[-1]))
    by = C.defaultdict(C.Counter)
    for r in r4:
        if r.get("action") and r.get("aug") == "base":
            b = min(r["step"], 12); by[b]["n"] += 1; by[b]["s"] += r["action"]["action_type"] == "status"
    print("P(status | step) base rows:", {b: "%.0f%%" % (100 * v["s"] / v["n"]) for b, v in sorted(by.items())})
    print("OVERALL", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main(*sys.argv[1:4])
