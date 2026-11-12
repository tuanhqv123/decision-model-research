"""On rows with a known gold element: how often each element rule keeps a creditable gold element, on screens with
more than 40 candidates (below 40 every rule keeps the same list)."""
import collections, json, os, sys
sys.path.insert(0, os.path.expanduser("~/d1work/code_v3/benchmarks/train"))
import ac_common as A
CAP = A.ELEMENT_CAP

def r0(els, W, H): return A.rule_cap40_reading_noime(els, W, H)
def _prio(order):
    def rule(els, W, H):
        cand = A._reading(els, [i for i in A.rule_interactive_or_labelled(els, W, H) if not A.is_keyboard(els[i])])
        if len(cand) <= CAP: return cand
        def tier(i):
            lab, inter = bool(els[i]["label"]), A._interactive(els[i])
            return order.index((lab, inter))
        return A._reading(els, sorted(cand, key=lambda i: (tier(i), cand.index(i)))[:CAP])
    return rule
R1 = _prio([(True, True), (True, False), (False, True), (False, False)])   # labelled first
R2 = _prio([(True, True), (False, True), (True, False), (False, False)])   # interactive first
files = {"AC train (AW-like, full flow)": "~/d1work/acdata/compact_awlike_full_v3.jsonl",
         "AC validation": "~/d1work/acdata/compact_validation_v3.jsonl",
         "AC-High test": "~/d1work/actest/data/compact_actest_v3.jsonl"}
for name, path in files.items():
    st = collections.Counter()
    for line in open(os.path.expanduser(path)):
        r = json.loads(line)
        if r["action"]["action_type"] not in A.SPATIAL or not r.get("target"): continue
        canon = [A.canonical_element(e) for e in r.get("elements") or []]
        n = len([i for i in A.rule_interactive_or_labelled(canon, r["W"], r["H"]) if not A.is_keyboard(canon[i])])
        st["spatial rows"] += 1
        if n <= CAP: continue
        st["rows over the cap"] += 1
        for tag, rule in (("R0 reading (now)", r0), ("R1 labelled-first", R1), ("R2 interactive-first", R2)):
            idx = rule(canon, r["W"], r["H"]); kept = [canon[i] for i in idx]
            el, _ = A.gold_target_el_v2(r, idx, kept)
            st[tag + ": gold kept"] += el is not None
    print(name, dict(st))

print("\n--- rows where R0 kept the gold but a new rule loses it ---")
for name, path in files.items():
    lost = collections.Counter(); kinds = collections.Counter()
    for line in open(os.path.expanduser(path)):
        r = json.loads(line)
        if r["action"]["action_type"] not in A.SPATIAL or not r.get("target"): continue
        canon = [A.canonical_element(e) for e in r.get("elements") or []]
        n = len([i for i in A.rule_interactive_or_labelled(canon, r["W"], r["H"]) if not A.is_keyboard(canon[i])])
        if n <= CAP: continue
        i0 = r0(canon, r["W"], r["H"]); g0, _ = A.gold_target_el_v2(r, i0, [canon[i] for i in i0])
        if g0 is None: continue
        ge = canon[i0[int(g0) - 1]]
        for tag, rule in (("R1", R1), ("R2", R2)):
            idx = rule(canon, r["W"], r["H"]); g, _ = A.gold_target_el_v2(r, idx, [canon[i] for i in idx])
            if g is None:
                lost[tag] += 1
                kinds[(tag, "labelled" if ge["label"] else "unlabelled", "interactive" if A._interactive(ge) else "static")] += 1
    print(name, dict(lost), dict(kinds))
