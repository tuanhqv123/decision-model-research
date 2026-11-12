"""Is the training label "→ no change" true? (2026-11-12) The label comes from the ELEMENT LIST (`effect`, computed by
ac_common.effect_changed on forest_to_ui_elements). Each row also carries `pixel_change`: the fraction of pixels that
differ between this screenshot and the next one. A step whose elements are "unchanged" but whose pixels changed a lot
is a label the model was taught that the screen contradicts.
  part 1: effect x pixel_change, per action type (row = the action taken; label = its effect)
  part 2: the 41 rows "previous scroll had no change, gold = scroll again": read by hand (goal, step instructions)
    python effect_vs_pixels.py DATA.jsonl"""
import collections as C, json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]; ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
def pb(p): return "n/a" if p is None else "<1%" if p < .01 else "1-5%" if p < .05 else "5-20%" if p < .2 else ">=20%"
t = C.defaultdict(C.Counter)
for r in rows:
    if r.get("effect") is None: continue
    t[(r["action"]["action_type"], "changed" if r["effect"] else "NO CHANGE")][pb(r.get("pixel_change"))] += 1
print("part 1: label from the element list  x  pixels changed after the action")
print("%-34s %6s  %s" % ("action, label", "n", "  ".join("%7s" % b for b in ("<1%", "1-5%", "5-20%", ">=20%", "n/a"))))
for k in sorted(t):
    n = sum(t[k].values()); print("%-34s %6d  %s" % (", ".join(k), n, "  ".join("%6.0f%%" % (100 * t[k][b] / n) for b in ("<1%", "1-5%", "5-20%", ">=20%", "n/a"))))
print("\npart 2: previous scroll labelled 'no change', gold is scroll again")
k = 0
for e, st in ep.items():
    for s, r in st.items():
        h = (r.get("history_v3") or [""])[-1]
        if s - 1 in st and st[s - 1]["action"]["action_type"] == "scroll" and h.endswith("→ no change") and r["action"]["action_type"] == "scroll":
            p = st[s - 1]; k += 1
            print("%2d ep %s step %d | goal: %s" % (k, e, s, r["goal"][:90]))
            print("     prev: scroll %s, pixels changed %s | step instruction: %s" % (p["action"].get("direction"), p.get("pixel_change"), p.get("local_goal")))
            print("     gold: scroll %s | step instruction: %s" % (r["action"].get("direction"), r.get("local_goal")))
