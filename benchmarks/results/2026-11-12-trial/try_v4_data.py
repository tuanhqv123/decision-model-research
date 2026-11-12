"""Trial of the v4 data fix (2026-11-12), measured on the data itself before any training.
  1. drop training TARGETS that teach a futile repeat: gold repeats an action whose history says "no change" and the
     repeat ALSO changed nothing (the rows stay in history; only the target row is removed)
  2. add rows that teach "after a failed action, do the right thing": copy a real row (same screen, same gold) and append to
     its history one click on ANOTHER clickable element of that screen with "→ no change". The screen is unchanged by
     construction, so the gold stays correct. Rate: one copy for FRAC of eligible rows.
Then the two checks: P(gold repeats previous | no change) must fall below P(... | changed), and I(effect word; gold type) rises.
    python try_v4_data.py CODE_DIR DATA.jsonl OUT.jsonl FRAC"""
import collections as C, json, math, random, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
src, out, frac = sys.argv[2], sys.argv[3], float(sys.argv[4]); random.seed(0)
RULE = A.CONTRACTS["v3"]["element_rule"]
rows = [json.loads(l) for l in open(src)]; ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
def tl(r):
    t = r.get("target")
    if not t: return None
    cx, cy = (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2; best = None
    for e in map(A.canonical_element, r.get("elements") or []):
        b = e["bbox"]
        if b and b["x_min"] <= cx <= b["x_max"] and b["y_min"] <= cy <= b["y_max"] and (best is None or (b["x_max"]-b["x_min"])*(b["y_max"]-b["y_min"]) < (best["bbox"]["x_max"]-best["bbox"]["x_min"])*(best["bbox"]["y_max"]-best["bbox"]["y_min"])): best = e
    return best
def repeats(p, r):
    pa, g = p["action"]["action_type"], r["action"]["action_type"]
    if g != pa: return False
    if pa == "scroll": return p["action"].get("direction") == r["action"].get("direction")
    if pa in ("click", "long_press"): return tl(p) is not None and tl(r) is not None and A._effect_key(tl(p)) == A._effect_key(tl(r))
    return False
def stats(pairs):
    t = C.defaultdict(C.Counter); joint = C.Counter()
    for prev, r, w in pairs:
        if prev["action"]["action_type"] == "wait": continue
        t[w]["n"] += 1; t[w]["rep"] += repeats(prev, r); joint[(w, r["action"]["action_type"])] += 1
    N = sum(joint.values()); pw, pg = C.Counter(), C.Counter()
    for (w, g), v in joint.items(): pw[w] += v; pg[g] += v
    mi = sum(v / N * math.log2((v / N) / (pw[w] / N * pg[g] / N)) for (w, g), v in joint.items())
    return {w: "%d rows, repeat %.1f%%" % (t[w]["n"], 100 * t[w]["rep"] / t[w]["n"]) for w in t}, mi
def word(r):
    h = (r.get("history_v3") or [""])[-1]
    return "no change" if h.endswith("→ no change") else "changed" if h.endswith("→ screen changed") else None
before = [(ep[r["ep"]][r["step"] - 1], r, word(r)) for r in rows if r["step"] - 1 in ep[r["ep"]] and word(r)]
print("v3 data:", *stats(before))
drop = set()
for prev, r, w in before:
    nxt = ep[r["ep"]].get(r["step"] + 1)
    if w == "no change" and repeats(prev, r) and nxt is not None and word(nxt) == "no change": drop.add((r["ep"], r["step"]))
kept = [r for r in rows if (r["ep"], r["step"]) not in drop]
added = []
for r in kept:
    if r.get("source") == "aw_generator" or r["action"]["action_type"] == "status" or random.random() > frac: continue
    kl, _ = A.select_elements(r["elements"], r["W"], r["H"], RULE)
    g = tl(r); cand = [e for e in kl if e["flags"]["clickable"] and (g is None or A._effect_key(e) != A._effect_key(g))]
    if not cand: continue
    x = random.choice(cand)
    fake = A.history_entry_v3({"action_type": "click"}, x, False)
    nr = dict(r, history_v3=list(r.get("history_v3") or []) + [fake], aug="nochange_other_click", step=r["step"] + 0.5)
    added.append((nr, x))
pairs = [(ep[r["ep"]][r["step"] - 1], r, word(r)) for r in kept if r["step"] - 1 in ep[r["ep"]] and word(r)]
# for an added row the 'previous action' is the fake click on x
for nr, x in added:
    fake_prev = {"action": {"action_type": "click"}, "target": x["bbox"], "elements": nr["elements"]}
    pairs.append((fake_prev, nr, "no change"))
print("dropped futile-repeat targets:", len(drop), "| added rows:", len(added), "| total rows:", len(kept) + len(added))
print("v4 trial data:", *stats(pairs))
with open(out, "w") as fo:
    for r in kept: fo.write(json.dumps(r, separators=(",", ":")) + "\n")
    for nr, _ in added: fo.write(json.dumps(nr, separators=(",", ":")) + "\n")
