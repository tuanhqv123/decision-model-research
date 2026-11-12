"""Paired AC-High comparison of two per-step files (score_actest.py output): McNemar, per gold action, pred-type confusion of losses.
Usage: python3 compare_actest.py A.jsonl B.jsonl"""
import json, sys, collections, math
def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l); d[(r["episode_id"], r["step"])] = r
    return d
A, B = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(A) & set(B)); print("paired steps", len(keys), "A only", len(set(A)-set(B)), "B only", len(set(B)-set(A)))
ea = sum(A[k]["exact_match"] for k in keys); eb = sum(B[k]["exact_match"] for k in keys)
w = sum((not A[k]["exact_match"]) and B[k]["exact_match"] for k in keys); l = sum(A[k]["exact_match"] and not B[k]["exact_match"] for k in keys)
z = (w - l) / math.sqrt(w + l) if w + l else 0
print(f"exact A {ea/len(keys):.4f}  B {eb/len(keys):.4f}  B wins {w} losses {l}  McNemar z {z:.2f}")
by = collections.defaultdict(lambda: [0, 0, 0, 0])
conf = collections.Counter()
for k in keys:
    g = A[k]["gt"]["action"]; a, b = A[k]["exact_match"], B[k]["exact_match"]
    s = by[g]; s[0] += 1; s[1] += a; s[2] += b
    if a and not b: conf[(g, B[k]["pred"].get("action"))] += 1
print("gold action        n     A      B    Δ")
for g, (n, a, b, _) in sorted(by.items(), key=lambda x: -x[1][0]):
    print(f"{g:15s} {n:5d} {a/n:.3f}  {b/n:.3f}  {b/n-a/n:+.3f}")
print("A right -> B wrong, (gold, B pred) top:")
for (g, p), c in conf.most_common(12): print(f"  {g:12s} -> {p}: {c}")
