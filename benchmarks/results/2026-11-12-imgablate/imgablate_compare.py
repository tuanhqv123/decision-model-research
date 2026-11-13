"""Real vs grey screenshots, same checkpoint, same 827 validation rows: what the image contributes."""
import json, sys, collections
A = {(r["episode_id"], r["step"]): r for r in map(json.loads, open(sys.argv[1]))}
B = {(r["episode_id"], r["step"]): r for r in map(json.loads, open(sys.argv[2]))}
k = sorted(set(A) & set(B))
w = sum(A[x]["correct"] and not B[x]["correct"] for x in k); l = sum(B[x]["correct"] and not A[x]["correct"] for x in k)
print("rows %d | real %d | grey %d | real-only right %d | grey-only right %d" % (len(k), sum(A[x]["correct"] for x in k), sum(B[x]["correct"] for x in k), w, l))
by = collections.defaultdict(lambda: [0, 0, 0])
for x in k:
    g = A[x]["gold"]["action_type"]; by[g][0] += 1; by[g][1] += A[x]["correct"]; by[g][2] += B[x]["correct"]
for g, (n, a, b) in sorted(by.items(), key=lambda t: -t[1][0]):
    print("  %-14s n=%3d real %3d grey %3d" % (g, n, a, b))
same = sum(A[x]["pred"] == B[x]["pred"] and A[x].get("el_idx") == B[x].get("el_idx") for x in k)
print("identical decision (action and element) %d/%d" % (same, len(k)))
