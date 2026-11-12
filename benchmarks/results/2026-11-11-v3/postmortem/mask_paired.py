"""Masking diagnostic (loop_detection.mask on) vs v3 (mask off): same checkpoint, same tasks, same seed 30; paired by task.
    python mask_paired.py V3_EPISODE_DIRS... -- MASK_EPISODE_DIRS..."""
import glob, json, os, sys
def load(dirs):
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.jsonl")):
            if ".abandoned." in f: continue
            r = [json.loads(l) for l in open(f) if '"type": "result"' in l or '"type":"result"' in l]
            if r and not (r[-1].get("setup_error") or r[-1].get("unscored")): out[r[-1]["task_name"]] = bool(r[-1].get("success"))
    return out
i = sys.argv.index("--"); a, b = load(sys.argv[1:i]), load(sys.argv[i + 1:])
common = sorted(set(a) & set(b))
w = sum(b[t] and not a[t] for t in common); l = sum(a[t] and not b[t] for t in common)
print("tasks scored in both:", len(common)); print("v3 (no mask): %d/%d" % (sum(a[t] for t in common), len(common)))
print("mask on:      %d/%d" % (sum(b[t] for t in common), len(common))); print("mask wins %d, mask losses %d" % (w, l))
print("changed tasks:", [(t, a[t], b[t]) for t in common if a[t] != b[t]])
