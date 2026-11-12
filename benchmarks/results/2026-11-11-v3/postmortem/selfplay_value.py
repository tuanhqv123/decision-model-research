"""Why argmax self-play adds little: which tasks did self-play solve, and were they already solved by v3 (seed 30)?
    python selfplay_value.py SELFPLAY_EPISODE_DIR V3_EPISODE_DIR [V3_EPISODE_DIR ...]"""
import glob, json, os, sys
def res(dirs):
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.jsonl")):
            if ".abandoned." in f: continue
            r = [json.loads(l) for l in open(f) if '"type": "result"' in l]
            if r and not (r[-1].get("setup_error") or r[-1].get("unscored")): out[r[-1]["task_name"]] = bool(r[-1].get("success"))
    return out
sp, v3 = res([sys.argv[1]]), res(sys.argv[2:])
common = sorted(set(sp) & set(v3))
print("self-play finished tasks:", len(sp), "| successes:", sum(sp.values()))
print("self-play successes:", sorted(t for t in sp if sp[t]))
new = [t for t in common if sp[t] and not v3[t]]; lost = [t for t in common if v3[t] and not sp[t]]
print("solved by self-play AND already by v3: %d | NEW (v3 failed): %d %s | v3 solved, self-play failed: %d %s" % (
    sum(sp[t] and v3[t] for t in common), len(new), new, len(lost), lost))
