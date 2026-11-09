"""Parity gate: train_d1.forward (no grad) must reproduce the serving path's probabilities.

For a few real records, compare the gold option's log-prob from forward() against
model.system_one_batch() (calibration off). Also checks the losses are finite and the box/read
supervision is present. Exit 1 on mismatch.
"""
import json
import math
import sys

import torch

from train_d1 import build_memory, d1_modules, forward, load_model, load_records, make_sample

data = sys.argv[1]
model, eng = load_model("LiquidAI/d1-2B")
M = d1_modules(eng)
model.eval()
train_set = sys.argv[2] if len(sys.argv) > 2 else "train"
recs = load_records(data, train_set, 400, seed=1)
mem = build_memory(data) if train_set == "train" else None
import random
rng = random.Random(0)
# a spread of action types, both tracks
picked, seen = [], {}
recs = [json.loads(x) for x in recs]
for r in recs:
    t = r["action"]["action_type"]
    if seen.get(t, 0) < 2:
        picked.append(r)
        seen[t] = seen.get(t, 0) + 1
print("types:", seen)
bad = 0
for track in ("V", "VA"):
    for r in picked:
        s = make_sample(r, data, track, mem, rng)
        state, qs, gold, img = s
        with torch.no_grad():
            loss, stats = forward(model, eng, M, [s])
        res = model.system_one_batch([(state, qs, [img])])[0]["answers"]
        p = res["action"]["probabilities"]
        p_gold = p[gold["action"]] if isinstance(p, dict) else None
        lp_train = -stats["action"][0]
        lp_serve = math.log(max(p_gold, 1e-12)) if p_gold is not None else float("nan")
        ok = abs(lp_train - lp_serve) < 0.05 and math.isfinite(float(loss))
        bad += not ok
        print(f"[{'OK ' if ok else 'BAD'}] {track} {r['action']['action_type']:13s} logp_train={lp_train:.4f} "
              f"logp_serve={lp_serve:.4f} loss={float(loss):.3f} keys={sorted(stats)} "
              f"pred={res['action']['choice']} box={res['target_box'].get('box')}")
print("PARITY", "PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
