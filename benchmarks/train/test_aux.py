"""Gate: auxiliary grounding samples go through the training forward (finite loss with presence/edge/hit terms)
and through the serving path (a box comes back). Exit 1 on failure."""
import math, random, sys, torch
from train_d1 import W_LOSS, d1_modules, forward, load_model, load_records, make_sample
data, aux = sys.argv[1], sys.argv[2]
W_LOSS["hit"] = 1.0
model, eng = load_model("LiquidAI/d1-2B"); M = d1_modules(eng); model.eval()
recs = load_records(data, aux, 8, seed=2); rng = random.Random(0); bad = 0
for rec in recs:
    s = make_sample(rec, data, "V", None, rng)
    with torch.no_grad():
        loss, st = forward(model, eng, M, [s])
    box = model.system_one_batch([(s[0], s[1], [s[3]])])[0]["answers"]["target_box"]
    ok = math.isfinite(float(loss)) and {"presence", "edge", "hit"} <= set(st) and "found" in box
    bad += not ok
    print(f"[{'OK ' if ok else 'BAD'}] {s[0][:50]!r:55s} loss={float(loss):.3f} p_hit={st.get('p_hit',[0])[0]:.3f} found={box.get('found',0):.2f}")
print("AUX", "PASS" if bad == 0 else f"FAIL ({bad})"); sys.exit(1 if bad else 0)
