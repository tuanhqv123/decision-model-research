"""Gate: gradients with gradient checkpointing == without (same sample, same weights).
Catches recompute running under a different attention mask. Exit 1 on mismatch."""
import random, sys, torch
from train_d1 import build_memory, d1_modules, forward, load_model, load_records, make_sample
data = sys.argv[1]
model, eng = load_model("LiquidAI/d1-2B"); M = d1_modules(eng); model.train()
mem = build_memory(data); rng = random.Random(0)
recs = load_records(data, "train", 50, seed=3)
s = [make_sample(recs[i], data, "VA", mem, rng) for i in range(1)]
probe = [p for n, p in model.named_parameters() if "layers.20" in n and p.dim() == 2][:3]
def grads(ckpt):
    model.zero_grad(set_to_none=True)
    if ckpt: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else: model.gradient_checkpointing_disable()
    loss, st = forward(model, eng, M, s, backward_scale=1.0)
    print("ckpt" if ckpt else "plain", "loss", float(loss), "tokens", st["tokens"], "peak GB",
          round(torch.cuda.max_memory_allocated()/2**30, 1), flush=True)
    torch.cuda.reset_peak_memory_stats()
    return [p.grad.float().clone() for p in probe]
g0, g1 = grads(False), grads(True)
rel = max(float((a - b).norm() / (a.norm() + 1e-12)) for a, b in zip(g0, g1))
print("max rel grad diff", rel)
print("GRADCKPT", "PASS" if rel < 0.05 else "FAIL")
sys.exit(0 if rel < 0.05 else 1)
