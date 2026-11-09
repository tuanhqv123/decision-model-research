"""Gate: center_in_range_logprob == Monte-Carlo P(lo <= (A+B)/2 <= hi) on the 0-999 grid; gradients flow."""
import math, sys, torch
sys.argv = ["x"]
from train_d1 import center_in_range_logprob
torch.manual_seed(0)
n, bad = 1000, 0
vals = torch.arange(n) / 1000.0
for t in range(6):
    la = torch.randn(n) * (3 if t % 2 else 0.5); lb = torch.randn(n) * 2
    if t >= 4:  # peaked, like a trained edge head
        la = -((torch.arange(n) - 300 - 20 * t) / 6.0) ** 2; lb = -((torch.arange(n) - 420) / 9.0) ** 2
    la, lb = la.log_softmax(0).requires_grad_(), lb.log_softmax(0)
    lo, hi = sorted(torch.rand(2).tolist()) if t < 4 else (0.35, 0.37)
    exact = center_in_range_logprob(la, lb, lo, hi).exp().item()
    A = torch.multinomial(la.exp().detach(), 200000, True); B = torch.multinomial(lb.exp(), 200000, True)
    c = (vals[A] + vals[B]) / 2
    mc = ((c >= lo) & (c <= hi)).float().mean().item()
    center_in_range_logprob(la, lb, lo, hi).backward()
    ok = abs(exact - mc) < 0.006 and la.grad is not None and torch.isfinite(la.grad).all()
    bad += not ok
    print(f"[{'OK ' if ok else 'BAD'}] range [{lo:.3f},{hi:.3f}] exact {exact:.4f}  monte-carlo {mc:.4f}  grad-norm {la.grad.norm():.3e}")
print("HITLOSS", "PASS" if bad == 0 else "FAIL"); sys.exit(1 if bad else 0)
