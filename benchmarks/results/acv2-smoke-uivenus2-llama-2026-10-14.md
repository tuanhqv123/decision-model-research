# AC-v2 official protocol — smoke-uivenus2-llama

n=4 (leaderboard subset, 398 steps) | protocol: GUI-Libra native-agent (goal only, T=greedy)
Scoring: eval_official.py port (a11y-tree element grounding + equivalences).

| metric | value |
|---|---|
| **step accuracy (Pass@1)** | 0.2500  (1/4) |
| grounding accuracy (spatial steps, n=1) | 0.0000 |
| **ECE (conf vs correct, scalar-bin)** | 0.6801 |
| Brier (scalar) | 0.5782 |
| mean confidence | 0.885 |
| conf if correct / if wrong | 0.909 / 0.876 |
| latency median / p95 (ms) | 9259 / 34689 |
| coverage–risk |  t>=0.3: cov=1.00 risk=0.750 /   t>=0.5: cov=1.00 risk=0.750 /   t>=0.7: cov=1.00 risk=0.750 /   t>=0.9: cov=0.25 risk=0.000 |

Leaderboard reference (AC-v2 High Pass@1): Mem-W-8B 68.59 · GUI-Libra-8B 64.3 ·
Mem-W-4B 63.07 · UI-Venus-1.5-8B 61.06.

---
> **RETRACTED (2026-10-28): SCORER BUG.** Same `score_row` port bug as the
> 2026-10-15 baseline: spatial steps counted correct on action-type alone.
> Accuracy/ECE/coverage numbers invalid (n=4 smoke anyway). Scorer fixed and
> differentially validated vs official `eval_official.py` (312/312, 0 mismatches).
