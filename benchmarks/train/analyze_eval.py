"""Paper table from eval_d1 jsonl: step accuracy next to trivial baselines and the leaderboard, per-type
accuracy, calibration (ECE, Brier, AUROC of confidence vs correctness) and the selective-prediction curve
(act only when confidence >= tau: coverage vs risk) — the act/ask/refuse gate.

Usage: python analyze_eval.py results/eval-<tag>-V.jsonl [more.jsonl ...]
"""
import json
import sys

import numpy as np

LEADERBOARD = {"Mem-W-8B": 68.59, "GUI-Libra-8B": 64.3, "Mem-W-4B": 63.07, "GPT-5+UGround": 61.3,
               "UI-Venus-1.5-8B": 61.06, "Qwen3-VL-8B": 54.8, "Qwen3-VL-2B": 45.48}


def ece(c, y, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (c > lo) & (c <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(c[m].mean() - y[m].mean())
    return e


def auroc(c, y):
    pos, neg = c[y == 1], c[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    y = np.array([float(r["correct"]) for r in rows])
    c = np.array([r["conf"] for r in rows])
    gold = [r["gold"]["action_type"] for r in rows]
    print(f"\n=== {path}  (n={n})")
    print(f"step accuracy      : {100*y.mean():.2f}%  ({int(y.sum())}/{n})")
    print(f"type accuracy      : {100*np.mean([r['pred']==g for r,g in zip(rows,gold)]):.2f}%")
    for name, v in [("always-status", np.mean([g == "status" for g in gold])),
                    ("always-click (no point)", 0.0)]:
        print(f"baseline {name:16s}: {100*v:.2f}%")
    sp = [i for i, g in enumerate(gold) if g in ("click", "long_press", "type_text")]
    ground_lb = np.mean([bool(rows[i]["grounding"]) for i in sp]) if sp else float("nan")
    print(f"leaderboard-def    : Type {100*np.mean([r['pred']==g for r,g in zip(rows,gold)]):.2f} | "
          f"Ground {100*ground_lb:.2f}  (eval_rl.py: hits / all gold-spatial steps; Mem-W-8B 81.16 / 71.30)")
    beat = [k for k, v in LEADERBOARD.items() if 100 * y.mean() > v]
    print(f"beats leaderboard  : {beat or 'none'}")
    for t in sorted(set(gold)):
        m = np.array([g == t for g in gold])
        print(f"  {t:14s} {int(y[m].sum()):3d}/{int(m.sum()):3d} = {100*y[m].mean():5.1f}%")
    print(f"ECE {ece(c, y):.4f} | Brier {np.mean((c-y)**2):.4f} | AUROC(conf->correct) {auroc(c, y):.3f} "
          f"| mean conf ok/bad {c[y==1].mean() if y.any() else 0:.3f}/{c[y==0].mean() if (1-y).any() else 0:.3f}")
    print("selective (act iff conf>=tau):  tau  coverage  risk  acc_on_covered")
    for tau in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = c >= tau
        if m.any():
            print(f"   {tau:4.2f}   {100*m.mean():6.1f}%  {100*(1-y[m].mean()):5.1f}%  {100*y[m].mean():5.1f}%")
    lat = [r["latency_ms"] for r in rows]
    print(f"latency median {np.median(lat):.0f} ms/step (batched)")
