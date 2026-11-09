"""Score an N1 checkpoint on the pilot holdout: accuracy, ECE, Brier, AUROC, majority baseline.

The pilot used a subset of the AC-v2 curated decisions whose images were on the server, split
85/15 into compact_n1tr.jsonl / compact_n1va.jsonl. This script evaluates only the holdout, in the
same serving path as the leaderboard run (model.system_one_batch, zero generated tokens).

Usage: python n1_analyze.py --ckpt runs/n1p-ce/ckpt --data acdata --tag ce
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_d1 import ece, run_eval, val_row  # noqa: E402
from train_d1 import load_model  # noqa: E402


def auroc(score, y):
    """Rank-based AUROC of a confidence score against a binary correctness label."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = score.argsort()
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def brier(probs, y, labels):
    """Multiclass Brier over the action distribution, averaged over rows."""
    vals = []
    for p, yy in zip(probs, y):
        v = np.array([p.get(l, 0.0) for l in labels], float)
        t = np.zeros(len(labels))
        t[labels.index(yy)] = 1.0
        vals.append(((v - t) ** 2).sum())
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="acdata")
    ap.add_argument("--valfile", default="compact_n1va.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = [val_row(json.loads(l), a.data) for l in open(f"{a.data}/{a.valfile}")]
    rows = [r for r in rows if os.path.exists(r["screenshot"])]
    print(f"[eval] {len(rows)} holdout rows", flush=True)

    model, _ = load_model(a.ckpt)
    out, summary = run_eval(model, rows, "V", None, a.batch)

    conf = [o["conf"] for o in out]
    y = [o["correct"] for o in out]
    golds = [o["gold"]["action_type"] for o in out]
    preds = [o["pred"] for o in out]
    labels = sorted(set(golds) | set(preds))
    maj = max(set(golds), key=golds.count)

    res = dict(summary)
    res.update({
        "tag": a.tag,
        "auroc_conf_correct": auroc(conf, y),
        "brier_action": brier([o["probs"] for o in out], [o["pred"] for o in out], labels),
        "ece_action_bins10": ece(conf, [int(v) for v in y], 10),
        "ece_action_bins20": ece(conf, [int(v) for v in y], 20),
        "mean_conf": float(np.mean(conf)),
        "acc_on_conf_ge_0.9": (float(np.mean([yy for c, yy in zip(conf, y) if c >= 0.9])),
                               int(sum(c >= 0.9 for c in conf))),
        "majority_class": maj,
        "majority_baseline_acc": float(np.mean([g == maj for g in golds])),
        "n": len(out),
    })
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/n1-{a.tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
