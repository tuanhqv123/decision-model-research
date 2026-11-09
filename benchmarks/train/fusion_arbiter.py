"""Learned arbiter for two-checkpoint fusion — zero GPU, uses only saved readouts.

The hand rule (`box_conf x found`) picks the right checkpoint 76% of the time on the 21 rows where the two
checkpoints disagree by >=60 px, but the resulting fusion only reaches +1.26 pts (p=0.42) because a single
scalar is a weak arbiter. This fits a small logistic model on those features instead.

Protocol (the same discipline as every other arm in this programme):
  * features come from the saved prediction dumps only — nothing recomputed through the model;
  * the arbiter is fitted on VAL (827 rows), then evaluated ONCE on TEST (398);
  * the fusion only fires when the two predicted points are >= tau px apart, tau also fitted on val;
  * scoring uses each checkpoint's own recorded `correct` flag, so the fusion cannot flatter itself.

Usage:
  python3 benchmarks/train/fusion_arbiter.py \
    --val-e2 <E2 val jsonl> --val-arm <arm val jsonl> \
    --test-e2 <E2 test jsonl> --test-arm <arm test jsonl> \
    --out benchmarks/results/2026-11-07-fusion-arbiter.json
"""
import argparse
import collections
import json
import math
import random
from math import comb

import numpy as np

SPATIAL = {"click", "long_press", "type_text"}


def load(path):
    return {(r["episode_id"], r["step"]): r for r in map(json.loads, open(path, encoding="utf-8"))}


def feats(r):
    """Everything available at test time for one checkpoint's decision on one row."""
    p = r.get("probs") or {}
    vals = np.array([float(v) for v in p.values()]) if p else np.array([1.0])
    ent = float(-(vals * np.log(np.clip(vals, 1e-12, 1))).sum())
    srt = np.sort(vals)[::-1]
    margin = float(srt[0] - srt[1]) if len(srt) > 1 else 0.0
    return {
        "conf": float(r.get("conf") or 0.0),
        "box_conf": float(r.get("box_conf") or 0.0),
        "found": float(r.get("found") or 0.0),
        "entropy": ent,
        "margin": margin,
        "is_spatial": 1.0 if r.get("pred") in SPATIAL else 0.0,
    }


def pair_features(er, ar):
    """Symmetric-ish encode: absolute features of each + their difference."""
    e, a = feats(er), feats(ar)
    p1, p2 = er.get("box_pt"), ar.get("box_pt")
    dist = math.dist(p1, p2) if (p1 and p2) else 0.0
    row = []
    for k in ("conf", "box_conf", "found", "entropy", "margin", "is_spatial"):
        row += [e[k], a[k], a[k] - e[k]]
    row += [min(dist, 600.0) / 600.0]
    return np.array(row, dtype=np.float64)


def build(base, arm, keys):
    X = np.stack([pair_features(base[k], arm[k]) for k in keys])
    # label: 1 = arm is correct and E2 is not (switching helps); 0 = the reverse.
    # Rows where they AGREE are uninformative for a *switch* decision and are marked -1.
    y = np.array([1 if (arm[k]["correct"] and not base[k]["correct"]) else
                  0 if (base[k]["correct"] and not arm[k]["correct"]) else -1 for k in keys])
    return X, y


def build_pooled(runs, keys):
    """The correct scheme for >2 checkpoints: one example per (row x run).

    The earlier version asked a mutually-exclusive question ("which single run is right"), which is
    undefined whenever two or more runs are right - i.e. on 57% of rows, which were then discarded.
    Per-run correctness is a well-posed binary label and uses every row.
    """
    X, y, meta = [], [], []
    names = list(runs)
    for k in keys:
        for n in names:
            r = runs[n][k]
            X.append(feats(r))
            y.append(1 if r["correct"] else 0)
            meta.append((k, n))
    return np.stack(X), np.array(y, dtype=np.float64), meta, names


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_logistic(X, y, l2=1.0, iters=400, lr=0.5):
    """Plain gradient descent — no sklearn dependency, deterministic."""
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        gw = X.T @ (p - y) / n + l2 * w / n
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def apply_rule(base, arm, keys, X, w, b, tau, p_thresh=0.5):
    """default E2; switch to arm only when far apart AND the arbiter prefers it."""
    correct, switched = 0, 0
    for k, x in zip(keys, X):
        er, ar = base[k], arm[k]
        p1, p2 = er.get("box_pt"), ar.get("box_pt")
        far = bool(p1 and p2 and math.dist(p1, p2) >= tau)
        if far:
            prob = 1 / (1 + math.exp(-max(min(float(x @ w + b), 30), -30)))
            if prob >= p_thresh:
                correct += int(ar["correct"]); switched += 1; continue
        correct += int(er["correct"])
    return correct / len(keys), switched


def mcnemar(base, arm, keys, X, w, b, tau, p_thresh=0.5):
    vec = []
    for k, x in zip(keys, X):
        er, ar = base[k], arm[k]
        p1, p2 = er.get("box_pt"), ar.get("box_pt")
        far = bool(p1 and p2 and math.dist(p1, p2) >= tau)
        use_arm = False
        if far:
            prob = 1 / (1 + math.exp(-max(min(float(x @ w + b), 30), -30)))
            use_arm = prob >= p_thresh
        vec.append(bool((ar if use_arm else er)["correct"]))
    b_ = sum(1 for k, v in zip(keys, vec) if (not base[k]["correct"]) and v)
    c_ = sum(1 for k, v in zip(keys, vec) if base[k]["correct"] and (not v))
    n = b_ + c_
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b_, c_) + 1)) / 2 ** n)
    return vec, b_, c_, p


def cluster_ci(base, keys, vec, n_boot=10000, seed=0):
    by = collections.defaultdict(list)
    for k, v in zip(keys, vec):
        by[base[k]["episode_id"]].append((int(base[k]["correct"]), int(v)))
    cl = list(by)
    rng = random.Random(seed)
    ds = []
    for _ in range(n_boot):
        rows = [r for _ in range(len(cl)) for r in by[cl[rng.randrange(len(cl))]]]
        ds.append((sum(y for _, y in rows) - sum(x for x, _ in rows)) / len(rows))
    ds.sort()
    return ds[int(0.025 * n_boot)], ds[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-e2", required=True)
    ap.add_argument("--val-arm", required=True)
    ap.add_argument("--test-e2", required=True)
    ap.add_argument("--test-arm", required=True)
    ap.add_argument("--out", default="benchmarks/results/2026-11-07-fusion-arbiter.json")
    a = ap.parse_args()

    V2, VA = load(a.val_e2), load(a.val_arm)
    T2, TA = load(a.test_e2), load(a.test_arm)
    vk = sorted(set(V2) & set(VA))
    tk = sorted(set(T2) & set(TA))

    Xv_all, yv_all = build(V2, VA, vk)
    # fit only on DISAGREEMENT rows (where a choice exists)
    mask = yv_all >= 0
    Xv, yv = Xv_all[mask], yv_all[mask]
    Xt_all, _ = build(T2, TA, tk)
    Xv_s, Xt_s = standardise(Xv, Xt_all)
    w, b = fit_logistic(Xv_s, yv.astype(np.float64))

    base_val = sum(V2[k]["correct"] for k in vk) / len(vk)
    # choose tau on val
    # Standardise once, with the statistics the model was fitted on, and reuse them everywhere.
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
    Xv_all_s = (Xv_all - mu) / sd
    best = (-1, None)
    for tau in (0, 20, 40, 60, 80, 100, 150):
        acc, sw = apply_rule(V2, VA, vk, Xv_all_s, w, b, tau)
        if acc > best[0]:
            best = (acc, tau)
    val_acc, tau = best

    base_test = sum(T2[k]["correct"] for k in tk) / len(tk)
    test_acc, switched = apply_rule(T2, TA, tk, Xt_s, w, b, tau)
    vec, gained, lost, p = mcnemar(T2, TA, tk, Xt_s, w, b, tau)
    lo, hi = cluster_ci(T2, tk, vec)

    out = {
        "protocol": "AC-v2 398 High, pure; E2 vs stage-2 arm; arbiter fitted on val 827, test scored once",
        "arbiter": "logistic on [conf, box_conf, found, entropy, margin, is_spatial] x {E2, arm, diff} + point distance",
        "fit_rows": int(mask.sum()),
        "tau_px": tau,
        "val": {"base": base_val, "fused": val_acc},
        "test": {"base": base_test, "fused": test_acc,
                 "delta_pts": round(100 * (test_acc - base_test), 2),
                 "gained": gained, "lost": lost, "mcnemar_p": round(p, 4),
                 "ci95_pts": [round(100 * lo, 2), round(100 * hi, 2)],
                 "rows_switched": switched},
        "pass_bar": {"delta_ge_1.5": bool(100 * (test_acc - base_test) >= 1.5),
                     "significant": bool(p < 0.05),
                     "verdict": "PASS" if (100 * (test_acc - base_test) >= 1.5 and p < 0.05) else "FAIL"},
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
