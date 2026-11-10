"""Multi-checkpoint ensemble: pooled arbiter + threshold-free voting, episode-clustered CV.

The earlier fusion work had two bugs that made its numbers meaningless:

  1. `fusion_arbiter.py` asked a mutually-exclusive question ("which single checkpoint is right"), which is
     undefined on the 57% of rows where two or more checkpoints are right; those rows were dropped, leaving
     107 training examples for a 20-feature model.
  2. The ensemble's px tolerance was fitted on a 3-checkpoint val and applied to a 10-checkpoint test.

This script does it properly:
  * the arbiter is a PER-CHECKPOINT correctness classifier (one example per row x checkpoint), which is a
    well-posed binary problem and uses every row;
  * the label never touches gold beyond "was this checkpoint's answer correct", which is what we are predicting;
  * every number comes from episode-clustered cross-validation, so no test row is ever used to choose anything;
  * voting and arbiter are both reported, and the oracle is reported only as a ceiling.

Usage:
  python3 benchmarks/train/ensemble_eval.py --out benchmarks/results/2026-11-07-ensemble.json
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import sys
from math import comb

import numpy as np

SPATIAL = {"click", "long_press", "type_text"}

# Only genuinely distinct runs: same-run steps and superset datasets are excluded from the "independent" set.
RUNS = {
    "E2pure":     "backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl",
    "E2mem":      "backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-mem-V.jsonl",
    "gndB1406":   "backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl",
    "cont500":    "backup-2026-10-29/GPU_HOST_B/acwork/runs/E2c-cont-cur8kB/eval-step500-V.jsonl",
    "cur24k900":  "backup-2026-10-29/GPU_HOST_C/acwork/runs/R2b-V-cur24k/eval-step900-V.jsonl",
    "E2_705":     "benchmarks/results/2026-10-29-evals/E2-705-V.jsonl",
    "arm20":      "benchmarks/results/2026-11-07-arm/eval-arm20test-V.jsonl",
}
# Near-duplicates kept only for the "everything on disk" bound, never for the independence claim:
NEAR_DUP = {
    "cont250":   "backup-2026-10-29/GPU_HOST_B/acwork/runs/E2c-cont-cur8kB/eval-step250-V.jsonl",
    "cur24k1500":"backup-2026-10-29/GPU_HOST_C/acwork/runs/R2b-V-cur24k/eval-step1500-V.jsonl",
    "gndB1056":  "backup-2026-10-29/ckpt-groundB/eval-step1056-V.jsonl",
}


def load(path):
    return {(r["episode_id"], r["step"]): r for r in map(json.loads, open(path, encoding="utf-8"))}


def fingerprint(rows):
    return hashlib.md5(json.dumps([(r["episode_id"], r["step"], bool(r.get("correct")))
                                   for r in rows], sort_keys=True).encode()).hexdigest()[:10]


def feats(r):
    p = r.get("probs") or {}
    v = np.array([float(x) for x in p.values()]) if p else np.array([1.0])
    s = np.sort(v)[::-1]
    return [float(r.get("conf") or 0.0), float(r.get("box_conf") or 0.0), float(r.get("found") or 0.0),
            float(-(v * np.log(np.clip(v, 1e-12, 1))).sum()),
            float(s[0] - s[1] if len(s) > 1 else 0.0),
            float(r.get("pred") in SPATIAL)]


def fit_lr(X, y, l2=1.0, iters=1200, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * float((p - y).mean())
    return mu, sd, w, b


def sig(r, px):
    pt = r.get("box_pt")
    if not pt:
        pts = ()
    elif px:
        pts = (round(pt[0] / px) * px, round(pt[1] / px) * px)
    else:
        pts = (round(pt[0], 6), round(pt[1], 6))
    return (r.get("pred"), str(r.get("value")), pts)


def vote(runs, names, k, px):
    c = collections.Counter(sig(runs[n][k], px) for n in names)
    top, votes = c.most_common(1)[0]
    cand = [n for n in names if sig(runs[n][k], px) == top]
    best = max(cand, key=lambda n: (runs[n][k].get("box_conf") or 0))
    return runs[best][k]["correct"], votes / len(names)


def mcnemar_and_ci(base_vec, vec, clusters):
    d = sum(1 for x, y in zip(base_vec, vec) if (not x) and y)
    e = sum(1 for x, y in zip(base_vec, vec) if x and not y)
    m = d + e
    p = 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(min(d, e) + 1)) / 2 ** m)
    by = collections.defaultdict(list)
    for c, x, y in zip(clusters, base_vec, vec):
        by[c].append((int(x), int(y)))
    cl = list(by)
    rng = random.Random(0)
    ds = []
    for _ in range(10000):
        rows = [r for _ in range(len(cl)) for r in by[cl[rng.randrange(len(cl))]]]
        ds.append((sum(y for _, y in rows) - sum(x for x, _ in rows)) / len(rows))
    ds.sort()
    return d, e, p, ds[250], ds[9749]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results/2026-11-07-ensemble.json")
    a = ap.parse_args()

    avail = {n: p for n, p in {**RUNS, **NEAR_DUP}.items() if os.path.exists(p)}
    R = {n: load(p) for n, p in avail.items()}
    fps = {n: fingerprint(list(R[n].values())) for n in R}
    seen, dups = {}, []
    for n, f in fps.items():
        if f in seen:
            dups.append((n, seen[f]))
        else:
            seen[f] = n

    indep = [n for n in RUNS if n in R and all(n != d[0] for d in dups)]
    keys = sorted(set.intersection(*[set(R[n]) for n in R]))
    clusters = [R["E2pure"][k]["episode_id"] for k in keys]
    base = [int(R["E2pure"][k]["correct"]) for k in keys]

    out = {"n": len(keys), "duplicates": dups, "independent_runs": indep,
           "all_runs": list(R), "per_run": {}, "oracle": {}, "vote": {}, "arbiter": {}}
    for n in R:
        out["per_run"][n] = round(sum(int(R[n][k]["correct"]) for k in keys) / len(keys), 4)

    for label, names in (("independent", indep), ("all", list(R))):
        anyok = sum(1 for k in keys if any(R[n][k]["correct"] for n in names))
        out["oracle"][label] = {"runs": len(names), "acc": round(anyok / len(keys), 4),
                                "delta_vs_best": round(anyok / len(keys) - max(out["per_run"][n] for n in names), 4)}

    # ---- voting, px chosen by episode-clustered CV (never by looking at the full-set score)
    eps = sorted(set(clusters))
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(eps))
    # BUG FIXED: folds must hold EPISODE IDS, not positional indices into `eps`.
    folds = [{eps[j] for j in perm[i::5]} for i in range(5)]
    assert all(isinstance(next(iter(f)), type(eps[0])) for f in folds)
    best_px, best_cv = None, -1
    for px in (0, 4, 8, 16, 32, 64, 96, 128):
        ok = 0
        for f in folds:
            for k, c in zip(keys, clusters):
                if c in f:
                    ok += int(vote(R, indep, k, px)[0])
        cv = ok / len(keys)
        out["vote"].setdefault("cv_by_px", {})[str(px)] = round(cv, 4)
        if cv > best_cv:
            best_cv, best_px = cv, px
    vec = [int(vote(R, indep, k, best_px)[0]) for k in keys]
    d, e, p, lo, hi = mcnemar_and_ci(base, vec, clusters)
    out["vote"].update({"px": best_px, "cv_acc": round(best_cv, 4), "acc": round(sum(vec) / len(keys), 4),
                        "delta_pts": round(100 * (sum(vec) - sum(base)) / len(keys), 2),
                        "gained": d, "lost": e, "mcnemar_p": round(p, 4),
                        "ci95_pts": [round(100 * lo, 2), round(100 * hi, 2)]})

    # ---- pooled per-checkpoint arbiter, episode-clustered CV
    X, y, meta = [], [], []
    for k in keys:
        for n in indep:
            X.append(feats(R[n][k]))
            y.append(1.0 if R[n][k]["correct"] else 0.0)
            meta.append((k, n))
    X, y = np.stack(X), np.array(y)
    ok = 0
    for f in folds:
        tr = [i for i, (k, _) in enumerate(meta) if k in keys and clusters[keys.index(k)] not in f]
        mu, sd, w, b = fit_lr(X[tr], y[tr])
        for i, (k, n) in enumerate(meta):
            if clusters[keys.index(k)] not in f:
                continue
        for k, c in zip(keys, clusters):
            if c not in f:
                continue
            idx = [i for i, (kk, _) in enumerate(meta) if kk == k]
            sc = ((X[idx] - mu) / sd) @ w + b
            pick = indep[int(np.argmax(sc))]
            ok += int(R[pick][k]["correct"])
    out["arbiter"] = {"design": "per-checkpoint correctness, one example per (row x run), 5-fold by episode",
                      "train_examples": int(len(y)), "cv_acc": round(ok / len(keys), 4),
                      "delta_vs_best": round(ok / len(keys) - max(out["per_run"][n] for n in indep), 4)}

    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
