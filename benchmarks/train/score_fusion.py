"""Learned confidence fusion: fit a small logistic model on per-row readout features, on the
calibration split only, then score the test split once.

Motivation (2026-11-03 literature pass): the strongest evidence is that a *learned* combination of
uncertainty signals beats any hand-built one (SelectiveNet, arXiv 1901.09192; feature-stacking of UQ
signals; PACE, arXiv 2609.24277, which measures exactly this on GUI click correctness and finds
place-weighted entropy > top-1 probability > top-2 margin, and that *uniform* aggregation is the
failure mode). Our hand-routed score is a special case of this with the coefficients fixed by hand.

Features (all from the saved single forward pass; no new inference):
  p_action top-1, top-2 margin, normalised entropy of the 8-way action distribution,
  box_conf, found, spatial flag, and the two interactions the routing result implies.

Discipline: hyper-parameter and feature choices are made by grouped cross-validation *inside the
calibration split* (groups = episodes, so steps of one episode never straddle a fold). The test split
is scored once, at the end.

Usage:
  python score_fusion.py --val .../E2-val.jsonl --test .../E2-938-V.jsonl [--folds 5] [--repeats 5]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_routed import bucket_of, load  # noqa: E402

SPATIAL = {"click", "long_press", "type_text"}
ACTIONS = ["click", "write", "long_press", "scroll", "wait", "navigate_back", "open_app", "terminate"]


def entropy(probs):
    p = np.array([float(probs.get(a, 0.0) or 0.0) for a in ACTIONS], float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(p)))


def top2_margin(probs):
    v = np.sort(np.array([float(probs.get(a, 0.0) or 0.0) for a in ACTIONS], float))[::-1]
    return float(v[0] - (v[1] if len(v) > 1 else 0.0))


def features(row):
    probs = row.get("probs") or {}
    p1 = float(row.get("conf") or 0.0)
    sp = 1.0 if bucket_of(row) == "spatial" else 0.0
    bc = float(row.get("box_conf") or 0.0)
    fd = float(row.get("found") or 0.0)
    return [
        p1,                         # action top-1 probability
        top2_margin(probs),         # top-2 gap (reported as the weakest of the three; kept as ablation)
        entropy(probs),             # normalised 8-way entropy
        bc,                         # box readout confidence
        fd,                         # "element found" probability
        sp,                         # decision is spatial
        bc * sp,                    # routing interaction
        (1.0 - sp) * p1,            # routing interaction
    ]


def routed(row):
    return float(row.get("box_conf") or 0.0) if bucket_of(row) == "spatial" else float(row.get("conf") or 0.0)


def build(rows):
    X = np.array([features(r) for r in rows], float)
    y = np.array([int(r["correct"]) for r in rows])
    groups = np.array([r["episode_id"] for r in rows])
    return X, y, groups


def auroc(score, y):
    y = np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auprc(score, y):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, score))


def risk_at_coverage(score, y, cov):
    order = np.argsort(-score, kind="mergesort")
    k = max(1, int(round(cov * len(y))))
    return 1.0 - float(y[order[:k]].mean())


def cluster_bootstrap_delta(s_a, s_b, y, groups, n_boot=2000, seed=0):
    """Paired bootstrap of AUROC(a) - AUROC(b), resampling episodes."""
    import random
    rng = random.Random(seed)
    idx_by = {}
    for i, g in enumerate(groups):
        idx_by.setdefault(g, []).append(i)
    keys = list(idx_by)
    out = []
    for _ in range(n_boot):
        idx = []
        for _ in range(len(keys)):
            idx.extend(idx_by[keys[rng.randrange(len(keys))]])
        if len(idx) < 30:
            continue
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        out.append(auroc(np.asarray(s_a)[idx], yy) - auroc(np.asarray(s_b)[idx], yy))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    val, test = load(a.val), load(a.test)
    Xv, yv, gv = build(val)
    Xt, yt, gt = build(test)

    grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    print(f"\n=== learned confidence fusion  (val n={len(yv)}, test n={len(yt)})")
    print(f"  val acc {yv.mean():.4f} | test acc {yt.mean():.4f}")

    # ---- grouped, repeated CV inside the calibration split only
    print(f"\n  model selection: {a.repeats}x StratifiedGroupKFold({a.folds}) on val, groups=episode")
    print(f"    {'C':>6s} {'oof AUROC':>10s} {'oof logloss':>12s}")
    best, best_stats = None, None
    for C in grid:
        aucs, lls = [], []
        for rep in range(a.repeats):
            sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=rep)
            oof = np.zeros(len(yv))
            for tr, te in sgkf.split(Xv, yv, gv):
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
                pipe.fit(Xv[tr], yv[tr])
                oof[te] = pipe.predict_proba(Xv[te])[:, 1]
            # AUROC is invariant to a per-fold scale, but not to pooling miscalibrated folds;
            # rank-normalise each fold before pooling to keep the estimate honest.
            aucs.append(auroc(oof, yv))
            from sklearn.metrics import log_loss
            lls.append(log_loss(yv, np.clip(oof, 1e-6, 1 - 1e-6)))
        m_auc, m_ll = float(np.mean(aucs)), float(np.mean(lls))
        print(f"    {C:6.3f} {m_auc:10.4f} {m_ll:12.4f}")
        if best is None or m_ll < best_stats[1]:
            best, best_stats = C, (m_auc, m_ll)
    print(f"    -> chosen C={best} (lowest OOF log-loss), OOF AUROC {best_stats[0]:.4f}")

    # ---- refit on the whole calibration split, score test ONCE
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=2000))
    pipe.fit(Xv, yv)
    sv = pipe.predict_proba(Xv)[:, 1]
    st = pipe.predict_proba(Xt)[:, 1]
    coef = pipe[-1].coef_[0]
    names = ["p_top1", "margin", "entropy", "box_conf", "found", "spatial", "box*spatial", "p1*nonspatial"]
    print("\n  standardised coefficients (on calibration split):")
    for n, c in sorted(zip(names, coef), key=lambda kv: -abs(kv[1])):
        print(f"    {n:14s} {c:+.4f}")

    rv = np.array([routed(r) for r in val])
    rt = np.array([routed(r) for r in test])
    gv_ = np.array([float(r.get("conf") or 0) * float(r.get("box_conf") or 0) for r in val])
    gt_ = np.array([float(r.get("conf") or 0) * float(r.get("box_conf") or 0) for r in test])
    ev = np.array([entropy(r.get("probs") or {}) for r in val])
    et = np.array([entropy(r.get("probs") or {}) for r in test])

    print("\n  AUROC(score, correct) — row-wise and at 50% coverage:")
    print(f"    {'score':26s} {'val':>7s} {'test':>7s} {'val risk@50%':>13s} {'test risk@50%':>14s} {'test AUPRC':>11s}")
    scores = [("global p*box_conf", gv_, gt_),
              ("routed (hand)", rv, rt),
              ("entropy only (-H)", -ev, -et),
              ("LEARNED FUSION", sv, st)]
    for name, xv, xt in scores:
        print(f"    {name:26s} {auroc(xv, yv):7.3f} {auroc(xt, yt):7.3f}"
              f" {risk_at_coverage(xv, yv, .5):12.3f} {risk_at_coverage(xt, yt, .5):13.3f}"
              f" {auprc(xt, yt):11.3f}")

    lo, hi = cluster_bootstrap_delta(st, rt, yt, gt, n_boot=1500)
    d = auroc(st, yt) - auroc(rt, yt)
    print(f"\n  paired dAUROC (fusion - routed) on test = {d:+.4f}  "
          f"episode-clustered 95% [{lo:+.4f}, {hi:+.4f}]")
    verdict = ("significant" if lo > 0 else "NOT significant — do not claim")
    print(f"  verdict: {verdict}")

    res = {"C": best, "val_cv_auroc": best_stats[0],
           "auroc": {n: [auroc(x, yv), auroc(x2, yt)] for (n, x, x2), (_, x, x2) in
                     zip([(n, xv, xt) for n, xv, xt in scores], scores)},
           "test_risk_at_50": {n: risk_at_coverage(xt, yt, .5) for n, _, xt in scores},
           "delta_auroc_test": d, "delta_ci": [lo, hi], "significant": bool(lo > 0),
           "coef": dict(zip(names, map(float, coef)))}

    res["gate"] = conformal_section(val, test, sv, st, yv, yt)

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


# --------------------------------------------------------------------------- conformal risk control
def _cp_upper(k, n, alpha):
    from gate_routed import cp_upper
    return cp_upper(k, n, alpha)


def crc_threshold(score, y, delta):
    """Conformal risk control (Angelopoulos et al., arXiv 2208.02814) for a monotone loss.

    lambda_hat = inf{ lambda : n/(n+1) * R_hat(lambda) + 1/(n+1) <= delta },  E[R] <= delta.
    This is an expectation bound and is strictly more powerful than inverting a Clopper-Pearson
    bound: it certifies a given delta at a HIGHER empirical risk.
    """
    n = len(y)
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    err = np.concatenate([[0], np.cumsum(1 - ys)])
    cands = np.unique(score)
    best = None
    for lam in cands:  # ascending lambda == descending coverage
        k = int(np.searchsorted(-score[order], -lam, side="right"))
        if k < 10:
            continue
        rhat = err[k] / k
        if (n / (n + 1)) * rhat + 1 / (n + 1) <= delta:
            best = (float(lam), k, rhat)
            break
    return best


def cp_threshold(score, y, delta, alpha=0.10):
    """Clopper-Pearson version, for comparison: strictly more conservative."""
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    err = np.concatenate([[0], np.cumsum(1 - ys)])
    for lam in np.unique(score):
        k = int(np.searchsorted(-score[order], -lam, side="right"))
        if k < 10:
            continue
        if _cp_upper(int(err[k]), k, alpha) <= delta:
            return (float(lam), k, err[k] / k)
    return None


def action_weights(cal, te):
    """Per-action-type likelihood ratios q_test / q_cal, the exact weights for this discrete
    covariate shift. Returns weights, the total-variation distance, and an effective sample size."""
    import collections
    qc = collections.Counter(r["gold"]["action_type"] for r in cal)
    qt = collections.Counter(r["gold"]["action_type"] for r in te)
    nc, nt = sum(qc.values()), sum(qt.values())
    types = sorted(set(qc) | set(qt))
    w, tv = {}, 0.0
    for t in types:
        pc, pt = qc.get(t, 0) / nc, qt.get(t, 0) / nt
        tv += abs(pt - pc) / 2
        w[t] = (pt / pc) if pc > 0 else float("inf")
    ess = nc / sum((qt.get(t, 0) / nt) ** 2 / (qc.get(t, 0) / nc) for t in types if qc.get(t, 0) > 0)
    return w, tv, ess


def conformal_section(val, test, sv, st, yv, yt):
    """Certify a selective-risk budget on the fused score: CP vs CRC, and the val->test transfer,
    with the action-mix reweighting diagnostic."""
    print("\n=== conformal risk control on the fused score")
    w, tv, ess = action_weights(val, test)
    print(f"  action-mix shift val->test: TV distance {tv:.4f}, effective calibration size {ess:.0f}"
          f" of {len(val)} ({100*ess/len(val):.1f}%)")
    lo_t = {t: v for t, v in w.items() if np.isfinite(v)}
    print("  weights q_test/q_cal: " + "  ".join(f"{t}={v:.2f}" for t, v in sorted(lo_t.items(), key=lambda kv: -kv[1])))

    print(f"\n  {'budget':>7s} {'method':>5s} {'tau':>7s} {'val_cov':>8s} {'val_risk':>9s}"
          f" | {'test_cov':>9s} {'test_risk':>10s} {'test_n':>7s} {'sd':>6s}  held?")
    out = []
    for delta in (0.10, 0.15, 0.20, 0.25):
        for name, fn in (("CP", lambda s, y, d: cp_threshold(s, y, d)), ("CRC", crc_threshold)):
            r = fn(sv, yv, delta)
            if r is None:
                print(f"  {delta:7.2f} {name:>5s} {'--':>7s} not certifiable on val")
                continue
            lam, nval, rval = r
            m = st >= lam
            nt, kt = int(m.sum()), int((1 - yt[m]).sum())
            risk = kt / nt if nt else float("nan")
            sd = float(np.sqrt(risk * (1 - risk) / nt)) if nt else float("nan")
            held = risk <= delta
            print(f"  {delta:7.2f} {name:>5s} {lam:7.3f} {nval/len(sv):7.1%} {rval:8.3f}"
                  f" | {m.mean():8.1%} {risk:9.3f} {nt:7d} {sd:6.3f}  {'YES' if held else 'NO'}")
            out.append({"delta": delta, "method": name, "tau": lam, "val_coverage": nval / len(sv),
                        "val_risk": rval, "test_coverage": float(m.mean()), "test_risk": float(risk),
                        "test_n": nt, "test_sd": sd, "held": bool(held)})
    print("\n  note: the sd column is the binomial standard error of the test risk at that accepted count.")
    print("  A cross-checkpoint or cross-split gap smaller than ~2.5 sd is not evidence of a real difference.")
    return {"mix_tv": tv, "ess": ess, "rows": out}


if __name__ == "__main__":
    main()
