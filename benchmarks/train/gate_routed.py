"""Type-routed calibrated gate for the single-pass decision model, with split-conformal / LTT risk
control fitted on the validation split and certified on the test split.

Why routed: on spatial rows the box readout's confidence is the honest signal (AUROC ~0.80) while the
action prior is weak (~0.69); on non-spatial rows the box readout is meaningless and multiplying by it
is actively harmful (AUROC drops to chance). One global p(action)*box_conf gate therefore averages two
incomparable signals. Route by bucket instead.

Transform: per-bucket empirical CDF fitted on the validation split, applied unchanged to test — so the
score is computable per row and there is no transductive leakage from test.

Risk control: for a budget delta, pick the smallest threshold whose one-sided Clopper-Pearson upper
bound on selective risk (errors / acted) on the *validation* split is <= delta, then report the test
coverage, empirical risk, and the certified bound.

Usage:
  python gate_routed.py --val results/eval-val.jsonl --test results/eval-test.jsonl \
      [--delta 0.10] [--alpha 0.1] [--json out.json]
"""
import json
import math
import sys

import numpy as np

SPATIAL = {"click", "long_press", "type_text"}
BUCKETS = ("spatial", "nonspatial")


# ----------------------------------------------------------------------------- metrics
def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier_murphy(p, y, bins=10):
    """Binary Brier with the Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    n = len(p)
    bs = float(np.mean((p - y) ** 2))
    ybar = float(y.mean())
    unc = ybar * (1 - ybar)
    rel = res = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if m.any():
            w = m.sum() / n
            rel += w * (p[m].mean() - y[m].mean()) ** 2
            res += w * (y[m].mean() - ybar) ** 2
    return {"brier": bs, "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(unc), "check": float(rel - res + unc)}


def aurc(score, y):
    """Area under the risk-coverage curve (lower is better); tau swept from high to low score."""
    score, y = np.asarray(score, float), np.asarray(y, float)
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    n = len(ys)
    curves = []
    for k in range(1, n + 1):
        curves.append((k / n, 1 - ys[:k].mean()))
    cov = np.array([c for c, _ in curves])
    risk = np.array([r for _, r in curves])
    return float(np.trapezoid(risk, cov))


def _logpmf(i, n, p):
    if p <= 0:
        return 0.0 if i == 0 else -math.inf
    if p >= 1:
        return 0.0 if i == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))


def cp_upper(k, n, alpha=0.1):
    """Exact one-sided Clopper-Pearson upper bound on a binomial proportion (k errors out of n).

    Closed form is the (1-alpha) quantile of Beta(k+1, n-k); use scipy when present, else bisect the
    binomial tail. The bound is exact either way, so the threshold search is deterministic."""
    if n == 0 or k >= n:
        return 1.0
    try:
        from scipy.stats import beta as _beta
        return float(_beta.ppf(1 - alpha, k + 1, n - k))
    except Exception:
        pass
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = sum(math.exp(_logpmf(i, n, mid)) for i in range(k + 1))
        if cdf > alpha:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


# ----------------------------------------------------------------------------- scoring
def bucket_of(row):
    return "spatial" if row["gold"]["action_type"] in SPATIAL else "nonspatial"


def raw_score(row):
    """The signal that is semantically responsible for the failure in this bucket."""
    if bucket_of(row) == "spatial":
        return float(row.get("box_conf") or 0.0)
    return float(row.get("conf") or 0.0)


def global_score(row):
    """What the repo previously used: p(action) * box_conf on every row."""
    return float(row.get("conf") or 0.0) * float(row.get("box_conf") or 0.0)


def fit_ecdf(val_rows):
    return {b: np.sort(np.array([raw_score(r) for r in val_rows if bucket_of(r) == b], float))
            for b in BUCKETS}


def routed(rows, ecdf):
    out = []
    for r in rows:
        b = bucket_of(r)
        ref = ecdf[b]
        if ref is None or len(ref) < 2:
            out.append(raw_score(r))
            continue
        # mid-rank ECDF: fraction of calibration scores <= s
        out.append(float(np.searchsorted(ref, raw_score(r), side="right") / len(ref)))
    return np.array(out)


# ----------------------------------------------------------------------------- risk control
def choose_threshold(vals, y, delta, alpha, max_cands=None):
    """Smallest threshold whose CP upper bound on selective risk is <= delta (coverage maximised).

    Scans candidates in ascending order (descending coverage) and returns the first feasible tau, i.e.
    the one with the highest certified coverage. Deterministic: the bound is exact and the default
    scans every distinct threshold."""
    cands = np.unique(vals)
    if max_cands and len(cands) > max_cands:
        idx = np.linspace(0, len(cands) - 1, max_cands).astype(int)
        cands = cands[idx]
    order = np.argsort(-vals, kind="mergesort")  # descending score
    ys = y[order]
    err_prefix = np.concatenate([[0], np.cumsum(1 - ys)])
    asc = -vals[order]  # ascending, so a prefix count of score >= tau is a searchsorted
    for tau in cands:  # ascending tau == descending coverage
        n_act = int(np.searchsorted(asc, -tau, side="right"))
        if n_act < 10:
            continue
        k_err = int(err_prefix[n_act])
        ub = cp_upper(k_err, n_act, alpha)
        if ub <= delta:
            return (float(tau), n_act / len(vals), k_err / n_act, ub)
    return None


# ----------------------------------------------------------------------------- main
def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def report(val_path, test_path, deltas=(0.05, 0.10, 0.20), alpha=0.1):
    val, test = load(val_path), load(test_path)
    yv = np.array([int(r["correct"]) for r in val])
    yt = np.array([int(r["correct"]) for r in test])
    print(f"\n=== type-routed gate")
    print(f"  val  n={len(val):4d}  acc={yv.mean():.4f}")
    print(f"  test n={len(test):4d}  acc={yt.mean():.4f}")

    ecdf = fit_ecdf(val)
    rv, rt = routed(val, ecdf), routed(test, ecdf)
    gv = np.array([global_score(r) for r in val])
    gt = np.array([global_score(r) for r in test])

    print("\n  AUROC(score, correct) — the mechanism claim:")
    print(f"    {'score':30s} {'val':>7s} {'test':>7s}")
    for name, sv, st in [("global p(action)*box_conf (old)", gv, gt),
                         ("p(action) only", np.array([r['conf'] for r in val]), np.array([r['conf'] for r in test])),
                         ("box_conf only", np.array([r['box_conf'] for r in val]), np.array([r['box_conf'] for r in test])),
                         ("TYPE-ROUTED (new)", rv, rt)]:
        print(f"    {name:30s} {auroc(sv, yv):7.3f} {auroc(st, yt):7.3f}")
    print("\n  AUROC within bucket (test):")
    for b in BUCKETS:
        mv = np.array([bucket_of(r) == b for r in val]); mt = np.array([bucket_of(r) == b for r in test])
        print(f"    {b:12s} n_val={int(mv.sum()):3d} n_test={int(mt.sum()):3d}"
              f"  p(action) {auroc(np.array([r['conf'] for r in test])[mt], yt[mt]):.3f}"
              f"   box_conf {auroc(np.array([r['box_conf'] or 0 for r in test])[mt], yt[mt]):.3f}"
              f"   routed {auroc(rt[mt], yt[mt]):.3f}")

    print("\n  AURC (lower is better):")
    print(f"    global  val {aurc(gv, yv):.4f}  test {aurc(gt, yt):.4f}")
    print(f"    routed  val {aurc(rv, yv):.4f}  test {aurc(rt, yt):.4f}")

    sp_v = np.array([bucket_of(r) == "spatial" for r in val])
    sp_t = np.array([bucket_of(r) == "spatial" for r in test])
    bc_v = np.array([r["box_conf"] or 0 for r in val])[sp_v]
    bc_t = np.array([r["box_conf"] or 0 for r in test])[sp_t]
    print("\n  box_conf on spatial rows — the honest readout:")
    for tag, p, y in [("val", bc_v, yv[sp_v]), ("test", bc_t, yt[sp_t])]:
        m = brier_murphy(p, y)
        print(f"    {tag:5s} mean={p.mean():.3f} acc={y.mean():.3f} ECE={ece(p, y):.4f} "
              f"Brier={m['brier']:.4f} = REL {m['reliability']:.4f} - RES {m['resolution']:.4f} "
              f"+ UNC {m['uncertainty']:.4f}")

    print(f"\n  risk control (threshold fitted on val, alpha={alpha}); coverage maximised")
    print(f"    {'delta':>6s} {'tau':>7s} {'val_cov':>8s} {'val_risk':>9s} {'val_UB':>7s}"
          f" | {'test_cov':>9s} {'test_risk':>10s} {'test_UB':>8s} {'test_n':>7s}  held?")
    out = []
    for d in deltas:
        r = choose_threshold(rv, yv, d, alpha)
        if r is None:
            print(f"    {d:6.2f}      --  no feasible threshold on val at alpha={alpha}")
            continue
        tau = r[0]
        m = rt >= tau
        n_act = int(m.sum())
        k_err = int((1 - yt[m]).sum())
        test_risk = k_err / n_act if n_act else float("nan")
        test_ub = cp_upper(k_err, n_act, alpha)
        held = test_risk <= d
        print(f"    {d:6.2f} {tau:7.3f} {100*r[1]:7.1f}% {100*r[2]:8.1f}% {100*r[3]:6.1f}%"
              f" | {100*m.mean():8.1f}% {100*test_risk:9.1f}% {100*test_ub:7.1f}% {n_act:7d}"
              f"  {'YES' if held else 'NO'}")
        out.append({"delta": d, "tau": tau, "val_coverage": r[1], "val_risk": r[2], "val_ub": r[3],
                    "test_coverage": float(m.mean()), "test_risk": test_risk, "test_ub": test_ub,
                    "test_n": n_act, "held": bool(held)})

    print("\n  operating curve (test, routed):")
    print(f"    {'tau':>6s} {'coverage':>9s} {'risk':>7s}")
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = rt >= tau
        if m.any():
            print(f"    {tau:6.2f} {100*m.mean():8.1f}% {100*(1-yt[m].mean()):6.1f}%")

    print("\n  attainable frontier on val (given this signal and n_val; alpha=%.2f):" % alpha)
    print(f"    {'certified risk <= delta':25s} {'max coverage':>13s}")
    frontier = []
    for d in (0.30, 0.25, 0.20, 0.15, 0.10, 0.05):
        r = choose_threshold(rv, yv, d, alpha)
        if r is None:
            print(f"    {d:25.2f} {'not certifiable':>13s}")
            frontier.append({"delta": d, "max_coverage": None})
        else:
            print(f"    {d:25.2f} {100*r[1]:12.1f}%")
            frontier.append({"delta": d, "max_coverage": r[1]})

    res = {"val": val_path, "test": test_path, "alpha": alpha,
           "auroc": {"global": [auroc(gv, yv), auroc(gt, yt)],
                     "routed": [auroc(rv, yv), auroc(rt, yt)]},
           "aurc": {"global": [aurc(gv, yv), aurc(gt, yt)],
                    "routed": [aurc(rv, yv), aurc(rt, yt)]},
           "frontier": frontier, "gate": out}

    # strict: the delta=0.10 row must EXIST and hold at >=50% coverage; an absent row is a FAIL.
    row10 = next((o for o in out if abs(o["delta"] - 0.10) < 1e-9), None)
    auroc_ok = auroc(rv, yv) >= 0.70 and auroc(rt, yt) >= 0.70
    gate_ok = bool(row10) and row10["held"] and row10["test_coverage"] >= 0.50
    held_deltas = [o["delta"] for o in out if o["held"]]
    print(f"\n  AUROC condition (routed >= 0.70 on val AND test)      : {'PASS' if auroc_ok else 'FAIL'}")
    print(f"  10%%-budget condition (held at coverage >= 50%%)        : "
          f"{'PASS' if gate_ok else 'FAIL'}"
          + ("" if row10 else "  (no feasible threshold on val at alpha=%.2f)" % alpha))
    print(f"  strongest certifiable budget held on test             : "
          f"{min(held_deltas) if held_deltas else 'none'}")
    print(f"  STAGE-0 PASS CONDITION                                : "
          f"{'PASS' if auroc_ok and gate_ok else 'FAIL'}")
    return res


def main():
    opt = {sys.argv[i]: sys.argv[i + 1] for i, a in enumerate(sys.argv)
           if a.startswith("--") and i + 1 < len(sys.argv)}
    if "--val" not in opt or "--test" not in opt:
        print(__doc__)
        return
    res = report(opt["--val"], opt["--test"], alpha=float(opt.get("--alpha", 0.1)))
    if "--json" in opt:
        json.dump(res, open(opt["--json"], "w"), indent=1)
        print(f"\nwrote {opt['--json']}")


if __name__ == "__main__":
    main()
