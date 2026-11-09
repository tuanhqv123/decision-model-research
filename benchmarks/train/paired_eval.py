"""Paired comparison of two AC-v2 eval jsonl files over the SAME rows: McNemar exact test on the
discordant pairs, paired bootstrap CI on the delta, and per-gold-type deltas.

This is what makes a +2 pt gain claimable on 398 rows. Independent bootstrap CIs on each accuracy
cannot separate 64.32 from 64.30 (that is why the AC-v2 leaderboard top reads as a tie); a paired test
over the same gold rows can resolve a consistent shift.

Usage: python paired_eval.py BASE.jsonl CAND.jsonl [--boot 10000] [--base-name E2] [--cand-name X]
Self-test: python paired_eval.py A.jsonl A.jsonl   ->  delta 0, b=c=0, p=1.0
"""
import json
import math
import random
import sys

import numpy as np

SPATIAL = {"click", "long_press", "type_text"}


def load(path):
    rows = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        rows[(r["episode_id"], r["step"])] = r
    return rows


def align(base, cand):
    keys = sorted(set(base) & set(cand))
    if len(keys) != len(base) or len(keys) != len(cand):
        print(f"[warn] row sets differ: base={len(base)} cand={len(cand)} common={len(keys)}")
    b = [base[k] for k in keys]
    c = [cand[k] for k in keys]
    return keys, b, c


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b (base-only correct) and c (cand-only correct)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap(base_c, cand_c, clusters, n_boot=10000, seed=0):
    """Paired bootstrap of the delta, resampling CLUSTERS (episodes), not rows.

    Steps within an episode are correlated, so a row-level bootstrap understates the interval. We
    resample episodes with replacement and take every row of each sampled episode.
    """
    rng = random.Random(seed)
    by_cluster = {}
    for i, k in enumerate(clusters):
        by_cluster.setdefault(k, []).append(i)
    keys = list(by_cluster)
    n = len(base_c)
    deltas = []
    for _ in range(n_boot):
        idx = []
        for _ in range(len(keys)):
            idx.extend(by_cluster[keys[rng.randrange(len(keys))]])
        if not idx:
            continue
        deltas.append(np.mean([cand_c[i] - base_c[i] for i in idx]))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def report(base_path, cand_path, n_boot=10000, base_name="base", cand_name="cand"):
    base, cand = load(base_path), load(cand_path)
    keys, b, c = align(base, cand)
    n = len(keys)
    bc = np.array([int(x["correct"]) for x in b])
    cc = np.array([int(x["correct"]) for x in c])

    only_b = int(np.sum((bc == 1) & (cc == 0)))
    only_c = int(np.sum((bc == 0) & (cc == 1)))
    both = int(np.sum((bc == 1) & (cc == 1)))
    neither = int(np.sum((bc == 0) & (cc == 0)))
    delta = float(cc.mean() - bc.mean())
    clusters = [k[0] for k in keys]
    lo, hi = paired_bootstrap(bc, cc, clusters, n_boot)
    p = mcnemar_exact(only_b, only_c)

    print(f"\n=== paired comparison, n={n} common rows, {len(set(clusters))} episodes")
    print(f"  {base_name:22s} {int(bc.sum()):3d}/{n} = {100*bc.mean():6.2f}%")
    print(f"  {cand_name:22s} {int(cc.sum()):3d}/{n} = {100*cc.mean():6.2f}%")
    print(f"  delta                  {100*delta:+.2f} pts   episode-clustered bootstrap95 "
          f"[{100*lo:+.2f}, {100*hi:+.2f}]")
    print(f"  McNemar exact          b({base_name} only)={only_b}  c({cand_name} only)={only_c}  p={p:.4g}")
    print(f"  agreement              both={both}  neither={neither}  "
          f"discordant={only_b+only_c} ({100*(only_b+only_c)/n:.1f}% of rows)")
    verdict = ("candidate significantly better" if p < 0.05 and delta > 0 else
               "candidate significantly worse" if p < 0.05 and delta < 0 else
               "no significant difference")
    print(f"  verdict                {verdict}")

    gold = [x["gold"]["action_type"] for x in b]
    print("\n  per gold type (delta must be read bucket by bucket: a win can hide a trade):")
    print(f"    {'type':14s} {'base':>10s} {'cand':>10s} {'delta':>8s}   note")
    for t in sorted(set(gold)):
        m = np.array([g == t for g in gold])
        nb, nc = int(bc[m].sum()), int(cc[m].sum())
        tot = int(m.sum())
        flag = ""
        if nc - nb <= -1 and tot >= 5:
            flag = "<- REGRESSION"
        print(f"    {t:14s} {nb:4d}/{tot:<5d} {nc:4d}/{tot:<5d} {nc-nb:+8d}   {flag}")

    print("\n  per bucket:")
    for name, pred in [("spatial gold", lambda g: g in SPATIAL),
                       ("non-spatial gold", lambda g: g not in SPATIAL)]:
        m = np.array([pred(g) for g in gold])
        nb, nc = int(bc[m].sum()), int(cc[m].sum())
        print(f"    {name:18s} base {nb:3d}/{int(m.sum()):3d}  cand {nc:3d}/{int(m.sum()):3d}"
              f"  delta {nc-nb:+d} rows ({100*(nc-nb)/n:+.2f} pts)")

    return {"n": n, "base_acc": float(bc.mean()), "cand_acc": float(cc.mean()), "delta": delta,
            "ci95": [lo, hi], "mcnemar_p": p, "b": only_b, "c": only_c}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {sys.argv[i]: sys.argv[i + 1] for i, a in enumerate(sys.argv) if a.startswith("--") and i + 1 < len(sys.argv)}
    args = [a for a in args if a not in opt.values()]
    if len(args) < 2:
        print(__doc__)
        return
    report(args[0], args[1], int(opt.get("--boot", 10000)),
           opt.get("--base-name", "base"), opt.get("--cand-name", "cand"))


if __name__ == "__main__":
    main()
