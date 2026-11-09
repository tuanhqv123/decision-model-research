"""Error-budget decomposition of an AC-v2 eval jsonl: where the misses actually are, and what fixing
each bucket is worth in points. Four buckets, plus the named slices worth attacking directly.

A  spatial gold, type correct, element wrong   (grounding)
B  spatial gold, type wrong
C1 non-spatial gold, type wrong
C2 non-spatial gold, type correct, value wrong  (e.g. open_app with the wrong app name)

Also prints oracle ceilings, so every later training stage has a pre-committed worth, and per-gold-type
accuracy next to the always-status baseline.

Usage: python error_budget.py results/eval-<tag>-V.jsonl [more.jsonl ...] [--json out.json]
"""
import json
import sys

import numpy as np

SPATIAL = {"click", "long_press", "type_text"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def decompose(rows):
    n = len(rows)
    corr = np.array([bool(r["correct"]) for r in rows])
    gold = [r["gold"]["action_type"] for r in rows]
    pred = [r["pred"] for r in rows]
    type_ok = np.array([p == g for p, g in zip(pred, gold)])
    spatial = np.array([g in SPATIAL for g in gold])

    buckets = {
        "A_spatial_type_ok_element_wrong": spatial & type_ok & ~corr,
        "B_spatial_type_wrong": spatial & ~type_ok,
        "C1_nonspatial_type_wrong": ~spatial & ~type_ok,
        "C2_nonspatial_value_wrong": ~spatial & type_ok & ~corr,
    }
    named = {
        "premature_status (acted as done, should act)": [p == "status" and g != "status" for p, g in zip(pred, gold)],
        "spurious_navigate_back": [p == "navigate_back" and g != "navigate_back" for p, g in zip(pred, gold)],
        "gold_scroll_missed": [g == "scroll" and not c for g, c in zip(gold, corr)],
        "open_app_wrong_value": [g == "open_app" and p == "open_app" and not c for g, p, c in zip(gold, pred, corr)],
        "gold_click_missed": [g == "click" and not c for g, c in zip(gold, corr)],
    }
    return n, corr, gold, pred, type_ok, spatial, buckets, named


def report(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n, corr, gold, pred, type_ok, spatial, buckets, named = decompose(rows)
    k = int(corr.sum())
    lo, hi = wilson(k, n)
    out = {"path": path, "n": n, "correct": k, "acc": k / n, "acc_ci95": [lo, hi]}

    print(f"\n=== {path}  (n={n})")
    print(f"step accuracy      : {100*k/n:.2f}%  ({k}/{n})  CI95 {100*lo:.1f}-{100*hi:.1f}")
    print(f"type accuracy      : {100*type_ok.mean():.2f}%   spatial gold rows: {int(spatial.sum())}")

    total_err = n - k
    print(f"\nerror budget ({total_err} errors is not one bucket):")
    cum = 0
    for name, m in buckets.items():
        c = int(np.sum(m))
        if c:
            print(f"  {name:34s} {c:4d} rows = {100*c/total_err:5.1f}% of errors   worth +{100*c/n:5.2f} pts")
            cum += c
    print(f"  {'sum of buckets':34s} {cum:4d} rows  (== {total_err} errors: {cum == total_err})")

    b = int(buckets["B_spatial_type_wrong"].sum())
    c1 = int(buckets["C1_nonspatial_type_wrong"].sum())
    c2 = int(buckets["C2_nonspatial_value_wrong"].sum())
    a = int(buckets["A_spatial_type_ok_element_wrong"].sum())
    print("\noracle ceilings (what each axis is worth if perfectly fixed):")
    print(f"  perfect grounding only      : {k+a:3d}/{n} = {100*(k+a)/n:5.2f}%   (+{100*a/n:.2f})")
    print(f"  perfect type only           : {k+b+c1:3d}/{n} = {100*(k+b+c1)/n:5.2f}%   (+{100*(b+c1)/n:.2f})")
    print(f"  grounding + type            : {k+a+b+c1:3d}/{n} = {100*(k+a+b+c1)/n:5.2f}%")
    print(f"  everything incl. values     : {k+a+b+c1+c2:3d}/{n} = {100*(k+a+b+c1+c2)/n:5.2f}%")
    print(f"  -> type errors {b+c1} vs grounding errors {a}: "
          f"{'TYPE IS THE BIGGER OR EQUAL BUCKET' if b+c1 >= a else 'grounding is bigger'}")

    print("\nnamed slices:")
    for name, m in named.items():
        c = int(np.sum(m))
        if c:
            print(f"  {name:46s} {c:3d} rows   worth +{100*c/n:5.2f} pts")

    print("\nper gold type:")
    for t in sorted(set(gold), key=lambda x: -sum(g == x for g in gold)):
        m = np.array([g == t for g in gold])
        print(f"  {t:14s} {int(corr[m].sum()):3d}/{int(m.sum()):3d} = {100*corr[m].mean():5.1f}%"
              f"   errors {int(m.sum())-int(corr[m].sum())}")

    maj = max(set(gold), key=gold.count)
    print(f"\nbaselines: always-{maj} {100*np.mean([g == maj for g in gold]):.2f}%"
          f" | always-status {100*np.mean([g == 'status' for g in gold]):.2f}%")

    out.update({
        "buckets": {k2: int(np.sum(v)) for k2, v in buckets.items()},
        "oracle": {"grounding_only": (k + a) / n, "type_only": (k + b + c1) / n,
                   "grounding_plus_type": (k + a + b + c1) / n, "perfect": (k + a + b + c1 + c2) / n},
        "named": {k2: int(np.sum(v)) for k2, v in named.items()},
        "per_type": {t: [int(corr[np.array([g == t for g in gold])].sum()),
                         int(sum(g == t for g in gold))] for t in sorted(set(gold))},
    })
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jout = None
    if "--json" in sys.argv:
        jout = sys.argv[sys.argv.index("--json") + 1]
        args = [a for a in args if a != jout]
    if not args:
        print(__doc__)
        return
    results = [report(p) for p in args]
    if jout:
        json.dump(results, open(jout, "w"), indent=1)
        print(f"\nwrote {jout}")


if __name__ == "__main__":
    main()
