"""Per-class logit adjustment for the action-type readout, fitted on the calibration split.

Stage-2 diagnosis (2026-11-03) found that 16 of the 19 gold-`scroll` errors are TYPE errors
(the model says `status` 9x or `click` 7x) and only 3 are direction errors. That is the classic
rare-class under-prediction signature, and the standard post-hoc repair is an additive per-class
logit offset (Menon et al., arXiv 2007.07314), fitted on held-out data.

This measures TYPE accuracy only. Converting a type gain into a step-accuracy gain also needs the
`scroll_dir` / `type_text` readouts for the switched-to class, which are not in the saved jsonl —
so the step-level version of this experiment needs a model re-run (see the note's blocker).

Usage: python class_bias.py [--val ...] [--test ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_eval import load, mcnemar_exact  # noqa: E402

OPT = ["click", "write", "long_press", "scroll", "wait", "navigate_back", "open_app", "terminate"]
OUT = {"write": "type_text", "terminate": "status"}
GRID = np.arange(-1.5, 1.51, 0.25)


def to_eval(name):
    return OUT.get(name, name)


def build(rows):
    P = np.array([[float((r.get("probs") or {}).get(k, 0.0) or 0.0) for k in OPT] for r in rows])
    P = np.clip(P, 1e-9, None)
    P = P / P.sum(1, keepdims=True)
    y = np.array([to_eval(r["gold"]["action_type"]) for r in rows])
    base = np.array([to_eval(str(OPT[i])) for i in P.argmax(1)])
    return np.log(P), y, base


def acc(pred, y):
    return float((pred == y).mean())


def fit_bias(L, y, passes=3):
    """Coordinate ascent on per-class additive logit offsets, maximising calibration-split accuracy."""
    b = np.zeros(len(OPT))
    labels = [to_eval(o) for o in OPT]
    cur = acc(np.array(labels)[np.argmax(L + b, 1)], y)
    for _ in range(passes):
        improved = False
        for j in range(len(OPT)):
            best_b, best_a = b[j], cur
            for g in GRID:
                b[j] = g
                a = acc(np.array(labels)[np.argmax(L + b, 1)], y)
                if a > best_a + 1e-12:
                    best_b, best_a = g, a
            b[j] = best_b
            if best_a > cur:
                cur, improved = best_a, True
        if not improved:
            break
    return b, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="benchmarks/results/2026-10-29-evals/E2-val.jsonl")
    ap.add_argument("--test", default="benchmarks/results/2026-10-29-evals/E2-938-V.jsonl")
    ap.add_argument("--out")
    a = ap.parse_args()

    vrows, trows = list(load(a.val).values()), list(load(a.test).values())
    Lv, yv, basev = build(vrows)
    Lt, yt, baset = build(trows)
    labels = [to_eval(o) for o in OPT]

    b, val_acc = fit_bias(Lv, yv)
    print(f"\n=== per-class logit adjustment (type only)")
    print(f"  fitted on {a.val.split('/')[-1]} (n={len(yv)})")
    for k, (name, v) in enumerate(zip(labels, b)):
        if abs(v) > 1e-9:
            print(f"    {name:14s} bias {v:+.2f}")
    pred_t = np.array(labels)[np.argmax(Lt + b, 1)]
    print(f"\n  {'':18s} {'val':>7s} {'test':>7s}")
    print(f"  {'unadjusted':18s} {acc(basev, yv):7.4f} {acc(baset, yt):7.4f}")
    print(f"  {'adjusted':18s} {val_acc:7.4f} {acc(pred_t, yt):7.4f}")

    b_ok = int((baset == yt).sum())
    c_ok = int((pred_t == yt).sum())
    bb = sum(1 for x, y in zip(baset == yt, pred_t == yt) if x and not y)
    cc = sum(1 for x, y in zip(baset == yt, pred_t == yt) if not x and y)
    p = mcnemar_exact(bb, cc)
    print(f"\n  test type errors: {len(yt)-b_ok} -> {len(yt)-c_ok}   (b={bb} regressed, c={cc} fixed)")
    print(f"  McNemar exact p = {p:.4g}")
    print(f"  worst-case step gain if every fixed type row also grounds correctly: "
          f"+{100*cc/len(yt):.2f} pts; net of regressions +{100*(cc-bb)/len(yt):.2f} pts")
    print("\n  BLOCKED (step-level): switching a row INTO `scroll`/`type_text` needs that class's "
          "direction/text readout, which is not in the saved jsonl. Needs a model re-run.")

    if a.out:
        json.dump({"bias": dict(zip(labels, map(float, b))),
                   "val_type_acc": [acc(basev, yv), val_acc],
                   "test_type_acc": [acc(baset, yt), acc(pred_t, yt)],
                   "test_fixed": cc, "test_regressed": bb, "mcnemar_p": p}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
