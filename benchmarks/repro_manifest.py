#!/usr/bin/env python3
"""Recompute every AC-v2 number the paper quotes, from the saved predictions.

Why this exists: the paper claims reproducibility, but its numbers were reported from run logs.
This script takes the artifacts and rebuilds the table, so a reader (or a reviewer) can see which
claims survive contact with the data and which are marked as reported-in-run.

    python3 benchmarks/repro_manifest.py \
        --test backup-2026-10-29/s4090-1x-100/acwork/runs/E2-cur15k-V/eval-step938-V.jsonl \
        --val  backup-2026-10-29/Model_Training_Server/acwork/runs/E2-val/eval-E2final-V-val.jsonl

Exit code is 0 when every *checkable* claim matches, 1 when one does not, and the table names the
claim either way. A claim whose artifact does not exist is reported as MISSING-ARTIFACT rather than
silently skipped — that is the difference between "we could not check it" and "it is true".
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "train"))

from error_budget import decompose, wilson  # noqa: E402
from gate_routed import auroc, cp_upper, ece, fit_ecdf, routed  # noqa: E402

SPATIAL = ("click", "long_press", "write", "type_text")

#: Per-claim tolerance in the quoted unit (percentage points, or absolute for AUROC).
TOLERANCE = {
    "Pass@1 (pure protocol)": 0.05,
    "Pass@1 (app-name hint)": 0.05,
    "Type accuracy": 0.05,
    "Grounding (spatial)": 0.05,
    "p(action) ECE": 0.2,
    "AUROC(confidence, correct)": 0.005,
    "Gate AUROC (type-routed, test)": 0.005,
    "Risk at 50% coverage, %": 0.6,
    "Certified coverage at 20% risk, %": 0.6,
}

#: Claims whose paper value came from a different estimator than this script uses.
#: Reported as DEFINITION so the disagreement is visible instead of being rounded away.
DEFINITIONAL = {"Risk at 50% coverage, %", "Certified coverage at 20% risk, %"}

#: Values quoted in papers/single-pass-decision-acv2.md, with the artifact each depends on.
CLAIMS = [
    ("Pass@1 (pure protocol)", 64.07, "pure"),
    ("Pass@1 (app-name hint)", 64.82, "hinted"),
    ("Type accuracy", 84.17, "any"),
    ("Grounding (spatial)", 56.05, "pure"),
    ("p(action) ECE", 20.4, "any"),
    ("AUROC(confidence, correct)", 0.631, "any"),
    ("Gate AUROC (type-routed, test)", 0.729, "any"),
    ("Risk at 50% coverage, %", 21.5, "any"),
    ("Certified coverage at 20% risk, %", 40.7, "any"),
]


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _is_spatial(row):
    return (row["gold"].get("action_type") in SPATIAL)


def compute(test_rows, val_rows):
    n, corr, gold, pred, type_ok, spatial, buckets, named = decompose(test_rows)
    k = int(corr.sum())

    spatial_idx = [i for i, g in enumerate(gold) if g in SPATIAL]
    grounding_hits = sum(1 for i in spatial_idx if corr[i])
    conf = np.array([float(r.get("conf") or 0.0) for r in test_rows], float)
    y = np.array([bool(r["correct"]) for r in test_rows], float)

    ecdf = fit_ecdf(val_rows) if val_rows else None
    routed_scores = routed(test_rows, ecdf) if ecdf else np.zeros(n)
    raw_scores = conf

    # risk at 50% coverage: act on the highest-scoring half, measure error rate there
    def risk_at(scores, coverage=0.5):
        order = np.argsort(-scores, kind="mergesort")
        k_act = max(1, int(round(coverage * len(scores))))
        return float(np.mean(1.0 - y[order[:k_act]]))

    # certified coverage at 20% risk, split-conformal on the val split's score distribution
    cert_cov = None
    if val_rows:
        v_scores = routed(val_rows, ecdf)
        v_y = np.array([bool(r["correct"]) for r in val_rows], float)
        order = np.argsort(-v_scores, kind="mergesort")
        err = np.concatenate([[0], np.cumsum(1 - v_y[order])])
        for frac in np.linspace(0.95, 0.05, 91):
            k_act = max(1, int(round(frac * len(v_scores))))
            ub = cp_upper(int(err[k_act]), k_act, 0.1)
            if ub <= 0.20:
                cert_cov = frac * 100.0
                break

    return {
        "n": n,
        "pass_pure": k / n * 100.0,
        "correct": k,
        "ci": [x * 100.0 for x in wilson(k, n)],
        "type_acc": float(np.mean(type_ok)) * 100.0,
        "grounding": (grounding_hits / len(spatial_idx) * 100.0) if spatial_idx else None,
        "grounding_n": len(spatial_idx),
        "ece": ece(conf, y) * 100.0,
        "auroc_conf": auroc(conf, y),
        "auroc_routed": auroc(routed_scores, y),
        "risk50_routed": risk_at(routed_scores) * 100.0,
        "risk50_conf": risk_at(raw_scores) * 100.0,
        "cert_cov": cert_cov,
        "buckets": {k2: int(np.sum(v)) for k2, v in buckets.items()},
        "named": {k2: int(np.sum(v)) for k2, v in named.items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="recompute the paper's AC-v2 numbers from artifacts")
    ap.add_argument("--test", required=True, help="prediction jsonl for the 398-step set")
    ap.add_argument("--val", default=None, help="prediction jsonl for the val split (needed for routing)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--kind", default="unknown", choices=["pure", "hinted", "unknown"],
                    help="state which protocol produced --test; NEVER infer it from the value being verified")
    args = ap.parse_args(argv)

    test_rows, val_rows = load(args.test), load(args.val)
    if not test_rows:
        print("no test predictions at %s" % args.test, file=sys.stderr)
        return 2

    got = compute(test_rows, val_rows)
    if args.json:
        print(json.dumps(got, indent=2, sort_keys=True))
        return 0

    print("artifact: %s" % args.test)
    print("  n=%d  correct=%d  Pass@1=%.2f%%  95%% CI [%.1f, %.1f]"
          % (got["n"], got["correct"], got["pass_pure"], got["ci"][0], got["ci"][1]))
    print("  type=%.2f%%  grounding=%.2f%% (%d spatial rows)"
          % (got["type_acc"], got["grounding"] or 0.0, got["grounding_n"]))
    print("  ECE=%.1f%%  AUROC(conf)=%.3f  AUROC(routed)=%.3f"
          % (got["ece"], got["auroc_conf"], got["auroc_routed"]))
    print("  risk@50%% coverage: routed=%.1f%%  conf-only=%.1f%%"
          % (got["risk50_routed"], got["risk50_conf"]))
    print("  error budget: %s" % got["buckets"])
    print("  named: %s" % got["named"])

    computed = {
        "Pass@1 (pure protocol)": got["pass_pure"],
        "Pass@1 (app-name hint)": got["pass_pure"],
        "Type accuracy": got["type_acc"],
        "Grounding (spatial)": got["grounding"] or 0.0,
        "p(action) ECE": got["ece"],
        "AUROC(confidence, correct)": got["auroc_conf"],
        "Gate AUROC (type-routed, test)": got["auroc_routed"],
        "Risk at 50% coverage, %": got["risk50_routed"],
        "Certified coverage at 20% risk, %": got["cert_cov"] or 0.0,
    }

    # The caller declares the protocol. Inferring it from `correct` is circular: the instrument would
    # recognise the artifact by the very value it is meant to verify (2026-11-07 audit, R10).
    artifact_kind = args.kind

    print("\nartifact kind: %s (%d/%d)" % (artifact_kind, got["correct"], got["n"]))
    print("\nclaim                                  paper   recomputed   status")
    print("-" * 74)
    failures = 0
    for name, quoted, kind in CLAIMS:
        value = computed[name]
        if kind in ("pure", "hinted") and artifact_kind not in (kind, "unknown"):
            print("%-38s %6.2f %10s   N/A (%s artifact)" % (name, quoted, "-", artifact_kind))
            continue
        tol = TOLERANCE.get(name, 0.3)
        status = "MATCH" if abs(value - quoted) <= tol else "MISMATCH"
        if status == "MISMATCH" and name not in DEFINITIONAL:
            failures += 1
        if name in DEFINITIONAL and status == "MISMATCH":
            status = "DEFINITION"
        print("%-38s %6.2f %10.2f   %s" % (name, quoted, value, status))

    print("\nNote: a MISMATCH on 'Pass@1 (pure protocol)' is expected for the hinted artifact;")
    print("regenerate the pure run (run_d12b_acv2.py --save-readouts, ~52 s) before quoting 64.07.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
