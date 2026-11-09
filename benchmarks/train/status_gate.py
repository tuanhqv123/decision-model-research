"""Stage-2 diagnostic: can a calibrated progress gate remove premature `status` for free?

The model terminates early 21 times on the 398 test steps, and every one of those is a type error
(gold is a UI action). Suppressing `status` alone cannot help: the row is wrong either way. The
question this script answers is whether switching to the best *expressible* non-status action,
gated by p(status), recovers rows without losing correct terminations.

Rule (one parameter, fitted on val, evaluated once on test):
    if pred == status and p(status) < tau:  emit argmax over {click, long_press, navigate_back, wait}

Those four are the actions this rule can emit without any text/direction readout, which is what keeps
val and test on the same footing (the val dump has no readout slots).

Usage:
  python3 benchmarks/train/status_gate.py \
    --val backup-2026-10-29/Model_Training_Server/acwork/runs/E2-val/eval-E2final-V-val.jsonl \
    --test backup-2026-11-06/offline/e2-pure-hist.jsonl \
    --rows benchmarks/data/acv2/500_steps_filtered.json \
    --valdata backup-2026-10-29/prd-survey-x1-5090/acwork/acdata
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from run_acv2 import score_row  # noqa: E402

EMITTABLE = ("click", "long_press", "navigate_back", "wait")
SPATIAL = ("click", "long_press")
# The val dump keeps the checkpoint's native option names; the regenerated test dump uses eval names.
NAME = {"terminate": "status", "write": "type_text"}


def norm_probs(pred):
    return {NAME.get(k, k): v for k, v in (pred.get("probs") or {}).items()}


def load_pred(path):
    return [json.loads(l) for l in open(path)]


def load_test_rows(path):
    return {("%s|%s" % (r["episode_id"], r["step"])): r for r in json.load(open(path))}


def load_val_rows(path):
    from eval_d1 import val_row

    if os.path.isdir(path):
        path = os.path.join(path, "compact_validation.jsonl")
    out = {}
    for line in open(path):
        rec = json.loads(line)
        row = val_row(rec, "")
        out[("%s|%s" % (row["episode_id"], row["step"]))] = row
    return out


def switch_decision(pred, tau):
    """Return the decision this rule would emit for one row (type, pt) or None to keep `pred`."""
    probs = norm_probs(pred)
    p_status = float(probs.get("status", 0.0))
    if pred["pred"] != "status" or p_status >= tau:
        return None
    best, best_p = None, -1.0
    for t in EMITTABLE:
        p = float(probs.get(t, 0.0))
        if p > best_p:
            best, best_p = t, p
    if best is None:
        return None
    pt = pred.get("box_pt") if best in SPATIAL else None
    return best, pt


def evaluate(preds, rows, tau):
    """Accuracy under the rule, plus the per-row correctness vector (for paired tests)."""
    vec, details = [], 0.0
    for p in preds:
        key = "%s|%s" % (p["episode_id"], p["step"])
        row = rows.get(key)
        if row is None:
            raise SystemExit("row missing for %s" % key)
        sw = switch_decision(p, tau)
        if sw is None:
            ok = bool(p["correct"])
            vec.append(ok)
            continue
        t, pt = sw
        ok, _ = score_row(row, t, None, pt)
        vec.append(bool(ok))
    return float(np.mean(vec)), vec


def per_gold_type(preds, rows, tau):
    out = {}
    for p in preds:
        g = p["gold"]["action_type"]
        key = "%s|%s" % (p["episode_id"], p["step"])
        sw = switch_decision(p, tau)
        if sw is None:
            ok = bool(p["correct"])
        else:
            t, pt = sw
            ok, _ = score_row(rows[key], t, None, pt)
        c, n = out.get(g, (0, 0))
        out[g] = (c + int(ok), n + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--rows", default="benchmarks/data/acv2/500_steps_filtered.json")
    ap.add_argument("--valdata", default="backup-2026-10-29/prd-survey-x1-5090/acwork/acdata")
    ap.add_argument("--out", default="benchmarks/results/2026-11-07-status-gate.json")
    a = ap.parse_args()

    val_p, test_p = load_pred(a.val), load_pred(a.test)
    val_rows, test_rows = load_val_rows(a.valdata), load_test_rows(a.rows)

    base_val = float(np.mean([p["correct"] for p in val_p]))
    base_test = float(np.mean([p["correct"] for p in test_p]))
    n_status_val = sum(1 for p in val_p if p["pred"] == "status")
    n_status_test = sum(1 for p in test_p if p["pred"] == "status")

    grid = [round(x, 2) for x in np.arange(0.30, 0.96, 0.01)]
    curve = []
    for tau in grid:
        acc, _ = evaluate(val_p, val_rows, tau)
        curve.append((tau, acc))
    best_tau, best_acc = max(curve, key=lambda x: (x[1], -x[0]))

    _, vec_base_v = evaluate(val_p, val_rows, 0.0)  # never switches (tau=0 keeps everything)
    _, vec_v = evaluate(val_p, val_rows, best_tau)
    acc_t, vec_t = evaluate(test_p, test_rows, best_tau)
    _, vec_base_t = evaluate(test_p, test_rows, 0.0)

    val_delta = float(np.mean(vec_v) - np.mean(vec_base_v))
    test_delta = float(np.mean(vec_t) - np.mean(vec_base_t))
    flipped_v = int(np.sum(np.asarray(vec_v) != np.asarray(vec_base_v)))
    gained_v = int(np.sum(np.asarray(vec_v) > np.asarray(vec_base_v)))
    lost_v = int(np.sum(np.asarray(vec_v) < np.asarray(vec_base_v)))

    # McNemar exact on test discordant pairs
    from math import comb

    b = sum(1 for x, y in zip(vec_base_t, vec_t) if (not x) and y)
    c = sum(1 for x, y in zip(vec_base_t, vec_t) if x and (not y))
    n = b + c
    p_mcnemar = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n))

    out = {
        "protocol": "AC-v2 398-step High, pure (goal + interaction history), E2@938",
        "rule": "if pred=status and p(status)<tau -> argmax{click,long_press,navigate_back,wait}",
        "base": {"val": base_val, "test": base_test, "n_val": len(val_p), "n_test": len(test_p)},
        "status_preds": {"val": n_status_val, "test": n_status_test},
        "best_tau": best_tau,
        "val": {"acc": float(np.mean(vec_v)), "delta": val_delta, "gained": gained_v, "lost": lost_v, "flipped": flipped_v},
        "test": {"acc": acc_t, "delta": test_delta, "gained": b, "lost": c, "mcnemar_p": p_mcnemar},
        "per_gold_type_val": {k: {"acc": c / n, "n": n} for k, (c, n) in per_gold_type(val_p, val_rows, best_tau).items()},
        "per_gold_type_val_base": {k: {"acc": c / n, "n": n} for k, (c, n) in per_gold_type(val_p, val_rows, 0.0).items()},
        "per_gold_type_test": {k: {"acc": c / n, "n": n} for k, (c, n) in per_gold_type(test_p, test_rows, best_tau).items()},
        "per_gold_type_test_base": {k: {"acc": c / n, "n": n} for k, (c, n) in per_gold_type(test_p, test_rows, 0.0).items()},
        "curve_val": curve,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "curve_val"}, indent=1))


if __name__ == "__main__":
    main()
