#!/usr/bin/env python3
"""Calibration and selective-risk analysis of the current model (v4 ckpt 2482), from saved artifacts only (2026-11-12).

Inputs (no model run):
  AC eval rows with probabilities  eval-<tag>-VA-val.jsonl   (conf = p(action), el_conf = p(element))
  AC per-step InfiGUI verdicts      perstep-<tag>.jsonl        (exact_match = the scorer of record)
  validation eval rows              eval-..._validation_v4-VA-val.jsonl (correct = eval_d1 scorer)
  AndroidWorld traces               <run>/episodes/*.jsonl     (step confidence, result row)

Scores:
  p_act   = p(action)
  product = p(action) x p(element) on steps whose predicted action uses an element, else p(action)
  min     = min(p(action), p(element)) on those steps, else p(action)

Outputs JSON with AUROC, ECE (15 equal-width bins), risk at fixed coverage, AURC, and a split-conformal check: episodes
of the test set are split 50/50 at random (200 seeds); on one half the lowest threshold whose Clopper-Pearson upper
bound on the error rate of the accepted steps is <= delta is chosen; on the other half coverage and risk are measured.

    python3 calib_current.py OUT.json
"""
import glob
import json
import math
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
AC = os.path.join(ROOT, "benchmarks", "results", "2026-11-12-v4", "actest")
AW = os.path.join(ROOT, "benchmarks", "results", "2026-11-12-maskenv")
AW_DIRS = ["aw116_v4g_s0", "aw116_v4g_s0_b", "aw116_v4g_s0_c"]
SPATIAL = {"click", "long_press", "write", "type_text"}


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties count half. labels: 1 = correct."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        rank_sum += avg * sum(l for _, l in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece(scores, labels, bins=15):
    tot, n = 0.0, len(scores)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [k for k, s in enumerate(scores) if (lo <= s < hi) or (b == bins - 1 and s == 1.0)]
        if idx:
            conf = sum(scores[k] for k in idx) / len(idx)
            acc = sum(labels[k] for k in idx) / len(idx)
            tot += len(idx) / n * abs(conf - acc)
    return tot


def risk_at_coverage(scores, labels, cov):
    order = sorted(range(len(scores)), key=lambda k: -scores[k])
    m = max(1, int(round(cov * len(order))))
    kept = order[:m]
    return 1.0 - sum(labels[k] for k in kept) / m


def aurc(scores, labels):
    order = sorted(range(len(scores)), key=lambda k: -scores[k])
    err, tot = 0, 0.0
    for m, k in enumerate(order, 1):
        err += 1 - labels[k]
        tot += err / m
    return tot / len(order)


def pick_threshold(scores, labels, delta, alpha=0.05):
    """Lowest threshold (largest coverage) whose accepted set has a one-sided Clopper-Pearson upper bound on its
    error rate <= delta (level alpha), over accepted sets of at least 30 steps. None if no such threshold."""
    import numpy as np
    from scipy.stats import beta
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(-s, kind="stable")
    err = np.cumsum(1 - y[order])
    m = np.arange(1, len(order) + 1)
    ub = np.where(err >= m, 1.0, beta.ppf(1 - alpha, err + 1, m - err))
    # fixed-sequence testing (Learn-then-Test): walk from the most confident steps down, stop at the first
    # accepted-set size >= 30 whose bound exceeds delta. Taking the largest passing set instead is not valid.
    last = None
    for i in range(29, len(order)):
        if ub[i] > delta:
            break
        last = i
    return None if last is None else float(s[order][last])


def summarize(scores, labels):
    n = len(labels)
    return {"n": n, "accuracy": sum(labels) / n, "auroc": auroc(scores, labels), "ece": ece(scores, labels),
            "mean_conf": sum(scores) / n, "aurc": aurc(scores, labels),
            "risk_at_cov": {str(c): risk_at_coverage(scores, labels, c) for c in (0.25, 0.5, 0.75, 1.0)}}


def load_ac(tag):
    ev = {}
    for l in open(os.path.join(AC, f"eval-{tag}-VA-val.jsonl")):
        r = json.loads(l)
        ev[(r["episode_id"], r["step"])] = r
    rows = []
    for l in open(os.path.join(AC, f"perstep-{tag}.jsonl")):
        p = json.loads(l)
        e = ev.get((p["episode_id"], p["step"]))
        if e is None:
            continue
        rows.append({"ep": p["episode_id"], "pred": e["pred"], "p_act": e["conf"], "p_el": e.get("el_conf"),
                     "ok": int(bool(p["exact_match"])), "type_ok": int(bool(p["type_match"]))})
    return rows


def scores_of(rows, kind):
    out = []
    for r in rows:
        sp = r["pred"] in SPATIAL and r["p_el"] is not None
        if kind == "p_act" or not sp:
            out.append(r["p_act"])
        elif kind == "product":
            out.append(r["p_act"] * r["p_el"])
        else:
            out.append(min(r["p_act"], r["p_el"]))
    return out


def conformal(rows, kind, deltas=(0.10, 0.15, 0.20), seeds=200, unit="episode"):
    """unit="episode": split whole episodes (steps of one episode stay together); unit="step": split steps."""
    if unit == "step":
        rows = [dict(r, ep=k) for k, r in enumerate(rows)]
    eps = sorted({r["ep"] for r in rows})
    sc = scores_of(rows, kind)
    out = {}
    for d in deltas:
        covs, risks, viol, none = [], [], 0, 0
        for s in range(seeds):
            rnd = random.Random(s)
            cal = set(rnd.sample(eps, len(eps) // 2))
            ci = [k for k, r in enumerate(rows) if r["ep"] in cal]
            ti = [k for k, r in enumerate(rows) if r["ep"] not in cal]
            t = pick_threshold([sc[k] for k in ci], [rows[k]["ok"] for k in ci], d)
            if t is None:
                none += 1
                continue
            acc = [k for k in ti if sc[k] >= t]
            cov = len(acc) / len(ti)
            risk = 1 - sum(rows[k]["ok"] for k in acc) / len(acc) if acc else 0.0
            covs.append(cov)
            risks.append(risk)
            viol += risk > d
        out[str(d)] = {"seeds": seeds, "no_threshold": none,
                       "coverage_mean": sum(covs) / len(covs) if covs else None,
                       "risk_mean": sum(risks) / len(risks) if risks else None,
                       "violation_rate": viol / len(covs) if covs else None}
    return out


def transfer(val_rows, test_rows, kind, deltas=(0.10, 0.15, 0.20)):
    """Threshold fitted on the validation split, applied to the test split (a distribution shift check)."""
    sv, st = scores_of(val_rows, kind), scores_of(test_rows, kind)
    out = {}
    for d in deltas:
        t = pick_threshold(sv, [r["ok"] for r in val_rows], d)
        if t is None:
            out[str(d)] = None
            continue
        acc = [k for k in range(len(test_rows)) if st[k] >= t]
        out[str(d)] = {"threshold": t, "test_coverage": len(acc) / len(test_rows),
                       "test_risk": 1 - sum(test_rows[k]["ok"] for k in acc) / len(acc) if acc else None}
    return out


def load_val():
    rows = []
    for l in open(os.path.join(AC, "eval-v4_s0_ckpt_2482_validation_v4-VA-val.jsonl")):
        e = json.loads(l)
        rows.append({"ep": e["episode_id"], "pred": e["pred"], "p_act": e["conf"], "p_el": e.get("el_conf"),
                     "ok": int(bool(e["correct"]))})
    return rows


def episode_level(rows, kind):
    sc = scores_of(rows, kind)
    by = {}
    for k, r in enumerate(rows):
        by.setdefault(r["ep"], []).append((sc[k], r["ok"]))
    s = [min(x for x, _ in v) for v in by.values()]
    y = [int(all(o for _, o in v)) for v in by.values()]
    return {"episodes": len(y), "all_right": sum(y), "auroc_min_conf": auroc(s, y)}


def load_aw():
    eps = []
    for d in AW_DIRS:
        for f in sorted(glob.glob(os.path.join(AW, d, "episodes", "*.jsonl"))):
            if ".abandoned." in f:
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            res = [r for r in rows if r.get("type") == "result"]
            if not res or res[-1].get("unscored") or res[-1].get("setup_error") or res[-1].get("score") is None:
                continue
            steps = [r for r in rows if r.get("type") == "step"]
            eps.append({"task": res[-1]["task_name"], "score": float(res[-1]["score"]), "steps": steps,
                         "declared": bool(res[-1].get("declared_done"))})
    return eps


def aw_analysis():
    eps = load_aw()
    out = {"tasks": len(eps), "score_sum": sum(e["score"] for e in eps)}
    done_conf, done_ok = [], []
    min_conf, succ = [], []
    step_conf, step_eff = [], []
    for e in eps:
        st = e["steps"]
        if not st:
            continue
        confs = [s["action_raw"]["confidence"] for s in st if s["action_raw"].get("confidence") is not None]
        min_conf.append(min(confs))
        succ.append(int(e["score"] >= 1.0))
        last = st[-1]
        if e["declared"] and last["action_raw"]["action"] in ("status", "answer"):
            done_conf.append(last["action_raw"]["confidence"])
            done_ok.append(int(e["score"] > 0))
        for s in st:
            if s.get("screen_changed") is not None and s["action_raw"]["action"] not in ("wait", "status", "answer"):
                step_conf.append(s["action_raw"]["confidence"])
                step_eff.append(int(bool(s["screen_changed"])))
    out["episode_min_conf_auroc_success"] = auroc(min_conf, succ)
    out["declared_done"] = {"n": len(done_ok), "right": sum(done_ok), "auroc": auroc(done_conf, done_ok),
                            "conf_right_mean": (sum(c for c, y in zip(done_conf, done_ok) if y) / max(1, sum(done_ok))),
                            "conf_wrong_mean": (sum(c for c, y in zip(done_conf, done_ok) if not y)
                                                / max(1, len(done_ok) - sum(done_ok)))}
    # ask gate on the termination decision: ask when p(done) < tau
    gate = []
    for tau in (0.9, 0.95, 0.98, 0.99, 0.995):
        asked = [y for c, y in zip(done_conf, done_ok) if c < tau]
        gate.append({"tau": tau, "asked": len(asked), "wrong_caught": len(asked) - sum(asked),
                     "right_asked": sum(asked), "wrong_total": len(done_ok) - sum(done_ok)})
    out["done_ask_gate"] = gate
    out["step_conf_vs_screen_changed"] = {"n": len(step_eff), "changed": sum(step_eff),
                                          "auroc": auroc(step_conf, step_eff)}
    return out


def main(out_path):
    res = {}
    val = load_val()
    for tag, name in (("v4_s0_ckpt_2482_actest_v4", "ac_high"), ("v4_s0_ckpt_2482_actest_v4_low", "ac_low")):
        if not os.path.exists(os.path.join(AC, f"eval-{tag}-VA-val.jsonl")):
            res[name] = "eval rows with probabilities not on disk"
            continue
        rows = load_ac(tag)
        r = {"n": len(rows)}
        for kind in ("p_act", "product", "min"):
            r[kind] = summarize(scores_of(rows, kind), [x["ok"] for x in rows])
        r["type_p_act"] = {"auroc": auroc([x["p_act"] for x in rows], [x["type_ok"] for x in rows]),
                           "ece": ece([x["p_act"] for x in rows], [x["type_ok"] for x in rows])}
        r["conformal_episode_split_product"] = conformal(rows, "product")
        r["conformal_episode_split_p_act"] = conformal(rows, "p_act")
        r["conformal_step_split_p_act"] = conformal(rows, "p_act", unit="step")
        r["transfer_val_to_test_p_act"] = transfer(val, rows, "p_act")
        r["transfer_val_to_test_product"] = transfer(val, rows, "product")
        r["episode_level_product"] = episode_level(rows, "product")
        res[name] = r
    res["validation"] = {k: summarize(scores_of(val, k), [x["ok"] for x in val]) for k in ("p_act", "product", "min")}
    res["androidworld"] = aw_analysis()
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main(sys.argv[1])
