#!/usr/bin/env python3
"""Numbers of the AndroidControl paper (2026-11-16), InfiGUI-R1 scorer, 8,444 test steps.

  DMA-2B (ours)  scored in AndroidControl's own action space (`--exclude-actions terminate`):
                 benchmarks/results/2026-11-16-ac-noterm/{eval,perstep}-ours_actest{_low,}_noterm*.jsonl
  d1-2B (base)   the recorded base evaluation (its default action space, "done" allowed; it answers "done" on some
                 steps): benchmarks/results/2026-11-12-base/perstep-base_actest{_low,}.jsonl and, for its step
                 confidence on AC-High, benchmarks/results/2026-11-12-calibration/conf-base-achigh.jsonl
Outputs JSON with Type / Grounding / SR / episode SR per setting, confidence metrics on AC-High (AUROC, ECE,
risk@50, AURC, type AUROC/ECE), episode-split conformal thresholds (200 seeds), the paired McNemar test, and
conf-{base,ours}-achigh-nt.jsonl for the figures.

    python3 paper_ac_final.py OUT.json
"""
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calib_current as C  # noqa: E402  (auroc, ece, risk_at_coverage, aurc, conformal, scores_of)

RES = os.path.join(HERE, "..", "results")
D = os.path.join(RES, "2026-11-16-ac-noterm")


def perstep(path):
    out = []
    for p in map(json.loads, open(path)):
        pred = p.get("pred")
        out.append({"ep": str(p["episode_id"]), "step": p["step"], "gt": p["gt"]["action"],
                    "pred": pred.get("action") if isinstance(pred, dict) else pred,
                    "ok": int(bool(p["exact_match"])), "type_ok": int(bool(p["type_match"]))})
    return out


def with_conf_ours(tag):
    ev = {(str(r["episode_id"]), r["step"]): r for r in map(json.loads, open(os.path.join(D, f"eval-{tag}-VA-val.jsonl")))}
    rows = perstep(os.path.join(D, f"perstep-{tag}.jsonl"))
    for r in rows:
        e = ev[(r["ep"], r["step"])]
        r["p_act"], r["p_el"] = e["conf"], e.get("el_conf")
    return rows


def with_conf_base():
    rows = []
    for c in map(json.loads, open(os.path.join(RES, "2026-11-12-calibration", "conf-base-achigh.jsonl"))):
        rows.append({"ep": str(c["episode_id"]), "step": c["step"], "pred": c["pred"], "p_act": c["p_action"],
                     "p_el": c.get("p_element"), "ok": int(bool(c["exact_match"])), "type_ok": int(bool(c["type_match"]))})
    return rows


def standard(rows):
    n = len(rows)
    g = [r for r in rows if r["type_ok"] and r["gt"] in ("click", "long_press")]
    eps = {}
    for r in rows:
        eps.setdefault(r["ep"], []).append(r["ok"])
    return {"steps": n, "type": round(100 * sum(r["type_ok"] for r in rows) / n, 2),
            "grounding": round(100 * sum(r["ok"] for r in g) / len(g), 2), "grounding_n": len(g),
            "sr": round(100 * sum(r["ok"] for r in rows) / n, 2), "correct": sum(r["ok"] for r in rows),
            "episode_sr": round(100 * sum(all(v) for v in eps.values()) / len(eps), 2),
            "episodes_all_right": sum(all(v) for v in eps.values()), "episodes": len(eps),
            "predicted_terminate": sum(r["pred"] in ("status", "terminate") for r in rows)}


def mcnemar(a, b):
    ka = {(r["ep"], r["step"]): r["ok"] for r in a}
    kb = {(r["ep"], r["step"]): r["ok"] for r in b}
    x = sum(ka[k] and not kb[k] for k in ka)
    y = sum(kb[k] and not ka[k] for k in ka)
    n = x + y
    return {"only_first": x, "only_second": y, "p_two_sided": min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)}


def main(out):
    R = {("ours", "low"): with_conf_ours("ours_actest_low_noterm"), ("ours", "high"): with_conf_ours("ours_actest_noterm"),
         ("base", "low"): perstep(os.path.join(RES, "2026-11-12-base", "perstep-base_actest_low.jsonl")),
         ("base", "high"): perstep(os.path.join(RES, "2026-11-12-base", "perstep-base_actest.jsonl"))}
    res = {"scoring": {"ours": "AndroidControl action space (terminate excluded)",
                       "base": "recorded base evaluation, terminate allowed"},
           "standard": {f"{m}_{s}": standard(R[(m, s)]) for (m, s) in R},
           "paired_ours_vs_base": {s: mcnemar(R[("ours", s)], R[("base", s)]) for s in ("low", "high")}}
    conf = {}
    for m, rows in (("base", with_conf_base()), ("ours", R[("ours", "high")])):
        y = [r["ok"] for r in rows]
        conf[m] = {k: C.summarize(C.scores_of(rows, k), y) for k in ("p_act", "product")}
        conf[m]["type_p_act"] = {"auroc": C.auroc([r["p_act"] for r in rows], [r["type_ok"] for r in rows]),
                                 "ece": C.ece([r["p_act"] for r in rows], [r["type_ok"] for r in rows])}
        conf[m]["conformal_episode_split_p_act"] = C.conformal(rows, "p_act")
        with open(os.path.join(D, f"conf-{m}-achigh-nt.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"episode_id": r["ep"], "step": r["step"], "p_action": r["p_act"],
                                    "exact_match": bool(r["ok"])}) + "\n")
    res["confidence_ac_high"] = conf
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main(sys.argv[1])
