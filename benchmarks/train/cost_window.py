"""Cost-window table: does the calibrated gate beat always-act and always-ask?

This is the number the audience note (T2 §A.5, §B.6) says a reviewer will grade and a builder will price.
It needs no model: everything comes from the saved gate/accuracy artifacts.

Normalise C_ask = 1 and let R = C_err / C_ask. For 1,000 decisions:

    always-act      : 1000 * (1 - acc) * R
    always-ask      : 1000
    gate at (c, r)  : 1000 * (c * r * R + (1 - c))

Assumption (stated, not measured): an ask is answered correctly and costs a fixed C_ask. The literature
prices the ask only through override/habituation rates (T2 §C.3), never in the product's own utility.

Usage:
  python3 benchmarks/train/cost_window.py \
    --gate benchmarks/results/2026-11-03-stage0/gate_test938.json \
    --fusion benchmarks/results/2026-11-03-stage0/score_fusion.json
"""
import argparse
import json


def breakeven_lower(acc, cov, risk):
    """Smallest R at which gate < always-act."""
    act = 1.0 - acc
    num = (1.0 - cov)
    den = act - cov * risk
    return float("inf") if den <= 0 else num / den


def breakeven_upper(cov, risk):
    """Largest R at which gate < always-ask (C_ask = 1)."""
    return float("inf") if risk <= 0 else 1.0 / risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--acc", type=float, default=0.6482,
                    help="accuracy of the SAME artifact the gate was fitted on (E2-938-V = 0.6482)")
    ap.add_argument("--out", default="benchmarks/results/2026-11-07-cost-window.json")
    a = ap.parse_args()

    g = json.load(open(a.gate))
    row = next(r for r in g["gate"] if abs(r["delta"] - 0.20) < 1e-9)
    cov, risk, ub = row["test_coverage"], row["test_risk"], row["test_ub"]

    policies = {
        "always_act": {"cost": lambda R: (1 - a.acc) * R},
        "always_ask": {"cost": lambda R: 1.0},
        "gate_empirical": {"cost": lambda R: cov * risk * R + (1 - cov)},
        "gate_upper_bound": {"cost": lambda R: cov * ub * R + (1 - cov)},
    }
    if a.fusion:
        f = json.load(open(a.fusion))
        risk_d = f.get("test_risk_at_50") or {}
        r50 = risk_d.get("LEARNED FUSION") if isinstance(risk_d, dict) else risk_d
        if r50 is not None:
            policies["fusion_risk@50"] = {"cost": lambda R: 0.5 * float(r50) * R + 0.5}

    grid = [0.5, 1, 2, 3, 4, 6, 8, 16]
    table = {name: {R: round(1000 * p["cost"](R), 1) for R in grid} for name, p in policies.items()}

    out = {
        "input": {"gate": a.gate, "fusion": a.fusion, "accuracy": a.acc},
        "operating_point": {"coverage": cov, "risk": risk, "cp_upper": ub, "test_n": row["test_n"]},
        "assumption": "C_ask = 1, an answered ask resolves the decision correctly; C_err = R",
        "table_cost_per_1000": table,
        "breakeven": {
            "gate_vs_always_act_R_gt": round(breakeven_lower(a.acc, cov, risk), 3),
            "gate_vs_always_act_R_gt_using_upper": round(breakeven_lower(a.acc, cov, ub), 3),
            "gate_vs_always_ask_R_lt": round(breakeven_upper(cov, risk), 3),
            "gate_vs_always_ask_R_lt_using_upper": round(breakeven_upper(cov, ub), 3),
        },
    }
    if "fusion_risk@50" in policies:
        out["breakeven"]["fusion_vs_always_act_R_gt"] = round(
            breakeven_lower(a.acc, 0.5, float(r50)), 3)
        out["breakeven"]["fusion_vs_always_ask_R_lt"] = round(breakeven_upper(0.5, float(r50)), 3)

    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
