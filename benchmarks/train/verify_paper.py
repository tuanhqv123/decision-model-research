"""Assert every number published in papers/calibrated-decision-interface-2026.md against the saved
artifacts in benchmarks/results/2026-11-03-stage0/.

This is the paper's provenance check: it fails loudly if a claim drifts from the data. NumPy only.

Usage: python verify_paper.py [--dir benchmarks/results/2026-11-03-stage0]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_routed import cp_upper  # noqa: E402
from paired_eval import load, mcnemar_exact  # noqa: E402

FAILS = []


def check(name, got, want, tol=0.0015):
    numeric = isinstance(got, (int, float)) and isinstance(want, (int, float)) \
        and not isinstance(got, bool) and not isinstance(want, bool)
    ok = (abs(float(got) - float(want)) <= tol) if numeric else (got == want)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:58s} got={got!r:>12} want={want!r}")
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="benchmarks/results/2026-11-03-stage0")
    a = ap.parse_args()
    D = a.dir

    eb = json.load(open(f"{D}/error_budget.json"))
    test, val, t705 = eb[0], eb[1], eb[2]

    print("\n== §2.2 / Table 1 — accuracy and error budget")
    check("test n", test["n"], 398)
    check("test accuracy", test["acc"], 0.6482)
    check("test bucket A (element wrong)", test["buckets"]["A_spatial_type_ok_element_wrong"], 68)
    check("test bucket B (spatial type wrong)", test["buckets"]["B_spatial_type_wrong"], 27)
    check("test bucket C1 (non-spatial type wrong)", test["buckets"]["C1_nonspatial_type_wrong"], 36)
    check("test bucket C2 (non-spatial value wrong)", test["buckets"]["C2_nonspatial_value_wrong"], 9)
    check("test oracle grounding only", test["oracle"]["grounding_only"], 0.8191)
    check("test oracle type only", test["oracle"]["type_only"], 0.8065)
    check("val n", val["n"], 827)
    check("val bucket A", val["buckets"]["A_spatial_type_ok_element_wrong"], 137)
    check("val bucket B", val["buckets"]["B_spatial_type_wrong"], 52)
    check("val bucket C1", val["buckets"]["C1_nonspatial_type_wrong"], 109)
    check("val bucket C2", val["buckets"]["C2_nonspatial_value_wrong"], 10)

    print("\n== §3.4 / Table 2 — grounding audit")
    au = json.load(open(f"{D}/grounding_audit.json"))["summary"]
    check("audit spatial joined", au["spatial_joined"], 198)
    check("audit exact hit rate", au["exact_hit_rate"], 0.6616)
    check("audit misses differ (different element)", au["cats"]["different_element"], 54)
    check("audit misses subtree", au["cats"]["subtree"], 12)
    check("audit misses same_label", au["cats"]["same_label"], 1)
    check("audit tolerant hit rate", au["tolerant_hit_rate"], 0.7273)
    check("audit ambiguous rows", au["ambiguous_rows"], 19)

    print("\n== §4 / Table 3 — score AUROC")
    g938 = json.load(open(f"{D}/gate_test938.json"))
    g705 = json.load(open(f"{D}/gate_test705.json"))
    check("AUROC global val", g938["auroc"]["global"][0], 0.677)
    check("AUROC global test@938", g938["auroc"]["global"][1], 0.649)
    check("AUROC routed val", g938["auroc"]["routed"][0], 0.760)
    check("AUROC routed test@938", g938["auroc"]["routed"][1], 0.729)
    check("AUROC routed test@705", g705["auroc"]["routed"][1], 0.709)
    check("AURC routed val", g938["aurc"]["routed"][0], 0.209, tol=0.002)
    check("AURC routed test@938", g938["aurc"]["routed"][1], 0.2006, tol=0.002)
    check("AURC global test@938", g938["aurc"]["global"][1], 0.2441, tol=0.002)

    print("\n== §5 / Table 4 — conformal risk control")
    r = [x for x in g938["gate"] if abs(x["delta"] - 0.20) < 1e-9][0]
    check("gate tau", round(r["tau"], 3), 0.527)
    check("gate val coverage", r["val_coverage"], 0.476, tol=0.003)
    check("gate test coverage", r["test_coverage"], 0.407, tol=0.003)
    check("gate test risk", r["test_risk"], 0.173, tol=0.003)
    check("gate test CP upper", r["test_ub"], 0.217, tol=0.003)
    check("gate test n accepted", r["test_n"], 162)
    check("gate held on test@938", r["held"], True)
    check("no 10% threshold exists", all(x["delta"] != 0.10 for x in g938["gate"]), True)

    print("\n== §6.1 — the risk floor forbids the 10% budget")
    fr = {x["delta"]: x["max_coverage"] for x in g938["frontier"]}
    check("frontier delta=0.20 coverage", fr[0.20], 0.476, tol=0.003)
    check("frontier delta=0.15 not certifiable", fr[0.15] is None, True)
    check("frontier delta=0.10 not certifiable", fr[0.10] is None, True)
    check("CP bound at 0 errors needs n>=22 for 10%", cp_upper(0, 22, 0.10) <= 0.10, True)

    print("\n== §6.2 — threshold does not transfer across checkpoints")
    r7 = [x for x in g705["gate"] if abs(x["delta"] - 0.20) < 1e-9][0]
    check("same tau on @938 holds", r["held"], True)
    check("same tau on @705 violates", r7["held"], False)
    check("gate test@705 risk", r7["test_risk"], 0.239, tol=0.003)
    check("gate test@705 coverage", r7["test_coverage"], 0.389, tol=0.003)

    print("\n== §4.4 / Table 3b — learned fusion (reported as NOT claimed)")
    fus = json.load(open(f"{D}/score_fusion.json"))
    check("fusion val AUROC", fus["auroc"]["LEARNED FUSION"][0], 0.767, tol=0.002)
    check("fusion test AUROC", fus["auroc"]["LEARNED FUSION"][1], 0.743, tol=0.002)
    check("routed test AUROC (for the delta)", fus["auroc"]["routed (hand)"][1], 0.730, tol=0.002)
    check("fusion test risk@50%", fus["test_risk_at_50"]["LEARNED FUSION"], 0.191, tol=0.003)
    check("routed test risk@50%", fus["test_risk_at_50"]["routed (hand)"], 0.226, tol=0.003)
    check("delta AUROC is NOT significant", fus["significant"], False)

    print("\n== §5.1 / Table 4b — CRC vs Clopper-Pearson")
    rows = {(g["delta"], g["method"]): g for g in fus["gate"]["rows"]}
    check("CP 20% test coverage", rows[(0.20, "CP")]["test_coverage"], 0.384, tol=0.004)
    check("CRC 20% test coverage", rows[(0.20, "CRC")]["test_coverage"], 0.455, tol=0.004)
    check("CRC 20% holds", rows[(0.20, "CRC")]["held"], True)
    check("CP 10% not certifiable", (0.10, "CP") not in rows, True)
    check("CRC 10% degenerate coverage", rows[(0.10, "CRC")]["test_coverage"], 0.035, tol=0.004)
    check("CRC 10% accepted steps", rows[(0.10, "CRC")]["test_n"], 14)
    check("val/test TV distance", fus["gate"]["mix_tv"], 0.106, tol=0.003)
    check("effective calibration size", fus["gate"]["ess"], 752, tol=2)

    print("\n== §3.5 / Table — Stage-2 diagnosis (per-class logit adjustment)")
    cb = json.load(open(f"{D}/class_bias.json"))
    check("class-bias val unadjusted type acc", cb["val_type_acc"][0], 0.8053, tol=0.002)
    check("class-bias val adjusted type acc", cb["val_type_acc"][1], 0.8210, tol=0.002)
    check("class-bias test unadjusted type acc", cb["test_type_acc"][0], 0.8417, tol=0.002)
    check("class-bias test adjusted type acc", cb["test_type_acc"][1], 0.8141, tol=0.002)
    check("class-bias test regressed rows", cb["test_regressed"], 18)
    check("class-bias test fixed rows", cb["test_fixed"], 7)
    check("class-bias McNemar p", round(cb["mcnemar_p"], 4), 0.0433, tol=0.0005)

    print("\n== §3.2 — paired checkpoint comparison")
    base = load("benchmarks/results/2026-10-29-evals/E2-705-V.jsonl")
    cand = load("benchmarks/results/2026-10-29-evals/E2-938-V.jsonl")
    keys = sorted(set(base) & set(cand))
    bc = [int(base[k]["correct"]) for k in keys]
    cc = [int(cand[k]["correct"]) for k in keys]
    b = sum(1 for x, y in zip(bc, cc) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(bc, cc) if x == 0 and y == 1)
    delta = sum(cc) / len(cc) - sum(bc) / len(bc)
    check("paired delta (705->938)", delta, 0.0352, tol=0.001)
    check("McNemar b", b, 8)
    check("McNemar c", c, 22)
    check("McNemar p", round(mcnemar_exact(b, c), 3), 0.016, tol=0.002)

    print(f"\n{'='*78}")
    if FAILS:
        print(f"FAILED {len(FAILS)} check(s): {FAILS}")
        sys.exit(1)
    print("ALL CHECKS PASS — every stage-0 number matches the saved artifacts.")
    print("(scope: stage-0 derived JSON only; the pure-protocol row, baselines, px-distances,")
    print(" the a11y diagnostic, latency and every online number are NOT asserted here.)")


if __name__ == "__main__":
    main()
