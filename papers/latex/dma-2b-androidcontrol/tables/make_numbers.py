#!/usr/bin/env python3
"""LaTeX macros for every number in the text of the AndroidControl paper, from paper_numbers.json
(benchmarks/train/paper_ac_final.py). Writes tables/numbers.tex; nothing is typed by hand."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
J = json.load(open(os.path.join(ROOT, "benchmarks/results/2026-11-16-ac-noterm/paper_numbers.json")))
S, CF, P = J["standard"], J["confidence_ac_high"], J["paired_ours_vs_base"]


def pct(x):
    return f"{100 * x:.1f}"


def macros():
    m = {"BASELOW": f"{S['base_low']['sr']:.1f}", "BASEHIGH": f"{S['base_high']['sr']:.1f}",
         "BASEDONELOW": f"{S['base_low']['predicted_terminate']}", "BASEDONEHIGH": f"{S['base_high']['predicted_terminate']}",
         "OURSESRLOW": f"{S['ours_low']['episode_sr']:.1f}", "OURSESRHIGH": f"{S['ours_high']['episode_sr']:.1f}",
         "BASEESRLOW": f"{S['base_low']['episode_sr']:.1f}", "BASEESRHIGH": f"{S['base_high']['episode_sr']:.1f}"}
    for key, pre in (("base", "BASE"), ("ours", "OURS")):
        c = CF[key]["p_act"]
        m[pre + "MEANCONF"] = pct(c["mean_conf"])
        m[pre + "GAP"] = f"{100 * (c['mean_conf'] - c['accuracy']):.1f}"
        m[pre + "ECE"] = f"{c['ece']:.3f}"
        m[pre + "AUROC"] = f"{c['auroc']:.3f}"
        m[pre + "PRODAUROC"] = f"{CF[key]['product']['auroc']:.3f}"
        m[pre + "RISK"] = pct(c["risk_at_cov"]["0.5"])
        m[pre + "AURC"] = f"{c['aurc']:.3f}"
        t = CF[key]["type_p_act"]
        m[pre + "TYPE"] = f"{t['auroc']:.3f} / {t['ece']:.3f}"
        m[pre + "TYPEAUROC"] = f"{t['auroc']:.3f}"
        m[pre + "TYPEECE"] = f"{t['ece']:.3f}"
        cf = CF[key]["conformal_episode_split_p_act"]
        g = cf["0.2"]
        found = g["seeds"] - g["no_threshold"]
        # a gate that exists in fewer than half of the splits is reported as "--" (the text gives the split count)
        m[pre + "GATE"] = f"{pct(g['coverage_mean'])} / {pct(g['risk_mean'])}" if found >= g["seeds"] / 2 else "--"
        m[pre + "TWENTYSPLITS"] = str(found)
        for d, name in (("0.2", "TWENTY"), ("0.15", "FIFTEEN"), ("0.1", "TEN")):
            x = cf[d]
            if x["coverage_mean"] is not None:
                m[pre + "COV" + name] = f"{100 * x['coverage_mean']:.0f}" if name == "TWENTY" else pct(x["coverage_mean"])
                m[pre + "RISK" + name] = pct(x["risk_mean"])
                m[pre + "VIOL" + name] = f"{100 * x['violation_rate']:.0f}"
            m[pre + name + "SPLITS"] = str(x["seeds"] - x["no_threshold"])
    pmax = max(P["low"]["p_two_sided"], P["high"]["p_two_sided"])
    m["PAIREDEXP"] = "100" if pmax == 0 else str(min(100, int(math.floor(-math.log10(pmax)))))
    return m


out = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros().items())) + "\n"
open(os.path.join(HERE, "numbers.tex"), "w").write(out)
print(out)
