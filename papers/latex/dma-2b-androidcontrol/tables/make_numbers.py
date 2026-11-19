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
                m[pre + "COV" + name] = pct(x["coverage_mean"])
                m[pre + "RISK" + name] = pct(x["risk_mean"])
                m[pre + "VIOL" + name] = f"{100 * x['violation_rate']:.0f}"
            m[pre + name + "SPLITS"] = str(x["seeds"] - x["no_threshold"])
    CUR = json.load(open(os.path.join(ROOT, "benchmarks/results/2026-11-16-ac-noterm/curated_summary.json")))
    for name, f in (("CURLOW", "easy_box_low"), ("CURBOX", "hard_box_high"), ("CURHARD", "hard_improved_high")):
        c = CUR[f]
        m[name] = f"{c['sr_pct']:.1f}"
    pmax = max(P["low"]["p_two_sided"], P["high"]["p_two_sided"])
    m["PAIREDEXP"] = "100" if pmax == 0 else str(min(100, int(math.floor(-math.log10(pmax)))))
    SD = json.load(open(os.path.join(ROOT, "benchmarks/results/2026-11-16-ac-noterm/seeds_summary.json")))
    m["SEEDN"] = str(len(SD["seeds"]))
    for sp, pre in (("low", "LOW"), ("high", "HIGH")):
        x = SD[sp + "_sr"]
        m["SEEDMEAN" + pre] = f"{x['mean']:.1f}"
        m["SEEDSD" + pre] = f"{x['sd']:.1f}" if x["sd"] is not None else "--"
        m["SEEDRANGE" + pre] = f"{x['max'] - x['min']:.1f}"
    m.update(bootstrap_and_errors())
    return m


def _perstep(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(str(r["episode_id"]), r["step"])] = r
    return out


def bootstrap_and_errors(reps=2000, seed=0):
    """95% intervals over episodes (resampled with replacement; steps of one episode are not independent) for our step
    success rate and for its paired difference from the base model; the base model's upper bound if every one of its
    "done" answers were replaced by a correct action; and the breakdown of our AndroidControl-High errors."""
    import random
    res = os.path.join(ROOT, "benchmarks/results")
    files = {"low": ("2026-11-16-ac-noterm/perstep-ours_actest_low_noterm.jsonl", "2026-11-12-base/perstep-base_actest_low.jsonl"),
             "high": ("2026-11-16-ac-noterm/perstep-ours_actest_noterm.jsonl", "2026-11-12-base/perstep-base_actest.jsonl")}
    m = {}
    for sp, (fo, fb) in files.items():
        ours, base = _perstep(os.path.join(res, fo)), _perstep(os.path.join(res, fb))
        assert ours.keys() == base.keys()
        ep = {}
        for k in ours:
            ep.setdefault(k[0], []).append((ours[k]["exact_match"], base[k]["exact_match"]))
        keys, rnd, sr, diff = list(ep), random.Random(seed), [], []
        for _ in range(reps):
            pick = [ep[rnd.choice(keys)] for _ in keys]
            n = sum(len(x) for x in pick)
            o = sum(a for x in pick for a, _ in x)
            b = sum(c for x in pick for _, c in x)
            sr.append(100 * o / n)
            diff.append(100 * (o - b) / n)
        sr.sort(), diff.sort()
        lo, hi = int(0.025 * reps), int(0.975 * reps) - 1
        P = sp.upper()
        m["OURSCILO" + P], m["OURSCIHI" + P] = f"{sr[lo]:.1f}", f"{sr[hi]:.1f}"
        n = len(ours)
        d = 100 * (sum(r["exact_match"] for r in ours.values()) - sum(r["exact_match"] for r in base.values())) / n
        m["DIFF" + P], m["DIFFCILO" + P], m["DIFFCIHI" + P] = f"{d:.1f}", f"{diff[lo]:.1f}", f"{diff[hi]:.1f}"
        b = S["base_" + sp]
        m["BASEBOUND" + P] = f"{100 * (b['correct'] + b['predicted_terminate']) / b['steps']:.1f}"
    high = _perstep(os.path.join(res, files["high"][0])).values()
    err = [r for r in high if not r["exact_match"]]
    act = lambda a: a["action"]  # noqa: E731
    m["ERRHIGH"] = f"{len(err):,}".replace(",", "{,}")
    m["ERRTYPEHIGH"] = f"{sum(not r['type_match'] for r in err):,}".replace(",", "{,}")
    wrong_target = [r for r in err if r["type_match"]]
    m["ERRARGHIGH"] = f"{len(wrong_target):,}".replace(",", "{,}")
    m["ERRCLICKTARGETHIGH"] = str(sum(act(r["gt"]) == "click" for r in wrong_target))
    pair = lambda r: {act(r["gt"]), act(r["pred"])}  # noqa: E731
    m["ERRCLICKSCROLLHIGH"] = str(sum(not r["type_match"] and pair(r) == {"click", "swipe"} for r in err))
    m["ERRCLICKWAITHIGH"] = str(sum(not r["type_match"] and pair(r) == {"click", "wait"} for r in err))
    return m


def seed_rows():
    """tables/tab_seeds_rows.tex: one row per training seed, then mean and standard deviation."""
    SD = json.load(open(os.path.join(ROOT, "benchmarks/results/2026-11-16-ac-noterm/seeds_summary.json")))
    rows = []
    for k, v in sorted(SD["seeds"].items()):
        name = f"Seed {k}" + (" (released)" if k == "0" else "")
        rows.append(f"{name} & {v['low']['sr']:.1f} & {v['high']['sr']:.1f} & {v['high']['episode_sr']:.1f} \\\\")
    f = lambda d: f"{d['mean']:.1f} $\\pm$ {d['sd']:.1f}"  # noqa: E731
    rows.append("\\midrule")
    rows.append(f"Mean $\\pm$ s.d. & {f(SD['low_sr'])} & {f(SD['high_sr'])} & {f(SD['high_episode_sr'])} \\\\")
    return "\n".join(rows) + "\n"


open(os.path.join(HERE, "tab_seeds_rows.tex"), "w").write(seed_rows())
out = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros().items())) + "\n"
open(os.path.join(HERE, "numbers.tex"), "w").write(out)
print(out)
