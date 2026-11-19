#!/usr/bin/env python3
"""Figures for the paper. Every plotted value is read from a repository artifact; nothing is typed in by hand.

    python3 make_figures.py            # writes fig_*.pdf next to this file

Artifacts (2026-11-16 AndroidControl-only paper):
  benchmarks/results/2026-11-16-ac-noterm/conf-{base,ours}-achigh-nt.jsonl   per-step confidence + InfiGUI verdicts,
  AndroidControl action space (written by benchmarks/train/paper_ac_final.py)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CAL = os.path.join(ROOT, "benchmarks", "results", "2026-11-16-ac-noterm")
OURS = "DMA-2B (ours)"

C = {"blue": "#3B6FB6", "lblue": "#DCE7F5", "orange": "#E07B39", "lorange": "#FBE3D3", "green": "#3A9A5B",
     "lgreen": "#DDF0E3", "red": "#C2423F", "lred": "#F6DCDB", "purple": "#7A5BA6", "lpurple": "#E9E1F3",
     "gray": "#6B6B6B", "lgray": "#EFEFEF", "ink": "#222222"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def load_conf(name):
    return [json.loads(l) for l in open(os.path.join(CAL, f"conf-{name}-achigh-nt.jsonl"))]


def risk_coverage(scores, ok):
    order = sorted(range(len(scores)), key=lambda k: -scores[k])
    cov, risk, err = [], [], 0
    for m, k in enumerate(order, 1):
        err += 1 - ok[k]
        cov.append(m / len(order))
        risk.append(err / m)
    return cov, risk


def reliability(scores, ok, bins=10):
    xs, ys, ns = [], [], []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [k for k, s in enumerate(scores) if lo <= s < hi or (b == bins - 1 and s == 1.0)]
        if len(idx) >= 20:
            xs.append(sum(scores[k] for k in idx) / len(idx))
            ys.append(sum(ok[k] for k in idx) / len(idx))
            ns.append(len(idx))
    return xs, ys, ns


# ---------------------------------------------------------------- Figure 1: risk-coverage and reliability
def fig_calibration():
    ours, base = load_conf("ours"), load_conf("base")
    fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.75), constrained_layout=True)
    for rows, name, col in ((base, "d1-2B (base)", C["gray"]), (ours, OURS, C["blue"])):
        s = [r["p_action"] for r in rows]
        y = [int(r["exact_match"]) for r in rows]
        cov, risk = risk_coverage(s, y)
        axs[0].plot(cov, risk, color=col, lw=1.2, label=name)
        xs, ys, _ = reliability(s, y)
        axs[1].plot(xs, ys, "o-", color=col, ms=2.5, lw=1.0, label=name)
    axs[0].set_xlabel("coverage (share of steps acted on)")
    axs[0].set_ylabel("selective risk (error rate)")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 0.7)
    axs[0].set_title("(a) risk–coverage, AC-High")
    # direct labels: a legend box overlaps one of the curves wherever it is put
    axs[0].text(0.06, 0.52, "d1-2B (base)", color=C["gray"], fontsize=7)
    axs[0].text(0.42, 0.04, OURS, color=C["blue"], fontsize=7)
    axs[1].plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=0.7)
    axs[1].set_xlabel("$p(\\hat a\\mid s)$ (bin mean)")
    axs[1].set_ylabel("step accuracy")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_title("(b) reliability, AC-High")
    fig.savefig(os.path.join(HERE, "fig_calibration.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2: per step vs whole episode
def fig_episode():
    fig, ax = plt.subplots(figsize=(3.35, 1.6), constrained_layout=True)
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "$\\geq$10"]
    width = 0.38
    for j, (name, tag, col) in enumerate((("d1-2B (base)", "base", C["gray"]), (OURS, "ours", C["blue"]))):
        by = {}
        for r in load_conf(tag):
            by.setdefault(r["episode_id"], []).append(r["exact_match"])
        acc = []
        for b in range(1, 11):
            g = [v for v in by.values() if (len(v) == b if b < 10 else len(v) >= 10)]
            acc.append(100.0 * sum(all(v) for v in g) / len(g))
        ax.bar([i + (j - 0.5) * width for i in range(10)], acc, width=width, color=col, label=name)
    ax.set_xticks(range(10))
    ax.set_xticklabels(labels)
    ax.set_xlabel("episode length (steps)")
    ax.set_ylabel("episodes fully correct (%)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(HERE, "fig_episode.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_calibration()
    fig_episode()
    print("figures written to", HERE)
