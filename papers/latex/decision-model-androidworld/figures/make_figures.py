#!/usr/bin/env python3
"""Figures for the paper. Every plotted value is read from a repository artifact; nothing is typed in by hand
except the labels of one real decision (Figure 1), which are copied from its trace row.

    python3 make_figures.py            # writes fig_*.pdf next to this file

Artifacts:
  benchmarks/results/2026-11-12-calibration/conf-{v4,base}-achigh.jsonl   per-step confidence + InfiGUI verdicts
  benchmarks/results/2026-11-12-maskenv/aw116_v4g_s0{,_b,_c}/episodes     AndroidWorld traces (v4, harness C)
  figures/asset_screen_wifi_step2.png                                     screenshot of the decision in Figure 1
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CAL = os.path.join(ROOT, "benchmarks", "results", "2026-11-12-calibration")
AW = os.path.join(ROOT, "benchmarks", "results", "2026-11-12-maskenv")
AW_DIRS = ["aw116_v4g_s0", "aw116_v4g_s0_b", "aw116_v4g_s0_c"]
TRACE = os.path.join(AW, "aw116_v4g_s0", "episodes", "p1_t0000_systemwifiturnonverify_s30_d1_v4g_s0.jsonl")

C = {"blue": "#3B6FB6", "lblue": "#DCE7F5", "orange": "#E07B39", "lorange": "#FBE3D3", "green": "#3A9A5B",
     "lgreen": "#DDF0E3", "red": "#C2423F", "lred": "#F6DCDB", "purple": "#7A5BA6", "lpurple": "#E9E1F3",
     "gray": "#6B6B6B", "lgray": "#EFEFEF", "ink": "#222222"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def load_conf(name):
    return [json.loads(l) for l in open(os.path.join(CAL, f"conf-{name}-achigh.jsonl"))]


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


# ---------------------------------------------------------------- Figure 1: pipeline with one real decision
def box(ax, x, y, w, h, fc, ec, text=None, size=7.5, weight="normal", r=0.012, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}", fc=fc, ec=ec, lw=0.8,
                                ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=C["ink"])


def arrow(ax, p, q, color=None, lw=0.9, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=color or C["gray"],
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_pipeline():
    rows = [json.loads(l) for l in open(TRACE)]
    step = [r for r in rows if r.get("type") == "step" and r["i"] == 1][0]
    ro = step["state_in"]["readouts"]
    p_act = ro["action"]["probabilities"]
    p_el = ro["target_el"]["probabilities"]
    model_ms = step["model_ms"]
    img = plt.imread(os.path.join(HERE, "asset_screen_wifi_step2.png"))

    fig = plt.figure(figsize=(7.0, 3.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # panel (a): generative agent
    box(ax, 0.005, 0.70, 0.99, 0.29, "#FAFAFA", "#CCCCCC", r=0.01)
    ax.text(0.015, 0.955, "(a) Generative VLM agent", fontsize=8, weight="bold", ha="left", va="center")
    box(ax, 0.03, 0.75, 0.13, 0.15, C["lblue"], C["blue"], "screenshot\n+ goal + history")
    box(ax, 0.21, 0.75, 0.15, 0.15, C["lorange"], C["orange"], "VLM decoder\n(token by token)")
    for k in range(5):
        box(ax, 0.40 + k * 0.045, 0.785, 0.038, 0.08, "white", C["orange"], ["Thought", ":", "...", "Action", "..."][k],
            size=6)
    ax.text(0.505, 0.745, "$T$ sequential decoding steps", fontsize=6.5, ha="center", color=C["gray"])
    box(ax, 0.66, 0.75, 0.12, 0.15, C["lgray"], C["gray"], "string parser\n(may fail)")
    box(ax, 0.83, 0.75, 0.14, 0.15, C["lgreen"], C["green"], "action\n(no calibrated\nprobability)", size=7)
    arrow(ax, (0.16, 0.825), (0.21, 0.825))
    arrow(ax, (0.36, 0.825), (0.40, 0.825))
    arrow(ax, (0.625, 0.825), (0.66, 0.825))
    arrow(ax, (0.78, 0.825), (0.83, 0.825))
    arrow(ax, (0.60, 0.875), (0.31, 0.905), rad=0.25, color=C["orange"], lw=0.7)

    # panel (b): decision model, one real step
    box(ax, 0.005, 0.01, 0.99, 0.67, "#FAFAFA", "#CCCCCC", r=0.01)
    ax.text(0.015, 0.65, "(b) Decision model (this work): one forward pass, zero output tokens", fontsize=8,
            weight="bold", ha="left", va="center")
    # screenshot asset
    iax = fig.add_axes([0.025, 0.05, 0.075, 0.54])
    iax.imshow(img)
    iax.axis("off")
    ax.text(0.0625, 0.025, "screenshot", fontsize=6.5, ha="center", color=C["gray"])
    # text state
    box(ax, 0.115, 0.06, 0.22, 0.53, C["lblue"], C["blue"], r=0.008)
    st = ("Goal: Turn wifi on.\n"
          "Current app: settings\n"
          "Previous actions:\n"
          "  Step 1: open_app settings\n"
          "     $\\rightarrow$ screen changed\n"
          "Elements on screen:\n"
          "  [4] Search settings\n"
          "  [6] Network & internet\n"
          "  [7] Mobile, Wi-Fi, hotspot\n"
          "  [8] Connected devices\n"
          "  ... (19 elements)")
    ax.text(0.123, 0.565, st, fontsize=6.3, ha="left", va="top", family="monospace", color=C["ink"],
            linespacing=1.25)
    ax.text(0.225, 0.075, "text state (same renderer at\ntraining, evaluation, serving)", fontsize=6, ha="center",
            color=C["blue"], style="italic")
    # model
    box(ax, 0.375, 0.17, 0.13, 0.32, C["lorange"], C["orange"], r=0.01)
    ax.text(0.44, 0.40, "d1-2B", fontsize=8.5, weight="bold", ha="center")
    ax.text(0.44, 0.31, "vision encoder\n+ LFM3 backbone\n+ readout heads", fontsize=6.5, ha="center", va="center")
    ax.text(0.44, 0.205, f"{model_ms:.0f} ms, 0 tokens", fontsize=6.5, ha="center", color=C["orange"])
    arrow(ax, (0.10, 0.33), (0.115, 0.33))
    arrow(ax, (0.335, 0.33), (0.375, 0.33))

    # readouts: two bar charts with the real probabilities
    acts = sorted(p_act.items(), key=lambda x: -x[1])[:4]
    els = sorted(p_el.items(), key=lambda x: -x[1])[:3]
    names = {"6": "[6] Network & internet", "7": "[7] Mobile, Wi-Fi, hotspot", "4": "[4] Search settings"}
    bax = fig.add_axes([0.60, 0.40, 0.15, 0.20])
    bax.barh([a for a, _ in acts][::-1], [p for _, p in acts][::-1], color=C["green"], height=0.6)
    bax.set_xlim(0, 1)
    bax.set_title("action $p(a\\mid s)$", fontsize=7, pad=2)
    bax.tick_params(length=2, pad=1)
    for i, (a, p) in enumerate(acts[::-1]):
        bax.text(min(p, 0.70) + 0.02, i, f"{p:.3f}", va="center", fontsize=6)
    eax = fig.add_axes([0.60, 0.10, 0.15, 0.17])
    eax.barh([names.get(e, f"[{e}]") for e, _ in els][::-1], [p for _, p in els][::-1], color=C["purple"], height=0.6)
    eax.set_xlim(0, 1)
    eax.set_title("target element $p(e\\mid s)$", fontsize=7, pad=2)
    eax.tick_params(length=2, pad=1)
    for i, (e, p) in enumerate(els[::-1]):
        eax.text(min(p, 0.70) + 0.02, i, f"{p:.3f}", va="center", fontsize=6)
    ax.text(0.675, 0.035, "+ scroll direction, app, text span, answer", fontsize=6, ha="center", color=C["gray"])
    arrow(ax, (0.505, 0.36), (0.535, 0.50), color=C["green"])
    arrow(ax, (0.505, 0.30), (0.535, 0.19), color=C["purple"])

    # gate and execution
    box(ax, 0.79, 0.36, 0.085, 0.20, C["lpurple"], C["purple"], "gate\nconfidence\n$\\geq\\tau$ ?", size=6.8)
    box(ax, 0.90, 0.44, 0.085, 0.13, C["lgreen"], C["green"], "act:\ntap [6]", size=6.8)
    box(ax, 0.90, 0.24, 0.085, 0.13, C["lred"], C["red"], "ask /\nescalate", size=6.8)
    arrow(ax, (0.765, 0.46), (0.79, 0.46))
    arrow(ax, (0.875, 0.48), (0.90, 0.505), color=C["green"])
    arrow(ax, (0.875, 0.42), (0.90, 0.31), color=C["red"])
    ax.text(0.89, 0.13, "harness maps the typed\nanswers to one device\naction (AndroidWorld)", fontsize=6,
            ha="center", color=C["gray"])
    fig.savefig(os.path.join(HERE, "fig_pipeline.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2: does confidence rank errors?
def fig_calibration():
    v4, base = load_conf("v4"), load_conf("base")
    fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.75), constrained_layout=True)
    for rows, name, col in ((base, "d1-2B (base)", C["gray"]), (v4, "DMA-2B (ours)", C["blue"])):
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
    axs[0].legend(frameon=False, loc="upper left", handlelength=1.2)
    axs[1].plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=0.7)
    axs[1].set_xlabel("$p(\\hat a\\mid s)$ (bin mean)")
    axs[1].set_ylabel("step accuracy")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_title("(b) reliability, AC-High")
    fig.savefig(os.path.join(HERE, "fig_calibration.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3: per step vs whole episode
def fig_episode():
    fig, ax = plt.subplots(figsize=(3.35, 1.6), constrained_layout=True)
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "$\\geq$10"]
    width = 0.38
    for j, (name, tag, col) in enumerate((("d1-2B (base)", "base", C["gray"]), ("DMA-2B (ours)", "v4", C["blue"]))):
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


# ---------------------------------------------------------------- Figure 4: termination confidence on AndroidWorld
def aw_terminations():
    out = []
    for d in AW_DIRS:
        for f in sorted(glob.glob(os.path.join(AW, d, "episodes", "*.jsonl"))):
            if ".abandoned." in f:
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            res = [r for r in rows if r.get("type") == "result"]
            if not res or res[-1].get("unscored") or res[-1].get("setup_error") or res[-1].get("score") is None:
                continue
            st = [r for r in rows if r.get("type") == "step"]
            if res[-1].get("declared_done") and st and st[-1]["action_raw"]["action"] in ("status", "answer"):
                out.append((st[-1]["action_raw"]["confidence"], float(res[-1]["score"]) > 0))
    return out


def fig_termination():
    t = aw_terminations()
    fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.95), constrained_layout=True)
    right = sorted(c for c, y in t if y)
    wrong = sorted(c for c, y in t if not y)
    bins = [0.3 + 0.05 * k for k in range(15)]
    axs[0].hist([min(max(c, 0.3), 0.999) for c in wrong], bins=bins, color=C["red"], alpha=0.75,
                label=f"not credited ({len(wrong)})")
    axs[0].hist([min(max(c, 0.3), 0.999) for c in right], bins=bins, color=C["green"], alpha=0.75,
                label=f"credited ({len(right)})")
    axs[0].set_xlabel("$p(\\mathrm{done}\\mid s)$")
    axs[0].set_ylabel("episodes")
    axs[0].set_title("(a) terminations")
    axs[0].legend(frameon=False, loc="upper left")
    taus = sorted({c for c, _ in t} | {1.01})
    xs, ys = [], []
    for tau in taus:
        xs.append(sum(1 for c in right if c < tau) / len(right))
        ys.append(sum(1 for c in wrong if c < tau) / len(wrong))
    axs[1].plot(xs, ys, color=C["purple"], lw=1.2)
    axs[1].plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=0.7)
    axs[1].set_xlabel("credited, escalated")
    axs[1].set_ylabel("premature, escalated")
    axs[1].set_title("(b) escalation trade-off")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    fig.savefig(os.path.join(HERE, "fig_termination.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_calibration()
    fig_episode()
    fig_termination()
    print("figures written to", HERE)
