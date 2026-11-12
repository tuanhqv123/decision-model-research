#!/usr/bin/env python3
"""AndroidControl comparison table (tab_ac.tex). Reported rows are copied from the cited tables:
InfiGUI-R1 Table 3 (arXiv 2504.14239), GUI-R1 Tables 2-3 (arXiv 2504.10458), MobileRL Table 2 (arXiv 2509.18119).
Our rows are recomputed from benchmarks/results/2026-11-12-{v4/actest,base}/perstep-*.jsonl (InfiGUI-R1 scorer)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
N = None
GROUPS = [
 ("Reported by InfiGUI-R1~\\citep{infiguir1}", [
  ("Claude$^{*}$", "--", (74.3, 0.0, 19.4, 63.7, 0.0, 12.5)),
  ("GPT-4o~\\citep{gpt4o}", "--", (74.3, 0.0, 19.4, 66.3, 0.0, 20.8)),
  ("Aria-UI~\\citep{ariaui}", "--", (N, 87.7, 67.3, N, 43.2, 10.2)),
  ("OS-Atlas-4B~\\citep{osatlas}", "4B", (91.9, 83.8, 80.6, 84.7, 73.8, 67.5)),
  ("Aguvis-7B~\\citep{aguvis}", "7B", (N, N, 80.5, N, N, 61.5)),
  ("Aguvis-72B~\\citep{aguvis}", "72B", (N, N, 84.4, N, N, 66.4)),
  ("UI-R1-3B~\\citep{uir1}", "3B", (94.3, 82.6, N, N, N, N)),
  ("GUI-R1-3B~\\citep{guir1}", "3B", (N, N, N, 58.0, 56.2, 46.6)),
  ("GUI-R1-7B~\\citep{guir1}", "7B", (N, N, N, 71.6, 65.6, 51.7)),
  ("UI-TARS-2B~\\citep{uitars2025}", "2B", (98.1, 87.3, 89.3, 81.2, 78.4, 68.9)),
  ("InfiGUI-R1-3B~\\citep{infiguir1}", "3B", (96.0, 93.2, 92.1, 82.7, 74.4, 71.1)),
 ]),
 ("Reported by GUI-R1, shared zero-shot prompt~\\citep{guir1}", [
  ("OS-Atlas-4B~\\citep{osatlas}", "4B", (64.6, 71.2, 40.6, 49.0, 49.5, 22.8)),
  ("Qwen2.5-VL-3B~\\citep{qwen25vl}", "3B", (62.0, 74.1, 59.3, 47.8, 46.5, 38.9)),
  ("Qwen2.5-VL-7B~\\citep{qwen25vl}", "7B", (83.4, 87.1, 62.5, 68.7, 59.7, 47.1)),
  ("UI-R1-3B~\\citep{uir1}", "3B", (79.2, 82.4, 66.4, 57.9, 55.7, 45.4)),
  ("GUI-R1-3B~\\citep{guir1}", "3B", (83.7, 81.6, 64.4, 58.0, 56.2, 46.6)),
  ("GUI-R1-7B~\\citep{guir1}", "7B", (85.2, 84.0, 66.5, 71.6, 65.6, 51.7)),
 ]),
 ("Reported by MobileRL~\\citep{mobilerl2025}", [
  ("UI-TARS-7B~\\citep{uitars2025}", "7B", (98.0, 89.3, 90.8, 83.7, 80.5, 72.5)),
  ("UI-TARS-72B~\\citep{uitars2025}", "72B", (98.1, 90.8, 91.3, 85.2, 81.5, 74.7)),
  ("UI-Genie-7B~\\citep{uigenie}", "7B", (98.1, 89.3, 94.3, 83.5, 82.9, 74.5)),
  ("UI-Genie-72B~\\citep{uigenie}", "72B", (98.3, 95.4, 94.8, 84.9, 86.3, 77.0)),
  ("MobileRL-7B~\\citep{mobilerl2025}", "7B", (96.9, 92.4, 91.4, 86.3, 71.5, 71.3)),
  ("MobileRL-9B~\\citep{mobilerl2025}", "9B", (97.9, 94.6, 94.3, 87.0, 74.5, 73.9)),
 ]),
]

def ours(tag):
    R = [json.loads(l) for l in open(os.path.join(ROOT, tag))]
    n = len(R)
    t = 100 * sum(r["type_match"] for r in R) / n
    g = [r for r in R if r["type_match"] and r["gt"]["action"] in ("click", "long_press")]
    gr = 100 * sum(r["exact_match"] for r in g) / len(g)
    sr = 100 * sum(r["exact_match"] for r in R) / n
    return round(t, 1), round(gr, 1), round(sr, 1)

b_low = ours("benchmarks/results/2026-11-12-base/perstep-base_actest_low.jsonl")
b_high = ours("benchmarks/results/2026-11-12-base/perstep-base_actest.jsonl")
o_low = ours("benchmarks/results/2026-11-12-v4/actest/perstep-v4_s0_ckpt_2482_actest_v4_low.jsonl")
o_high = ours("benchmarks/results/2026-11-12-v4/actest/perstep-v4_s0_ckpt_2482_actest_v4.jsonl")
OURS = [("d1-2B (base)~\\citep{liquidd12026}", "2B", b_low + b_high),
        ("\\textbf{DMA-2B (ours)}", "2B", o_low + o_high)]

allrows = [r for _, rs in GROUPS for r in rs] + OURS
best, second = [], []
for c in range(6):
    vals = sorted({r[2][c] for r in allrows if r[2][c] is not None}, reverse=True)
    best.append(vals[0]); second.append(vals[1])

def cell(v, c):
    if v is None:
        return "--"
    s = f"{v:.1f}"
    if v == best[c]:
        s = f"\\textbf{{{s}}}"
    elif v == second[c]:
        s = f"\\underline{{{s}}}"
    if c in (2, 5):
        s = f"\\cellcolor{{srbg}}{s}"
    return s

lines = []
def row(name, size, v):
    lines.append(f"{name} & {size} & " + " & ".join(cell(x, c) for c, x in enumerate(v)) + " \\\\")
for title, rs in GROUPS:
    lines.append(f"\\rowcolor{{grpbg}}\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
    for r in rs:
        row(*r)
lines.append("\\midrule")
lines.append("\\rowcolor{grpbg}\\multicolumn{8}{@{}l}{\\textit{This work: decision model, zero output tokens, InfiGUI-R1 scorer}} \\\\")
for r in OURS:
    row(*r)
open(os.path.join(HERE, "tab_ac_rows.tex"), "w").write("\n".join(lines) + "\n")
print(b_low, b_high, o_low, o_high)
