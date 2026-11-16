#!/usr/bin/env python3
"""AndroidControl tables of the paper (2026-11-16): one row per model, step success rate only, no citations inside the
table (sources are cited in the text). Reported rows are copied from:
  InfiGUI-R1 Table 3 (arXiv 2504.14239): GPT-4o, OS-Atlas-4B, Aguvis-7B/72B, UI-TARS-2B, InfiGUI-R1-3B
  GUI-R1 Tables 2-3 (arXiv 2504.10458), shared zero-shot prompt: Qwen2.5-VL-3B/7B, UI-R1-3B, GUI-R1-3B/7B
  MobileRL Table 2 (arXiv 2509.18119): UI-TARS-7B/72B, UI-Genie-7B/72B, MobileRL-7B/9B
Our rows come from benchmarks/results/2026-11-16-ac-noterm/paper_numbers.json (benchmarks/train/paper_ac_final.py).
Writes tab_ac_rows.tex (comparison) and tab_ours_rows.tex (base vs ours in Type / Grounding / SR / episode SR)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
NUM = json.load(open(os.path.join(ROOT, "benchmarks/results/2026-11-16-ac-noterm/paper_numbers.json")))["standard"]
OURS_NAME = "\\ours{}"

GROUPS = [
    ("General vision-language models, zero-shot", [
        ("GPT-4o", "--", 19.4, 20.8), ("Qwen2.5-VL-3B", "3B", 59.3, 38.9), ("Qwen2.5-VL-7B", "7B", 62.5, 47.1)]),
    ("GUI agents, up to 4B", [
        ("UI-R1-3B", "3B", 66.4, 45.4), ("GUI-R1-3B", "3B", 64.4, 46.6), ("OS-Atlas-4B", "4B", 80.6, 67.5),
        ("UI-TARS-2B", "2B", 89.3, 68.9), ("InfiGUI-R1-3B", "3B", 92.1, 71.1)]),
    ("GUI agents, 7B to 9B", [
        ("GUI-R1-7B", "7B", 66.5, 51.7), ("Aguvis-7B", "7B", 80.5, 61.5), ("UI-TARS-7B", "7B", 90.8, 72.5),
        ("MobileRL-7B", "7B", 91.4, 71.3), ("UI-Genie-7B", "7B", 94.3, 74.5), ("MobileRL-9B", "9B", 94.3, 73.9)]),
    ("GUI agents, 72B", [
        ("Aguvis-72B", "72B", 84.4, 66.4), ("UI-TARS-72B", "72B", 91.3, 74.7), ("UI-Genie-72B", "72B", 94.8, 77.0)]),
]
OURS = [("d1-2B (base)", "2B", NUM["base_low"]["sr"], NUM["base_high"]["sr"], False),
        (OURS_NAME, "2B", NUM["ours_low"]["sr"], NUM["ours_high"]["sr"], True)]

allv = [(r[2], r[3]) for _, rs in GROUPS for r in rs] + [(r[2], r[3]) for r in OURS]
best = [max(v[c] for v in allv) for c in (0, 1)]


def cell(v, c):
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if v == best[c] else s


lines = []
for title, rs in GROUPS:
    lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
    for name, size, lo, hi in rs:
        lines.append(f"\\quad {name} & {size} & yes & {cell(lo, 0)} & {cell(hi, 1)} \\\\")
    lines.append("\\addlinespace[2pt]")
lines.append("\\midrule")
lines.append("\\multicolumn{5}{@{}l}{\\textit{Decision model, one forward pass (this work)}} \\\\")
for name, size, lo, hi, ours in OURS:
    pre = "\\rowcolor{oursbg}" if ours else ""
    nm = f"\\textbf{{{name}}}" if ours else name
    lines.append(f"{pre}\\quad {nm} & {size} & no & {cell(lo, 0)} & {cell(hi, 1)} \\\\")
open(os.path.join(HERE, "tab_ac_rows.tex"), "w").write("\n".join(lines) + "\n")

det = []
for key, name in (("base", "d1-2B (base)"), ("ours", f"\\textbf{{{OURS_NAME}}}")):
    lo, hi = NUM[f"{key}_low"], NUM[f"{key}_high"]
    pre = "\\rowcolor{oursbg}" if key == "ours" else ""
    det.append(pre + f"{name} & {lo['type']:.1f} & {lo['grounding']:.1f} & {lo['sr']:.1f} & {lo['episode_sr']:.1f} & "
               f"{hi['type']:.1f} & {hi['grounding']:.1f} & {hi['sr']:.1f} & {hi['episode_sr']:.1f} \\\\")
open(os.path.join(HERE, "tab_ours_rows.tex"), "w").write("\n".join(det) + "\n")
print("\n".join(lines + det))
