#!/usr/bin/env python3
"""Re-derive every number in paper §5.12.1 from the captured primary table.

Purpose: §5.12.1 makes a claim that is arithmetic on published numbers. Arithmetic is exactly the
kind of claim that survives review when nobody re-runs it, so this script re-runs it from
``references/raw/maiui2026-androidworld-table9.txt`` — a verbatim capture of arXiv:2512.22047v1 —
and asserts each figure the paper prints.

    python scripts/verify_compound_accuracy.py

Exit 0 means the paper's table and the primary capture agree. Non-zero lists the disagreement.

The one free parameter is stated, not hidden: converting an end-to-end success rate to a per-step
rate needs the step count ``n``. The paper uses n = 10 and says so; this script uses the same n and
also prints the range for n in 5..15 so a reader can see how sensitive the inference is.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "references", "raw", "maiui2026-androidworld-table9.txt")
PAPER = os.path.join(REPO, "papers", "single-pass-decision-acv2.md")

# End-to-end AndroidWorld success rates, as captured verbatim from the report.
E2E = {
    "M3A (GPT-4 Turbo, a11y tree)": 0.306,     # arXiv:2405.14573v2, not MAI-UI
    "MAI-UI-2B SFT": 0.451,                    # Table 9
    "MAI-UI-2B RL": 0.491,                     # Table 9 / contribution bullet
    "MAI-UI-8B SFT": 0.647,                    # Table 9
    "MAI-UI-8B RL": 0.707,                     # Table 9
    "MAI-UI-32B SFT": 0.698,                   # Table 9
    "MAI-UI-32B RL": 0.733,                    # Table 9
    "MAI-UI-235B-A22B RL": 0.767,              # Table 9 / contribution bullet
}

OURS_PER_STEP = 0.6407                          # paper §5.1, 255/398


def per_step(e2e, n):
    return e2e ** (1.0 / n)


def main():
    raw = open(RAW, encoding="utf-8").read()
    fails = []

    def check(name, ok, detail=""):
        print("  [%s] %-56s %s" % ("PASS" if ok else "FAIL", name, detail))
        if not ok:
            fails.append(name)

    print("=" * 78)
    print("1. the raw capture actually contains the numbers we attribute to it")
    print("=" * 78)
    for label, needle in [
        ("Table 9 header", "Table 9: Online RL performance gains"),
        ("2B SFT 45.1", "MAI-UI-2B-SFT 45.1"),
        ("2B RL 49.1", "MAI-UI-2B-RL 49.1 (+4.0)"),
        ("8B SFT 64.7", "MAI-UI-8B-SFT 64.7"),
        ("8B RL 70.7", "MAI-UI-8B-RL 70.7 (+6.0)"),
        ("32B SFT 69.8", "MAI-UI-32B-SFT 69.8"),
        ("32B RL 73.3", "MAI-UI-32B-RL 73.3 (+3.5)"),
        ("235B 76.7", "76.7% on AndroidWorld"),
        ("116 tasks", "116 tasks across 20 Android apps"),
        ("Qwen3-VL-2B 36.4", "Qwen3-VL-2B 2B 36.4"),
        ("Ferret-UI Lite-3B 28.0", "Ferret-UI Lite-3B 28.0"),
    ]:
        check(label, needle in raw)

    print()
    print("=" * 78)
    print("2. P(task) = p^n at our own measured per-step accuracy")
    print("=" * 78)
    expected = {5: 10.8, 8: 2.8, 10: 1.2, 15: 0.1}
    for n, want in expected.items():
        got = 100.0 * (OURS_PER_STEP ** n) * 100.0 / 100.0
        check("64.07%% over %2d steps = %.1f%%" % (n, want), abs(got - want) < 0.05, "%.2f%%" % got)

    print()
    print("=" * 78)
    print("3. the per-step rates §5.12.1 tabulates (n = 10)")
    print("=" * 78)
    printed = {
        "M3A (GPT-4 Turbo, a11y tree)": 88.8,
        "MAI-UI-2B SFT": 92.3,
        "MAI-UI-2B RL": 93.1,
        "MAI-UI-8B SFT": 95.7,
        "MAI-UI-8B RL": 96.6,
        "MAI-UI-32B SFT": 96.5,
        "MAI-UI-32B RL": 96.9,
        "MAI-UI-235B-A22B RL": 97.4,
    }
    for label, want in printed.items():
        got = 100.0 * per_step(E2E[label], 10)
        check("%-30s -> %.1f%%" % (label, want), abs(got - want) < 0.06, "%.2f%%" % got)

    print()
    print("=" * 78)
    print("4. the SFT/RL deltas the paper quotes")
    print("=" * 78)
    for fam, sft, rl, want_pp in [("2B", 0.451, 0.491, 4.0), ("8B", 0.647, 0.707, 6.0),
                                  ("32B", 0.698, 0.733, 3.5)]:
        got_pp = 100.0 * (rl - sft)
        check("%s end-to-end delta = +%.1f pp" % (fam, want_pp), abs(got_pp - want_pp) < 0.06,
              "+%.2f" % got_pp)
    d_ps = 100.0 * (per_step(0.491, 10) - per_step(0.451, 10))
    check("2B RL is worth +0.8 per-step points", abs(d_ps - 0.8) < 0.06, "+%.2f" % d_ps)

    print()
    print("=" * 78)
    print("5. the '28 points' gap between us and MAI-UI-2B-SFT")
    print("=" * 78)
    gap = 100.0 * per_step(0.451, 10) - 100.0 * OURS_PER_STEP
    check("gap = 28 per-step points", abs(gap - 28.2) < 0.4, "%.2f" % gap)

    print()
    print("=" * 78)
    print("6. sensitivity to n — stated in the paper, printed here so it cannot be hidden")
    print("=" * 78)
    for n in (5, 8, 10, 15):
        print("     n=%2d : 2B-SFT %.1f%%   our e2e %.2f%%"
              % (n, 100.0 * per_step(0.451, n), 100.0 * OURS_PER_STEP ** n))

    print()
    print("=" * 78)
    print("7. the paper prints what this script computes")
    print("=" * 78)
    paper = open(PAPER, encoding="utf-8").read()
    sec = paper[paper.index("### 5.12.1"):paper.index("### 5.13")]
    sec_flat = re.sub(r"\s+", " ", sec)
    for fig in ("10.8% over 5 steps", "2.8% over 8", "1.2% over 10", "0.1% over 15",
                "**45.1%**", "**92.3%**", "93.1%", "95.7%", "96.6%", "97.4%", "88.8%",
                "+4.0 points", "+0.8 per-step", "28 points of per-step"):
        check("§5.12.1 states %r" % fig, fig in sec_flat)

    print()
    if fails:
        print("%d FAILED — §5.12.1 does not re-derive from the primary capture." % len(fails))
        return 1
    print("ALL CHECKS PASS — §5.12.1 re-derives from references/raw/maiui2026-androidworld-table9.txt")
    return 0


def OURSSAFE(p, n):
    """P(task) = p^n, kept as a named function so the intent is unmistakable."""
    return p ** n


if __name__ == "__main__":
    sys.exit(main())
