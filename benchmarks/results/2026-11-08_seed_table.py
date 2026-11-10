#!/usr/bin/env python3
"""Re-derive paper §5.9's seed table from the eval artifacts on this disk.

§5.9 is the paper's only surviving positive claim, so it must not rest on a table anyone typed. The two
recipes differ only in the training file: `compact_cur15k.jsonl` (15,000 rows) vs `compact_ctrl22k.jsonl`
(the same rows shuffled and duplicated to 22,500). Everything else is identical (init, 938 steps, lr, warmup,
720 px, track V, no memory).

Grounding is a **count out of 223** gold-spatial rows, so two runs can legitimately land on the same value.
That is why 132/223 = 59.19% appears three times in the 22.5k arm: it is a discrete quantity, not a padded
table. (Checked 2026-11-08 after suspecting padding; the suspicion was wrong.)

    python3 benchmarks/results/2026-11-08_seed_table.py
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SEEDS = os.path.join(ROOT, "backup-2026-11-07", "seed_evals")

# artifact -> (recipe, seed). Only files that exist are used; a missing one is reported, never guessed.
PLAN = {
    "eval-E2-nomem":    ("15k",   0),
    "eval-seed1-V":     ("15k",   1),
    "eval-s15k_s2-V":   ("15k",   2),
    "eval-ctrl-V":      ("22.5k", 0),
    "eval-ctrl_s2-V":   ("22.5k", 2),
    "eval-ctrl_s3-V":   ("22.5k", 3),
    "eval-ctrl_s4-V":   ("22.5k", 4),
}

SPATIAL_DEN = 223   # the official scorer's gold-spatial denominator on the 398-step protocol


def score(path):
    """Official definitions: step = `correct`; grounding = `grounding` True over the 223 spatial golds."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows)
    step = sum(1 for r in rows if r.get("correct"))
    gt = sum(1 for r in rows if r.get("grounding") is True)
    return n, step, gt


def main():
    # E2's pure artifact lives outside seed_evals (it is the project baseline).
    e2 = os.path.join(ROOT, "backup-2026-10-29", "GPU_HOST_A", "acwork", "runs",
                      "E2-nomem", "eval-E2nomem-V.jsonl")
    alt = os.path.join(ROOT, "backup-2026-10-29", "GPU_HOST_A", "acwork", "runs",
                       "E2-nomem", "eval-E2final-nomem-V.jsonl")
    e2 = e2 if os.path.exists(e2) else alt

    found = {}
    for key, (recipe, seed) in PLAN.items():
        path = e2 if key == "eval-E2-nomem" else os.path.join(SEEDS, key + ".jsonl")
        if not os.path.exists(path):
            continue
        n, step, gt = score(path)
        found.setdefault(recipe, []).append((seed, gt, n, step, os.path.basename(path)))

    print("§5.9 — re-derived from artifacts\n")
    print("%-7s %-5s %-30s %5s %8s %8s" % ("recipe", "seed", "artifact", "n", "step%", "ground%"))
    ok = True
    for recipe in ("15k", "22.5k"):
        runs = sorted(found.get(recipe, []))
        if not runs:
            print("%-7s  NO ARTIFACTS FOUND" % recipe)
            ok = False
            continue
        for seed, gt, n, step, name in runs:
            print("%-7s %-5d %-30s %5d %7.2f%% %7.2f%%" % (
                recipe, seed, name, n, step / n * 100, gt / SPATIAL_DEN * 100))
        vals = [gt / SPATIAL_DEN * 100 for _, gt, _, _, _ in runs]
        print("%-7s %-5s n=%d mean %.2f  sd %.2f  values %s\n" % (
            recipe, "", len(vals), st.mean(vals),
            st.stdev(vals) if len(vals) > 1 else 0.0,
            ", ".join("%.2f" % v for v in vals)))

    a = [gt for _, gt, _, _, _ in sorted(found.get("15k", []))]
    b = [gt for _, gt, _, _, _ in sorted(found.get("22.5k", []))]
    if len(a) >= 2 and len(b) >= 2:
        va = [x / SPATIAL_DEN * 100 for x in a]
        vb = [x / SPATIAL_DEN * 100 for x in b]
        diff = st.mean(vb) - st.mean(va)
        sa, sb = st.stdev(va), st.stdev(vb)
        pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
        se = pooled * (1 / len(va) + 1 / len(vb)) ** 0.5
        t = diff / se if se else float("inf")
        df = len(va) + len(vb) - 2
        print("difference %+.2f points, pooled sd %.2f, t = %.2f, df = %d" % (diff, pooled, t, df))
        # What §5.9 now states, after the 2026-11-08 correction to match these artifacts.
        print("PAPER TABLE CLAIMS: 15k 56.80 +/- 1.29 (n=3), 22.5k 59.42 +/- 0.45 (n=4), "
              "diff +2.62, t=3.54, df=5")
        means_ok = round(st.mean(va), 2) == 56.80 and round(st.mean(vb), 2) == 59.42
        stats_ok = round(diff, 2) == 2.62 and round(t, 2) == 3.54 and df == 5
        print("MEANS REPRODUCE:", "YES" if means_ok else "NO")
        print("DIFF/t/df REPRODUCE:", "YES" if stats_ok else "NO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
