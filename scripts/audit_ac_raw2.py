#!/usr/bin/env python3
"""Stage 2 — label sanity and real-world plausibility for AndroidControl raw records.

Checks the things a data scientist must check before trusting a label:
  - does the gold `target` bbox correspond to an actual node in `nodes`?
  - are coordinates within the declared screen W/H?
  - is the action type consistent with the target (click needs a target, wait should not)?
  - is the `status` row annotated or synthesised?
  - do duplicate labels exist on screen (two nodes with the same text)?
  - real-world plausibility: does the goal name an app that the action opens?

Run: python3 scripts/audit_ac_raw.py --stage 2
"""
import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ac_raw import default_root, iter_rows  # noqa: E402

SPATIAL = ("click", "long_press", "type_text")


def box(b):
    return (b["x_min"], b["y_min"], b["x_max"], b["y_max"]) if b else None


def contains(outer, inner, tol=2):
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and outer[2] + tol >= inner[2] and outer[3] + tol >= inner[3])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ba = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ba - inter)


def stage2(root, limit=None):
    print("=" * 78)
    print("STAGE 2 — label sanity / real-world plausibility")
    print("=" * 78)

    rows = 0
    with_target = 0
    target_matches_node = 0
    target_exact_node = 0
    target_no_node = 0
    oob = 0
    zero_area = 0
    inverted = 0
    type_vs_target = collections.Counter()
    status_rows = 0
    status_is_last = 0
    status_not_last = 0
    goal_open_app_mismatch = 0
    goal_open_app_checked = 0
    dup_label_rows = 0
    dup_label_rows_multi = 0
    label_counts = []
    eps = collections.defaultdict(list)

    for d, shard, ln in iter_rows(root):
        if d is None:
            continue
        rows += 1
        if limit and rows > limit:
            break
        a = d.get("action") or {}
        at = a.get("action_type")
        t = d.get("target")
        W, H = d.get("W"), d.get("H")
        eps[d["ep"]].append(d)

        if at == "status":
            status_rows += 1

        # ---- target geometry checks
        if t:
            with_target += 1
            bx = box(t)
            if bx[0] >= bx[2] or bx[1] >= bx[3]:
                inverted += 1
            if t.get("area", 0) <= 0:
                zero_area += 1
            if W and H and (bx[0] < 0 or bx[1] < 0 or bx[2] > W or bx[3] > H):
                oob += 1
            # ---- does a node match this target?
            best = None
            for n in d.get("nodes") or []:
                nb = box(n.get("bbox_pixels"))
                if not nb:
                    continue
                v = iou(nb, bx)
                if best is None or v > best:
                    best = v
            if best is None or best == 0:
                target_no_node += 1
            else:
                target_matches_node += 1
                if best > 0.999:
                    target_exact_node += 1

        type_vs_target[(at, t is not None)] += 1

        # ---- duplicate labels on screen (real ambiguity)
        lbl = collections.Counter()
        for n in d.get("nodes") or []:
            s = (n.get("text") or n.get("content_description") or "").strip()
            if s:
                lbl[s] += 1
        if lbl:
            label_counts.append(len(lbl))
            if any(v > 1 for v in lbl.values()):
                dup_label_rows += 1
                if sum(v for v in lbl.values() if v > 1) > 2:
                    dup_label_rows_multi += 1

        # ---- real-world plausibility: open_app names an app present in the goal?
        if at == "open_app":
            goal_open_app_checked += 1
            name = (a.get("app_name") or "").strip().lower()
            g = (d.get("goal") or "").lower()
            if name and name.split()[0] not in g:
                goal_open_app_mismatch += 1

    print(f"\nrows examined: {rows}")
    print("\n--- target geometry ---")
    print(f"rows with a target bbox      : {with_target} ({100*with_target/rows:.1f}%)")
    print(f"  target matches SOME node>0 : {target_matches_node} "
          f"({100*target_matches_node/max(with_target,1):.1f}% of targets)")
    print(f"  target matches a node EXACT: {target_exact_node} "
          f"({100*target_exact_node/max(with_target,1):.1f}%)")
    print(f"  target matches NO node     : {target_no_node}")
    print(f"  inverted bbox              : {inverted}")
    print(f"  zero area                  : {zero_area}")
    print(f"  out of bounds              : {oob}")

    print("\n--- action type vs target presence ---")
    for k in sorted(type_vs_target, key=lambda x: str(x)):
        print(f"  {str(k[0]):14s} has_target={k[1]}: {type_vs_target[k]}")

    print("\n--- status rows ---")
    print(f"status rows: {status_rows}")
    for ep, rs in eps.items():
        rs = sorted(rs, key=lambda r: r["step"])
        idx = [i for i, r in enumerate(rs) if r["action"].get("action_type") == "status"]
        if not idx:
            continue
        if idx == [len(rs) - 1]:
            status_is_last += 1
        else:
            status_not_last += 1
    print(f"  episodes where status is the FINAL row: {status_is_last}")
    print(f"  episodes where status is NOT final    : {status_not_last}")

    print("\n--- ambiguity: duplicate labels on one screen ---")
    print(f"rows with >=1 repeated label: {dup_label_rows} "
          f"({100*dup_label_rows/max(len(label_counts),1):.2f}%)")
    print(f"  of which >2 nodes share a label: {dup_label_rows_multi}")

    print("\n--- plausibility: open_app name vs goal text ---")
    print(f"open_app rows checked: {goal_open_app_checked}")
    print(f"  app name not found in goal text: {goal_open_app_mismatch} "
          f"({100*goal_open_app_mismatch/max(goal_open_app_checked,1):.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    stage2(a.root or default_root(), a.limit)
