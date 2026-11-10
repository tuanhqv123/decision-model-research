#!/usr/bin/env python3
"""Stage 4 — why does the gold node fail to qualify as a candidate?

Reports the exact attribute profile of the gold node, split by whether it
survives the candidate filter. This decides what the filter should be, instead
of guessing.

Run: python3 scripts/audit_ac_raw.py --stage 4
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ac_raw import default_root, iter_rows  # noqa: E402
from audit_ac_raw2 import box, iou  # noqa: E402

CONTAINERS = {
    "android.widget.FrameLayout", "android.widget.LinearLayout",
    "android.widget.RelativeLayout", "android.widget.ScrollView",
    "android.widget.GridView", "android.widget.ListView",
    "androidx.recyclerview.widget.RecyclerView",
    "androidx.viewpager.widget.ViewPager", "android.view.ViewGroup",
    "android.widget.TableLayout", "android.widget.HorizontalScrollView",
}


def lab(n):
    return ((n.get("text") or "").strip()
            or (n.get("content_description") or "").strip())


def stage4(root, limit=20000):
    print("=" * 78)
    print("STAGE 4 — profile of the GOLD node (why it does / does not qualify)")
    print("=" * 78)

    rows = 0
    spatial = 0
    in_set = 0
    out_set = 0
    out_cls = collections.Counter()
    out_reason = collections.Counter()
    in_cls = collections.Counter()
    out_has_text = collections.Counter()
    out_clickable = collections.Counter()
    out_area = []
    in_area = []
    W_H = (1080, 2400)

    for d, shard, ln in iter_rows(root):
        if d is None:
            continue
        rows += 1
        if limit and rows > limit:
            break
        a = d.get("action") or {}
        if a.get("action_type") not in ("click", "long_press", "type_text"):
            continue
        t = d.get("target")
        if not t:
            continue
        spatial += 1
        bx = box(t)
        nodes = d.get("nodes") or []
        best, bestv = None, 0.0
        for n in nodes:
            nb = box(n.get("bbox_pixels"))
            if not nb:
                continue
            v = iou(nb, bx)
            if v > bestv:
                bestv, best = v, n
        if best is None or bestv < 0.999:
            continue

        cn = best.get("class_name") or "?"
        area = best["bbox_pixels"]["area"]
        # does it pass the filter?
        reasons = []
        if not best.get("bbox_pixels"):
            reasons.append("no bbox")
        if best.get("is_visible") is False:
            reasons.append("not visible")
        if cn in CONTAINERS:
            reasons.append("container class")
        if not lab(best) and not (best.get("resource_name") or "").strip():
            reasons.append("no label/resource")
        if not reasons:
            in_set += 1
            in_cls[cn] += 1
            in_area.append(area)
        else:
            out_set += 1
            out_reason[reasons[0]] += 1
            out_cls[cn] += 1
            out_has_text[bool(lab(best))] += 1
            out_clickable[bool(best.get("is_clickable") or best.get("is_long_clickable") or best.get("is_editable"))] += 1
            out_area.append(area)

    print(f"\nspatial rows scanned: {spatial}")
    print(f"gold PASSES filter: {in_set} ({100*in_set/max(spatial,1):.1f}%)")
    print(f"gold FAILS  filter: {out_set} ({100*out_set/max(spatial,1):.1f}%)")

    print("\n--- why the gold fails ---")
    for k, v in out_reason.most_common():
        print(f"  {k:22s} {v:6d}  ({100*v/max(out_set,1):.1f}% of failures)")

    print("\n--- class_name of gold that FAILS (top 10) ---")
    for k, v in out_cls.most_common(10):
        print(f"  {k:48s} {v}")

    print("\n--- class_name of gold that PASSES (top 10) ---")
    for k, v in in_cls.most_common(10):
        print(f"  {k:48s} {v}")

    print("\n--- gold that fails: does it have text / clickable? ---")
    print(f"  has text: {dict(out_has_text)}")
    print(f"  is clickable/editable/long: {dict(out_clickable)}")

    import statistics as st
    if in_area:
        print(f"\ngold area (passes): median {st.median(in_area):.0f} px^2")
    if out_area:
        print(f"gold area (fails ) : median {st.median(out_area):.0f} px^2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--limit", type=int, default=20000)
    a = ap.parse_args()
    stage4(a.root or default_root(), a.limit)
