#!/usr/bin/env python3
"""Stage 3 — is the 73% duplicate-label rate real ambiguity, or an artifact?

The check: duplicate labels counted over ALL nodes include container nodes that
repeat their child's text, and nodes with no text at all. A *candidate* list is
what a selector actually chooses from, so the ambiguity that matters is the
ambiguity within the candidate set, not within the raw tree.

Also measures: how often is the gold target node itself duplicated (i.e. a
different node with the same label sits elsewhere on screen)?

Run: python3 scripts/audit_ac_raw.py --stage 3
"""
import argparse
import collections
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ac_raw import default_root, iter_rows  # noqa: E402
from audit_ac_raw2 import box, iou, contains  # noqa: E402

CONTAINERS = {
    "android.widget.FrameLayout", "android.widget.LinearLayout",
    "android.widget.RelativeLayout", "android.widget.ScrollView",
    "android.widget.GridView", "android.widget.ListView",
    "androidx.recyclerview.widget.RecyclerView",
    "androidx.viewpager.widget.ViewPager", "android.view.ViewGroup",
    "android.widget.TableLayout", "android.widget.HorizontalScrollView",
}


def label_of(n):
    t = (n.get("text") or "").strip()
    if t:
        return t
    cd = (n.get("content_description") or "").strip()
    if cd:
        return cd
    return ""


def candidate(n, drop_container=True, need_label=True):
    b = n.get("bbox_pixels")
    if not b or b.get("area", 0) <= 0:
        return False
    if n.get("is_visible") is False:
        return False
    if b["x_min"] >= b["x_max"] or b["y_min"] >= b["y_max"]:
        return False
    if drop_container and (n.get("class_name") in CONTAINERS):
        return False
    if need_label and not label_of(n) and not (n.get("resource_name") or "").strip():
        return False
    return True


def stage3(root, limit=None):
    print("=" * 78)
    print("STAGE 3 — is the duplicate-label rate real?")
    print("=" * 78)

    rows = 0
    all_sets = []
    cand_sets = []
    gold_label_dup = 0
    gold_in_cands = 0
    gold_iou1 = 0
    spatial = 0
    cand_sizes = []
    all_sizes = []
    unlabeled_nodes = 0
    total_nodes = 0
    container_nodes = 0

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
        nodes = d.get("nodes") or []
        total_nodes += len(nodes)

        lbl_all = collections.Counter()
        for n in nodes:
            s = label_of(n)
            if not s:
                unlabeled_nodes += 1
                continue
            if n.get("class_name") in CONTAINERS:
                container_nodes += 1
            lbl_all[s] += 1
        all_sets.append(lbl_all)
        all_sizes.append(len(nodes))

        cands = [n for n in nodes if candidate(n)]
        cand_sizes.append(len(cands))
        lbl_c = collections.Counter(label_of(n) for n in cands if label_of(n))
        cand_sets.append(lbl_c)

        bx = box(t)
        # the gold node = the node with the highest IoU with the target bbox
        best, bestv = None, 0.0
        for n in nodes:
            nb = box(n.get("bbox_pixels"))
            if not nb:
                continue
            v = iou(nb, bx)
            if v > bestv:
                bestv, best = v, n
        if best is not None and bestv > 0.999:
            gold_iou1 += 1
        if best is not None and any(candidate(c) for c in [best]):
            gold_in_cands += 1
        gl = label_of(best) if best else ""
        if gl and lbl_c.get(gl, 0) > 1:
            gold_label_dup += 1

    def dup_rate(sets):
        n = sum(1 for s in sets if any(v > 1 for v in s.values()))
        return n, len(sets)

    a_dup, a_n = dup_rate(all_sets)
    c_dup, c_n = dup_rate(cand_sets)

    print(f"\nspatial rows with a target: {spatial}  (of {rows} rows)")
    print(f"gold bbox matches a node at IoU>0.999: {gold_iou1} "
          f"({100*gold_iou1/max(spatial,1):.1f}%)")

    print("\n--- duplicate labels: ALL nodes vs CANDIDATE nodes ---")
    print(f"over ALL nodes      : {a_dup}/{a_n} rows have a repeated label "
          f"({100*a_dup/max(a_n,1):.1f}%)")
    print(f"over CANDIDATE nodes: {c_dup}/{c_n} rows have a repeated label "
          f"({100*c_dup/max(c_n,1):.1f}%)")
    print("  -> the drop from ALL to CANDIDATE is the size of the artifact.")

    print("\n--- is the GOLD node's own label ambiguous? ---")
    print(f"gold label shared by another candidate: {gold_label_dup} "
          f"({100*gold_label_dup/max(spatial,1):.1f}% of spatial rows)")
    print("  -> this is the ambiguity that a selector cannot resolve. It bounds accuracy.")

    print("\n--- node counts ---")
    print(f"nodes per screen      : median {st.median(all_sizes):.0f} mean {st.mean(all_sizes):.1f} "
          f"p90 {sorted(all_sizes)[int(.9*len(all_sizes))]} max {max(all_sizes)}")
    print(f"candidates per screen : median {st.median(cand_sizes):.0f} mean {st.mean(cand_sizes):.1f} "
          f"p90 {sorted(cand_sizes)[int(.9*len(cand_sizes))]} max {max(cand_sizes)}")
    print(f"nodes with NO label at all: {unlabeled_nodes}/{total_nodes} "
          f"({100*unlabeled_nodes/max(total_nodes,1):.1f}%)")
    print(f"container-class nodes with a label: {container_nodes}")
    print(f"gold node is in the candidate set: {gold_in_cands}/{spatial} "
          f"({100*gold_in_cands/max(spatial,1):.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    stage3(a.root or default_root(), a.limit)
