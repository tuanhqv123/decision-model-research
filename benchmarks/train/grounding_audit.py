"""Grounding audit: what are the "wrong element" clicks actually hitting?

The AC-v2 scorer credits a spatial step iff the predicted point falls inside the *smallest* a11y
node containing the gold point. That is an exact-element criterion. It cannot distinguish

  (a) a genuinely different widget            -> real element-selection error
  (b) the same widget, a different subtree    -> scorer brittleness (parent/child/overlap)
  (c) a sibling/duplicate with the same label -> label ambiguity in the dataset
  (d) a gold node so small the task is near-impossible

This script joins saved predictions with the eval set's full accessibility trees and partitions the
misses into those buckets, plus the gold node area distribution and per-node label ambiguity. It is an
audit of the *measurement*, not a new score: nothing here changes the official number.

Usage:
  python grounding_audit.py --rows benchmarks/data/acv2/500_steps_filtered.json \
      --pred benchmarks/results/2026-10-29-evals/E2-938-V.jsonl [--json out.json]
"""
import argparse
import json

import numpy as np

SPATIAL = {"click", "long_press", "type_text"}


def walk(nodes, out):
    for n in nodes:
        if isinstance(n, dict):
            b = n.get("bbox_pixels")
            if b and b.get("width", 0) > 0 and b.get("height", 0) > 0:
                out.append(n)
        elif isinstance(n, list):
            walk(n, out)
    return out


def flat(tree):
    return walk(tree, [])


def label_of(n):
    return " ".join(str(n.get("text") or n.get("content_description") or n.get("hint_text") or "").split())


def contains(b, x, y):
    return b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"]


def smallest(nodes, x, y):
    best, best_area = None, float("inf")
    for n in nodes:
        b = n["bbox_pixels"]
        if contains(b, x, y):
            a = b["width"] * b["height"]
            if a < best_area:
                best, best_area = n, a
    return best, best_area


def iou(a, b):
    x0, y0 = max(a["x_min"], b["x_min"]), max(a["y_min"], b["y_min"])
    x1, y1 = min(a["x_max"], b["x_max"]), min(a["y_max"], b["y_max"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / ua if ua > 0 else 0.0


def is_ancestor(anc, desc):
    a, d = anc["bbox_pixels"], desc["bbox_pixels"]
    return (a["x_min"] <= d["x_min"] and a["y_min"] <= d["y_min"]
            and a["x_max"] >= d["x_max"] and a["y_max"] >= d["y_max"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="benchmarks/data/acv2/500_steps_filtered.json")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()

    samples = {(r["episode_id"], r["step"]): r for r in json.load(open(a.rows))}
    preds = [json.loads(l) for l in open(a.pred) if l.strip()]

    spatial = [p for p in preds if p["gold"]["action_type"] in SPATIAL]
    n_sp = len(spatial)
    hits = misses = 0
    cats = {"same_label": 0, "subtree": 0, "iou_ge_0.5": 0, "overlaps_gold": 0, "different_element": 0}
    no_pred_node = 0
    gold_areas, hit_areas, miss_areas = [], [], []
    ambiguous = 0
    ambiguous_and_missed = 0
    examples = []

    for p in spatial:
        key = (p["episode_id"], p["step"])
        s = samples.get(key)
        if not s or p.get("pt") is None:
            continue
        nodes = flat(s["accessibility_tree"])
        gx, gy = p["gold"]["x"], p["gold"]["y"]
        px, py = p["pt"]
        gn, _ = smallest(nodes, gx, gy)
        if gn is None:
            continue
        gb = gn["bbox_pixels"]
        gold_areas.append(gb["width"] * gb["height"])

        glabel = label_of(gn)
        same = [n for n in nodes if glabel and label_of(n) == glabel]
        if len(same) > 1:
            ambiguous += 1

        if contains(gb, px, py):
            hits += 1
            hit_areas.append(gb["width"] * gb["height"])
            continue

        misses += 1
        miss_areas.append(gb["width"] * gb["height"])
        if len(same) > 1:
            ambiguous_and_missed += 1
        pn, _ = smallest(nodes, px, py)
        if pn is None:
            no_pred_node += 1
            cats["different_element"] += 1
            continue
        plabel = label_of(pn)
        if glabel and plabel == glabel:
            cats["same_label"] += 1
            k = "same_label"
        elif is_ancestor(pn, gn) or is_ancestor(gn, pn):
            cats["subtree"] += 1
            k = "subtree"
        elif iou(gb, pn["bbox_pixels"]) >= 0.5:
            cats["iou_ge_0.5"] += 1
            k = "iou_ge_0.5"
        elif iou(gb, pn["bbox_pixels"]) > 0:
            cats["overlaps_gold"] += 1
            k = "overlaps_gold"
        else:
            cats["different_element"] += 1
            k = "different_element"
        if len(examples) < 400:
            examples.append({"key": key, "bucket": k, "gold_label": glabel[:40],
                             "pred_label": plabel[:40], "iou": round(iou(gb, pn["bbox_pixels"]), 3)})

    tot = hits + misses
    print(f"\n=== grounding audit  ({a.pred})")
    print(f"  spatial rows joined                : {tot}")
    print(f"  exact-element hit (official metric) : {hits}/{tot} = {100*hits/tot:.2f}%")
    print(f"  misses                             : {misses}")
    print("\n  anatomy of the misses:")
    for k in ("same_label", "subtree", "iou_ge_0.5", "overlaps_gold", "different_element"):
        c = cats[k]
        print(f"    {k:20s} {c:4d}  = {100*c/misses:5.1f}% of misses   = {100*c/tot:5.2f} pts of accuracy")
    tolerant = hits + cats["same_label"] + cats["subtree"] + cats["iou_ge_0.5"]
    print(f"\n  tolerance curve (diagnostic only, NOT a new headline):")
    print(f"    exact element            : {100*hits/tot:6.2f}%")
    print(f"    + same label (ambiguity) : {100*(hits+cats['same_label'])/tot:6.2f}%")
    print(f"    + subtree relation       : {100*(hits+cats['same_label']+cats['subtree'])/tot:6.2f}%")
    print(f"    + IoU>=0.5               : {100*tolerant/tot:6.2f}%")
    ga, ha, ma = map(np.array, (gold_areas, hit_areas, miss_areas))
    if len(ga):
        print(f"\n  gold element area (px^2), fraction of screen:")
        scr = [s["screenshot_width"] * s["screenshot_height"] for s in samples.values()]
        frac = ga / np.mean(scr)
        print(f"    median {np.median(frac)*100:.3f}%   p10 {np.percentile(frac,10)*100:.4f}%   "
              f"p90 {np.percentile(frac,90)*100:.2f}%")
        for lim in (0.0005, 0.001, 0.005, 0.01, 0.05):
            m = frac <= lim
            if m.sum() >= 5:
                print(f"    gold area <= {lim*100:.2f}% of screen: n={int(m.sum()):3d}  "
                      f"(share of all spatial rows {100*m.mean():.1f}%)")
    print(f"\n  label ambiguity: gold label shared by >1 node on {ambiguous}/{tot} rows "
          f"({100*ambiguous/tot:.1f}%); {ambiguous_and_missed} of those were missed")
    if len(ha) and len(ma):
        print(f"  gold area when hit: median {np.median(ha):,.0f} px^2 | when missed: {np.median(ma):,.0f} px^2")

    res = {"pred": a.pred, "spatial_joined": tot, "hits": hits,
           "exact_hit_rate": hits / tot if tot else None, "cats": cats,
           "tolerant_hit_rate": tolerant / tot if tot else None,
           "ambiguous_rows": ambiguous, "ambiguous_missed": ambiguous_and_missed}
    if a.json:
        json.dump({"summary": res, "examples": examples}, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
