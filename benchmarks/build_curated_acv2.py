"""Build a corrected choice-over-K decision dataset from AC-v2 (a11y-guided).

Fixes vs the old builder:
  * Gold uses **bbox E_bbox** (minimal-area UI element containing the GT point),
    matching AndroidControl-Curated's intent-aligned grounding — NOT first-match
    in tree order (which mis-grounded nested elements).
  * Candidate labels are **de-duplicated** (two identical candidates would be
    unanswerable for a classifier).
  * Non-spatial gold = `candidates.index(gt_type)` (exact), not off-by-one.
  * K-truncation keeps gold and never drops it (K-truncation-safe).
  * Rows whose action_type can't be grounded (e.g. `status`) are skipped.

Output JSONL per row:
  episode_id, step, goal, step_instruction, image, candidates (K), gold (idx), gt.
"""
import argparse
import json
import os

DATA = "benchmarks/data/acv2"
ROWS = f"{DATA}/500_steps_filtered.json"
DEFAULT_OUT = "benchmarks/data/acv2_candidates_curated.jsonl"

# non-spatial action types we expose as always-present candidates
NON_SPATIAL = ("wait", "navigate_back", "scroll", "open_app", "terminate")
# spatial action types whose GT point maps to an element bbox
SPATIAL = ("click", "long_press", "type_text")


def node_label(node):
    for key in ("text", "content_description", "resource_name", "class_name"):
        v = node.get(key)
        if v:
            return str(v).strip()
    return None


def bbox_contains(node, pt):
    bb = node.get("bbox_pixels")
    if not bb:
        return False
    return bb["x_min"] <= pt[0] <= bb["x_max"] and bb["y_min"] <= pt[1] <= bb["y_max"]


def node_area(node):
    bb = node.get("bbox_pixels")
    if not bb:
        return None
    return bb.get("area")


def candidate_list(row):
    """Return (candidates, gold) or (None, -1) if the row can't be grounded."""
    tree = row.get("accessibility_tree", [])
    gt = row.get("action", {})
    gt_type = (gt.get("action_type") or "").lower()
    gt_x, gt_y = gt.get("x"), gt.get("y")

    candidates = []          # unique candidate strings, in order
    cand_index = {}          # node id -> candidate index (first occurrence)
    gold = -1

    # 1. spatial affordances from a11y tree
    for node in tree:
        if not node.get("is_visible", True):
            continue
        clickable = node.get("is_clickable", False)
        editable = node.get("is_editable", False)
        label = node_label(node)
        if not (clickable or editable) or not label:
            continue
        kind = "write" if editable else "click"
        cand = f"{kind}: {label}"
        if cand not in cand_index:
            cand_index[cand] = len(candidates)
            candidates.append(cand)

    # 2. non-spatial action candidates (always present)
    for t in NON_SPATIAL:
        if t not in candidates:
            candidates.append(t)

    # 3. resolve gold
    if gt_type in SPATIAL and gt_x is not None and gt_y is not None:
        pt = (gt_x, gt_y)
        # E_bbox: minimal-area candidate node containing the point
        best_idx = -1
        best_area = None
        for node in tree:
            if not node.get("is_visible", True):
                continue
            if not (node.get("is_clickable", False) or node.get("is_editable", False)):
                continue
            label = node_label(node)
            if not label or not bbox_contains(node, pt):
                continue
            cand = ("write" if node.get("is_editable", False) else "click") + f": {label}"
            if cand not in cand_index:
                continue
            area = node_area(node)
            if best_area is None or (area is not None and area < best_area):
                best_area = area
                best_idx = cand_index[cand]
        gold = best_idx
    elif gt_type in NON_SPATIAL:
        try:
            gold = candidates.index(gt_type)
        except ValueError:
            gold = -1

    if gold is None or gold == -1:
        return None, -1
    return candidates, gold


def truncate(candidates, gold, k):
    """Keep gold + k-1 distractors, preserving gold's index (K-truncation-safe)."""
    if not k or len(candidates) <= k:
        return candidates, gold
    distract = [i for i in range(len(candidates)) if i != gold]
    keep = sorted(distract[: k - 1] + [gold])
    new_cands = [candidates[i] for i in keep]
    new_gold = keep.index(gold)
    return new_cands, new_gold


def build(rows, limit=None, k=None):
    out = []
    skipped = {"unmatched": 0}
    for r in rows:
        cands, gold = candidate_list(r)
        if gold == -1:
            skipped["unmatched"] += 1
            continue
        cands, gold = truncate(cands, gold, k)
        out.append({
            "episode_id": r.get("episode_id"),
            "step": r.get("step"),
            "goal": r.get("goal"),
            "step_instruction": r.get("step_instruction"),
            "image": f"benchmarks/data/acv2_shots/images/{r.get('screenshot')}",
            "candidates": cands,
            "gold": gold,
            "gt": r.get("action"),
        })
        if limit and len(out) >= limit:
            break
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=6, help="max candidates (0 = all)")
    args = ap.parse_args()

    rows = json.load(open(ROWS))
    if args.limit:
        rows = rows[:args.limit]
    out, skipped = build(rows, limit=args.limit, k=args.k)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    ok = all(0 <= o["gold"] < len(o["candidates"]) for o in out)
    print(f"wrote {len(out)} rows -> {args.out} | gold-in-range: {ok} | skipped(unmatched): {skipped['unmatched']}")
    from collections import Counter
    print("gold action-type dist:", Counter(o['gt']['action_type'] for o in out))
    if out:
        print("sample candidates:", out[0]["candidates"][:5])
        print("sample gold:", out[0]["gold"], "| gt:", out[0]["gt"])


if __name__ == "__main__":
    main()
