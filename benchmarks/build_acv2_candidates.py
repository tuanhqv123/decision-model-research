"""Build a choice-over-K candidate dataset from AC-v2 (a11y-guided, text-state).

Purpose: produce the input format for our calibrated single-pass decision head.
The head scores K candidate actions (real UI affordances from each step's
accessibility tree + the 8-type action space) and returns a calibrated prob
vector; the gold is the candidate that matches the ground-truth action.

This is the a11y-guided (text-state) path — DIFFERENT modality from the strict
vision-only High protocol. Never present it as beating SOTA on the unmodified
High-v2 protocol (see notes/sota-arch-deep.md §10).

Output: JSONL with, per row:
  episode_id, step, goal, step_instruction, image (screenshot path),
  candidates (list of K action strings), gold (index into candidates),
  gt (raw ground-truth action dict).

Usage:
  python3 benchmarks/build_acv2_candidates.py [--out ...] [--limit N] [--k K]
"""
import argparse
import json
import os
import re

DATA = "benchmarks/data/acv2"
ROWS = f"{DATA}/500_steps_filtered.json"
DEFAULT_OUT = "benchmarks/data/acv2_candidates.jsonl"


# ---- helpers -------------------------------------------------------------

def node_label(node):
    """A short human/agent-readable label for an a11y node."""
    for key in ("text", "content_description", "resource_name", "class_name"):
        v = node.get(key)
        if v:
            return str(v)
    return None


def find_click_target(node, pt):
    """Return True if the node's bbox (pixels) contains the point (x, y)."""
    bb = node.get("bbox_pixels")
    if not bb:
        return False
    return bb["x_min"] <= pt[0] <= bb["x_max"] and bb["y_min"] <= pt[1] <= bb["y_max"]


def candidate_list(row):
    """Build (candidates, gold_index) for a row. gold_index = -1 if not matched."""
    tree = row.get("accessibility_tree", [])
    gt = row.get("action", {})
    gt_type = (gt.get("action_type") or "").lower()
    gt_text = (gt.get("text") or "").strip()

    candidates = []
    gold = -1
    # Collect tappable / editable affordances from the a11y tree.
    for node in tree:
        if not node.get("is_visible", True):
            continue
        clickable = node.get("is_clickable", False)
        editable = node.get("is_editable", False)
        label = node_label(node)
        if not clickable and not editable:
            continue
        if not label:
            continue
        kind = "write" if editable else "click"
        cand = f"{kind}: {label}"
        # Determine gold for spatial steps (click / write) via point containment.
        if gold == -1 and gt_type in ("click", "long_press", "type_text") and gt.get("x") is not None:
            if find_click_target(node, (gt["x"], gt["y"])):
                gold = len(candidates)
        candidates.append(cand)

    # Non-spatial action candidates (always present, so the head can choose them).
    for t in ("wait", "navigate_back", "scroll", "open_app", "terminate"):
        cand = t
        candidates.append(cand)

    # If the GT is itself a non-spatial action type, locate the exact candidate
    # string in the candidate list (robust to append order / de-dup), instead of
    # the old off-by-one `len(candidates) - offset` formula which indexed the
    # WRONG candidate for every non-spatial action.
    if gold == -1 and gt_type in ("wait", "navigate_back", "scroll", "open_app", "terminate"):
        try:
            gold = candidates.index(gt_type)
        except ValueError:
            gold = -1

    return candidates, gold


def build(rows, limit=None, k=None):
    out = []
    for r in rows:
        cands, gold = candidate_list(r)
        if gold == -1:
            # skip rows we can't ground (better to keep the split clean)
            continue
        if k and len(cands) > k:
            # Keep the gold candidate plus the k-1 nearest non-gold distractors,
            # preserving gold's index validity (never truncate gold out).
            if gold == -1:
                cands = cands[:k]
            else:
                distract = [i for i in range(len(cands)) if i != gold]
                keep = sorted(distract[:k - 1] + [gold])
                cands = [cands[i] for i in keep]
                gold = keep.index(gold)
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=6, help="max candidates per row (0 = all)")
    args = ap.parse_args()

    rows = json.load(open(ROWS))
    if args.limit:
        rows = rows[:args.limit]
    out = build(rows, limit=args.limit, k=args.k)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    n = len(out)
    golds = [o["gold"] for o in out]
    # sanity: gold index in range
    ok = all(0 <= g < len(o["candidates"]) for o, g in zip(out, golds))
    print(f"wrote {n} rows -> {args.out} | gold-in-range: {ok}")
    if out:
        print("sample candidates:", out[0]["candidates"][:5])
        print("sample gold:", out[0]["gold"], "| gt:", out[0]["gt"])


if __name__ == "__main__":
    main()
