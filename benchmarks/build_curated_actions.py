"""Build a calibrated single-pass action-type decision dataset from AndroidControl-Curated.

LOCKED benchmark (2026-10-15): AndroidControl-Curated (high split, bbox-corrected),
task L1 = action-type choice over K coarse types, evaluated on a held-out episode split.

Data source: benchmarks/data/curated/android_control_high_bbox.json
  row -> image, instruction(goal), history, step_instruction, gt_action,
         gt_min_bbox, gt_max_bbox, gt_coordinate
Images: all curated images are present locally (acv2_shots/images/<ep>/screenshot_<n>.png),
  mapped via android_control_images/<ep>/step_<n>.png -> <ep>/screenshot_<n>.png.

Output JSONL per row:
  episode_id, step, goal, step_instruction, image, action (coarse), gt_action (raw),
  has_bbox (bool), coord (or null), bbox (or null).

Usage:
  python3 benchmarks/build_curated_actions.py --out benchmarks/data/curated_actions.jsonl
"""
import argparse, json, os, re, collections

SRC = "benchmarks/data/curated/android_control_high_bbox.json"
SHOTS = "benchmarks/data/acv2_shots/images"


def coarse_action(a):
    a = (a or "")
    if a.startswith("input_text"): return "input_text"
    if a.startswith("open_app"):   return "open_app"
    if a.startswith("swipe"):      return "swipe"
    if a.startswith("long_press"): return "long_press"
    if a.startswith("navigate_back"): return "navigate_back"
    if a.startswith("wait"):       return "wait"
    if a.startswith("scroll"):     return "scroll"
    if a.startswith("click"):      return "click"
    return "other"


def local_image(img):
    m = re.match(r"android_control_images/(\d+)/step_(\d+)\.png", img)
    if not m: return None
    return f"{SHOTS}/{m.group(1)}/screenshot_{m.group(2)}.png"


def build(out, limit=None):
    rows = json.load(open(SRC))
    if limit: rows = rows[:limit]
    ep = collections.defaultdict(list)
    for r in rows:
        loc = local_image(r["image"])
        if not loc or not os.path.exists(loc):
            continue
        raw = r["gt_action"]
        has_bbox = bool(r["gt_max_bbox"]) or bool(r["gt_min_bbox"])
        coord = r["gt_coordinate"] or None
        bb = r["gt_min_bbox"] or r["gt_max_bbox"] or None
        ep[r["image"].split("/")[1]].append({
            "episode_id": int(r["image"].split("/")[1]),
            "step": int(r["image"].split("/")[2].split(".")[0].split("_")[1]),
            "goal": r.get("instruction") or "",
            "step_instruction": r.get("step_instruction") or "",
            "image": loc,
            "action": coarse_action(raw),
            "gt_action": raw,
            "has_bbox": has_bbox,
            "coord": coord,
            "bbox": bb,
        })
    # flatten keeping episode-grouping implicit via episode_id
    all_rows = [x for ep_rows in ep.values() for x in ep_rows]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for x in all_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_rows)} rows -> {out}")
    print("action dist:", dict(collections.Counter(x['action'] for x in all_rows)))
    print("rows with coord:", sum(1 for x in all_rows if x['coord']), "| with bbox:", sum(1 for x in all_rows if x['bbox']))
    return all_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/data/curated_actions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    build(args.out, args.limit)
