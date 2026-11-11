"""AndroidControl (GCS TFRecords) -> step records for d1-2B fine-tuning.

One record per decision the AC-v2 eval asks for:
  - every UI action step i  : state = screenshot_i + goal, label = action_i
  - one status step per ep  : state = screenshot_T (final, after the last action), label = status/successful
    (this mirrors how the AC-v2 398 set builds its 87 status rows: the screen after the last UI action)

Labels use the eval's naming (click, long_press, type_text, scroll, open_app, navigate_back, wait, status).
`nodes` keeps the eval's flat a11y format (bbox_pixels dict etc.) so the eval's candidate builder and scorer
apply unchanged. `target` = the SMALLEST node containing the gold point (exactly the scorer's target element).
type_text has no point in AndroidControl; like the AC-v2 set, it is located at the focused editable node,
else at the previous click.

Test-split episodes are dropped (and the 350 AC-v2 eval episodes are asserted absent).

Usage: python preprocess_ac.py --raw ~/acwork/ac_raw --out ~/acwork/acdata --workers 8
"""
import argparse
import glob
import gzip
import io
import json
import os
import struct
from multiprocessing import Pool

IMG_W = 720  # stored width; 1080x2400 -> 720x1600 (aspect kept)


def save_screenshot(png, path):
    """THE screenshot renderer for every stored AndroidControl image (training, validation, and the test set used only
    for scoring): RGB, IMG_W wide, aspect kept, bicubic, JPEG quality 90."""
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    im.resize((IMG_W, round(im.height * IMG_W / im.width)), Image.BICUBIC).save(path, quality=90)


def records(path):
    with gzip.open(path, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                return
            n = struct.unpack("<Q", h)[0]
            f.read(4)
            d = f.read(n)
            f.read(4)
            yield d


def flat_nodes(forest_bytes):
    from android_env.proto.a11y import android_accessibility_forest_pb2 as P
    forest = P.AndroidAccessibilityForest()
    forest.ParseFromString(forest_bytes)
    out = []
    for w in forest.windows:
        for n in w.tree.nodes:
            b = n.bounds_in_screen
            x0, y0, x1, y1 = b.left, b.top, b.right, b.bottom
            out.append({
                "text": n.text or None,
                "content_description": n.content_description or None,
                "hint_text": n.hint_text or None,
                "class_name": n.class_name or None,
                "bbox_pixels": {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1,
                                "center": [(x0 + x1) / 2, (y0 + y1) / 2],
                                "width": x1 - x0, "height": y1 - y0, "area": (x1 - x0) * (y1 - y0)},
                "is_clickable": n.is_clickable, "is_editable": n.is_editable,
                "is_scrollable": n.is_scrollable, "is_focused": n.is_focused,
                "is_long_clickable": n.is_long_clickable,
                "is_visible": n.is_visible_to_user, "package_name": n.package_name or None,
            })
    return out


def smallest(nodes, x, y):
    best, area = None, float("inf")
    for n in nodes:
        b = n["bbox_pixels"]
        if b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"]:
            a = (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
            if a < area:
                best, area = n, a
    return best


def convert_action(a):
    t = a["action_type"]
    if t == "input_text":
        return {"action_type": "type_text", "text": a.get("text", "")}
    if t == "scroll":
        return {"action_type": "scroll", "direction": a["direction"]}
    if t == "open_app":
        return {"action_type": "open_app", "app_name": a.get("app_name", "")}
    if t in ("click", "long_press"):
        return {"action_type": t, "x": a["x"], "y": a["y"]}
    return {"action_type": t}  # navigate_back, navigate_home, wait


def do_shard(args):
    path, out_dir, split_of = args
    from tfrecord import example_pb2
    from PIL import Image
    name = os.path.basename(path)
    out_path = f"{out_dir}/records/{name}.jsonl"
    if os.path.exists(out_path + ".done"):
        return name, "skip"
    recs = []
    for raw in records(path):
        ex = example_pb2.Example()
        ex.ParseFromString(raw)
        f = ex.features.feature
        ep = f["episode_id"].int64_list.value[0]
        split = split_of.get(ep)
        if split not in ("train", "validation"):
            continue
        goal = f["goal"].bytes_list.value[0].decode()
        shots = list(f["screenshots"].bytes_list.value)
        trees = list(f["accessibility_trees"].bytes_list.value)
        Ws = list(f["screenshot_widths"].int64_list.value)
        Hs = list(f["screenshot_heights"].int64_list.value)
        acts = [json.loads(x) for x in f["actions"].bytes_list.value]
        steps_instr = [x.decode() for x in f["step_instructions"].bytes_list.value]
        os.makedirs(f"{out_dir}/imgs/{ep}", exist_ok=True)
        for i, png in enumerate(shots):
            p = f"{out_dir}/imgs/{ep}/{i}.jpg"
            if not os.path.exists(p):
                save_screenshot(png, p)
        last_click = None
        T = len(acts)
        for i in range(T + 1):
            nodes = flat_nodes(trees[i])
            if i == T:
                action = {"action_type": "status", "goal_status": "successful"}
                instr = "The task is complete."
            else:
                action = convert_action(acts[i])
                instr = steps_instr[i] if i < len(steps_instr) else ""
            target = None
            if action["action_type"] in ("click", "long_press"):
                target = smallest(nodes, action["x"], action["y"])
                last_click = (action["x"], action["y"])
            elif action["action_type"] == "type_text":
                foc = [n for n in nodes if n["is_focused"] and n["is_editable"]]
                if foc:
                    target = min(foc, key=lambda n: n["bbox_pixels"]["area"])
                elif last_click:
                    target = smallest(nodes, *last_click)
                if target:
                    action["x"], action["y"] = target["bbox_pixels"]["center"]
            recs.append({
                "ep": ep, "step": i, "n_steps": T, "split": split, "goal": goal, "step_instruction": instr,
                "img": f"imgs/{ep}/{i}.jpg", "W": Ws[i], "H": Hs[i], "action": action,
                "target": target["bbox_pixels"] if target else None, "nodes": nodes,
            })
    os.makedirs(f"{out_dir}/records", exist_ok=True)
    with open(out_path, "w") as fo:
        for r in recs:
            fo.write(json.dumps(r) + "\n")
    open(out_path + ".done", "w").close()
    return name, len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-eps", default=None, help="json list of AC-v2 eval episode ids (asserted not in train)")
    a = ap.parse_args()
    splits = json.load(open(f"{a.raw}/splits.json"))
    split_of = {ep: s for s, eps in splits.items() for ep in eps}
    if a.eval_eps:
        ev = json.load(open(a.eval_eps))
        bad = [e for e in ev if split_of.get(e) in ("train", "validation")]
        assert not bad, f"eval episodes in train/val: {bad[:10]}"
        print(f"[leak-check] {len(ev)} eval episodes: all outside train/validation", flush=True)
    shards = sorted(p for p in glob.glob(f"{a.raw}/android_control-*") if os.path.exists(p + ".done"))
    print(f"[prep] {len(shards)} shards ready; splits: { {k: len(v) for k, v in splits.items()} }", flush=True)
    os.makedirs(f"{a.out}/records", exist_ok=True)
    with Pool(a.workers) as pool:
        for name, n in pool.imap_unordered(do_shard, [(p, a.out, split_of) for p in shards]):
            print(f"[prep] {name}: {n}", flush=True)


if __name__ == "__main__":
    main()
