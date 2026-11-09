"""d1-2B AC-v2 two-stage decision protocol — self-contained KAGGLE script kernel.

GPU T4. Data pulled from public sources (no uploads):
  - 398-step JSON: GUI-Libra GitHub raw (leaderboard subset)
  - images: demisama/UGround-Offline-Evaluation AndroidControl_images.zip (6.6GB)
Results: /kaggle/working/data/results/<TAG>.jsonl + stdout summary.
Scoring = official port (differential-tested vs eval_official.py, 398/398).
"""
import json, os, sys, subprocess, time, zipfile, urllib.request

# ================= CONFIG (edit for smoke vs full) =================
TAG = "d12b2s-full-v"
LIMIT = 0            # 0 = full 398
MODALITY = "vision"   # vision | ax
ONLY_STATUS = False
THR = 0.5
BATCH = 8
IMG_MAX_H = 1280
RESUME = True
# ====================================================================

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/data/results", exist_ok=True)

# ---- fail-fast GPU guard (runbook golden rule)
import torch
assert torch.cuda.is_available(), "NO GPU — aborting before slow CPU load"
print("GPU:", torch.cuda.get_device_name(0), flush=True)

# ---- transformers: d1-2B needs a recent 5.x that registers 'axial' rope
# (older 5.x raises KeyError: 'axial' in ROPE_INIT_FUNCTIONS). Colab 5.20.2 / Mac 5.19 work.
try:
    import importlib.metadata as _md
    _v = _md.version("transformers")
    print("transformers before:", _v, flush=True)
    if tuple(int(x) for x in _v.split(".")[:2]) < (5, 19):
        print("upgrading transformers -> >=5.19 (need 'axial' rope)", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5.19"], check=True)
    print("transformers after:", _md.version("transformers"), flush=True)
except Exception as e:
    print("transformers check failed (continuing):", e, flush=True)

# ---- data prep (public sources)
def _prep():
    jp = f"{WORK}/data/500_steps_filtered.json"
    if not os.path.exists(jp):
        t0 = time.time()
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/GUI-Libra/GUI-Libra/main/"
            "evaluation/offline_evaluation/AndroidControl/data/500_steps_filtered.json", jp)
        print(f"json {os.path.getsize(jp)/1e6:.1f}MB in {time.time()-t0:.0f}s", flush=True)
    rows = json.load(open(jp))
    if MODALITY != "vision":
        print(f"PREP_DONE rows={len(rows)} (ax modality: images skipped)", flush=True)
        return
    need = sorted({r["screenshot"] for r in rows})
    zp = f"{WORK}/AndroidControl_images.zip"
    if not os.path.exists(zp):
        t0 = time.time()
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/demisama/UGround-Offline-Evaluation/"
            "resolve/main/AndroidControl_images.zip", zp)
        print(f"zip {os.path.getsize(zp)/1e9:.2f}GB in {time.time()-t0:.0f}s", flush=True)
    zf = zipfile.ZipFile(zp)
    endmap = {}
    for n in zf.namelist():
        p = n.split("/")
        if len(p) >= 2:
            endmap[f"{p[-2]}/{p[-1]}"] = n
    missing = []
    for s in need:
        dst = f"{WORK}/data/images/{s}"
        if os.path.exists(dst):
            continue
        src = endmap.get(s)
        if src is None:
            missing.append(s)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with zf.open(src) as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
    zf.close()
    if os.path.exists(zp):
        os.remove(zp)  # keep kernel OUTPUT dataset small (zip bloats it, hangs `kernels output`)
        print("zip removed after extract (output dataset stays small)", flush=True)
    bad = sum(1 for r in rows if not os.path.exists(f"{WORK}/data/images/{r['screenshot']}"))
    assert bad == 0, f"{bad} rows unresolved"
    print(f"PREP_DONE rows={len(rows)} missing={len(missing)}", flush=True)

_prep()

# ================= runner =================
import numpy as np

DATA = "/kaggle/working/data"
ROWS = f"{DATA}/500_steps_filtered.json"
IMAGES = f"{DATA}/images"
RESULTS = f"{DATA}/results"
MODEL_ID = "LiquidAI/d1-2B"

# ---- official scoring (port of eval_official.py, differential-tested 398/398)
def _bbox_contains(bbox, x, y):
    return bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]


def _smallest_bbox_node(x, y, tree):
    best_node, best_bbox, best_area = None, None, float("inf")
    for node in tree:
        if isinstance(node, dict):
            bbox = node.get("bbox_pixels")
            if bbox and _bbox_contains(bbox, x, y):
                area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])
                if area < best_area:
                    best_node, best_bbox, best_area = node, bbox, area
        elif isinstance(node, list):
            n, b, a = _smallest_bbox_node(x, y, node)
            if n and a < best_area:
                best_node, best_bbox, best_area = n, b, a
    return best_node, best_bbox


ACTION_MAPPING = {
    "longpress": "long_press", "openapp": "open_app", "terminate": "status",
    "write": "type_text", "navigateback": "navigate_back",
    "back": "navigate_back", "home": "navigate_home",
}


def build_pred_action(action_type, value):
    pat = action_type.lower()
    if pat in ACTION_MAPPING:
        pat = ACTION_MAPPING[pat]
    if "click" in pat:
        return {"action_type": "click"}
    if "long_press" in pat:
        return {"action_type": "long_press"}
    if "type_text" in pat:
        return {"action_type": "type_text", "text": (value or "").lower()}
    if "navigate_back" in pat:
        return {"action_type": "navigate_back"}
    if "open_app" in pat:
        return {"action_type": "open_app", "app_name": (value or "").lower()}
    if "scroll" in pat or "swipe" in pat:
        v = (value or "").lower()
        if "swipe" in pat:
            if "up" in v:
                v2 = "down"
            elif "down" in v:
                v2 = "up"
            elif "left" in v:
                v2 = "right"
            else:
                v2 = "left"
        else:
            v2 = v
        return {"action_type": "scroll", "direction": v2}
    if "status" in pat:
        return {"action_type": "status",
                "goal_status": "successful" if "fail" not in (value or "") else "failed"}
    return {"action_type": pat}


def score_row(row, action_type, value, pt):
    gold = row["action"]
    tree = row.get("accessibility_tree") or []
    pat = (action_type or "").lower()
    if pat in ACTION_MAPPING:
        pat = ACTION_MAPPING[pat]
    pred_action = build_pred_action(pat, value)
    correct = False
    grounding = None
    if pred_action["action_type"] in ["click", "long_press", "type_text"] and \
            gold["action_type"] == pred_action["action_type"]:
        grounding = False
        if pt is not None:
            x, y = float(pt[0]), float(pt[1])
            gnode, gbbox = _smallest_bbox_node(gold.get("x", -1), gold.get("y", -1), tree)
            if gbbox and _bbox_contains(gbbox, x, y):
                if gold["action_type"] == "type_text":
                    correct = (str(gold.get("text", "")).lower() == (value or "").lower())
                else:
                    correct = True
            grounding = correct
    else:
        keys = [k for k in gold if k in pred_action]
        if keys and all(str(gold[k]).lower() == str(pred_action[k]).lower() for k in keys):
            correct = True
    if not correct and pt is not None and tree:
        x, y = float(pt[0]), float(pt[1])
        el, _ = _smallest_bbox_node(x, y, tree)
        if el:
            text = (el.get("text") or "").lower()
            content = (el.get("content_description") or "").lower()
            if {pred_action["action_type"], gold["action_type"]} == {"click", "open_app"}:
                app = (gold.get("app_name") or "").lower()
                if (text and app in text) or (content and app in content):
                    correct, grounding = True, True
            if {pred_action["action_type"], gold["action_type"]} == {"click", "navigate_back"}:
                if (text and "back" in text) or (content and "back" in content):
                    correct, grounding = True, True
    return correct, grounding


def iter_nodes(tree):
    if isinstance(tree, dict):
        yield tree
        for v in tree.values():
            if isinstance(v, (dict, list)):
                yield from iter_nodes(v)
    elif isinstance(tree, list):
        for c in tree:
            yield from iter_nodes(c)


def pick_candidates(tree, W, H):
    seen = set()
    cands = []
    for n in iter_nodes(tree):
        if not isinstance(n, dict) or n.get("is_visible") not in (True, "True"):
            continue
        if n.get("package_name") == "com.android.systemui":
            continue
        b = n.get("bbox_pixels")
        if not b or b.get("width", 0) <= 0 or b.get("height", 0) <= 0:
            continue
        key = (b["x_min"], b["y_min"], b["x_max"], b["y_max"])
        if key in seen:
            continue
        seen.add(key)
        label = (n.get("text") or n.get("content_description") or n.get("hint_text") or "")
        actionable = n.get("is_clickable") in (True, "True") or \
                     n.get("is_editable") in (True, "True") or \
                     n.get("is_scrollable") in (True, "True")
        if not (label or actionable):
            continue
        cands.append({"label": str(label)[:80], "bbox": b})
    cands.sort(key=lambda c: c["bbox"]["area"])
    out = []
    for i, c in enumerate(cands, 1):
        b = c["bbox"]
        out.append({
            "idx": i, "label": c["label"], "bbox": b,
            "pos": (f"x:{round(100*b['x_min']/W)}-{round(100*b['x_max']/W)}% "
                    f"y:{round(100*b['y_min']/H)}-{round(100*b['y_max']/H)}%"),
        })
    return out


def ece_binary(conf, labels, n_bins=10):
    n = len(conf)
    if n == 0 or n_bins <= 0:
        return 0.0
    counts = [0] * n_bins
    csum = [0.0] * n_bins
    asum = [0.0] * n_bins
    for c, lab in zip(conf, labels):
        b = min(int(min(max(c, 0.0), 1.0) * n_bins), n_bins - 1)
        counts[b] += 1
        csum[b] += c
        asum[b] += 1.0 if lab else 0.0
    total = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        total += (counts[b] / n) * abs(asum[b] / counts[b] - csum[b] / counts[b])
    return total


def brier_binary(conf, labels):
    n = len(conf)
    if n == 0:
        return 0.0
    return sum((c - (1.0 if lab else 0.0)) ** 2 for c, lab in zip(conf, labels)) / n


ACTION_CRITERIA = {
    "click": "Tap or click a specific UI element",
    "write": "Enter text into an input field",
    "long_press": "Press and hold a UI element",
    "scroll": "Scroll a view in a direction",
    "wait": "Pause to let the UI load or update",
    "navigate_back": "Press the system Back button",
    "open_app": "Launch an app by its name",
    "terminate": "End the task (only if it is already fully done)",
}


def build_questions(with_box, cands=None):
    q = {
        "task_done": {
            "type": "noul",
            "instructions": "The task described in the goal is fully and successfully completed "
                            "on this screen (nothing left to do).",
        },
        "action": {"type": "choice",
                   "instructions": "The next UI action to perform to advance the task.",
                   "criteria": ACTION_CRITERIA},
    }
    if with_box:
        q["target"] = {"type": "box",
                       "instructions": "The UI element the next action acts on (button, text, icon, "
                                       "field). Leave unfound if the action does not target an element."}
    if cands is not None:
        q["target_el"] = {"type": "choice",
                          "instructions": "The element the next action acts on.",
                          "criteria": {str(c["idx"]): (c["label"] or f"element at {c['pos']}")
                                       for c in cands}}
    q.update({
        "scroll_dir": {"type": "choice", "instructions": "The direction, if the next action is a scroll.",
                       "criteria": {"up": "content moves up (swipe up)", "down": "content moves down (swipe down)",
                                    "left": "content moves left", "right": "content moves right"}},
        "app_name": {"type": "text", "instructions": "The exact app name, if the next action launches an app."},
        "type_text": {"type": "text", "instructions": "The exact text to type, if the next action enters text."},
    })
    return q


def resolve_shot(row):
    p = os.path.join(IMAGES, row["screenshot"])
    if os.path.exists(p):
        return p
    for d in os.listdir(IMAGES):
        q = os.path.join(IMAGES, d, row["screenshot"])
        if os.path.exists(q):
            return q
    raise FileNotFoundError(row["screenshot"])


def load_rows(limit=0, only_status=False):
    rows = json.load(open(ROWS))
    if only_status:
        rows = [r for r in rows if r["action"]["action_type"] == "status"]
    return rows[:limit] if limit else rows


def load_model():
    from transformers import AutoModel
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    model.eval()
    print(f"[load] {MODEL_ID} on cuda in {time.time()-t0:.0f}s", flush=True)
    try:
        t0 = time.time()
        model.compile()
        print(f"[load] compile() done in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[load] compile() skipped: {e}", flush=True)
    return model


def run(model, cfg):
    from PIL import Image
    rows = load_rows(cfg["limit"], cfg["only_status"])
    out_path = f"{RESULTS}/{cfg['tag']}.jsonl"
    os.makedirs(RESULTS, exist_ok=True)
    done, out = set(), []
    if cfg["resume"] and os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line)
            out.append(d)
            done.add((d["episode_id"], d["step"]))
        print(f"[resume] {len(out)} already done; {len(rows)-len(done)} remaining", flush=True)
    rows = [r for r in rows if (r["episode_id"], r["step"]) not in done]
    with_box = cfg["modality"] == "vision"
    with_el = cfg["modality"] == "ax"
    print(f"[run] {cfg['tag']} | modality={cfg['modality']} | thr={cfg['thr']} | n={len(rows)} | batch={cfg['batch']}",
          flush=True)
    t_start = time.time()
    for i in range(0, len(rows), cfg["batch"]):
        chunk = rows[i:i + cfg["batch"]]
        calls, meta = [], []
        for r in chunk:
            cands = pick_candidates(r.get("accessibility_tree") or [],
                                    r["screenshot_width"], r["screenshot_height"]) if with_el else None
            q = build_questions(with_box, cands)
            state = r["goal"]
            if with_el:
                lines = [f"Goal: {r['goal']}",
                         "Interactive elements visible on screen (id: label, position):"]
                for c in cands:
                    lines.append(f"[{c['idx']}] {c['label'] or '(no label)'}  ({c['pos']})")
                state = "\n".join(lines) + "\nAnswer using the element ids from this list."
            media = None
            if with_box:
                img = Image.open(resolve_shot(r)).convert("RGB")
                w, h = img.size
                if h > cfg["img_max_h"]:
                    img = img.resize((int(w * cfg["img_max_h"] / h), cfg["img_max_h"]), Image.LANCZOS)
                media = [img]
            calls.append((state, q, media))
            meta.append(cands)
        t0 = time.time()
        batch_out = model.system_one_batch(calls)
        wall = time.time() - t0
        for r, cands, res in zip(chunk, meta, batch_out):
            a = res["answers"]
            p_done = float(a["task_done"].get("noul", 0.0))
            action = a["action"]["choice"]
            conf = float(a["action"].get("confidence") or 0.0)
            finished = p_done >= cfg["thr"]
            final_action, value, pt, conf_final = None, None, None, None
            if finished:
                final_action, value, conf_final = "terminate", "successful", p_done
            else:
                final_action = action
                conf_final = conf  # always defined (fix: was None when box not found -> ECE crash)
                if action == "scroll":
                    value = a["scroll_dir"]["choice"]
                elif action == "open_app":
                    value = a["app_name"].get("text")
                elif action == "write":
                    value = a["type_text"].get("text")
                if with_box:
                    box = a["target"].get("box")
                    if box and (a["target"].get("found") or 0) >= 0.5:
                        pt = ((box["left"] + box["right"]) / 2 * r["screenshot_width"],
                              (box["top"] + box["bottom"]) / 2 * r["screenshot_height"])
                        conf_final = min(conf, float(a["target"].get("confidence") or 0.0))
                if with_el:
                    chosen = str(a["target_el"].get("choice") or "").strip()
                    cand = next((c for c in cands if str(c["idx"]) == chosen), None)
                    if cand:
                        pt = (cand["bbox"]["center"][0], cand["bbox"]["center"][1])
            correct, grounding = score_row(r, final_action, value, pt)
            out.append({
                "episode_id": r["episode_id"], "step": r["step"],
                "goal": r["goal"][:200],
                "p_done": p_done, "finished": finished,
                "flat_action": action, "action_type": final_action, "value": value,
                "point_2d": pt, "gold": r["action"], "conf": conf_final,
                "latency_ms": wall * 1000 / len(chunk),
                "input_tokens": res.get("usage", {}).get("input_tokens"),
                "correct": correct, "grounding": grounding,
                "gold_status": r["action"]["action_type"] == "status",
                "action_probs": a["action"].get("probabilities"),
            })
            with open(out_path, "a") as f:
                f.write(json.dumps(out[-1]) + "\n")
            n = len(out)
            acc = sum(o["correct"] for o in out) / n
            tag = "OK " if out[-1]["correct"] else "MISS"
            fin = "DONE" if finished else "ACT"
            print(f"  [{tag}] {r['episode_id']}/{r['step']} p_done={p_done:.2f} {fin} "
                  f"pred={final_action} gold={r['action']['action_type']} acc={acc:.2f} "
                  f"({wall/len(chunk)*1000:.0f}ms/step)", flush=True)
        if (i // cfg["batch"] + 1) % 5 == 0:
            el = time.time() - t_start
            eta = el / max(i + 1, 1) * (len(rows) - i - 1)
            print(f"  [progress] {i+len(chunk)}/{len(rows)} elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)
    summarize(cfg, out)


def summarize(cfg, out):
    n = len(out)
    acc = sum(o["correct"] for o in out) / n
    g_steps = [o for o in out if o["grounding"] is not None]
    g_acc = sum(1 for o in g_steps if o["grounding"]) / len(g_steps) if g_steps else float("nan")
    conf = [o["conf"] for o in out]
    y = [1.0 if o["correct"] else 0.0 for o in out]
    lats = sorted(o["latency_ms"] for o in out)
    st = [o for o in out if o["gold_status"]]
    ns = [o for o in out if not o["gold_status"]]
    done_recall = sum(1 for o in st if o["p_done"] >= 0.5) / len(st) if st else float("nan")
    pd_st = float(np.mean([o["p_done"] for o in st])) if st else float("nan")
    false_done = sum(1 for o in ns if o["p_done"] >= 0.5) / len(ns) if ns else float("nan")
    pd_ns = float(np.mean([o["p_done"] for o in ns])) if ns else float("nan")
    by_type = {}
    for at in set(o["gold"]["action_type"] for o in out):
        sub = [o for o in out if o["gold"]["action_type"] == at]
        by_type[at] = f"{sum(o['correct'] for o in sub)}/{len(sub)}"
    print("=" * 70, flush=True)
    print(f"RESULT {cfg['tag']} | modality={cfg['modality']} | n={n}", flush=True)
    print(f"  step accuracy: {acc:.4f} ({sum(o['correct'] for o in out)}/{n})", flush=True)
    print(f"  grounding: {g_acc:.4f} (n={len(g_steps)})", flush=True)
    print(f"  completion recall (gold-status n={len(st)}): {done_recall:.4f} (mean p_done={pd_st:.3f})", flush=True)
    print(f"  false-completion (non-status n={len(ns)}): {false_done:.4f} (mean p_done={pd_ns:.3f})", flush=True)
    print(f"  ECE: {ece_binary(conf, y):.4f}  Brier: {brier_binary(conf, y):.4f}", flush=True)
    print(f"  latency median/p95 ms: {lats[n//2]:.0f} / {lats[int(0.95*n)]:.0f}", flush=True)
    print(f"  by gold type: {by_type}", flush=True)
    print(f"  refs: flat vision 15.83% | flat a11y 28.39% | best trivial 21.86% | "
          f"SOTA visual 45.48(Qwen3-VL-2B)..68.59(Mem-W-8B)", flush=True)
    print("DONE_RUN", flush=True)


if __name__ == "__main__":
    cfg = {
        "tag": TAG, "modality": MODALITY, "limit": LIMIT, "only_status": ONLY_STATUS,
        "thr": THR, "img_max_h": IMG_MAX_H, "batch": BATCH, "resume": RESUME,
    }
    t0 = time.time()
    run(load_model(), cfg)
    print(f"total wall: {time.time()-t0:.0f}s", flush=True)
    # shrink /kaggle/working -> output dataset stays tiny (images/json were only needed for the run)
    import shutil
    for d in (f"{WORK}/data/images",):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    jp = f"{WORK}/data/500_steps_filtered.json"
    if os.path.exists(jp):
        os.remove(jp)
    print("WORK_CLEANUP done (output = results only)", flush=True)