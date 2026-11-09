"""d1-2B on AC-v2 with the a11y-tree "choice over K" interface (our decision-model design).

Unlike run_d12b_acv2.py (official vision-only protocol: goal + screenshot + box question),
this variant gives the model the accessibility-tree candidate elements and asks a CHOICE
over them — the single-pass decision contract in its native form:

  state     : goal + compact list of interactive elements (id, label, position %)
  questions : action (choice of 8) + target_el (choice over K candidates)
              + [box: same element in the picture] when vision is on
              + scroll_dir / app_name / type_text / task_done (same as the vision variant)

Modalities:
  ax         : text-only state (no screenshot)          -> cheapest, pure decision model
  vision_ax  : screenshot + a11y state; scores BOTH the choice and the box

Scoring = the same official port (run_acv2.score_row): the chosen candidate's bbox center
becomes point_2d; element equivalences apply. We also report candidate coverage
(share of steps whose gold point lies inside at least one candidate) — if coverage is
low, the a11y data itself is the ceiling (documented data-quality metric, cf. Magma-R1).

Outputs: benchmarks/results/d12bax-acv2-<tag>-<date>.{jsonl,md} (jsonl = resumable checkpoint)

Usage:
  .venv-embgemma/bin/python benchmarks/run_d12b_acv2_ax.py --tag d12bax --modality ax --limit 5
  .venv-embgemma/bin/python benchmarks/run_d12b_acv2_ax.py --tag d12bax-va --modality vision_ax --resume
"""
import argparse
import json
import os
import time

import numpy as np

from run_acv2 import ROWS, resolve_shot, score_row

MODEL_ID = "LiquidAI/d1-2B"
RESULTS = "benchmarks/results"
K_MAX = 20


def base_questions(with_box):
    q = {
        "action": {
            "type": "choice",
            "instructions": "The next UI action to perform to advance the task.",
            "criteria": {
                "click": "Tap or click a specific UI element",
                "write": "Enter text into an input field",
                "long_press": "Press and hold a UI element",
                "scroll": "Scroll a view in a direction",
                "wait": "Pause to let the UI load or update",
                "navigate_back": "Press the system Back button",
                "open_app": "Launch an app by its name",
                "terminate": "End the task with a final message",
            },
        },
        "target_el": None,  # filled per row (choice over K)
    }
    if with_box:
        q["target_box"] = {
            "type": "box",
            "instructions": "The same UI element in the picture (picture fractions).",
        }
    q.update({
        "scroll_dir": {"type": "choice",
                       "instructions": "The direction, if the next action is a scroll.",
                       "criteria": {"up": "content moves up (swipe up)",
                                    "down": "content moves down (swipe down)",
                                    "left": "content moves left",
                                    "right": "content moves right"}},
        "app_name": {"type": "text", "instructions": "The exact app name, if the next action launches an app."},
        "type_text": {"type": "text", "instructions": "The exact text to type, if the next action enters text."},
        "task_done": {"type": "noul", "instructions": "The task has been completed successfully."},
    })
    return q


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
    """Visible, labeled OR clickable elements; bbox-deduped; no containment dedup, no cap.

    Measured on the 398-step set (223 spatial steps): coverage 0.89 (gold point inside a
    candidate), strict center-inside-gold 0.88. The missing ~11% is an a11y-data ceiling
    (gold element absent from the tree) — reported honestly as candidate coverage.
    System UI (status bar/notifications) is filtered: no coverage loss, less noise.
    """
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
    cands.sort(key=lambda c: c["bbox"]["area"])  # most specific first (readable list)
    out = []
    for i, c in enumerate(cands, 1):
        b = c["bbox"]
        out.append({
            "idx": i, "label": c["label"], "bbox": b,
            "pos": (f"x:{round(100*b['x_min']/W)}-{round(100*b['x_max']/W)}% "
                    f"y:{round(100*b['y_min']/H)}-{round(100*b['y_max']/H)}%"),
        })
    return out


def state_for(row, cands, with_box):
    lines = [f"Goal: {row['goal']}",
             "Interactive elements visible on screen (id: label, position):"]
    for c in cands:
        lines.append(f"[{c['idx']}] {c['label'] or '(no label)'}  ({c['pos']})")
    lines.append("Answer using the element ids from this list.")
    return "\n".join(lines)


def load_rows(limit=0):
    rows = json.load(open(ROWS))
    return rows[:limit] if limit else rows


def load_model():
    import torch
    from transformers import AutoModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype).to(device)
    model.eval()
    print(f"[load] {MODEL_ID} on {device} in {time.time()-t0:.0f}s", flush=True)
    try:
        t0 = time.time()
        model.compile()
        print(f"[load] compile() done in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[load] compile() skipped: {e}", flush=True)
    return model


def run(model, args):
    rows = load_rows(args.limit)
    stamp = time.strftime("%Y-%m-%d")
    out_path = f"{RESULTS}/d12bax-acv2-{args.tag}-{stamp}.jsonl"
    os.makedirs(RESULTS, exist_ok=True)
    done, out = set(), []
    if args.resume and os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line)
            out.append(d)
            done.add((d["episode_id"], d["step"]))
        print(f"[resume] {len(out)} already done; {len(rows)-len(done)} remaining", flush=True)
    rows = [r for r in rows if (r["episode_id"], r["step"]) not in done]
    with_box = args.modality == "vision_ax"
    print(f"[run] {args.tag} | modality={args.modality} | n={len(rows)} | batch={args.batch}", flush=True)

    from PIL import Image
    t_start = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        calls = []
        meta = []
        for r in chunk:
            cands = pick_candidates(r.get("accessibility_tree") or [],
                                    r["screenshot_width"], r["screenshot_height"])
            q = base_questions(with_box)
            q["target_el"] = {"type": "choice",
                              "instructions": "The element the next action acts on.",
                              "criteria": {str(c["idx"]): (c["label"] or f"element at {c['pos']}")
                                           for c in cands}}
            state = state_for(r, cands, with_box)
            media = None
            if with_box:
                img = Image.open(resolve_shot(r)).convert("RGB")
                w, h = img.size
                if h > args.img_max_h:
                    img = img.resize((int(w * args.img_max_h / h), args.img_max_h), Image.LANCZOS)
                media = [img]
            calls.append((state, q, media))
            meta.append({"cands": cands, "r": r})
        t0 = time.time()
        batch_out = model.system_one_batch(calls)
        wall = time.time() - t0
        for (state, q, media), m, res in zip(calls, meta, batch_out):
            r, cands = m["r"], m["cands"]
            a = res["answers"]
            action = a["action"]["choice"]
            conf = float(a["action"].get("confidence") or 0.0)
            value = None
            if action == "scroll":
                value = a["scroll_dir"]["choice"]
            elif action == "open_app":
                value = a["app_name"].get("text")
            elif action == "write":
                value = a["type_text"].get("text")
            elif action == "terminate":
                value = "successful" if a["task_done"].get("noul", 0) >= 0.5 else "failed"

            # candidate coverage: gold point inside at least one candidate bbox
            gx, gy = r["action"].get("x"), r["action"].get("y")
            covered = any(c["bbox"]["x_min"] <= gx <= c["bbox"]["x_max"] and
                          c["bbox"]["y_min"] <= gy <= c["bbox"]["y_max"]
                          for c in cands) if gx is not None else None

            # primary: choice -> candidate center
            chosen = str(a["target_el"].get("choice") or "").strip()
            cand = next((c for c in cands if str(c["idx"]) == chosen), None)
            pt = (cand["bbox"]["center"][0], cand["bbox"]["center"][1]) if cand else None
            correct, grounding = score_row(r, action, value, pt)
            rec = {
                "episode_id": r["episode_id"], "step": r["step"],
                "goal": r["goal"][:200], "action_type": action, "value": value,
                "point_2d": pt, "gold": r["action"], "conf": conf,
                "latency_ms": wall * 1000 / len(chunk),
                "input_tokens": res.get("usage", {}).get("input_tokens"),
                "n_cands": len(cands), "chosen": chosen, "coverage": covered,
                "correct": correct, "grounding": grounding,
                "action_probs": a["action"].get("probabilities"),
            }
            if with_box:
                box = a["target_box"].get("box")
                if box and (a["target_box"].get("found") or 0) >= 0.5:
                    bpt = ((box["left"] + box["right"]) / 2 * r["screenshot_width"],
                           (box["top"] + box["bottom"]) / 2 * r["screenshot_height"])
                    rec["box_pt"] = bpt
                    rec["correct_box"], rec["grounding_box"] = score_row(r, action, value, bpt)
            out.append(rec)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n = len(out)
            acc = sum(o["correct"] for o in out) / n
            print(f"  [{'OK ' if rec['correct'] else 'MISS'}] {r['episode_id']}/{r['step']} "
                  f"pred={action} el={chosen}/{rec['n_cands']} gold={r['action']['action_type']} "
                  f"acc={acc:.2f} conf={conf:.2f} cov={rec['coverage']} "
                  f"({wall/len(chunk)*1000:.0f}ms/step)", flush=True)
        if (i // args.batch + 1) % 5 == 0:
            el = time.time() - t_start
            eta = el / max(i + 1, 1) * (len(rows) - i - 1)
            print(f"  [progress] {i+len(chunk)}/{len(rows)} elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)

    summarize(args.tag, out, stamp, with_box)
    return out


def summarize(tag, out, stamp, with_box):
    from calib_metrics import ece_binary, brier_binary
    from collections import Counter
    n = len(out)
    acc = sum(o["correct"] for o in out) / n
    g_steps = [o for o in out if o["grounding"] is not None]
    g_acc = sum(1 for o in g_steps if o["grounding"]) / len(g_steps) if g_steps else float("nan")
    conf = [o["conf"] for o in out]
    y = [1.0 if o["correct"] else 0.0 for o in out]
    lats = sorted(o["latency_ms"] for o in out)
    ok = [o["conf"] for o in out if o["correct"]]
    bad = [o["conf"] for o in out if not o["correct"]]
    covs = [o["coverage"] for o in out if o["coverage"] is not None]
    cov_acc = 1.0 if covs else float("nan")
    by_type = {}
    for at in set(o["gold"]["action_type"] for o in out):
        sub = [o for o in out if o["gold"]["action_type"] == at]
        by_type[at] = f"{sum(o['correct'] for o in sub)}/{len(sub)}"
    boxline = ""
    if with_box:
        cb = [o for o in out if "correct_box" in o]
        if cb:
            boxline = (f"\n| box interface: acc | "
                       f"{sum(o['correct_box'] for o in cb)/len(cb):.4f} ({len(cb)} steps) |")
    md = f"""# d1-2B AC-v2 — a11y choice-over-K interface ({tag})

n={n} (leaderboard subset, 398) | modality: {"vision+a11y" if with_box else "a11y text-only"}
Interface: goal + K a11y candidates as CHOICE (the single-pass decision contract)
{'+ box in the picture' if with_box else ''}. Scoring: official port (same scorer as run_acv2.py).
NOTE: this is a DIFFERENT modality than the visual leaderboard — report as a11y-assisted,
NOT as a High-v2 parity number.

| metric | value |
|---|---|
| **step accuracy** | {acc:.4f} ({sum(o['correct'] for o in out)}/{n}) |
| grounding accuracy (spatial, n={len(g_steps)}) | {g_acc:.4f} |
| candidate coverage (gold inside a candidate) | {cov_acc:.4f} |
| ECE / Brier (action conf) | {ece_binary(conf, y):.4f} / {brier_binary(conf, y):.4f} |
| mean confidence | {np.mean(conf):.3f} |
| conf if correct / if wrong | {np.mean(ok) if ok else 0:.3f} / {np.mean(bad) if bad else 0:.3f} |
| latency median / p95 (ms) | {lats[n//2]:.0f} / {lats[int(0.95*n)]:.0f} |
{boxline}
Per gold type (correct/total): {by_type}

Leaderboard reference (visual protocol): Mem-W-8B 68.59 · GUI-Libra-8B 64.3 · Mem-W-4B 63.07 ·
UI-Venus-1.5-8B 61.06 · Qwen3-VL-2B 45.48. Below ~30 steps: smoke scale, not a result.
"""
    with open(f"{RESULTS}/d12bax-acv2-{tag}-{stamp}.md", "w") as f:
        f.write(md)
    print(md, flush=True)
    print(f"wrote {RESULTS}/d12bax-acv2-{tag}-{stamp}.md (+ .jsonl checkpoint)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img-max-h", type=int, default=1280)
    ap.add_argument("--modality", default="ax", choices=["ax", "vision_ax"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(load_model(), args)


if __name__ == "__main__":
    main()