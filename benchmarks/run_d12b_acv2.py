"""d1-2B (Liquid, single-pass decision model) on the AC-v2 398-step leaderboard subset.

Maps the official native-agent task onto d1-2B's typed questions in ONE forward pass:
  action       : choice over the 8 native action types
  target       : box  -> element to act on (fractions of the picture)
  scroll_dir   : choice up/down/left/right   (used when action=scroll)
  app_name     : text                        (used when action=open_app)
  type_text    : text                        (used when action=write)
  task_done    : noul                        (used when action=terminate)

Scoring = the SAME port of eval_official.py used by run_acv2.py (score_row):
element-level grounding via the row's a11y tree + action equivalences, so results
are directly comparable with the leaderboard (Mem-W-8B 68.59, GUI-Libra-8B 64.3,
Mem-W-4B 63.07, UI-Venus-1.5-8B 61.06, Qwen3-VL-2B 45.48).

Resource notes (from the d1-2B card):
  * model.compile() once, then system_one_batch for throughput
  * screenshots pre-downscaled (box answers are picture fractions -> scale back to
    the original screenshot_width/height for scoring)
  * state kept to goal-only (official "High" protocol); 5K-token states are ~10x slower

Outputs (resumable; the jsonl IS the checkpoint):
  benchmarks/results/d12b-acv2-<tag>-<date>.jsonl   per-step log (append-only)
  benchmarks/results/d12b-acv2-<tag>-<date>.md      summary

Usage:
  .venv-embgemma/bin/python benchmarks/run_d12b_acv2.py --tag d12b --limit 5   # smoke
  .venv-embgemma/bin/python benchmarks/run_d12b_acv2.py --tag d12b --resume    # full
"""
import argparse
import json
import os
import time

import numpy as np

from run_acv2 import ROWS, resolve_shot, score_row

MODEL_ID = "LiquidAI/d1-2B"
RESULTS = "benchmarks/results"

QUESTIONS = {
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
    "target": {
        "type": "box",
        "instructions": "The UI element the next action acts on (button, text, icon, "
                        "field). Leave unfound if the action does not target an element.",
    },
    "scroll_dir": {
        "type": "choice",
        "instructions": "The direction, if the next action is a scroll.",
        "criteria": {"up": "content moves up (swipe up)", "down": "content moves down (swipe down)",
                     "left": "content moves left", "right": "content moves right"},
    },
    "app_name": {"type": "text", "instructions": "The exact app name, if the next action launches an app."},
    "type_text": {"type": "text", "instructions": "The exact text to type, if the next action enters text."},
    "task_done": {"type": "noul", "instructions": "The task has been completed successfully."},
}


def load_rows(limit=0):
    rows = json.load(open(ROWS))
    return rows[:limit] if limit else rows


def load_model():
    import torch
    from transformers import AutoModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype)
    model = model.to(device)
    model.eval()
    print(f"[load] {MODEL_ID} on {device}/{dtype} in {time.time()-t0:.0f}s", flush=True)
    try:
        t0 = time.time()
        if os.environ.get("D1_NO_COMPILE", "") != "1":
            model.compile()
            print(f"[load] compile() done in {time.time()-t0:.0f}s", flush=True)
        else:
            print("[load] compile() SKIPPED (D1_NO_COMPILE=1)", flush=True)
    except Exception as e:  # compile is a speed-up, not a requirement
        print(f"[load] compile() skipped: {e}", flush=True)
    return model


def step_to_call(row, max_img_h=1280):
    from PIL import Image
    path = resolve_shot(row)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if h > max_img_h:
        img = img.resize((int(w * max_img_h / h), max_img_h), Image.LANCZOS)
    state = row["goal"]  # official High protocol: goal + screenshot, no history
    media = [img]
    return state, QUESTIONS, media, row["screenshot_width"], row["screenshot_height"]


def answer_to_action(a, W, H):
    """Map d1-2B answers -> (action_type, value, point_2d, conf, raw) for score_row."""
    action = a["action"]["choice"]
    conf = float(a["action"].get("confidence") or 0.0)
    raw = {"action_probs": a["action"].get("probabilities"),
           "box": a["target"].get("box"), "box_found": a["target"].get("found"),
           "box_conf": a["target"].get("confidence")}

    value = None
    pt = None
    box_conf = None
    if action == "scroll":
        value = a["scroll_dir"]["choice"]
    elif action == "open_app":
        value = a["app_name"].get("text")
    elif action == "write":
        value = a["type_text"].get("text")
    elif action == "terminate":
        value = "successful" if a["task_done"].get("noul", 0) >= 0.5 else "failed"

    box = a["target"].get("box")
    if box and (a["target"].get("found") or 0) >= 0.5:
        pt = ((box["left"] + box["right"]) / 2.0 * W, (box["top"] + box["bottom"]) / 2.0 * H)
        box_conf = float(a["target"].get("confidence") or 0.0)
    if pt is not None and box_conf is not None:
        conf = min(conf, box_conf)  # conservative step confidence
    return action, value, pt, conf, raw


def run(model, args):
    rows = load_rows(args.limit)
    stamp = time.strftime("%Y-%m-%d")
    out_path = f"{RESULTS}/d12b-acv2-{args.tag}-{stamp}.jsonl"
    os.makedirs(RESULTS, exist_ok=True)
    done, out = set(), []
    if args.resume and os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line)
            out.append(d)
            done.add((d["episode_id"], d["step"]))
        print(f"[resume] {len(out)} already done; {len(rows)-len(done)} remaining", flush=True)
    rows = [r for r in rows if (r["episode_id"], r["step"]) not in done]
    print(f"[run] {args.tag} | modality={args.modality} | n={len(rows)} | batch={args.batch} | "
          f"img_h<={args.img_max_h}", flush=True)

    t_start = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        calls = [step_to_call(r, args.img_max_h) for r in chunk]
        t0 = time.time()
        batch_out = model.system_one_batch(
            [(st, q, m) for st, q, m, _w, _h in calls])
        wall = time.time() - t0
        for row, (st, q, m, W, H), res in zip(chunk, calls, batch_out):
            a = res["answers"]
            action, value, pt, conf, raw = answer_to_action(a, W, H)
            correct, grounding = score_row(row, action, value, pt)
            rec = {
                "episode_id": row["episode_id"], "step": row["step"],
                "goal": row["goal"][:200],
                "action_type": action, "value": value,
                "point_2d": pt,
                "gold": row["action"],
                "conf": conf, "latency_ms": wall * 1000 / len(chunk),
                "input_tokens": res.get("usage", {}).get("input_tokens"),
                "correct": correct, "grounding": grounding,
                **{k: raw[k] for k in raw if raw[k] is not None},
            }
            if args.save_readouts:
                rec["readouts"] = a
            out.append(rec)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n = len(out)
            acc = sum(o["correct"] for o in out) / n
            mark = "OK " if rec["correct"] else "MISS"
            print(f"  [{mark}] {row['episode_id']}/{row['step']} pred={action} "
                  f"gold={row['action']['action_type']} acc={acc:.2f} conf={conf:.2f} "
                  f"({wall/len(chunk)*1000:.0f}ms/step)", flush=True)
        if (i // args.batch + 1) % 5 == 0:
            el = time.time() - t_start
            eta = el / max(i + 1, 1) * (len(rows) - i - 1)
            print(f"  [progress] {i+len(chunk)}/{len(rows)} elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)

    summarize(args.tag, out, stamp)
    return out


def summarize(tag, out, stamp):
    from calib_metrics import ece_binary, brier_binary
    n = len(out)
    acc = sum(o["correct"] for o in out) / n
    g_steps = [o for o in out if o["grounding"] is not None]
    g_acc = sum(1 for o in g_steps if o["grounding"]) / len(g_steps) if g_steps else float("nan")
    conf = [o["conf"] for o in out]
    y = [1.0 if o["correct"] else 0.0 for o in out]
    lats = sorted(o["latency_ms"] for o in out)
    ok = [o["conf"] for o in out if o["correct"]]
    bad = [o["conf"] for o in out if not o["correct"]]
    cov = []
    for t in (0.3, 0.5, 0.7, 0.9):
        sel = [i for i in range(n) if conf[i] >= t]
        if sel:
            cov.append(f"  t>={t}: cov={len(sel)/n:.2f} risk={1-np.mean([y[i] for i in sel]):.3f}")
    from collections import Counter
    by_type = {}
    for at in set(o["gold"]["action_type"] for o in out):
        sub = [o for o in out if o["gold"]["action_type"] == at]
        by_type[at] = f"{sum(o['correct'] for o in sub)}/{len(sub)}"
    md = f"""# d1-2B on AC-v2 (single-pass decision, {tag})

n={n} (leaderboard subset, 398 steps) | protocol: AC-v2 High (goal + screenshot, no history)
Interface: 6 typed questions in ONE forward pass (choice action + box target + values).
Scoring: eval_official.py port (a11y-tree element grounding + equivalences) — same scorer as run_acv2.py.

| metric | value |
|---|---|
| **step accuracy (Pass@1)** | {acc:.4f} ({sum(o['correct'] for o in out)}/{n}) |
| grounding accuracy (spatial steps, n={len(g_steps)}) | {g_acc:.4f} |
| ECE (conf vs correct, scalar-bin) | {ece_binary(conf, y):.4f} |
| Brier (scalar) | {brier_binary(conf, y):.4f} |
| mean confidence | {np.mean(conf):.3f} |
| conf if correct / if wrong | {np.mean(ok) if ok else 0:.3f} / {np.mean(bad) if bad else 0:.3f} |
| latency median / p95 (ms) | {lats[n//2]:.0f} / {lats[int(0.95*n)]:.0f} |
| coverage–risk | {' / '.join(cov) or 'n/a'} |

Per gold action type (correct/total): {by_type}

Leaderboard reference (AC-v2 High Pass@1): Mem-W-8B 68.59 · GUI-Libra-8B 64.3 ·
Mem-W-4B 63.07 · UI-Venus-1.5-8B 61.06 · Qwen3-VL-2B 45.48.
NOTE: numbers below ~30 steps are smoke-test scale, not a result.
"""
    with open(f"{RESULTS}/d12b-acv2-{tag}-{stamp}.md", "w") as f:
        f.write(md)
    print(md, flush=True)
    print(f"wrote {RESULTS}/d12b-acv2-{tag}-{stamp}.md (+ .jsonl checkpoint)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img-max-h", type=int, default=1280)
    ap.add_argument("--modality", default="vision", choices=["vision"])  # ax variants later
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-readouts", action="store_true",
                    help="persist every slot readout (choice/probabilities/text) per row; "
                         "stage 2 needs the readouts of the class a post-processor switches TO")
    args = ap.parse_args()
    run(load_model(), args)


if __name__ == "__main__":
    main()