"""d1-2B on AC-v2 with the TWO-STAGE decision protocol (our thesis architecture).

Decouples task-completion detection from action selection:
  Stage 1: noul   "Is the task complete on this screen?"  -> p_done
  Stage 2: choice action (8 types) + grounding target (box in vision mode,
           or choice over K a11y candidates in ax mode) + value questions
  Rule:    p_done >= threshold  ->  Terminate (status/successful)
           otherwise             ->  the stage-2 action

Rationale: d1-2B zero-shot scored 0/87 on completion (status) steps when terminate
was buried as one of 8 choice options; a dedicated binary completion read is the
decision-model-native way (and is exactly the act/ask/refuse gate's success head).

No training data contains terminate steps (train trajectories end mid-task), so this
first measures how much completion is detectable ZERO-SHOT with the right interface.

Outputs: benchmarks/results/d12b2s-acv2-<tag>-<date>.{jsonl,md} (jsonl = resumable)

Usage:
  .venv-embgemma/bin/python benchmarks/run_d12b_twostage.py --tag d12b2s-v --modality vision --limit 5
  .venv-embgemma/bin/python benchmarks/run_d12b_twostage.py --tag d12b2s-v --modality vision --resume
"""
import argparse
import json
import os
import time

import numpy as np

from run_acv2 import ROWS, resolve_shot, score_row
from run_d12b_acv2 import load_model
from run_d12b_acv2_ax import pick_candidates, base_questions

MODEL_ID = "LiquidAI/d1-2B"
RESULTS = "benchmarks/results"

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


def load_rows(limit=0, only_status=False):
    rows = json.load(open(ROWS))
    if only_status:
        rows = [r for r in rows if r["action"]["action_type"] == "status"]
    return rows[:limit] if limit else rows


def run(model, args):
    from PIL import Image
    rows = load_rows(args.limit, args.only_status)
    stamp = time.strftime("%Y-%m-%d")
    out_path = f"{RESULTS}/d12b2s-acv2-{args.tag}-{stamp}.jsonl"
    os.makedirs(RESULTS, exist_ok=True)
    done, out = set(), []
    if args.resume and os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line)
            out.append(d)
            done.add((d["episode_id"], d["step"]))
        print(f"[resume] {len(out)} already done; {len(rows)-len(done)} remaining", flush=True)
    rows = [r for r in rows if (r["episode_id"], r["step"]) not in done]
    with_box = args.modality == "vision"
    with_el = args.modality == "ax"
    print(f"[run] {args.tag} | modality={args.modality} | thr={args.thr} | n={len(rows)} | batch={args.batch}",
          flush=True)

    t_start = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
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
                if h > args.img_max_h:
                    img = img.resize((int(w * args.img_max_h / h), args.img_max_h), Image.LANCZOS)
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

            # two-stage decision rule
            finished = p_done >= args.thr
            final_action, value, pt, conf_final = None, None, None, None
            if finished:
                final_action, value, conf_final = "terminate", "successful", p_done
            else:
                final_action = action
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
                else:
                    conf_final = conf
                if with_el:
                    chosen = str(a["target_el"].get("choice") or "").strip()
                    cand = next((c for c in cands if str(c["idx"]) == chosen), None)
                    if cand:
                        pt = (cand["bbox"]["center"][0], cand["bbox"]["center"][1])
            correct, grounding = score_row(r, final_action, value, pt)
            gold_status = r["action"]["action_type"] == "status"
            out.append({
                "episode_id": r["episode_id"], "step": r["step"],
                "goal": r["goal"][:200],
                "p_done": p_done, "finished": finished,
                "flat_action": action, "action_type": final_action, "value": value,
                "point_2d": pt, "gold": r["action"], "conf": conf_final,
                "latency_ms": wall * 1000 / len(chunk),
                "input_tokens": res.get("usage", {}).get("input_tokens"),
                "correct": correct, "grounding": grounding,
                "gold_status": gold_status,
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
    # completion head quality
    st = [o for o in out if o["gold_status"]]
    ns = [o for o in out if not o["gold_status"]]
    if st:
        done_recall = sum(1 for o in st if o["p_done"] >= 0.5) / len(st)
        pd_st = float(np.mean([o["p_done"] for o in st]))
    else:
        done_recall, pd_st = float("nan"), float("nan")
    if ns:
        false_done = sum(1 for o in ns if o["p_done"] >= 0.5) / len(ns)
        pd_ns = float(np.mean([o["p_done"] for o in ns]))
    else:
        false_done, pd_ns = float("nan"), float("nan")
    md = f"""# d1-2B AC-v2 — two-stage decision protocol ({tag})

n={n} | rule: p_done>={args_thr} -> terminate/successful, else stage-2 action
Scoring: official port (same as run_acv2.py). NOTE: below ~30 steps = smoke scale.

| metric | value |
|---|---|
| **step accuracy** | {acc:.4f} ({sum(o['correct'] for o in out)}/{n}) |
| grounding accuracy (spatial, n={len(g_steps)}) | {g_acc:.4f} |
| completion recall (gold-status steps, n={len(st)}) | {done_recall:.4f} (mean p_done={pd_st:.3f}) |
| false-completion rate (non-status, n={len(ns)}) | {false_done:.4f} (mean p_done={pd_ns:.3f}) |
| ECE / Brier | {ece_binary(conf, y):.4f} / {brier_binary(conf, y):.4f} |
| latency median / p95 (ms) | {lats[n//2]:.0f} / {lats[int(0.95*n)]:.0f} |

Flat-interface reference (same model): vision 15.83% · a11y 28.39% · best trivial baseline 21.86%
(always status-successful). SOTA visual: 45.48 (Qwen3-VL-2B) ... 68.59 (Mem-W-8B).
"""
    with open(f"{RESULTS}/d12b2s-acv2-{tag}-{stamp}.md", "w") as f:
        f.write(md)
    print(md, flush=True)
    print(f"wrote {RESULTS}/d12b2s-acv2-{tag}-{stamp}.md (+ .jsonl checkpoint)", flush=True)


args_thr = 0.5


def main():
    global args_thr
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-status", action="store_true", help="only the gold-status (task-complete) steps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img-max-h", type=int, default=1280)
    ap.add_argument("--modality", default="vision", choices=["vision", "ax"])
    ap.add_argument("--thr", type=float, default=0.5, help="p_done threshold for terminate")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args_thr = args.thr
    run(load_model(), args)


if __name__ == "__main__":
    main()