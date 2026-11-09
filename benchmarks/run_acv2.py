"""AC-v2 official protocol runner (reproduces the GUI-Libra/Mem-W leaderboard harness).

Protocol (from GUI-Libra repo, evaluation/offline_evaluation/AndroidControl):
  data   : 398 steps (benchmarks/data/acv2/500_steps_filtered.json — the leaderboard subset)
  shots  : benchmarks/data/acv2_shots/<episode_id>/screenshot_<step>.png
  prompt : native-agent system prompt (8 action types) + goal + <answer> JSON template
           (level=high: goal only; interaction history empty, as in the harness)
  decode : temperature 0 (greedy), max 128 tokens  (Pass@1)
  score  : port of eval_official.py — element-level grounding via the row's
           accessibility tree + their action equivalences (click<->open_app etc.)
  extra  : we ALSO emit plan/ans files in the official format so the original
           eval_official.py can be run as a cross-check, and we add a
           calibration profile (ECE/Brier/coverage-risk on confidence).

Backends:
  hf    : transformers Qwen3-VL + MPS (inclusionAI/UI-Venus-1.5-8B or local path)
  llama : llama.cpp mtmd HTTP server (GGUF + mmproj)

Usage:
  python3 benchmarks/run_acv2.py --backend hf --model inclusionAI/UI-Venus-1.5-8B --tag uivenus15-8b
  python3 benchmarks/run_acv2.py --backend llama --base-url http://127.0.0.1:8081 --tag uivenus2-9b-gguf
"""
import argparse
import json
import os
import re
import time

import numpy as np

DATA = "benchmarks/data/acv2"
ROWS = f"{DATA}/500_steps_filtered.json"
SHOTS = "benchmarks/data/acv2_shots"
RESULTS = "benchmarks/results"

SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of actions to complete the task. "
    "You need to choose actions from the the following list:"
    """
action_type: Click, action_target: Element description, value: None, point_2d: [x, y]
    ## Explanation: Tap or click a specific UI element and provide its coordinates

action_type: Write, action_target: Element description or None, value: Text to enter, point_2d: [x, y] 
    ## Explanation: Enter text into a specific input field or at the current focus if coordinate is None

action_type: LongPress, action_target: Element description, value: None, point_2d: [x, y]
    ## Explanation: Press and hold on a specific UI element (mobile only) and provide its coordinates

action_type: Scroll, action_target: None, value: "up" | "down" | "left" | "right", point_2d: None
    ## Explanation: Scroll a view or container in the specified direction

action_type: Wait, action_target: None, value: Number of seconds, point_2d: None
    ## Explanation: Pause execution to allow the UI to load or update

action_type: NavigateBack, action_target: None, value: None, point_2d: None
    ## Explanation: Press the system "Back" button

action_type: OpenApp, action_target: None, value: App name, point_2d: None
    ## Explanation: Launch an app by its name (mobile only)

action_type: Terminate, action_target: None, value: End-task message, point_2d: None
    ## Explanation: Signal the end of the current task with a final message
"""
)

ANSWER_TEMPLATE = """The response should be structured in the following format:
<answer>
{
  "action_type": "the type of action to perform, e.g., Click, Write, Scroll, Answer, etc. Please follow the system prompt for available actions.",
  "action_target": "the description of the target of the action, such as the color, text, or position on the screen of the UI element to interact with",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y] # the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100]
}
</answer>"""


def user_text(row, img_size):
    # Strict AC-v2 "High v2": goal + screenshot only, NO interaction history.
    q = ("Please generate the next move according to the UI screenshot {} "
         "and the instruction.\n\nInstruction: {}")
    return q.format("(original image size {}x{})".format(img_size[0], img_size[1]),
                    row["goal"]) + "\n" + ANSWER_TEMPLATE


# ------------------------------------------------------------- official scoring
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


def parse_response(response):
    def grab(pattern):
        m = re.search(pattern, response or "", re.DOTALL)
        return m.group(1).strip().strip(",").strip('"') if m else ""
    a = grab(r'"action_type"\s*:\s*"([^"]+)"')
    v = grab(r'"value"\s*:\s*"([^"]*)"')
    t = grab(r'"action_target"\s*:\s*"([^"]*)"')
    m = re.search(r'"point_2d"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]', response or "")
    pt = (float(m.group(1)), float(m.group(2))) if m else None
    return a, v, t, pt


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
    """Port of eval_official.py per-row logic (incl. equivalences).

    Official structure (verified against benchmarks/data/acv2/eval_official.py):
      SPATIAL branch (pred type in click/long_press/type_text AND gold type matches):
        the predicted point must land inside the a11y bbox of the SMALLEST element
        containing the gold point; type_text additionally needs the exact text.
        **There is NO key-subset fallback in this branch** (a 2026-10-28 bug had
        applied the fallback here, inflating click steps to type-only accuracy).
      ELSE branch (non-spatial, or type mismatch): key-subset equality over shared
        keys (every action carries 'action_type', so types must match).
      Equivalences (bidirectional, credited via the element at the PREDICTED point):
        click<->open_app (element label contains gold app_name; official quirk:
        if gold is click, app_name is "" so any labeled element credits it),
        click<->navigate_back (element label contains "back").
    """
    gold = row["action"]
    tree = row.get("accessibility_tree") or []
    pat = (action_type or "").lower()
    if pat in ACTION_MAPPING:
        pat = ACTION_MAPPING[pat]
    pred_action = build_pred_action(pat, value)

    correct = False
    grounding = None  # None = not a grounding step; True/False = was grounded

    if pred_action["action_type"] in ["click", "long_press", "type_text"] and \
            gold["action_type"] == pred_action["action_type"]:
        # SPATIAL: point must land on the gold element. No fallback.
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
        # NON-SPATIAL: key-subset check (action_type always shared -> types must match).
        keys = [k for k in gold if k in pred_action]
        if keys and all(str(gold[k]).lower() == str(pred_action[k]).lower() for k in keys):
            correct = True

    # equivalences: click <-> open_app / navigate_back (bidirectional, official logic)
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


# ------------------------------------------------------------------ backends
def resolve_shot(row):
    for base in (SHOTS, os.path.join(SHOTS, "images")):
        p = os.path.join(base, row["screenshot"])
        if os.path.exists(p):
            return p
    for d in os.listdir(SHOTS):
        q = os.path.join(SHOTS, d, row["screenshot"])
        if os.path.exists(q):
            return q
    raise FileNotFoundError(row["screenshot"])


def run_hf(args, rows, out=None, partial=None):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if getattr(args, "load_4bit", False):
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    model.eval()
    if out is None:
        out = []
    for i, r in enumerate(rows):
        img = Image.open(resolve_shot(r)).convert("RGB")
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text(r, img.size)}]},
        ]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                     do_sample=args.temperature > 0,
                                     temperature=1.0 if args.temperature == 0 else args.temperature,
                                     output_scores=True, return_dict_in_generate=True)
        lat = time.time() - t0
        gen = gen_out.sequences
        p_len = inputs["input_ids"].shape[1]
        toks = gen[0][p_len:]
        # Native per-token log-probs from generate()'s scores. Avoids a fragile
        # second forward whose attention_mask/mm_token_type_ids won't match the
        # longer generated sequence (Qwen3-VL get_rope_index shape mismatch).
        scores = gen_out.scores
        if scores:
            logits_stack = torch.stack(scores, dim=1).float()          # [1, n_new, vocab]
            logp = torch.log_softmax(logits_stack, dim=-1)
            tok_lp = logp[0, torch.arange(len(toks)), toks].tolist()
            conf = float(np.exp(np.mean(tok_lp))) if tok_lp else 0.0
        else:
            conf = 0.0
        resp = processor.tokenizer.decode(gen[0][p_len:], skip_special_tokens=True)
        a, v, t, pt = parse_response(resp)
        correct, grounding = score_row(r, a, v, pt)
        out.append({
            "episode_id": r["episode_id"], "step": r["step"],
            "action_type": a, "value": v, "element_description": t,
            "point_2d": pt, "response": resp,
            "conf": conf, "latency_ms": lat * 1000,
            "correct": correct, "grounding": grounding,
        })
        if partial:
            with open(partial, "a") as pf:
                pf.write(json.dumps(out[-1]) + "\n")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  acc={sum(o['correct'] for o in out)/(i+1):.3f} "
                  f"conf={conf:.3f} lat={lat*1000:.0f}ms", flush=True)
    return out


def run_llama(args, rows, out=None, partial=None):
    import io
    import base64
    import urllib.request
    from PIL import Image

    if out is None:
        out = []
    for i, r in enumerate(rows):
        img = Image.open(resolve_shot(r)).convert("RGB")
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        body = json.dumps({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": user_text(r, img.size)}]}],
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "logprobs": True, "top_logprobs": 1,
        }).encode()
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        lat = time.time() - t0
        resp = data["choices"][0]["message"]["content"] or ""
        # some models emit reasoning first; include it so the JSON is reachable
        rc = data["choices"][0]["message"].get("reasoning_content") or ""
        if not resp.strip() and rc.strip():
            resp = rc
        conf = 1.0
        lp = data["choices"][0].get("logprobs")
        if lp and lp.get("content"):
            vals = [c["logprob"] for c in lp["content"] if c.get("logprob") is not None]
            if vals:
                conf = float(np.exp(np.mean(vals)))
        a, v, t, pt = parse_response(resp)
        correct, grounding = score_row(r, a, v, pt)
        out.append({
            "episode_id": r["episode_id"], "step": r["step"],
            "action_type": a, "value": v, "element_description": t,
            "point_2d": pt, "response": resp,
            "conf": conf, "latency_ms": lat * 1000,
            "correct": correct, "grounding": grounding,
        })
        if partial:
            with open(partial, "a") as pf:
                pf.write(json.dumps(out[-1]) + "\n")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  acc={sum(o['correct'] for o in out)/(i+1):.3f} "
                  f"conf={conf:.3f} lat={lat*1000:.0f}ms", flush=True)
    return out


# ------------------------------------------------------------------ summary
def summarize(tag, out):
    import sys
    sys.path.insert(0, "benchmarks")
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
    # coverage-risk curve
    cov = []
    for t in (0.3, 0.5, 0.7, 0.9):
        sel = [i for i in range(n) if conf[i] >= t]
        if sel:
            cov.append(f"  t>={t}: cov={len(sel)/n:.2f} risk={1-np.mean([y[i] for i in sel]):.3f}")
    md = f"""# AC-v2 official protocol — {tag}

n={n} (leaderboard subset, 398 steps) | protocol: GUI-Libra native-agent (goal only, T=greedy)
Scoring: eval_official.py port (a11y-tree element grounding + equivalences).

| metric | value |
|---|---|
| **step accuracy (Pass@1)** | {acc:.4f}  ({sum(o['correct'] for o in out)}/{n}) |
| grounding accuracy (spatial steps, n={len(g_steps)}) | {g_acc:.4f} |
| **ECE (conf vs correct, scalar-bin)** | {ece_binary(conf, y):.4f} |
| Brier (scalar) | {brier_binary(conf, y):.4f} |
| mean confidence | {np.mean(conf):.3f} |
| conf if correct / if wrong | {np.mean(ok) if ok else 0:.3f} / {np.mean(bad) if bad else 0:.3f} |
| latency median / p95 (ms) | {lats[n//2]:.0f} / {lats[int(0.95*n)]:.0f} |
| coverage–risk |{' / '.join(cov) or 'n/a'} |

Leaderboard reference (AC-v2 High Pass@1): Mem-W-8B 68.59 · GUI-Libra-8B 64.3 ·
Mem-W-4B 63.07 · UI-Venus-1.5-8B 61.06.
"""
    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    with open(f"{RESULTS}/acv2-{tag}-{stamp}.md", "w") as f:
        f.write(md)
    with open(f"{RESULTS}/acv2-{tag}-{stamp}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    # official-format files (cross-check with eval_official.py)
    with open(f"{RESULTS}/acv2-{tag}-{stamp}-plan.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps({k: o[k] for k in
                                ("episode_id", "step", "action", "action_type", "element_description", "value", "response")
                                if k in o}) + "\n")
    with open(f"{RESULTS}/acv2-{tag}-{stamp}-ans.jsonl", "w") as f:
        for o in out:
            if o["point_2d"] is not None:
                f.write(json.dumps({"episode_id": o["episode_id"], "step": o["step"],
                                    "output": f"({o['point_2d'][0]:.1f}, {o['point_2d'][1]:.1f})",
                                    "scale": 1.0}) + "\n")
    print(md)
    print("wrote", f"{RESULTS}/acv2-{tag}-{stamp}.md + .jsonl + plan/ans")


def main():
    global DATA, ROWS, SHOTS, RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "llama"], required=True)
    ap.add_argument("--model", default="inclusionAI/UI-Venus-1.5-8B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--shots", default=SHOTS)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--load-4bit", action="store_true", help="load model in NF4 bitsandbytes (fits a 16GB T4)")
    ap.add_argument("--resume", action="store_true", help="skip rows already in the partial results jsonl")
    args = ap.parse_args()
    DATA = args.data
    SHOTS = args.shots
    RESULTS = args.results
    ROWS = f"{DATA}/500_steps_filtered.json"
    rows = json.load(open(ROWS))
    if args.limit:
        rows = rows[: args.limit]
    partial = f"{RESULTS}/acv2-{args.tag}-partial.jsonl"
    out0 = []
    seen = set()
    if args.resume and os.path.exists(partial):
        for line in open(partial):
            d = json.loads(line)
            out0.append(d)
            seen.add((d["episode_id"], d["step"]))
        print(f"resume: {len(out0)} already done; running {len(rows)-len(seen)} remaining")
        rows = [r for r in rows if (r["episode_id"], r["step"]) not in seen]
    print(f"AC-v2 run: {args.tag} | backend={args.backend} | n={len(rows)} | T={args.temperature}")
    out = run_hf(args, rows, out0, partial) if args.backend == "hf" else run_llama(args, rows, out0, partial)
    summarize(args.tag, out)


if __name__ == "__main__":
    main()