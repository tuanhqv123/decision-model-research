"""T1: run a SOTA mobile GUI agent on the AndroidControl offline protocol.

Protocol (documented, on-device-realistic — screenshot + text, NO a11y tree):
  input : screenshot (state) + goal + step_instruction
  output: ONE action as a JSON object
  scoring:
    type_match   : predicted action type == ground-truth type (aliases normalized)
    step_match   : type + key args (tap coords within 100/1000, text fuzzy>=0.8, app fuzzy>=0.8)
    grounding    : tap only, |dx|<=100 and |dy|<=100 in 0-1000 normalized coords
  confidence: exp( mean logprob over generated tokens ) (hf backend: exact; llama: if logprobs)

Backends:
  hf    : transformers + MPS, e.g. inclusionAI/UI-Venus-1.5-8B (full logprobs)
  llama : llama.cpp mtmd HTTP server (GGUF + mmproj), e.g. UI-Venus-2-9B Q4

Usage:
  python3 benchmarks/run_sota_ac.py --backend hf --model <path-or-repo> --limit 300 --tag uivenus15
  python3 benchmarks/run_sota_ac.py --backend llama --base-url http://127.0.0.1:8081 --tag uivenus2-gguf
"""
import argparse
import difflib
import json
import re
import time

import numpy as np
from PIL import Image

MANIFEST = "benchmarks/data/ac_test_manifest.jsonl"
RESULTS_DIR = "benchmarks/results"

TYPE_ALIASES = {
    "click": "tap", "tap": "tap", "long_press": "long_press", "double_click": "double",
    "input_text": "type", "type": "type", "type_text": "type", "enter": "type",
    "scroll": "swipe", "swipe": "swipe", "scroll_down": "swipe", "scroll_up": "swipe",
    "open_app": "open_app", "navigate": "open_app", "launch": "open_app", "app": "open_app",
    "navigate_back": "back", "back": "back", "go_back": "back",
    "navigate_home": "home", "home": "home", "go_home": "home",
    "wait": "wait", "stop": "stop", "finish": "stop", "done": "stop", "answer": "stop",
}


def norm_type(t):
    if t is None:
        return "???"
    return TYPE_ALIASES.get(str(t).strip().lower(), str(t).strip().lower())


def norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def parse_json_out(s):
    s = str(s or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.M).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def score(pred_raw, gt, img_size):
    w, h = img_size
    p = parse_json_out(pred_raw)
    out = {"parse_ok": p is not None, "pred": p, "type_match": False,
           "step_match": False, "grounding": None, "pred_type": None}
    if p is None:
        return out
    pt = norm_type(p.get("action"))
    gt_t = norm_type(gt.get("action_type"))
    out["pred_type"] = pt
    out["type_match"] = pt == gt_t
    if pt != gt_t:
        return out
    ok = True
    if gt_t == "tap":
        try:
            gx = float(gt["x"]) / w * 1000
            gy = float(gt["y"]) / h * 1000
            px = float(p.get("x"))
            py = float(p.get("y"))
            # model may emit pixel or 0-1000 coords; normalize if pixel-ish
            if px > 1000:
                px = px / w * 1000
            if py > 1000:
                py = py / h * 1000
            grounded = abs(px - gx) <= 100 and abs(py - gy) <= 100
            out["grounding"] = bool(grounded)
            ok = grounded
        except Exception:
            ok = False
    elif gt_t == "type":
        ok = sim(p.get("text"), gt.get("text")) >= 0.8 or norm_text(p.get("text")) == norm_text(gt.get("text"))
    elif gt_t == "open_app":
        ok = sim(p.get("app") or p.get("text"), gt.get("app_name")) >= 0.8
    elif gt_t == "swipe":
        gdir = (gt.get("direction") or "").lower()
        pdir = str(p.get("direction") or "").lower()
        ok = (not gdir) or pdir == gdir
    # back/home/wait/stop: type match is the step
    out["step_match"] = bool(ok)
    return out


# ------------------------------------------------------------------ hf backend
def run_hf(args, rows):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True)
    model.eval()

    out = []
    for i, r in enumerate(rows):
        img = Image.open(r["img"]).convert("RGB")
        prompt = (
            "You are an expert Android GUI agent. You are shown a screenshot of an Android phone and a user goal.\n"
            f"User goal: {r['goal']}\n"
            f"Current step instruction: {r['step_instruction']}\n\n"
            "Decide the SINGLE next action to execute now.\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"action": "<tap|type|swipe|open_app|back|home|wait|stop>",\n'
            ' "x": <0-1000>, "y": <0-1000>,          (tap only; 0,0=top-left; 1000,1000=bottom-right)\n'
            ' "text": "<string>",                    (type only)\n'
            ' "direction": "<up|down|left|right>",   (swipe only)\n'
            ' "app": "<app name>"}                   (open_app only)'
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) and v.is_floating_point() or
                  (isinstance(v, torch.Tensor) and v.dtype in (torch.int64, torch.int32)) else v
                  for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        lat = time.time() - t0
        # logprobs: one forward over full sequence
        full = gen[0]
        with torch.no_grad():
            logits = model(input_ids=full.unsqueeze(0), **{
                k: v for k, v in inputs.items()
                if k not in ("input_ids", "attention_mask") and v is not None
            }).logits
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        p_len = inputs["input_ids"].shape[1]
        toks = full[p_len:]
        tok_lp = logp[torch.arange(p_len - 1, p_len - 1 + len(toks))][
            torch.arange(len(toks)), toks].tolist()
        conf = float(np.exp(np.mean(tok_lp))) if tok_lp else 0.0
        pred = model.processor.tokenizer.decode(gen[0][p_len:], skip_special_tokens=True) \
            if hasattr(model, "processor") else processor.tokenizer.decode(gen[0][p_len:], skip_special_tokens=True)
        s = score(pred, r["action"], r["img_size"])
        s.update({"row_id": r["row_id"], "step": r["step"], "conf": conf,
                  "latency_ms": lat * 1000, "pred_raw": pred[:400]})
        out.append(s)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}  type={sum(o['type_match'] for o in out)/(i+1):.3f} "
                  f"step={sum(o['step_match'] for o in out)/(i+1):.3f}  conf={conf:.3f} "
                  f"lat={lat*1000:.0f}ms", flush=True)
    return out


# ----------------------------------------------------------------- llama backend
def run_llama(args, rows):
    import urllib.request
    from base64 import b64encode

    results = []
    for i, r in enumerate(rows):
        img = Image.open(r["img"]).convert("RGB")
        b64 = b64encode(img.tobytes()).decode()  # placeholder — use PIL->bytes below
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = b64encode(buf.getvalue()).decode()
        prompt = (
            f"User goal: {r['goal']}\nCurrent step instruction: {r['step_instruction']}\n"
            "Decide the SINGLE next action on the Android phone in the screenshot.\n"
            'Respond with ONLY a JSON object: {"action":"<tap|type|swipe|open_app|back|home|wait|stop>",'
            '"x":<0-1000>,"y":<0-1000>,"text":"...","direction":"up|down|left|right","app":"..."} '
            "(include only the fields that apply to the chosen action)."
        )
        body = json.dumps({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt}]}],
            "max_tokens": args.max_tokens, "temperature": 0,
            "logprobs": True, "top_logprobs": 1,
        }).encode()
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        lat = time.time() - t0
        pred = data["choices"][0]["message"]["content"]
        conf = 1.0
        lp = data["choices"][0].get("logprobs")
        if lp and lp.get("content"):
            vals = [c["logprob"] for c in lp["content"] if c.get("logprob") is not None]
            if vals:
                conf = float(np.exp(np.mean(vals)))
        s = score(pred, r["action"], r["img_size"])
        s.update({"row_id": r["row_id"], "step": r["step"], "conf": conf,
                  "latency_ms": lat * 1000, "pred_raw": str(pred)[:400]})
        results.append(s)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}  type={sum(o['type_match'] for o in results)/(i+1):.3f} "
                  f"step={sum(o['step_match'] for o in results)/(i+1):.3f}  conf={conf:.3f} "
                  f"lat={lat*1000:.0f}ms", flush=True)
    return results


# ------------------------------------------------------------------ summary
def summarize(tag, out):
    n = len(out)
    n_taps = sum(1 for o in out if o.get("grounding") is not None)
    n_grounded = sum(1 for o in out if o.get("grounding") is True)
    type_acc = sum(o["type_match"] for o in out) / n
    step_acc = sum(o["step_match"] for o in out) / n
    parse_ok = sum(o["parse_ok"] for o in out) / n
    lats = sorted(o["latency_ms"] for o in out)
    confs = [o["conf"] for o in out]
    conf_ok = [o["conf"] for o in out if o["step_match"]]
    conf_bad = [o["conf"] for o in out if not o["step_match"]]
    import sys
    sys.path.insert(0, "benchmarks")
    from calib_metrics import ece, brier
    pairs = [(o["conf"], 1.0 if o["step_match"] else 0.0) for o in out]
    md = f"""# SOTA on AndroidControl (offline, screenshot+text protocol) — {tag}

Protocol: screenshot + goal + step_instruction -> one JSON action.
No a11y tree (on-device-realistic variant of AC High v2). tol=100/1000 for taps.

| metric | value |
|---|---|
| n steps | {n} |
| JSON parse ok | {parse_ok:.3f} |
| **type accuracy** | {type_acc:.3f} |
| **step accuracy** (type+args) | {step_acc:.3f} |
| grounding (taps only, n={n_taps}) | {n_grounded/n_taps if n_taps else float('nan'):.3f} |
| latency median / p95 (ms) | {lats[n//2]:.0f} / {lats[int(0.95*n)]:.0f} |
| mean confidence | {np.mean(confs):.3f} |
| conf if correct / if wrong | {np.mean(conf_ok) if conf_ok else 0:.3f} / {np.mean(conf_bad) if conf_bad else 0:.3f} |
| **ECE (conf vs step-correct)** | {ece(confs, [p[1] for p in pairs]):.4f} |
| Brier | {brier(confs, [p[1] for p in pairs]):.4f} |
"""
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    with open(f"{RESULTS_DIR}/sota-ac-{tag}-{stamp}.md", "w") as f:
        f.write(md)
    with open(f"{RESULTS_DIR}/sota-ac-{tag}-{stamp}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(md)
    print("wrote", f"{RESULTS_DIR}/sota-ac-{tag}-{stamp}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "llama"], required=True)
    ap.add_argument("--model", default="inclusionAI/UI-Venus-1.5-8B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(MANIFEST)][: args.limit]
    print(f"running {args.tag} on {len(rows)} steps (backend={args.backend})")
    if args.backend == "hf":
        out = run_hf(args, rows)
    else:
        out = run_llama(args, rows)
    summarize(args.tag, out)


if __name__ == "__main__":
    main()