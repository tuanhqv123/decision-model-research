"""d1-2B decision API: one HTTP call = one forward pass, zero generated tokens.

POST /v1/decide   {goal, history[], image_b64 (PNG), nodes[] (a11y nodes, optional), track "V"|"VA",
                   elements[] (harness alias for nodes), return_readouts bool}
GET  /health, /v1/info

The state and the questions are built by the SAME functions training and evaluation use (ac_common.build_state /
build_questions / pick_candidates, train_d1.load_image), and the model is called exactly as eval_d1.run_eval does,
so an API answer equals the offline evaluation's answer for the same input (checked by serving/test_client.py).

Env: D1_MODEL (checkpoint dir), D1_IMG_W (picture width: 720 for the E2 checkpoint, 858 for the grounding run),
D1_MEM_FRAC (cap on this process's GPU memory, protects the other services on the card), D1_API_TOKEN.
"""
import base64
import io
import os
import sys
import threading
import time

import torch
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ac_common import (FROM_OPTION, SPATIAL, app_intent_for, build_questions, build_state,  # noqa: E402
                        pick_candidates)
from train_d1 import load_image  # noqa: E402

MODEL_PATH = os.environ.get("D1_MODEL", "models/e2-final")
TOKEN = os.environ.get("D1_API_TOKEN", "")
if os.environ.get("D1_MEM_FRAC"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["D1_MEM_FRAC"]))

from transformers import AutoModel  # noqa: E402

_t0 = time.time()
MODEL = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16).cuda()
MODEL.engine.calibration = None  # raw readouts, as trained (calibration is applied by the caller)
LOCK = threading.Lock()
STATS = {"requests": 0, "errors": 0, "load_s": round(time.time() - _t0, 1), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
app = FastAPI(title="d1-2B decision API")


class DecideReq(BaseModel):
    goal: str
    history: list[str] = []
    image_b64: str
    nodes: list[dict] | None = None
    elements: list[dict] | None = None  # harness spelling of `nodes`; accepted for compatibility
    track: str = "V"
    return_readouts: bool = False

    def a11y(self):
        return self.nodes if self.nodes is not None else self.elements


def _check(token):
    if TOKEN and token != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def _normalise(nodes):
    """AndroidWorld UI elements -> the node dicts ac_common.pick_candidates reads (adds width/height when absent)."""
    out = []
    for n in nodes or []:
        b = dict(n.get("bbox_pixels") or {})
        if not b or not all(k in b for k in ("x_min", "x_max", "y_min", "y_max")):
            continue
        b.setdefault("width", b["x_max"] - b["x_min"])
        b.setdefault("height", b["y_max"] - b["y_min"])
        b.setdefault("center", [(b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2])
        out.append({**n, "bbox_pixels": b})
    return out


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH, "img_width": int(os.environ.get("D1_IMG_W", "720")),
            "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), **STATS}


@app.get("/v1/info")
def info():
    return {"model": MODEL_PATH, "tracks": ["V", "VA"], "actions": list(FROM_OPTION.values()),
            "note": "V = screenshot + goal + history; VA adds the numbered element list (needs `nodes`). "
                    "A checkpoint trained only on V (e2-final) is valid for V."}


@app.post("/v1/decide")
def decide(req: DecideReq, x_api_key: str | None = Header(default=None)):
    _check(x_api_key)
    t_req = time.time()
    try:
        raw = Image.open(io.BytesIO(base64.b64decode(req.image_b64)))
        W, H = raw.size
        track, cands = req.track, None
        if track == "VA":
            cands = pick_candidates(_normalise(req.a11y()), W, H)
            if not cands:
                track = "V"
        img = load_image(io.BytesIO(base64.b64decode(req.image_b64)))
        # Same line the trainer renders, so the pointer read has the app name to copy when the goal only
        # implies it ("Turn wifi on" -> settings). Identical string on both paths by construction.
        state = build_state(req.goal, track, cands, None, req.history,
                            app_intent=app_intent_for(req.goal))
        qs = build_questions(track, cands)
        with LOCK, torch.inference_mode():
            t0 = time.time()
            ans = MODEL.system_one_batch([(state, qs, [img])])[0]["answers"]
            model_ms = (time.time() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        STATS["errors"] += 1
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
    STATS["requests"] += 1

    probs = ans["action"]["probabilities"]
    option = ans["action"]["choice"]
    action = FROM_OPTION[option]
    conf = float(ans["action"].get("confidence") or max(probs.values()))
    out = {"action": action, "confidence": conf, "probabilities": {FROM_OPTION[k]: float(v) for k, v in probs.items()},
           "track": track, "screen": [W, H]}
    if action == "scroll":
        out["direction"] = ans["scroll_dir"]["choice"]
    elif action == "open_app":
        out["app_name"] = ans["app_name"].get("text") or ""
    elif action == "type_text":
        out["text"] = ans["type_text"].get("text") or ""
    box = ans["target_box"].get("box")
    box_conf = ans["target_box"].get("confidence")
    if box:
        out["box"] = box
        out["box_point"] = [(box["left"] + box["right"]) / 2 * W, (box["top"] + box["bottom"]) / 2 * H]
    out["box_confidence"] = box_conf
    point = out.get("box_point")
    if cands is not None:
        ch = str(ans["target_el"].get("choice") or "")
        c = next((c for c in cands if str(c["idx"]) == ch), None)
        out["element"] = {"idx": int(ch) if ch.isdigit() else None, "confidence": ans["target_el"].get("confidence"),
                          "label": c["label"] if c else None}
        if c is not None:
            out["element_point"] = list(c["bbox"]["center"])
            point = out["element_point"]
    out["point"] = point if action in SPATIAL else None
    out["gate_score"] = conf * (float(box_conf or 0.0) if action in SPATIAL else 1.0)
    out["latency_ms"] = {"model": round(model_ms, 1), "total": round((time.time() - t_req) * 1000, 1)}
    if req.return_readouts:
        # Every slot readout of this forward pass. Stage 2's post-processors need the readout of
        # the class they switch TO, which the per-action summary above cannot provide.
        out["readouts"] = ans
    return out


@app.on_event("startup")
def warmup():
    """One decision on a blank screen: compiles the Triton kernels now, not on the first real request."""
    blank = Image.new("RGB", (1080, 2400), (255, 255, 255))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    t0 = time.time()
    decide(DecideReq(goal="warm up", history=[], image_b64=base64.b64encode(buf.getvalue()).decode(), track="V"), x_api_key=TOKEN)
    STATS["requests"] = 0
    STATS["warmup_s"] = round(time.time() - t0, 1)
