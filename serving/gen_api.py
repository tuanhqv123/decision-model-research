"""Generative-arm (gen_v1) decision API: the SAME endpoints, request and response fields as serving/d1_api.py, so the
AndroidWorld harness (benchmarks/aw/awenv/model_client.py -> adapter.map_decision) runs it unchanged.

POST /v1/decide   {goal, history[], image_b64 (PNG), track "V", nodes[]/elements[] (accepted, unused: track V),
                   return_readouts bool}
GET  /health, /v1/info

One request = one greedy generation (every logits processor off, at most GEN_MAX_NEW_TOKENS tokens) from the prompt
training used (gen_common.messages: ac_common.build_state + the frozen gen_v1 instruction; train_d1.load_image).

Differences from d1_api, all explicit in the response:
  * `action` is "" when the generated text does not parse; `raw` always carries the generated text and
    `parse_error` the reason. The adapter then records an invalid action (`unmappable_action:empty`) — never a
    silent wait.
  * `confidence` = product of the probabilities of the generated tokens that spell the action_type value
    (raw logits, temperature 1); `confidence_source` names it; null when the value cannot be located.
  * `probabilities` is null: generation yields one sequence, not a distribution over the action options.
  * `box`, `box_point`, `box_confidence` are null (no box readout); `gate_score` = `confidence`.
  * `tokens_generated` = new tokens including the end-of-turn token; `latency_ms.model` covers prefill + decode.

Env: GEN_MODEL (checkpoint dir with contract.json name gen_v1 + gen_prompt.json), GEN_MAX_NEW_TOKENS (default
gen_model.DEFAULT_MAX_NEW_TOKENS), GEN_MEM_FRAC (cap on this process's GPU memory), D1_IMG_W (picture width, 720 as
in training), D1_API_TOKEN (same header the harness client sends), GEN_DEVICE (cuda; "cpu" = float32, checks only).
Run:  uvicorn gen_api:app --app-dir serving --host 127.0.0.1 --port 8701
"""
import base64
import io
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "benchmarks", "train"), os.path.join(HERE, "..", "benchmarks", "gen"), HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)  # repo layout, or every module copied flat next to this file
from gen_common import (CONTRACT_NAME, INSTRUCTION_SHA256, TRACKS, action_type_confidence,  # noqa: E402
                        check_contract, decision, parse_action)

MODEL_PATH = os.environ.get("GEN_MODEL", "models/gen-v1")
TOKEN = os.environ.get("D1_API_TOKEN", "")
CONFIDENCE_SOURCE = "action_type_token_probability_product"
STATE: Dict[str, Any] = {"engine": None, "contract": None}  # filled at startup, or injected by tests
LOCK = threading.Lock()
STATS = {"requests": 0, "errors": 0, "parse_failures": 0, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
app = FastAPI(title="gen_v1 generative decision API")


class DecideReq(BaseModel):
    goal: str
    history: List[str] = []
    image_b64: str
    nodes: Optional[List[dict]] = None
    elements: Optional[List[dict]] = None
    track: str = "V"
    return_readouts: bool = False


def load_engine(path=MODEL_PATH):
    """Refuse anything but a gen_v1 checkpoint trained with this code's instruction, then load it on the GPU."""
    doc = check_contract(path)
    import torch
    from gen_model import DEFAULT_MAX_NEW_TOKENS, GenEngine, load
    device = os.environ.get("GEN_DEVICE", "cuda")
    frac = os.environ.get("GEN_MEM_FRAC")
    if frac and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac))
    t0 = time.time()
    model, processor = load(path, device, torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()
    STATS["load_s"] = round(time.time() - t0, 1)
    n = int(os.environ.get("GEN_MAX_NEW_TOKENS", "0") or 0) or DEFAULT_MAX_NEW_TOKENS
    return GenEngine(model, processor, n), doc


def _check(token):
    if TOKEN and token != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def respond(g, W, H, track="V", return_readouts=False):
    """GenEngine.run output -> the d1_api response body (minus latency.total, added by the endpoint)."""
    parsed, err, flags = parse_action(g["text"])
    dec = decision(parsed, W, H)
    conf = action_type_confidence(g["pieces"], g["probs"]) if parsed else None
    out = {"action": dec["action"], "confidence": conf, "confidence_source": CONFIDENCE_SOURCE if conf is not None
           else None, "probabilities": None, "track": track, "screen": [W, H]}
    for k in ("direction", "app_name", "text"):
        if k in dec:
            out[k] = dec[k]
    out.update({"box": None, "box_point": None, "box_confidence": None, "point": dec["point"],
                "gate_score": conf, "raw": g["text"], "parse_error": err, "parse_flags": flags,
                "tokens_generated": int(g["tokens_generated"]), "hit_eos": bool(g["hit_eos"]),
                "latency_ms": {"model": round(float(g["model_ms"]), 1)}})
    if return_readouts:
        out["readouts"] = {"ids": g.get("ids"), "pieces": g["pieces"], "probs": g["probs"]}
    return out


@app.get("/health")
def health():
    eng = STATE["engine"]
    mem = None
    try:
        import torch
        mem = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    doc = STATE["contract"] or {}
    return {"ok": eng is not None, "model": MODEL_PATH, "contract": doc.get("name"), "tracks": doc.get("tracks"),
            "img_width": int(os.environ.get("D1_IMG_W", "720")), "instruction_sha256": INSTRUCTION_SHA256,
            "decoding": "greedy", "max_new_tokens": getattr(eng, "max_new_tokens", None), "gpu_mem_gb": mem,
            **STATS}


@app.get("/v1/info")
def info():
    return {"model": MODEL_PATH, "contract": CONTRACT_NAME, "tracks": list(TRACKS),
            "actions": ["click", "type_text", "long_press", "scroll", "wait", "navigate_back", "navigate_home",
                        "open_app", "status"],
            "note": "Generative control arm: the action is generated as one JSON object (greedy). V track only. "
                    "action '' = the output did not parse; see raw and parse_error."}


@app.post("/v1/decide")
def decide(req: DecideReq, x_api_key: Optional[str] = Header(default=None)):
    _check(x_api_key)
    t_req = time.time()
    eng = STATE["engine"]
    if eng is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    tracks = (STATE["contract"] or {}).get("tracks") or list(TRACKS)
    if req.track not in tracks:
        STATS["errors"] += 1
        raise HTTPException(status_code=400, detail=f"ValueError: track {req.track!r} was not trained for this "
                                                    f"checkpoint (trained: {tracks})")
    try:
        from train_d1 import load_image
        raw = Image.open(io.BytesIO(base64.b64decode(req.image_b64)))
        W, H = raw.size
        img = load_image(io.BytesIO(base64.b64decode(req.image_b64)))
        with LOCK:
            g = eng.run(req.goal, req.history, img)
    except Exception as e:  # noqa: BLE001
        STATS["errors"] += 1
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
    STATS["requests"] += 1
    out = respond(g, W, H, req.track, req.return_readouts)
    STATS["parse_failures"] += out["action"] == ""
    out["latency_ms"]["total"] = round((time.time() - t_req) * 1000, 1)
    return out


@app.on_event("startup")
def startup():
    """Load the checkpoint (unless an engine was injected) and decode once on a blank screen."""
    if STATE["engine"] is None:
        STATE["engine"], STATE["contract"] = load_engine()
        blank = Image.new("RGB", (1080, 2400), (255, 255, 255))
        buf = io.BytesIO()
        blank.save(buf, format="PNG")
        t0 = time.time()
        decide(DecideReq(goal="warm up", history=[], image_b64=base64.b64encode(buf.getvalue()).decode()),
               x_api_key=TOKEN)
        STATS["requests"] = 0
        STATS["parse_failures"] = 0
        STATS["warmup_s"] = round(time.time() - t0, 1)
