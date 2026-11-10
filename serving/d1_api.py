"""d1-2B decision API: one HTTP call = one forward pass, zero generated tokens.

POST /v1/decide   {goal, history[], image_b64 (PNG), nodes[] (a11y nodes, optional), track "V"|"VA",
                   elements[] (harness alias for nodes), return_readouts bool}
GET  /health, /v1/info

The state and the questions are built by the SAME functions training and evaluation use (ac_common.build_state /
build_questions / pick_candidates, train_d1.load_image), and the model is called exactly as eval_d1.run_eval does,
so an API answer equals the offline evaluation's answer for the same input (checked by serving/test_client.py).
`build_request` assembles them without touching the model, so the train/serve parity is testable on a CPU
(tests/test_d1_api_v2.py).

Contract v2 (an element-list checkpoint, `contract.json` name "v2"): the request MUST carry the AndroidWorld
element list (`elements`, or `nodes`) on track VA; it is canonicalised, filtered by the checkpoint's stamped
`element_rule` and rendered by ac_common.element_inputs — the training path — and the model CHOOSES an element
(`target_el`; there is no box). `point` is the chosen element's centre; `element` = {idx (rendered id), label,
confidence, aw_index (position in the request's list)}. A request without elements is refused with HTTP 400.

Env: D1_MODEL (checkpoint dir), D1_IMG_W (picture width: 720 for the E2 checkpoint, 858 for the grounding run),
D1_MEM_FRAC (cap on this process's GPU memory, protects the other services on the card), D1_API_TOKEN.
"""
import base64
import io
import os
import sys
import threading
import time
from typing import List, Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ac_common import (CONTRACTS, FROM_OPTION, SPATIAL, app_intent_for, build_state, chosen_element,  # noqa: E402
                        element_centre, element_inputs, element_label, pick_candidates, questions_for,
                        read_contract)
from train_d1 import load_image  # noqa: E402

MODEL_PATH = os.environ.get("D1_MODEL", "models/e2-final")
# The questions a checkpoint is asked must be the ones it was trained on. `contract.json` next to the
# weights says which; a checkpoint without one is refused at startup rather than served a guess
# (2026-11-10: E2, trained on 8 options + 5 questions, was being served 9 + 6).
CONTRACT = read_contract(MODEL_PATH)
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
    # typing.Optional/List rather than `X | None`: identical on the server's 3.11, and importable on 3.9 so the
    # train/serve parity test can run on a CPU-only machine.
    goal: str
    history: List[str] = []
    image_b64: str
    nodes: Optional[List[dict]] = None
    elements: Optional[List[dict]] = None  # harness spelling of `nodes`; accepted for compatibility
    track: str = "V"
    return_readouts: bool = False

    def a11y(self):
        return self.nodes if self.nodes is not None else self.elements


def _check(token):
    if TOKEN and token != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def _normalise(nodes):
    """UI elements -> the node dicts `ac_common.pick_candidates` reads.

    Accepts TWO shapes, because the harness and the raw AndroidWorld controller disagree:
      * `bbox_pixels` as a dict with x_min/x_max/y_min/y_max (+ optional width/height/center)
      * `bbox` as a [x_min, y_min, x_max, y_max] list, which is what the harness used to send
    The old version only understood the first, so every element was dropped, `cands` came back
    empty, `track` silently fell back to "V" — and the response then read `ans["target_el"]` and
    raised `KeyError: target_el` (HTTP 500) because a V request has no such question.
    """
    out = []
    for n in nodes or []:
        b = dict(n.get("bbox_pixels") or {})
        if not b:
            flat = n.get("bbox")
            if isinstance(flat, (list, tuple)) and len(flat) == 4:
                b = {"x_min": flat[0], "y_min": flat[1], "x_max": flat[2], "y_max": flat[3]}
        if not b or not all(k in b for k in ("x_min", "x_max", "y_min", "y_max")):
            continue
        b.setdefault("width", b["x_max"] - b["x_min"])
        b.setdefault("height", b["y_max"] - b["y_min"])
        b.setdefault("center", [(b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2])
        # The harness sends `clickable`; pick_candidates reads the AndroidWorld spellings.
        merged = {**n, "bbox_pixels": b}
        if "is_clickable" not in merged:
            merged["is_clickable"] = bool(n.get("clickable"))
        if "is_visible" not in merged:
            merged["is_visible"] = True
        out.append(merged)
    return out


def _element_contract(contract):
    return bool(CONTRACTS[contract["name"]].get("elements"))


def build_request(contract, goal, history, nodes, W, H, track):
    """Everything the model is asked for one request, built without the model.

    Returns {"track", "state", "questions", "cands" (v1 VA candidates or None), "kept", "keep_idx" (v2 kept
    canonical elements and their positions in `nodes`, or None)}. Raises ValueError for a request the
    checkpoint's contract does not define; `decide` turns that into HTTP 400.
    """
    if _element_contract(contract):
        if not nodes:
            raise ValueError(f"this checkpoint speaks contract {contract['name']!r}: it chooses among the screen's "
                             "elements, and the request carries no `elements` (send the AndroidWorld element list "
                             "with track 'VA'); there is no fallback without them")
        if track != "VA":
            raise ValueError(f"contract {contract['name']!r} is served on track 'VA' only, not {track!r}")
        state, kept, keep_idx = element_inputs(goal, history, nodes, W, H, contract["element_rule"])
        return {"track": "VA", "state": state, "questions": questions_for(contract, "VA", kept), "cands": None,
                "kept": kept, "keep_idx": keep_idx}
    cands = None
    if track == "VA":
        cands = pick_candidates(_normalise(nodes), W, H)
        if not cands:
            track = "V"
    # `app_intent` is a scaffold line ("Application for this task: <app>") that only helps if
    # the checkpoint was TRAINED with it. `ac_common.ALLOW_APP_INTENT` is False, so training
    # rows never contain it — but this call passed it unconditionally, adding a line the model
    # had never seen to every serve-time state. Gate on the same flag the trainer uses, so the
    # two paths render the same string by construction rather than by intention.
    from ac_common import ALLOW_APP_INTENT
    state = build_state(
        goal, track, cands, None, history,
        app_intent=app_intent_for(goal) if ALLOW_APP_INTENT else None,
    )
    return {"track": track, "state": state, "questions": questions_for(contract, track, cands), "cands": cands,
            "kept": None, "keep_idx": None}


@app.get("/health")
def health():
    rule = {"element_rule": CONTRACT["element_rule"]} if _element_contract(CONTRACT) else {}
    return {"ok": True, "model": MODEL_PATH, "contract": CONTRACT["name"], "tracks": CONTRACT["tracks"], **rule,
            "img_width": int(os.environ.get("D1_IMG_W", "720")),
            "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), **STATS}


@app.get("/v1/info")
def info():
    return {"model": MODEL_PATH, "tracks": ["V", "VA"], "actions": list(FROM_OPTION.values()),
            "note": "V = screenshot + goal + history; VA adds the numbered element list (needs `nodes`). "
                    "A checkpoint trained only on V (e2-final) is valid for V."}


@app.post("/v1/decide")
def decide(req: DecideReq, x_api_key: Optional[str] = Header(default=None)):
    _check(x_api_key)
    t_req = time.time()
    try:
        raw = Image.open(io.BytesIO(base64.b64decode(req.image_b64)))
        W, H = raw.size
        built = build_request(CONTRACT, req.goal, req.history, req.a11y(), W, H, req.track)
        track, cands, state, qs = built["track"], built["cands"], built["state"], built["questions"]
        img = load_image(io.BytesIO(base64.b64decode(req.image_b64)))
        with LOCK, torch.inference_mode():
            t0 = time.time()
            ans = MODEL.system_one_batch([(state, qs, [img])])[0]["answers"]
            model_ms = (time.time() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        STATS["errors"] += 1
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
    STATS["requests"] += 1
    if built["kept"] is not None:
        return _respond_elements(req, ans, built, W, H, model_ms, t_req)

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
    # is_done: independent p(task complete). The adapter uses it to refuse an early `status`
    # (31 of 104 AndroidWorld episodes declared done at median step 7 on tasks needing 8-15).
    if "is_done" in ans:
        try:
            out["is_done"] = float(ans["is_done"].get("noul"))
        except (AttributeError, TypeError, ValueError):
            pass
    if box:
        out["box"] = box
        out["box_point"] = [(box["left"] + box["right"]) / 2 * W, (box["top"] + box["bottom"]) / 2 * H]
    out["box_confidence"] = box_conf
    point = out.get("box_point")
    # `if cands is not None` was true for an EMPTY list, so a request whose candidates had all
    # been filtered out still tried to read `ans["target_el"]` — a question that only exists when
    # track is VA. That is the `KeyError: target_el` / HTTP 500 seen on 2026-11-10.
    if cands:
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


def _respond_elements(req, ans, built, W, H, model_ms, t_req):
    """The response of an element contract (v2): the point is the CHOSEN element's centre; there is no box."""
    probs = ans["action"]["probabilities"]
    action = FROM_OPTION[ans["action"]["choice"]]
    conf = float(ans["action"].get("confidence") or max(probs.values()))
    out = {"action": action, "confidence": conf, "probabilities": {FROM_OPTION[k]: float(v) for k, v in probs.items()},
           "track": built["track"], "screen": [W, H], "contract": CONTRACT["name"],
           "element_rule": CONTRACT["element_rule"], "n_elements": len(built["kept"])}
    if action == "scroll":
        out["direction"] = ans["scroll_dir"]["choice"]
    elif action == "open_app":
        out["app_name"] = ans["app_name"].get("text") or ""
    elif action == "type_text":
        out["text"] = ans["type_text"].get("text") or ""
    el = ans.get("target_el") or {}   # absent only when no element survived the rule
    el_id, e = chosen_element(ans, built["kept"])
    out["element"] = {"idx": el_id, "label": element_label(e) if e is not None else None,
                      "confidence": el.get("confidence"),
                      "aw_index": built["keep_idx"][el_id - 1] if el_id is not None else None}
    point = None
    if e is not None:
        out["element_point"] = element_centre(e)
        point = out["element_point"]
    out["point"] = point if action in SPATIAL else None
    # Same shape as v1's gate (action confidence x placement confidence), with the element's confidence in
    # place of the box's, since v2 has no box.
    out["gate_score"] = conf * (float(el.get("confidence") or 0.0) if action in SPATIAL else 1.0)
    out["latency_ms"] = {"model": round(model_ms, 1), "total": round((time.time() - t_req) * 1000, 1)}
    if req.return_readouts:
        out["readouts"] = ans   # includes target_el's probabilities over the rendered ids (the harness masks on them)
    return out


@app.on_event("startup")
def warmup():
    """One decision on a blank screen: compiles the Triton kernels now, not on the first real request.
    An element contract needs an element list, so its warm-up sends one synthetic element on track VA."""
    blank = Image.new("RGB", (1080, 2400), (255, 255, 255))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    t0 = time.time()
    kw = {"track": "V"}
    if _element_contract(CONTRACT):
        kw = {"track": "VA", "elements": [{"text": "warm up", "class_name": "android.widget.Button",
                                           "package_name": "warm.up", "is_clickable": True,
                                           "bbox_pixels": {"x_min": 0, "y_min": 0, "x_max": 1080, "y_max": 240}}]}
    decide(DecideReq(goal="warm up", history=[], image_b64=base64.b64encode(buf.getvalue()).decode(), **kw),
           x_api_key=TOKEN)
    STATS["requests"] = 0
    STATS["warmup_s"] = round(time.time() - t0, 1)
