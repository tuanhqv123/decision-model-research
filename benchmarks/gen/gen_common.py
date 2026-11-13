"""The generative control arm (Q1): ONE definition of its prompt, its target text, its coordinate grid and its
parser, shared by make_gen_data, train_gen, eval_gen and serving/gen_api — so the trained function is the served one.

What is matched with the one-pass arm R1 (contract `v1`, track V):
  * the state string is exactly `ac_common.build_state(goal, "V", None, None, history)` (no app memory, no
    app-intent line — `ALLOW_APP_INTENT` is honoured the same way train_d1 and d1_api honour it);
  * the picture is `train_d1.load_image` (720 px wide, bicubic, JPEG q90 round-trip);
  * the action vocabulary is `ac_common.ACTION_VOCAB` and its descriptions are R1's option criteria
    (`ac_common.ACTIONS`, `ac_common.SCROLL`), written once into the frozen INSTRUCTION below.

What differs, by construction (the variable under test): the answer is GENERATED text — one compact JSON object —
instead of slot readouts.

Pure Python (no torch): importable by tests, by the data builder and by the server alike.
"""
import hashlib
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from ac_common import (ACTION_VOCAB, ALLOW_APP_INTENT, SPATIAL, app_intent_for,  # noqa: E402
                       build_state)

CONTRACT_NAME = "gen_v1"
TRACKS = ("V",)
GRID = 1000                     # coordinates are thousandths of the screenshot: integers 0..999
DIRECTIONS = ("up", "down", "left", "right")
END_OF_TURN = "<|im_end|>"      # the chat template's end-of-turn token; supervised, so the model learns to stop

# FROZEN for contract gen_v1. A checkpoint trained with this text records its sha256 in gen_prompt.json and the
# server refuses a checkpoint whose recorded sha differs (PLAN §6 rule 5). Never edit in place: a new wording is a
# new contract (gen_v2). The descriptions are R1's own option criteria from ac_common.ACTIONS / ac_common.SCROLL
# as of 2026-11-10 (tests/test_gen_common.py pins the equality), in R1's option order.
INSTRUCTION = (
    "Choose the next UI action to perform to advance the task.\n"
    "Answer with one JSON object and nothing else.\n"
    "\"action_type\" is one of:\n"
    "- click: Tap or click a specific UI element\n"
    "- type_text: Enter text into an input field\n"
    "- long_press: Press and hold a UI element\n"
    "- scroll: Scroll a view in a direction\n"
    "- wait: Pause to let the UI load or update\n"
    "- navigate_back: Press the system Back button\n"
    "- navigate_home: Go to the device home screen\n"
    "- open_app: Launch an app by its name\n"
    "- status: End the task: the goal is already achieved on this screen\n"
    "For click, long_press and type_text add \"x\" and \"y\": the centre of the UI element the action acts on, "
    "as integers from 0 to 999 (thousandths of the screenshot's width and height).\n"
    "For type_text add \"text\": the exact text to type.\n"
    "For open_app add \"app_name\": the exact app name.\n"
    "For scroll add \"direction\": up (content moves up (swipe up)), down (content moves down (swipe down)), "
    "left (content moves left) or right (content moves right).\n"
    "Write the JSON compactly, with keys in this order: action_type, x, y, then text, app_name or direction."
)
# No worked example on purpose: with one ({"action_type":"click","x":500,"y":120}) the BASE model copied it verbatim
# zero-shot on a type_text row (CPU check 2026-11-10), i.e. an example is a point prior, not a format hint.
INSTRUCTION_SHA256 = hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------- prompt
def state_text(goal, history):
    """R1's state string, byte for byte: the same call train_d1.make_sample (track V, --no-memory) and d1_api make."""
    return build_state(goal, "V", None, None, history or [],
                       app_intent=app_intent_for(goal) if ALLOW_APP_INTENT else None)


def user_text(goal, history):
    """The text half of the user turn: the state, a blank line, the fixed output-format instruction."""
    return state_text(goal, history) + "\n\n" + INSTRUCTION


def messages(goal, history):
    """The chat the processor renders: one user turn, picture first (as in the model card), then the text."""
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(goal, history)}]}]


# ----------------------------------------------------------------------------------------------- coordinates
def to_grid(px, size):
    """Pixel -> thousandths: round-half-up(px / size * 1000), clamped to 0..999 — the 0.000-0.999 grid d1's box edges
    live on. Round trip error <= size / 2000 (0.54 px on a 1080 px axis) inside the screen.

    Chosen over floor-and-decode-at-bin-centre by measurement (2026-11-10, gold bbox centres): decoded gold point
    outside its own bbox 2/8,387 train and 1/474 validation spatial rows (all zero-width or zero-height boxes),
    versus 8/8,387 and 1/474. Every decodable point n * size / 1000 is also a centre d1's box readout can express
    ((top + bottom) / 2 on the same grid), so the generative grid is never finer than the one-pass one."""
    return int(min(GRID - 1, max(0, math.floor(float(px) * GRID / float(size) + 0.5))))


def from_grid(n, size):
    """Thousandths -> pixel: n * size / 1000."""
    return min(GRID - 1, max(0, float(n))) * float(size) / GRID


def gold_point(rec):
    """Pixel point the target encodes: centre of the gold target bbox if present, else the gold click point.

    The bbox centre is what R1's box readout is supervised towards (its four edges), so both arms aim at the same
    point. None for a non-spatial action or a spatial row with neither (0 such rows in cur15k_v1 / validation_v1)."""
    a = rec["action"]
    if {"input_text": "type_text"}.get(a["action_type"], a["action_type"]) not in SPATIAL:
        return None
    t = rec.get("target")
    if t:
        return (t["x_min"] + t["x_max"]) / 2.0, (t["y_min"] + t["y_max"]) / 2.0
    if a.get("x") is not None and a.get("y") is not None:
        return float(a["x"]), float(a["y"])
    return None


# ----------------------------------------------------------------------------------------------- target
def target_obj(rec):
    """The gold answer as a dict with a fixed key order. Raises ValueError for anything the contract cannot say."""
    a = rec["action"]
    at = a["action_type"]
    at = {"input_text": "type_text"}.get(at, at)
    if at not in ACTION_VOCAB:
        raise ValueError(f"action_type {at!r} not in ACTION_VOCAB")
    obj = {"action_type": at}
    if at in SPATIAL:
        pt = gold_point(rec)
        if pt is None:
            raise ValueError(f"spatial action {at} without target bbox or point")
        obj["x"] = to_grid(pt[0], rec["W"])
        obj["y"] = to_grid(pt[1], rec["H"])
    if at == "type_text":
        obj["text"] = str(a.get("text") or "")
    elif at == "open_app":
        obj["app_name"] = str(a.get("app_name") or "")
    elif at == "scroll":
        d = str(a.get("direction") or "").strip().lower()
        if d not in DIRECTIONS:
            raise ValueError(f"scroll direction {d!r}")
        obj["direction"] = d
    return obj


def dumps(obj):
    """One compact JSON object: no spaces, non-ASCII kept as is (the tokenizer sees the characters, not escapes)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def target_text(rec):
    return dumps(target_obj(rec))


# ----------------------------------------------------------------------------------------------- parsing
_ACTION_VALUE = re.compile(r'"action_type"\s*:\s*"([^"]*)"')


def _strip(text):
    s = text or ""
    if "</think>" in s:                 # the template pre-fills an empty think block; tolerate a generated one
        s = s.split("</think>")[-1]
    return s.replace(END_OF_TURN, "")


def _coord(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return int(min(GRID - 1, max(0, round(v))))


def parse_action(text):
    """Generated text -> (action dict or None, error code or None, flags list).

    Strict where the harness needs a value (an unknown action, a spatial action without numeric x/y, a scroll
    without a known direction, type_text/open_app without a string) and lenient only where nothing is invented:
    out-of-range coordinates are clamped to 0..999 (flag `clamped`), floats are rounded, extra keys and trailing
    text are ignored (flags `extra_keys`, `trailing_text`)."""
    s = _strip(text)
    i = s.find("{")
    if i < 0:
        return None, "no_json_object", []
    try:
        obj, end = json.JSONDecoder().raw_decode(s[i:])
    except json.JSONDecodeError:
        return None, "invalid_json", []
    if not isinstance(obj, dict):
        return None, "not_an_object", []
    flags = []
    if s[i + end:].strip():
        flags.append("trailing_text")
    at = obj.get("action_type")
    at = at.strip().lower() if isinstance(at, str) else None
    if at not in ACTION_VOCAB:
        return None, "unknown_action_type", flags
    out = {"action_type": at}
    allowed = {"action_type"}
    if at in SPATIAL:
        allowed |= {"x", "y"}
        x, y = _coord(obj.get("x")), _coord(obj.get("y"))
        if x is None or y is None:
            return None, "missing_point", flags
        if any(v < 0 or v > GRID - 1 for v in (obj["x"], obj["y"])):
            flags.append("clamped")
        out["x"], out["y"] = x, y
    if at == "type_text":
        allowed.add("text")
        if not isinstance(obj.get("text"), str):
            return None, "missing_text", flags
        out["text"] = obj["text"]
    elif at == "open_app":
        allowed.add("app_name")
        if not isinstance(obj.get("app_name"), str):
            return None, "missing_app_name", flags
        out["app_name"] = obj["app_name"]
    elif at == "scroll":
        allowed.add("direction")
        d = obj.get("direction")
        d = d.strip().lower() if isinstance(d, str) else None
        if d not in DIRECTIONS:
            return None, "bad_direction", flags
        out["direction"] = d
    if set(obj) - allowed:
        flags.append("extra_keys")
    return out, None, flags


def decision(parsed, W, H):
    """Parsed action -> the decision fields d1_api returns (point in PIXELS of the W x H screenshot sent)."""
    if parsed is None:
        return {"action": "", "point": None}
    at = parsed["action_type"]
    out = {"action": at, "point": None}
    if at in SPATIAL:
        out["point"] = [from_grid(parsed["x"], W), from_grid(parsed["y"], H)]
    if at == "type_text":
        out["text"] = parsed["text"]
    elif at == "open_app":
        out["app_name"] = parsed["app_name"]
    elif at == "scroll":
        out["direction"] = parsed["direction"]
    return out


def scorer_args(parsed, W, H):
    """(action_type, value, point) exactly as eval_d1.run_eval hands them to run_acv2.score_row."""
    if parsed is None:
        return "", None, None
    at = parsed["action_type"]
    value = {"scroll": parsed.get("direction"), "open_app": parsed.get("app_name"),
             "type_text": parsed.get("text"), "status": "successful"}.get(at)
    pt = None
    if at in SPATIAL:  # the leaderboard scorer reads no point for a non-spatial prediction
        pt = (from_grid(parsed["x"], W), from_grid(parsed["y"], H))
    return at, value, pt


# ----------------------------------------------------------------------------------------------- confidence
def action_type_confidence(pieces, probs):
    """Product of the probabilities of the generated tokens that spell the action_type VALUE.

    `pieces[k]` is the text token k added to the decoded output, `probs[k]` the probability the model gave that
    token (softmax of the raw logits, temperature 1). A token counts if its characters overlap the value inside
    `"action_type":"<value>"`. Returns None when the value cannot be located (e.g. a parse failure)."""
    text = "".join(pieces)
    m = _ACTION_VALUE.search(text)
    if not m or m.end(1) == m.start(1):
        return None
    lo, hi = m.start(1), m.end(1)
    conf, pos, used = 1.0, 0, 0
    for piece, p in zip(pieces, probs):
        start, pos = pos, pos + len(piece)
        if start < hi and pos > lo:
            conf *= float(p)
            used += 1
    return conf if used else None


# ----------------------------------------------------------------------------------------------- contract
def write_contract(ckpt_dir):
    """contract.json = {"name": "gen_v1", "tracks": ["V"]} (deliberately NOT in ac_common.CONTRACTS: d1_api refuses a
    generative checkpoint) + gen_prompt.json recording the exact instruction the weights were trained with."""
    with open(os.path.join(ckpt_dir, "contract.json"), "w") as fh:
        json.dump({"name": CONTRACT_NAME, "tracks": list(TRACKS)}, fh, indent=1)
    with open(os.path.join(ckpt_dir, "gen_prompt.json"), "w") as fh:
        json.dump({"format": CONTRACT_NAME, "instruction": INSTRUCTION, "sha256": INSTRUCTION_SHA256,
                   "grid": GRID, "end_of_turn": END_OF_TURN, "enable_thinking": False}, fh, indent=1)


def check_contract(ckpt_dir):
    """Refuse a checkpoint that is not a gen_v1 checkpoint trained with THIS instruction. Returns the contract."""
    with open(os.path.join(ckpt_dir, "contract.json")) as fh:
        doc = json.load(fh)
    if doc.get("name") != CONTRACT_NAME:
        raise ValueError(f"contract {doc.get('name')!r} is not {CONTRACT_NAME!r}")
    with open(os.path.join(ckpt_dir, "gen_prompt.json")) as fh:
        prompt = json.load(fh)
    if prompt.get("sha256") != INSTRUCTION_SHA256:
        raise ValueError("gen_prompt.json sha256 differs from this code's INSTRUCTION: the checkpoint was trained "
                         "with another prompt")
    doc.setdefault("tracks", list(TRACKS))
    return doc
