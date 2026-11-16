"""Run DMA-2B on one AndroidControl test step and decode the answer into an action.

The example is step 1 of AndroidControl test episode 27: goal "Open the Cx file Explorer and rename the Flowers
folder to Flora.", step instruction "Click on the square icon next to the Flowers folder". In our evaluation the
model chose `click` (p = 0.999) on element [13] (p = 0.988), which is the correct action. This script rebuilds the
same inputs with `ac_common.py` (the code used for training and evaluation), runs one forward pass, and compares the
result with the recording.

    python example.py                     # model from this directory, GPU
    python example.py --model <repo id>   # or from the Hub
    python example.py --no-instruction    # goal only (the AndroidControl-High setting)
    python example.py --inputs-only       # print the state and the questions; no model, no GPU

AndroidControl's action space has no "done" action, so `status` is not taken here unless --allow-done is given.
"""
import argparse
import io
import json
import os

import ac_common as v

HERE = os.path.dirname(os.path.abspath(__file__))


def load_image(path, width=720):
    """As in training (`train_d1.load_image`): RGB, resized to 720 px wide (bicubic), JPEG q90 round trip."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return im


def build_inputs(goal, history, elements, width, height, apps, instruction=None):
    """State text, questions and kept elements for one step. `width`, `height`: the original screen size in pixels
    (element boxes are in those pixels). `instruction`: the step instruction, or None for goal only."""
    rule = v.CONTRACTS["v4"]["element_rule"]    # "all_reading_noime"
    state, kept, keep_idx = v.element_inputs_v3(goal, history, elements, width, height, rule, instruction)
    questions = v.build_questions("VA", kept, contract="v4", apps=apps)
    return state, questions, kept, keep_idx


def decode(ans, kept, keep_idx, elements, apps, allow_done=False):
    """Model answers -> an action dict. Without `allow_done`, `status` is never taken: the best other action of the
    same forward pass is used (AndroidControl's action space)."""
    probs = {v.FROM_OPTION[k]: float(p) for k, p in ans["action"]["probabilities"].items()}
    allowed = {k: p for k, p in probs.items() if allow_done or k != "status"}
    action = max(allowed, key=allowed.get)
    out = {"action": action, "probability": probs[action]}
    if action in v.SPATIAL:                          # click, long_press, type_text
        el_id, el = v.chosen_element(ans, kept)
        if el is None:
            out["json_action"] = None
            return out
        x, y = v.element_centre(el)
        out["element"] = {"id": el_id, "label": v.element_label(el),
                          "probability": float(ans["target_el"]["probabilities"][str(el_id)])}
        if action == "type_text":
            out["json_action"] = {"action_type": "input_text", "x": round(x), "y": round(y),
                                  "text": ans["type_text"].get("text") or "", "clear_text": True}
        else:
            out["json_action"] = {"action_type": action, "x": round(x), "y": round(y)}
    elif action == "scroll":
        out["json_action"] = {"action_type": "scroll", "direction": ans["scroll_dir"]["choice"]}
    elif action == "open_app":
        out["json_action"] = {"action_type": "open_app", "app_name": v.decode_app_choice(ans["app_name"], apps)}
    elif action == "answer":
        _, text = v.answer_text(ans, elements, kept, keep_idx)
        out["json_action"] = {"action_type": "answer", "text": text}
    elif action == "status":
        out["json_action"] = None                    # the instruction or task is complete
    else:                                            # wait, navigate_back, navigate_home
        out["json_action"] = {"action_type": action}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=HERE, help="local directory or Hub repo id")
    ap.add_argument("--example", default=os.path.join(HERE, "examples", "ac_test_ep27_step1.json"))
    ap.add_argument("--no-instruction", action="store_true")
    ap.add_argument("--allow-done", action="store_true")
    ap.add_argument("--inputs-only", action="store_true")
    a = ap.parse_args()

    ex = json.load(open(a.example))
    screenshot = os.path.join(os.path.dirname(a.example), os.path.basename(a.example).replace(".json", ".jpg"))
    instruction = None if a.no_instruction else ex.get("instruction")
    state, questions, kept, keep_idx = build_inputs(ex["goal"], ex["history"], ex["elements"], ex["width"],
                                                    ex["height"], ex["apps"], instruction)
    if a.inputs_only:
        print(state)
        print("\nquestions:", list(questions))
        return

    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    model.engine.calibration = None   # raw readouts, as trained and as evaluated
    with torch.inference_mode():
        ans = model.system_one_batch([(state, questions, [load_image(screenshot)])])[0]["answers"]

    out = decode(ans, kept, keep_idx, ex["elements"], ex["apps"], a.allow_done)
    print(json.dumps(out, indent=1))
    rec = ex.get("recorded")
    if rec and instruction is not None and not a.allow_done:
        same = out["action"] == rec["action"] and str((out.get("element") or {}).get("id")) == str(rec["target_el"])
        print(f"recorded: {rec['action']} p={rec['action_probability']:.3f}, element [{rec['target_el']}] "
              f"p={rec['target_el_probability']:.3f} -> {'same decision' if same else 'DIFFERENT decision'}")


if __name__ == "__main__":
    main()
