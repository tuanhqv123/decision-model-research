"""DMA-2B demo: replay three AndroidWorld episodes step by step, and re-run the model live on any step.

Each step shows what the model read (screenshot, goal, history, element list), the probability it gave to every
action and to the top elements, and the AndroidWorld action that was executed. The recorded answers come from the
evaluation run (model on a GPU). "Run the model now" repeats the forward pass on this Space's CPU.
"""
import importlib.util
import json
import os
import threading
import time

import gradio as gr
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageDraw

MODEL_ID = "tuanhqv123/DMA-2B"
HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name):
    """`ac_common.py` and `example.py` from the model repository: the same input code as training and evaluation."""
    path = hf_hub_download(MODEL_ID, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v = _load_module("ac_common")
E = _load_module("example")
APPS = [a["name"] for a in json.load(open(hf_hub_download(MODEL_ID, "aw_apps.json")))]

EPISODES = {
    "Turn wifi on.  (SystemWifiTurnOnVerify)": "wifi_on",
    "Count an ingredient in a recipe note  (NotesRecipeIngredientCount)": "recipe_count",
    "Reply to an SMS  (SimpleSmsReply)": "sms_reply",
}
DATA = {k: json.load(open(os.path.join(HERE, "examples", f"{f}.json"))) for k, f in EPISODES.items()}

ACTION_NAMES = {"click": "click", "write": "type text", "long_press": "long press", "scroll": "scroll",
                "wait": "wait", "navigate_back": "back", "navigate_home": "home", "open_app": "open app",
                "terminate": "done (end task)", "answer": "answer"}

_model, _lock = None, threading.Lock()


def get_model():
    global _model
    with _lock:
        if _model is None:
            import torch
            from transformers import AutoModel
            path = snapshot_download(MODEL_ID)
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            # float32 on CPU: about 3x faster than bfloat16 on CPUs without bf16 units; same decisions in our test
            _model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.float32).eval()
            _model.engine.calibration = None
        return _model


def inputs_for(ep, step):
    st = ep["steps"][step]
    png = os.path.join(HERE, "examples", st["screenshot"])
    state, questions, kept, keep_idx = E.build_inputs(ep["goal"], st["history"], st["elements"], png, APPS)
    return st, png, state, questions, kept, keep_idx


def render(ep, step, ans, note):
    st, png, state, _, kept, keep_idx = inputs_for(ep, step)
    out = E.decode(ans, st["history"], kept, keep_idx, st["elements"], APPS)
    img = Image.open(png).convert("RGB")
    draw = ImageDraw.Draw(img)
    action_probs = {ACTION_NAMES.get(k, k): float(p) for k, p in ans["action"]["probabilities"].items()}
    el_probs = {}
    if "target_el" in ans:
        for k, p in ans["target_el"]["probabilities"].items():
            if k.isdigit() and 1 <= int(k) <= len(kept):
                el_probs[f"[{k}] {v.element_label(kept[int(k) - 1])[:40]}"] = float(p)
    el = out.get("element")
    if out["action"] == "answer":  # the answer is the text of the chosen element: show that element too
        el_id, e = v.chosen_element(ans, kept)
        if e is not None:
            el = {"id": el_id, "label": v.element_label(e),
                  "probability": float(ans["target_el"]["probabilities"][str(el_id)])}
    if el and out["action"] in v.SPATIAL + ("answer",):
        b = kept[el["id"] - 1]["bbox"]
        draw.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=(230, 40, 40), width=8)
        x, y = v.element_centre(kept[el["id"] - 1])
        draw.ellipse([x - 22, y - 22, x + 22, y + 22], outline=(230, 40, 40), width=8)
    a = out["action"]
    headline = f"**Decision:** `{a}` with probability **{out['probability']:.3f}**"
    if el and a in v.SPATIAL + ("answer",):
        headline += f" on element **[{el['id']}] {el['label']}** (p = {el['probability']:.3f})"
    if out.get("json_action") is not None:
        headline += f"\n\n**AndroidWorld action:** `{json.dumps(out['json_action'], ensure_ascii=False)}`"
    elif a == "status":
        headline += "\n\n**The episode ends here** (the model declares the task complete)."
    history = "\n".join(f"{i}. {h}" for i, h in enumerate(st["history"], 1)) or "none (first step)"
    info = (f"**Goal:** {ep['goal']}\n\n**Step {step + 1} of {len(ep['steps'])}.** Previous actions:\n\n{history}"
            f"\n\n{note}")
    return img, info, headline, action_probs, el_probs, state


def show(ep_name, step):
    ep = DATA[ep_name]
    step = min(int(step) - 1, len(ep["steps"]) - 1)
    st = ep["steps"][step]
    note = (f"_Recorded answer from the evaluation run (GPU, {st['model_ms']:.0f} ms). "
            "Press **Run the model now** to repeat it on this Space's CPU._")
    return render(ep, step, st["readouts"], note)


def run_live(ep_name, step):
    ep = DATA[ep_name]
    step = min(int(step) - 1, len(ep["steps"]) - 1)
    st, png, state, questions, kept, _ = inputs_for(ep, step)
    import torch
    model = get_model()
    t0 = time.time()
    with torch.inference_mode():
        ans = model.system_one_batch([(state, questions, [E.load_image(png)])])[0]["answers"]
    secs = time.time() - t0
    rec = st["readouts"]
    same = ans["action"]["choice"] == rec["action"]["choice"] and (
        "target_el" not in rec or ans["target_el"]["choice"] == rec["target_el"]["choice"])
    note = (f"_Live run on this Space's CPU: {secs:.0f} s (one forward pass, float32). "
            f"{'Same decision as the recorded GPU run.' if same else 'Different decision from the recorded GPU run.'}_")
    return render(ep, step, ans, note)


def steps_for(ep_name):
    n = len(DATA[ep_name]["steps"])
    return gr.Slider(minimum=1, maximum=n, step=1, value=1, label=f"Step (1 to {n})")


INTRO = """
# DMA-2B: one forward pass per phone action

[DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B) is a 2B **decision model** for Android agents. At each step it
reads the screenshot, the goal, what it has done so far and the list of on-screen elements, and **chooses** the next
action from fixed options (click, type, scroll, open app, done, ...) and the element to act on. It writes no text,
so its output can always be executed, and every choice comes with a probability.

Pick one of three AndroidWorld tasks it solved and step through them. The red box is the element it chose.
"""

EXPLAIN = """
### How to read this page
- **Action probabilities**: the model's probability for each possible action at this step. They sum to 1. A
  confident step puts almost everything on one action.
- **Element probabilities**: for click, long press and type, which element of the list it acts on. The tap
  lands at the centre of that element (red circle).
- **What the model reads**: the exact text input, built by the same code as in training. Together with the
  screenshot (resized to 720 px wide) it is the whole input.

### Results (more in the [model card](https://huggingface.co/tuanhqv123/DMA-2B))
| | Type | Grounding | Step success |
|---|---|---|---|
| AndroidControl-Low (step instruction given) | 96.6 | 90.1 | 89.3 |
| AndroidControl-High (goal only) | 74.3 | 78.4 | 61.7 |

*Type*: % of steps with the right kind of action. *Grounding*: % of clicks on the right element (among steps
with the right type). *Step success*: % of steps entirely right. On AndroidWorld (116 live tasks) it completes
**21.6%** of the tasks, at 135 ms per decision on a GPU.
"""

with gr.Blocks(title="DMA-2B demo") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        ep_dd = gr.Dropdown(list(EPISODES), value=list(EPISODES)[0], label="Task")
        step_sl = gr.Slider(minimum=1, maximum=len(DATA[list(EPISODES)[0]]["steps"]), step=1, value=1,
                            label="Step")
    with gr.Row():
        with gr.Column(scale=2):
            shot = gr.Image(type="pil", label="Screen (red: chosen element)", height=640)
        with gr.Column(scale=3):
            info = gr.Markdown()
            decision = gr.Markdown()
            live = gr.Button("Run the model now (CPU, about 1 minute)", variant="primary")
            with gr.Row():
                act = gr.Label(num_top_classes=5, label="Action probabilities")
                els = gr.Label(num_top_classes=5, label="Element probabilities")
    with gr.Accordion("What the model reads (text input)", open=False):
        state = gr.Textbox(lines=20, show_label=False)
    gr.Markdown(EXPLAIN)

    outs = [shot, info, decision, act, els, state]
    ep_dd.change(steps_for, ep_dd, step_sl).then(show, [ep_dd, step_sl], outs)
    step_sl.change(show, [ep_dd, step_sl], outs)
    live.click(run_live, [ep_dd, step_sl], outs)
    demo.load(show, [ep_dd, step_sl], outs)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
