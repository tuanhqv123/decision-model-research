"""Demo data for the DMA-2B Space: three AndroidControl test episodes, AC-Low setting (step instruction shown), every
step run through the released checkpoint with all readouts. Episodes are chosen by a fixed rule from the evaluation
verdicts: the first two episodes of 4-6 steps (by episode id) with every step right and different apps, and the first
one with exactly one wrong step. Each step records whether the decision matches the human's action (InfiGUI scorer)."""
import json, os, shutil, sys, time
sys.path.insert(0, os.path.expanduser("~/d1pilot/hf_space_build"))
import ac_common as v
from example import load_image, decode

D = os.path.expanduser("~/d1pilot")
OUT = os.path.join(D, "hf_space_build", "space")
CK = os.path.join(D, "runs/exec2_s0/ckpt_7000")
rows = {}
for l in open(os.path.join(D, "actest/data/compact_actest_v4.jsonl")):
    r = json.loads(l)
    rows.setdefault(r["ep"], []).append(r)
verdict = {(int(p["episode_id"]), p["step"]): bool(p["exact_match"])
           for p in map(json.loads, open(os.path.join(D, "gate_exec2/perstep-exec2_actest_low_noterm.jsonl")))}
evp = {(int(p["episode_id"]), p["step"]): p for p in map(json.loads, open(os.path.join(D, "gate_exec2/eval-exec2_actest_low_noterm-VA-val.jsonl")))}
gold_ps = {(int(p["episode_id"]), p["step"]): p["gt"] for p in map(json.loads, open(os.path.join(D, "gate_exec2/perstep-exec2_actest_low_noterm.jsonl")))}

def app(rs):
    return rs[0]["action"].get("app_name") or rs[0]["goal"].split()[0]

good, one_err, apps_seen = [], [], set()
for ep in sorted(rows):
    rs = sorted(rows[ep], key=lambda r: r["step"])
    if not 4 <= len(rs) <= 6 or rs[0]["action"]["action_type"] != "open_app":
        continue
    wrong = sum(not verdict[(ep, r["step"])] for r in rs)
    a = rs[0]["action"]["app_name"]
    if wrong == 0 and len(good) < 2 and a not in apps_seen:
        good.append(ep); apps_seen.add(a)
    elif wrong == 1 and not one_err and a not in apps_seen:
        one_err.append(ep); apps_seen.add(a)
    if len(good) == 2 and one_err:
        break
chosen = good + one_err
print("episodes", chosen)

def same_as_eval(ep, step, out):
    e = evp[(ep, step)]
    el = (out.get("element") or {}).get("id")
    return e["pred"] == out["action"] and (el is None or e.get("el_idx") == el)


import torch
from transformers import AutoModel
model = AutoModel.from_pretrained(CK, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
model.engine.calibration = None
LAB = {"click": "click", "write": "type text", "long_press": "long press", "scroll": "scroll", "wait": "wait",
       "navigate_back": "back", "navigate_home": "home", "open_app": "open app", "terminate": "done", "answer": "answer"}
os.makedirs(os.path.join(OUT, "screens"), exist_ok=True)
data = []
for ep in chosen:
    rs = sorted(rows[ep], key=lambda r: r["step"])
    steps = []
    for r in rs:
        W, H = r["W"], r["H"]
        state, kept, keep_idx = v.element_inputs_v3(r["goal"], r["history_v3"], r["elements"], W, H,
                                                    "all_reading_noime", r["local_goal"])
        qs = v.build_questions("VA", kept, contract="v4", apps=r["apps"])
        img_src = os.path.join(D, "actest/data", r["img"])
        shot = "screens/ep%d_%d.jpg" % (ep, r["step"])
        shutil.copy(img_src, os.path.join(OUT, shot))
        t0 = time.time()
        with torch.inference_mode():
            ans = model.system_one_batch([(state, qs, [load_image(img_src)])])[0]["answers"]
        ms = (time.time() - t0) * 1000
        out = decode(ans, kept, keep_idx, r["elements"], r["apps"])
        ap = sorted(((LAB[k], float(p)) for k, p in ans["action"]["probabilities"].items()), key=lambda x: -x[1])
        act = out["action"]
        box, second = None, {"title": "", "rows": []}
        if act in v.SPATIAL:
            pr = sorted(((int(k), float(p)) for k, p in ans["target_el"]["probabilities"].items()), key=lambda x: -x[1])[:5]
            second = {"title": "Element probabilities (top 5)",
                      "rows": [["[%d] %s" % (i, v.element_label(kept[i - 1])), p] for i, p in pr]}
            if out.get("element"):
                b = kept[out["element"]["id"] - 1]["bbox"]
                s = 720.0 / W
                box = [round(b["x_min"] * s), round(b["y_min"] * s), round(b["x_max"] * s), round(b["y_max"] * s)]
        elif act == "scroll":
            second = {"title": "Scroll direction", "rows": sorted(([k, float(p)] for k, p in ans["scroll_dir"]["probabilities"].items()), key=lambda x: -x[1])}
        elif act == "open_app":
            second = {"title": "App (top 5)", "rows": sorted(([k, float(p)] for k, p in ans["app_name"]["probabilities"].items()), key=lambda x: -x[1])[:5]}
        g = r["action"]
        steps.append({"screenshot": shot, "instruction": r["local_goal"], "history": r["history_v3"], "state": state,
                      "action": act, "probability": out["probability"], "element": out.get("element"), "box": box,
                      "json_action": out.get("json_action"), "action_probs": ap, "second": second,
                      "model_ms": round(ms, 1), "human": {k: g[k] for k in g if k in ("action_type", "app_name", "direction", "text")},
                      "matches_human": (verdict[(ep, r["step"])] if same_as_eval(ep, r["step"], out) else None)})
    data.append({"title": rs[0]["goal"][:60] + ("…" if len(rs[0]["goal"]) > 60 else ""), "episode": ep,
                 "goal": rs[0]["goal"], "steps": steps})
    print(ep, [s["action"] for s in steps], [s["matches_human"] for s in steps])
json.dump(data, open(os.path.join(OUT, "data.json"), "w"), indent=1)
