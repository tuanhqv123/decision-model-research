"""eval_d1 jsonl -> the GUI-Libra harness plan-file format, so the leaderboard's own eval_rl.py scores our
decisions exactly as it scores a generative agent's answers (pixel point_2d, relative_coord=0)."""
import json
import sys

NAME = {"click": "Click", "type_text": "Write", "long_press": "LongPress", "scroll": "Scroll", "wait": "Wait",
        "navigate_back": "NavigateBack", "open_app": "OpenApp", "status": "Terminate"}
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w") as f:
    for line in open(src):
        e = json.loads(line)
        pt = [round(e["pt"][0]), round(e["pt"][1])] if e.get("pt") else [-100, -100]
        val = e["value"] if e["value"] not in (None, "") else "None"
        ans = {"action_type": NAME[e["pred"]], "action_target": "None", "value": val, "point_2d": pt}
        f.write(json.dumps({"episode_id": e["episode_id"], "step": e["step"], "action_type": NAME[e["pred"]],
                            "value": val, "point_2d": pt,
                            "response": "<answer>\n" + json.dumps(ans, ensure_ascii=False) + "\n</answer>"}) + "\n")
print("wrote", dst)
