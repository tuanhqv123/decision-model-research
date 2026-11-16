"""AC-High without `status`: re-decode the SAVED readouts of the standard run, taking the best non-terminate action of the
same pass (what `eval_d1 --exclude-actions terminate` does), then score with InfiGUI's scorer."""
import json, os, sys
W = os.path.expanduser("~/d1work/actest")
sys.path.insert(0, f"{W}/code/benchmarks/train")
from ac_common import FROM_OPTION, decode_app_choice
from score_actest import score, to_infigui_pred
apps = {}
for l in open(f"{W}/data/compact_actest_v3.jsonl"):
    r = json.loads(l); apps[(r["ep"], r["step"])] = r["apps"]
preds, changed, same = {}, 0, 0
for l in open(f"{W}/eval/eval-v3_s0_ckpt_1824_actest_v3-VA-val.jsonl"):
    r = json.loads(l); key = (int(r["episode_id"]), int(r["step"]))
    pred, value, pt = r["pred"], r.get("value"), r.get("pt")
    if pred == "status":
        ans = r["readouts"]; probs = ans["action"]["probabilities"]
        choice = max((k for k in probs if k != "terminate"), key=lambda k: probs[k])
        pred = FROM_OPTION[choice]
        if pred == "scroll": value = (ans.get("scroll_dir") or {}).get("choice")
        elif pred == "type_text": value = (ans.get("type_text") or {}).get("text") or ""
        elif pred == "open_app": value = decode_app_choice(ans.get("app_name"), apps[key])
        else: value = None
        pt = r.get("el_pt") if pred in ("click", "long_press", "type_text") else None
        changed += 1
    else:
        same += 1
    preds[key] = to_infigui_pred(pred, value, pt)
s = score(preds, json.load(open(f"{W}/infigui_test.json")))
print("re-decoded", changed, "status rows; unchanged", same)
print("AC-HIGH-NOSTATUS " + json.dumps(s))
