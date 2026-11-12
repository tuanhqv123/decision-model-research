"""Replay recorded states (the step where v4 emitted `answer`) on the served checkpoint; print what it answers now.
python3 replay_answers.py URL TOKEN RUN_DIR [RUN_DIR...]"""
import base64, glob, json, os, sys, urllib.request
url, tok, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
def ask(si, img):
    body = json.dumps({"goal": si["goal"], "history": si["history"], "image_b64": img, "track": "VA",
                       "elements": si["elements"], "return_readouts": True}).encode()
    req = urllib.request.Request(url + "/v1/decide", data=body, headers={"content-type": "application/json", "x-api-key": tok})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())
for r in runs:
    for f in sorted(glob.glob(os.path.join(r, "episodes", "*.jsonl"))):
        if ".abandoned." in f: continue
        rows = [json.loads(l) for l in open(f)]
        for s in rows:
            if s.get("type") != "step" or (s.get("action_executed") or {}).get("action_type") != "answer": continue
            p = os.path.join(r, "screens", os.path.basename(s.get("screenshot") or ""))
            img = base64.b64encode(open(p, "rb").read()).decode()
            d = ask(s["state_in"], img)
            ap = (d.get("readouts") or {}).get("action", {}).get("probabilities", {})
            top = sorted(ap, key=lambda k: -ap[k])[:3]
            print("%-45s v4 answered %-30r | now: %-10s %s answer_value=%s text=%r" % (
                rows[0]["task_name"], s["action_executed"]["text"][:30], d.get("action"),
                ["%s:%.2f" % (k, ap[k]) for k in top], d.get("answer_value"), (d.get("answer_text") or "")[:30]), flush=True)
