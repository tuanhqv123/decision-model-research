"""Gate 1 for recovery training (2026-11-12): recorded AndroidWorld states one step AFTER a wrong navigation, where
going back is the right next action (hand-picked from the v3h error analysis). Reports P(navigate_back), its rank and
the top-1 action for each served model.  python3 probe_recover.py RUN_DIR[,RUN_DIR] NAME=URL [NAME=URL ...]"""
import base64, glob, json, os, sys, urllib.request
STATES = [("MarkorDeleteAllNotes", 2), ("MarkorDeleteNewestNote", 2), ("MarkorDeleteNote", 2), ("MarkorMoveNote", 3),
          ("MarkorMoveNote", 5), ("ExpenseAddMultipleFromGallery", 3), ("SaveCopyOfReceiptTaskEval", 6),
          ("NotesTodoItemCount", 2), ("SimpleCalendarAddOneEventInTwoWeeks", 8)]
runs = sys.argv[1].split(","); servers = [a.split("=", 1) for a in sys.argv[2:]]
def load(task):
    for r in runs:
        for f in glob.glob(os.path.join(r, "episodes", "*.jsonl")):
            if ".abandoned." in f: continue
            rows = [json.loads(l) for l in open(f)]
            if rows[0]["task_name"] == task: return r, [x for x in rows if x.get("type") == "step"]
def ask(url, si, img):
    body = json.dumps({"goal": si["goal"], "history": si["history"], "image_b64": img, "track": "VA",
                       "elements": si["elements"], "return_readouts": True}).encode()
    req = urllib.request.Request(url + "/v1/decide", data=body, headers={"content-type": "application/json", "x-api-key": "tok"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())
tot = {n: [0, 0.0] for n, _ in servers}
for task, i in STATES:
    got = load(task)
    if not got or len(got[1]) <= i: print("%-38s step %d | missing" % (task, i)); continue
    r, st = got; s = st[i]
    img = base64.b64encode(open(os.path.join(r, "screens", os.path.basename(s["screenshot"])), "rb").read()).decode()
    out = []
    for n, u in servers:
        d = ask(u, s["state_in"], img); ap = (d.get("readouts") or {}).get("action", {}).get("probabilities", {})
        rk = sorted(ap, key=lambda k: -ap[k]); p = ap.get("navigate_back", 0.0)
        tot[n][0] += rk[0] == "navigate_back"; tot[n][1] += p
        out.append("%s: P(back)=%.3f rank %d top1=%s" % (n, p, rk.index("navigate_back") + 1 if "navigate_back" in rk else -1, rk[0]))
    print("%-38s step %d | %s" % (task, i, " | ".join(out)), flush=True)
for n, (c, p) in tot.items(): print("%s: back top-1 %d/%d, mean P(back) %.3f" % (n, c, len(STATES), p / len(STATES)))
