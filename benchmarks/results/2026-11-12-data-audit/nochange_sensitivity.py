"""Does v3 READ the "→ no change" line? (2026-11-12) Counterfactual replay on the exact inputs v3 saw on AndroidWorld.
States: self-play seed 1001 (v3 ckpt_1824), every step right after an executed non-wait action whose history line says
"→ no change". The recorded state_in (goal, history, elements) + the saved screenshot are sent again, in 3 variants:
  A as recorded (last line "... → no change")   B last line rewritten to "... → screen changed"   C last line removed
Measured: does the decision REPEAT the failed action (same type; for click: chosen point inside the element it hit;
for scroll: same direction)? If A ≈ B, the model ignores the effect label. Replay check: A must equal the recorded raw decision.
    python3 nochange_sensitivity.py RUN_DIR URL TOKEN"""
import base64, collections as C, glob, json, os, sys, urllib.request
run, url, tok = sys.argv[1:4]
def decide(state, img_b64):
    body = json.dumps({"goal": state["goal"], "history": state["history"], "image_b64": img_b64, "track": "VA",
                       "elements": state["elements"], "return_readouts": False}).encode()
    req = urllib.request.Request(url + "/v1/decide", data=body, headers={"content-type": "application/json", "x-api-key": tok})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
def repeats(d, prev):
    a = (d.get("action") or "").lower(); pa = prev["action_executed"]["action_type"]
    pa = {"input_text": "type_text"}.get(pa, pa)
    if a != pa: return False
    if a in ("click", "long_press", "type_text"):
        b = (prev.get("click_target") or {}).get("bbox"); pt = d.get("element_point") or d.get("point")
        return bool(b and pt and b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3])
    if a == "scroll": return (d.get("direction") or "") == (prev["action_executed"].get("direction") or "")
    return True
c = C.Counter(); by = C.Counter(); n = 0
for f in sorted(glob.glob(os.path.join(run, "episodes", "*.jsonl"))):
    if ".abandoned." in f: continue
    st = sorted([json.loads(l) for l in open(f) if '"type": "step"' in l], key=lambda s: s["i"])
    for k in range(1, len(st)):
        p, s = st[k - 1], st[k]
        ex = p.get("action_executed") or {}
        if not ex or ex.get("action_type") == "wait" or not (p.get("history_line") or "").endswith("→ no change"): continue
        si = s.get("state_in"); shot = s.get("screenshot")
        if not si or not si["history"] or not shot: continue
        path = os.path.join(run, "screens", os.path.basename(shot))
        if not os.path.exists(path): continue
        img = base64.b64encode(open(path, "rb").read()).decode()
        last = si["history"][-1]; assert last.endswith("→ no change"), last
        A = decide(si, img)
        B = decide(dict(si, history=si["history"][:-1] + [last[:-len("→ no change")] + "→ screen changed"]), img)
        Cc = decide(dict(si, history=si["history"][:-1]), img)
        raw = s.get("action_raw") or {}
        n += 1; kind = {"input_text": "type_text"}.get(ex["action_type"], ex["action_type"])
        c["replay A == recorded action"] += (A.get("action") == raw.get("action"))
        for name, d in (("A as recorded (no change)", A), ("B says screen changed", B), ("C line removed", Cc)):
            r = repeats(d, p); c[name + ": repeats failed action"] += r; by[(kind, name)] += r
            c[name + ": says status"] += (d.get("action") == "status")
        c["A vs B: same action type"] += A.get("action") == B.get("action")
        c["A vs B: same action AND same point"] += A.get("action") == B.get("action") and A.get("element_point") == B.get("element_point")
        by[(kind, "n")] += 1
print("states:", n)
for k, v in c.items(): print("  %-48s %d (%.0f%%)" % (k, v, 100 * v / max(n, 1)))
print("by the failed action's type:")
for kind in sorted({k for k, _ in by}): print("  %-10s n=%d  " % (kind, by[(kind, "n")]) + "  ".join("%s %d" % (nm.split(" ")[0], by[(kind, nm)]) for nm in ("A as recorded (no change)", "B says screen changed", "C line removed")))
