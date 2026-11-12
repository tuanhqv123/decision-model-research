"""Behavioural probes of v3 (ckpt_1824) with controlled edits of REAL AndroidWorld inputs (2026-11-12).
CheckList-style: each probe changes ONE thing in the input and reads the model's own probabilities (readouts).
States: self-play seed 1001 (state_in recorded + screenshot). Nothing is executed on a device.
  P1 DIR  effect word: last history line "→ no change" vs "→ screen changed"      expect P(repeat) lower with "no change"
  P2 DIR  repeated failure: the same failed line 1, 2, 3, 4 times in a row         expect P(repeat) to FALL as it stacks
  P3 INV  position: k neutral "wait" lines inserted at the START of the history    expect P(status) flat in k
  P4 DIR  unfinished goal: on states the checker says are DONE, append a clause    expect P(status) to fall
          that is not done yet ("Then do it once more.")
  P5 DIR  focus: after a click on a field, mark that field is_focused=True          expect P(type_text) to rise
          (AW never reports focus; training shows it on 93% of these screens)
    python3 probe_v3.py RUN_DIR URL TOKEN"""
import base64, collections as C, glob, json, os, random, statistics as S, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ac_common as A
run, url, tok = sys.argv[1:4]; random.seed(0)
def ask(si, img, **edit):
    st = dict(si, **edit)
    body = json.dumps({"goal": st["goal"], "history": st["history"], "image_b64": img, "track": "VA", "elements": st["elements"], "return_readouts": True}).encode()
    req = urllib.request.Request(url + "/v1/decide", data=body, headers={"content-type": "application/json", "x-api-key": tok})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
def pact(d, k): return (d.get("readouts") or {}).get("action", {}).get("probabilities", {}).get(k, 0.0)
def p_el_in(d, st, box, rule):
    """probability mass the target_el head puts on elements whose centre lies in `box` [x0,y0,x1,y1]"""
    probs = (d.get("readouts") or {}).get("target_el", {}).get("probabilities") or {}
    kept, _ = A.select_elements(st["elements"], 1080, 2400, rule)
    m = 0.0
    for i, e in enumerate(kept, 1):
        c = A.element_centre(e)
        if box and box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]: m += probs.get(str(i), 0.0)
    return m
OPT = {"click": "click", "scroll": "scroll", "input_text": "write", "long_press": "long_press", "open_app": "open_app"}
eps = []
for f in sorted(glob.glob(os.path.join(run, "episodes", "*.jsonl"))):
    if ".abandoned." in f: continue
    st = sorted([json.loads(l) for l in open(f) if '"type": "step"' in l], key=lambda s: s["i"])
    eps.append(st)
def img_of(s):
    p = os.path.join(run, "screens", os.path.basename(s.get("screenshot") or ""))
    return base64.b64encode(open(p, "rb").read()).decode() if os.path.isfile(p) else None
res = C.defaultdict(list)
for st in eps:
    for k, s in enumerate(st):
        si = s.get("state_in"); img = img_of(s) if si else None
        if not si or not img: continue
        prev = st[k - 1] if k else None; ex = (prev or {}).get("action_executed") or {}
        done = k > 0 and (prev.get("probe_score") or 0) >= 1
        # P1 + P2: right after a non-wait action labelled "no change"
        if prev and ex.get("action_type") in OPT and (prev.get("history_line") or "").endswith("→ no change") and si["history"]:
            box = (prev.get("click_target") or {}).get("bbox"); typ = OPT[ex["action_type"]]; last = si["history"][-1]
            base = si["history"][:-1]
            for name, h in (("P1 no change", si["history"]), ("P1 screen changed", base + [last[:-len("→ no change")] + "→ screen changed"])):
                d = ask(si, img, history=h); rule = d.get("element_rule")
                res[name + " | P(same type)"].append(pact(d, typ))
                if ex["action_type"] == "click": res[name + " | P(same element)"].append(pact(d, "click") * p_el_in(d, si, box, rule))
            if ex["action_type"] == "click":
                for j in (1, 2, 3, 4):
                    d = ask(si, img, history=base + [last] * j); rule = d.get("element_rule")
                    res["P2 failed %dx | P(click same element)" % j].append(pact(d, "click") * p_el_in(d, si, box, rule))
        # P3: early, not-done states
        if not done and s["i"] <= 3 and random.random() < 0.6:
            for kk in (0, 3, 6, 10):
                d = ask(si, img, history=["wait"] * kk + si["history"]); res["P3 +%d wait lines at start | P(status)" % kk].append(pact(d, "terminate"))
        # P4: done states
        if done:
            for name, g in (("as is", si["goal"]), ("+ unfinished clause", si["goal"].rstrip(".") + ". Then do it once more.")):
                d = ask(si, img, goal=g); res["P4 done state, goal %s | P(status)" % name].append(pact(d, "terminate"))
        # P5: right after a click on an editable field
        t = (prev or {}).get("click_target") or {}
        if prev and ex.get("action_type") == "click" and "e" in str(t.get("flags") or ""):
            box = t.get("bbox")
            els = [dict(e) for e in si["elements"]]
            hit = 0
            for e in els:
                b = e.get("bbox_pixels") or {}
                if e.get("is_editable") and b and box and abs(b.get("x_min", -9) - box[0]) <= 2 and abs(b.get("y_min", -9) - box[1]) <= 2: e["is_focused"] = True; hit += 1
            if hit:
                for name, E in (("as served (no focus)", si["elements"]), ("field marked focused", els)):
                    d = ask(si, img, elements=E); res["P5 %s | P(type_text)" % name].append(pact(d, "write"))
for k in sorted(res):
    v = res[k]; print("%-60s n=%3d  mean %.3f  median %.3f" % (k, len(v), S.mean(v), S.median(v)))
json.dump(res, open("probe_v3.json", "w"))
