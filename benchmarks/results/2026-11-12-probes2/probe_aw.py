"""Behavioural probes, rebuilt after the 2026-11-12 review (decision 15 in docs/research/2026-11-12-V4-WORKLOG-vi.md).
States: the seed-30 AndroidWorld run of v3 under harness V8 (aw116_v3h_s0*, no action masking), `state_in` + screenshot.
The SAME states are sent to every checkpoint, so v3 and v4 are compared on identical inputs. Nothing runs on a device.
Each probe edits ONE thing and records P(option) and the top-1 action, per episode (bootstrap is by episode).
  P1 DIR effect word: last line "→ no change" vs "→ screen changed" (after a click/scroll/type)   want P(repeat) lower
  P2 DIR the same failed click line 1..4 times                                                     want P(repeat) to fall
  P3 INV k neutral "wait" lines at the START of the history, early steps                           want P(status) flat
  P4 DIR states where the task IS done (final step of a successful episode): goal as is vs a second,
         not-done task of the same app in the held-out phrasing "First, {a} Then, {b}"             want P(status) to fall
  P5 DIR right after a click on an editable field, mark it focused                                 want P(type_text) flat
         (v4 drops focus; AW never reports it)
    python3 probe_aw.py OUT.jsonl URL TOKEN RUN_DIR [RUN_DIR...]"""
import base64, glob, json, os, random, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ac_common as A  # noqa: E402

out, url, tok, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
random.seed(0)
PROBE = "First, {a} Then, {b}"
OPT = {"click": "click", "scroll": "scroll", "input_text": "write", "long_press": "long_press", "open_app": "open_app"}


def ask(si, img, **edit):
    st = dict(si, **edit)
    body = json.dumps({"goal": st["goal"], "history": st["history"], "image_b64": img, "track": "VA",
                       "elements": st["elements"], "return_readouts": True}).encode()
    req = urllib.request.Request(url + "/v1/decide", data=body, headers={"content-type": "application/json", "x-api-key": tok})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def probs(d):
    return (d.get("readouts") or {}).get("action", {}).get("probabilities", {}) or {}


def p_el_in(d, st, box):
    pr = (d.get("readouts") or {}).get("target_el", {}).get("probabilities") or {}
    kept, _ = A.select_elements(st["elements"], 1080, 2400, d.get("element_rule"))
    m = 0.0
    for i, e in enumerate(kept, 1):
        c = A.element_centre(e)
        if box and box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]:
            m += pr.get(str(i), 0.0)
    return m


eps = []
for r in runs:
    for f in sorted(glob.glob(os.path.join(r, "episodes", "*.jsonl"))):
        if ".abandoned." in f:
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        st = sorted([x for x in rows if x.get("type") == "step"], key=lambda s: s["i"])
        res = [x for x in rows if x.get("type") == "result"]
        if st and res:
            eps.append((r, res[-1], st))
goals_by_app = {}
for r, res, st in eps:
    goals_by_app.setdefault(res["task_name"][:5], set()).add(st[0]["state_in"]["goal"] if st[0].get("state_in") else None)


def img_of(r, s):
    p = os.path.join(r, "screens", os.path.basename(s.get("screenshot") or ""))
    return base64.b64encode(open(p, "rb").read()).decode() if os.path.isfile(p) else None


fo = open(out, "w")


def rec(probe, variant, ep, d, **kw):
    p = probs(d)
    fo.write(json.dumps(dict(probe=probe, variant=variant, ep=ep, top1=max(p, key=p.get) if p else None, p=p, **kw)) + "\n")
    fo.flush()


for r, res, st in eps:
    ep = res["task_name"]
    for k, s in enumerate(st):
        si = s.get("state_in")
        img = img_of(r, s) if si else None
        if not si or not img:
            continue
        prev = st[k - 1] if k else None
        ex = (prev or {}).get("action_executed") or {}
        hist = si["history"]
        if prev and ex.get("action_type") in OPT and hist and hist[-1].endswith("→ no change"):
            box = (prev.get("click_target") or {}).get("bbox")
            typ, last, base = OPT[ex["action_type"]], hist[-1], hist[:-1]
            for v, h in (("no change", hist), ("screen changed", base + [last[:-len("→ no change")] + "→ screen changed"])):
                d = ask(si, img, history=h)
                rec("P1", v, ep, d, same=typ, p_same_el=(p_el_in(d, si, box) if ex["action_type"] == "click" else None))
            if ex["action_type"] == "click":
                for j in (1, 2, 3, 4):
                    d = ask(si, img, history=base + [last] * j)
                    rec("P2", "%dx" % j, ep, d, same="click", p_same_el=p_el_in(d, si, box))
        last_success = (res.get("score") or 0) >= 1 and k == len(st) - 1
        if s["i"] <= 3 and not last_success and random.random() < 0.5:
            for kk in (0, 3, 6, 10):
                rec("P3", "+%d" % kk, ep, ask(si, img, history=["wait"] * kk + hist))
        if last_success:
            others = sorted(g for g in goals_by_app.get(ep[:5], ()) if g and g != si["goal"])
            if others:
                b = random.choice(others)
                rec("P4", "as is", ep, ask(si, img))
                rec("P4", "+ second task", ep, ask(si, img, goal=PROBE.format(a=si["goal"].strip(), b=b[0].lower() + b[1:])))
        t = (prev or {}).get("click_target") or {}
        if prev and ex.get("action_type") == "click" and "e" in str(t.get("flags") or ""):
            box, els, hit = t.get("bbox"), [dict(e) for e in si["elements"]], 0
            for e in els:
                bp = e.get("bbox_pixels") or {}
                if e.get("is_editable") and bp and box and abs(bp.get("x_min", -9) - box[0]) <= 2 and abs(bp.get("y_min", -9) - box[1]) <= 2:
                    e["is_focused"] = True
                    hit += 1
            if hit:
                rec("P5", "as served", ep, ask(si, img))
                rec("P5", "focused", ep, ask(si, img, elements=els))
print("done", out)
