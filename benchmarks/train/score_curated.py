#!/usr/bin/env python3
"""Score eval_d1 predictions on AndroidControl-Curated (arXiv 2510.18488; HF batwBMW/AndroidControl_Curated).

The rules are re-implemented from the authors' `src/eval/utils.py` (no license is stated, so the file is not copied):
  * a step is right when the action type is right and, for click / long_press, the centre of the predicted box lies
    inside the ground-truth box (`gt_max_bbox`, absolute pixels of the original screenshot);
  * typed text and app names are compared by containment or token F1 >= 0.5;
  * scroll: "scroll:<direction>" is turned into "swipe:<opposite>" (their `correct_type`), so our scroll direction
    (the direction the content moves) maps to their swipe label;
  * the task-improved file adds `candidate_actions` (other valid actions): any match counts, and the rows with a
    `revised_task` are re-run with that task (the other rows keep the prediction of the unrevised high-level run).
Rows are matched to AndroidControl test steps by their image path `android_control_images/<episode>/step_<i>.png`.

    python3 score_curated.py CURATED.json PRED.jsonl [--improved] [--revised PRED_REVISED.jsonl] [--out OUT.json]
"""
import argparse
import json
import re


def correct_type(t):
    if "tap" in t:
        t = t.replace("tap", "click")
    if "scroll" in t:
        t = t.replace("scroll", "swipe")
        for a, b in (("up", "down"), ("down", "up"), ("left", "right"), ("right", "left")):
            if a in t:
                t = t.replace(a, b)
                break
    if "press_back" in t:
        t = t.replace("press_back", "navigate_back")
    if "type" in t:
        t = t.replace("type", "input_text")
    return t


def text_match(gt, pred):
    pred, gt = pred.replace("[", "").replace("]", ""), gt.replace("[", "").replace("]", "")
    if gt in pred or pred in gt:
        return True
    p, g = set(pred.lower().split()), set(gt.lower().split())
    if len(p) == 1 and len(g) == 1:
        a, b = next(iter(p)), next(iter(g))
        if a in b or b in a:
            return True
    common = p & g
    prec = len(common) / len(p) if p else 0
    rec = len(common) / len(g) if g else 0
    return prec + rec > 0 and 2 * prec * rec / (prec + rec) >= 0.5


def type_ok(gt, pred):
    if pred is None:
        return False
    gt, pred = correct_type(gt), correct_type(pred)
    if ":" in gt and ":" in pred:
        g1, g2 = gt.split(":")[0], gt.split(":")[1]
        p1, p2 = pred.split(":")[0], pred.split(":")[1]
        return g1 == p1 and text_match(g2, p2)
    return gt.lower() == pred.lower()


def inside(gt_box, box):
    if box is None or not gt_box or len(gt_box) != 4:
        return False
    cx, cy = (int(box[0]) + int(box[2])) / 2, (int(box[1]) + int(box[3])) / 2
    return gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]


def single(gt_type, gt_box, ptype, pbox):
    if gt_type not in ("click", "long_press"):
        t = type_ok(gt_type, ptype)
        return None, t, t
    b, t = inside(gt_box, pbox), type_ok(gt_type, ptype)
    return b, t, b and t


def to_curated(p):
    """(action type string, box) of one eval_d1 prediction in the authors' answer vocabulary."""
    a, v = p["pred"], p.get("value")
    if a in ("click", "long_press"):
        pt = p.get("pt") or p.get("el_pt")
        box = [int(pt[0]) - 1, int(pt[1]) - 1, int(pt[0]) + 1, int(pt[1]) + 1] if pt else None
        return a, box
    if a == "type_text":
        return "input_text:%s" % (v or ""), None
    if a == "open_app":
        return "open_app:%s" % (v or ""), None
    if a == "scroll":
        return "scroll:%s" % (v or ""), None
    return a, None   # wait, navigate_back, navigate_home, status


def key(row):
    m = re.search(r"/(\d+)/step_(\d+)\.png$", row["image"])
    return int(m.group(1)), int(m.group(2))


def gt_type_of(row):
    t = row["gt_action"]
    if row.get("gt_input_text") not in (None, "no input text"):
        t = t + ":" + row["gt_input_text"].lower()
    return correct_type(t) if "scroll" in t else t


def score(rows, preds, improved=False, revised=None):
    out = {"rows": 0, "missing": 0, "type": 0, "grounding_n": 0, "grounding": 0, "sr": 0}
    per = []
    for r in rows:
        k = key(r)
        p = (revised or {}).get(k) if (improved and "revised_task" in r) else preds.get(k)
        out["rows"] += 1
        if p is None:
            out["missing"] += 1
            per.append({"key": k, "ok": False})
            continue
        ptype, pbox = to_curated(p)
        golds = [(gt_type_of(r), r.get("gt_bbox") if "gt_bbox" in r else r.get("gt_max_bbox"))]
        if improved:
            golds += [(c.get("action_type", ""), c.get("action_bounds", [])) for c in r.get("candidate_actions") or []]
        flags = [single(gt, gb, ptype, pbox) for gt, gb in golds]
        b = any(f[0] for f in flags) if improved else flags[0][0]
        t = any(f[1] for f in flags)
        s = any(f[2] for f in flags)
        out["type"] += t
        if golds[0][0].split(":")[0] in ("click", "long_press"):
            out["grounding_n"] += 1
            out["grounding"] += bool(b)
        out["sr"] += s
        per.append({"key": k, "ok": bool(s), "type_ok": bool(t)})
    n = out["rows"]
    out.update({"type_pct": round(100 * out["type"] / n, 2), "sr_pct": round(100 * out["sr"] / n, 2),
                "grounding_pct": round(100 * out["grounding"] / max(1, out["grounding_n"]), 2)})
    return out, per


def load_preds(path):
    return {(int(p["episode_id"]), p["step"]): p for p in map(json.loads, open(path))} if path else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("curated")
    ap.add_argument("pred")
    ap.add_argument("--improved", action="store_true")
    ap.add_argument("--revised")
    ap.add_argument("--out")
    a = ap.parse_args()
    res, per = score(json.load(open(a.curated)), load_preds(a.pred), a.improved, load_preds(a.revised))
    print(json.dumps(res))
    if a.out:
        json.dump({"summary": res, "per_row": per}, open(a.out, "w"))


if __name__ == "__main__":
    main()
