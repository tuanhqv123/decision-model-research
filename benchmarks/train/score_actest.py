#!/usr/bin/env python3
"""Score v3 predictions on AndroidControl-High with InfiGUI's scorer, unchanged (`vendor/infigui_eval.py`).

Test set: InfiGUI's processed AndroidControl test (`InfiX-ai/android_control_test`): the 1,543 official test episodes,
8,444 steps; every `step_check_pam` was checked against Google's raw action for the same episode and step (all equal,
2026-11-11). Predictions: `eval_d1.py` rows (`episode_id`, `step`, `pred`, `value`, `pt`) on `compact_actest_v3`.
Coordinates are original screen pixels, so the scorer is called with resized size = original size.

    python score_actest.py --pred eval-<tag>-VA-val.jsonl --test android_control_test.json [--out per_step.jsonl]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vendor.infigui_eval import evaluate_android_control_action  # noqa: E402

# A swipe whose start -> end direction InfiGUI's `predict_direction` reads back as the label.
_SWIPE = {"up": ([500, 900], [500, 100]), "down": ([500, 100], [500, 900]),
          "left": ([900, 500], [100, 500]), "right": ([100, 500], [900, 500])}
_NOWHERE = [-10 ** 6, -10 ** 6]  # a click with no chosen element: outside every box and every distance
# InfiGUI's check_text credits `pred in gt`; '' is in every string, so an empty answer would score as correct.
_NO_TEXT = "\u2205<no text>"


def _text(value):
    return value.strip() if isinstance(value, str) and value.strip() else _NO_TEXT


def to_infigui_pred(pred, value, pt):
    """One v3 decision (eval_d1 names) as InfiGUI's action dict."""
    if pred in ("click", "long_press"):
        return {"action": pred, "coordinate": list(pt) if pt else list(_NOWHERE)}
    if pred == "type_text":
        return {"action": "type", "text": _text(value)}
    if pred == "open_app":
        return {"action": "open", "text": _text(value)}
    if pred == "scroll":
        if value not in _SWIPE:
            return {"action": "invalid"}
        start, end = _SWIPE[value]
        return {"action": "swipe", "coordinate": list(start), "coordinate2": list(end)}
    if pred == "navigate_back":
        return {"action": "system_button", "button": "Back"}
    if pred == "navigate_home":
        return {"action": "system_button", "button": "Home"}
    if pred == "wait":
        return {"action": "wait"}
    if pred == "status":
        return {"action": "terminate", "status": "success"}
    return {"action": "invalid"}


def score(preds, test, per_step=None):
    """preds: {(episode_id, step): InfiGUI action dict}. Every test step counts; a missing prediction is wrong."""
    n = exact = typ = missing = 0
    for x in test:
        ep, W, H = x["episode"]["episode_id"], x["width"], x["height"]
        for i, gt in enumerate(x["step_check_pams"]):
            n += 1
            p = preds.get((ep, i))
            if p is None:
                missing += 1
                t, e = False, False
            else:
                t, e = evaluate_android_control_action(p, gt, W, H, W, H, pred_type="abs_resized",
                                                       gt_type="abs_resized")
            typ += bool(t)
            exact += bool(e)
            if per_step is not None:
                per_step.append({"episode_id": ep, "step": i, "gt": gt, "pred": p, "type_match": bool(t),
                                 "exact_match": bool(e)})
    return {"steps": n, "exact": exact, "type": typ, "missing": missing,
            "exact_match_acc": 100.0 * exact / n if n else 0.0, "type_match_acc": 100.0 * typ / n if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    preds = {}
    for line in open(a.pred):
        r = json.loads(line)
        preds[(int(r["episode_id"]), int(r["step"]))] = to_infigui_pred(r["pred"], r.get("value"), r.get("pt"))
    rows = [] if a.out else None
    s = score(preds, json.load(open(a.test)), rows)
    print("AC-HIGH " + json.dumps(s), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
