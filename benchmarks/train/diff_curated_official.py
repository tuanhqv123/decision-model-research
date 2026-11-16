#!/usr/bin/env python3
"""Differential test: our AndroidControl-Curated scorer (score_curated.py) vs the authors' own scoring functions.

The authors' code (github.com/zuitbjc1096/AndroidControl_Curated, src/eval/utils.py) is not vendored here (no license).
Pass the path of a clone with --official. For every row, our prediction is written as the answer string their parser
reads ("{'action_type': ..., 'bbox_2d': [...]}"), the ground truth is built exactly as their
`get_all_response_android_batch` builds `action_bounds`, and their `calculate_single_android` /
`calculate_multi_android` gives the official flags. These are compared row by row with ours.

    python3 diff_curated_official.py --official <clone>/src/eval --curated android_control_high_bbox.json \
        --pred eval-...jsonl [--improved --revised eval-...jsonl]
"""
import argparse
import contextlib
import io
import json
import sys

from score_curated import correct_type, key, load_preds, score, to_curated


def official_gt(row, improved):
    """`action_bounds` as built in the authors' evaluate_actions_androidControl_vllm.py (lines 88-114)."""
    if "gt_input_text" in row and row["gt_input_text"] != "no input text":
        t = row["gt_action"] + ":" + row["gt_input_text"].lower()
    else:
        t = row["gt_action"]
    if "scroll" in t:
        t = correct_type(t)
    gt = [t, row["gt_bbox"] if "gt_bbox" in row else row["gt_max_bbox"]]
    if improved:
        gt = [gt]
    for c in row.get("candidate_actions") or []:
        gt.append([c.get("action_type", ""), c.get("action_bounds", [])])
    return gt


def answer_string(pred):
    ptype, pbox = to_curated(pred)
    d = {"action_type": ptype}
    if pbox is not None:
        d["bbox_2d"] = pbox
    return repr(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="directory holding the authors' utils.py")
    ap.add_argument("--curated", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--improved", action="store_true")
    ap.add_argument("--revised")
    a = ap.parse_args()
    sys.path.insert(0, a.official)
    import utils as official  # the authors' module

    rows = json.load(open(a.curated))
    preds, revised = load_preds(a.pred), load_preds(a.revised)
    ours, per = score(rows, preds, a.improved, revised)
    n = mism = sr = 0
    examples = []
    for r, mine in zip(rows, per):
        k = key(r)
        p = (revised or {}).get(k) if (a.improved and "revised_task" in r) else preds.get(k)
        if p is None:
            continue
        gt = official_gt(r, a.improved)
        with contextlib.redirect_stdout(io.StringIO()):   # their functions print every row
            flags = (official.calculate_multi_android(gt, answer_string(p), 1080, 2400) if a.improved
                     else official.calculate_single_android(gt, answer_string(p), 1080, 2400))
        n += 1
        sr += bool(flags[2])
        if bool(flags[2]) != bool(mine["ok"]):
            mism += 1
            if len(examples) < 10:
                examples.append({"key": k, "gt": gt, "answer": answer_string(p), "official": flags, "ours": mine})
    print(json.dumps({"rows": n, "official_sr": sr, "official_sr_pct": round(100 * sr / n, 2),
                      "ours_sr": ours["sr"], "ours_sr_pct": ours["sr_pct"], "mismatches": mism,
                      "examples": examples}, indent=1, default=str))


if __name__ == "__main__":
    main()
