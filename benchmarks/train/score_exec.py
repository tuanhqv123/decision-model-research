#!/usr/bin/env python3
"""Score a checkpoint in the EXECUTOR role (2026-11-13): eval_d1 predictions on a build_exec_data file
(`--instruction-per-row`). Reports, with denominators:
  action rows   step success (the eval's own `correct`), and false "done" (predicted `status`) — split into rows inside a
                local goal (more actions still needed) and rows that are the local goal's last action
  done rows     recall: `status` predicted on the screen after the local goal's last action
  p(status)     AUROC of the model's status probability, done rows vs action rows (the harness thresholds it)
Rows of goal-only episodes (no instruction) are left out.
    python3 score_exec.py PRED.jsonl EXEC.jsonl [OUT.json]
"""
import ast
import json
import sys


def p_status(row):
    probs = row.get("probs") or {}
    if isinstance(probs, str):
        probs = ast.literal_eval(probs)
    return float(probs.get("status", probs.get("terminate", 0.0)))


def auroc(pos, neg):
    """P(score of a random positive > score of a random negative), ties counted half."""
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2  # 1-based average rank of the tie block
        rank_sum += avg * sum(lab for _, lab in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def score(preds, rows):
    pred = {(str(p["episode_id"]), p["step"]): p for p in preds}
    done_keys = {(str(r["ep"]).split(":", 1)[1], r["step"]) for r in rows if r["aug"] == "span_done"}
    acc = {"inside": [0, 0, 0], "last": [0, 0, 0]}  # rows, correct, predicted status
    done = [0, 0]
    pos, neg, missing = [], [], 0
    for r in rows:
        if r["aug"] == "goal":
            continue
        p = pred.get((str(r["ep"]), r["step"]))
        if p is None:
            missing += 1
            continue
        said_done = p["pred"] in ("status", "terminate")
        if r["aug"] == "span_done":
            done[0] += 1
            done[1] += said_done
            pos.append(p_status(p))
        else:
            k = "last" if (str(r["ep"]), r["step"] + 1) in done_keys else "inside"
            acc[k][0] += 1
            acc[k][1] += bool(p["correct"])
            acc[k][2] += said_done
            neg.append(p_status(p))
    n_act = acc["inside"][0] + acc["last"][0]
    said = acc["inside"][2] + acc["last"][2] + done[1]

    def pct(a, b):
        return round(100 * a / b, 2) if b else None

    return {
        "action rows": n_act,
        "step success %": pct(acc["inside"][1] + acc["last"][1], n_act),
        "false done on action rows": "%d/%d" % (acc["inside"][2] + acc["last"][2], n_act),
        "false done inside a local goal": "%d/%d" % (acc["inside"][2], acc["inside"][0]),
        "false done on its last action": "%d/%d" % (acc["last"][2], acc["last"][0]),
        "done rows": done[0],
        "done recall %": pct(done[1], done[0]),
        "done precision %": pct(done[1], said),
        "AUROC p(status) done vs action": None if not pos else round(auroc(pos, neg), 4),
        "rows without a prediction": missing,
    }


def main(pred_path, exec_path, out=None):
    res = score([json.loads(l) for l in open(pred_path)], [json.loads(l) for l in open(exec_path)])
    print(json.dumps(res, indent=1))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
