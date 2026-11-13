#!/usr/bin/env python3
"""Evaluate the `is_done` noul readout on held-out episodes.

Why this exists: AndroidWorld task success at n=116 has an MDE of ~14 points
(docs/research/A5-parts/Q4-power-analysis.md), so it cannot certify a change. `is_done` is scored over
tens of thousands of ROWS, so it can.

CRITICAL, and the reason for `--exclude-terminate`: `status` rows and `is_done=yes` rows are the SAME
13,740 rows. A model that learns "if the action is terminate then done" gets is_done right for free. The
honest number is the AUROC **conditioned on action != terminate**, which is reported as `auroc_nonterm`.

Run:
  python3 benchmarks/train/eval_isdone.py --ckpt <dir> --data <compact dir> --out <json>
"""
import argparse
import json
import os
import sys


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney), ties handled at 0.5. No sklearn."""
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc needs at least one positive and one negative")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def risk_coverage(scores, labels, coverage):
    """Error rate among the top-`coverage` share by score (higher score = more confident = done)."""
    n = len(scores)
    if n == 0:
        raise ValueError("empty input")
    k = max(1, min(n, int(-(-coverage * n // 1))))   # ceil
    order = sorted(range(n), key=lambda i: -scores[i])[:k]
    return sum(1 for i in order if labels[i] == 0) / k


def _run_cli(args):
    """Serve the validation rows through the checkpoint's own `system_one_batch` — the same forward
    pass the trainer and the deployed API use — so the evaluated function is the trained one."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_d1 as T                     # noqa: E402
    from ac_common import build_questions, build_state  # noqa: E402

    rows = []
    p = os.path.join(args.data, "compact_validation.jsonl")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    with open(p) as fh:
        for line in fh:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    model, _eng = T.load_model(args.ckpt)
    model.engine.calibration = None          # raw readouts, as trained

    scores, labels, scores_nt, labels_nt = [], [], [], []
    for b0 in range(0, len(rows), args.batch):
        chunk = rows[b0:b0 + args.batch]
        reqs = []
        for rec in chunk:
            state = build_state(rec["goal"], "V", None, None, rec.get("history") or [])
            reqs.append((state, build_questions("V"), [T.load_image(f"{args.data}/{rec['img']}")]))
        res = model.system_one_batch(reqs)
        for rec, rs in zip(chunk, res):
            ans = rs["answers"]
            if "is_done" not in ans:
                continue
            d = ans["is_done"]
            p_yes = float(d.get("noul") if isinstance(d, dict) else d)
            y = 1 if (rec.get("step") == rec.get("n_steps")) else 0
            scores.append(p_yes)
            labels.append(y)
            if (rec.get("action") or {}).get("action_type") != "status":
                scores_nt.append(p_yes)
                labels_nt.append(y)

    if not scores:
        raise SystemExit("no is_done readouts produced — is the checkpoint trained with the is_done slot?")
    res = {
        "n": len(labels),
        "n_pos": sum(labels),
        "auroc": auroc(scores, labels) if 0 < sum(labels) < len(labels) else None,
        "coverage": args.coverage,
        "risk": risk_coverage(scores, labels, args.coverage),
        "n_nonterm": len(labels_nt),
        "n_pos_nonterm": sum(labels_nt),
        "auroc_nonterm": (auroc(scores_nt, labels_nt)
                          if 0 < sum(labels_nt) < len(labels_nt) else None),
        "note": "auroc_nonterm is the honest number: status rows and is_done=yes are the same rows.",
    }
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    _run_cli(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
