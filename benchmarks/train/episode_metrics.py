#!/usr/bin/env python3
"""Whole-task view of a per-step validation file written by `eval_d1.py` (one JSON row per step).

Offline validation is teacher-forced: every step sees the demonstrator's correct history. This adds the task view on top
of step accuracy: a goal is solved only if EVERY one of its steps is right, and the first wrong step of an unsolved goal
is placed at the start (first step), the end (last step) or the middle. It cannot measure recovery from the model's own
mistakes; only the online AndroidWorld run does.

    python episode_metrics.py eval-<tag>-VA-val.jsonl [more.jsonl ...]
"""
import collections
import json
import sys


def episode_metrics(rows):
    by_ep = collections.defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r)
    first_err = collections.Counter()
    solved = 0
    for steps in by_ep.values():
        steps.sort(key=lambda r: r["step"])
        wrong = [i for i, r in enumerate(steps) if not r["correct"]]
        if not wrong:
            solved += 1
            continue
        i = wrong[0]
        first_err["start" if i == 0 else "end" if i == len(steps) - 1 else "middle"] += 1
    premature = sum(1 for r in rows if r["pred"] in ("status", "terminate") and (r["gold"] or {}).get("action_type") != "status")
    return {"steps": len(rows), "step_correct": sum(bool(r["correct"]) for r in rows), "goals": len(by_ep),
            "goals_all_correct": solved, "first_error_at": dict(first_err), "premature_status": premature}


def main(paths):
    for p in paths:
        m = episode_metrics([json.loads(line) for line in open(p) if line.strip()])
        print(f"{p}\n  steps right {m['step_correct']}/{m['steps']} ({100 * m['step_correct'] / m['steps']:.1f}%)  "
              f"goals with every step right {m['goals_all_correct']}/{m['goals']}  first error at {m['first_error_at']}  "
              f"premature status {m['premature_status']}")


if __name__ == "__main__":
    main(sys.argv[1:])
