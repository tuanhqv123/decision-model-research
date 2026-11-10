"""Rewrite a compact corpus's `history` as the rendered GOLD ACTIONS of each row's earlier steps.

Why (decision 2026-11-10, K3): AndroidControl's `history` is the annotators' step instructions
("Open the YouTube Music app"), and AC-v2's test file feeds the same oracle text. AndroidWorld supplies
no such text: the agent's history is its own executed actions. This makes training history identical in
kind and in format to serve-time history, using `ac_common.render_history_line` — the same function the
harness agent uses (benchmarks/aw/awenv/agent.py).

A row whose episode is missing any earlier step is DROPPED and counted, never filled in.

Usage:
  python rerender_history.py --episodes acdata/compact_train.jsonl --rows acdata/compact_cur15k.jsonl \
      --out acdata/compact_cur15k_v1.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ac_common import render_history_line  # noqa: E402


def rerender(rows, episodes):
    """Return (new_rows, stats). `episodes` must contain every step of the rows' episodes."""
    by_ep = {}
    for r in episodes:
        by_ep.setdefault(r["ep"], {})[int(r["step"])] = r["action"]
    out, gap = [], 0
    for r in rows:
        steps = by_ep.get(r["ep"], {})
        prev = [steps.get(j) for j in range(int(r["step"]))]
        if any(a is None for a in prev):
            gap += 1
            continue
        new = dict(r)
        new["history"] = [render_history_line(a) for a in prev]
        new["history_source"] = "rendered_gold_actions_v1"
        out.append(new)
    return out, {"rows": len(rows), "kept": len(out), "dropped_gap": gap}


def _read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True, help="files holding every step of the episodes")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    episodes = [r for p in a.episodes for r in _read(p)]
    out, stats = rerender(_read(a.rows), episodes)
    with open(a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("RERENDER " + json.dumps({**stats, "out": a.out}), flush=True)
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
