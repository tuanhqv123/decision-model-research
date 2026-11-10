#!/usr/bin/env python3
"""Per-episode failure analysis for AndroidWorld runs.

Every failed episode is classified from its own trajectory, not from the summary line. The
questions, in order, are the ones that decide what to fix:

  1. Did the episode end because the agent declared done, or because steps ran out?
     Upstream has NO time limit (`episode_runner.py` loops `for step_n in range(max_n_steps)`),
     so "timeout" is the wrong word: it means the step budget was exhausted.
  2. If it declared done, WAS the screen right? A premature `status` and a correct-but-unjudged
     finish look identical in the summary and have opposite fixes.
  3. Was the action itself mappable, executed, and did the screen change? An invalid action is a
     harness/interface defect; a no-op on a changed screen is a model defect.
  4. For spatial actions: did the point land inside the element the model named? This separates
     "picked the wrong widget" from "cannot place a point on a widget".
  5. Did the model loop on the same action, or did it explore?

Usage:
    python3 analyze_failures.py /out            # inside the container
    python3 analyze_failures.py ~/awruns/run    # on the host
"""
import argparse
import collections
import glob
import json
import os
import sys


def load(path):
    meta, steps = None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "result":
            meta = rec
        elif "action_raw" in rec:
            steps.append(rec)
    return meta, steps


def _norm(s):
    return (s or "").strip().lower()


def classify(meta, steps):
    """Return a list of mechanism tags with supporting evidence."""
    tags = []
    n_steps = len(steps)
    declared = bool(meta.get("declared_done"))
    budget = meta.get("step_budget") or 0

    if meta.get("success"):
        return ["SUCCESS"]

    # ---- how the episode ended
    if declared and n_steps < budget:
        tags.append("DECLARED_DONE_EARLY")
    elif n_steps >= budget:
        tags.append("BUDGET_EXHAUSTED")

    # ---- action-level facts
    invalid = [s for s in steps if s.get("invalid_action")]
    if invalid:
        reasons = collections.Counter(s.get("reason") for s in invalid)
        tags.append("INVALID_ACTION x%d %s" % (len(invalid), dict(reasons)))

    still = [s for s in steps if s.get("screen_changed") is False]
    if still:
        tags.append("NO_SCREEN_CHANGE x%d" % len(still))

    # ---- repetition: the same decision on the same screen
    sigs = []
    for s in steps:
        a = s.get("action_raw") or {}
        sigs.append((a.get("action"), tuple(a.get("point") or []), a.get("text"),
                     a.get("direction"), a.get("app_name")))
    if sigs:
        counts = collections.Counter(sigs)
        top, k = counts.most_common(1)[0]
        if k >= 4:
            tags.append("REPEAT x%d (%s)" % (k, top[0]))

    # ---- spatial grounding proxy: did any point land on an element?
    # The trace records the a11y element count, not the elements themselves, so this uses the
    # model's own point vs the screenshots captured alongside; a full check needs the a11y tree.
    spatial = [s for s in steps
               if (s.get("action_raw") or {}).get("action") in ("click", "long_press", "type_text")]
    if spatial:
        tags.append("spatial_actions=%d" % len(spatial))

    # ---- did it ever type what the goal asked for?
    typed = [(s.get("action_raw") or {}).get("text") for s in steps
             if (s.get("action_raw") or {}).get("action") == "type_text"]
    if typed:
        tags.append("typed=%d %r" % (len(typed), typed[0][:26] if typed[0] else ""))

    return tags or ["UNCLASSIFIED"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    ep_dir = os.path.join(a.run_root, "episodes")
    files = sorted(glob.glob(os.path.join(ep_dir, "*.jsonl")))
    if not files:
        raise SystemExit("no episodes under %s" % ep_dir)

    buckets = collections.Counter()
    ok = 0
    print("=" * 100)
    for path in files:
        meta, steps = load(path)
        if not meta:
            continue
        name = meta.get("task_name") or os.path.basename(path)
        success = bool(meta.get("success"))
        ok += success
        tags = classify(meta, steps)
        for t in tags:
            buckets[t.split(" x")[0].split(" (")[0]] += 1

        print("%-42s %-5s steps=%-3s/%-3s %s" % (
            name[:42], "OK" if success else "FAIL",
            len(steps), meta.get("step_budget") or 0,
            " | ".join(tags)))
        if a.verbose:
            for s in steps:
                act = s.get("action_raw") or {}
                line = "    i=%-2s %-11s conf=%s" % (
                    s.get("i"), act.get("action"),
                    ("%.2f" % act["confidence"]) if isinstance(act.get("confidence"), (int, float)) else "-")
                if act.get("point"):
                    line += " pt=%s" % ([round(v) for v in act["point"]],)
                if act.get("text"):
                    line += " %r" % str(act["text"])[:24]
                if act.get("app_name"):
                    line += " app=%s" % act["app_name"]
                if act.get("direction"):
                    line += " dir=%s" % act["direction"]
                if s.get("screen_changed") is False:
                    line += " [no-change]"
                if s.get("invalid_action"):
                    line += " [INVALID:%s]" % s.get("reason")
                print(line)

    print("=" * 100)
    print("total=%d  success=%d  SR=%.1f%%" % (len(files), ok, 100.0 * ok / max(1, len(files))))
    print()
    print("MECHANISM TALLY (a task can appear in several):")
    for tag, n in buckets.most_common():
        print("  %-28s %d" % (tag, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
