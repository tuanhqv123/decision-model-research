#!/usr/bin/env python3
"""How far two runs of the same weights on the same task instances agree (Q10; host equivalence).

    python3 replay_agreement.py <episodes_dir_A> <episodes_dir_B> [--label-a X] [--label-b Y]

Each directory holds `<episode_id>.jsonl` files (abandoned partial files are ignored); a task is identified by its
result line's `task_name`, and the instance by the seed the harness derives from (seed, task name, index), so two
runs of one config replay the same instances.

Three levels of agreement, from coarse to fine:
  * outcome: success (official score > 0.5) and declared-done, per task;
  * trajectory: the sequence of SEMANTIC decisions — action type, the element a click landed on (its bbox, or none),
    typed text, scroll direction, app name. Raw click pixels are NOT compared: the screen differs between runs from
    the first frame (status bar), so the model's point jitters by a few pixels while choosing the same element;
  * step count alone is NOT reported as agreement: most failed episodes stop at their step budget, so equal step
    counts follow from the budget whatever the actions were (2026-11-10: 20/20 equal step counts, 14/20 equal
    trajectories).
"""
import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List, Tuple

Run = Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]]


def load_run(ep_dir: str) -> Run:
    out: Run = {}
    for path in sorted(glob.glob(os.path.join(ep_dir, "*.jsonl"))):
        if ".abandoned." in os.path.basename(path):
            continue
        with open(path, "r", encoding="utf-8") as fp:
            recs = [json.loads(line) for line in fp if line.strip()]
        res = [r for r in recs if r.get("type") == "result"]
        if not res:
            continue
        name = res[-1]["task_name"]
        if name in out:
            raise ValueError("task %s has two scored episodes in %s" % (name, ep_dir))
        out[name] = ([r for r in recs if r.get("type") == "step"], res[-1])
    return out


def semantic_action(step: Dict[str, Any]) -> tuple:
    e = step.get("action_executed") or {}
    target = step.get("click_target") or {}
    bbox = tuple(target["bbox"]) if isinstance(target, dict) and target.get("bbox") else None
    return (e.get("action_type"), bbox, e.get("text"), e.get("direction"), e.get("app_name"))


def _success(r: Dict[str, Any]) -> bool:
    return (r.get("score") or 0.0) > 0.5


def _excluded(r: Dict[str, Any]) -> bool:
    """Same rule as paired_sr: a task that raised or could not be scored is not an outcome of either policy."""
    return bool(r.get("setup_error") or r.get("unscored") or r.get("env_crash")) or r.get("score") is None


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def compare_runs(a: Run, b: Run) -> Dict[str, Any]:
    common = sorted(set(a) & set(b))
    excluded = [t for t in common if _excluded(a[t][1]) or _excluded(b[t][1])]
    tasks = [t for t in common if t not in excluded]
    flips, done_diffs, first_div = [], [], {}
    same = 0
    for t in tasks:
        (sa, ra), (sb, rb) = a[t], b[t]
        if _success(ra) != _success(rb):
            flips.append(t)
        if bool(ra.get("declared_done")) != bool(rb.get("declared_done")):
            done_diffs.append(t)
        ka, kb = [semantic_action(s) for s in sa], [semantic_action(s) for s in sb]
        if ka == kb:
            same += 1
        else:
            first_div[t] = next((i for i, (x, y) in enumerate(zip(ka, kb)) if x != y), min(len(ka), len(kb)))
    return {"n_tasks": len(tasks), "excluded": excluded, "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a)),
            "success_flips": flips, "done_diffs": done_diffs, "identical_trajectories": same,
            "first_divergence": first_div, "successes": (sum(_success(a[t][1]) for t in tasks),
                                                          sum(_success(b[t][1]) for t in tasks))}


def render(out: Dict[str, Any], la: str, lb: str) -> str:
    n = out["n_tasks"]
    pct = lambda lo_hi: "n/a" if lo_hi[0] is None else "[%.1f%%, %.1f%%]" % tuple(100 * v for v in lo_hi)  # noqa: E731
    div = n - out["identical_trajectories"]
    return "\n".join([
        "%s vs %s on %d common tasks scored in both (only in %s: %d, only in %s: %d)" % (la, lb, n, la, len(out["only_a"]), lb,
                                                                            len(out["only_b"])),
        "  excluded in either run (set aside): %d %s" % (len(out["excluded"]), out["excluded"]),
        "  successes %d vs %d" % out["successes"],
        "  success flips %d/%d, Wilson 95%% %s: %s" % (len(out["success_flips"]), n,
                                                    pct(wilson(len(out["success_flips"]), n)), out["success_flips"]),
        "  declared-done differences %d/%d: %s" % (len(out["done_diffs"]), n, out["done_diffs"]),
        "  semantically divergent trajectories %d/%d, Wilson 95%% %s; first divergence step: %s" % (
            div, n, pct(wilson(div, n)), out["first_divergence"]),
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args(argv)
    print(render(compare_runs(load_run(args.a), load_run(args.b)), args.label_a, args.label_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
