"""D06 — the four matched conditions (B1/B2/B3/P1) over one locked task manifest.

The guide's Step 13-14, implemented so that the *only* difference between runs is the coordinator policy:

  * one task manifest file, hashed and stored with every run;
  * the same action model, planner, adapters, step budget and timeout for all conditions;
  * per-run records of instance id, seed, completion category, elapsed monotonic time, planner calls, action
    count, model latency and terminal reason;
  * technical failures are recorded, never dropped (guide §56).

Usage
  python3 -m awenv.coordinator_bench --manifest configs/manifest_v1.json --policies B1,B2,B3,P1 \
      --out runs/coord_v1 --dry-run
  (drop --dry-run on a host with the emulator + model service)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from .coordinator import CONTINUE, UPDATE, Coordinator, Plan


# ------------------------------------------------------------------ the manifest (the locked population)
def load_manifest(path: str) -> Dict[str, Any]:
    m = json.load(open(path, encoding="utf-8"))
    if "tasks" not in m or not m["tasks"]:
        raise ValueError("manifest needs a non-empty 'tasks' list")
    for t in m["tasks"]:
        for k in ("task_name", "seed"):
            if k not in t:
                raise ValueError("every task needs %r: %r" % (k, t))
    m["sha256"] = hashlib.sha256(json.dumps(m["tasks"], sort_keys=True).encode()).hexdigest()
    return m


# ------------------------------------------------------------------ one episode under one policy
def run_episode(env: Any, action_model: Any, coordinator: Coordinator, planner,
                task: Dict[str, Any], step_budget: int = 15) -> Dict[str, Any]:
    """Drive one task. Returns a record with every field the guide's Steps 6 and 15 require."""
    t0 = time.monotonic_ns()
    goal = getattr(task, "goal", None) or task["task_name"]
    rec: Dict[str, Any] = {
        "task_name": task["task_name"], "seed": task["seed"],
        "environment_revision": os.environ.get("AW_ENV_REV", "unknown"),
        "policy": coordinator.policy,
        "planner_calls": 0, "coordinator_decisions": 0, "actions": 0,
        "invalid_actions": 0, "model_ms": [], "recovery_events": 0,
        "terminal_reason": None, "evaluator_status": None,
    }
    try:
        env.reset()
        obs_id = "obs_0"
        rec["coordinator_decisions"] += 1
        d = coordinator.decide(goal, "", obs_id, last_action_ok=True)
        if d["choice"] == UPDATE:
            coordinator.apply_update(goal, "")
            rec["planner_calls"] += 1

        for i in range(step_budget):
            observation = getattr(env, "screen_text", "") or ""
            observation_id = "obs_%d" % (i + 1)

            # 1. coordinator decision — before any action (guide §22)
            d = coordinator.decide(goal, observation, observation_id, last_action_ok=True)
            rec["coordinator_decisions"] += 1
            if d["choice"] == UPDATE:
                if coordinator.apply_update(goal, observation) is None:
                    rec["terminal_reason"] = "planner_failed"
                    rec["recovery_events"] += 1
                    break
                rec["planner_calls"] += 1
            elif d["choice"] != CONTINUE:
                raise AssertionError("coordinator returned an illegal decision %r" % d["choice"])
            coordinator.observe(observation_id)

            # 2. action selection happens only after the plan is settled
            tm = time.monotonic()
            step = action_model.step(goal, i)
            rec["model_ms"].append((time.monotonic() - tm) * 1000.0)
            rec["actions"] += 1
            if step.get("invalid_action"):
                rec["invalid_actions"] += 1
            if step.get("done"):
                rec["terminal_reason"] = "declared_done"
                break
        else:
            rec["terminal_reason"] = "step_budget"

        rec["evaluator_status"] = bool(task.is_successful(env))
    except Exception as exc:                                  # never hide a technical failure
        rec["terminal_reason"] = "technical_error"
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["evaluator_status"] = None
    rec["elapsed_ns"] = time.monotonic_ns() - t0
    rec["elapsed_s"] = rec["elapsed_ns"] / 1e9
    ms = rec["model_ms"] or [0.0]
    rec["model_ms_p50"] = sorted(ms)[len(ms) // 2]
    rec["decisions"] = [d for d in coordinator.decisions]
    return rec


# ------------------------------------------------------------------ the comparison table (guide §57, §62)
def summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Rebuildable from raw records only: counts per category, per policy, with denominators shown."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by.setdefault(r["policy"], []).append(r)
    out = {}
    for pol, rs in by.items():
        n = len(rs)
        ok = sum(1 for r in rs if r["evaluator_status"] is True)
        bad = sum(1 for r in rs if r["evaluator_status"] is False)
        err = sum(1 for r in rs if r["evaluator_status"] is None)
        times_ok = sorted(r["elapsed_s"] for r in rs if r["evaluator_status"] is True)
        out[pol] = {
            "attempted": n, "success": ok, "judged_failure": bad, "technical_error": err,
            "completion_rate_pct": round(100 * ok / n, 2) if n else None,
            "planner_calls_mean": round(sum(r["planner_calls"] for r in rs) / n, 2) if n else None,
            "actions_mean": round(sum(r["actions"] for r in rs) / n, 2) if n else None,
            "invalid_actions_total": sum(r["invalid_actions"] for r in rs),
            "model_ms_p50": round(sorted([r["model_ms_p50"] for r in rs])[len(rs) // 2], 1) if n else None,
            "elapsed_s_p50_success": round(times_ok[len(times_ok) // 2], 2) if times_ok else None,
            "timeout_rate_pct": round(100 * sum(1 for r in rs if r["terminal_reason"] == "step_budget") / n, 2) if n else None,
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--policies", default="B1,B2,B3,P1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-budget", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the manifest and policy wiring without an emulator or model")
    a = ap.parse_args(argv)

    m = load_manifest(a.manifest)
    print("[manifest] %d tasks, sha256 %s" % (len(m["tasks"]), m["sha256"][:16]))
    policies = [p.strip() for p in a.policies.split(",") if p.strip()]
    for p in policies:
        if p not in ("B1", "B2", "B3", "P1"):
            raise SystemExit("unknown policy %r" % p)

    os.makedirs(a.out, exist_ok=True)
    json.dump({"manifest_sha256": m["sha256"], "tasks": len(m["tasks"]), "policies": policies,
               "step_budget": a.step_budget}, open(os.path.join(a.out, "run_config.json"), "w"), indent=1)

    if a.dry_run:
        for p in policies:
            c = Coordinator(policy=p, planner=lambda g, o: ["step one", "step two"])
            d = c.decide(m["tasks"][0]["task_name"], "", "obs_0", last_action_ok=True)
            print("  %s -> %s (%s)" % (p, d["choice"], d["reason"]))
        print("[dry-run] wiring OK; no environment used")
        return 0

    from .awenv_bridge import make_environment, make_action_model   # host-only imports
    env = make_environment()
    model = make_action_model()
    planner = lambda goal, obs: [goal]           # single-subgoal plan: the honest minimum
    records = []
    for policy in policies:
        for task in m["tasks"]:
            coord = Coordinator(policy=policy, planner=planner)
            r = run_episode(env, model, coord, planner, task, a.step_budget)
            records.append(r)
            with open(os.path.join(a.out, "episodes.jsonl"), "a", encoding="utf-8") as fp:
                fp.write(json.dumps(r) + "\n")
    table = summarise(records)
    json.dump(table, open(os.path.join(a.out, "table.json"), "w"), indent=1)
    print(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
