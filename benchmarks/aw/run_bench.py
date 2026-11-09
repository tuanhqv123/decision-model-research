#!/usr/bin/env python3
"""Run an AndroidWorld benchmark from a config, or print the plan without touching a host.

Two entry points matter:

* ``--dry-run`` prints the episode plan (protocol, pairing keys, budgets) and exits. It runs
  anywhere, including a laptop with no emulator, so a protocol can be reviewed before it is
  allowed to spend an hour.
* a real run builds the in-process AndroidWorld environment, the d1 model client, the
  recorder and the runner, then reports through ``awenv.report`` with every denominator.

Design rules carried over from the plan: the environment is in-process (the HTTP server
cannot return an a11y tree), the model is remote, ``app_name`` fallback is off unless the
run declares itself diagnostic, and the protocol string is in every artefact.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from awenv.baselines import build_policy  # noqa: E402
from awenv.report import compare_nightly, format_markdown  # noqa: E402
from awenv.runner import EpisodeSpec, Runner, new_run_id, plan_episodes  # noqa: E402

REQUIRED = ("protocol", "agent", "model", "environment", "suite")


class ConfigError(RuntimeError):
    """A config that would produce an uninterpretable run."""


@dataclass
class Plan:
    protocol: str
    agent: str
    checkpoint: Optional[str]
    specs: List[EpisodeSpec] = field(default_factory=list)
    step_budget: int = 20
    wall_clock_budget: Optional[float] = None
    allow_app_fallback: bool = False
    diagnostic: bool = False
    stuck_k: int = 3
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        return len(self.specs)


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    if not os.path.exists(path):
        raise ConfigError("config not found: %s" % path)
    with open(path, "r", encoding="utf-8") as fp:
        return dict(yaml.safe_load(fp) or {})


def plan_from_config(cfg: Dict[str, Any]) -> Plan:
    """Validate a config and turn it into episodes. No device, no model, no side effects."""
    for key in REQUIRED:
        if key not in cfg:
            raise ConfigError("config is missing %r" % key)

    suite = cfg["suite"]
    policy = cfg.get("policy") or {}
    diagnostic = bool(suite.get("diagnostic"))
    allow_fallback = bool(policy.get("allow_app_fallback"))
    if allow_fallback and not diagnostic:
        raise ConfigError(
            "allow_app_fallback=true inflates SR; set suite.diagnostic=true to declare the run "
            "as diagnostic, or turn the fallback off for a headline run"
        )

    tasks = suite.get("tasks")
    if tasks == "all":
        tasks = _all_tasks()
    elif not isinstance(tasks, list) or not tasks:
        raise ConfigError("suite.tasks must be a non-empty list or 'all'")

    seed = int(suite.get("seed", 30))
    agent = str(cfg["agent"])
    specs = plan_episodes([(name, 0) for name in tasks], seed=seed, agent=agent)
    for spec, name in zip(specs, tasks):
        spec.checkpoint = cfg.get("checkpoint")

    loop = policy.get("loop_detection") or {}
    return Plan(
        protocol=str(cfg["protocol"]),
        agent=agent,
        checkpoint=cfg.get("checkpoint"),
        specs=specs,
        step_budget=int(suite.get("step_budget", 20)),
        wall_clock_budget=_optional_float((cfg.get("episode") or {}).get("wall_clock_budget_s")),
        allow_app_fallback=allow_fallback,
        diagnostic=diagnostic,
        stuck_k=int(loop.get("repeat", 3)),
        config=cfg,
    )


def _optional_float(value):
    return None if value is None else float(value)


def validate_task_names(names: List[str], known: List[str]) -> None:
    """Fail before touching the device: a typo'd task name otherwise costs a whole run."""
    missing = [n for n in names if n not in known]
    if missing:
        raise ConfigError("unknown task name(s): %s (the suite has %d tasks)"
                          % (", ".join(missing[:5]), len(known)))


def _all_tasks() -> List[str]:
    try:
        from android_world import suite_assets  # noqa: F401
    except Exception as exc:
        raise ConfigError(
            "suite.tasks: all requires android_world to enumerate the suite (import failed: %s)" % exc
        )
    from android_world import registry

    reg = registry.TaskRegistry()
    return sorted(t.name for t in reg.get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY).values())


def describe(plan: Plan) -> str:
    lines = [
        "protocol   : %s" % plan.protocol,
        "agent      : %s (checkpoint %s)" % (plan.agent, plan.checkpoint or "n/a"),
        "tasks      : %d" % plan.n_tasks,
        "step budget: %d" % plan.step_budget,
        "wall clock : %s" % ("%ss" % plan.wall_clock_budget if plan.wall_clock_budget else "none"),
        "fallback   : %s" % ("ON (diagnostic run)" if plan.allow_app_fallback else "off"),
        "loop break : after %d identical actions" % plan.stuck_k,
        "",
        "episodes:",
    ]
    for spec in plan.specs[:12]:
        lines.append("  %-46s pairing=%s" % (spec.episode_id, spec.pairing_key))
    if plan.n_tasks > 12:
        lines.append("  ... %d more" % (plan.n_tasks - 12))
    return "\n".join(lines)


# --------------------------------------------------------------------- real run


def _build_env(cfg: Dict[str, Any]):
    """AndroidWorld's in-process environment, through our bridge. Host-only import."""
    from awenv.awenv_bridge import build

    env_cfg = cfg["environment"]
    # emulator_setup installs the task APKs and writes the user-data fixtures. It is slow
    # (~5-8 min) and idempotent, so it defaults ON: a fresh container otherwise dies on the
    # first task that needs an app (AUDIORECORDER_DATA in AudioRecorderRecordAudio). A config
    # may set environment.emulator_setup=false to skip it on a pod that was already set up.
    return build(
        console_port=int(env_cfg.get("console_port", 5554)),
        adb_path=env_cfg.get("adb_path", "adb"),
        emulator_setup=bool(env_cfg.get("emulator_setup", True)),
    )


def _build_task_registry(cfg: Dict[str, Any]):
    from android_world import registry

    return registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)


def _load_replay(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """episode_id -> instance params, from a previous run's report."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        report = json.load(fp)
    return {e["episode_id"]: e["instance_params"] for e in report.get("episodes", []) if e.get("instance_params")}


def run(plan: Plan, run_root: str, limit: Optional[int] = None,
        replay_params: Optional[str] = None) -> Dict[str, Any]:
    from awenv.agent import AgentCore
    from awenv.model_client import ModelClient
    from awenv.recorder import Recorder

    cfg = plan.config
    model_cfg = cfg["model"]
    client = ModelClient(base_url=model_cfg.get("url"), token=model_cfg.get("token"))
    registry = _build_task_registry(cfg)
    validate_task_names([s.task_name for s in plan.specs], list(registry))
    env = _build_env(cfg)

    recorder = Recorder(run_root, new_run_id(plan.agent), config=cfg)
    specs = plan.specs[:limit] if limit else plan.specs

    replay = _load_replay(replay_params)
    task_cache: Dict[str, Any] = {}

    def task_factory(spec: EpisodeSpec):
        # The registry maps a task name to its class; construction is ``cls(params)``.
        # ``generate_random_params`` takes no seed in this version, so determinism across runs
        # comes from replaying the params recorded in run 1 — not from a seed that does not exist.
        task_class = registry[spec.task_name]
        params = (replay.get(spec.episode_id) or task_class.generate_random_params())
        task = task_class(params)
        task.aw_params = params
        task_cache[spec.episode_id] = params
        return task

    def agent_factory(environment):
        name = plan.agent
        if name in ("wait", "random"):
            model = build_policy(name, seed=specs[0].seed)
        else:
            model = client
        return AgentCore(
            model,
            environment,
            allow_app_fallback=plan.allow_app_fallback,
            stuck_k=plan.stuck_k,
            track=str(model_cfg.get("track", "V")),
            return_readouts=bool(model_cfg.get("return_readouts")),
        )

    runner = Runner(
        recorder=recorder,
        env_factory=lambda: env,
        agent_factory=agent_factory,
        task_factory=task_factory,
        protocol=plan.protocol,
        checkpoint=plan.checkpoint,
    )
    out = runner.run(
        specs,
        step_budget=plan.step_budget,
        wall_clock_budget=plan.wall_clock_budget,
        goal_of=lambda spec, task: getattr(task, "goal", ""),
    )
    report = _report(recorder)
    report["run"] = out
    # A run that stopped early must say so in the report itself. On 2026-11-08 a suite declared 116
    # tasks printed "SR (all) 10.0% (1/10)" and nothing else, which reads as a finished benchmark.
    report["completeness"] = {
        "planned": out.get("planned", len(specs)),
        "completed": len(recorder.episodes()),
        "complete": len(recorder.episodes()) >= out.get("planned", len(specs)),
    }
    return report


def _report(recorder) -> Dict[str, Any]:
    """Aggregates plus the per-episode success vector.

    The vector is what makes Gate B checkable: two runs at the same seed must produce the same
    one, and a diff names the episode that moved instead of just changing a total.
    """
    from awenv.report import build_report

    episodes = recorder.episodes()
    report = build_report(episodes, recorder.steps())
    report["episodes"] = [
        {
            "episode_id": e.get("episode_id"),
            "task_name": e.get("task_name"),
            "seed": e.get("seed"),
            "success": bool(e.get("success")),
            "instance_params": e.get("instance_params"),
            "n_steps": e.get("n_steps"),
            "env_crash": bool(e.get("env_crash")),
            "timeout": bool(e.get("timeout")),
        }
        for e in sorted(episodes, key=lambda r: str(r.get("episode_id")))
    ]
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AndroidWorld benchmark runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay-params", default=None,
                    help="JSON report from a previous run; reuse its task params so episodes are paired")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)

    try:
        plan = plan_from_config(load_config(args.config))
    except ConfigError as exc:
        print("config error: %s" % exc, file=sys.stderr)
        return 2

    if args.dry_run:
        print(describe(plan))
        return 0

    if args.run_id:
        run_root = os.path.join(args.run_root, args.run_id)
    else:
        run_root = args.run_root
    report = run(plan, run_root, limit=args.limit, replay_params=args.replay_params)

    baseline_path = ((plan.config.get("regression") or {}).get("baseline"))
    alerts = []
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as fp:
            alerts = compare_nightly(report, json.load(fp))

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_markdown(report, title="AndroidWorld %s" % plan.agent))
        for alert in alerts:
            print("ALERT: %s" % alert)

    return 1 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
