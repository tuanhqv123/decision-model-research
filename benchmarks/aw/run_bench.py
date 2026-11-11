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
import hashlib
import json
import os
import random
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


TRAINING_IMG_WIDTH = 720  # every checkpoint since 2026-11-11 is trained on 720-px screenshots (stored AC width)


def model_format_error(health: Dict[str, Any], history_format: str, img_width: int = TRAINING_IMG_WIDTH):
    """Why the served checkpoint cannot be run with this harness config, or None.

    The model's `/health` names its contract; a contract with `history: v3` reads v3 history lines (element + effect)
    and every other contract reads v1 lines. The harness writes whatever `model.history_format` says, and the server
    resizes screenshots to its `D1_IMG_W`. A disagreement on either would run all 116 tasks on inputs the checkpoint
    never saw, and still print a number (2026-11-11)."""
    from awenv import agent as _agent  # noqa: F401  (puts train/ on sys.path)
    from ac_common import CONTRACTS
    name = health.get("contract")
    if name not in CONTRACTS:
        return "served checkpoint reports contract %r, which this harness does not know" % (name,)
    need = CONTRACTS[name].get("history", "v1")
    if need != history_format:
        return ("served checkpoint speaks contract %r, which reads %s history lines, but model.history_format is %r"
                % (name, need, history_format))
    served = health.get("img_width")
    if served != img_width:
        return "served image width is %r px but the checkpoint was trained at %d px (set D1_IMG_W)" % (served, img_width)
    return None


@dataclass
class Plan:
    protocol: str
    agent: str
    checkpoint: Optional[str]
    specs: List[EpisodeSpec] = field(default_factory=list)
    step_budget: int = 20
    # Per-task budget, keyed by task name. Upstream allocates
    # int(10 * task.complexity) per task (suite_utils.py:_allocate_step_budget),
    # so a single flat number is a protocol deviation. When `step_budgets` is
    # given it wins over `step_budget`; dump_step_budgets.py produces the table.
    step_budgets: Dict[str, int] = field(default_factory=dict)
    wall_clock_budget: Optional[float] = None
    allow_app_fallback: bool = False
    diagnostic: bool = False
    stuck_k: int = 3
    loop_break: bool = False
    # Q2-ii verified-failure masking (awenv/masking.py). A policy, so off unless configured.
    mask: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    def budget_for(self, task_name: str) -> int:
        """Upstream per-task budget, falling back to the flat value."""
        return int(self.step_budgets.get(task_name, self.step_budget))

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
        tasks = _all_tasks(suite.get("step_budgets_file"))
    elif not isinstance(tasks, list) or not tasks:
        raise ConfigError("suite.tasks must be a non-empty list or 'all'")

    seed = int(suite.get("seed", 30))
    agent = str(cfg["agent"])
    specs = plan_episodes([(name, 0) for name in tasks], seed=seed, agent=agent)
    for spec, name in zip(specs, tasks):
        spec.checkpoint = cfg.get("checkpoint")

    loop = policy.get("loop_detection") or {}
    # Per-task budgets. `step_budgets_file` is a JSON map produced inside the
    # android_world container by dump_step_budgets.py; when absent we fall back to
    # the flat `step_budget`, and that fallback is recorded as a deviation.
    budgets: Dict[str, int] = {}
    budgets_path = suite.get("step_budgets_file")
    if budgets_path:
        import json as _json
        with open(budgets_path) as fh:
            raw = _json.load(fh)
        budgets = {k: int(v["max_steps"]) for k, v in raw.items() if "max_steps" in v}
    return Plan(
        protocol=str(cfg["protocol"]),
        agent=agent,
        checkpoint=cfg.get("checkpoint"),
        specs=specs,
        step_budget=int(suite.get("step_budget", 20)),
        step_budgets=budgets,
        wall_clock_budget=_optional_float((cfg.get("episode") or {}).get("wall_clock_budget_s")),
        allow_app_fallback=allow_fallback,
        diagnostic=diagnostic,
        stuck_k=int(loop.get("repeat", 3)),
        loop_break=bool(loop.get("break", False)),
        mask=bool(loop.get("mask", False)),
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


def _all_tasks(step_budgets_file: Optional[str] = None) -> List[str]:
    """Every task in the android_world family.

    Preferred source is the budget table produced by `dump_step_budgets.py` INSIDE the
    container: it already enumerates exactly the 116 tasks with their complexity, so reading it
    needs no import. The registry is only a fallback, because importing it fails in this image —
    `from android_world import suite_assets` raises `ImportError` against the installed
    `/usr/local/lib/python3.11/site-packages/android_world`, which is a different copy from the
    source tree at `/app` that the server actually runs. Enumerating the suite must not depend
    on which of the two copies Python happens to resolve.
    """
    if step_budgets_file:
        try:
            import json as _json

            with open(step_budgets_file) as fh:
                names = sorted(_json.load(fh))
            if names:
                return names
        except Exception:  # noqa: BLE001 - fall through to the registry
            pass
    try:
        from android_world import registry
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            "suite.tasks: all needs either suite.step_budgets_file (preferred) or an importable "
            "android_world registry; import failed: %s" % exc
        )
    reg = registry.TaskRegistry()
    return sorted(t.name for t in reg.get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY).values())


def describe(plan: Plan) -> str:
    lines = [
        "protocol   : %s" % plan.protocol,
        "agent      : %s (checkpoint %s)" % (plan.agent, plan.checkpoint or "n/a"),
        "tasks      : %d" % plan.n_tasks,
        "step budget: %s" % (
            "per-task (android_world int(10*complexity))" if plan.step_budgets
            else "%d (FLAT — protocol deviation)" % plan.step_budget
        ),
        "wall clock : %s" % ("%ss" % plan.wall_clock_budget if plan.wall_clock_budget else "none"),
        "fallback   : %s" % ("ON (diagnostic run)" if plan.allow_app_fallback else "off"),
        "loop       : detect after %d identical actions; override %s; mask %s" % (
            plan.stuck_k, "ON (navigate_back)" if plan.loop_break else "off",
            "ON (verified no-effect pairs -> model's next-best)" if plan.mask else "off"),
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


def instance_seed(seed: int, name: str, i: int) -> int:
    """Upstream's per-instance seed (`suite_utils.py:149-153`)."""
    return int(hashlib.sha256(f"{seed}_{name}_{i}".encode()).hexdigest(), 16) % (2 ** 32)


def instantiate_task(task_class: Any, name: str, seed: int, i: int = 0,
                     replay: Optional[Dict[str, Any]] = None) -> Any:
    """Build the task exactly as `suite_utils._instantiate_task` does; replayed params are used as-is."""
    if replay:
        params = dict(replay)
    else:
        s = instance_seed(seed, name, i)
        random.seed(s)
        params = task_class.generate_random_params()
        params["seed"] = s
    task = task_class(params)
    task.aw_params = params
    return task


def _load_replay(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """episode_id -> instance params, from a previous run's report."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        report = json.load(fp)
    return {e["episode_id"]: e["instance_params"] for e in report.get("episodes", []) if e.get("instance_params")}


def run(plan: Plan, run_root: str, limit: Optional[int] = None,
        replay_params: Optional[str] = None, max_new: Optional[int] = None) -> Dict[str, Any]:
    from awenv.agent import AgentCore
    from awenv.model_client import ModelClient
    from awenv.recorder import Recorder

    cfg = plan.config
    model_cfg = cfg["model"]
    client = ModelClient(base_url=model_cfg.get("url"), token=model_cfg.get("token"))
    if plan.agent not in ("wait", "random", "status_now"):
        err = model_format_error(client.health(), str(model_cfg.get("history_format", "v1")),
                                 int(model_cfg.get("img_width", TRAINING_IMG_WIDTH)))
        if err:
            raise ConfigError(err)
    registry = _build_task_registry(cfg)
    validate_task_names([s.task_name for s in plan.specs], list(registry))
    env = _build_env(cfg)

    recorder = Recorder(run_root, new_run_id(plan.agent), config=cfg)
    specs = plan.specs[:limit] if limit else plan.specs

    replay = _load_replay(replay_params)
    task_cache: Dict[str, Any] = {}

    def task_factory(spec: EpisodeSpec):
        # The official seed-30 instance (`suite_utils.py:97-103, 149-153`), unless run 1's params
        # are being replayed. The old comment here said "generate_random_params takes no seed";
        # it uses the global `random` module and the CALLER seeds it — which we never did, so
        # every run before 2026-11-10 drew a fresh random instance set.
        task = instantiate_task(registry[spec.task_name], spec.task_name, spec.seed, spec.task_idx,
                                replay=replay.get(spec.episode_id))
        task_cache[spec.episode_id] = task.aw_params
        return task

    def agent_factory(environment):
        name = plan.agent
        if name in ("wait", "random", "status_now"):
            model = build_policy(name, seed=specs[0].seed)
        else:
            model = client
        return AgentCore(model, environment, **agent_kwargs(plan))

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
        budget_of=plan.budget_for,
        max_new=max_new,
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


def agent_kwargs(plan: Plan) -> Dict[str, Any]:
    """Every AgentCore setting a config controls, in one place so a test can see it reach the agent."""
    model_cfg = plan.config.get("model") or {}
    return dict(
        allow_app_fallback=plan.allow_app_fallback,
        stuck_k=plan.stuck_k,
        loop_break=plan.loop_break,
        mask=plan.mask,
        track=str(model_cfg.get("track", "V")),
        return_readouts=bool(model_cfg.get("return_readouts")),
        # 0.0 = gate OFF, which is upstream's behaviour and the default. See
        # benchmarks/results/2026-11-10-episodic-s0-fails-prereg.md: `is_done` is a copy of
        # the action label on this corpus, so there is no threshold with an operating point.
        done_threshold=float(model_cfg.get("done_threshold", 0.0)),
        # contract-v3 checkpoints read v3 history entries (element + effect); every other contract reads v1 lines
        history_format=str(model_cfg.get("history_format", "v1")),
    )


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
    # Runs live OUTSIDE the code tree by default. The harness itself gets reinstalled into
    # /tmp/awh on every emulator restart, which wiped /tmp/awh/aw/runs repeatedly during
    # 2026-11-09/10 debugging — real episodes were lost while the summary still printed.
    ap.add_argument("--run-root", default=os.environ.get("AW_RUN_ROOT", "/tmp/awruns"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None)
    # Stop after N NEW episodes, leaving the rest for the next chunk. The a11y forwarder needs a
    # container restart every ~15-20 min of uptime, so a 116-task suite cannot be one process.
    ap.add_argument("--max-new", type=int, default=None)
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
        # Joining is idempotent: passing a run-root that ALREADY ends in the run id must not
        # create a nested `<root>/<id>/<id>`. That nesting happened on 2026-11-10 with
        # `--run-root /out/aw116_episodic --run-id aw116_episodic`, and it silently broke
        # resumption: the recorder opened a fresh index.db two levels down, found no completed
        # episodes, and re-ran work that was already on disk — while the summary looked normal.
        if os.path.basename(os.path.normpath(args.run_root)) == args.run_id:
            run_root = args.run_root
        else:
            run_root = os.path.join(args.run_root, args.run_id)
    else:
        run_root = args.run_root
    report = run(plan, run_root, limit=args.limit, replay_params=args.replay_params,
                 max_new=args.max_new)

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
