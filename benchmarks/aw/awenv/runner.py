"""Tier B: turn a task suite into episodes that can be resumed, paired and audited.

Four properties are load-bearing, and each exists because its absence corrupted an earlier
number: an episode is skipped only when a ``result`` line proves it finished; the pairing
key is written into the header rather than asserted in a write-up; the step budget ends an
episode as a timeout, never as a success; and an exception is classified as an action error
or an environment crash by asking the environment, not by guessing from the traceback.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_PROTOCOL = "aw-main-v1"


class InfraAbort(RuntimeError):
    """The device could not be reset: stop the chunk WITHOUT recording a result for this episode.

    A dead accessibility forwarder is not a task outcome. Recording it as a setup error excluded nine
    consecutive B0 tasks from the official denominator and, because a result line existed, every later
    resume skipped them (2026-11-10). With no result line the next round (container restart +
    preflight) runs the same task again.
    """


@dataclass
class EpisodeSpec:
    task_name: str
    task_idx: int
    seed: int
    agent: str
    episode_id: str
    instance_params: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[str] = None

    @property
    def pairing_key(self) -> str:
        return "%s|%d|%d" % (self.task_name, self.task_idx, self.seed)


def _slug(name: str) -> str:
    """The episode-id token: the whole task name, lower-cased, never truncated.

    Until 2026-11-10 this cut names at 24 characters and called the result "collision-free". It
    was not: `SimpleCalendarAddOneEvent` and `SimpleCalendarAddOneEventTomorrow` both became
    `simplecalendaraddoneeven`, so the second task shared the first one's id and resumption
    skipped it as already finished. Six such groups hid 12 of the 116 tasks from every 116-task
    run. Names of 24 characters or fewer get the same token as before, so their existing results
    still resume; a longer name now gets its full token. `plan_episodes` refuses any duplicate.
    """
    keep = [c if (c.isalnum() or c == "-") else "-" for c in str(name)]
    return "".join(keep).strip("-").lower() or "task"


def plan_episodes(
    tasks: Sequence[Any],
    seed: int = 30,
    agent: str = "d1_sft",
    prefix: str = "p1",
    instance_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[EpisodeSpec]:
    """Deterministic episode plan. ``tasks`` is a sequence of (task_name, task_idx)."""
    specs = []
    for item in tasks:
        name, idx = item if isinstance(item, (tuple, list)) else (item, 0)
        key = "%s|%d" % (name, idx)
        specs.append(
            EpisodeSpec(
                task_name=name,
                task_idx=int(idx),
                seed=int(seed),
                agent=agent,
                episode_id="%s_t%04d_%s_s%d_%s" % (prefix, int(idx), _slug(name), int(seed), agent),
                instance_params=(instance_params or {}).get(key, {"task_idx": int(idx)}),
            )
        )
    # Resumption treats an id with a result as finished, so two episodes sharing an id means the
    # second is silently never run. Fail here, where it is visible, instead.
    seen: Dict[str, str] = {}
    for spec in specs:
        if spec.episode_id in seen:
            raise ValueError("duplicate episode id %r for %s|%d (already used by %s)"
                             % (spec.episode_id, spec.task_name, spec.task_idx, seen[spec.episode_id]))
        seen[spec.episode_id] = "%s|%d" % (spec.task_name, spec.task_idx)
    return specs


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 1)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 1)


def _p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1)


class Runner:
    """Runs episodes one at a time. Never concurrent inside a run."""

    def __init__(
        self,
        recorder: Any,
        env_factory: Callable[[], Any],
        agent_factory: Callable[[Any], Any],
        task_factory: Callable[[EpisodeSpec], Any],
        protocol: str = DEFAULT_PROTOCOL,
        checkpoint: Optional[str] = None,
        save_screenshots: bool = True,
    ):
        self.recorder = recorder
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.task_factory = task_factory
        self.protocol = protocol
        self.checkpoint = checkpoint
        self.save_screenshots = save_screenshots

    def run(
        self,
        specs: Sequence[EpisodeSpec],
        step_budget: int = 20,
        wall_clock_budget: Optional[float] = None,
        goal_of: Optional[Callable[[EpisodeSpec, Any], str]] = None,
        budget_of: Optional[Callable[[str], int]] = None,
        max_new: Optional[int] = None,
    ) -> Dict[str, Any]:
        attempted = skipped = 0
        n_action_errors = 0
        started = time.time()

        for spec in specs:
            if self.recorder.has_result(spec.episode_id):
                skipped += 1
                continue
            # `max_new` bounds a CHUNK, not the suite. The a11y forwarder degrades after tens of
            # minutes of emulator uptime and the only reliable fix is recreating the container, so
            # a 116-task suite is run as several chunks with a restart between them. Each chunk
            # stops after `max_new` new episodes and the next one resumes from what is on disk.
            # `--limit` cannot do this: it truncates the SPEC list before skipping, so a chunk
            # whose first N specs are already done would attempt nothing.
            if max_new is not None and attempted >= max_new:
                break
            attempted += 1
            # Upstream budgets per task (int(10*complexity)). The table (`budget_of`) is only the
            # fallback: `_run_episode` lets the task instance's own complexity win.
            budget = budget_of(spec.task_name) if budget_of else step_budget
            n_action_errors += self._run_episode(spec, budget, goal_of, wall_clock_budget)

        return {
            "attempted": attempted,
            "skipped": skipped,
            "planned": len(specs),
            "completed": attempted,
            "n_action_errors": n_action_errors,
            "elapsed_s": round(time.time() - started, 1),
        }

    # ------------------------------------------------------------------ episode
    def _run_episode(
        self,
        spec: EpisodeSpec,
        step_budget: int,
        goal_of: Optional[Callable[[EpisodeSpec, Any], str]],
        wall_clock_budget: Optional[float] = None,
    ) -> int:
        env = self.env_factory()
        task = self.task_factory(spec)
        agent = self.agent_factory(env)
        # The instance decides its own budget (`suite_utils.py:459`): information-retrieval tasks set
        # `self.complexity` from their proto in `__init__` (`information_retrieval.py:70`), which a
        # class-level table cannot see.
        step_budget = resolve_budget(task, step_budget)
        action_errors = 0

        header = {
            "episode_id": spec.episode_id,
            "task_family": "android_world",
            "task_name": spec.task_name,
            "task_idx": spec.task_idx,
            "seed": spec.seed,
            "instance_params": _jsonable(getattr(task, "aw_params", None) or spec.instance_params),
            "pairing_key": spec.pairing_key,
            "agent": spec.agent,
            "checkpoint": spec.checkpoint or self.checkpoint,
            "protocol": self.protocol,
            "step_budget": step_budget,
            "harness_commit": _harness_id(),
            "coverage": _coverage_of(task),
        }
        self.recorder.start_episode(header)

        env_crash = False
        wall_clock_exceeded = False
        steps: List[Dict[str, Any]] = []
        i = 0
        # ``wall_clock_budget`` is PER EPISODE. Until 2026-11-08 it was checked once per episode in
        # the outer loop, so a 116-task suite stopped after the first ten episodes and printed a
        # confident "1/10" with no hint that 106 tasks were never attempted. A per-episode cap can
        # only bind inside the episode, so it is enforced here and recorded on the episode.
        episode_deadline = (time.time() + wall_clock_budget) if wall_clock_budget else None
        setup_error = None
        try:
            # ORDER IS UPSTREAM'S: initialise the task, THEN go home. `suite_utils.py:246-247` calls
            # `task.initialize_task(env)` and only then `run_episode`, whose first act is
            # `agent.reset(task.start_on_home_screen)` -> `env.reset(go_home=True)`
            # (`episode_runner.py:79`; `start_on_home_screen = True` for every non-MiniWoB task,
            # `task_eval.py:40`). Until 2026-11-10 we reset first, so any app that initialisation
            # left open became the agent's first screen.
            try:
                _init_task(task, env)
            except Exception as exc:  # noqa: BLE001
                # A task that cannot prepare its own device state must not end the suite. On
                # 2026-11-08 `ExpenseDeleteDuplicates` raised `no such table: map_markers` inside
                # initialize_task and killed a 116-task run at episode 42, losing 74 tasks of
                # coverage to one app's missing schema. Record it as a setup failure and continue.
                setup_error = "%s: %s" % (type(exc).__name__, exc)
                env_crash = not _alive(env)
            # ``env.reset()`` raises when the a11y dump fails on a live device — the same transient
            # that AndroidWorld retries internally. On 2026-11-08 it escaped and killed a 116-task run
            # at episode 93 while `adb devices` showed `device` and `/health` said success. Retry a
            # couple of times, and if it still fails treat it as this episode's setup failure rather
            # than the suite's.
            for _attempt in range(0 if setup_error else 3):
                try:
                    env.reset()
                    setup_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    setup_error = "reset: %s: %s" % (type(exc).__name__, exc)
                    if not _alive(env):
                        env_crash = True
                        break
                    time.sleep(2.0 * (_attempt + 1))
            if setup_error and setup_error.startswith("reset:"):
                raise InfraAbort("%s (episode %s left unscored for a re-run)" % (setup_error, spec.episode_id))
            goal = goal_of(spec, task) if goal_of else getattr(task, "goal", "")

            while not setup_error and i < step_budget:
                if episode_deadline is not None and time.time() > episode_deadline:
                    wall_clock_exceeded = True
                    break
                try:
                    record = agent.step(goal, i)
                except Exception as exc:
                    if not _alive(env):
                        env_crash = True
                        break
                    action_errors += 1
                    record = _error_step(i, exc)
                    record["env_crash_candidate"] = True
                record = self._attach_image(record, spec.episode_id, i)
                self.recorder.add_step(record)
                steps.append(record)
                i += 1
                if record.get("done"):
                    break

            checker_error = None
            task_score = None
            if not env_crash and not setup_error:
                try:
                    # A FLOAT, not a bool: composite tasks return 0.5 and upstream averages the
                    # floats (`suite_utils.py:248, 684-716`). `bool(0.5)` counted half credit as a
                    # full success until 2026-11-10.
                    raw = task.is_successful(env)
                    # Any numeric scalar (incl. numpy bool_/int64/float64); None or text is not a score.
                    try:
                        if raw is None or isinstance(raw, (str, bytes)):
                            raise TypeError
                        task_score = float(raw)
                    except (TypeError, ValueError):
                        checker_error = "non-numeric checker result: %r" % (raw,)
                except Exception as exc:  # noqa: BLE001 - a checker must never end the run
                    # A raising checker is not proof that the environment died. AndroidWorld's
                    # own a11y accessor raises when it cannot dump the tree (observed on the
                    # DeskClock timer screen, 2026-11-06) while the device stays usable. Ask
                    # the environment before putting a crash in the table.
                    checker_error = "%s: %s" % (type(exc).__name__, exc)
                    env_crash = not _alive(env)
            declared = bool(steps and steps[-1].get("done"))
            # UPSTREAM REQUIRES BOTH. `suite_utils.py:261` is
            #     agent_successful = task_successful if interaction_results.done else 0.0
            # i.e. a correct device state does not count unless the agent DECLARED done. The
            # harness used to record `success = task.is_successful(env)` alone, which is a
            # different metric: it asks "is the world right" and ignores "did the agent say so".
            # That inflates success for any agent that stops early and understates none, and it
            # makes our numbers incomparable with every published AndroidWorld score.
            # `agent_successful = task_successful if interaction_results.done else 0.0`; an episode
            # that was never judged (setup error, crash, raising or non-numeric checker) has no score
            # — upstream gives it NaN and its mean skips it (`suite_utils.py:544-558`).
            score = None if task_score is None else (task_score if declared else 0.0)
            # upstream prints "Task Successful" iff agent_successful > 0.5 (`suite_utils.py:263`)
            success = score is not None and score > 0.5
            task_successful = task_score is not None and task_score > 0.5
            # An episode whose checker raised was never judged. It must not be counted as a model
            # outcome: on 2026-11-09 twelve such episodes ran to completion (n_steps 4-25) and were
            # reported as 6 timeouts and 6 premature terminations, because `timeout` and
            # `declared_done` describe what the AGENT did and the missing verdict was never surfaced.
            # Any consumer that puts a checker-unavailable episode in a model bucket is now wrong.
            unscored = bool(checker_error) and not env_crash
            result = {
                "success": success,
                "score": score,
                "task_score": task_score,
                # the two upstream factors, kept apart so a post-mortem can tell
                # "world was right but agent never said done" from "world was wrong"
                "task_successful": task_successful,
                "setup_error": setup_error,
                "declared_done": declared,
                "unscored": unscored,
                "checker_unavailable": bool(checker_error) or env_crash,
                "checker_error": checker_error,
                "n_steps": len(steps),
                "n_invalid_actions": sum(1 for s in steps if s.get("invalid_action")),
                "n_action_errors": action_errors,
                "env_crash": env_crash,
                "emulator_restarts": 0,
                "timeout": (not success) and (not declared) and not setup_error and not unscored and (
                    len(steps) >= step_budget or wall_clock_exceeded),
                "wall_clock_exceeded": wall_clock_exceeded,
                "latency": _latency(steps),
                # 0 for the one-pass model by construction; the generative arm reports its own count.
                "tokens_generated": sum(int((s.get("action_raw") or {}).get("tokens_generated") or 0) for s in steps),
                "task_name": spec.task_name,
                "task_idx": spec.task_idx,
                # The goal is the task's identity: without it in the record there is no way to
                # check that the agent was asked the right thing. Traces from 2026-11-10 could
                # not answer that question at all, and the same omission is what made the
                # 2026-11-07 "goal-only vs history" mix-up invisible for days.
                "goal": goal,
                "seed": spec.seed,
                "agent": spec.agent,
                "checkpoint": spec.checkpoint or self.checkpoint,
            }
            self.recorder.finish_episode(result)
        finally:
            _teardown_task(task, env)
        return action_errors

    def _attach_image(self, record: Dict[str, Any], episode_id: str, i: int) -> Dict[str, Any]:
        if not self.save_screenshots:
            return record
        shot = record.pop("screenshot_pixels", None)
        if shot is None:
            return record
        try:
            import numpy as np
            from PIL import Image

            path = self.recorder.step_image_path(episode_id, i)
            Image.fromarray(np.asarray(shot, dtype=np.uint8)).save(path)
            record["screenshot"] = path
        except Exception as exc:
            record["screenshot_error"] = "%s: %s" % (type(exc).__name__, exc)
        return record


def resolve_budget(task: Any, fallback: int) -> int:
    """`int(10 * task.complexity)` when the instance carries a numeric complexity, else `fallback`."""
    c = getattr(task, "complexity", None)
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return int(10 * c)
    return int(fallback)


def _jsonable(value: Any) -> Any:
    """Task params can contain datetimes and enums; the trajectory is JSON, so coerce early.

    AndroidWorld's `generate_random_params` returns things like a CalendarEvent, which killed a run
    at the first calendar task *after* the episode finished (found 2026-11-06).
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _init_task(task: Any, env: Any) -> None:
    """AndroidWorld calls it ``initialize_task``; fakes in our tests call it ``initialize``."""
    for name in ("initialize_task", "initialize"):
        fn = getattr(task, name, None)
        if callable(fn):
            fn(env)
            return
    raise AttributeError("task exposes neither initialize_task nor initialize: %r" % type(task))


def _teardown_task(task: Any, env: Any = None) -> None:
    """AndroidWorld's `tear_down(env)` resets the apps and closes recents.

    Called with no `env` it raised TypeError every single episode, and the bare `except`
    below swallowed it — so no episode was ever torn down. `TaskEval.tear_down` is
    `tear_down(self, env)` (task_eval.py:182) and both upstream callers pass the env
    (suite_utils.py:288, server/android_server.py:183). A leftover foreground app is exactly
    how one task's screen leaks into the next task's first observation.
    """
    fn = getattr(task, "tear_down", None)
    if not callable(fn):
        return
    try:
        fn(env)
    except TypeError:
        # Some fakes in our own tests take no argument; keep those working, but only after
        # the real signature has been tried.
        try:
            fn()
        except Exception:
            pass
    except Exception:
        pass


def _alive(env: Any, attempts: int = 2, delay: float = 1.0) -> bool:
    """Dead only after repeated disagreement.

    A single failed probe is a transient adb hiccup, and treating it as a crash would put a
    false crash in the table — which is exactly the kind of number that got quoted without
    checking (found on the 5-task gate run, 2026-11-06).
    """
    probe = getattr(env, "is_alive", None)
    if not callable(probe):
        return True
    for attempt in range(attempts):
        try:
            if probe():
                return True
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(delay)
    return False


def _coverage_of(task: Any) -> Optional[Dict[str, Any]]:
    """Gold action types per task, when AndroidWorld can report them.

    ``None`` when it cannot: a fabricated 1.0 would hide the action-space gap that
    makes SR_reachable different from SR_all.
    """
    getter = getattr(task, "gold_action_types", None)
    if not callable(getter):
        return None
    try:
        gold = list(getter() or [])
    except Exception:
        return None
    if not gold:
        return None
    from .adapter import EMITTABLE

    return {
        "emittable": sum(1 for a in gold if a in EMITTABLE),
        "total": len(gold),
        "types": gold,
    }


def _error_step(i: int, exc: Exception) -> Dict[str, Any]:
    return {
        "i": i,
        "action_raw": {"action": None},
        "action_executed": None,
        "confidence": None,
        "invalid_action": False,
        "action_ok": False,
        "screen_changed": None,
        "emitted": False,
        "error": "%s: %s" % (type(exc).__name__, exc),
        "done": False,
        "model_ms": None,
        "env_ms": None,
    }


def _latency(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    model = [s["model_ms"] for s in steps if s.get("model_ms") is not None]
    env = [s["env_ms"] for s in steps if s.get("env_ms") is not None]
    return {
        "model_p50": _median(model),
        "model_p95": _p95(model),
        "env_p50": _median(env),
        "env_total_ms": round(sum(env), 1) if env else None,
        "e2e_total_ms": round(sum(model) + sum(env), 1) if (model or env) else None,
    }


def _harness_id() -> str:
    """git sha when there is a checkout, package hash in the image."""
    sha = _git_sha()
    if sha != "unknown":
        return sha
    try:
        from . import harness_version

        return harness_version()
    except Exception:
        return "unknown"


def _git_sha() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=_repo_root(),
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _repo_root() -> str:
    import os

    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def new_run_id(agent: str = "d1_sft") -> str:
    return "%s-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), agent, uuid.uuid4().hex[:6])
