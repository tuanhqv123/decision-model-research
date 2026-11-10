"""Metric definitions: the single source of truth for every number the harness reports.

Two properties are enforced here rather than in the reporting code:

* Every metric carries its denominator. A rate without a denominator is not returned.
* ASR and SR are different quantities and never share a function. A harness that
  conflates them reports "90% action accuracy, 0% tasks" as success.

Episode and step records are plain dicts matching the trajectory schema in the plan
(``runs/<run_id>/episodes/<episode_id>.jsonl``).
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SPATIAL_ACTIONS = ("click", "long_press")
CLICK_ACTIONS = ("click", "long_press")


@dataclass(frozen=True)
class Ratio:
    """A rate that cannot be reported without knowing what it is a rate of."""

    num: int
    den: int

    @property
    def value(self) -> Optional[float]:
        return (self.num / self.den) if self.den else None

    def as_dict(self) -> Dict[str, Any]:
        return {"num": self.num, "den": self.den, "value": self.value}


def wilson(k: int, n: int, z: float = 1.959963985) -> Tuple[Optional[float], Optional[float]]:
    """95% Wilson interval for a binomial proportion."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _action_of(step: Dict[str, Any]) -> str:
    raw = step.get("action_raw") or {}
    return (raw.get("action") or step.get("action") or "").strip().lower()


def _executed(step: Dict[str, Any]) -> bool:
    return bool(step.get("action_ok")) and not bool(step.get("invalid_action"))


# ---------------------------------------------------------------- task level


def success_rate(episodes: Sequence[Dict[str, Any]]) -> Ratio:
    return Ratio(sum(1 for e in episodes if e.get("success")), len(episodes))


def timeout_rate(episodes: Sequence[Dict[str, Any]]) -> Ratio:
    return Ratio(sum(1 for e in episodes if e.get("timeout")), len(episodes))


def crash_rate(episodes: Sequence[Dict[str, Any]]) -> Ratio:
    return Ratio(sum(1 for e in episodes if e.get("env_crash")), len(episodes))


def declared_done_rate(episodes: Sequence[Dict[str, Any]]) -> Ratio:
    return Ratio(sum(1 for e in episodes if e.get("declared_done")), len(episodes))


def steps_to_success(episodes: Sequence[Dict[str, Any]]) -> Optional[float]:
    steps = sorted(e.get("n_steps", 0) for e in episodes if e.get("success"))
    if not steps:
        return None
    mid = len(steps) // 2
    return float(steps[mid]) if len(steps) % 2 else (steps[mid - 1] + steps[mid]) / 2.0


def recovery_rate(
    episodes: Sequence[Dict[str, Any]], steps_by_episode: Dict[str, Sequence[Dict[str, Any]]]
) -> Ratio:
    """P(success | episode saw at least one stuck or action-error event).

    A model that never recovers from a mistake cannot finish a multi-step task, so
    this is reported separately from SR instead of being averaged into it.
    """
    stressed = 0
    recovered = 0
    for ep in episodes:
        eid = ep.get("episode_id")
        steps = steps_by_episode.get(eid, [])
        stress = any(
            bool(s.get("invalid_action"))
            or bool(s.get("error"))
            or s.get("action_ok") is False
            or s.get("screen_changed") is False
            for s in steps
        )
        if stress:
            stressed += 1
            recovered += 1 if ep.get("success") else 0
    return Ratio(recovered, stressed)


# ---------------------------------------------------------------- step level


def emitted_actions(steps: Sequence[Dict[str, Any]]) -> int:
    """Steps that produced an executable action.

    A terminal ``status`` step is a declaration, not an action: it stays in the
    trajectory (so the termination is attributable) but sits in no action denominator.
    """
    return sum(1 for s in steps if s.get("emitted") is not False)


def _emitted(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in steps if s.get("emitted") is not False]


def asr_exec(steps: Sequence[Dict[str, Any]]) -> Ratio:
    """Executor acceptance rate. Says nothing about whether the action was correct."""
    acts = _emitted(steps)
    return Ratio(sum(1 for s in acts if s.get("action_ok")), len(acts))


def invalid_action_rate(steps: Sequence[Dict[str, Any]]) -> Ratio:
    acts = _emitted(steps)
    return Ratio(sum(1 for s in acts if s.get("invalid_action")), len(acts))


def env_reject_rate(steps: Sequence[Dict[str, Any]]) -> Ratio:
    acts = _emitted(steps)
    return Ratio(
        sum(1 for s in acts if s.get("action_ok") is False and not s.get("invalid_action")),
        len(acts),
    )


def stuck_rate(steps: Sequence[Dict[str, Any]]) -> Ratio:
    executed = [s for s in steps if _executed(s)]
    return Ratio(sum(1 for s in executed if s.get("screen_changed") is False), len(executed))


def grounding_error_proxy(steps: Sequence[Dict[str, Any]]) -> Ratio:
    """Click point outside every clickable a11y node.

    A proxy only: online there is no gold coordinate, so this must never be presented
    as the AC-v2 grounding number.
    """
    spatial = [s for s in steps if _action_of(s) in CLICK_ACTIONS and s.get("grounding_error") is not None]
    return Ratio(sum(1 for s in spatial if s.get("grounding_error")), len(spatial))


# -------------------------------------------------- coverage-adjusted success


def coverage_split(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Separate "the model cannot express the action" from "the model is wrong".

    Each episode carries ``coverage = {"emittable": n, "total": m}``: how many gold
    action types its solution needs vs how many of those this checkpoint can emit.
    """
    all_eps = list(episodes)
    reachable = []
    emittable = 0
    total = 0
    for ep in all_eps:
        cov = ep.get("coverage") or {}
        e_n = int(cov.get("emittable", 0))
        e_t = int(cov.get("total", 0))
        emittable += e_n
        total += e_t
        if e_t and e_n == e_t:
            reachable.append(ep)
    return {
        "sr_all": success_rate(all_eps).as_dict(),
        "sr_reachable": success_rate(reachable).as_dict(),
        "coverage": Ratio(emittable, total).as_dict(),
        "n_reachable": len(reachable),
        "n_all": len(all_eps),
    }


# ---------------------------------------------------------------- calibration


def proxy_correct(step: Dict[str, Any]) -> bool:
    """Proxy label for confidence: the action ran and the screen actually changed."""
    return bool(step.get("action_ok")) and (step.get("screen_changed") is not False)


def ece(confs: Sequence[float], labels: Sequence[bool], bins: int = 10) -> Optional[float]:
    pairs = [(float(c), bool(y)) for c, y in zip(confs, labels) if c is not None]
    if not pairs:
        return None
    total = len(pairs)
    err = 0.0
    edges = [i / bins for i in range(bins + 1)]
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        chunk = [(c, y) for c, y in pairs if (lo <= c < hi) or (i == bins - 1 and c == hi)]
        if not chunk:
            continue
        conf = sum(c for c, _ in chunk) / len(chunk)
        acc = sum(1 for _, y in chunk if y) / len(chunk)
        err += (len(chunk) / total) * abs(conf - acc)
    return err


def coverage_risk(
    confs: Sequence[float],
    labels: Sequence[bool],
    taus: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95),
) -> List[Dict[str, Any]]:
    """Selective-prediction curve: act iff confidence >= tau."""
    out = []
    for tau in taus:
        kept = [(float(c), bool(y)) for c, y in zip(confs, labels) if c is not None and c >= tau]
        n = len(confs)
        acc = (sum(1 for _, y in kept if y) / len(kept)) if kept else None
        out.append(
            {
                "tau": tau,
                "coverage": (len(kept) / n) if n else None,
                "risk": (1.0 - acc) if acc is not None else None,
            }
        )
    return out


# ---------------------------------------------------------------- aggregate


def _latency_block(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Model and environment latency, reported apart: the env dominates end-to-end."""

    def _p50(vals):
        if not vals:
            return None
        ordered = sorted(vals)
        return round(ordered[len(ordered) // 2], 1)

    def _p95(vals):
        if not vals:
            return None
        ordered = sorted(vals)
        return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1)

    model = [s["model_ms"] for s in steps if s.get("model_ms") is not None]
    env = [s["env_ms"] for s in steps if s.get("env_ms") is not None]
    return {
        "model_p50": _p50(model),
        "model_p95": _p95(model),
        "env_p50": _p50(env),
        "env_total_ms": round(sum(env), 1) if env else None,
        "n_steps_with_latency": len(model),
    }


def official_sr(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """AndroidWorld's own success rate, computed the way `suite_utils` computes it.

    Per episode: the float `score` (`agent_successful`, 0.5 = partial credit). An episode whose task
    raised — setup error, unscored checker, crashed device — has no score: upstream records `NaN`
    (`suite_utils.py:544-558`) and its pandas `mean()` skips it. SR is the mean over task templates
    of each template's mean score (`suite_utils.py:684-716`). Records written before 2026-11-10 carry
    only `success`; they are read as 1.0 / 0.0.
    """
    per: Dict[str, List[float]] = {}
    excluded: List[str] = []
    for e in episodes:
        name = e.get("task_name") or e.get("episode_id") or "?"
        if e.get("setup_error") or e.get("unscored") or e.get("env_crash"):
            excluded.append(name)
            continue
        score = e.get("score")
        if score is None:
            if "success" not in e:
                excluded.append(name)
                continue
            score = 1.0 if e.get("success") else 0.0
        per.setdefault(name, []).append(float(score))
    per_task = {k: sum(v) / len(v) for k, v in per.items()}
    excluded = sorted(set(excluded) - set(per_task))
    return {
        "sr": (sum(per_task.values()) / len(per_task)) if per_task else None,
        "n_tasks_scored": len(per_task),
        "n_tasks_excluded": len(excluded),
        "excluded": excluded,
        "per_task": per_task,
    }


def summarize(
    episodes: Sequence[Dict[str, Any]],
    steps: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Every metric the harness is allowed to report, with denominators."""
    by_ep: Dict[str, List[Dict[str, Any]]] = {}
    for s in steps:
        by_ep.setdefault(s.get("episode_id"), []).append(s)

    confs = [s.get("confidence") for s in steps]
    labels = [proxy_correct(s) for s in steps]
    sr = success_rate(episodes)
    lo, hi = wilson(sr.num, sr.den)
    cov = coverage_split(episodes)

    return {
        "protocol_note": "proxy metrics are labelled; ASR is not SR",
        "task": {
            "official": official_sr(episodes),
            "sr": sr.as_dict(),
            "sr_wilson95": [lo, hi],
            "sr_all": cov["sr_all"],
            "sr_reachable": cov["sr_reachable"],
            "coverage": cov["coverage"],
            "n_reachable": cov["n_reachable"],
            "n_all": cov["n_all"],
            "timeout_rate": timeout_rate(episodes).as_dict(),
            "crash_rate": crash_rate(episodes).as_dict(),
            "declared_done_rate": declared_done_rate(episodes).as_dict(),
            "steps_to_success_median": steps_to_success(episodes),
            "recovery_rate": recovery_rate(episodes, by_ep).as_dict(),
        },
        "step": {
            "n_emitted_actions": emitted_actions(steps),
            "asr_exec": asr_exec(steps).as_dict(),
            "invalid_action_rate": invalid_action_rate(steps).as_dict(),
            "env_reject_rate": env_reject_rate(steps).as_dict(),
            "stuck_rate": stuck_rate(steps).as_dict(),
            "grounding_error_proxy": grounding_error_proxy(steps).as_dict(),
        },
        "latency": _latency_block(steps),
        "calibration_proxy": {
            "ece10": ece(confs, labels, bins=10),
            "coverage_risk": coverage_risk(confs, labels),
        },
    }
