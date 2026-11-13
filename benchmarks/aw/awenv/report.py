"""Aggregation and paired comparison.

Two rules:

* Every reported number carries its denominator, and proxy metrics are labelled as
  proxies in the output itself, not in a footnote someone will drop.
* Agents are compared by **paired bootstrap over shared episode ids**, because the
  same task instances are reused across agents. Comparing two unpaired point
  estimates on 116 tasks is how a 2-task difference gets published as a result.
"""
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import Ratio, summarize, wilson


def paired_ids(episodes_a: Sequence[Dict[str, Any]], episodes_b: Sequence[Dict[str, Any]]) -> List[str]:
    a = {e.get("episode_id") for e in episodes_a}
    b = {e.get("episode_id") for e in episodes_b}
    return sorted(i for i in (a & b) if i)


def paired_bootstrap_delta(
    episodes_a: Sequence[Dict[str, Any]],
    episodes_b: Sequence[Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Delta in success rate (a - b) with a percentile CI over shared episodes."""
    index_a = {e["episode_id"]: 1 if e.get("success") else 0 for e in episodes_a if e.get("episode_id")}
    index_b = {e["episode_id"]: 1 if e.get("success") else 0 for e in episodes_b if e.get("episode_id")}
    ids = sorted(set(index_a) & set(index_b))
    if not ids:
        return {"n_pairs": 0, "delta": None, "ci": [None, None], "excludes_zero": None}

    deltas = []
    rng = random.Random(seed)
    n = len(ids)
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        da = sum(index_a[i] for i in sample) / n
        db = sum(index_b[i] for i in sample) / n
        deltas.append(da - db)
    deltas.sort()

    observed = sum(index_a[i] for i in ids) / n - sum(index_b[i] for i in ids) / n
    lo = deltas[int((alpha / 2) * n_boot)]
    hi = deltas[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {
        "n_pairs": n,
        "delta": observed,
        "ci": [lo, hi],
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def _fmt(ratio: Dict[str, Any]) -> str:
    if not isinstance(ratio, dict) or "num" not in ratio:
        return "—"
    value = ratio.get("value")
    if value is None:
        return "n/a (0/%d)" % ratio["den"]
    return "%.1f%% (%d/%d)" % (100.0 * value, ratio["num"], ratio["den"])


def format_markdown(report: Dict[str, Any], title: str = "AndroidWorld run") -> str:
    task = report["task"]
    step = report["step"]
    cal = report["calibration_proxy"]
    lo, hi = task["sr_wilson95"]
    lines = ["## " + title, ""]

    # Completeness first, never after the table. A partial run that prints its SR before saying it
    # stopped at 10/116 is indistinguishable from a finished benchmark, and that is exactly the
    # misread this harness is supposed to prevent.
    comp = report.get("completeness") or {}
    if comp:
        done, planned = comp.get("completed"), comp.get("planned")
        if comp.get("complete"):
            lines.append("**Completeness: %s/%s tasks attempted — complete.**" % (done, planned))
        else:
            lines.append(
                "**Completeness: %s/%s tasks attempted — INCOMPLETE. The success rate below is over "
                "the tasks that ran, not the suite.**" % (done, planned))
        lines.append("")

    # The number comparable to published AndroidWorld results, before anything else.
    off = task.get("official") or {}
    if off:
        sr_txt = "n/a" if off.get("sr") is None else "%.1f%%" % (100.0 * off["sr"])
        line = "**Official SR (suite_utils): %s over %d tasks; %d excluded (raised)" % (
            sr_txt, off.get("n_tasks_scored", 0), off.get("n_tasks_excluded", 0))
        if off.get("excluded"):
            line += ": " + ", ".join(off["excluded"])
        lines += [line + ".**", ""]

    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append("| **SR (all)** | %s |" % _fmt(task["sr"]))
    if lo is not None:
        lines.append("| SR 95%% Wilson | %.1f%% – %.1f%% |" % (100 * lo, 100 * hi))
    lines.append("| **SR (reachable action space)** | %s |" % _fmt(task["sr_reachable"]))
    lines.append("| action-space coverage | %s |" % _fmt(task["coverage"]))
    lines.append("| timeout rate | %s |" % _fmt(task["timeout_rate"]))
    lines.append("| emulator crash rate | %s |" % _fmt(task["crash_rate"]))
    lines.append("| declared done rate | %s |" % _fmt(task["declared_done_rate"]))
    steps_med = task.get("steps_to_success_median")
    lines.append("| steps to success (median) | %s |" % ("—" if steps_med is None else "%.0f" % steps_med))
    lines.append("| recovery rate (after stress) | %s |" % _fmt(task["recovery_rate"]))
    lines.append("")
    lines.append("| step metric | value |")
    lines.append("|---|---|")
    lines.append("| emitted actions | %d |" % step["n_emitted_actions"])
    lines.append("| ASR_exec (executor accepted) | %s |" % _fmt(step["asr_exec"]))
    lines.append("| invalid action rate | %s |" % _fmt(step["invalid_action_rate"]))
    lines.append("| env reject rate | %s |" % _fmt(step["env_reject_rate"]))
    lines.append("| stuck rate (no screen change) | %s |" % _fmt(step["stuck_rate"]))
    lines.append("| grounding error (**proxy**) | %s |" % _fmt(step["grounding_error_proxy"]))
    lines.append("")

    ece = cal.get("ece10")
    lines.append("Calibration (**proxy label**: action executed and screen changed)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append("| ECE (10 bins) | %s |" % ("—" if ece is None else "%.3f" % ece))
    lines.append("")
    lines.append("| τ | coverage | risk |")
    lines.append("|---|---|---|")
    for row in cal.get("coverage_risk", []):
        cov = "—" if row["coverage"] is None else "%.1f%%" % (100 * row["coverage"])
        risk = "—" if row["risk"] is None else "%.1f%%" % (100 * row["risk"])
        lines.append("| %.2f | %s | %s |" % (row["tau"], cov, risk))
    lines.append("")
    lines.append(
        "_ASR is not SR: an action can execute cleanly and still fail the task. "
        "Grounding and calibration are proxies, not the AC-v2 numbers._"
    )
    return "\n".join(lines)


def build_report(episodes: Sequence[Dict[str, Any]], steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return summarize(episodes, steps)


# ------------------------------------------------------------ nightly regression

DEFAULT_THRESHOLDS = {
    "sr_drop_tasks": 2,       # absolute drop in successful episodes
    "crash_rate": 0.0,        # any crash at all
    "model_p50_regress": 0.25,
    "invalid_action_delta": 0.05,
}


def _value(report, group, name, default=None):
    section = (report or {}).get(group) or {}
    payload = section.get(name)
    if isinstance(payload, dict):
        return payload.get("value", default)
    return payload if payload is not None else default


def _num(report, group, name, default=0):
    section = (report or {}).get(group) or {}
    payload = section.get(name)
    if isinstance(payload, dict):
        return payload.get("num", default)
    return default


def compare_nightly(current, baseline, thresholds=None):
    """Alerts for a nightly run against a stored baseline. Empty list means healthy."""
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    alerts = []

    sr_now = _num(current, "task", "sr")
    sr_base = _num(baseline, "task", "sr")
    if sr_base - sr_now >= th["sr_drop_tasks"]:
        alerts.append("SR dropped %d task(s): %d -> %d" % (sr_base - sr_now, sr_base, sr_now))

    crash = (current.get("task") or {}).get("crash_rate") or {}
    if (crash.get("value") or 0.0) > th["crash_rate"]:
        alerts.append("crash rate %s/%s exceeds %.2f" % (
            crash.get("num", 0), crash.get("den", 0), th["crash_rate"]))

    p50_now = (current.get("latency") or {}).get("model_p50")
    p50_base = (baseline.get("latency") or {}).get("model_p50")
    if p50_now and p50_base:
        ratio = float(p50_now) / float(p50_base) - 1.0
        if ratio > th["model_p50_regress"]:
            alerts.append("latency p50 regressed %.0f%% (%.1f -> %.1f ms)" % (
                ratio * 100.0, p50_base, p50_now))

    inv_now = _value(current, "step", "invalid_action_rate") or 0.0
    inv_base = _value(baseline, "step", "invalid_action_rate") or 0.0
    if inv_now - inv_base > th["invalid_action_delta"]:
        alerts.append("invalid-action rate rose %.1f pp (%.1f%% -> %.1f%%)" % (
            (inv_now - inv_base) * 100.0, inv_base * 100.0, inv_now * 100.0))

    return alerts
