"""Error taxonomy: turn a run into counts per failure bucket.

Gate C's deliverable is a taxonomy with counts, not a single success rate. The buckets are
mutually exclusive per episode by a fixed priority, so the counts add up to the number of
failed episodes instead of double-counting one episode with several symptoms.

Priority (first match wins), because a later symptom is usually a consequence of an earlier one:
    env_crash → timeout → unmappable_action → stuck → premature_status → wrong_element → wrong_value
"""

from typing import Any, Dict, List, Optional, Sequence

BUCKETS = (
    "env_crash",
    "timeout",
    "unmappable_action",
    "stuck",
    "premature_status",
    "wrong_element",
    "wrong_value",
    "unknown",
)


def _by_episode(steps: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for step in steps:
        out.setdefault(step.get("episode_id"), []).append(step)
    return out


def _episode_bucket(episode: Dict[str, Any], steps: List[Dict[str, Any]]) -> Optional[str]:
    if episode.get("env_crash"):
        return "env_crash"
    if episode.get("timeout"):
        return "timeout"

    if any(s.get("invalid_action") for s in steps):
        return "unmappable_action"
    if any(s.get("stuck_event") for s in steps):
        return "stuck"

    actions = [(s.get("action_raw") or {}).get("action") for s in steps]
    if "status" in actions and not episode.get("success"):
        return "premature_status"

    spatial = [s for s in steps if (s.get("action_raw") or {}).get("action") in ("click", "long_press", "type_text")]
    if spatial and any(s.get("screen_changed") is False for s in spatial):
        return "wrong_element"

    if any(s.get("action_ok") is False for s in steps):
        return "wrong_value"

    return "unknown" if not episode.get("success") else None


def summarize_taxonomy(
    episodes: Sequence[Dict[str, Any]],
    steps: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Counts per bucket over failed episodes, with the success rate beside them."""
    by_ep = _by_episode(steps)
    failed = [e for e in episodes if not e.get("success")]

    counts = {b: 0 for b in BUCKETS}
    examples: Dict[str, str] = {}
    for episode in failed:
        eid = episode.get("episode_id")
        bucket = _episode_bucket(episode, by_ep.get(eid, [])) or "unknown"
        counts[bucket] += 1
        examples.setdefault(bucket, eid)

    return {
        "n_episodes": len(episodes),
        "n_failed": len(failed),
        "n_succeeded": len(episodes) - len(failed),
        "counts": counts,
        "examples": examples,
        "action_mix": _action_mix(steps),
    }


def _action_mix(steps: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """What the model actually emitted — the coverage gap lives here, not in the SR."""
    mix: Dict[str, int] = {}
    for step in steps:
        action = (step.get("action_raw") or {}).get("action") or "none"
        mix[action] = mix.get(action, 0) + 1
    return dict(sorted(mix.items(), key=lambda kv: -kv[1]))


def load_run(episodes_path: str, steps_path: str) -> Dict[str, Any]:
    """Summarize a run from the two JSONL streams the recorder writes."""
    import json
    import os

    def _load(path):
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fp:
            return [json.loads(line) for line in fp if line.strip()]

    return summarize_taxonomy(_load(episodes_path), _load(steps_path))
