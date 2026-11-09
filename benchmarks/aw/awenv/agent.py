"""One decision + one execution per step, recorded so failures stay attributable.

The rule that shapes this module: every ambiguity of the 2026-11-02 smoke run has to be
impossible to reconstruct. Nothing is inferred from a step counter, no action degrades
silently, and the history the model sees is built from *what executed*, not from what the
model asked for.

``AgentCore`` is deliberately free of AndroidWorld imports so it can be tested against
fakes; ``make_android_world_agent`` is the thin integration shim loaded on the host.
"""

import time
from typing import Any, Dict, List, Optional

from .adapter import MappedAction, map_decision
from .liveness import image_hash

LOOP_BREAK_ACTION = {"action_type": "navigate_back"}
LOOP_BREAK_TAG = "loop_break"


def _signature(mapped: MappedAction) -> str:
    return "|".join(
        "%s:%s:%s" % (a.get("action_type"), a.get("x"), a.get("y")) for a in mapped.actions
    )


def _raw_of(dec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": dec.get("action"),
        "point": dec.get("point"),
        "text": dec.get("text"),
        "app_name": dec.get("app_name"),
        "direction": dec.get("direction"),
        "confidence": dec.get("confidence"),
        "gate_score": dec.get("gate_score"),
    }


def _history_line(dec: Dict[str, Any], mapped: MappedAction, ok: bool) -> str:
    """History entry: the action that ran, in the model's own vocabulary, plus its outcome."""
    outcome = "ok" if ok else "error"
    action = (dec.get("action") or "").strip().lower()
    point = dec.get("point") or [None, None]
    x, y = (list(point) + [None, None])[:2]

    if action in ("click", "long_press") and x is not None:
        return "%s (%s,%s) -> %s" % (action, int(x), int(y), outcome)
    if action == "type_text" and x is not None:
        return 'type_text (%s,%s) "%s" -> %s' % (int(x), int(y), dec.get("text") or "", outcome)
    if action == "scroll":
        return "scroll %s -> %s" % (dec.get("direction") or "?", outcome)
    if action == "open_app":
        app = mapped.actions[0].get("app_name") if mapped.actions else dec.get("app_name")
        return "open_app %s -> %s" % (app or "?", outcome)
    if action == "status":
        return "status -> declare done"
    return "%s -> %s" % (action or "empty", outcome)


class AgentCore:
    """One model, one environment, one step at a time."""

    name = "d1_2b"

    def __init__(
        self,
        model: Any,
        env: Any,
        allow_app_fallback: bool = False,
        stuck_k: int = 3,
        track: str = "V",
        return_readouts: bool = False,
    ):
        self.model = model
        self.env = env
        self.allow_app_fallback = allow_app_fallback
        self.stuck_k = stuck_k
        self.track = track
        self.return_readouts = return_readouts
        self.history: List[str] = []
        self._seen: List[Any] = []

    def _elements(self) -> Optional[List[Dict[str, Any]]]:
        if self.track != "VA":
            return None
        getter = getattr(self.env, "elements", None)
        return list(getter()) if callable(getter) else None

    def _n_elements(self) -> int:
        getter = getattr(self.env, "elements", None)
        if not callable(getter):
            return 0
        try:
            return len(getter() or [])
        except Exception:
            return 0

    def step(self, goal: str, i: int) -> Dict[str, Any]:
        """Run one decision and one execution; return the step record."""
        before = self.env.screenshot()
        before_hash = image_hash(before)

        started = time.time()
        dec = self.model.decide(
            goal=goal,
            history=list(self.history),
            pixels=before,
            elements=self._elements(),
            track=self.track,
            return_readouts=self.return_readouts,
        )
        model_ms = _model_ms(dec, (time.time() - started) * 1000.0)

        mapped = map_decision(dec, allow_app_fallback=self.allow_app_fallback, goal=goal)
        raw = _raw_of(dec)

        if mapped.done:
            record = self._record(i, raw, mapped, before_hash, None, None, model_ms, 0.0)
            record["screenshot_pixels"] = before  # exactly what the model saw; the runner saves it
            record["action_ok"] = None
            record["done"] = True
            self.history.append(_history_line(dec, mapped, True))
            return record

        stuck_event = self._is_loop(_signature(mapped), before_hash)
        if stuck_event:
            mapped = MappedAction(
                actions=[dict(LOOP_BREAK_ACTION)],
                scaffolding=[LOOP_BREAK_TAG],
                reason="loop_detected",
            )
            self._seen = []

        env_started = time.time()
        ok, error, rejected = True, None, False
        try:
            for action in mapped.actions:
                out = self.env.execute(action) or {}
                if isinstance(out, dict) and out.get("ok") is False:
                    ok, rejected = False, True
                    error = out.get("error") or "executor_rejected"
                    break
        except Exception as exc:  # an execution crash must not end the episode
            ok, error = False, "%s: %s" % (type(exc).__name__, exc)
        env_ms = (time.time() - env_started) * 1000.0

        after_hash = image_hash(self.env.screenshot())
        screen_changed = after_hash != before_hash

        record = self._record(
            i, raw, mapped, before_hash, mapped.actions[0] if mapped.actions else None,
            ok, model_ms, env_ms,
        )
        record["screenshot_pixels"] = before
        record.update(
            {
                "screen_changed": screen_changed,
                "screen_hash_after": after_hash,
                "action_ok": ok,
                "error": error,
                "env_reject": rejected,
                "stuck_event": stuck_event,
                "done": False,
                "history_line": _history_line(dec, mapped, ok),
            }
        )
        self.history.append(record["history_line"])
        return record

    def _record(
        self,
        i: int,
        raw: Dict[str, Any],
        mapped: MappedAction,
        before_hash: str,
        executed: Optional[Dict[str, Any]],
        model_ok: Optional[bool],
        model_ms: float,
        env_ms: float,
    ) -> Dict[str, Any]:
        return {
            "i": i,
            "action_raw": raw,
            "action_executed": executed,
            "confidence": raw.get("confidence"),
            "gate_score": raw.get("gate_score"),
            "invalid_action": bool(mapped.invalid),
            "reason": mapped.reason,
            "scaffolding": list(mapped.scaffolding),
            "screen_hash": before_hash,
            "model_ms": model_ms,
            "env_ms": env_ms,
            "a11y_elements_n": self._n_elements(),
            "emitted": executed is not None,
        }

    def _is_loop(self, sig: str, screen_hash: str) -> bool:
        """k identical decisions on an unchanged screen is a loop, not persistence."""
        self._seen.append((sig, screen_hash))
        if len(self._seen) < self.stuck_k:
            return False
        return len(set(self._seen[-self.stuck_k:])) == 1


def _model_ms(dec: Dict[str, Any], fallback: float) -> float:
    lat = dec.get("latency_ms")
    if isinstance(lat, dict):
        value = lat.get("model")
        if value is not None:
            return float(value)
    if isinstance(lat, (int, float)):
        return float(lat)
    return round(fallback, 1)


def make_android_world_agent(core: AgentCore, env: Any, name: Optional[str] = None):
    """Build the AndroidWorld agent class on the host.

    Kept to the minimum surface: AndroidWorld's ``EnvironmentInteractingAgent`` owns the
    loop and the environment handle; ``core`` owns every decision and every record.
    """
    from android_world.env import interface as aw_interface

    class _D1Agent(aw_interface.EnvironmentInteractingAgent):
        def __init__(self, environment):
            super().__init__(environment)
            self.core = core
            self.name = name or core.name
            self._i = 0

        def step(self, goal):
            record = self.core.step(goal, self._i)
            self._i += 1
            done = bool(record.get("done"))
            return aw_interface.AgentInteractionResult(done=done, data=record)

    return _D1Agent(env)
