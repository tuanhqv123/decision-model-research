"""Decision -> AndroidWorld JSONAction mapping.

Two rules make this module trustworthy:

1. No silent defaults. If a decision cannot be executed faithfully, the result is
   flagged ``invalid`` and the executed action is recorded as a ``wait``. It is never
   quietly treated as a clean no-op.
2. Any scaffolding that changes what the model meant (e.g. deriving ``app_name`` from
   goal keywords) is stamped into ``scaffolding`` so it reaches the trajectory record.

The model emits 8 of AndroidWorld's 13 action types. The gap is declared in
``NOT_EMITTED`` and counted by the metrics layer, not hidden here.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

EMITTABLE = (
    "answer",
    "click",
    "long_press",
    "type_text",
    "scroll",
    "open_app",
    "navigate_back",
    "navigate_home",
    "wait",
    "status",
)

AW_ACTION_SPACE = (
    "click",
    "double_tap",
    "input_text",
    "keyboard_enter",
    "long_press",
    "navigate_home",
    "navigate_back",
    "open_app",
    "scroll",
    "swipe",
    "wait",
    "status",
    "answer",
    "infeasible",
)

NOT_EMITTED = tuple(a for a in AW_ACTION_SPACE if a not in EMITTABLE)

SPATIAL = ("click", "long_press", "type_text")
SCROLL_DIRECTIONS = ("up", "down", "left", "right")

CANON_APPS = [
    "android world", "audio recorder", "broccoli app", "camera", "chrome", "clipper", "clock",
    "contacts", "dialer", "files", "joplin", "markor", "miniwob", "open tracks sports tracker",
    "osmand", "pro expense", "retro music", "settings", "simple calendar pro", "simple draw pro",
    "simple gallery pro", "simple sms messenger", "tasks", "vlc",
]

APP_ALIASES = {
    "stopwatch": "clock", "timer": "clock", "alarm": "clock", "desk clock": "clock",
    "contact": "contacts", "phone": "dialer", "call": "dialer",
    "browser": "chrome", "google chrome": "chrome", "web": "chrome",
    "note": "markor", "notes": "markor", "text editor": "markor",
    "message": "simple sms messenger", "messages": "simple sms messenger", "sms": "simple sms messenger",
    "messenger": "simple sms messenger", "text": "simple sms messenger",
    "calendar": "simple calendar pro", "todo": "tasks", "to do": "tasks", "task": "tasks",
    "gallery": "simple gallery pro", "photos": "simple gallery pro", "photo": "simple gallery pro",
    "draw": "simple draw pro", "sketch": "simple draw pro", "whiteboard": "simple draw pro",
    "expense": "pro expense", "expenses": "pro expense", "recorder": "audio recorder",
    "music": "retro music", "player": "retro music", "maps": "osmand", "files app": "files",
}

APP_HINTS = {
    "contact": "contacts", "stopwatch": "clock", "timer": "clock", "alarm": "clock", "clock": "clock",
    "note": "markor", "markor": "markor", "sms": "simple sms messenger", "message": "simple sms messenger",
    "text ": "simple sms messenger", "calendar": "simple calendar pro", "task": "tasks", "to-do": "tasks",
    "expense": "pro expense", "receipt": "pro expense", "gallery": "simple gallery pro",
    "photo": "simple gallery pro", "picture": "simple gallery pro", "file": "files", "music": "retro music",
    "song": "retro music", "audio": "audio recorder", "record": "audio recorder", "camera": "camera",
    "video": "camera", "photo ": "camera", "browser": "chrome", "web": "chrome", "http": "chrome",
    "setting": "settings", "draw": "simple draw pro", "sketch": "simple draw pro", "joplin": "joplin",
    "osmand": "osmand", "navigation": "osmand", "vlc": "vlc",
}


def canon_app(name: Optional[str]) -> str:
    """Normalise a free-form app name into AndroidWorld's own vocabulary."""
    n = (name or "").strip().lower()
    if not n:
        return ""
    if n in APP_ALIASES:
        return APP_ALIASES[n]
    if n in CANON_APPS:
        return n
    for c in CANON_APPS:
        if n in c or c in n:
            return c
    return n


def app_from_goal(goal: Optional[str]) -> str:
    """Scaffolding: derive an app name from goal keywords. Only used when explicitly enabled."""
    g = (goal or "").lower()
    hits = [(k, v) for k, v in APP_HINTS.items() if k in g]
    return max(hits, key=lambda kv: len(kv[0]))[1] if hits else ""


@dataclass
class MappedAction:
    """Result of mapping one model decision to executable AndroidWorld actions."""

    actions: List[Dict[str, Any]] = field(default_factory=list)
    done: bool = False
    invalid: bool = False
    scaffolding: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    @property
    def primary(self) -> Optional[str]:
        return self.actions[0]["action_type"] if self.actions else None


def _point(dec: Dict[str, Any]):
    pt = dec.get("point") or [None, None]
    x, y = (pt + [None, None])[:2] if isinstance(pt, (list, tuple)) else (None, None)
    if x is None or y is None:
        return None
    return int(round(float(x))), int(round(float(y)))


def map_decision(
    dec: Dict[str, Any],
    allow_app_fallback: bool = False,
    goal: str = "",
    done_threshold: float = 0.0,
) -> MappedAction:
    """Map a decision dict to AndroidWorld actions, flagging anything not faithful.

    ``dec`` fields: action, point, text, app_name, direction, and optionally is_done
    (the p(task complete) readout).

    ``done_threshold = 0.0`` (the default) DISABLES the is_done gate, which is the correct
    configuration. Upstream lets the agent decide `done`; gating it is a protocol deviation,
    and the signal it was gated on turned out to be null — see
    benchmarks/results/2026-11-10-episodic-s0-fails-prereg.md. Measured on 824 validation
    rows: `step == n_steps` and `action == status` are the SAME 137 rows, so `is_done` is a
    copy of the action label (AUROC 0.960 including terminal rows, `auroc_nonterm` not
    computable because there are 0 positives among the 687 non-status rows, risk 0.670 at 50%
    coverage). A non-zero threshold re-enables a deviation whose signal does not exist.
    """
    action = (dec.get("action") or "").strip().lower()

    if action == "status":
        # Kept for a deliberately controlled experiment, never as the default. The readout
        # cannot separate "finished" from "not finished" on this corpus: refusing a `status`
        # on `is_done` is refusing it on a restatement of the action label, and the threshold
        # therefore has no operating point that is better than not gating at all.
        p_done = dec.get("is_done")
        if done_threshold > 0.0 and p_done is not None and float(p_done) < done_threshold:
            return MappedAction(
                actions=[{"action_type": "wait"}],
                invalid=False,
                scaffolding=["status_blocked_by_is_done:%.3f" % float(p_done)],
                reason="is_done_below_threshold",
            )
        return MappedAction(actions=[], done=True)

    if action not in EMITTABLE:
        return MappedAction(
            actions=[{"action_type": "wait"}],
            invalid=True,
            reason="unmappable_action:%s" % (action or "empty"),
        )

    if action == "answer":
        # contract v4: AndroidWorld stores the LAST answer text (interface.py interaction_cache); the agent turns the
        # chosen element's full text into the accumulated, comma-joined answer before execution.
        text = str(dec.get("answer_text") or "").strip()
        if not text:
            return MappedAction(actions=[{"action_type": "wait"}], invalid=True, reason="answer_missing")
        return MappedAction(actions=[{"action_type": "answer", "text": text}])

    if action in ("click", "long_press"):
        pt = _point(dec)
        if pt is None:
            return MappedAction(
                actions=[{"action_type": "wait"}],
                invalid=True,
                reason="point_missing",
            )
        return MappedAction(actions=[{"action_type": action, "x": pt[0], "y": pt[1]}])

    if action == "type_text":
        pt = _point(dec)
        text = dec.get("text") or ""
        if pt is None:
            return MappedAction(
                actions=[{"action_type": "wait"}],
                invalid=True,
                reason="point_missing",
            )
        if not text:
            return MappedAction(
                actions=[{"action_type": "click", "x": pt[0], "y": pt[1]}],
                invalid=True,
                reason="text_missing",
            )
        # ONE action, not two. Upstream `actuation.py:77-110` shows `input_text`
        # already clicks the focus point itself (`action.x/y` -> click_action),
        # sleeps 1s, optionally clears, types, and presses ENTER. Emitting a
        # separate `click` first therefore tapped the field twice, which on some
        # screens (Markor's filename dialog) closes or re-selects the widget.
        # `clear_text=True` is required: without it typed text is APPENDED to
        # whatever the field already held.
        return MappedAction(
            actions=[
                {
                    "action_type": "input_text",
                    "x": pt[0],
                    "y": pt[1],
                    "text": text,
                    "clear_text": True,
                }
            ]
        )

    if action == "scroll":
        direction = (dec.get("direction") or "").strip().lower()
        if direction not in SCROLL_DIRECTIONS:
            return MappedAction(
                actions=[{"action_type": "wait"}],
                invalid=True,
                reason="bad_scroll_direction:%s" % (direction or "empty"),
            )
        return MappedAction(actions=[{"action_type": "scroll", "direction": direction}])

    if action == "open_app":
        scaffolding: List[str] = []
        name = canon_app(dec.get("app_name"))
        if not name and allow_app_fallback:
            name = canon_app(app_from_goal(goal))
            if name:
                scaffolding.append("app_name_fallback")
        if not name:
            return MappedAction(
                actions=[{"action_type": "wait"}],
                invalid=True,
                reason="app_name_missing",
            )
        return MappedAction(
            actions=[{"action_type": "open_app", "app_name": name}],
            scaffolding=scaffolding,
        )

    return MappedAction(actions=[{"action_type": action}])
