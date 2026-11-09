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
    "click",
    "long_press",
    "type_text",
    "scroll",
    "open_app",
    "navigate_back",
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
) -> MappedAction:
    """Map a decision dict to AndroidWorld actions, flagging anything not faithful.

    ``dec`` fields: action, point, text, app_name, direction.
    """
    action = (dec.get("action") or "").strip().lower()

    if action == "status":
        return MappedAction(actions=[], done=True)

    if action not in EMITTABLE:
        return MappedAction(
            actions=[{"action_type": "wait"}],
            invalid=True,
            reason="unmappable_action:%s" % (action or "empty"),
        )

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
        return MappedAction(
            actions=[
                {"action_type": "click", "x": pt[0], "y": pt[1]},
                {"action_type": "input_text", "text": text},
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
