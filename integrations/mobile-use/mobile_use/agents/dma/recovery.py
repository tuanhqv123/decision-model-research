"""Self-correction: unexpected pop-ups and stuck states.

`detect_popup` recognises the system dialogs that interrupt a task, from the element list alone, and returns the
element to press. They are handled before the planner or the executor sees the screen, because neither was asked about
them and both tend to treat them as part of the task:
  app_not_responding  "X isn't responding" -> Wait (a second time on the same app -> Close app)
  app_crashed         "X has stopped" / "keeps stopping" -> Close app / OK
  permission          the Android permission controller -> "While using the app" / "Allow" (only if allowed by policy)
  system_alert        a one-button system notice ("OK", "Got it") owned by the system UI
Everything else is left to the planner, which sees "BLOCKED" or "no change" reports and re-plans (`StuckDetector`).
"""
from typing import Any, Dict, List, Optional

PERMISSION_PACKAGES = ("com.google.android.permissioncontroller", "com.android.permissioncontroller")
SYSTEM_PACKAGES = ("android", "com.android.systemui")


def _label(e: Dict[str, Any]) -> str:
    return " ".join(str(e.get(k) or "") for k in ("text", "content_description")).strip()


def _find(elements, *names):
    for name in names:
        for e in elements:
            if _label(e).lower() == name.lower() and (e.get("is_clickable") or e.get("class_name", "").endswith("Button")):
                return e
    return None


def centre(e) -> Dict[str, int]:
    b = e["bbox_pixels"]
    return {"x": (b["x_min"] + b["x_max"]) // 2, "y": (b["y_min"] + b["y_max"]) // 2}


def detect_popup(elements: List[Dict[str, Any]], seen_anr: int = 0,
                 allow_permissions: bool = True) -> Optional[Dict[str, Any]]:
    """{"kind", "press": element, "reason"} for a known interrupting dialog, else None."""
    text = " ".join(_label(e) for e in elements).lower()
    pkgs = {e.get("package_name") for e in elements}
    if "isn't responding" in text or "is not responding" in text:
        target = _find(elements, "Close app") if seen_anr else _find(elements, "Wait")
        target = target or _find(elements, "Close app", "OK")
        if target:
            return {"kind": "app_not_responding", "press": target,
                    "reason": "an app stopped responding; pressing %r" % _label(target)}
    if "has stopped" in text or "keeps stopping" in text:
        target = _find(elements, "Close app", "OK", "Close")
        if target:
            return {"kind": "app_crashed", "press": target, "reason": "an app crashed; pressing %r" % _label(target)}
    if pkgs & set(PERMISSION_PACKAGES):
        target = _find(elements, "While using the app", "Allow", "Only this time") if allow_permissions else \
            _find(elements, "Don't allow", "Deny")
        if target:
            return {"kind": "permission", "press": target,
                    "reason": "permission dialog; pressing %r (policy allow_permissions=%s)" % (_label(target), allow_permissions)}
    if pkgs and pkgs <= set(SYSTEM_PACKAGES) | {None}:
        buttons = [e for e in elements if e.get("is_clickable") and _label(e)]
        if len(buttons) == 1 and _label(buttons[0]).lower() in ("ok", "got it", "dismiss"):
            return {"kind": "system_alert", "press": buttons[0], "reason": "system notice; pressing %r" % _label(buttons[0])}
    return None


class StuckDetector:
    """Counts consecutive executed actions that changed nothing on screen."""

    def __init__(self, limit: int = 2):
        self.limit, self.count = limit, 0

    def update(self, changed: Optional[bool]) -> bool:
        self.count = 0 if changed else self.count + 1
        return self.count >= self.limit
