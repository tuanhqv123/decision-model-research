"""Bind AndroidWorld's actual environment API to the one the runner expects.

AndroidWorld's `AsyncAndroidEnv` is synchronous but shaped differently from this harness:
`get_state(wait_to_stabilize=...)` returns a `State(pixels, forest, ui_elements, auxiliaries)`,
`execute_action` wants a `JSONAction`, and liveness is adb's problem rather than the object's.

The shim is deliberately thin and assertion-free: it does no scoring, no retries and no
interpretation, so a failure recorded above it is a model or executor failure, not a shim
decision. Its behaviour on the real emulator is tested by `test_device_gate_a.py`; the one rule
that needs no emulator — one `State` per observation (H5) — is pinned by
`tests/test_bridge_observation.py`.
"""

from typing import Any, Dict, List, Optional


def _port_is_serving(port: int) -> bool:
    import socket

    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _to_json_action(action: Dict[str, Any]):
    from android_world.env.json_action import JSONAction

    known = {"action_type", "index", "x", "y", "text", "direction", "goal_status",
             "app_name", "keycode", "clear_text"}
    payload = {k: v for k, v in action.items() if k in known}
    return JSONAction(**payload)


def _element_dicts(state: Any) -> List[Dict[str, Any]]:
    """UI elements in a flat form the adapter and the grounding proxy can read.

    The VA track does not merely need a rectangle: `ac_common.pick_candidates` keeps an element
    only if `is_visible` is true AND it is labelled or actionable, and it reads `bbox_pixels` as a
    DICT with `width`/`height`. The old version sent `bbox` as a LIST and a single boolean
    `clickable`, so every element was dropped twice over on the server side and `track: VA`
    answered HTTP 500 with `KeyError: target_el` (measured 2026-11-10). Send the flags and the
    geometry in the form the consumer reads, and keep the flat `bbox`/`clickable` keys because
    the adapter and the trace already use them.
    """
    out = []
    for i, el in enumerate(getattr(state, "ui_elements", None) or []):
        bbox = getattr(el, "bbox_pixels", None)
        box = None
        box_dict = None
        if bbox is not None:
            box = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
            box_dict = {
                "x_min": int(bbox.x_min), "y_min": int(bbox.y_min),
                "x_max": int(bbox.x_max), "y_max": int(bbox.y_max),
                "width": int(bbox.x_max - bbox.x_min),
                "height": int(bbox.y_max - bbox.y_min),
                "center": [(bbox.x_min + bbox.x_max) / 2.0, (bbox.y_min + bbox.y_max) / 2.0],
            }
        out.append(
            {
                "id": i,
                "text": getattr(el, "text", None) or "",
                "content_description": getattr(el, "content_description", None) or "",
                "class": getattr(el, "class_name", None) or "",
                "package_name": getattr(el, "package_name", None) or "",
                "clickable": bool(getattr(el, "is_clickable", False)),
                "bbox": box,
                # the keys pick_candidates actually reads
                "bbox_pixels": box_dict,
                "is_visible": bool(getattr(el, "is_visible", True)),
                "is_clickable": bool(getattr(el, "is_clickable", False)),
                "is_editable": bool(getattr(el, "is_editable", False)),
                "is_scrollable": bool(getattr(el, "is_scrollable", False)),
                "hint_text": getattr(el, "hint_text", None) or "",
                # State flags (Q3, PLAN-2026-11-10). They were dropped here, so a switch that
                # flipped `is_checked` looked like an action that did nothing (F2's toggle
                # blindness) and no trace could say whether a click landed on a checkable widget.
                # Measurement only: `pick_candidates` reads none of them, so the VA state the
                # model receives is unchanged.
                "is_checked": _flag(el, "is_checked"),
                "is_checkable": _flag(el, "is_checkable"),
                "is_selected": _flag(el, "is_selected"),
                "is_focused": _flag(el, "is_focused"),
                "is_long_clickable": _flag(el, "is_long_clickable"),
                # an element whose enabled state AndroidWorld did not report is not claimed disabled
                "is_enabled": _flag(el, "is_enabled", default=True),
                "resource_name": getattr(el, "resource_name", None) or "",
            }
        )
    return out


def _flag(el: Any, name: str, default: bool = False) -> bool:
    """An AndroidWorld `UIElement` boolean; upstream leaves unset fields as None."""
    value = getattr(el, name, None)
    return default if value is None else bool(value)


class AndroidWorldEnv:
    """Synchronous, runner-shaped view of an AndroidWorld environment."""

    def __init__(self, env: Any, adb_path: Optional[str] = None):
        self._env = env
        self._last_state = None
        self._obs = None  # the ONE State the current step is observed through (H5)
        self._adb = adb_path

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        self._obs = None
        self._last_state = self._env.reset(go_home=True)
        # Every official agent hides the pointer-location overlay right after reset
        # (m3a.py:365, t3a.py:295, seeact.py:132). Until 2026-11-10 we never did, so each screenshot
        # carried a coordinate bar and touch trails that no official agent sees (H10).
        hide = getattr(self._env, "hide_automation_ui", None)
        if callable(hide):
            hide()

    # AndroidWorld's own `is_successful(env)` implementations call `env.get_state()` directly (and
    # occasionally other members of the wrapped environment). Without this forwarding every checker
    # that reads state raised `AttributeError: 'AndroidWorldEnv' object has no attribute 'get_state'`,
    # the runner recorded `success=False`, and 12 episodes that RAN to completion were silently scored
    # as model failures — 6 folded into "timeout" and 6 into "premature status" (found 2026-11-09).
    # Delegation is the honest default: anything this wrapper does not define is the wrapped env's.
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)

    def close(self) -> None:
        closer = getattr(self._env, "close", None)
        if callable(closer):
            closer()

    @property
    def controller(self):
        """AndroidWorld's raw controller: task setup uses it directly (datetime, a11y, packages)."""
        return getattr(self._env, "controller", None)

    # ---------------------------------------------------------------- observation
    def _state(self, stabilize: bool = False) -> Any:
        state = self._env.get_state(wait_to_stabilize=stabilize)
        self._last_state = state
        return state

    def _observation(self) -> Any:
        """One stabilised `State` per step: pixels and ui_elements of the SAME moment.

        Upstream agents read a single `State` per step. Until 2026-11-10 `screenshot()` (stabilised)
        and `elements()` (not) each called `get_state()`, so the model could get the pixels of one
        screen and the elements of another. `execute()` and `reset()` drop the cache, so the next
        read after an action is a fresh observation — never the previous screen.
        """
        if self._obs is None:
            self._obs = self._state(stabilize=True)
        return self._obs

    def screenshot(self):
        self._obs = None  # a screenshot request starts a new observation
        return self._observation().pixels

    def elements(self) -> List[Dict[str, Any]]:
        return _element_dicts(self._observation())

    def foreground(self) -> str:
        """Foreground activity name: the only honest way to check that `open_app` landed."""
        value = getattr(self._env, "foreground_activity_name", "")
        return value() if callable(value) else str(value or "")

    def screen_size(self):
        state = self._state()
        pixels = getattr(state, "pixels", None)
        if pixels is None:
            return None
        return int(pixels.shape[1]), int(pixels.shape[0])

    # ---------------------------------------------------------------- action
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one mapped action. Exceptions propagate: the agent records them."""
        self._obs = None
        self._env.execute_action(_to_json_action(action))
        return {"ok": True, "error": None}

    # ---------------------------------------------------------------- health
    def is_alive(self) -> bool:
        try:
            state = self._env.get_state(wait_to_stabilize=False)
        except Exception:
            return False
        pixels = getattr(state, "pixels", None)
        return pixels is not None and getattr(pixels, "size", 0) > 0


def detect_console_port(adb_path: str = "adb") -> Optional[int]:
    """Find the running emulator's console port instead of assuming 5554.

    With host networking the emulator takes the next free port when 5554 is held by a
    leftover process, and a hardcoded port then fails with "device not found" while the
    emulator is perfectly healthy two ports up (found 2026-11-06).
    """
    import re
    import subprocess

    try:
        out = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        match = re.match(r"emulator-(\d+)\s+device", line.strip())
        if match:
            return int(match.group(1))
    return None


def build(console_port: Optional[int] = 5554, adb_path: str = "adb", emulator_setup: bool = False) -> AndroidWorldEnv:
    from android_world.env import env_launcher

    if console_port in (None, 0):
        console_port = detect_console_port(adb_path) or 5554
    elif not _port_is_serving(console_port):
        console_port = detect_console_port(adb_path) or console_port

    env = env_launcher.load_and_setup_env(
        console_port=console_port,
        emulator_setup=emulator_setup,
        adb_path=adb_path,
    )
    return AndroidWorldEnv(env, adb_path=adb_path)
