"""The probe ladder must name the layer that failed, and must not raise.

The failure this guards against: the emulator process OOM-killed while ``/health``
still returned 200, so tasks silently scored 0.
"""
import numpy as np
import pytest

from awenv.liveness import (
    EnvironmentHealthMonitor,
    image_hash,
    is_black,
)


def _ok(_prev=None):
    return True, {}


def _frame(prev=None):
    px = np.zeros((4, 4, 3), dtype=np.uint8)
    px[0, 0, 0] = 1
    return True, {"hash": image_hash(px), "frame": px}


def _monitor(handlers, restart=None):
    defaults = {
        "http": _ok,
        "device": _ok,
        "screenshot": _frame,
        "noop": _ok,
    }
    defaults.update(handlers)
    return EnvironmentHealthMonitor(
        http_health=defaults["http"],
        device_state=defaults["device"],
        screenshot=defaults["screenshot"],
        noop_action=defaults["noop"],
        restart=restart,
    )


def test_all_layers_pass():
    result = _monitor({}).probe()
    assert result.ok is True
    assert result.layer_failed is None
    assert [l.name for l in result.layers] == ["http", "device", "screenshot", "noop"]
    assert result.skipped == []


def test_dead_device_stops_the_ladder_at_the_named_layer():
    calls = []

    def screenshot(prev=None):
        calls.append("screenshot")
        return _frame(prev)

    result = _monitor({"device": lambda: (False, {"devices": []}), "screenshot": screenshot}).probe()
    assert result.ok is False
    assert result.layer_failed == "device"
    assert calls == [], "a later layer must not run after a hard failure"


def test_health_alone_is_not_enough():
    def dead_device():
        return False, {"devices": [], "note": "adbd not running"}

    result = _monitor({"https": _ok, "device": dead_device}).probe()
    assert result.ok is False
    assert result.layer_failed == "device"


def test_black_frame_fails_the_screenshot_layer():
    """A probe that returns ``ok=0`` for a black frame must stop the ladder there."""

    def black(prev=None):
        px = np.zeros((4, 4, 3), dtype=np.uint8)
        return False, {"hash": image_hash(px), "reason": "black_frame"}

    result = _monitor({"screenshot": black}).probe()
    assert result.ok is False
    assert result.layer_failed == "screenshot"
    assert result.layers[-1].evidence["reason"] == "black_frame"


def test_exception_in_a_probe_fails_that_layer_not_the_process():
    def boom(prev=None):
        raise ConnectionError("device 'emulator-5554' not found")

    result = _monitor({"noop": boom}).probe()
    assert result.ok is False
    assert result.layer_failed == "noop"
    assert "ConnectionError" in result.layers[-1].evidence["exception"]


def test_skipped_layer_is_declared_not_hidden():
    result = _monitor({"device": lambda: (None, {"skipped": "adb not installed"})}).probe()
    assert result.ok is True
    assert result.skipped == ["device"]


def test_ensure_alive_restarts_then_recovers():
    state = {"alive": False, "restarts": 0}

    def device():
        return state["alive"], {"devices": ["emulator-5554"] if state["alive"] else []}

    def restart():
        state["restarts"] += 1
        state["alive"] = True
        return True

    result = _monitor({"device": device}, restart=restart).ensure_alive()
    assert state["restarts"] == 1
    assert result.ok is True
    assert result.restarts == 1


def test_ensure_alive_gives_up_after_max_restarts():
    def device():
        return False, {}

    result = EnvironmentHealthMonitor(
        http_health=_ok,
        device_state=device,
        screenshot=_frame,
        noop_action=_ok,
        restart=lambda: False,
        max_restarts=2,
    ).ensure_alive()
    assert result.ok is False
    assert result.restarts == 2


def test_black_frame_detection():
    assert is_black(np.zeros((2, 2, 3), dtype=np.uint8)) is True
    assert is_black(np.ones((2, 2, 3), dtype=np.uint8)) is False
    assert is_black(None) is True


def test_image_hash_is_stable_and_discriminating():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 1] = 9
    assert image_hash(a) == image_hash(a.copy())
    assert image_hash(a) != image_hash(b)


def test_layer_filter_runs_only_the_requested_layers():
    calls = []

    def record(name, ok=True):
        def fn(prev=None):
            calls.append(name)
            return ok, {"name": name}

        return fn

    from awenv.liveness import EnvironmentHealthMonitor

    monitor = EnvironmentHealthMonitor(
        http_health=record("http"),
        device_state=record("device"),
        screenshot=record("screenshot"),
        noop_action=record("noop"),
        layers=["http", "noop"],
    )
    result = monitor.probe()

    assert result.ok is True
    assert calls == ["http", "noop"]


def test_unknown_layer_name_is_rejected_loudly():
    import pytest

    from awenv.liveness import monitor_for

    with pytest.raises(ValueError, match="unknown layer"):
        monitor_for("http://127.0.0.1:9", layers=["http", "telepathy"])


def test_noop_probe_sends_the_required_screenshot_parameter():
    """Regression: android_world@HEAD returns 422 for a bare GET /screenshot (found 2026-11-08).

    The noop layer re-reads the screen after the action; if it forgets `wait_to_stabilize` every live gate run
    reports a dead device while the device is fine.
    """
    from awenv import liveness
    import inspect

    src = inspect.getsource(liveness.noop_action_probe)
    assert "wait_to_stabilize" in src, "noop probe must pass wait_to_stabilize or the server returns 422"
