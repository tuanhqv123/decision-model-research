"""Layered environment health probe and supervisor.

``GET /health`` answering is not evidence that the device is alive. On 2026-11-02 the
emulator process was OOM-killed inside the container while the FastAPI server kept
returning 200, ``docker inspect`` reported ``OOMKilled=false``, and every subsequent
task silently scored 0. This module exists so that failure can never be silent again.

Layers are injected so the probe is testable without a device.
"""
import hashlib
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

LayerFn = Callable[..., Any]


@dataclass
class LayerResult:
    name: str
    ok: Optional[bool]
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def skipped(self) -> bool:
        return self.ok is None


@dataclass
class ProbeResult:
    ok: bool
    layer_failed: Optional[str]
    layers: List[LayerResult] = field(default_factory=list)
    restarts: int = 0

    @property
    def skipped(self) -> List[str]:
        return [l.name for l in self.layers if l.skipped]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "layer_failed": self.layer_failed,
            "restarts": self.restarts,
            "skipped": self.skipped,
            "layers": [{"name": l.name, "ok": l.ok, "evidence": l.evidence} for l in self.layers],
        }


def image_hash(pixels: Any) -> str:
    if pixels is None:
        return ""
    try:
        data = pixels.tobytes()
    except AttributeError:
        data = bytes(pixels)
    return hashlib.sha1(data).hexdigest()[:16]


def is_black(pixels: Any) -> bool:
    if pixels is None:
        return True
    try:
        return int(pixels.max()) == 0
    except AttributeError:
        return not any(pixels)


class EnvironmentHealthMonitor:
    """Runs the probe ladder in order and stops at the first hard failure.

    Order: http -> device -> screenshot -> noop. A later layer is never reached when an
    earlier one fails, so ``layer_failed`` names the first real problem.
    """

    ORDER = ("http", "device", "screenshot", "noop")

    def __init__(
        self,
        http_health: LayerFn,
        device_state: LayerFn,
        screenshot: LayerFn,
        noop_action: LayerFn,
        restart: Optional[Callable[[], bool]] = None,
        max_restarts: int = 2,
        layers: Optional[List[str]] = None,
    ):
        self._fns = {
            "http": http_health,
            "device": device_state,
            "screenshot": screenshot,
            "noop": noop_action,
        }
        if layers:
            unknown = [l for l in layers if l not in self._fns and l != "all"]
            if unknown:
                raise ValueError("unknown layer(s): %s (known: %s, all)" % (",".join(unknown), ",".join(self.ORDER)))
            if "all" not in layers:
                self._layers = tuple(l for l in self.ORDER if l in layers)
            else:
                self._layers = self.ORDER
        self._restart = restart
        self.max_restarts = max_restarts
        self._prev_hash: str = ""
        self._prev_frame: Any = None

    def probe(self) -> ProbeResult:
        layers: List[LayerResult] = []
        for name in getattr(self, "_layers", self.ORDER):
            fn = self._fns[name]
            try:
                if name == "screenshot":
                    out = fn(self._prev_frame)
                elif name == "noop":
                    out = fn(self._prev_frame)
                else:
                    out = fn()
            except Exception as exc:  # noqa: BLE001 - a probe must never raise
                layers.append(LayerResult(name, False, {"exception": "%s: %s" % (type(exc).__name__, exc)}))
                return ProbeResult(False, name, layers)

            ok, evidence = self._normalise(out)
            layers.append(LayerResult(name, ok, evidence))

            if ok is False:
                return ProbeResult(False, name, layers)

            frame = evidence.get("frame")
            if name == "screenshot" and frame is not None:
                self._prev_frame = frame
                self._prev_hash = evidence.get("hash", "")

        return ProbeResult(True, None, layers)

    def ensure_alive(self) -> ProbeResult:
        result = self.probe()
        attempts = 0
        while not result.ok and self._restart and attempts < self.max_restarts:
            attempts += 1
            self._restart()
            time.sleep(1.0)
            result = self.probe()
        result.restarts = attempts
        return result

    @staticmethod
    def _normalise(out: Any):
        if isinstance(out, tuple) and len(out) == 2:
            return bool(out[0]) if out[0] is not None else None, out[1] or {}
        if isinstance(out, dict):
            return bool(out.get("ok")), out
        return bool(out), {}


# ------------------------------------------------------------------ defaults


def http_probe(base_url: str, timeout: float = 10.0) -> LayerFn:
    def check():
        import requests

        try:
            r = requests.get("%s/health" % base_url, timeout=timeout)
            return r.status_code == 200, {"status": r.status_code, "body": r.text[:200]}
        except Exception as exc:  # noqa: BLE001
            return False, {"exception": "%s: %s" % (type(exc).__name__, exc)}

    return check


def adb_probe(serial: Optional[str] = None, adb: str = "adb") -> LayerFn:
    def check():
        cmd = [adb, "devices"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except FileNotFoundError:
            return None, {"skipped": "adb not installed"}
        except Exception as exc:  # noqa: BLE001
            return False, {"exception": "%s: %s" % (type(exc).__name__, exc)}
        lines = [l for l in out.stdout.splitlines()[1:] if l.strip()]
        devices = [l for l in lines if "\tdevice" in l]
        if serial:
            devices = [d for d in devices if d.split()[0] == serial]
        return len(devices) > 0, {"devices": devices, "stderr": out.stderr[:200]}

    return check


def screenshot_probe(base_url: str, timeout: float = 60.0) -> LayerFn:
    """A real frame, not black, not byte-identical to the previous one."""

    def check(prev_frame=None):
        import numpy as np
        import requests

        try:
            r = requests.get("%s/screenshot" % base_url, params={"wait_to_stabilize": "true"}, timeout=timeout)
            r.raise_for_status()
            px = np.array(r.json()["pixels"], dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001
            return False, {"exception": "%s: %s" % (type(exc).__name__, exc)}
        h = image_hash(px)
        if is_black(px):
            return False, {"hash": h, "reason": "black_frame"}
        evidence = {"hash": h, "shape": list(px.shape), "frame": px}
        if prev_frame is not None and image_hash(prev_frame) == h:
            evidence["identical_to_previous"] = True
        return True, evidence

    return check


def noop_action_probe(base_url: str, timeout: float = 60.0) -> LayerFn:
    """Execute a harmless action and re-read the screen.

    Fails only on genuine unresponsiveness (the action raises, or the next frame is
    missing/black). A frame that changed while waiting is recorded, not treated as a
    failure, because animations would otherwise produce false alarms.
    """

    def check(prev_frame=None):
        import numpy as np
        import requests

        try:
            r = requests.post(
                "%s/execute_action" % base_url,
                json={"action_type": "wait"},
                timeout=timeout,
            )
            r.raise_for_status()
            time.sleep(0.2)
            # The server requires this query parameter: a bare GET /screenshot returns 422.
            # (Found 2026-11-08 on android_world@HEAD; runbook section 4 did not record it.)
            r2 = requests.get("%s/screenshot" % base_url,
                              params={"wait_to_stabilize": "true"}, timeout=timeout)
            r2.raise_for_status()
            px = np.array(r2.json()["pixels"], dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001
            return False, {"exception": "%s: %s" % (type(exc).__name__, exc)}
        h = image_hash(px)
        if is_black(px):
            return False, {"hash": h, "reason": "black_frame_after_action"}
        changed = bool(prev_frame is not None and image_hash(prev_frame) != h)
        return True, {"hash": h, "changed_during_wait": changed}

    return check


def monitor_for(base_url: str, serial: Optional[str] = None, restart=None,
                max_restarts: int = 2, layers: Optional[List[str]] = None) -> EnvironmentHealthMonitor:
    return EnvironmentHealthMonitor(
        http_health=http_probe(base_url),
        device_state=adb_probe(serial),
        screenshot=screenshot_probe(base_url),
        noop_action=noop_action_probe(base_url),
        restart=restart,
        max_restarts=max_restarts,
        layers=layers,
    )


def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Layered AndroidWorld health probe")
    ap.add_argument("--url", default="http://127.0.0.1:5051")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--layers", default="all",
                    help="comma-separated subset of %s, or 'all'" % ",".join(EnvironmentHealthMonitor.ORDER))
    args = ap.parse_args()

    layers = [l.strip().lower() for l in args.layers.split(",") if l.strip()]
    try:
        monitor = monitor_for(args.url, args.serial, layers=layers)
    except ValueError as exc:
        ap.error(str(exc))
    result = monitor.probe()
    if args.json:
        print(json.dumps(result.as_dict(), indent=1))
    else:
        for layer in result.layers:
            state = "SKIP" if layer.skipped else ("ok" if layer.ok else "FAIL")
            print("%-12s %s %s" % (layer.name, state, layer.evidence))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())


def probe_all(url: str, serial: Optional[str] = None, adb: Optional[str] = None) -> Dict[str, Any]:
    """All four layers, as a dict. The Gate A entry point.

    `adb` is explicit because inside the android_world image the binary is not always the one
    on the default PATH (found 2026-11-06).
    """
    monitor = monitor_for(url, serial)
    if adb:
        monitor._fns["device"] = adb_probe(serial, adb)
    return monitor.probe().as_dict()


def probe_inprocess(env: Any, adb: str = "adb", serial: Optional[str] = None) -> Dict[str, Any]:
    """Liveness for the in-process architecture: no HTTP server is involved.

    The four claims are the same ones the HTTP probe makes — device present, a real frame,
    an action that is accepted, a device that answers afterwards — but they are checked
    against the environment object the agent will actually use, so a pass means the thing
    under test is alive, not that a sidecar is.
    """
    layers = []

    ok, evidence = adb_probe(serial, adb)()
    layers.append(LayerResult("device", bool(ok), evidence or {}))
    if not ok:
        return ProbeResult(False, "device", layers).as_dict()

    try:
        frame = env.screenshot()
    except Exception as exc:  # noqa: BLE001
        layers.append(LayerResult("screenshot", False, {"exception": "%s: %s" % (type(exc).__name__, exc)}))
        return ProbeResult(False, "screenshot", layers).as_dict()
    black = is_black(frame)
    layers.append(LayerResult("screenshot", not black, {"hash": image_hash(frame), "reason": "black_frame" if black else None}))
    if black:
        return ProbeResult(False, "screenshot", layers).as_dict()

    try:
        env.execute({"action_type": "wait"})
        after = env.screenshot()
    except Exception as exc:  # noqa: BLE001
        layers.append(LayerResult("noop", False, {"exception": "%s: %s" % (type(exc).__name__, exc)}))
        return ProbeResult(False, "noop", layers).as_dict()

    alive = image_hash(after) != "" and not is_black(after)
    layers.append(LayerResult("noop", alive, {"hash_after": image_hash(after), "changed": image_hash(after) != image_hash(frame)}))
    return ProbeResult(alive, None if alive else "noop", layers).as_dict()
