"""Control module: screenshot, UI extraction (UIAutomator XML + optional OCR), and actions (tap, swipe, type, ...).

It wraps mobile-use's ADB `Environment` (screenshot and action execution) and adds what the agent needs on top:
  * `elements()`  the accessibility hierarchy from `uiautomator dump`, as element dicts (the shape DMA-2B was trained
                  on), retried because the dump is sometimes empty while the screen changes;
  * `ocr_labels()` text found by OCR inside elements that have no accessibility label (icons drawn as images, custom
                  views); optional, used only when the hierarchy leaves an interactive element unnamed;
  * `act()`       one high-level action (click, long_press, type, scroll, back, home, open_app, wait), retried once
                  if ADB fails.
"""
import io
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _bool(x) -> bool:
    return str(x).lower() == "true"


def parse_hierarchy(xml: str) -> List[Dict[str, Any]]:
    """`uiautomator dump` XML -> element dicts (text, content_description, class_name, bbox_pixels, is_* flags)."""
    xml = xml[xml.find("<?xml"):] if "<?xml" in xml else xml
    out = []
    for n in ET.fromstring(xml).iter("node"):
        m = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", n.get("bounds", ""))
        if not m:
            continue
        x1, y1, x2, y2 = map(int, m.groups())
        cls = n.get("class", "")
        out.append({"text": n.get("text") or None, "content_description": n.get("content-desc") or None,
                    "hint_text": n.get("hint") or None, "class_name": cls,
                    "bbox_pixels": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
                    "is_checkable": _bool(n.get("checkable")), "is_checked": _bool(n.get("checked")),
                    "is_clickable": _bool(n.get("clickable")), "is_editable": "EditText" in cls,
                    "is_enabled": _bool(n.get("enabled")), "is_focusable": _bool(n.get("focusable")),
                    "is_focused": _bool(n.get("focused")), "is_long_clickable": _bool(n.get("long-clickable")),
                    "is_scrollable": _bool(n.get("scrollable")), "is_selected": _bool(n.get("selected")),
                    "is_visible": True, "package_name": n.get("package") or None,
                    "resource_name": n.get("resource-id") or None})
    return out


def ocr_to_elements(result, package: Optional[str] = None, min_conf: float = 0.5) -> List[Dict[str, Any]]:
    """RapidOCR boxes ([[4 corner points], text, confidence], ...) -> clickable text elements in reading order."""
    out = []
    for box, text, conf in result or []:
        if float(conf) < min_conf or not str(text).strip():
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append({"text": str(text).strip(), "content_description": None, "hint_text": None,
                    "class_name": "android.widget.TextView",
                    "bbox_pixels": {"x_min": int(min(xs)), "y_min": int(min(ys)), "x_max": int(max(xs)),
                                    "y_max": int(max(ys))},
                    "is_clickable": True, "is_editable": False, "is_enabled": True, "is_visible": True,
                    "package_name": package, "text_source": "ocr"})
    out.sort(key=lambda e: (e["bbox_pixels"]["y_min"] // 20, e["bbox_pixels"]["x_min"]))
    return out


def swipe_for_scroll(direction: str, W: int, H: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Start and end of the finger for a scroll: the content moves in `direction`, so the finger moves the other way."""
    cx, cy = W // 2, H // 2
    return {"down": ((cx, int(H * 0.7)), (cx, int(H * 0.3))), "up": ((cx, int(H * 0.3)), (cx, int(H * 0.7))),
            "right": ((int(W * 0.8), cy), (int(W * 0.2), cy)), "left": ((int(W * 0.2), cy), (int(W * 0.8), cy))}[direction]


class Device:
    def __init__(self, serial: str = "emulator-5554", adb_host: str = "127.0.0.1", adb_port: int = 5037,
                 wait_after_action_s: float = 1.5, packages: Optional[Dict[str, str]] = None, ocr: bool = False):
        from mobile_use.environment.mobile_environ import Environment   # ADB through mobile-use
        self.env = Environment(serial_no=serial, host=adb_host, port=adb_port, go_home=False,
                               wait_after_action_seconds=wait_after_action_s)
        self.d = self.env._d
        self.packages = {k.lower(): v for k, v in (packages or {}).items()}
        self._ocr = None
        self.fill_labels = ocr      # OCR labels for unnamed elements (the OCR fallback is always on)

    # ------------------------------------------------------------------ observation
    def screenshot(self):
        return self.env.get_state().pixels.convert("RGB")

    def current_package(self) -> str:
        return self.d.app_current().package

    def _u2(self):
        if getattr(self, "_u2dev", None) is None:
            try:
                import uiautomator2 as u2
                self._u2dev = u2.connect(self.d.serial)
            except Exception as exc:  # noqa: BLE001 - optional, the plain dump is the next source
                logger.warning("uiautomator2 unavailable: %s", exc)
                self._u2dev = False
        return self._u2dev or None

    def elements(self, tries: int = 3) -> List[Dict[str, Any]]:
        """The accessibility hierarchy, from the first source that works:
          1. uiautomator2 (reads the tree directly, ~0.1 s, also on screens that never become idle);
          2. `uiautomator dump` (fails with "could not get idle state" on animated screens);
          3. OCR text boxes on the screenshot."""
        dev = self._u2()
        if dev is not None:
            try:
                els = parse_hierarchy(dev.dump_hierarchy())
                if len(els) > 1:
                    return els
            except Exception as exc:  # noqa: BLE001
                logger.warning("uiautomator2 dump failed: %s", exc)
        for _ in range(tries):
            self.d.shell(["rm", "-f", "/sdcard/agent_ui.xml"])
            said = self.d.shell(["uiautomator", "dump", "/sdcard/agent_ui.xml"], timeout=20)
            if "idle state" in said:          # the screen never idles: retrying cannot help
                break
            xml = self.d.shell(["cat", "/sdcard/agent_ui.xml"])
            try:
                els = parse_hierarchy(xml) if "<?xml" in xml else []
            except ET.ParseError:
                els = []
            if len(els) > 1:
                return els
            time.sleep(1.0)
        logger.warning("UI hierarchy unavailable; falling back to OCR elements")
        return self.ocr_elements(self.screenshot())

    def _ocr_engine(self):
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr = RapidOCR()
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR unavailable: %s", exc)
                return None
        return self._ocr

    def ocr_elements(self, image) -> List[Dict[str, Any]]:
        engine = self._ocr_engine()
        if engine is None:
            return []
        import numpy as np
        result, _ = engine(np.array(image))
        return ocr_to_elements(result, self.current_package())

    def ocr_labels(self, image, elements: List[Dict[str, Any]]) -> int:
        """Fill `text` of unlabeled interactive elements from OCR boxes inside them. Returns how many were filled."""
        if not self.fill_labels or self._ocr_engine() is None:
            return 0
        import numpy as np
        result, _ = self._ocr(np.array(image))
        filled = 0
        for e in elements:
            if e.get("text") or e.get("content_description") or not (e["is_clickable"] or e["is_editable"]):
                continue
            b = e["bbox_pixels"]
            words = [t for box, t, conf in (result or []) if conf > 0.5 and
                     b["x_min"] <= sum(p[0] for p in box) / 4 <= b["x_max"] and
                     b["y_min"] <= sum(p[1] for p in box) / 4 <= b["y_max"]]
            if words:
                e["text"], e["text_source"] = " ".join(words)[:60], "ocr"
                filled += 1
        return filled

    # ------------------------------------------------------------------ actions
    def act(self, kind: str, **p) -> None:
        from mobile_use.schema.schema import Action
        W, H = self.screenshot().size if kind == "scroll" else (0, 0)
        if kind in ("click", "long_press"):
            a = [Action(name=kind, parameters={"coordinate": (int(p["x"]), int(p["y"]))})]
        elif kind == "type":
            a = [Action(name="click", parameters={"coordinate": (int(p["x"]), int(p["y"]))})] if "x" in p else []
            a.append(Action(name="type", parameters={"text": p["text"]}))
        elif kind == "scroll":
            s, e = swipe_for_scroll(p["direction"], W, H)
            a = [Action(name="swipe", parameters={"coordinate": s, "coordinate2": e})]
        elif kind == "back":
            a = [Action(name="press_back", parameters={})]
        elif kind == "home":
            a = [Action(name="press_home", parameters={})]
        elif kind == "open_app":
            name = p["app"]
            a = [Action(name="open", parameters={"text": self.packages.get(name.lower(), name)})]
        elif kind == "wait":
            a = [Action(name="wait", parameters={"time": p.get("seconds", 2.0)})]
        else:
            raise ValueError("unknown action %r" % kind)
        for action in a:
            for attempt in range(2):
                try:
                    self.env.execute_action(action)
                    break
                except Exception as exc:  # noqa: BLE001 - one retry, then the error goes to the caller
                    logger.warning("ADB action %s failed (%s), attempt %d", action.name, exc, attempt + 1)
                    if attempt:
                        raise
                    time.sleep(1.0)
