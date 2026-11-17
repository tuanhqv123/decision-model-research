"""Control and monitoring dashboard for DMAgent.

    python -m mobile_use.dma_dashboard --apps-json app_list.json --port 7870

Open http://<host>:7870 and set the three endpoints in Settings (the phone's ADB serial or host:port, the DMA-2B API,
the planner LLM); each can be tested from there. They are saved to --settings (default ~/.dmagent.json, readable by
its owner only) and loaded at the next start. They can also be given at start, as flags or environment variables
(DEVICE, DMA_URL, DMA_API_TOKEN, LLM_URL, LLM_KEY, LLM_MODEL); values saved from the page take precedence.
Then type a task and start it; the page shows the live screen (a stream of a few frames per second, with the agent's
last tap marked), every planner thought and local goal, every DMA-2B action with its probability (click one for its
decision card: the screen it read with the chosen element boxed, the question, the answer, each action's probability), recoveries from pop-ups, reflection checks, and the
results of earlier tasks. When the agent needs a person (a sensitive action, a low-confidence action or a question), an
approval panel appears; the run waits until it is answered. "Stop" ends the run after the current action. While no
task runs, clicking the screen taps the device at that point, and Back / Home press those keys.
"""
import argparse
import io
import json
import os
import threading
import time

import uvicorn
from typing import Callable, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from PIL import ImageDraw
from pydantic import BaseModel

from mobile_use.agents.dma.safety import Policy, QueueApprover
from mobile_use.agents.dma_agent import DMAgent

PAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dma_dashboard.html")


class Start(BaseModel):
    task: str


class Tap(BaseModel):
    x: float      # 0..1 of the image width
    y: float      # 0..1 of the image height


class Key(BaseModel):
    key: str      # "back" | "home"


class Answer(BaseModel):
    id: int
    approved: bool
    note: str = ""


SECRETS = ("dma_token", "llm_key")
REQUIRED = ("device", "dma_url", "llm_url", "llm_model")
FIELDS = REQUIRED + SECRETS + ("apps_json", "confidence", "max_steps")


class Config(BaseModel):
    device: Optional[str] = None
    dma_url: Optional[str] = None
    dma_token: Optional[str] = None
    llm_url: Optional[str] = None
    llm_key: Optional[str] = None
    llm_model: Optional[str] = None
    confidence: Optional[float] = None
    max_steps: Optional[int] = None


def public(settings: dict) -> dict:
    """The settings as the page may see them: every endpoint, never a key (only whether one is set)."""
    out = {k: settings.get(k) for k in FIELDS if k not in SECRETS}
    out.update({k + "_set": bool(settings.get(k)) for k in SECRETS})
    return out


def missing(settings: dict) -> List[str]:
    return [k for k in REQUIRED if not settings.get(k)]


def load_settings(path: Optional[str], base: dict) -> dict:
    """The start values, overridden by what was last saved from the page."""
    out = {k: v for k, v in base.items() if v not in (None, "")}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            out.update({k: v for k, v in json.load(f).items() if k in FIELDS and v not in (None, "")})
    return out


def save_settings(path: str, settings: dict):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)    # holds keys: owner only
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({k: settings.get(k) for k in FIELDS if settings.get(k) is not None}, f, indent=1)


def info_of(settings: dict, info: Optional[dict] = None) -> dict:
    out = dict(info or {})
    out.update({"serial": settings.get("device"), "llm_model": settings.get("llm_model"),
                "confidence": settings.get("confidence", out.get("confidence")),
                "max_steps": settings.get("max_steps", out.get("max_steps"))})
    return {k: v for k, v in out.items() if v is not None}


def build_app(agent: Optional[DMAgent], approver: QueueApprover, info: dict = None, settings: dict = None,
              make_agent: Optional[Callable[[dict], DMAgent]] = None, settings_file: Optional[str] = None,
              check: Optional[Callable[[dict], dict]] = None) -> FastAPI:
    """`info`: the setup the page shows (device serial, LLM model, where DMA-2B runs, the ask threshold).
    `settings`: the endpoints; `make_agent(settings)` builds an agent for new ones (saved to `settings_file`);
    `check(settings)` tests each endpoint. `agent` is None until the endpoints are set."""
    app = FastAPI(title="DMA-2B mobile agent")
    state = {"thread": None, "summary": None, "runs": []}
    lock = threading.Lock()
    cfg = {"settings": dict(settings or {}), "info": info_of(settings, info) if settings else dict(info or {})}
    box = {"agent": agent}

    class _Current:   # every endpoint reaches the agent of the current settings
        def __getattr__(self, name):
            if box["agent"] is None:
                raise HTTPException(409, "set the endpoints in Settings first")
            return getattr(box["agent"], name)
    agent = _Current()

    def worker(task):
        a = box["agent"]                      # the settings cannot change while it runs
        try:
            for _ in a.iter_run(task):
                pass
        except Exception as exc:  # noqa: BLE001 - show the failure instead of a dead page
            a.outcome = "error: %s" % exc
        state["summary"] = dict(a.summary(), id=len(state["runs"]))
        state["runs"].append({"summary": state["summary"], "events": list(getattr(a, "events", []))})

    @app.get("/", response_class=HTMLResponse)
    def page():
        with open(PAGE_FILE, encoding="utf-8") as f:
            return f.read()

    @app.post("/api/start")
    def start(req: Start):
        with lock:
            if box["agent"] is None:
                return {"ok": False, "error": "set the endpoints in Settings first"}
            if state["thread"] and state["thread"].is_alive():
                return {"ok": False, "error": "a task is running"}
            state["summary"] = None
            state["thread"] = threading.Thread(target=worker, args=(req.task.strip(),), daemon=True)
            state["thread"].start()
            return {"ok": True}

    @app.post("/api/stop")
    def stop():
        agent.policy.stop.set()
        return {"ok": True}

    @app.post("/api/answer")
    def answer(req: Answer):
        return {"ok": approver.answer(req.id, req.approved, req.note)}

    @app.get("/api/state")
    def get_state():
        running = bool(state["thread"] and state["thread"].is_alive())
        events = list(getattr(box["agent"], "events", []))[-200:]
        return {"running": running, "events": events, "pending": approver.pending,
                "summary": state["summary"], "runs": [r["summary"] for r in state["runs"][-20:]],
                "info": cfg["info"], "configured": box["agent"] is not None,
                "missing": missing(cfg["settings"]) if make_agent else []}

    @app.get("/api/config")
    def get_config():
        return public(cfg["settings"])

    @app.post("/api/config")
    def set_config(req: Config):
        """New endpoints: a key left empty keeps the saved one. Refused while a task runs."""
        with lock:
            if busy():
                return {"ok": False, "error": "a task is running"}
            if make_agent is None:
                return {"ok": False, "error": "this dashboard was started without settings"}
            new = dict(cfg["settings"])
            new.update({k: (v.strip() if isinstance(v, str) else v) for k, v in req.model_dump().items()
                        if v not in (None, "")})
            if missing(new):
                cfg["settings"] = new
                return {"ok": False, "error": "missing: " + ", ".join(missing(new))}
            try:
                fresh = make_agent(new)
            except Exception as exc:  # noqa: BLE001 - report it on the page; the old agent stays
                return {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
            box["agent"], cfg["settings"] = fresh, new
            cfg["info"] = info_of(new, cfg["info"])
            if settings_file:
                save_settings(settings_file, new)
            return {"ok": True, "config": public(new)}

    @app.post("/api/config/test")
    def test_config(req: Config):
        """Reach each endpoint with the given values (empty ones: the saved values)."""
        trial = dict(cfg["settings"])
        trial.update({k: v for k, v in req.model_dump().items() if v not in (None, "")})
        out = (check or check_endpoints)(trial)
        if (out.get("dma") or {}).get("device"):
            cfg["info"]["dma_device"] = out["dma"]["device"]
        return out

    @app.get("/api/runs/{i}")
    def get_run(i: int):
        if not 0 <= i < len(state["runs"]):
            raise HTTPException(404, "no such run")
        return state["runs"][i]

    def busy():
        return bool(state["thread"] and state["thread"].is_alive())

    @app.get("/api/frames/{n}.jpg")
    def frame(n: int):
        """The screen a DMA-2B decision was made on (its card in the chat shows it with the chosen element boxed)."""
        jpg = getattr(box["agent"], "frames", {}).get(n)
        if jpg is None:
            raise HTTPException(404, "no such frame")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    def marked(img):
        """The frame with the agent's last tap circled, while a task runs."""
        ev = next((e for e in reversed(getattr(box["agent"], "events", [])) if e.get("point")), None)
        if busy() and ev:
            img = img.copy()
            x, y = ev["point"]
            ImageDraw.Draw(img).ellipse([x - 40, y - 40, x + 40, y + 40], outline=(27, 108, 128), width=10)
        return img

    @app.post("/api/tap")
    def tap(req: Tap):
        if busy():
            return {"ok": False, "error": "a task is running"}
        w, h = agent.device.screenshot().size
        x, y = int(round(min(max(req.x, 0.0), 1.0) * w)), int(round(min(max(req.y, 0.0), 1.0) * h))
        agent.device.act("click", x=x, y=y)
        return {"ok": True, "x": x, "y": y}

    @app.post("/api/key")
    def press(req: Key):
        if busy() or req.key not in ("back", "home"):
            return {"ok": False, "error": "a task is running" if busy() else "unknown key"}
        agent.device.act(req.key)
        return {"ok": True}

    @app.get("/api/stream")
    def stream(frames: int = 0, fps: float = 4.0):
        """MJPEG: a few frames per second of the device screen (`frames` > 0 stops after that many; for tests)."""
        def gen():
            sent = 0
            while frames <= 0 or sent < frames:
                t0 = time.time()
                if box["agent"] is None:            # no phone set yet
                    time.sleep(1.0)
                    if frames > 0:
                        return
                    continue
                try:
                    grab = getattr(agent.device, "fast_screenshot", agent.device.screenshot)
                    img = marked(grab().convert("RGB"))
                    img.thumbnail((540, 1200))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n"
                    sent += 1
                except Exception:  # noqa: BLE001 - a missed frame must not end the stream
                    pass
                time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/screen.png")
    def screen():
        img = marked(agent.last_image if busy() and agent.last_image is not None else agent.device.screenshot())
        img = img.copy()
        img.thumbnail((540, 1200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})

    return app


def connect_device(device: str) -> str:
    """An emulator or phone reached over the network ("host:port") is connected with `adb connect` first."""
    if ":" in device:
        import adbutils
        adbutils.adb.connect(device, timeout=10)
    return device


def check_endpoints(s: dict) -> dict:
    """Reach the phone (ADB), the DMA-2B API (/health) and the LLM (a one-token completion); never raises."""
    import requests
    out = {}
    try:
        import adbutils
        dev = adbutils.adb.device(connect_device(s["device"]))
        out["device"] = {"ok": dev.shell("echo ok").strip() == "ok", "model": dev.prop.model}
    except Exception as exc:  # noqa: BLE001
        out["device"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    try:
        r = requests.get(s["dma_url"].rstrip("/") + "/health", timeout=10,
                         headers={"x-api-key": s.get("dma_token") or "tok"})
        r.raise_for_status()
        h = r.json()
        out["dma"] = {"ok": bool(h.get("ok", True)), "device": h.get("device", "cuda")}
    except Exception as exc:  # noqa: BLE001
        out["dma"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    try:
        r = requests.post(s["llm_url"], timeout=60, headers={"Authorization": "Bearer " + (s.get("llm_key") or "")},
                          json={"model": s["llm_model"], "max_tokens": 1,
                                "messages": [{"role": "user", "content": "Reply with OK."}]})
        r.raise_for_status()
        out["llm"] = {"ok": "choices" in r.json()}
    except Exception as exc:  # noqa: BLE001
        out["llm"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    return out


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser(description="DMAgent dashboard. Endpoints can be set later in the page.")
    ap.add_argument("--dma-url", default=env("DMA_URL"))
    ap.add_argument("--llm-url-file")
    ap.add_argument("--llm-key-file")
    ap.add_argument("--llm-model", default=env("LLM_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--apps-json", default=env("APPS_JSON"))
    ap.add_argument("--serial", default=env("DEVICE", "emulator-5554"))
    ap.add_argument("--settings", default=env("DMAGENT_SETTINGS", os.path.expanduser("~/.dmagent.json")))
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--confidence", type=float, default=0.79)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7870)
    a = ap.parse_args()
    read = lambda f: open(f).read().strip() if f else None   # noqa: E731
    settings = load_settings(a.settings, {
        "device": a.serial, "dma_url": a.dma_url, "dma_token": env("DMA_API_TOKEN"),
        "llm_url": read(a.llm_url_file) or env("LLM_URL"), "llm_key": read(a.llm_key_file) or env("LLM_KEY"),
        "llm_model": a.llm_model, "apps_json": a.apps_json, "confidence": a.confidence, "max_steps": a.max_steps})
    approver = QueueApprover(timeout_s=600)

    def make_agent(s):
        from mobile_use.agents.dma.executor import Executor
        agent = DMAgent(dma_url=s["dma_url"], llm_url=s["llm_url"], llm_key=s.get("llm_key") or "",
                        llm_model=s["llm_model"], serial=connect_device(s["device"]), apps_json=s.get("apps_json"),
                        ocr=a.ocr, approver=approver, max_steps=int(s.get("max_steps") or 40),
                        executor=Executor(s["dma_url"], token=s.get("dma_token") or "tok"),
                        policy=Policy(confidence_threshold=float(s.get("confidence", 0.79)),
                                      max_steps=int(s.get("max_steps") or 40)))
        agent.reset()
        return agent

    agent = None
    if not missing(settings):
        try:
            agent = make_agent(settings)
        except Exception as exc:  # noqa: BLE001 - start anyway; the page shows Settings
            print("could not reach the endpoints (%s); set them in the page" % exc)
    info = {"confidence": a.confidence, "max_steps": a.max_steps}
    if settings.get("dma_url"):
        try:   # where DMA-2B runs (GPU or CPU), as the API reports it; the page never shows the API's address
            import requests
            info["dma_device"] = requests.get(settings["dma_url"].rstrip("/") + "/health", timeout=5, headers={
                "x-api-key": settings.get("dma_token") or "tok"}).json().get("device", "cuda")
        except Exception:  # noqa: BLE001
            pass
    app = build_app(agent, approver, info, settings=settings, make_agent=make_agent, settings_file=a.settings)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
