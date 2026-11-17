"""Control and monitoring dashboard for DMAgent.

    python -m mobile_use.dma_dashboard --dma-url http://<dma-api>:8700 --llm-url-file llm_url.txt \
        --llm-key-file llm_key.txt --apps-json app_list.json --serial emulator-5554 --port 7870

Open http://<host>:7870. Type a task and start it; the page shows the live screen (a stream of a few frames per
second, with the agent's last tap marked), every planner
thought and local goal, every DMA-2B action with its probability, recoveries from pop-ups, reflection checks, and the
results of earlier tasks. When the agent needs a person (a sensitive action, a low-confidence action or a question), an
approval panel appears; the run waits until it is answered. "Stop" ends the run after the current action. While no
task runs, clicking the screen taps the device at that point, and Back / Home press those keys.
"""
import argparse
import io
import os
import threading
import time

import uvicorn
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


def build_app(agent: DMAgent, approver: QueueApprover, info: dict = None) -> FastAPI:
    """`info`: the setup the page shows (device serial, LLM model, where DMA-2B runs, the ask threshold)."""
    app = FastAPI(title="DMA-2B mobile agent")
    state = {"thread": None, "summary": None, "runs": []}
    lock = threading.Lock()

    def worker(task):
        try:
            for _ in agent.iter_run(task):
                pass
        except Exception as exc:  # noqa: BLE001 - show the failure instead of a dead page
            agent.outcome = "error: %s" % exc
        state["summary"] = dict(agent.summary(), id=len(state["runs"]))
        state["runs"].append({"summary": state["summary"], "events": list(getattr(agent, "events", []))})

    @app.get("/", response_class=HTMLResponse)
    def page():
        with open(PAGE_FILE, encoding="utf-8") as f:
            return f.read()

    @app.post("/api/start")
    def start(req: Start):
        with lock:
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
        return {"running": running, "events": list(getattr(agent, "events", []))[-200:], "pending": approver.pending,
                "summary": state["summary"], "runs": [r["summary"] for r in state["runs"][-20:]],
                "info": info or {}}

    @app.get("/api/runs/{i}")
    def get_run(i: int):
        if not 0 <= i < len(state["runs"]):
            raise HTTPException(404, "no such run")
        return state["runs"][i]

    def busy():
        return bool(state["thread"] and state["thread"].is_alive())

    def marked(img):
        """The frame with the agent's last tap circled, while a task runs."""
        ev = next((e for e in reversed(getattr(agent, "events", [])) if e.get("point")), None)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dma-url", required=True)
    ap.add_argument("--llm-url-file", required=True)
    ap.add_argument("--llm-key-file", required=True)
    ap.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--apps-json")
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--confidence", type=float, default=0.79)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7870)
    a = ap.parse_args()
    approver = QueueApprover(timeout_s=600)
    agent = DMAgent(dma_url=a.dma_url, llm_url=open(a.llm_url_file).read().strip(),
                    llm_key=open(a.llm_key_file).read().strip(), llm_model=a.llm_model, serial=a.serial,
                    apps_json=a.apps_json, ocr=a.ocr, approver=approver, max_steps=a.max_steps,
                    policy=Policy(confidence_threshold=a.confidence, max_steps=a.max_steps))
    agent.reset()
    info = {"serial": a.serial, "llm_model": a.llm_model, "confidence": a.confidence, "max_steps": a.max_steps}
    try:   # where DMA-2B runs (GPU or CPU), as the API reports it; the page never shows the API's address
        import requests
        h = requests.get(a.dma_url.rstrip("/") + "/health", timeout=5,
                         headers={"x-api-key": os.environ.get("DMA_API_TOKEN", "tok")}).json()
        info["dma_device"] = h.get("device", "cuda")
    except Exception:  # noqa: BLE001
        pass
    uvicorn.run(build_app(agent, approver, info), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
