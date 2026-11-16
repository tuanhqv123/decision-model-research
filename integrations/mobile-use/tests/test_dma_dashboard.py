"""Dashboard endpoints with a fake agent: manual control and the live stream."""
import threading

from fastapi.testclient import TestClient
from PIL import Image

from mobile_use.agents.dma.safety import Policy, QueueApprover
from mobile_use.dma_dashboard import build_app


class FakeDevice:
    def __init__(self):
        self.acts = []

    def screenshot(self):
        return Image.new("RGB", (1080, 2400), (40, 40, 40))

    fast_screenshot = screenshot

    def act(self, kind, **p):
        self.acts.append((kind, p))


class FakeAgent:
    def __init__(self):
        self.device, self.events, self.last_image, self.policy = FakeDevice(), [], None, Policy()
        self.release = threading.Event()

    def iter_run(self, task):
        self.release.wait(5)
        return iter(())

    def summary(self):
        return {"task": "t", "outcome": "finished"}


def client():
    agent = FakeAgent()
    return agent, TestClient(build_app(agent, QueueApprover()))


def test_a_tap_on_the_screen_image_taps_the_device_at_the_same_relative_point():
    agent, c = client()
    assert c.post("/api/tap", json={"x": 0.5, "y": 0.25}).json()["ok"]
    assert agent.device.acts == [("click", {"x": 540, "y": 600})]
    assert c.post("/api/key", json={"key": "back"}).json()["ok"]
    assert agent.device.acts[-1] == ("back", {})


def test_manual_control_is_refused_while_a_task_runs():
    agent, c = client()
    c.post("/api/start", json={"task": "t"})
    r = c.post("/api/tap", json={"x": 0.5, "y": 0.5}).json()
    agent.release.set()
    assert r["ok"] is False and agent.device.acts == []


def test_the_stream_sends_jpeg_frames():
    agent, c = client()
    with c.stream("GET", "/api/stream?frames=2") as r:
        body = b"".join(r.iter_bytes())
    assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert body.count(b"\xff\xd8") == 2           # two JPEG frames
