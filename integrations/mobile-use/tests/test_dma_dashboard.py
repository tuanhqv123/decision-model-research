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


def _finish(agent, c, task="Turn off Wi-Fi"):
    c.post("/api/start", json={"task": task})
    agent.release.set()
    for _ in range(100):
        if not c.get("/api/state").json()["running"]:
            break
        threading.Event().wait(0.02)


def test_a_finished_run_keeps_its_trajectory_and_the_state_lists_runs_without_it():
    agent, c = client()
    agent.events = [{"t": 0.1, "type": "start", "task": "Turn off Wi-Fi"},
                    {"t": 2.0, "type": "action", "action": "click Internet", "probability": 0.999}]
    _finish(agent, c)
    runs = c.get("/api/state").json()["runs"]
    assert len(runs) == 1 and "events" not in runs[0] and runs[0]["id"] == 0
    run = c.get("/api/runs/0").json()
    assert run["summary"]["outcome"] == "finished"
    assert [e["type"] for e in run["events"]] == ["start", "action"]
    assert c.get("/api/runs/7").status_code == 404


def test_the_state_carries_the_setup_shown_in_the_page():
    agent = FakeAgent()
    c = TestClient(build_app(agent, QueueApprover(), info={"serial": "emulator-5554", "llm_model": "Qwen/Qwen3.5-27B",
                                                         "dma_device": "cpu", "confidence": 0.79}))
    info = c.get("/api/state").json()["info"]
    assert info["serial"] == "emulator-5554" and info["confidence"] == 0.79


def test_the_page_is_the_chat_console_with_the_phone_beside_it():
    _, c = client()
    html = c.get("/").text
    assert 'id="composer"' in html and 'id="phone"' in html and "/api/stream" in html


def test_the_frame_a_decision_was_made_on_is_served_by_its_id():
    agent, c = client()
    agent.frames = {3: b"\xff\xd8jpeg"}
    r = c.get("/api/frames/3.jpg")
    assert r.status_code == 200 and r.content == b"\xff\xd8jpeg" and r.headers["content-type"] == "image/jpeg"
    assert c.get("/api/frames/4.jpg").status_code == 404


# ---------------------------------------------------------------- settings: endpoints set from the page
import json  # noqa: E402
import os  # noqa: E402

FULL = {"device": "emulator-5554", "dma_url": "http://dma:8700", "dma_token": "secret-tok",
        "llm_url": "http://llm/v1/chat/completions", "llm_key": "secret-key", "llm_model": "m1",
        "confidence": 0.79, "max_steps": 40}


def configured(tmp_path, settings=FULL, agent="default"):
    made = []

    def make_agent(s):
        made.append(dict(s))
        return FakeAgent()
    first = FakeAgent() if agent == "default" else agent
    app = build_app(first, QueueApprover(), settings=dict(settings), make_agent=make_agent,
                    settings_file=str(tmp_path / "dmagent.json"),
                    check=lambda s: {"device": {"ok": True}, "dma": {"ok": True, "device": "cpu"},
                                     "llm": {"ok": False, "error": "401"}})
    return first, made, TestClient(app)


def test_the_page_reads_the_endpoints_but_never_the_keys(tmp_path):
    _, _, c = configured(tmp_path)
    cfg = c.get("/api/config").json()
    assert cfg["dma_url"] == "http://dma:8700" and cfg["llm_model"] == "m1" and cfg["device"] == "emulator-5554"
    assert cfg["dma_token_set"] is True and cfg["llm_key_set"] is True
    assert "secret" not in json.dumps(cfg)


def test_saving_new_endpoints_rebuilds_the_agent_keeps_unsent_keys_and_writes_a_private_file(tmp_path):
    old, made, c = configured(tmp_path)
    r = c.post("/api/config", json={"llm_model": "m2", "device": "emulator-host:5555"}).json()
    assert r["ok"] is True and made[-1]["llm_model"] == "m2" and made[-1]["llm_key"] == "secret-key"
    assert c.post("/api/key", json={"key": "home"}).json()["ok"] and old.device.acts == []   # the new agent acts
    f = tmp_path / "dmagent.json"
    assert json.loads(f.read_text())["device"] == "emulator-host:5555" and oct(os.stat(f).st_mode & 0o777) == "0o600"
    assert c.get("/api/state").json()["info"]["llm_model"] == "m2"


def test_endpoints_cannot_change_while_a_task_runs(tmp_path):
    agent, made, c = configured(tmp_path)
    c.post("/api/start", json={"task": "t"})
    r = c.post("/api/config", json={"llm_model": "m2"}).json()
    agent.release.set()
    assert r["ok"] is False and made == []


def test_without_endpoints_the_page_asks_for_setup_and_no_task_starts(tmp_path):
    _, _, c = configured(tmp_path, settings={"device": "emulator-5554"}, agent=None)
    st = c.get("/api/state").json()
    assert st["configured"] is False and set(st["missing"]) == {"dma_url", "llm_url", "llm_model"}
    assert c.post("/api/start", json={"task": "t"}).json()["ok"] is False


def test_saved_settings_are_loaded_at_the_next_start(tmp_path):
    _, _, c = configured(tmp_path)
    c.post("/api/config", json={"llm_model": "m3"})
    from mobile_use.dma_dashboard import load_settings
    s = load_settings(str(tmp_path / "dmagent.json"), {"llm_model": "from-flags", "dma_url": "x"})
    assert s["llm_model"] == "m3" and s["llm_key"] == "secret-key"


def test_each_endpoint_can_be_tested_from_the_page(tmp_path):
    _, _, c = configured(tmp_path)
    r = c.post("/api/config/test", json={}).json()
    assert r["device"]["ok"] and r["dma"]["device"] == "cpu" and r["llm"] == {"ok": False, "error": "401"}
