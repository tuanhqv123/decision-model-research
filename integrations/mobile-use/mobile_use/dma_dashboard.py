"""Control and monitoring dashboard for DMAgent.

    python -m mobile_use.dma_dashboard --dma-url http://<dma-api>:8700 --llm-url-file llm_url.txt \
        --llm-key-file llm_key.txt --apps-json app_list.json --serial emulator-5554 --port 7870

Open http://<host>:7870. Type a task and start it; the page shows the live screen (with the last tap), every planner
thought and local goal, every DMA-2B action with its probability, recoveries from pop-ups, reflection checks, and the
results of earlier tasks. When the agent needs a person (a sensitive action, a low-confidence action or a question), an
approval panel appears; the run waits until it is answered. "Stop" ends the run after the current action.
"""
import argparse
import io
import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from PIL import ImageDraw
from pydantic import BaseModel

from mobile_use.agents.dma.safety import Policy, QueueApprover
from mobile_use.agents.dma_agent import DMAgent

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>DMA-2B mobile agent</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e2e2df;--acc:#2563eb;--ok:#2e7d4f;--bad:#c2423f;--warn:#b7791f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card);display:flex;gap:10px;align-items:center}
header h1{font-size:18px;margin:0 16px 0 0}input[type=text]{flex:1;padding:8px 10px;border:1px solid var(--line);border-radius:8px;font:inherit}
button{padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);cursor:pointer;font:inherit}
button.go{background:var(--acc);color:#fff;border-color:var(--acc)}button.stop{color:var(--bad);border-color:var(--bad)}
main{display:grid;grid-template-columns:360px 1fr 320px;gap:16px;padding:16px}@media(max-width:1100px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
img{width:100%;border-radius:8px;display:block}h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);margin:0 0 8px}
.ev{border-bottom:1px solid var(--line);padding:6px 0}.ev .k{font-weight:600}.t{color:var(--mute);font-size:12px;margin-right:6px}
.plan .k{color:var(--acc)}.action .k{color:var(--ok)}.recovery .k,.blocked .k{color:var(--warn)}.approval_request .k,.question .k{color:var(--bad)}
.reflection .k{color:#7a5ba6}#events{max-height:78vh;overflow:auto}#approve{display:none;border:2px solid var(--bad)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px;text-align:left}
.status{font-weight:600}</style></head><body>
<header><h1>DMA-2B mobile agent</h1><input id="task" type="text" placeholder="Task, e.g. Turn off Wi-Fi">
<button class="go" onclick="start()">Start</button><button class="stop" onclick="stop()">Stop</button>
<span id="status" class="status"></span></header>
<main>
<div class="card"><h2>Screen</h2><img id="screen" src="/api/screen.png"></div>
<div class="card"><h2>Timeline</h2><div id="events"></div></div>
<div><div class="card" id="approve"><h2>Needs you</h2><div id="req"></div>
<input id="note" type="text" placeholder="answer / note (optional)" style="width:100%;margin:8px 0">
<button class="go" onclick="answer(true)">Approve / Send</button> <button onclick="answer(false)">Reject</button></div>
<div class="card" style="margin-top:16px"><h2>Summary</h2><div id="summary">-</div></div>
<div class="card" style="margin-top:16px"><h2>Earlier tasks</h2><table id="runs"></table></div></div>
</main>
<script>
const $=id=>document.getElementById(id);let pending=null;
function esc(s){return String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function line(e){const d=Object.assign({},e);delete d.t;delete d.type;delete d.summary;
 let txt="";if(e.type==="plan")txt=(e.thought?esc(e.thought)+" → ":"")+"<b>"+esc(e.next)+"</b> "+esc(e.instruction||e.text||e.question||"");
 else if(e.type==="action")txt=esc(e.action)+" (p="+e.probability+") "+(e.changed?"screen changed":"no change");
 else txt=esc(JSON.stringify(d));return `<div class="ev ${e.type}"><span class="t">${e.t}s</span><span class="k">${e.type}</span> ${txt}</div>`}
async function poll(){const s=await (await fetch("/api/state")).json();
 $("status").textContent=s.running?"running":(s.summary?("last: "+s.summary.outcome):"idle");
 $("events").innerHTML=s.events.map(line).join("");$("events").scrollTop=1e9;
 pending=s.pending;$("approve").style.display=pending?"block":"none";
 if(pending)$("req").innerHTML="<b>"+esc(pending.kind)+"</b>: "+esc(pending.reason)+(pending.action?"<br>action: "+esc(pending.action):"")+(pending.local_goal?"<br>local goal: "+esc(pending.local_goal):"");
 $("summary").innerHTML=s.summary?Object.entries(s.summary).map(([k,v])=>esc(k)+": <b>"+esc(v)+"</b>").join("<br>"):"-";
 $("runs").innerHTML="<tr><th>task</th><th>outcome</th><th>steps</th><th>s</th></tr>"+s.runs.map(r=>`<tr><td>${esc(r.task)}</td><td>${esc(r.outcome)}</td><td>${r.steps}</td><td>${r.seconds}</td></tr>`).join("");
 $("screen").src="/api/screen.png?"+Date.now()}
async function start(){await fetch("/api/start",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({task:$("task").value})})}
async function stop(){await fetch("/api/stop",{method:"POST"})}
async function answer(ok){if(!pending)return;await fetch("/api/answer",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({id:pending.id,approved:ok,note:$("note").value})});$("note").value=""}
setInterval(poll,1500);poll();
</script></body></html>"""


class Start(BaseModel):
    task: str


class Answer(BaseModel):
    id: int
    approved: bool
    note: str = ""


def build_app(agent: DMAgent, approver: QueueApprover) -> FastAPI:
    app = FastAPI(title="DMA-2B mobile agent")
    state = {"thread": None, "summary": None, "runs": []}
    lock = threading.Lock()

    def worker(task):
        try:
            for _ in agent.iter_run(task):
                pass
        except Exception as exc:  # noqa: BLE001 - show the failure instead of a dead page
            agent.outcome = "error: %s" % exc
        state["summary"] = agent.summary()
        state["runs"].append(state["summary"])

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

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
                "summary": state["summary"], "runs": state["runs"][-20:]}

    @app.get("/api/screen.png")
    def screen():
        running = bool(state["thread"] and state["thread"].is_alive())
        img = agent.last_image if running and agent.last_image is not None else agent.device.screenshot()
        img = img.copy()
        ev = next((e for e in reversed(getattr(agent, "events", [])) if e.get("point")), None)
        if running and ev:
            x, y = ev["point"]
            d = ImageDraw.Draw(img)
            d.ellipse([x - 40, y - 40, x + 40, y + 40], outline=(229, 72, 77), width=10)
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
    uvicorn.run(build_app(agent, approver), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
