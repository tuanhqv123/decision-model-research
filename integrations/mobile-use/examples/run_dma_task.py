"""Run one task with DMAgent (LLM planner + DMA-2B) from the command line and save a trace.

    python examples/run_dma_task.py "Turn off Wi-Fi" --serial emulator-5554 --dma-url http://<dma-api>:8700 \
        --llm-url-file llm_url.txt --llm-key-file llm_key.txt --apps-json app_list.json --out run_wifi [--ask]

Without --ask every human-in-the-loop request is approved automatically (and recorded); with --ask the terminal asks.
"""
import argparse
import json
import logging
import os

from mobile_use.agents.dma.safety import AutoApprover, Policy
from mobile_use.agents.dma_agent import DMAgent


class TerminalApprover:
    def __init__(self):
        self.requests = []

    def ask(self, req):
        self.requests.append(req)
        print("\n[NEEDS YOU] %s: %s %s" % (req["kind"], req["reason"], req.get("action") or ""))
        reply = input("approve? [y/N] or type an answer: ").strip()
        return {"approved": reply.lower() in ("y", "yes") or (req["kind"] == "question" and bool(reply)),
                "note": reply, "by": "person"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--dma-url", required=True)
    ap.add_argument("--llm-url-file", required=True)
    ap.add_argument("--llm-key-file", required=True)
    ap.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--apps-json")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--out", default="dma_run")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(a.out, exist_ok=True)
    agent = DMAgent(dma_url=a.dma_url, llm_url=open(a.llm_url_file).read().strip(),
                    llm_key=open(a.llm_key_file).read().strip(), llm_model=a.llm_model, serial=a.serial,
                    apps_json=a.apps_json, ocr=a.ocr, max_steps=a.max_steps,
                    approver=TerminalApprover() if a.ask else AutoApprover(True), policy=Policy(max_steps=a.max_steps))
    for k, step in enumerate(agent.iter_run(a.task)):
        if agent.last_image is not None:
            agent.last_image.save(os.path.join(a.out, "step_%02d.png" % k))
        print("STEP %d: %s" % (k, step.action_desc), flush=True)
    agent.device.screenshot().save(os.path.join(a.out, "final.png"))
    json.dump({"summary": agent.summary(), "events": agent.events, "history": agent.history, "log": agent.log},
              open(os.path.join(a.out, "trace.json"), "w"), indent=1, default=str)
    print(json.dumps(agent.summary(), indent=1))


if __name__ == "__main__":
    main()
