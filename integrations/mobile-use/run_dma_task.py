"""Run one task on a device with DMAgent (LLM planner + DMA-2B) and save a trace.

    python examples/run_dma_task.py "Turn off Wi-Fi" --serial emulator-5554 --dma-url http://172.17.0.1:8711 \
        --llm-url-file /tmp/llm_url --llm-key-file /tmp/llm_key --apps-json /tmp/aw_apps.json --out /tmp/dma_run
"""
import argparse
import json
import logging
import os
import time

from mobile_use.agents.dma_agent import DMAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--dma-url", required=True)
    ap.add_argument("--llm-url-file", required=True)
    ap.add_argument("--llm-key-file", required=True)
    ap.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--apps-json")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--out", default="dma_run")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(a.out, exist_ok=True)
    agent = DMAgent(env={"serial_no": a.serial, "go_home": True, "wait_after_action_seconds": 2.0},
                    max_steps=a.max_steps, dma_url=a.dma_url, llm_url=open(a.llm_url_file).read().strip(),
                    llm_key=open(a.llm_key_file).read().strip(), llm_model=a.llm_model, apps_json=a.apps_json)
    trace, t0 = [], time.time()
    for k, step in enumerate(agent.iter_run(a.task)):
        if step.curr_env_state is not None:
            step.curr_env_state.pixels.save(os.path.join(a.out, "step_%02d.png" % k))
        trace.append({"step": k, "action": step.action.name if step.action else None,
                      "parameters": step.action.parameters if step.action else None, "desc": step.action_desc,
                      "thought": step.thought})
        print("STEP %d: %s" % (k, step.action_desc), flush=True)
    agent.env.get_state().pixels.save(os.path.join(a.out, "final.png"))
    json.dump({"task": a.task, "steps": trace, "llm_calls": agent.llm_calls, "planner_log": agent.log,
               "dma_history": agent.history, "answer": agent.answer, "seconds": round(time.time() - t0, 1)},
              open(os.path.join(a.out, "trace.json"), "w"), indent=1, default=str)
    print("DONE in %.0fs, %d steps, %d LLM calls" % (time.time() - t0, len(trace), agent.llm_calls))


if __name__ == "__main__":
    main()
