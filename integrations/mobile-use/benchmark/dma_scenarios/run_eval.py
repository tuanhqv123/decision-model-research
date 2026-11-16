"""Run the scenario suite with DMAgent and report completion rate and performance.

    python benchmark/dma_scenarios/run_eval.py --dma-url ... --llm-url-file ... --llm-key-file ... \
        --apps-json app_list.json --out results/ [--only wifi_off note_create] [--repeat 1]

Per task: setup over ADB, Home, the agent run (human-in-the-loop answered automatically and counted), then the check
on the device. Writes results.json (one row per run, with the agent's summary and events) and results.md (the table).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import by_name  # noqa: E402

from mobile_use.agents.dma.safety import AutoApprover, Policy  # noqa: E402
from mobile_use.agents.dma_agent import DMAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dma-url", required=True)
    ap.add_argument("--llm-url-file", required=True)
    ap.add_argument("--llm-key-file", required=True)
    ap.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--apps-json")
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--out", default="dma_eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    approver = AutoApprover(approve=True)
    agent = DMAgent(dma_url=a.dma_url, llm_url=open(a.llm_url_file).read().strip(),
                    llm_key=open(a.llm_key_file).read().strip(), llm_model=a.llm_model, serial=a.serial,
                    apps_json=a.apps_json, ocr=a.ocr, approver=approver, max_steps=a.max_steps,
                    policy=Policy(max_steps=a.max_steps, max_seconds=900))
    d = agent.device.d
    rows = []
    for rep in range(a.repeat):
        for sc in by_name(a.only):
            for pkg in set(sc.apps) | {"com.android.settings", "net.gsantner.markor", "com.google.android.contacts"}:
                d.shell("am force-stop %s" % pkg)     # every task starts from a fresh app state
            d.shell("settings put system pointer_location 0")   # a developer overlay that adds noise to OCR
            for cmd in sc.setup:
                d.shell(cmd)
            d.keyevent("HOME")
            time.sleep(2)
            n_req = len(approver.requests)
            try:
                for _ in agent.iter_run(sc.task):
                    pass
                err = None
            except Exception as exc:  # noqa: BLE001 - one broken run must not end the suite
                err = "%s: %s" % (type(exc).__name__, exc)
            s = agent.summary()
            time.sleep(1)
            ok = bool(sc.check(d, agent.answer))
            row = {"scenario": sc.name, "category": sc.category, "repeat": rep, "success": ok, "error": err,
                   "human_requests": len(approver.requests) - n_req, **s, "events": agent.events}
            rows.append(row)
            print("%-16s %-7s steps=%-3s %5.0fs llm=%-3s outcome=%s" % (sc.name, "OK" if ok else "FAIL", s["steps"],
                                                                     s["seconds"], s["llm_calls"], s["outcome"]),
                  flush=True)
            json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    write_table(rows, os.path.join(a.out, "results.md"))


def write_table(rows, path):
    n = len(rows)
    ok = sum(r["success"] for r in rows)
    avg = lambda k: sum((r.get(k) or 0) for r in rows) / n  # noqa: E731
    lines = ["| Scenario | Category | Result | Steps | Time (s) | LLM calls | Recoveries | Human requests |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %d | %.0f | %d | %d | %d |" % (
            r["scenario"], r["category"], "✓" if r["success"] else "✗", r["steps"], r["seconds"], r["llm_calls"],
            r["recoveries"], r["human_requests"]))
    model = [r["model_ms_avg"] for r in rows if r.get("model_ms_avg")]
    llm = [r["llm_ms_avg"] for r in rows if r.get("llm_ms_avg")]
    lines += ["", "Completion rate: **%d/%d = %.1f%%**." % (ok, n, 100.0 * ok / n),
              "Mean per task: %.1f steps, %.0f s, %.1f LLM calls." % (avg("steps"), avg("seconds"), avg("llm_calls")),
              "Mean latency: DMA-2B %.0f ms per decision, LLM %.0f ms per call." % (
                  sum(model) / max(1, len(model)), sum(llm) / max(1, len(llm)))]
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
