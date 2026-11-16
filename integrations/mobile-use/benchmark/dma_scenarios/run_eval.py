"""Run the scenario suite with DMAgent and report completion rate and performance.

    python benchmark/dma_scenarios/run_eval.py --dma-url ... --llm-url-file ... --llm-key-file ... \
        --apps-json app_list.json --out results/ [--only wifi_off note_create] [--repeat 1]

Per task: setup over ADB, Home, the agent run (human-in-the-loop answered automatically and counted), then the check
on the device. Writes results.json (one row per run, with the agent's summary and events) and results.md (the table).
"""
import argparse
import json
import math
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
            for cmd in sc.teardown:
                d.shell(cmd)
            # strict: the device check passes AND the agent itself declared the task done (or answered)
            strict = ok and s["outcome"] in ("finished", "answered")
            row = {"scenario": sc.name, "category": sc.category, "held_out": sc.held_out, "repeat": rep,
                   "success": ok, "strict": strict, "error": err,
                   "human_requests": len(approver.requests) - n_req, **s, "events": agent.events}
            rows.append(row)
            print("%-16s %-7s steps=%-3s %5.0fs llm=%-3s outcome=%s" % (sc.name, "OK" if ok else "FAIL", s["steps"],
                                                                     s["seconds"], s["llm_calls"], s["outcome"]),
                  flush=True)
            json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    write_table(rows, os.path.join(a.out, "results.md"))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return max(0.0, m - c), min(1.0, m + c)


def rate(rows, key):
    k, n = sum(r[key] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return "%d/%d = %.1f%% (95%% CI %.1f–%.1f%%)" % (k, n, 100.0 * k / max(1, n), 100 * lo, 100 * hi)


def write_table(rows, path):
    names = list(dict.fromkeys(r["scenario"] for r in rows))
    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    lines = ["| Scenario | Set | Category | Success | Strict | Steps | Time (s) | LLM calls | Human requests |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        rs = [r for r in rows if r["scenario"] == name]
        lines.append("| %s | %s | %s | %d/%d | %d/%d | %.1f | %.0f | %.1f | %.1f |" % (
            name, "held-out" if rs[0].get("held_out") else "tuned", rs[0]["category"],
            sum(r["success"] for r in rs), len(rs), sum(r.get("strict", False) for r in rs), len(rs),
            mean([r["steps"] for r in rs]), mean([r["seconds"] for r in rs]), mean([r["llm_calls"] for r in rs]),
            mean([r["human_requests"] for r in rs])))
    lines.append("")
    for label, sel in (("Held-out", [r for r in rows if r.get("held_out")]),
                       ("Tuned", [r for r in rows if not r.get("held_out")]), ("All", rows)):
        if sel:
            lines.append("%s: completed %s; strict %s." % (label, rate(sel, "success"), rate(sel, "strict")))
    model = [r["model_ms_avg"] for r in rows if r.get("model_ms_avg")]
    llm = [r["llm_ms_avg"] for r in rows if r.get("llm_ms_avg")]
    lines += ["Mean per run: %.1f steps, %.0f s, %.1f LLM calls." % (
                  mean([r["steps"] for r in rows]), mean([r["seconds"] for r in rows]),
                  mean([r["llm_calls"] for r in rows])),
              "Mean latency per call (client side of the shared GPU / LLM endpoint): DMA-2B %.0f ms, LLM %.0f ms." % (
                  mean(model), mean(llm))]
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
