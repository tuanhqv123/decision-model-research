"""Check the API against the offline evaluation, then measure latency.

1. PARITY: for N AC-v2 rows, send (goal, history, screenshot) and compare the answer with what eval_d1 recorded for the same
   row with the same checkpoint (action, click point). Batch effects in bf16 can move a probability in the 3rd decimal;
   the action and the point must agree.
2. LATENCY: single-request wall time p50/p95 (model-only and end to end), sequential requests.
Needs an SSH tunnel:  ssh -N -f -L 8700:127.0.0.1:8700 GPU_HOST_B
Usage: python test_client.py <eval_jsonl> [N=12] [--latency 30]
"""
import base64
import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.expanduser("~/Workspace/research")
URL = os.environ.get("D1_URL", "http://127.0.0.1:8700")
TOKEN = open(os.path.expanduser("~/.d1-api-token")).read().strip()


def call(path, payload=None):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"content-type": "application/json", "x-api-key": TOKEN})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=60))
    return r, (time.time() - t0) * 1000


def main():
    ev = [json.loads(l) for l in open(sys.argv[1])]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rows = {(r["episode_id"], r["step"]): r for r in json.load(open(f"{ROOT}/benchmarks/data/acv2/500_steps_filtered.json"))}
    print("health:", call("/health")[0])
    ok = bad = 0
    lat_model, lat_total = [], []
    for e in ev[:n]:
        r = rows[(e["episode_id"], e["step"])]
        img = base64.b64encode(open(f"{ROOT}/benchmarks/data/acv2_shots/images/{r['screenshot']}", "rb").read()).decode()
        out, wall = call("/v1/decide", {"goal": r["goal"], "history": r.get("previous_actions") or [], "image_b64": img, "track": "V"})
        same_action = out["action"] == e["pred"]
        same_pt = True
        if e["pred"] in ("click", "long_press", "type_text") and e.get("pt") and out.get("point"):
            same_pt = np.hypot(out["point"][0] - e["pt"][0], out["point"][1] - e["pt"][1]) < 25  # px on a 1080x2400 screen
        good = same_action and same_pt
        ok += good
        bad += not good
        lat_model.append(out["latency_ms"]["model"])
        lat_total.append(wall)
        print(f"[{'OK ' if good else 'DIFF'}] {e['gold']['action_type']:13s} api={out['action']:13s} eval={e['pred']:13s} "
              f"conf={out['confidence']:.2f} gate={out['gate_score']:.2f} model={out['latency_ms']['model']:.0f}ms")
    print(f"\nPARITY {ok}/{ok + bad}  (action and click point agree with the offline evaluation)")
    if "--latency" in sys.argv:
        k = int(sys.argv[sys.argv.index("--latency") + 1])
        e = ev[0]; r = rows[(e["episode_id"], e["step"])]
        img = base64.b64encode(open(f"{ROOT}/benchmarks/data/acv2_shots/images/{r['screenshot']}", "rb").read()).decode()
        payload = {"goal": r["goal"], "history": r.get("previous_actions") or [], "image_b64": img, "track": "V"}
        m, t = [], []
        for _ in range(k):
            out, wall = call("/v1/decide", payload); m.append(out["latency_ms"]["model"]); t.append(wall)
        print(f"LATENCY single request x{k}: model p50 {np.percentile(m,50):.0f} ms / p95 {np.percentile(m,95):.0f} ms | "
              f"end to end (incl. upload + tunnel) p50 {np.percentile(t,50):.0f} ms / p95 {np.percentile(t,95):.0f} ms")


main()
