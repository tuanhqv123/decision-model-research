"""Decision latency of a served DMA-2B checkpoint on AndroidControl-High test steps.

One request = one decision = one forward pass. Requests are sent one at a time (batch 1) to `/v1/decide` with the
goal, the step's history, the screenshot and the step's full element list, as the AC-High evaluation feeds them.
Recorded per request:
  model_ms   server-side time of the forward pass (reported by the API as latency_ms.model)
  server_ms  server-side time of the whole request: decode image, build state, forward pass, build response
  client_ms  wall clock seen by the client on the same host (adds HTTP and JSON encoding)
  n_elements number of elements sent
GPU utilisation of the card is sampled once per request, so contention from other processes is visible in the output.
The first `--warmup` requests are not counted.

  python latency_api.py --rows compact_actest_v4.jsonl --images <dir> --url http://127.0.0.1:8711 --n 500 --out lat.json
"""
import argparse
import base64
import json
import random
import statistics
import subprocess
import time
import urllib.request


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return int(out[0])
    except Exception:  # noqa: BLE001 - utilisation is context, not the measurement
        return None


def summary(rows, key):
    xs = [r[key] for r in rows]
    return {"median": round(statistics.median(xs), 1), "p95": round(pct(xs, 0.95), 1),
            "mean": round(statistics.mean(xs), 1), "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8711")
    ap.add_argument("--token", default="tok")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.rows)]
    random.seed(a.seed)
    pick = random.sample(rows, a.n + a.warmup)
    out = []
    for i, r in enumerate(pick):
        img = base64.b64encode(open("%s/%s" % (a.images, r["img"]), "rb").read()).decode()
        req = {"goal": r["goal"], "history": r.get("history_v3") or r.get("history") or [], "image_b64": img,
               "elements": r["elements"], "track": "VA"}
        util = gpu_util()
        t0 = time.time()
        http = urllib.request.Request(a.url + "/v1/decide", data=json.dumps(req).encode(), method="POST",
                                      headers={"x-api-key": a.token, "content-type": "application/json"})
        with urllib.request.urlopen(http, timeout=120) as resp:
            body = json.loads(resp.read())
        client_ms = (time.time() - t0) * 1000
        lat = body.get("latency_ms") or {}
        if i < a.warmup:
            continue
        out.append({"ep": r["ep"], "step": r["step"], "n_elements": len(r["elements"]), "gpu_util": util,
                    "model_ms": lat.get("model"), "server_ms": lat.get("total"), "client_ms": round(client_ms, 1)})

    by_len = {}
    for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 10 ** 6)):
        sub = [r for r in out if lo <= r["n_elements"] < hi]
        if sub:
            by_len["%d-%s" % (lo, hi if hi < 10 ** 6 else "")] = summary(sub, "model_ms")
    utils = [r["gpu_util"] for r in out if r["gpu_util"] is not None]
    res = {"n": len(out), "warmup_excluded": a.warmup, "seed": a.seed, "rows_file": a.rows.split("/")[-1],
           "model_ms": summary(out, "model_ms"), "server_ms": summary(out, "server_ms"),
           "client_ms": summary(out, "client_ms"), "model_ms_by_elements": by_len,
           "elements": {"median": statistics.median(r["n_elements"] for r in out),
                        "p95": pct([r["n_elements"] for r in out], 0.95)},
           "gpu_util_at_request": {"median": statistics.median(utils) if utils else None,
                                   "share_ge_50": round(sum(u >= 50 for u in utils) / len(utils), 3) if utils else None},
           "per_request": out}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_request"}, indent=1))


if __name__ == "__main__":
    main()
