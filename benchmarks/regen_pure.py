#!/usr/bin/env python3
"""Regenerate the pure-protocol AC-v2 predictions through the served checkpoint.

Why this exists: the 64.32% headline has no artifact on disk. The saved run is the
app-name-hinted one (258/398), and the script that produced it is Mac/MPS-only and pinned to the
*base* model id — so re-running it would not reproduce the thing we report.

This script asks the deployed API instead, so the artifact is produced by the same forward pass the
paper's latency and calibration numbers come from, and with ``return_readouts`` every slot readout is
saved (the stage-2 blocker: a post-processor needs the readout of the class it switches *to*).

    python3 benchmarks/regen_pure.py --url http://127.0.0.1:8700 --out benchmarks/results/e2-pure.jsonl

Rows are scored with the official port (`run_acv2.score_row`), so the number is comparable to the
existing artifacts and to `benchmarks/repro_manifest.py`.
"""
import argparse
import base64
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_acv2 import ROWS, resolve_shot, score_row  # noqa: E402

SPATIAL = ("click", "long_press", "type_text")


def answer_to_row(answer):
    """API decision -> (action, value, point, confidence) for the official scorer."""
    action = answer.get("action")
    value = None
    if action == "scroll":
        value = answer.get("direction")
    elif action == "open_app":
        value = answer.get("app_name")
    elif action == "type_text":
        value = answer.get("text")
    elif action == "status":
        value = "successful"
    point = answer.get("point")
    return action, value, point, float(answer.get("confidence") or 0.0)


def decide(url, token, goal, image_path, history=None, track="V", timeout=120):
    import urllib.request

    with open(image_path, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode()
    body = json.dumps(
        {"goal": goal, "history": list(history or []), "image_b64": b64,
         "track": track, "return_readouts": True}
    ).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-API-Key"] = token
    request = urllib.request.Request(url.rstrip("/") + "/v1/decide", data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="regenerate AC-v2 predictions through the served model")
    ap.add_argument("--url", default="http://127.0.0.1:8700")
    ap.add_argument("--token", default=os.environ.get("D1_API_TOKEN"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--track", default="V")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    rows = json.load(open(ROWS))
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if args.resume and os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            rec = json.loads(line)
            done.add((rec["episode_id"], rec["step"]))
        print("[resume] %d already done" % len(done), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    n_ok = 0
    n_done = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            key = (row["episode_id"], row["step"])
            if key in done:
                continue
            path = resolve_shot(row)
            try:
                # The leaderboard protocol is goal + Interaction History. A goal-only run is the
                # mistake this project already made once; `previous_actions` is the history.
                answer = decide(args.url, args.token, row["goal"], path,
                                history=row.get("previous_actions"), track=args.track)
            except Exception as exc:  # noqa: BLE001
                print("  [ERR] %s/%s %s" % (row["episode_id"], row["step"], exc), flush=True)
                continue
            action, value, point, conf = answer_to_row(answer)
            correct, grounding = score_row(row, action, value, point)
            n_done += 1
            n_ok += int(bool(correct))
            rec = {
                "episode_id": row["episode_id"],
                "step": row["step"],
                "gold": row["action"],
                "pred": action,
                "value": value,
                "pt": point,
                "conf": conf,
                "correct": correct,
                "grounding": grounding,
                "probs": answer.get("probabilities"),
                "box_conf": answer.get("box_confidence"),
                "box_pt": answer.get("box_point"),
                "readouts": answer.get("readouts"),
                "latency_ms": (answer.get("latency_ms") or {}).get("model"),
                "harness": "served-api",
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                seen = i + 1
                print("  %d/%d acc=%.3f elapsed=%.0fs" % (seen, len(rows), n_ok / max(n_done, 1), time.time() - t0),
                      flush=True)

    print("done: %d predicted (%d correct, %.2f%%) in %.0fs -> %s"
          % (n_done, n_ok, 100.0 * n_ok / max(n_done, 1), time.time() - t0, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
