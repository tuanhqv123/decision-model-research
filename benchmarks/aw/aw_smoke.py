"""Online AndroidWorld smoke run: emulator (HTTP) + d1-2B decision API (HTTP).

One episode = AndroidWorld task -> loop {screenshot -> /v1/decide -> execute_action}
until the model says `status` (done) or the step budget is spent; the score is
AndroidWorld's own `is_successful(env)` read from device state.

Both endpoints are expected on localhost through SSH tunnels:
  ssh -N -f -L 5001:127.0.0.1:5000 GPU_HOST_A
  ssh -N -f -L 8700:127.0.0.1:8700 GPU_HOST_B

Usage:
  python aw_smoke.py --tasks ContactsAddContact,ClockStopWatchRunning --max-steps 12
  python aw_smoke.py --list
  python aw_smoke.py --dry-run
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import time

import numpy as np
import requests
from PIL import Image

AW_URL = os.environ.get("AW_URL", "http://127.0.0.1:5001")
D1_URL = os.environ.get("D1_URL", "http://127.0.0.1:8700")
def _token():
    if os.environ.get("D1_API_TOKEN"):
        return os.environ["D1_API_TOKEN"].strip()
    for p in ("~/.d1-api-token", "~/d1-api/.env"):
        path = os.path.expanduser(p)
        if os.path.exists(path):
            t = open(path).read().strip()
            if "=" in t:
                for line in t.splitlines():
                    if line.startswith("D1_API_TOKEN="):
                        return line.split("=", 1)[1].strip()
            return t
    raise SystemExit("no D1 token: set D1_API_TOKEN or ~/.d1-api-token")


D1_TOKEN = _token()

SPATIAL = {"click", "long_press", "type_text"}

# AndroidWorld matches app names against its own vocabulary (adb_utils._PATTERN_TO_ACTIVITY and
# setup_device/apps.py). The decision model emits free-form names ("stopwatch"), so normalise first.
CANON_APPS = [
    "android world", "audio recorder", "broccoli app", "camera", "chrome", "clipper", "clock",
    "contacts", "dialer", "files", "joplin", "markor", "miniwob", "open tracks sports tracker",
    "osmand", "pro expense", "retro music", "settings", "simple calendar pro", "simple draw pro",
    "simple gallery pro", "simple sms messenger", "tasks", "vlc",
]
APP_ALIASES = {
    "stopwatch": "clock", "timer": "clock", "alarm": "clock", "desk clock": "clock",
    "contact": "contacts", "phone": "dialer", "call": "dialer",
    "browser": "chrome", "google chrome": "chrome", "web": "chrome",
    "note": "markor", "notes": "markor", "text editor": "markor",
    "message": "simple sms messenger", "messages": "simple sms messenger", "sms": "simple sms messenger",
    "messenger": "simple sms messenger", "text": "simple sms messenger",
    "calendar": "simple calendar pro", "todo": "tasks", "to do": "tasks", "task": "tasks",
    "gallery": "simple gallery pro", "photos": "simple gallery pro", "photo": "simple gallery pro",
    "draw": "simple draw pro", "sketch": "simple draw pro", "whiteboard": "simple draw pro",
    "expense": "pro expense", "expenses": "pro expense", "recorder": "audio recorder",
    "music": "retro music", "player": "retro music", "maps": "osmand", "files app": "files",
}


def canon_app(name):
    n = (name or "").strip().lower()
    if n in APP_ALIASES:
        return APP_ALIASES[n]
    if n in CANON_APPS:
        return n
    for c in CANON_APPS:
        if n and (n in c or c in n):
            return c
    return n


# The V-track checkpoint reads app names from a text slot, so over the pixels-only HTTP server it
# returns an EMPTY name. This goal-keyword fallback is harness scaffolding, NOT model capability:
# report it as such. (VA track needs the a11y tree, which /screenshot does not expose.)
APP_HINTS = {
    "contact": "contacts", "stopwatch": "clock", "timer": "clock", "alarm": "clock", "clock": "clock",
    "note": "markor", "markor": "markor", "sms": "simple sms messenger", "message": "simple sms messenger",
    "text ": "simple sms messenger", "calendar": "simple calendar pro", "task": "tasks", "to-do": "tasks",
    "expense": "pro expense", "receipt": "pro expense", "gallery": "simple gallery pro",
    "photo": "simple gallery pro", "picture": "simple gallery pro", "file": "files", "music": "retro music",
    "song": "retro music", "audio": "audio recorder", "record": "audio recorder", "camera": "camera",
    "video": "camera", "photo ": "camera", "browser": "chrome", "web": "chrome", "http": "chrome",
    "setting": "settings", "draw": "simple draw pro", "sketch": "simple draw pro", "joplin": "joplin",
    "osmand": "osmand", "navigation": "osmand", "vlc": "vlc",
}


def app_from_goal(goal):
    g = (goal or "").lower()
    hits = [(k, v) for k, v in APP_HINTS.items() if k in g]
    return max(hits, key=lambda kv: len(kv[0]))[1] if hits else ""


def log(msg):
    print(msg, flush=True)


class AW:
    def __init__(self, base):
        self.base = base

    def _get(self, path, **params):
        r = requests.get(f"{self.base}{path}", params=params, timeout=180)
        r.raise_for_status()
        return r.json()

    def _post(self, path, body=None, **params):
        r = requests.post(f"{self.base}{path}", params=params, json=body, timeout=180)
        r.raise_for_status()
        return r.json()

    def health(self):
        return self._get("/health")

    def reset(self, go_home=True):
        return self._post("/reset", go_home=go_home)

    def screenshot(self, wait=True):
        px = np.array(self._get("/screenshot", wait_to_stabilize=wait)["pixels"], dtype=np.uint8)
        if px.ndim == 1:
            raise RuntimeError(f"unexpected flat pixels: {px.shape}")
        return px

    def execute(self, action):
        return self._post("/execute_action", action)

    def task_list(self):
        return self._get("/suite/task_list", max_index=-1)["task_list"]

    def task_length(self, task):
        return self._get("/suite/task_length", task_type=task)["length"]

    def reinit(self, n=1, seed=30):
        return self._get("/suite/reinitialize", n_task_combinations=n, seed=seed, task_family="android_world")

    def init_task(self, task, idx):
        return self._post("/task/initialize", task_type=task, task_idx=idx)

    def goal(self, task, idx):
        return self._get("/task/goal", task_type=task, task_idx=idx)["goal"]

    def score(self, task, idx):
        return self._get("/task/score", task_type=task, task_idx=idx)["score"]

    def teardown(self, task, idx):
        return self._post("/task/tear_down", task_type=task, task_idx=idx)


class D1:
    def __init__(self, base, token):
        self.base = base
        self.h = {"content-type": "application/json", "x-api-key": token}

    def health(self):
        return requests.get(f"{self.base}/health", timeout=60).json()

    def decide(self, goal, history, px):
        buf = io.BytesIO()
        Image.fromarray(px).save(buf, format="PNG")
        payload = {"goal": goal, "history": history,
                   "image_b64": base64.b64encode(buf.getvalue()).decode(), "track": "V"}
        t0 = time.time()
        r = requests.post(f"{self.base}/v1/decide", data=json.dumps(payload), headers=self.h, timeout=120)
        r.raise_for_status()
        out = r.json()
        out["wall_ms"] = (time.time() - t0) * 1000
        return out


def to_action(dec, goal=""):
    """d1 decision -> AndroidWorld JSONAction dict (None = model finished)."""
    a = dec["action"]
    if a == "status":
        return None
    if a in ("click", "long_press"):
        x, y = dec.get("point") or [None, None]
        if x is None:  # a spatial action with no point cannot be executed
            return {"action_type": "wait"}
        return {"action_type": a, "x": int(x), "y": int(y)}
    if a == "type_text":
        x, y = dec.get("point") or [None, None]
        txt = dec.get("text") or ""
        if x is None:
            return {"action_type": "wait"}
        return {"action_type": "click", "x": int(x), "y": int(y), "_then_text": txt}
    if a == "scroll":
        return {"action_type": "scroll", "direction": dec.get("direction") or "down"}
    if a == "open_app":
        name = canon_app(dec.get("app_name") or "") or app_from_goal(goal)
        if not name:
            return {"action_type": "navigate_home"}
        return {"action_type": "open_app", "app_name": name}
    if a == "navigate_back":
        return {"action_type": "navigate_back"}
    if a == "wait":
        return {"action_type": "wait"}
    return {"action_type": "wait"}


def run_episode(aw, d1, task, idx, max_steps, verbose, policy="d1"):
    aw.reset(go_home=True)
    aw.init_task(task, idx)
    goal = aw.goal(task, idx)
    history = []
    lat, gates, done = [], [], False
    n_err = 0
    t0 = time.time()
    for step in range(max_steps):
        if policy == "wait":
            aw.execute({"action_type": "wait"})
            continue
        px = aw.screenshot(wait=True)
        if policy == "random":
            h, w = px.shape[:2]
            kind = random.choice(["click", "click", "scroll", "wait", "navigate_back"])
            if kind == "click":
                aw.execute({"action_type": "click", "x": random.randrange(w), "y": random.randrange(h)})
            elif kind == "scroll":
                aw.execute({"action_type": "scroll", "direction": random.choice(["up", "down"])})
            else:
                aw.execute({"action_type": kind})
            continue
        dec = d1.decide(goal, history, px)
        lat.append(dec["latency_ms"]["model"])
        gates.append(dec.get("gate_score") or 0.0)
        act = to_action(dec, goal)
        if verbose:
            extra = dec.get("app_name") or dec.get("text") or dec.get("direction") or ""
            log(f"    [{step:02d}] {dec['action']:13s} conf={dec['confidence']:.2f} "
                f"gate={(dec.get('gate_score') or 0):.2f} {dec['latency_ms']['model']:.0f}ms {extra}")
        if act is None:
            done = True
            break
        text = act.pop("_then_text", None)
        try:
            aw.execute(act)
            if text is not None:
                aw.execute({"action_type": "input_text", "text": text})
        except Exception as e:  # noqa: BLE001 - a bad action must not abort the episode
            n_err += 1
            log(f"    !! action failed ({type(e).__name__}: {str(e)[:100]}) -> navigate_home")
            try:
                aw.execute({"action_type": "navigate_home"})
            except Exception:  # noqa: BLE001
                pass
            history.append(f"{dec['action']} FAILED")
        history.append(f"{dec['action']} {json.dumps({k: v for k, v in act.items() if k != 'action_type'})}")
    steps = len(lat) if policy == "d1" else max_steps
    score = aw.score(task, idx)
    aw.teardown(task, idx)
    return {"task": task, "idx": idx, "goal": goal, "policy": policy, "score": float(score),
            "declared_done": done, "steps": steps, "action_errors": n_err,
            "wall_s": round(time.time() - t0, 1),
            "model_ms_p50": float(np.percentile(lat, 50)) if lat else None,
            "gate_mean": float(np.mean(gates)) if gates else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="ContactsAddContact,ClockStopWatchRunning")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--n-combos", type=int, default=1)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--aw-url", default=AW_URL)
    ap.add_argument("--d1-url", default=D1_URL)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--policy", default="d1", choices=["d1", "wait", "random"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    aw = AW(args.aw_url)
    log(f"aw health: {aw.health()}")
    if args.list:
        print(json.dumps(aw.task_list(), indent=1))
        return

    d1 = None
    if args.policy == "d1":
        d1 = D1(args.d1_url, D1_TOKEN)
        log(f"d1 health: {d1.health()}")
    if args.dry_run:
        log("dry run ok")
        return

    log(f"reinitialize suite: n={args.n_combos} seed={args.seed}")
    log(f"  {aw.reinit(args.n_combos, args.seed)}")

    results = []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        n = aw.task_length(task)
        for idx in range(n):
            log(f"== {task}[{idx}] ({idx + 1}/{n})")
            try:
                r = run_episode(aw, d1, task, idx, args.max_steps, not args.quiet, args.policy)
            except Exception as e:  # noqa: BLE001
                log(f"   ERROR {type(e).__name__}: {e}")
                r = {"task": task, "idx": idx, "score": 0.0, "error": f"{type(e).__name__}: {e}"}
            results.append(r)
            log(f"   score={r.get('score')} done={r.get('declared_done')} steps={r.get('steps')} "
                f"wall={r.get('wall_s')}s model_p50={r.get('model_ms_p50')}")

    ok = sum(1 for r in results if r.get("score"))
    log(f"\nSUCCESS {ok}/{len(results)}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        log(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
