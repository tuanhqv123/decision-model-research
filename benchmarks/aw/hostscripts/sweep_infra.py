#!/usr/bin/env python3
"""Set aside episodes that an emulator death turned into non-outcomes, so resumption runs them again.

    sweep_infra.py <run_dir> [--max-retries 2]        (prints the episode ids it moved)

Why (2026-11-11, R1 seed 2): the emulator died in the 3rd episode of a chunk ("adb: device 'emulator-5554' not found") and
the 10 remaining episodes of the chunk were recorded as setup errors, each indexed `status='ok'`, so resumption would have
counted them as finished and never run them. The between-chunk preflight cannot see a death INSIDE a chunk.

An episode is infrastructure-lost when its result line says `env_crash`, or its setup error is an AdbControllerError. Such
episodes are MOVED (never deleted) to `<run_dir>/abandoned_infra/<id>.attemptN.jsonl` and their index rows removed. A task
that KILLED the emulator during its own episode (steps > 0) and has already been set aside `--max-retries` times stays as recorded: it kills the emulator every time, which is a result
about the task, and an endless retry loop is worse than an excluded episode. Genuine setup errors (a task whose own
initialisation raises) are left alone. A VICTIM — an episode that never started because the emulator was already dead
(AdbControllerError at setup, 0 steps) — says nothing about its task and is always set aside, however often.
When anything is set aside, the container's recent log is saved next to it (`emu_log_<utc>.txt`), because the next
round's restart discards it.
"""
import argparse
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import time


def _result(path):
    res = None
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if '"result"' in line:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") == "result":
                    res = rec
    return res


def is_infra_failure(res):
    return bool(res.get("env_crash")) or "AdbControllerError" in str(res.get("setup_error") or "")


def is_victim(res):
    """Never started: the emulator was dead before this episode's first step."""
    return "AdbControllerError" in str(res.get("setup_error") or "") and not res.get("n_steps")


def docker_log(container="aw-emu", tail=600):
    try:
        out = subprocess.run(["docker", "logs", "--tail", str(tail), container], capture_output=True, text=True,
                             timeout=30)
        return "\n".join(l for l in (out.stdout + out.stderr).splitlines() if "closure of unused" not in l) + "\n"
    except Exception as exc:  # noqa: BLE001 - evidence capture must never break the sweep
        return "docker logs failed: %r\n" % (exc,)


def sweep(run_dir, max_retries=2, emulator_log=docker_log):
    moved = []
    aside = os.path.join(run_dir, "abandoned_infra")
    for path in sorted(glob.glob(os.path.join(run_dir, "episodes", "*.jsonl"))):
        name = os.path.basename(path)
        if ".abandoned." in name:
            continue
        res = _result(path)
        if res is None or not is_infra_failure(res):
            continue
        eid = name[:-len(".jsonl")]
        tried = len(glob.glob(os.path.join(aside, eid + ".attempt*.jsonl")))
        if tried >= max_retries and not is_victim(res):
            continue
        os.makedirs(aside, exist_ok=True)
        shutil.move(path, os.path.join(aside, "%s.attempt%d.jsonl" % (eid, tried + 1)))
        db = sqlite3.connect(os.path.join(run_dir, "index.db"))
        db.execute("DELETE FROM episodes WHERE episode_id = ?", (eid,))
        db.commit()
        db.close()
        moved.append(eid)
    if moved and emulator_log is not None:
        with open(os.path.join(aside, "emu_log_%s.txt" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())), "w") as fp:
            fp.write(emulator_log())
    return moved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args(argv)
    for eid in sweep(args.run_dir, args.max_retries):
        print(eid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
