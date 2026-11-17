"""Run a queue of GPU jobs on a shared GPU without blocking anyone.

    python gpu_share.py --queue queue.json --log share_queue.out

Rules:
- A job starts only when the GPU has had no other user for `--quiet-s` seconds (default 600) and at least
  `--need-mib` MiB are free (default 21500). Processes already on the GPU when this script starts (a resident service)
  are tolerated, matched by command line so a restart under a new pid is still tolerated.
- While a job runs, it yields as soon as another process appears on the GPU, or a tolerated one grows by more than
  `--grow-mib` MiB (default 1024).
- A `train` job yields by SIGUSR1: the trainer finishes its optimizer step, saves its whole state and exits 75
  (`resume_state.py`); started again, it continues from that state. Any other job is stopped and later run again from
  the start.
- A job that exits with an error is started again (a train job resumes from its last saved state); after three errors
  in a row the queue stops and the log says QUEUE FAILED.

queue.json: [{"name": ..., "kind": "train" | "eval" | "cpu", "cmd": "<shell command>", "done_if": "<path>"}]
`done_if` (optional) must exist after the job for it to count as done. A `cpu` job does not use the GPU and runs at once.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple

PAUSED_EXIT = 75


@dataclass
class Snapshot:
    free_mib: int
    apps: List[Tuple[int, int, str]] = field(default_factory=list)    # (pid, used MiB, command line)


class Gate:
    def __init__(self, residents: Set[str], need_mib: int = 21500, grow_mib: int = 1024, quiet_s: float = 600):
        self.residents, self.need, self.grow, self.quiet = set(residents), need_mib, grow_mib, quiet_s
        self.baseline_others = 0
        self.clear_since: Optional[float] = None

    def _others(self, snap: Snapshot, ours: Iterable[int]) -> Tuple[list, int]:
        ours = set(ours)
        foreign = [(pid, cmd) for pid, _, cmd in snap.apps if pid not in ours and cmd not in self.residents]
        return foreign, sum(m for pid, m, _ in snap.apps if pid not in ours)

    def started(self, snap: Snapshot, ours: Iterable[int] = ()):
        """Record how much memory the tolerated processes use when our job starts."""
        self.baseline_others = self._others(snap, ours)[1]

    def yield_reason(self, snap: Snapshot, ours: Iterable[int]) -> Optional[str]:
        foreign, others = self._others(snap, ours)
        if foreign:
            pid, cmd = foreign[0]
            return "another process on the GPU (pid %d, %s)" % (pid, os.path.basename(cmd.split(" ")[0]) or "?")
        if others > self.baseline_others + self.grow:
            return "the other GPU users need more GPU memory (%d MiB, was %d)" % (others, self.baseline_others)
        return None

    def may_start(self, snap: Snapshot, now: float) -> bool:
        foreign, _ = self._others(snap, ())
        if foreign or snap.free_mib < self.need:
            self.clear_since = None
            return False
        if self.clear_since is None:
            self.clear_since = now
        return now - self.clear_since >= self.quiet


# ---------------------------------------------------------------- the machine
def procs() -> dict:
    """pid -> (parent pid, command line), from `ps` (Linux and macOS)."""
    out = {}
    for line in subprocess.run(["ps", "-A", "-o", "pid=,ppid=,args="], capture_output=True, text=True,
                               timeout=60).stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = (int(parts[1]), parts[2] if len(parts) > 2 else "")
    return out


def cmdline(pid: int) -> str:
    return procs().get(pid, (0, ""))[1]


def read_snapshot() -> Snapshot:
    q = lambda args: subprocess.run(["nvidia-smi"] + args + ["--format=csv,noheader,nounits"],  # noqa: E731
                                    capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    free = int(q(["--query-gpu=memory.free"])[0].strip())
    apps, table = [], procs()
    for line in q(["--query-compute-apps=pid,used_memory"]):
        pid, used = [x.strip() for x in line.split(",")]
        apps.append((int(pid), int(used), table.get(int(pid), (0, ""))[1]))
    return Snapshot(free, apps)


def tree(root: int, table: Optional[dict] = None) -> Set[int]:
    """root and all its descendants."""
    children = {}
    for pid, (ppid, _) in (table if table is not None else procs()).items():
        children.setdefault(ppid, []).append(pid)
    out, todo = set(), [root]
    while todo:
        p = todo.pop()
        out.add(p)
        todo += children.get(p, [])
    return out


class Log:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def stop(p: subprocess.Popen, job: dict, log: Log, grace_s: float):
    if job["kind"] == "train":
        # the trainer itself, not the shell around it: a shell killed by SIGUSR1 would leave the trainer running
        table = procs()
        trainers = [q for q in tree(p.pid, table) if "train_d1.py" in table.get(q, (0, ""))[1]] or [p.pid]
        for q in trainers:
            os.kill(q, signal.SIGUSR1)        # finish the step, save the state, exit 75
    else:
        os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        log("%s did not stop within %d s; killed" % (job["name"], grace_s))
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def run_queue(jobs: List[dict], gate: Gate, log: Log, poll_s: float = 30, wait_s: float = 60) -> bool:
    for job in jobs:
        if job.get("done_if") and os.path.exists(job["done_if"]):
            log("%s already done (%s exists)" % (job["name"], job["done_if"]))
            continue
        errors = 0
        while True:
            if job["kind"] != "cpu":
                waited = False
                while not gate.may_start(read_snapshot(), time.time()):
                    if not waited:
                        log("%s waiting for the GPU to be free" % job["name"])
                        waited = True
                    time.sleep(wait_s)
                gate.started(read_snapshot())
            log("%s launched" % job["name"])
            out = open(job.get("out", os.devnull), "a")
            p = subprocess.Popen(job["cmd"], shell=True, executable="/bin/bash", start_new_session=True,
                                 stdout=out, stderr=subprocess.STDOUT)
            yielded = None
            while p.poll() is None:
                time.sleep(poll_s)
                if job["kind"] == "cpu" or p.poll() is not None:
                    continue
                try:
                    why = gate.yield_reason(read_snapshot(), tree(p.pid))
                except Exception as exc:  # noqa: BLE001 - a failed reading is not a reason to stop
                    log("could not read the GPU (%s)" % exc)
                    continue
                if why:
                    yielded = why
                    log("%s yields: %s" % (job["name"], why))
                    stop(p, job, log, grace_s=1200 if job["kind"] == "train" else 120)
            out.close()
            rc = p.returncode
            if yielded:
                log("%s paused (exit %s); will continue when the GPU is free" % (job["name"], rc))
                continue
            if rc == 0 and (not job.get("done_if") or os.path.exists(job["done_if"])):
                log("%s done" % job["name"])
                break
            errors += 1
            log("%s exited %s%s (error %d of 3)" % (job["name"], rc,
                "" if rc != 0 else ", but %s is missing" % job.get("done_if"), errors))
            if errors >= 3:
                log("QUEUE FAILED at %s" % job["name"])
                return False
    log("QUEUE DONE")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--need-mib", type=int, default=21500)
    ap.add_argument("--grow-mib", type=int, default=1024)
    ap.add_argument("--quiet-s", type=float, default=600)
    a = ap.parse_args()
    log = Log(a.log)
    snap = read_snapshot()
    residents = {cmd for _, _, cmd in snap.apps}
    log("GPU users tolerated (already running): %d process(es); %d MiB free" % (len(residents), snap.free_mib))
    ok = run_queue(json.load(open(a.queue)), Gate(residents, a.need_mib, a.grow_mib, a.quiet_s), log)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
