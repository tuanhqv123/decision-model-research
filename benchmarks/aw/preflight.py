#!/usr/bin/env python3
"""Pre-flight gate: refuse to start a benchmark chunk unless every invariant holds.

Why this exists. On 2026-11-10 the same suite failed five times in a row, and each failure cost
a full emulator restart (~6 min) to discover:

  1. `suite.tasks: all` died on `ImportError: suite_assets` — two copies of android_world in the
     image, and the installed one is not the one the server runs.
  2. Results were written to the container's /tmp because the container had no `/out` mount, so
     everything would have been destroyed by the next restart.
  3. A freshly created container has no `pyyaml`, so `load_config` died *after* the restart.
  4. A stale harness was silently used because `tar`'s output went to /dev/null; the chunk then
     ran the previous build while printing `harness_ok`.
  5. `--run-root` + `--run-id` nested the run directory two levels deep, so the recorder opened
     a fresh index.db and re-ran episodes that were already complete.

None of those is a modelling question. Each is a check that costs seconds. This script runs
them all and exits non-zero if any fails, so a chunk either runs on a verified environment or
does not run at all.

It works in both directions:
  * from the HOST: also checks the container exists, /out is a real bind mount of the host run
    directory, and the run directory is not nested;
  * from INSIDE the container (how the chunk runner calls it): checks only what the benchmark
    process will actually see, because host-side facts are not observable there.

Usage:
    python3 preflight.py --run-root /home/u/awruns/aw116 --run-id aw116
"""
import argparse
import json
import os
import subprocess
import sys

CONTAINER = "aw-emu"
OK, BAD = "PASS", "FAIL"


def sh(args, timeout=120):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, "", "%s: %s" % (type(exc).__name__, exc)


def run(cmd, timeout=120):
    return sh(["bash", "-c", cmd], timeout)


def in_container():
    """True when running INSIDE the container.

    `sudo docker ps` is not available there and says nothing about anything; calling it made the
    first version of this gate report "container running: FAIL" on a healthy host, which would
    have blocked a valid chunk.
    """
    return os.path.exists("/.dockerenv") or os.environ.get("AW_IN_CONTAINER") == "1"


class Checker:
    def __init__(self):
        self.results = []

    def add(self, name, ok, detail=""):
        self.results.append((name, OK if ok else BAD, detail))
        return ok

    def report(self):
        width = max(len(r[0]) for r in self.results) if self.results else 10
        for name, status, detail in self.results:
            print("  %-*s  %s  %s" % (width, name, status, detail[:88]))
        failed = [r for r in self.results if r[1] == BAD]
        print()
        if failed:
            print("PREFLIGHT FAILED: %d of %d checks" % (len(failed), len(self.results)))
            for name, _, detail in failed:
                print("  - %s: %s" % (name, detail[:200]))
            return 1
        print("PREFLIGHT PASS: all %d checks" % len(self.results))
        return 0


def check_inside(c, expect_tasks, expect_contract=None, model_url="http://172.17.0.1:8701"):
    """Everything the benchmark process can verify from inside the container."""

    # harness present AND the right build. A stale copy silently produces results from
    # yesterday's code; budget support is the marker for this build.
    rc, out, _ = run("test -f /tmp/awh/aw/run_bench.py && echo yes")
    c.add("harness installed", "yes" in out, out[:60])
    rc, out, _ = run("grep -c step_budgets_file /tmp/awh/aw/run_bench.py || true")
    try:
        n = int(out.strip() or 0)
    except ValueError:
        n = 0
    c.add("harness is current build", n >= 3, "%d markers" % n)

    # the run-id join must be idempotent (no <root>/<id>/<id>)
    rc, out, _ = run("grep -q 'basename(os.path.normpath' /tmp/awh/aw/run_bench.py && echo yes")
    c.add("no-nest fix present", "yes" in out, out[:40])

    # Every task of the suite must get its own episode id. Until 2026-11-10 `_slug` cut names at 24
    # characters, so 12 of the 116 tasks shared an id with a shorter-named task and resumption
    # skipped them in every 116-task run, while each chunk still printed a normal report. This
    # plans the real suite with the installed harness and counts ids: an old harness gives 104/116.
    rc, out, _ = run(
        "cd /tmp/awh/aw && python3 -c 'import json; from awenv.runner import plan_episodes as p;"
        "b=list(json.load(open(\"/tmp/budgets.json\"))); "
        "print(len({s.episode_id for s in p([(n, 0) for n in b])}), len(b))' 2>&1 | tail -1")
    ids_tasks = out.split()
    c.add("episode ids unique over the suite",
          len(ids_tasks) == 2 and ids_tasks[0] == ids_tasks[1] and ids_tasks[1] != "0",
          "%s ids / %s tasks" % tuple((ids_tasks + ["?", "?"])[:2]))

    # dependencies a FRESH container lacks
    for mod in ("yaml", "numpy", "PIL", "requests"):
        rc, out, _ = run("python3 -c 'import %s' 2>&1 && echo ok" % mod)
        c.add("python: %s" % mod, "ok" in out, out[:60])

    # the registry the server uses must be importable; `tasks: all` can fall back to the budget
    # table, but task CONSTRUCTION needs it, so a broken import fails every episode at setup
    rc, out, _ = run(
        "cd /app && PYTHONPATH=/app python3 -c 'from android_world import registry;"
        "r=registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY);"
        "print(len(r))' 2>&1 | tail -1", timeout=180)
    try:
        n_tasks = int(out.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        n_tasks = 0
    c.add("registry importable", n_tasks == expect_tasks,
          "%d tasks (want %d): %s" % (n_tasks, expect_tasks, out[:60]))

    # per-task budget must cover every task, else the flat budget is used (a deviation)
    rc, out, _ = run("python3 -c \"import json;print(len(json.load(open('/tmp/budgets.json'))))\""
                     " 2>&1 | tail -1")
    try:
        n_bud = int(out.strip())
    except Exception:  # noqa: BLE001
        n_bud = 0
    c.add("budget table complete", n_bud == expect_tasks,
          "%d entries (want %d)" % (n_bud, expect_tasks))

    # /out must be a writable directory, and it must be the SAME one the host sees. If it is
    # not mounted, results die with the container, which needs a restart every ~15-20 min.
    rc, out, _ = run("touch /out/.preflight && echo ok")
    c.add("/out writable", "ok" in out, out[:60])

    # model API reachable FROM THE CONTAINER: the tunnel is host-side, so only this proves the
    # whole chain (container -> host bridge -> server)
    # the URL is the run config's own model.url (2026-11-12: a second server on :8710 failed this check while healthy)
    rc, out, _ = run('curl -s -m 10 -H "x-api-key: tok" %s/health' % model_url.rstrip("/"), timeout=60)
    c.add("model reachable", '"ok":true' in out.replace(" ", ""), out[:80])

    # The served checkpoint must speak the contract this run expects (2026-11-10: E2 was being served
    # questions it was never trained on). `--expect-contract` comes from the run's own config.
    if expect_contract:
        flat = out.replace(" ", "")
        c.add("model contract = %s" % expect_contract, '"contract":"%s"' % expect_contract in flat, out[:120])

    # The agent renders history with train/ac_common.render_history_line — the TRAINING renderer.
    # A harness tarball without train/ac_common.py would fail at the first step of every episode.
    rc, out, _ = run("cd /tmp/awh/aw && PYTHONPATH=/tmp/awh/aw python3 -c "
                     "'from awenv.agent import render_history_line as r; print(r({\"action_type\": \"scroll\", "
                     "\"direction\": \"down\"}))' 2>&1 | tail -1")
    c.add("agent uses the training history renderer", out.strip() == "scroll down", out[:80])

    # a11y must work on the DEVICE. /health answers even when the forwarder has stopped sending
    # protos; reset+screenshot is the only honest probe.
    rc, out, _ = run("python3 /tmp/probe.py --attempts 2 2>&1 | tail -1", timeout=400)
    c.add("a11y chain alive", "A11Y_PROBE PASS" in out, out[:90])

    # the resume index must agree with the files on disk, or a chunk silently re-runs work.
    # A zero-byte index.db is what the 2026-11-10 nesting left behind: the episodes were moved
    # back by hand but the database was empty, so `has_result` returned False for everything.
    #
    # Files whose episode has no `result` line are NOT counted as ok: the episode was interrupted
    # before it could be scored, so re-running it is correct and re-using it would record a
    # scoring that never happened. They are reported separately so the discrepancy is visible
    # instead of blocking the gate forever.
    idx = "/out/index.db"
    ep_dir = "/out/episodes"
    try:
        names = [f for f in os.listdir(ep_dir) if f.endswith(".jsonl")]
    except OSError:
        names = []
    completed = unfinished = 0
    for f in names:
        try:
            with open(os.path.join(ep_dir, f), encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError:
            continue
        if '"type": "result"' in body or '"type":"result"' in body:
            completed += 1
        else:
            unfinished += 1
    if os.path.isfile(idx) and os.path.getsize(idx) > 0:
        import sqlite3

        try:
            db = sqlite3.connect(idx)
            done = db.execute("select count(*) from episodes where status='ok'").fetchone()[0]
            c.add("resume index consistent", done == completed,
                  "index %d ok, %d scored files (+%d unfinished)"
                  % (done, completed, unfinished))
        except Exception as exc:  # noqa: BLE001
            c.add("resume index consistent", False,
                  "%s: %s (repair: rebuild_index.py)" % (type(exc).__name__, exc))
    elif completed:
        c.add("resume index consistent", False,
              "index.db empty/missing but %d scored episode files exist -> resume WILL re-run "
              "them; repair with rebuild_index.py" % completed)
    else:
        c.add("resume index consistent", True, "first chunk (nothing scored yet)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--expect-tasks", type=int, default=116)
    ap.add_argument("--expect-contract", default="", help="contract the served checkpoint must report")
    ap.add_argument("--model-url", default="http://172.17.0.1:8701", help="the run config's model.url")
    a = ap.parse_args()
    c = Checker()

    if in_container():
        check_inside(c, a.expect_tasks, a.expect_contract or None, a.model_url)
        return c.report()

    rc, out, _ = sh(["sudo", "-n", "docker", "ps", "--format", "{{.Names}}"])
    c.add("container running", CONTAINER in out.split(), out.replace("\n", " ")[:80])
    if CONTAINER not in out.split():
        return c.report()

    rc, out, _ = sh(["sudo", "-n", "docker", "inspect", CONTAINER, "--format",
                     "{{range .Mounts}}{{.Source}}|{{.Destination}}{{println}}{{end}}"])
    mounted = ""
    for line in out.splitlines():
        if line.endswith("|/out"):
            mounted = line.rsplit("|", 1)[0]
    host_root = os.path.abspath(a.run_root)
    c.add("/out bind-mounted", mounted == host_root,
          "mount=%r expected=%r" % (mounted, host_root))
    if mounted != host_root:
        return c.report()

    c.add("/out visible on host", os.path.isdir(host_root), host_root)
    nested = os.path.join(host_root, a.run_id)
    c.add("no nested run dir", not os.path.isdir(nested), nested)

    # delegate the in-container facts to the container itself, so there is one implementation
    inner = ("python3 /tmp/preflight.py --run-root /out --run-id %s --expect-tasks %d"
             % (a.run_id, a.expect_tasks))
    rc, out, err = sh(["sudo", "-n", "docker", "exec", CONTAINER, "bash", "-c", inner], timeout=600)
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("PREFLIGHT"):
            continue
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[-1] in (OK, BAD):
            continue
        if len(parts) >= 3 and parts[1] in (OK, BAD):
            c.add("inner: " + parts[0], parts[1] == OK, parts[2] if len(parts) > 2 else "")
        elif len(parts) >= 2 and parts[1] in (OK, BAD):
            c.add("inner: " + parts[0], parts[1] == OK, "")
    c.add("inner gate passed", rc == 0, (out.splitlines()[-1] if out else err)[:90])
    return c.report()


if __name__ == "__main__":
    sys.exit(main())
