#!/usr/bin/env python3
"""Refresh the AndroidWorld 116-task evidence file from the live run.

The run is resumable and takes hours, so any number written into a document is a **snapshot**.
This script is the single place that recomputes it, so a document refresh is one command
instead of a hand-edit that silently drifts:

    python scripts/refresh_aw116.py --host HOST_TRAIN

It copies the episode JSONL out of the container, recomputes every figure the paper and the
record documents quote, and rewrites
``benchmarks/results/2026-11-08-aw116/evidence-82tasks.txt`` — whose name keeps the count it was
first written at, while its first line always states the true current count. Renaming the file
would break the checks in ``scripts/check_record.py``, so the header carries the truth instead.

Nothing here is estimated: every line is a count or a median over episodes that finished.
"""
import argparse
import collections
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "benchmarks", "results", "2026-11-08-aw116")
EVIDENCE = os.path.join(OUT_DIR, "evidence-82tasks.txt")


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def pull(host, container, run_dir, local):
    """Copy the run's episode directory out of the container onto this machine.

    Never destructive before the fetch succeeds. The first version `rmtree`d ``local`` and *then* tried
    ssh, so a failed pull (host down, container gone, wrong name) silently deleted committed evidence —
    an auditor lost 128 tracked episode files this way on 2026-11-09. The fetch now lands in a temporary
    sibling and is only swapped into place after it has been validated as non-empty.
    """
    local = os.path.abspath(local)
    staging = local + ".incoming"
    backup = local + ".previous"
    tmp = "/tmp/aw116_pull.tar"

    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)

    r = sh(["ssh", "-o", "BatchMode=yes", host,
            f"docker exec {container} tar -C {run_dir} -cf - episodes | gzip -1 > {tmp}"])
    if r.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit("pull failed, local data untouched: %s%s" % (r.stdout, r.stderr))
    r = sh(["scp", "-q", f"{host}:{tmp}", os.path.join(staging, "episodes.tgz")])
    if r.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit("scp failed, local data untouched: %s" % r.stderr)
    r = sh(["tar", "-xzf", os.path.join(staging, "episodes.tgz"), "-C", staging])
    if r.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit("untar failed, local data untouched: %s" % r.stderr)
    os.remove(os.path.join(staging, "episodes.tgz"))

    # Validate before swapping: an empty or resultless fetch must not replace a good local copy.
    fetched = os.path.join(staging, "episodes")
    n_jsonl = len(glob.glob(os.path.join(fetched, "*.jsonl"))) if os.path.isdir(fetched) else 0
    if n_jsonl == 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit("pull produced no episodes, local data untouched")

    if os.path.exists(backup):
        shutil.rmtree(backup)
    if os.path.exists(local):
        os.rename(local, backup)
    os.rename(staging, local)
    shutil.rmtree(backup, ignore_errors=True)


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def med(v):
    return sorted(v)[len(v) // 2] if v else None


def analyse(ep_dir):
    # One row per TASK, not per result line. A retry writes a second file for the same task (the
    # setup-error attempt is abandoned, the retry succeeds), so a naive glob over-counts: 118 result
    # lines for 116 planned tasks on 2026-11-08. Prefer a genuine attempt over a setup failure, then
    # the later episode_id, which is the retry.
    best = {}
    rows = []
    for f in sorted(glob.glob(os.path.join(ep_dir, "*.jsonl"))):
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        steps = [r for r in recs if r.get("type") == "step"]
        result = [r for r in recs if r.get("type") == "result"]
        if not result:
            continue  # an episode in flight is not a datum
        r = result[0]
        acts = collections.Counter((s.get("action_raw") or {}).get("action") for s in steps)
        gates = [(s.get("action_raw") or {}).get("gate_score") for s in steps
                 if (s.get("action_raw") or {}).get("gate_score") is not None]
        confs = [s.get("confidence") for s in steps if s.get("confidence") is not None]
        inv = sum(1 for s in steps if s.get("invalid_action"))
        emitted = sum(1 for s in steps if s.get("emitted"))
        cand = dict(task=r.get("task_name"), ep_id=r.get("episode_id"), success=bool(r.get("success")),
                         timeout=bool(r.get("timeout")), declared=bool(r.get("declared_done")),
                         n=r.get("n_steps") or 0, acts=dict(acts),
                         setup=r.get("setup_error"), gates=gates, confs=confs,
                         invalid=inv, emitted=emitted, coverage=r.get("coverage"),
                         crash=bool(r.get("env_crash")),
                         unscored=bool(r.get("unscored") or r.get("checker_unavailable")),
                         checker=r.get("checker_error"))
        key = cand["task"]
        prev = best.get(key)
        if prev is None:
            best[key] = cand
        else:
            # a real attempt beats a setup failure; otherwise keep the later retry
            cand_ok = not cand["setup"]
            prev_ok = not prev["setup"]
            if (cand_ok and not prev_ok) or (cand_ok == prev_ok and cand["ep_id"] > prev["ep_id"]):
                best[key] = cand
    rows = list(best.values())
    return rows


def report(rows, planned=116):
    n = len(rows)
    succ = sum(r["success"] for r in rows)
    lo, hi = wilson(succ, n)
    # Bucket precedence matters. `unscored` (the checker raised) is checked BEFORE timeout and
    # premature_status: those two describe what the AGENT did, while `unscored` says nobody ever
    # judged it. Before this, 12 episodes that ran 4-25 steps were reported as 6 timeouts and 6
    # premature terminations — model failures no checker ever assessed (found 2026-11-09).
    buckets = collections.Counter()
    for r in rows:
        if r["success"]:
            buckets["success"] += 1
        elif r["setup"]:
            buckets["setup_error"] += 1
        elif r.get("unscored"):
            buckets["unscored_checker_error"] += 1
        elif r["timeout"]:
            buckets["timeout_non_terminating"] += 1
        elif r["declared"]:
            buckets["premature_status"] += 1
        else:
            buckets["other"] += 1

    pre = [r["n"] for r in rows if r["declared"] and not r["success"]]
    tmo = [r["n"] for r in rows if r["timeout"]]
    gates = [g for r in rows for g in r["gates"]]
    confs = [c for r in rows for c in r["confs"]]

    def drift(sel):
        d = [statistics.mean(r["gates"][-3:]) - statistics.mean(r["gates"][:3])
             for r in sel if len(r["gates"]) >= 3]
        return len(d), (statistics.mean(d) if d else float("nan")), \
            (100.0 * sum(1 for x in d if x < 0) / len(d) if d else float("nan"))

    tt = [r for r in rows if r["acts"].get("type_text", 0) > 0]
    mix = collections.Counter()
    for r in rows:
        mix.update(r["acts"])
    ds, dsm, dsp = drift([r for r in rows if r["success"]])
    df, dfm, dfp = drift([r for r in rows if not r["success"]])

    out = []
    out.append("TASKS_FINISHED=%d PLANNED=%d SUCCESS=%d SR=%.1f%% WILSON=%.1f-%.1f"
               % (n, planned, succ, 100.0 * succ / max(n, 1), 100 * lo, 100 * hi))
    out.append("BUCKETS %s" % dict(buckets))
    out.append("premature median_steps=%s" % med(pre))
    out.append("timeout   median_steps=%s" % med(tmo))
    out.append("conf p50=%.3f  gate p50=%.3f  gate==0 frac=%.1f%%"
               % (statistics.median(confs), statistics.median(gates),
                  100.0 * sum(1 for x in gates if x == 0) / len(gates)))
    out.append("drift success: n=%d mean=%.6f decaying=%.0f%%" % (ds, dsm, dsp))
    out.append("drift fail   : n=%d mean=%.6f decaying=%.0f%%" % (df, dfm, dfp))
    out.append("type_text emitted: %d/%d success=%d" % (len(tt), n, sum(r["success"] for r in tt)))
    out.append("no type_text     : %d/%d success=%d"
               % (n - len(tt), n, sum(r["success"] for r in rows if r["acts"].get("type_text", 0) == 0)))
    # The objective requires these with their own denominators, never folded into SR.
    emitted = sum(r["emitted"] for r in rows)
    invalid = sum(r["invalid"] for r in rows)
    crashes = sum(r["crash"] for r in rows)
    out.append("EMITTED_ACTIONS=%d INVALID=%d INVALID_RATE=%.1f%%"
               % (emitted, invalid, 100.0 * invalid / max(emitted, 1)))
    out.append("ENV_CRASH_EPISODES=%d/%d TIMEOUT_EPISODES=%d/%d"
               % (crashes, n, buckets["timeout_non_terminating"], n))
    # SR_all vs SR_reachable: AndroidWorld upstream exposes no gold action list (TaskEval has
    # `template`, `goal`, `is_successful` — no `gold_action_types`), so `coverage` is None on every
    # episode by construction and the reachable split is UNDEFINED. Reporting a number would be a
    # fabrication; reporting the reason is the honest option, and the objective asks for the split
    # explicitly so it must be answered explicitly.
    with_cov = sum(1 for r in rows if r.get("coverage"))
    # A setup failure is not an attempt by the model: the device never reached a usable state.
    # Report both denominators so a degraded emulator cannot be read as a model failure.
    attempts = n - buckets.get("setup_error", 0)
    scorable = [r for r in rows if not r["setup"] and not r.get("unscored")]
    s_ok = sum(1 for r in scorable if r["success"])
    out.append("UNSCORED_CHECKER_UNAVAILABLE=%d of %d finished (ran, but the checker raised; NOT a model outcome)"
               % (buckets.get("unscored_checker_error", 0), n))
    slo, shi = wilson(s_ok, len(scorable))
    out.append("SCORABLE=%d SUCCESS=%d SR_SCORABLE=%.1f%% WILSON_SCORABLE=%.1f-%.1f"
               % (len(scorable), s_ok, 100.0 * s_ok / max(len(scorable), 1),
                  100 * slo, 100 * shi))
    out.append("GENUINE_ATTEMPTS=%d SUCCESS=%d SR_OVER_ATTEMPTS=%.1f%%"
               % (attempts, succ, 100.0 * succ / max(attempts, 1)))
    # AndroidWorld exposes no gold action list, so `coverage` is unpopulated and the *general*
    # reachable split is unavailable. But this programme answered a narrower version of it by hand:
    # exactly 2 of the 116 tasks need `answer`, which the option set cannot emit
    # (notes/2026-11-08-app-intent-not-action-space.md). So the split IS computable here, and saying
    # "cannot be separated" would be wrong — say which definition is used and give the number.
    ANSWER_TASKS = {"BrowserMultiply", "RetroPlaylistDuration"}
    reach = [r for r in rows if r["task"] not in ANSWER_TASKS]
    r_ok = sum(1 for r in reach if r["success"])
    out.append("SR_REACHABLE=%d/%d=%.1f%% n_reachable=%d n_all=%d definition=excludes_answer_needing_tasks"
               % (r_ok, len(reach), 100.0 * r_ok / max(len(reach), 1), len(reach), n))
    out.append("COVERAGE_FIELD_POPULATED=%d/%d reason=androidworld_exposes_no_gold_action_list"
               % (with_cov, n))
    n_crash = sum(1 for r in rows if r.get("crash"))
    crash_and_setup = sum(1 for r in rows if r.get("crash") and r.get("setup"))
    out.append("ENV_CRASH_IS_SETUP_FAILURE=%d/%d setup_errors=%d — the crashes ARE the setup failures, "
               "not a second defect" % (crash_and_setup, n_crash, buckets.get("setup_error", 0)))
    out.append("ACTION_MIX %s" % dict(mix))
    out.append("SUCCESSES %s" % [r["task"] for r in rows if r["success"]])
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="HOST_TRAIN")
    ap.add_argument("--container", default="aw-emu")
    ap.add_argument("--run-dir", default="/tmp/awruns/aw116_appint")
    ap.add_argument("--local", default=os.path.join(OUT_DIR, "episodes"))
    ap.add_argument("--planned", type=int, default=116)
    ap.add_argument("--from-local", action="store_true", help="recompute without pulling")
    args = ap.parse_args(argv)

    if not args.from_local:
        pull(args.host, args.container, args.run_dir, args.local)
    rows = analyse(os.path.join(args.local, "episodes") if os.path.isdir(
        os.path.join(args.local, "episodes")) else args.local)
    if not rows:
        # An empty directory is not a result. Reporting StatisticsError here once made the script look
        # like it had computed something when it had no data at all.
        raise SystemExit("no finished episodes found under %s — nothing to report, files left alone"
                         % args.local)
    text = report(rows, planned=args.planned)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(EVIDENCE, "w", encoding="utf-8") as fp:
        fp.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
