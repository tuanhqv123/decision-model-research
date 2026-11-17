"""The GPU-sharing gate: our job yields as soon as someone else starts using the GPU or needs more of its memory, and
starts again only after the GPU has been free of other users for a quiet period. Processes that were already on the
GPU before the queue started (a resident service) are tolerated, also after they restart under a new pid."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "train"))
from gpu_share import Gate, Snapshot  # noqa: E402

RES = "python -m service"


def snap(free, *apps):
    return Snapshot(free_mib=free, apps=list(apps))       # apps: (pid, used_mib, cmd)


def gate():
    return Gate(residents={RES}, need_mib=21500, grow_mib=1024, quiet_s=600)


def test_a_new_process_of_someone_else_makes_our_job_yield():
    g = gate()
    g.started(snap(6000, (10, 2700, RES), (99, 15000, "ours")), ours={99})
    assert g.yield_reason(snap(6000, (10, 2700, RES), (99, 15000, "ours")), ours={99}) is None
    why = g.yield_reason(snap(4000, (10, 2700, RES), (99, 15000, "ours"), (50, 2000, "python train.py")), ours={99})
    assert why and "another process" in why


def test_the_resident_service_needing_more_memory_makes_our_job_yield():
    g = gate()
    g.started(snap(6000, (10, 2700, RES), (99, 15000, "ours")), ours={99})
    assert g.yield_reason(snap(5800, (10, 2900, RES), (99, 15000, "ours")), ours={99}) is None   # small drift
    why = g.yield_reason(snap(4000, (10, 4200, RES), (99, 15000, "ours")), ours={99})
    assert why and "more GPU memory" in why


def test_our_own_memory_growth_never_makes_us_yield():
    g = gate()
    g.started(snap(9000, (10, 2700, RES), (99, 12000, "ours")), ours={99})
    assert g.yield_reason(snap(3000, (10, 2700, RES), (99, 18000, "ours")), ours={99}) is None


def test_we_start_again_only_after_a_quiet_period_with_enough_free_memory():
    g = gate()
    busy = snap(4000, (10, 2700, RES), (50, 16000, "python other.py"))
    free = snap(21800, (10, 2700, RES))
    assert not g.may_start(busy, now=0)
    assert not g.may_start(free, now=100)                  # the quiet period starts here
    assert not g.may_start(free, now=650)
    assert not g.may_start(busy, now=680)                  # someone came back: the period restarts
    assert not g.may_start(free, now=700)
    assert g.may_start(free, now=1300)


def test_too_little_free_memory_keeps_us_waiting_even_without_other_processes():
    g = gate()
    assert not g.may_start(snap(20000, (10, 2700, RES)), now=0)
    assert not g.may_start(snap(20000, (10, 2700, RES)), now=5000)


def test_a_resident_restarted_under_a_new_pid_is_still_tolerated():
    g = gate()
    g.started(snap(6000, (10, 2700, RES), (99, 15000, "ours")), ours={99})
    assert g.yield_reason(snap(6000, (11, 2700, RES), (99, 15000, "ours")), ours={99}) is None


def test_the_queue_yields_a_job_runs_it_again_when_free_and_stops_after_three_errors(tmp_path, monkeypatch):
    import gpu_share as gs
    flag = tmp_path / "done"
    snaps = iter([snap(21800, (10, 2700, RES))] * 2                                   # free: start
                 + [snap(4000, (10, 2700, RES), (50, 9000, "python other.py"))]      # someone came: yield
                 + [snap(21800, (10, 2700, RES))] * 1000)                            # free again
    monkeypatch.setattr(gs, "read_snapshot", lambda: next(snaps))
    monkeypatch.setattr(gs, "tree", lambda pid: {pid})
    lines = []
    log = lambda m: lines.append(m)  # noqa: E731
    jobs = [{"name": "ev", "kind": "eval", "cmd": "sleep 0.3; touch %s" % flag, "done_if": str(flag)},
            {"name": "bad", "kind": "eval", "cmd": "exit 3"}]
    ok = gs.run_queue(jobs, Gate({RES}, need_mib=21500, quiet_s=0), log, poll_s=0.05, wait_s=0)
    assert not ok and flag.exists()
    assert any(l.startswith("ev yields: another process") for l in lines)
    assert "ev done" in lines and sum(l.startswith("ev launched") for l in lines) == 2
    assert lines[-1] == "QUEUE FAILED at bad" and sum("error" in l for l in lines) == 3


def test_a_train_job_is_paused_by_signalling_the_trainer_itself_not_the_shell_around_it(tmp_path):
    import subprocess
    import gpu_share as gs
    script = tmp_path / "fake_train_d1.py"
    script.write_text("import signal, sys, time\n"
                      "signal.signal(signal.SIGUSR1, lambda *a: (open(sys.argv[1], 'w').write('saved'), sys.exit(75)))\n"
                      "print('ready', flush=True)\n"
                      "time.sleep(30)\n")
    saved = tmp_path / "saved"
    # a shell that does not exec its last command, as a queue command line may be
    p = subprocess.Popen("true; %s %s %s; echo after" % (sys.executable, script, saved), shell=True,
                         executable="/bin/bash", start_new_session=True, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    gs.stop(p, {"name": "t", "kind": "train"}, lambda m: None, grace_s=20)
    assert saved.read_text() == "saved"                 # the trainer got the pause and saved its state
    assert "after" in p.stdout.read()                   # and the shell, whose command line names the trainer too,
                                                        # was left alone to finish
