"""The completion-config generator: run exactly the suite tasks a run directory has not scored."""
import os
import sqlite3
import subprocess
import sys

import yaml

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "hostscripts", "make_completion_cfg.py")
PARENT = "agent: d1_e2_b0prime\nsuite:\n  seed: 30\n  tasks: all\n"


def _rundir(tmp_path, scored, other=()):
    os.makedirs(tmp_path / "run")
    db = sqlite3.connect(str(tmp_path / "run" / "index.db"))
    db.execute("CREATE TABLE episodes (episode_id TEXT, task_name TEXT, status TEXT)")
    for n in scored:
        db.execute("INSERT INTO episodes VALUES (?,?,?)", ("id-" + n, n, "ok"))
    for n in other:  # an abandoned episode is NOT a scored one
        db.execute("INSERT INTO episodes VALUES (?,?,?)", ("ab-" + n, n, "abandoned"))
    db.commit()
    return str(tmp_path / "run")


def _make(tmp_path, rundir, names):
    (tmp_path / "parent.yaml").write_text(PARENT)
    (tmp_path / "tasks.txt").write_text("\n".join(names) + "\n")
    out = tmp_path / "out.yaml"
    proc = subprocess.run([sys.executable, SCRIPT, str(tmp_path / "parent.yaml"), rundir,
                           str(tmp_path / "tasks.txt"), str(out)], capture_output=True, text=True)
    return proc, out


def test_lists_exactly_the_unscored_tasks_in_suite_order(tmp_path):
    names = ["A", "B", "SimpleCalendarAddOneEvent", "SimpleCalendarAddOneEventTomorrow"]
    rundir = _rundir(tmp_path, scored=["A", "SimpleCalendarAddOneEvent"], other=["B"])
    proc, out = _make(tmp_path, rundir, names)
    assert proc.returncode == 0, proc.stderr
    cfg = yaml.safe_load(out.read_text())
    assert cfg["suite"]["tasks"] == ["B", "SimpleCalendarAddOneEventTomorrow"]
    assert cfg["agent"] == "d1_e2_b0prime"  # everything but `tasks` is the parent's


def test_nothing_to_run_writes_nothing_and_says_so(tmp_path):
    rundir = _rundir(tmp_path, scored=["A", "B"])
    proc, out = _make(tmp_path, rundir, ["A", "B"])
    assert proc.returncode == 3
    assert not out.exists()


# ---------------------------------------------------------------- scored_count.py
COUNT = os.path.join(HERE, "..", "hostscripts", "scored_count.py")


def _count(rundir):
    proc = subprocess.run([sys.executable, COUNT, rundir], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return int(proc.stdout.strip())


def test_scored_count_ignores_an_unfinished_episode_file(tmp_path):
    """R1 s0, 2026-11-10: a container death left episodes/<id>.jsonl with no 'ok' row. It is renamed
    .abandoned. only when the same task restarts, so counting files read 105 with 104 scored, and a run
    with 115 scored + 1 partial would have stopped as complete."""
    rundir = _rundir(tmp_path, scored=["A", "B"], other=["C"])
    os.makedirs(os.path.join(rundir, "episodes"))
    for ep in ["id-A", "id-B", "id-D"]:  # id-D: a partial episode with no index row
        with open(os.path.join(rundir, "episodes", ep + ".jsonl"), "w") as fp:
            fp.write("{}\n")
    assert _count(rundir) == 2


def test_scored_count_of_a_fresh_run_dir_is_zero_and_creates_nothing(tmp_path):
    os.makedirs(tmp_path / "fresh")
    assert _count(str(tmp_path / "fresh")) == 0
    assert os.listdir(str(tmp_path / "fresh")) == []


# ---- sweep_infra: an emulator that dies mid-chunk must not leave "scored" episodes behind --------------------------
import importlib.util  # noqa: E402
import json  # noqa: E402


def _load_sweep():
    spec = importlib.util.spec_from_file_location("sweep_infra", os.path.join(HERE, "..", "hostscripts", "sweep_infra.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _episode(run, eid, **res):
    os.makedirs(os.path.join(run, "episodes"), exist_ok=True)
    with open(os.path.join(run, "episodes", eid + ".jsonl"), "w") as fp:
        fp.write(json.dumps({"type": "step"}) + "\n")
        fp.write(json.dumps(dict({"type": "result", "episode_id": eid, "task_name": eid.upper()}, **res)) + "\n")
    db = sqlite3.connect(os.path.join(run, "index.db"))
    db.execute("CREATE TABLE IF NOT EXISTS episodes (episode_id TEXT, task_name TEXT, status TEXT)")
    db.execute("INSERT INTO episodes VALUES (?,?,?)", (eid, eid.upper(), "ok"))
    db.commit()


def test_sweep_sets_aside_dead_emulator_episodes_and_keeps_real_outcomes(tmp_path):
    sw = _load_sweep()
    run = str(tmp_path)
    _episode(run, "good", score=1.0, env_crash=False, setup_error=None)
    _episode(run, "crashed", score=None, env_crash=True, setup_error=None)
    _episode(run, "adbdead", score=None, env_crash=True, setup_error="AdbControllerError: adb: device 'emulator-5554' not found")
    _episode(run, "realsetup", score=None, env_crash=False, setup_error="no such table: map_markers")
    moved = sw.sweep(run, max_retries=2)
    assert sorted(moved) == ["adbdead", "crashed"]
    assert sorted(os.listdir(os.path.join(run, "episodes"))) == ["good.jsonl", "realsetup.jsonl"]
    assert len([f for f in os.listdir(os.path.join(run, "abandoned_infra")) if f.endswith(".jsonl")]) == 2  # never deleted
    ids = [r[0] for r in sqlite3.connect(os.path.join(run, "index.db")).execute("select episode_id from episodes")]
    assert sorted(ids) == ["good", "realsetup"]  # so resumption will run the two again


def test_sweep_gives_up_after_max_retries_so_a_task_that_kills_the_emulator_is_not_retried_forever(tmp_path):
    sw = _load_sweep()
    run = str(tmp_path)
    for attempt in range(3):
        _episode(run, "killer", score=None, env_crash=True, setup_error=None)
        moved = sw.sweep(run, max_retries=2)
        assert (moved == ["killer"]) == (attempt < 2)
        if attempt == 2:  # third time the episode stays, recorded as the crash it is
            assert os.path.exists(os.path.join(run, "episodes", "killer.jsonl"))
        else:
            assert not os.path.exists(os.path.join(run, "episodes", "killer.jsonl"))
        db = sqlite3.connect(os.path.join(run, "index.db"))
        db.execute("DELETE FROM episodes WHERE episode_id='killer'") if attempt == 2 else None
        db.commit()


def test_sweep_always_retries_a_victim_of_an_earlier_death(tmp_path):
    """An episode that never started because the emulator was ALREADY dead (setup AdbControllerError, 0 steps) is not
    a result about its task; it must be re-run however often that happens, unlike the task that killed the emulator."""
    sw = _load_sweep()
    run = str(tmp_path)
    for attempt in range(4):
        _episode(run, "victim", score=None, env_crash=True, n_steps=0,
                 setup_error="AdbControllerError: adb: device 'emulator-5554' not found")
        assert sw.sweep(run, max_retries=2) == ["victim"]


def test_sweep_writes_the_emulator_log_when_it_sets_something_aside(tmp_path):
    sw = _load_sweep()
    run = str(tmp_path)
    _episode(run, "crashed", score=None, env_crash=True, n_steps=8, setup_error=None)
    sw.sweep(run, max_retries=2, emulator_log=lambda: "qemu: fatal\n")
    logs = [f for f in os.listdir(os.path.join(run, "abandoned_infra")) if f.startswith("emu_log_")]
    assert len(logs) == 1 and open(os.path.join(run, "abandoned_infra", logs[0])).read() == "qemu: fatal\n"
