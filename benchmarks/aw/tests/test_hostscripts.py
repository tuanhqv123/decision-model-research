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
