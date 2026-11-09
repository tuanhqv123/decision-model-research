"""Recording must survive a killed process without losing or duplicating a step."""
import json
import os

import pytest

from awenv.recorder import Recorder


def _header(eid="p1_t0007_s30_d1", **kw):
    h = {
        "episode_id": eid,
        "task_family": "android_world",
        "task_name": "ClockStopWatchRunning",
        "task_idx": 0,
        "seed": 30,
        "agent": "d1_2b",
        "checkpoint": "E2@938",
        "protocol": "aw-main-v1",
        "step_budget": 20,
    }
    h.update(kw)
    return h


def _step(i, **kw):
    rec = {
        "i": i,
        "action_raw": {"action": "click", "point": [1, 2]},
        "action_executed": {"action_type": "click", "x": 1, "y": 2},
        "confidence": 0.7,
        "gate_score": 0.5,
        "invalid_action": False,
        "action_ok": True,
        "screen_changed": True,
        "model_ms": 100.0,
        "env_ms": 900.0,
    }
    rec.update(kw)
    return rec


def _result(**kw):
    rec = {
        "success": False,
        "declared_done": False,
        "n_steps": 2,
        "n_invalid_actions": 0,
        "n_action_errors": 0,
        "env_crash": False,
        "timeout": True,
        "latency": {"model_p50": 100.0, "e2e_total_ms": 2000.0},
        "tokens_generated": 0,
    }
    rec.update(kw)
    return rec


def test_writes_header_steps_result(tmp_path):
    rec = Recorder(str(tmp_path), "run1")
    rec.start_episode(_header())
    rec.add_step(_step(0))
    rec.add_step(_step(1))
    rec.finish_episode(_result())

    assert rec.has_result("p1_t0007_s30_d1") is True
    lines = [json.loads(l) for l in open(rec.episode_path("p1_t0007_s30_d1"))]
    assert [l["type"] for l in lines] == ["episode", "step", "step", "result"]
    assert len(rec.episodes()) == 1
    assert len(rec.steps()) == 2
    rec.close()


def test_pairing_key_is_recorded(tmp_path):
    rec = Recorder(str(tmp_path), "run1")
    rec.start_episode(_header())
    rec.finish_episode(_result())
    header = json.loads(open(rec.episode_path("p1_t0007_s30_d1")).readline())
    assert (header["task_name"], header["task_idx"], header["seed"]) == ("ClockStopWatchRunning", 0, 30)
    rec.close()


def test_rerun_abandons_previous_without_losing_it(tmp_path):
    rec = Recorder(str(tmp_path), "run1")
    rec.start_episode(_header())
    rec.add_step(_step(0))
    rec.finish_episode(_result())
    rec.close()

    rec2 = Recorder(str(tmp_path), "run1")
    assert rec2.has_result("p1_t0007_s30_d1") is True
    rec2.start_episode(_header())
    assert rec2.has_result("p1_t0007_s30_d1") is False
    abandoned = [f for f in os.listdir(rec2.ep_dir) if ".abandoned." in f]
    assert abandoned, "the previous attempt must be preserved, not silently overwritten"
    rec2.close()


def test_resume_after_partial_episode(tmp_path):
    rec = Recorder(str(tmp_path), "run1")
    rec.start_episode(_header())
    rec.add_step(_step(0))
    rec.close()

    rec2 = Recorder(str(tmp_path), "run1")
    assert rec2.has_result("p1_t0007_s30_d1") is False
    rec2.start_episode(_header())
    rec2.add_step(_step(0))
    rec2.add_step(_step(1))
    rec2.finish_episode(_result())
    lines = [json.loads(l) for l in open(rec2.episode_path("p1_t0007_s30_d1"))]
    assert [l["type"] for l in lines] == ["episode", "step", "step", "result"]
    assert [l["i"] for l in lines if l["type"] == "step"] == [0, 1]
    rec2.close()


def test_steps_are_indexed_with_episode_id(tmp_path):
    rec = Recorder(str(tmp_path), "run1")
    rec.start_episode(_header())
    rec.add_step(_step(0))
    rec.finish_episode(_result())
    steps = rec.steps()
    assert all(s["episode_id"] == "p1_t0007_s30_d1" for s in steps)
    rec.close()


def test_add_step_before_start_is_an_error(tmp_path):
    rec = Recorder(str(tmp_path), "run1")
    with pytest.raises(RuntimeError):
        rec.add_step(_step(0))
    rec.close()


def test_report_counts_episodes_from_earlier_resume_runs(tmp_path):
    """Every resume is a new run_id, so scoping the report to run_id lost finished episodes.

    On 2026-11-08 a 116-task suite restarted four times; the final process reported
    "Completeness: 0/116 tasks attempted" while 104 finished episodes were on disk.
    A resumable run is a DIRECTORY, not a process.
    """
    from awenv.recorder import Recorder

    first = Recorder(str(tmp_path), "run-a", config={"protocol": "aw-main-v1"})
    first.start_episode({"episode_id": "ep1", "task_name": "A", "task_idx": 0, "seed": 30})
    first.add_step({"i": 0, "episode_id": "ep1", "action_ok": True, "confidence": 0.9})
    first.finish_episode({"success": True, "declared_done": True, "n_steps": 1})

    second = Recorder(str(tmp_path), "run-b", config={"protocol": "aw-main-v1"})
    second.start_episode({"episode_id": "ep2", "task_name": "B", "task_idx": 1, "seed": 30})
    second.add_step({"i": 0, "episode_id": "ep2", "action_ok": True, "confidence": 0.9})
    second.finish_episode({"success": False, "declared_done": False, "n_steps": 1})

    ids = {e["episode_id"] for e in second.episodes()}
    assert ids == {"ep1", "ep2"}, "a resumed report must include episodes from earlier runs"
    assert len(second.steps()) == 2


def test_report_does_not_pick_up_another_directorys_episodes(tmp_path):
    """Scoping by directory must not leak episodes from a different run root on a shared index."""
    from awenv.recorder import Recorder

    a = Recorder(str(tmp_path / "a"), "run-a", config={"protocol": "aw-main-v1"})
    a.start_episode({"episode_id": "in-a", "task_name": "A", "task_idx": 0, "seed": 30})
    a.finish_episode({"success": True, "declared_done": True, "n_steps": 0})

    b = Recorder(str(tmp_path / "b"), "run-b", config={"protocol": "aw-main-v1"})
    assert b.episodes() == [], "a separate run root must not see the other root's episodes"
