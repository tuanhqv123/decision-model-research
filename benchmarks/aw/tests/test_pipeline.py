"""Wiring test: adapter -> recorder -> metrics, with no device and no model.

This is the Gate B preview. If a number cannot travel from a decision to the report
with its denominator intact, the harness is wrong before any emulator is involved.
"""
from awenv.adapter import map_decision
from awenv.metrics import summarize
from awenv.recorder import Recorder


def _run_episode(rec, episode_id, decisions, success):
    rec.start_episode(
        {
            "episode_id": episode_id,
            "task_name": "ClockStopWatchRunning",
            "task_idx": 0,
            "seed": 30,
            "agent": "d1_2b",
            "checkpoint": "E2@938",
            "protocol": "aw-main-v1",
            "step_budget": 20,
            "coverage": {"emittable": 2, "total": 2},
        }
    )
    invalid = 0
    errors = 0
    for i, dec in enumerate(decisions):
        mapped = map_decision(dec, allow_app_fallback=False, goal="turn on the stopwatch")
        invalid += 1 if mapped.invalid else 0
        rec.add_step(
            {
                "i": i,
                "action_raw": {"action": dec.get("action"), "confidence": dec.get("confidence", 0.5)},
                "action_executed": mapped.actions[0] if mapped.actions else None,
                "confidence": dec.get("confidence", 0.5),
                "gate_score": dec.get("confidence", 0.5) * 0.9,
                "invalid_action": mapped.invalid,
                "action_ok": not mapped.invalid,
                "screen_changed": not mapped.invalid,
                "grounding_error": None,
                "model_ms": 100.0,
                "env_ms": 900.0,
            }
        )
        if mapped.done:
            break
    rec.finish_episode(
        {
            "success": success,
            "declared_done": any(d.get("action") == "status" for d in decisions),
            "n_steps": len(decisions),
            "n_invalid_actions": invalid,
            "n_action_errors": errors,
            "env_crash": False,
            "timeout": not success,
            "latency": {"model_p50": 100.0, "e2e_total_ms": 1000.0 * len(decisions)},
            "tokens_generated": 0,
        }
    )


def test_numbers_travel_from_decision_to_report(tmp_path):
    rec = Recorder(str(tmp_path), "run1")
    _run_episode(
        rec,
        "p1_a",
        [
            {"action": "open_app", "app_name": "stopwatch", "confidence": 0.9},
            {"action": "click", "point": [500, 1200], "confidence": 0.8},
            {"action": "status", "confidence": 0.7},
        ],
        success=True,
    )
    _run_episode(
        rec,
        "p1_b",
        [
            {"action": "click", "point": [None, None], "confidence": 0.6},
            {"action": "double_tap", "point": [1, 2], "confidence": 0.4},
        ],
        success=False,
    )
    advisory = rec.episodes()
    steps = rec.steps()
    rec.close()

    report = summarize(advisory, steps)
    task = report["task"]
    step = report["step"]

    assert task["sr"]["num"] == 1 and task["sr"]["den"] == 2
    assert task["coverage"]["num"] == 4 and task["coverage"]["den"] == 4
    assert step["invalid_action_rate"]["num"] == 2
    assert step["asr_exec"]["den"] == 5
    assert step["asr_exec"]["value"] == 3 / 5
    assert report["calibration_proxy"]["ece10"] is not None


def test_status_ends_the_episode_but_is_not_success(tmp_path):
    rec = Recorder(str(tmp_path), "run1")
    _run_episode(rec, "p1_c", [{"action": "status", "confidence": 0.99}], success=False)
    episodes = rec.episodes()
    rec.close()
    report = summarize(episodes, [])
    assert report["task"]["declared_done_rate"]["value"] == 1.0
    assert report["task"]["sr"]["value"] == 0.0
