"""The nightly comparison is the only thing that makes a regression visible.

Without it the harness runs at 03:00, writes a number nobody reads, and a silent breakage
survives until someone quotes the stale number. Each alert below is a failure the program
has already had once: a crashed emulator, a latency jump from GPU contention, and an
invalid-action rate that crept up as the action space drifted.
"""
from awenv.report import compare_nightly, summarize


def _report(sr=8, n=8, crashes=0, model_p50=100.0, invalid=0.10):
    episodes = [
        {
            "episode_id": "e%d" % i,
            "success": i < sr,
            "env_crash": i < crashes,
            "timeout": False,
            "coverage": None,
        }
        for i in range(n)
    ]
    steps = [
        {
            "episode_id": "e%d" % i,
            "i": 0,
            "action_raw": {"action": "click"},
            "confidence": 0.9,
            "action_ok": True,
            "invalid_action": i < round(invalid * n),
            "screen_changed": True,
            "model_ms": model_p50,
            "env_ms": 10.0,
            "emitted": True,
        }
        for i in range(n)
    ]
    return summarize(episodes, steps)


def test_identical_runs_raise_nothing():
    assert compare_nightly(_report(), _report()) == []


def test_sr_drop_of_two_tasks_alerts():
    alerts = compare_nightly(_report(sr=5), _report(sr=8))
    assert any("SR" in a for a in alerts)


def test_sr_drop_of_one_task_does_not_alert():
    assert compare_nightly(_report(sr=7), _report(sr=8)) == []


def test_any_crash_alerts_even_when_sr_holds():
    alerts = compare_nightly(_report(crashes=1), _report(crashes=0))
    assert any("crash" in a.lower() for a in alerts)


def test_latency_regression_of_30_percent_alerts():
    alerts = compare_nightly(_report(model_p50=130.0), _report(model_p50=100.0))
    assert any("latency" in a.lower() or "p50" in a.lower() for a in alerts)


def test_latency_regression_under_the_threshold_is_quiet():
    assert compare_nightly(_report(model_p50=120.0), _report(model_p50=100.0)) == []


def test_latency_at_exactly_the_threshold_is_quiet():
    """The plan says "> 25%", so 25.0% is not an alert."""
    assert compare_nightly(_report(model_p50=125.0), _report(model_p50=100.0)) == []


def test_invalid_action_rise_of_five_points_alerts():
    alerts = compare_nightly(_report(invalid=0.20), _report(invalid=0.10))
    assert any("invalid" in a.lower() for a in alerts)


def test_thresholds_are_configurable():
    alerts = compare_nightly(
        _report(sr=5, model_p50=200.0),
        _report(sr=8, model_p50=100.0),
        thresholds={"sr_drop_tasks": 5, "model_p50_regress": 2.0, "crash_rate": 0.0,
                    "invalid_action_delta": 0.05},
    )
    assert alerts == []
