"""Paired comparison and denominator-carrying output."""
from awenv.report import format_markdown, paired_bootstrap_delta, paired_ids
from awenv.metrics import summarize


def _ep(eid, success):
    return {"episode_id": eid, "success": success}


def test_paired_ids_uses_intersection():
    a = [_ep("x", True), _ep("y", False)]
    b = [_ep("y", True), _ep("z", True)]
    assert paired_ids(a, b) == ["y"]


def test_identical_agents_have_zero_delta():
    a = [_ep("1", True), _ep("2", False), _ep("3", True), _ep("4", False)]
    b = [_ep("1", True), _ep("2", False), _ep("3", True), _ep("4", False)]
    out = paired_bootstrap_delta(a, b, n_boot=500)
    assert out["delta"] == 0.0
    assert out["n_pairs"] == 4
    assert out["excludes_zero"] is False


def test_dominant_agent_has_positive_delta():
    a = [_ep(str(i), True) for i in range(20)]
    b = [_ep(str(i), False) for i in range(20)]
    out = paired_bootstrap_delta(a, b, n_boot=500)
    assert out["delta"] == 1.0
    assert out["excludes_zero"] is True


def test_unpaired_returns_empty_not_a_number():
    out = paired_bootstrap_delta([_ep("1", True)], [_ep("2", True)])
    assert out["n_pairs"] == 0
    assert out["delta"] is None


def test_markdown_carries_denominators_and_proxy_labels():
    episodes = [{"episode_id": "e1", "success": True, "n_steps": 3,
                 "coverage": {"emittable": 2, "total": 2}}]
    steps = [{"episode_id": "e1", "i": 0, "action_raw": {"action": "click"},
              "action_ok": True, "invalid_action": False, "screen_changed": True,
              "confidence": 0.8}]
    md = format_markdown(summarize(episodes, steps), title="t")
    assert "1/1" in md
    assert "proxy" in md.lower()
    assert "ASR is not SR" in md


def test_incomplete_run_says_so_before_the_table():
    """Ten of 116 tasks must never read as a finished benchmark."""
    report = {"task": {"sr": {"num": 1, "den": 10, "value": 0.1}, "sr_wilson95": [0.02, 0.40],
                        "sr_reachable": {"num": 0, "den": 0, "value": None},
                        "coverage": {"num": 0, "den": 0, "value": None},
                        "timeout_rate": {"num": 0, "den": 10, "value": 0.0},
                        "crash_rate": {"num": 0, "den": 10, "value": 0.0},
                        "declared_done_rate": {"num": 6, "den": 10, "value": 0.6},
                        "steps_to_success_median": 3.0,
                        "recovery_rate": {"num": 0, "den": 0, "value": None}},
              "step": {"n_emitted_actions": 0, "asr_exec": {"num": 0, "den": 0, "value": None},
                       "invalid_action_rate": {"num": 0, "den": 0, "value": None},
                       "env_reject_rate": {"num": 0, "den": 0, "value": None},
                       "stuck_rate": {"num": 0, "den": 0, "value": None},
                       "grounding_error_proxy": {"num": 0, "den": 0, "value": None}},
              "calibration_proxy": {"ece10": None, "coverage_risk": []},
              "completeness": {"planned": 116, "completed": 10, "complete": False}}

    out = format_markdown(report, title="t")

    assert "INCOMPLETE" in out
    assert "10/116" in out
    assert out.index("INCOMPLETE") < out.index("SR (all)"), "the caveat must come before the number"


def test_complete_run_is_labelled_complete():
    report = {"task": {"sr": {"num": 2, "den": 2, "value": 1.0}, "sr_wilson95": [0.5, 1.0],
                        "sr_reachable": {"num": 0, "den": 0, "value": None},
                        "coverage": {"num": 0, "den": 0, "value": None},
                        "timeout_rate": {"num": 0, "den": 2, "value": 0.0},
                        "crash_rate": {"num": 0, "den": 2, "value": 0.0},
                        "declared_done_rate": {"num": 2, "den": 2, "value": 1.0},
                        "steps_to_success_median": 2.0,
                        "recovery_rate": {"num": 0, "den": 0, "value": None}},
              "step": {"n_emitted_actions": 0, "asr_exec": {"num": 0, "den": 0, "value": None},
                       "invalid_action_rate": {"num": 0, "den": 0, "value": None},
                       "env_reject_rate": {"num": 0, "den": 0, "value": None},
                       "stuck_rate": {"num": 0, "den": 0, "value": None},
                       "grounding_error_proxy": {"num": 0, "den": 0, "value": None}},
              "calibration_proxy": {"ece10": None, "coverage_risk": []},
              "completeness": {"planned": 2, "completed": 2, "complete": True}}

    out = format_markdown(report, title="t")

    assert "complete" in out.lower()
    assert "INCOMPLETE" not in out
