"""Metric definitions: denominators and the ASR/SR distinction."""
from awenv.metrics import (
    Ratio,
    asr_exec,
    coverage_risk,
    coverage_split,
    ece,
    emitted_actions,
    grounding_error_proxy,
    invalid_action_rate,
    recovery_rate,
    stuck_rate,
    success_rate,
    summarize,
    wilson,
)


def _step(i=0, action="click", ok=True, invalid=False, changed=True, conf=0.8, grounding=None):
    rec = {
        "i": i,
        "action_raw": {"action": action},
        "action_ok": ok,
        "invalid_action": invalid,
        "screen_changed": changed,
        "confidence": conf,
    }
    if grounding is not None:
        rec["grounding_error"] = grounding
    return rec


def _ep(eid="e1", success=False, steps=2, coverage=None, **kw):
    rec = {
        "episode_id": eid,
        "success": success,
        "n_steps": steps,
        "timeout": False,
        "env_crash": False,
        "declared_done": False,
    }
    if coverage is not None:
        rec["coverage"] = coverage
    rec.update(kw)
    return rec


def test_ratio_without_denominator_has_no_value():
    assert Ratio(0, 0).value is None
    assert Ratio(1, 4).as_dict() == {"num": 1, "den": 4, "value": 0.25}


def test_wilson_bounds_and_empty():
    lo, hi = wilson(1, 6)
    assert 0.0 <= lo < 1 / 6 < hi <= 1.0
    assert wilson(0, 0) == (None, None)


def test_sr_and_asr_are_different_quantities():
    steps = [_step(i=i, ok=(i < 9)) for i in range(10)]
    episodes = [_ep("e1", success=False)]
    assert asr_exec(steps).value == 0.9
    assert success_rate(episodes).value == 0.0
    out = summarize(episodes, steps)
    assert out["task"]["sr"]["value"] == 0.0
    assert out["step"]["asr_exec"]["value"] == 0.9


def test_invalid_action_denominator_is_all_emitted_actions():
    steps = [_step(invalid=True, ok=False), _step(), _step()]
    assert invalid_action_rate(steps).as_dict() == {"num": 1, "den": 3, "value": 1 / 3}


def test_grounding_proxy_only_counts_spatial_actions():
    steps = [
        _step(action="click", grounding=False),
        _step(action="long_press", grounding=True),
        _step(action="scroll", grounding=None),
        _step(action="click", grounding=None),
    ]
    assert grounding_error_proxy(steps).as_dict() == {"num": 1, "den": 2, "value": 0.5}


def test_stuck_rate_denominator_is_executed_actions():
    steps = [
        _step(ok=True, changed=False),
        _step(ok=True, changed=True),
        _step(ok=False, changed=False),
        _step(ok=False, invalid=True),
    ]
    assert stuck_rate(steps).as_dict() == {"num": 1, "den": 2, "value": 0.5}


def test_recovery_rate_conditions_on_stress():
    episodes = [_ep("a", success=True), _ep("b", success=False), _ep("c", success=True)]
    steps_by = {
        "a": [_step(ok=False)],
        "b": [_step(invalid=True)],
        "c": [_step(ok=True)],
    }
    assert recovery_rate(episodes, steps_by).as_dict() == {"num": 1, "den": 2, "value": 0.5}


def test_coverage_split_separates_expressible_from_not():
    episodes = [
        _ep("a", success=True, coverage={"emittable": 3, "total": 3}),
        _ep("b", success=False, coverage={"emittable": 3, "total": 4}),
        _ep("c", success=True, coverage={"emittable": 2, "total": 2}),
    ]
    out = coverage_split(episodes)
    assert out["sr_all"]["value"] == 2 / 3
    assert out["sr_reachable"]["value"] == 1.0
    assert out["n_reachable"] == 2
    assert out["coverage"]["num"] == 8 and out["coverage"]["den"] == 9


def test_ece_zero_when_calibrated():
    confs = [0.0] * 10 + [1.0] * 10
    labels = [False] * 10 + [True] * 10
    assert ece(confs, labels) == 0.0


def test_ece_positive_when_overconfident():
    confs = [0.9] * 10
    labels = [False] * 10
    assert abs(ece(confs, labels) - 0.9) < 1e-9


def test_coverage_risk_monotone_in_tau():
    confs = [0.4, 0.6, 0.75, 0.85, 0.96]
    labels = [True, True, True, True, True]
    curve = coverage_risk(confs, labels, taus=(0.5, 0.9))
    assert curve[0]["coverage"] == 0.8
    assert curve[1]["coverage"] == 0.2
    assert curve[0]["risk"] == 0.0


def test_summarize_carries_denominators_everywhere():
    out = summarize([_ep("a", success=True)], [_step()])
    for group in ("task", "step"):
        for name, payload in out[group].items():
            if isinstance(payload, dict) and "num" in payload:
                assert set(payload) == {"num", "den", "value"}, name
    assert out["task"]["sr"]["den"] == 1
    assert out["task"]["sr_wilson95"][0] is not None


def test_terminal_status_step_is_not_in_the_action_denominator():
    """A `status` emits no action, so it must not dilute ASR or the invalid rate."""
    ok_step = _step()
    status_step = {"episode_id": "e1", "action_raw": {"action": "status"},
                   "action_ok": None, "invalid_action": False, "emitted": False}

    asr = asr_exec([ok_step, status_step])
    assert asr.num == 1 and asr.den == 1
    assert invalid_action_rate([ok_step, status_step]).den == 1
    assert emitted_actions([ok_step, status_step]) == 1
