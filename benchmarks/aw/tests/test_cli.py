"""The CLI must be runnable, and checkable, without an emulator.

``--dry-run`` is not a convenience: it is how the protocol in a config gets reviewed before
a 116-task run spends an hour on it. If the plan it prints is wrong, the run is wrong, and
this is the cheapest place to find out.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_bench import ConfigError, load_config, plan_from_config  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _cfg(name="nightly_smoke.yaml"):
    return load_config(os.path.join(CONFIGS, name))


def test_loads_a_config_and_keeps_the_protocol_string():
    cfg = _cfg()
    assert cfg["protocol"] == "aw-main-v1"
    assert cfg["agent"] == "d1_sft"


def test_missing_protocol_is_a_config_error():
    cfg = _cfg()
    del cfg["protocol"]
    with pytest.raises(ConfigError, match="protocol"):
        plan_from_config(cfg)


def test_dry_run_plan_matches_the_configured_suite():
    plan = plan_from_config(_cfg())

    assert plan.n_tasks == 8
    assert [s.task_name for s in plan.specs][0] == "ContactsAddContact"
    assert plan.step_budget == 12
    assert plan.specs[0].pairing_key == "ContactsAddContact|0|30"


def test_allow_app_fallback_must_be_off_for_a_headline_run():
    cfg = _cfg()
    cfg["policy"]["allow_app_fallback"] = True
    with pytest.raises(ConfigError, match="fallback"):
        plan_from_config(cfg)


def test_fallback_is_allowed_when_the_run_declares_itself_diagnostic():
    cfg = _cfg()
    cfg["policy"]["allow_app_fallback"] = True
    cfg["suite"]["diagnostic"] = True
    plan = plan_from_config(cfg)
    assert plan.allow_app_fallback is True
    assert plan.diagnostic is True


def test_all_tasks_config_is_rejected_without_android_world():
    """`tasks: all` needs the suite enumeration from AndroidWorld, not a guess."""
    cfg = _cfg("main_116.yaml")
    with pytest.raises(ConfigError, match="android_world"):
        plan_from_config(cfg)


def test_unknown_task_name_is_rejected_before_any_device_work():
    from run_bench import ConfigError as CE
    from run_bench import validate_task_names

    with pytest.raises(CE, match="unknown task name"):
        validate_task_names(["ContactsAddContact", "TypoTask"], ["ContactsAddContact"])
    validate_task_names(["ContactsAddContact"], ["ContactsAddContact"])


def test_plan_carries_the_loop_detection_setting():
    plan = plan_from_config(_cfg())
    assert plan.stuck_k == 3


def test_report_carries_the_per_episode_success_vector(tmp_path):
    """Gate B compares two runs by this vector; a total cannot localise a difference."""
    from awenv.recorder import Recorder
    from run_bench import _report

    rec = Recorder(str(tmp_path), "r1", config={})
    for i, ok in enumerate([True, False]):
        eid = "p1_t000%d_gate_s30_d1_sft" % i
        rec.start_episode({"episode_id": eid, "task_name": "T%d" % i, "seed": 30})
        rec.finish_episode({"episode_id": eid, "episode_id_ok": True, "success": ok, "n_steps": 3})
    report = _report(rec)

    assert [e["success"] for e in report["episodes"]] == [True, False]
    assert report["episodes"][0]["episode_id"].startswith("p1_")
