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
    assert plan.loop_break is False  # detection only unless `loop.break: true`


def test_masking_is_off_unless_the_config_asks_for_it():
    """Q2-ii verified-failure masking is a policy, so it is opt-in and visible in the dry run."""
    from run_bench import describe

    plan = plan_from_config(_cfg())
    assert plan.mask is False
    assert "mask off" in describe(plan)

    cfg = _cfg()
    cfg["policy"]["loop_detection"]["mask"] = True
    plan = plan_from_config(cfg)
    assert plan.mask is True
    assert "mask ON" in describe(plan)


def test_the_mask_flag_reaches_the_agent():
    """`run()` must hand `plan.mask` to AgentCore, exactly like `loop_break`."""
    import run_bench

    cfg = _cfg()
    cfg["policy"]["loop_detection"]["mask"] = True
    assert run_bench.agent_kwargs(plan_from_config(cfg))["mask"] is True
    assert run_bench.agent_kwargs(plan_from_config(_cfg()))["mask"] is False


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


def test_all_tasks_can_be_enumerated_without_importing_android_world(tmp_path):
    """`suite.tasks: all` must not depend on an importable `android_world`.

    In this image `from android_world import suite_assets` raises ImportError against the
    installed /usr/local/lib/python3.11/site-packages/android_world, which is a different copy
    from the /app source tree the server runs. A 116-task run died on that import before
    touching the device. The budget table already lists every task, so it is the source.
    """
    import json

    import run_bench

    budgets = tmp_path / "budgets.json"
    budgets.write_text(json.dumps({
        "CameraTakePhoto": {"complexity": 1.0, "max_steps": 10},
        "MarkorCreateNote": {"complexity": 1.4, "max_steps": 14},
    }))

    names = run_bench._all_tasks(str(budgets))
    assert names == ["CameraTakePhoto", "MarkorCreateNote"]


# ---------------------------------------------------------------- official instances (H2)
# Upstream android_world@55ddaf5 suite_utils.py:97-103, 149-153:
#   instance_seed = int(sha256(f"{seed}_{name}_{i}").hexdigest(), 16) % 2**32
#   random.seed(instance_seed); params = task.generate_random_params(); params["seed"] = instance_seed
import hashlib  # noqa: E402
import random  # noqa: E402

from run_bench import instance_seed, instantiate_task  # noqa: E402


class _RandomTask:
    calls = 0

    def __init__(self, params):
        self.params = params

    @classmethod
    def generate_random_params(cls):
        cls.calls += 1
        return {"x": random.random()}


def test_instance_seed_matches_upstream_formula():
    expected = int(hashlib.sha256(b"30_ContactsAddContact_0").hexdigest(), 16) % 2**32
    assert instance_seed(30, "ContactsAddContact", 0) == expected


def test_instantiate_is_deterministic():
    a = instantiate_task(_RandomTask, "T", 30)
    random.random()  # disturb the global RNG between the two instantiations
    b = instantiate_task(_RandomTask, "T", 30)
    assert a.aw_params["x"] == b.aw_params["x"]
    assert a.aw_params["seed"] == instance_seed(30, "T", 0)


def test_replay_params_are_not_reseeded():
    _RandomTask.calls = 0
    t = instantiate_task(_RandomTask, "T", 30, replay={"x": 7})
    assert t.aw_params["x"] == 7
    assert _RandomTask.calls == 0
