"""AC-High test rows for v3: one fixed app list for every row, no terminal `status` rows (the benchmark has none), and
every benchmark step present exactly once with the same action type."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_actest_v3 import TYPE_OF, alignment, app_list  # noqa: E402


def test_app_list_is_fixed_deduplicated_case_insensitively_and_capped():
    apps = app_list(["Gmail", "clock"], ["gmail", "Zoom", "Clock"], cap=702)
    assert apps == ["clock", "Gmail", "Zoom"]
    try:
        app_list([f"a{i}" for i in range(703)], [], cap=702)
        assert False, "a list longer than d1's option codes must be refused"
    except SystemExit:
        pass


def test_alignment_finds_missing_extra_and_type_mismatch():
    test = [{"episode": {"episode_id": 1}, "step_check_pams": [{"action": "click"}, {"action": "swipe"}]},
            {"episode": {"episode_id": 2}, "step_check_pams": [{"action": "type"}]}]
    rows = [{"ep": 1, "step": 0, "action": {"action_type": "click"}},
            {"ep": 1, "step": 1, "action": {"action_type": "scroll"}},
            {"ep": 2, "step": 0, "action": {"action_type": "open_app"}},
            {"ep": 3, "step": 0, "action": {"action_type": "wait"}}]
    a = alignment(rows, test)
    assert a["benchmark_steps"] == 3 and a["matched"] == 2 and a["missing"] == 0
    assert a["type_mismatch"] == 1 and a["extra"] == 1
    assert TYPE_OF["input_text"] == "type" and TYPE_OF["type_text"] == "type" and TYPE_OF["scroll"] == "swipe"
