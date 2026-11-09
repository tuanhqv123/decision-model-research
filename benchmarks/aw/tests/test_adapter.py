"""The adapter must never turn an unfaithful decision into a clean-looking action."""
from awenv.adapter import (
    AW_ACTION_SPACE,
    EMITTABLE,
    NOT_EMITTED,
    map_decision,
)


def test_click_with_point():
    m = map_decision({"action": "click", "point": [540.4, 1200.6]})
    assert m.actions == [{"action_type": "click", "x": 540, "y": 1201}]
    assert not m.invalid and not m.done


def test_click_without_point_is_flagged_not_silent():
    m = map_decision({"action": "click", "point": [None, None]})
    assert m.invalid is True
    assert m.reason == "point_missing"
    assert m.actions == [{"action_type": "wait"}]


def test_long_press_with_point():
    m = map_decision({"action": "long_press", "point": [10, 20]})
    assert m.actions[0]["action_type"] == "long_press"


def test_type_text_expands_to_click_then_input():
    m = map_decision({"action": "type_text", "point": [5, 6], "text": "hello world"})
    assert [a["action_type"] for a in m.actions] == ["click", "input_text"]
    assert m.actions[1]["text"] == "hello world"
    assert not m.invalid


def test_type_text_without_text_is_flagged():
    m = map_decision({"action": "type_text", "point": [5, 6], "text": ""})
    assert m.invalid is True
    assert m.reason == "text_missing"


def test_scroll_direction_validated():
    assert map_decision({"action": "scroll", "direction": "down"}).actions[0]["direction"] == "down"
    bad = map_decision({"action": "scroll", "direction": "sideways"})
    assert bad.invalid is True
    assert bad.reason.startswith("bad_scroll_direction")


def test_open_app_alias_is_canonicalised():
    m = map_decision({"action": "open_app", "app_name": "stopwatch"})
    assert m.actions == [{"action_type": "open_app", "app_name": "clock"}]
    assert m.scaffolding == []


def test_open_app_without_name_is_invalid_by_default():
    m = map_decision({"action": "open_app", "app_name": ""}, goal="add a contact", allow_app_fallback=False)
    assert m.invalid is True
    assert m.reason == "app_name_missing"
    assert m.scaffolding == []


def test_app_fallback_is_stamped_when_enabled():
    m = map_decision({"action": "open_app", "app_name": ""}, goal="add a contact", allow_app_fallback=True)
    assert m.actions == [{"action_type": "open_app", "app_name": "contacts"}]
    assert m.scaffolding == ["app_name_fallback"]


def test_status_terminates():
    m = map_decision({"action": "status"})
    assert m.done is True
    assert m.actions == []


def test_unmappable_action_is_counted():
    m = map_decision({"action": "double_tap", "point": [1, 2]})
    assert m.invalid is True
    assert m.actions == [{"action_type": "wait"}]
    assert m.reason == "unmappable_action:double_tap"


def test_empty_action_is_counted():
    m = map_decision({})
    assert m.invalid is True
    assert m.reason == "unmappable_action:empty"


def test_action_space_gap_is_declared():
    # type_text is our model's name for click+input_text, not an AndroidWorld action.
    assert set(EMITTABLE) | set(NOT_EMITTED) == set(AW_ACTION_SPACE) | {"type_text"}
    assert "keyboard_enter" in NOT_EMITTED
    assert "input_text" not in EMITTABLE
    assert len(EMITTABLE) == 8
