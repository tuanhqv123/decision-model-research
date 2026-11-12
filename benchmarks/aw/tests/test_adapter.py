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


def test_type_text_is_one_input_text_action():
    """Upstream `actuation.py` makes input_text click the focus point itself, so a
    separate click would tap the field twice. It also needs clear_text, otherwise
    typed text appends to whatever the field already held."""
    m = map_decision({"action": "type_text", "point": [5, 6], "text": "hello world"})
    assert [a["action_type"] for a in m.actions] == ["input_text"]
    one = m.actions[0]
    assert one["text"] == "hello world"
    assert one["x"] == 5 and one["y"] == 6
    assert one["clear_text"] is True
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
    # 9 since 2026-11-09: navigate_home was in the vocabulary and the corpus (29 raw rows) but had
    # been omitted from the adapter, so the model could learn an action it could never emit.
    # 10 since 2026-11-12: contract v4 emits `answer` (25 information-retrieval tasks score only through it).
    assert len(EMITTABLE) == 10
    assert "answer" in EMITTABLE and "answer" not in NOT_EMITTED
    assert "navigate_home" in EMITTABLE
    assert "navigate_home" not in NOT_EMITTED


def test_status_is_blocked_when_is_done_is_low():
    """The reason the is_done readout exists: 31/104 episodes declared done at median step 7."""
    m = map_decision({"action": "status", "is_done": 0.12}, done_threshold=0.5)
    assert not m.done
    assert m.actions == [{"action_type": "wait"}]
    assert any(s.startswith("status_blocked_by_is_done") for s in m.scaffolding)


def test_status_is_allowed_when_is_done_is_high():
    m = map_decision({"action": "status", "is_done": 0.88}, done_threshold=0.5)
    assert m.done and m.actions == []


def test_status_without_is_done_behaves_as_before():
    """A checkpoint without the readout must not change behaviour."""
    m = map_decision({"action": "status"})
    assert m.done


def test_navigate_home_maps_through():
    m = map_decision({"action": "navigate_home"})
    assert m.actions == [{"action_type": "navigate_home"}]
    assert not m.invalid


def test_is_done_gate_is_off_by_default():
    """`status` must terminate unless a caller deliberately asks for the gate.

    The gate is a protocol deviation and the signal behind it does not exist: on 824 validation
    rows `step == n_steps` and `action == status` are the same 137 rows, so `is_done` restates the
    action label (benchmarks/results/2026-11-10-episodic-s0-fails-prereg.md). Leaving it on by
    default silently makes every success rate incomparable with the recorded ones.
    """
    m = map_decision({"action": "status", "is_done": 0.01})
    assert m.done is True, "a very low is_done must still terminate when the gate is off"
    assert m.actions == []
    assert m.reason is None


def test_is_done_gate_can_still_be_switched_on_explicitly():
    """Kept only so a controlled experiment can be run; never a default."""
    m = map_decision({"action": "status", "is_done": 0.01}, done_threshold=0.5)
    assert m.done is False
    assert m.reason == "is_done_below_threshold"
    assert [a["action_type"] for a in m.actions] == ["wait"]
