"""Training history must be what the serving agent would write: the rendered gold actions of earlier steps.

AndroidControl's `history` field is the annotators' step INSTRUCTIONS ("Open the YouTube Music app"),
which AndroidWorld can never supply at run time. This rewrites it from the episode's own gold actions with
`ac_common.render_history_line`, the function the agent uses for executed actions.
"""
from benchmarks.train.rerender_history import rerender

EP = [
    {"ep": 7, "step": 0, "goal": "g", "history": [], "action": {"action_type": "open_app", "app_name": "Clock"}},
    {"ep": 7, "step": 1, "goal": "g", "history": ["Open the clock app"], "action": {"action_type": "click", "x": 1, "y": 2}},
    {"ep": 7, "step": 2, "goal": "g", "history": ["Open the clock app", "Tap timer"],
     "action": {"action_type": "type_text", "text": "5"}},
    {"ep": 7, "step": 3, "goal": "g", "history": ["a", "b", "c"], "action": {"action_type": "status"}},
]


def test_history_is_the_rendered_gold_actions_of_earlier_steps():
    out, stats = rerender(EP, EP)
    assert [r["history"] for r in out] == [
        [], ["open_app Clock"], ["open_app Clock", "click"], ["open_app Clock", "click", "type_text 5"]]
    assert stats == {"rows": 4, "kept": 4, "dropped_gap": 0}


def test_rows_with_a_missing_earlier_step_are_dropped_not_guessed():
    rows = [EP[3]]
    episodes = [EP[0], EP[2], EP[3]]  # step 1 missing
    out, stats = rerender(rows, episodes)
    assert out == [] and stats["dropped_gap"] == 1


def test_the_input_rows_are_not_mutated():
    rows = [dict(EP[1])]
    rerender(rows, EP)
    assert rows[0]["history"] == ["Open the clock app"]
