import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_card import card_stats  # noqa: E402


def test_counts_full_episodes_and_history():
    rows = [
        {"ep": 1, "step": 0, "n_steps": 1, "history": [], "action": {"action_type": "click"}},
        {"ep": 1, "step": 1, "n_steps": 1, "history": ["click"], "action": {"action_type": "status"}},
        {"ep": 2, "step": 3, "n_steps": 4, "history": ["a", "b", "c"], "action": {"action_type": "click"}},
    ]
    s = card_stats(rows)
    assert s["episodes"] == 2 and s["episodes_with_every_step"] == 1
    assert s["steps_kept_per_episode(6=6+)"] == {1: 1, 2: 1}
    assert s["history_len_per_row(5=5+)"] == {0: 1, 1: 1, 3: 1}
    assert s["sources"] == {"androidcontrol": 3}


def test_executor_rows_count_under_their_source_episode_and_every_action_step_counts_as_full():
    rows = [
        {"ep": 5, "step": 0, "n_steps": 2, "history_v3": [], "action": {"action_type": "click"}},
        {"ep": 5, "step": 1, "n_steps": 2, "history_v3": ["x"], "action": {"action_type": "click"}},
        {"ep": "done:5", "step": 2, "n_steps": 2, "history_v3": ["x", "y"], "action": {"action_type": "status"}},
    ]
    s = card_stats(rows)
    assert s["episodes"] == 1 and s["episodes_with_every_step"] == 1
