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
