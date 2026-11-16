"""score_exec on a hand-built executor file: denominators, the inside/last split, and AUROC."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_exec as S  # noqa: E402


def test_auroc_matches_hand_counts():
    assert S.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert S.auroc([0.5], [0.5]) == 0.5
    assert S.auroc([0.3, 0.9], [0.5]) == 0.5


def test_score_splits_action_rows_and_done_rows():
    rows = [{"ep": 7, "step": 0, "aug": "span"}, {"ep": 7, "step": 1, "aug": "span"},
            {"ep": "done:7", "step": 2, "aug": "span_done"}, {"ep": 8, "step": 0, "aug": "goal"}]
    preds = [{"episode_id": 7, "step": 0, "pred": "click", "correct": True, "probs": "{'status': 0.1}"},
             {"episode_id": 7, "step": 1, "pred": "status", "correct": False, "probs": "{'status': 0.7}"},
             {"episode_id": "done:7", "step": 2, "pred": "status", "correct": True, "probs": "{'status': 0.9}"}]
    r = S.score(preds, rows)
    assert r["action rows"] == 2 and r["step success %"] == 50.0
    assert r["false done inside a local goal"] == "0/1" and r["false done on its last action"] == "1/1"
    assert r["done rows"] == 1 and r["done recall %"] == 100.0 and r["done precision %"] == 50.0
    assert r["AUROC p(status) done vs action"] == 1.0 and r["rows without a prediction"] == 0


def test_the_last_action_of_an_episode_without_a_done_row_is_a_last_action_not_inside():
    # an AndroidControl test episode ends at its last action: its last local goal has no done row after it
    rows = [{"ep": 7, "step": 0, "aug": "span", "local_goal": "a"}, {"ep": 7, "step": 1, "aug": "span", "local_goal": "a"},
            {"ep": "done:7", "step": 2, "aug": "span_done", "local_goal": "a"},
            {"ep": 7, "step": 2, "aug": "span", "local_goal": "b"}, {"ep": 7, "step": 3, "aug": "span", "local_goal": "b"},
            {"ep": 9, "step": 0, "aug": "span", "local_goal": "b"}]
    preds = [{"episode_id": e, "step": s, "pred": "status", "correct": False, "probs": "{'status': 0.6}"}
             for e, s in ((7, 0), (7, 1), (7, 2), (7, 3), (9, 0))] + \
            [{"episode_id": "done:7", "step": 2, "pred": "status", "correct": True, "probs": "{'status': 0.9}"}]
    r = S.score(preds, rows)
    assert r["false done inside a local goal"] == "2/2"  # (7,0) and (7,2)
    assert r["false done on its last action"] == "3/3"  # (7,1) before a done row; (7,3) and (9,0) end their episode
