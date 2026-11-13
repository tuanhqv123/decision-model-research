"""Q10 / host equivalence: how far two runs of the same weights on the same task instances agree."""
from replay_agreement import compare_runs, semantic_action


def _step(x, y, bbox, action_type="click", **kw):
    e = {"action_type": action_type, "x": x, "y": y}
    e.update(kw)
    return {"type": "step", "action_executed": e, "click_target": {"bbox": bbox, "label": "L"} if bbox else {}}


def _res(task, score=0.0, done=False):
    return {"type": "result", "task_name": task, "score": score, "declared_done": done, "n_steps": 0,
            "setup_error": None, "unscored": False, "env_crash": False}


def test_click_jitter_on_the_same_element_is_the_same_decision():
    assert semantic_action(_step(646, 2180, [429, 2074, 650, 2295])) == \
        semantic_action(_step(649, 2182, [429, 2074, 650, 2295]))


def test_jitter_that_leaves_the_element_is_a_different_decision():
    """AudioRecorderRecordAudio, 2026-11-10: 646 -> 654 px moved the click off a [429..650] element."""
    assert semantic_action(_step(646, 2180, [429, 2074, 650, 2295])) != semantic_action(_step(654, 2182, None))


def test_compare_counts_outcome_flips_and_first_semantic_divergence():
    a = {"T1": ([_step(1, 1, [0, 0, 9, 9]), _step(2, 2, [0, 0, 9, 9])], _res("T1", 1.0, True)),
         "T2": ([_step(1, 1, [0, 0, 9, 9])], _res("T2"))}
    b = {"T1": ([_step(1, 1, [0, 0, 9, 9]), _step(50, 50, [40, 40, 60, 60])], _res("T1", 0.0, True)),
         "T2": ([_step(3, 3, [0, 0, 9, 9])], _res("T2"))}
    out = compare_runs(a, b)
    assert out["n_tasks"] == 2
    assert out["success_flips"] == ["T1"]
    assert out["done_diffs"] == []
    assert out["identical_trajectories"] == 1
    assert out["first_divergence"] == {"T1": 1}


def test_a_typed_text_change_is_a_divergence_even_on_the_same_field():
    a = {"T": ([_step(1, 1, [0, 0, 9, 9], "input_text", text="DCIM")], _res("T"))}
    b = {"T": ([_step(1, 1, [0, 0, 9, 9], "click")], _res("T"))}
    assert compare_runs(a, b)["first_divergence"] == {"T": 0}


def test_runs_with_no_common_task_render_instead_of_crashing():
    from replay_agreement import render
    text = render(compare_runs({"T1": ([], _res("T1"))}, {"T2": ([], _res("T2"))}), "A", "B")
    assert "0 common tasks" in text and "n/a" in text


def test_an_episode_excluded_in_either_run_is_set_aside_not_compared():
    """Q12, 2026-11-10: B0-overlay's dead-a11y setup errors (0 steps) were counted as diverging at step 0."""
    bad = _res("T2")
    bad["setup_error"] = "reset: RuntimeError: Could not get a11y tree."
    a = {"T1": ([_step(1, 1, [0, 0, 9, 9])], _res("T1")), "T2": ([], bad)}
    b = {"T1": ([_step(1, 1, [0, 0, 9, 9])], _res("T1")), "T2": ([_step(1, 1, [0, 0, 9, 9])], _res("T2", 1.0, True))}
    out = compare_runs(a, b)
    assert out["excluded"] == ["T2"]
    assert out["n_tasks"] == 1
    assert out["success_flips"] == [] and out["done_diffs"] == [] and out["first_divergence"] == {}
