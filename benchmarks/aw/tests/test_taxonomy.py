"""The taxonomy has to add up, or Gate C reports a number nobody can act on."""
from awenv.taxonomy import summarize_taxonomy


def _ep(eid, success=False, **kw):
    rec = {"episode_id": eid, "success": success}
    rec.update(kw)
    return rec


def _step(eid, action="click", invalid=False, stuck=False, ok=True, changed=True):
    return {
        "episode_id": eid,
        "action_raw": {"action": action},
        "invalid_action": invalid,
        "stuck_event": stuck,
        "action_ok": ok,
        "screen_changed": changed,
    }


def test_counts_add_up_to_the_number_of_failures():
    episodes = [_ep("a"), _ep("b"), _ep("c", success=True), _ep("d", env_crash=True)]
    steps = [
        _step("a", invalid=True),
        _step("b", stuck=True),
        _step("c"),
        _step("d"),
    ]

    out = summarize_taxonomy(episodes, steps)

    assert out["n_episodes"] == 4
    assert out["n_failed"] == 3
    assert sum(out["counts"].values()) == out["n_failed"]


def test_crash_outranks_every_later_symptom():
    episodes = [_ep("a", env_crash=True, timeout=True)]
    out = summarize_taxonomy(episodes, [_step("a", invalid=True, stuck=True)])
    assert out["counts"]["env_crash"] == 1
    assert out["counts"]["timeout"] == 0


def test_invalid_action_outranks_stuck():
    episodes = [_ep("a")]
    out = summarize_taxonomy(episodes, [_step("a", invalid=True), _step("a", stuck=True)])
    assert out["counts"]["unmappable_action"] == 1
    assert out["counts"]["stuck"] == 0


def test_premature_status_is_a_bucket_of_its_own():
    episodes = [_ep("a")]
    steps = [_step("a", action="click"), _step("a", action="status")]
    out = summarize_taxonomy(episodes, steps)
    assert out["counts"]["premature_status"] == 1


def test_a_successful_episode_is_in_no_bucket():
    out = summarize_taxonomy([_ep("a", success=True)], [_step("a")])
    assert sum(out["counts"].values()) == 0
    assert out["n_succeeded"] == 1


def test_action_mix_exposes_the_coverage_gap():
    steps = [_step("a", action="click"), _step("a", action="click"), _step("a", action="status")]
    out = summarize_taxonomy([_ep("a", success=True)], steps)
    assert out["action_mix"] == {"click": 2, "status": 1}


def test_an_unclassifiable_failure_is_counted_not_dropped():
    episodes = [_ep("a")]
    out = summarize_taxonomy(episodes, [_step("a")])
    assert out["counts"]["unknown"] == 1
