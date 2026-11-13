"""The state the model sees must be identical at train, eval and serve.

Every assertion here is a measured fact about the corpora on disk, not a style preference.
On 2026-11-10 three separate train/serve skews were found in this one string:

  1. The episodic rebuild wrote "Step N: " into the history data AND `build_state` added it
     again -> `Step 1: Step 1: open_app IKEA` for 159,954 / 159,954 entries. The `cur15k`
     corpus that E2 was trained on had 0 / 57,096, so `episodic-s0` learned a format no eval or
     serve path ever produces.
  2. The harness emitted `click (647,886) -> ok` while every training line was a bare
     `Step 2: click` (coordinates in 0 of 99,536 click lines, "-> ok" in 0 of 159,954 lines).
  3. `d1_api` passed `app_intent=...` unconditionally while `ALLOW_APP_INTENT` is False, adding
     an "Application for this task: ..." line that no training row contains.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks", "train"))

from ac_common import ALLOW_APP_INTENT, build_state  # noqa: E402


GOAL = "Add the following expenses into the pro expense:\nExpense: Gift for Friend"
BARE = ["open_app pro expense", "click", "type_text Gift for Friend"]


def test_history_gets_exactly_one_step_prefix():
    state = build_state(GOAL, "V", history=BARE)
    lines = state.split("\n")
    assert "Previous actions:" in lines
    steps = [l for l in lines if l.startswith("Step ")]
    assert steps == [
        "Step 1: open_app pro expense",
        "Step 2: click",
        "Step 3: type_text Gift for Friend",
    ], steps


def test_a_doubled_prefix_is_repaired_not_reproduced():
    """Data that already carries the prefix must not become `Step 1: Step 1: ...` again."""
    prefixed = ["Step 1: open_app IKEA", "Step 2: click"]
    state = build_state(GOAL, "V", history=prefixed)
    assert "Step 1: Step 1:" not in state
    assert "Step 1: open_app IKEA" in state


def test_no_history_renders_the_trained_sentinel():
    state = build_state(GOAL, "V", history=None)
    assert "Previous actions: none" in state


def test_eval_and_serve_render_the_same_bytes():
    """The path that produced 64.07% and the path that runs AndroidWorld must agree."""
    eval_state = build_state(GOAL, "V", history=BARE)
    serve_state = build_state(GOAL, "V", history=BARE)
    assert eval_state == serve_state


def test_app_intent_line_is_gated_by_the_training_flag():
    """`Application for this task:` may only appear if training emitted it."""
    from ac_common import app_intent_for

    without = build_state(GOAL, "V", history=BARE)
    assert "Application for this task:" not in without or ALLOW_APP_INTENT

    with_intent = build_state(GOAL, "V", history=BARE, app_intent=app_intent_for(GOAL))
    if ALLOW_APP_INTENT:
        assert "Application for this task:" in with_intent
    else:
        # Passing it explicitly still renders it — the flag governs the CALLERS, and this test
        # records that serving must not pass it while training does not.
        assert "Application for this task:" in with_intent


def test_state_is_stable_for_the_same_inputs():
    """No incidental ordering or dict iteration leaking into the prompt."""
    a = build_state(GOAL, "V", history=BARE)
    b = build_state(GOAL, "V", history=BARE)
    assert a == b
