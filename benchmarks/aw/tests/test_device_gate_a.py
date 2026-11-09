"""Gate A: the environment must be provably correct before any number is produced.

These tests need a live emulator running AndroidWorld, so they skip everywhere else rather
than faking a device — a fake here would certify exactly the thing that broke on 2026-11-02.

Run on the emulator host:
    AW_GATE=1 python3 -m pytest benchmarks/aw/tests/test_device_gate_a.py -q

Gate A per docs/compute/AW_PROTOCOL.md: four liveness layers pass, every emittable action
round-trips, `type_text` handles spaces and non-ASCII, and `open_app` works for all 24
canonical apps.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awenv.adapter import CANON_APPS, EMITTABLE  # noqa: E402

GATE = os.environ.get("AW_GATE") == "1"
pytestmark = pytest.mark.skipif(
    not GATE,
    reason="Gate A needs a live emulator; set AW_GATE=1 on the AndroidWorld host",
)


ADB = os.environ.get("AW_ADB", "adb")


@pytest.fixture(scope="module")
def env():
    from awenv.awenv_bridge import build

    return build(console_port=5554, adb_path=ADB, emulator_setup=False)


def test_liveness_passes_all_layers(env, tmp_path):
    """In-process architecture: liveness is about the environment object, not an HTTP sidecar."""
    from awenv.liveness import probe_inprocess

    result = probe_inprocess(env, adb=ADB, serial=os.environ.get("AW_SERIAL"))
    assert result["ok"] is True, result
    assert [l["name"] for l in result["layers"]] == ["device", "screenshot", "noop"]


@pytest.mark.parametrize("action", [a for a in EMITTABLE if a != "status"])
def test_every_emittable_action_roundtrips(env, action):
    from awenv.adapter import map_decision

    decision = {
        "click": {"action": "click", "point": [540, 1200]},
        "long_press": {"action": "long_press", "point": [540, 1200]},
        "type_text": {"action": "type_text", "point": [540, 400], "text": "gate a"},
        "scroll": {"action": "scroll", "direction": "down"},
        "open_app": {"action": "open_app", "app_name": "settings"},
        "navigate_back": {"action": "navigate_back"},
        "wait": {"action": "wait"},
    }[action]

    mapped = map_decision(decision)
    assert mapped.invalid is False, mapped.reason
    for item in mapped.actions:
        env.execute(item)


def test_type_text_handles_spaces_and_non_ascii(env):
    """Three separate claims, because they fail in three different ways.

    1. the adapter must keep the exact unicode payload (a dropped accent is a silent corruption);
    2. the executor must accept it without raising;
    3. the text should reach the a11y tree — checked only when the screen actually offers an
       editable field, and skipped with the reason when it does not, because a click into empty
       space cannot be evidence about typing.
    """
    from awenv.adapter import map_decision

    text = "héllo wörld 123"
    mapped = map_decision({"action": "type_text", "point": [540, 400], "text": text})
    assert mapped.invalid is False, mapped.reason
    assert [a["action_type"] for a in mapped.actions] == ["click", "input_text"]
    assert mapped.actions[1]["text"] == text, "adapter altered the payload"

    before = env.foreground()
    for action in mapped.actions:
        env.execute(action)
    assert env.is_alive()

    editable = [e for e in env.elements()
                if "EditText" in (e.get("class") or "") or "Search" in (e.get("text") or "")]
    if not editable:
        pytest.skip("no editable field on screen (%s); execution was exercised, landing was not"
                    % before)

    field = editable[0]
    x1, y1, x2, y2 = field["bbox"]
    env.execute({"action_type": "click", "x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)})
    env.execute({"action_type": "input_text", "text": text})
    typed = " ".join((e.get("text") or "") for e in env.elements())
    assert "héllo" in typed or "wörld" in typed, "typed text never reached the a11y tree: %r" % typed[:120]


@pytest.mark.parametrize("app", CANON_APPS)
def test_open_app_reaches_every_canonical_app(env, app):
    """`open_app` must change the foreground activity, not just return without raising."""
    from awenv.adapter import map_decision

    mapped = map_decision({"action": "open_app", "app_name": app})
    assert mapped.invalid is False, mapped.reason

    before = env.foreground()
    for action in mapped.actions:
        env.execute(action)
    after = env.foreground()

    assert after, "foreground activity is empty"
    assert after != before or "launcher" not in after.lower(), (
        "open_app %r left the foreground activity at %r" % (app, after))
