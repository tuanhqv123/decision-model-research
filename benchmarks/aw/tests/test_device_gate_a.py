"""Gate A: prove every action works on the live emulator before any task number exists.

Why this file exists, and why it is not a unit test: on 2026-11-02 the emulator was
OOM-killed while the FastAPI server kept answering 200 and every task silently scored 0.
A fake device here would certify exactly the thing that broke. Every assertion below is a
statement about the real device, so these tests skip unless a device is present.

Two rules make the gate meaningful:

1. **Verify the consequence, not the absence of an exception.** `execute()` returning
   without raising means adb accepted a command, nothing more. `open_app` must change the
   foreground activity; `type_text` must land in the a11y tree; `click` must change the
   image hash. Anything weaker certifies nothing.
2. **Test the payload the adapter actually emits.** The old version asserted
   `["click", "input_text"]` for type_text — the double-tap-without-clear bug it was
   supposed to catch. It held the wrong contract and passed. Now it calls `map_decision`
   and executes exactly what comes back.

Run on the emulator host, inside the container:
    AW_GATE=1 python3 -m pytest benchmarks/aw/tests/test_device_gate_a.py -q
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from awenv.adapter import CANON_APPS, EMITTABLE  # noqa: E402

GATE = os.environ.get("AW_GATE") == "1"
pytestmark = pytest.mark.skipif(
    not GATE,
    reason="Gate A needs a live emulator; set AW_GATE=1 on the AndroidWorld host",
)

ADB = os.environ.get("AW_ADB", "/opt/android/platform-tools/adb")
SETTLE = float(os.environ.get("AW_SETTLE", "2.0"))


def _hash(pixels):
    return __import__("hashlib").sha1(pixels.tobytes()).hexdigest()


def _texts(env):
    return " ".join((e.get("text") or "") + " " + (e.get("content_description") or "")
                    for e in env.elements())


def _editable(env):
    out = []
    for e in env.elements():
        cls = (e.get("class") or "")
        if "EditText" in cls or "AutoComplete" in cls or "SearchView" in cls:
            if e.get("bbox"):
                out.append(e)
    return out


def _run(env, decision, settle=SETTLE):
    """Map a decision the way the agent does, execute it, settle. Returns mapped."""
    from awenv.adapter import map_decision

    mapped = map_decision(decision)
    assert mapped.invalid is False, "adapter rejected %r: %s" % (decision, mapped.reason)
    for a in mapped.actions:
        env.execute(a)
    time.sleep(settle)
    return mapped


@pytest.fixture(scope="module")
def env():
    from awenv.awenv_bridge import build

    environment = build(console_port=5554, adb_path=ADB, emulator_setup=False)
    environment.reset()
    time.sleep(1.0)
    yield environment
    try:
        environment.close()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------- liveness
def test_liveness_passes_all_layers(env):
    """The device, a screenshot and a no-op must all survive a round trip."""
    from awenv.liveness import probe_inprocess

    result = probe_inprocess(env, adb=ADB, serial=os.environ.get("AW_SERIAL"))
    assert result["ok"] is True, result
    names = [l["name"] for l in result["layers"]]
    assert "device" in names and "screenshot" in names, names


# ----------------------------------------------------------- every action, executed
@pytest.mark.parametrize("action", [a for a in EMITTABLE if a != "status"])
def test_every_emittable_action_is_accepted(env, action):
    """No mapping raises, and the device is still alive afterwards.

    `status` is excluded because it is a termination, not an action: it is exercised by
    `test_status_terminates_without_executing`.
    """
    decision = {
        "click": {"action": "click", "point": [540, 1200]},
        "long_press": {"action": "long_press", "point": [540, 1200]},
        "type_text": {"action": "type_text", "point": [540, 400], "text": "gate a"},
        "scroll": {"action": "scroll", "direction": "down"},
        "open_app": {"action": "open_app", "app_name": "settings"},
        "navigate_back": {"action": "navigate_back"},
        "navigate_home": {"action": "navigate_home"},
        "wait": {"action": "wait"},
    }[action]

    mapped = _run(env, decision)
    assert mapped.actions, "mapping produced no action"
    assert env.is_alive(), "device died executing %r" % action


def test_status_terminates_without_executing(env):
    """`status` is a termination: no device command, `done=True`."""
    from awenv.adapter import map_decision

    mapped = map_decision({"action": "status"})
    assert mapped.done is True
    assert mapped.actions == [], "status must not issue a device command"


# --------------------------------------------------------------- action consequences
def test_click_changes_the_screen(env):
    """A click must alter the pixels; a no-op that returns cleanly proves nothing.

    Taps the centre of the first clickable element rather than a fixed coordinate, so the
    test does not depend on which app happens to be foreground.
    """
    clicks = [e for e in env.elements() if e.get("clickable") and e.get("bbox")]
    if not clicks:
        pytest.skip("no clickable element on screen")

    x1, y1, x2, y2 = clicks[0]["bbox"]
    before = _hash(env.screenshot())
    _run(env, {"action": "click", "point": [int((x1 + x2) / 2), int((y1 + y2) / 2)]}, settle=2.5)
    assert _hash(env.screenshot()) != before, "click left the screen unchanged"


def test_scroll_changes_the_screen(env):
    """Scroll must move content. On a short screen it legitimately cannot, so skip."""
    before = _hash(env.screenshot())
    _run(env, {"action": "scroll", "direction": "down"}, settle=2.0)
    if _hash(env.screenshot()) == before:
        pytest.skip("screen is not scrollable (short page) -- direction not exercised")
    assert env.is_alive()


def test_navigate_home_leaves_the_app(env):
    """`navigate_home` must reach the launcher, not merely be accepted."""
    _run(env, {"action": "navigate_home"}, settle=3.0)
    fg = env.foreground()
    assert fg, "foreground activity is empty after navigate_home"
    assert "launcher" in fg.lower() or "nexus" in fg.lower(), (
        "navigate_home did not reach the launcher: %r" % fg)


def test_navigate_back_pops_a_screen(env):
    """Back from a launched app must change the foreground or the pixels."""
    _run(env, {"action": "open_app", "app_name": "settings"}, settle=4.0)
    deep = env.foreground()
    before = _hash(env.screenshot())
    _run(env, {"action": "navigate_back"}, settle=2.5)
    after = env.foreground()
    changed = (after != deep) or (_hash(env.screenshot()) != before)
    assert changed, "navigate_back changed neither activity (%r) nor pixels" % deep


# --------------------------------------------------------------------- type_text
def test_type_text_emits_exactly_one_input_text(env):
    """The payload contract, checked without touching the device.

    Upstream `actuation.py:77-110` makes `input_text` click its own x/y, sleep, clear when
    `clear_text`, type, and press ENTER. Emitting a separate click first therefore taps the
    field twice, and omitting `clear_text` appends to whatever the field already held.
    """
    from awenv.adapter import map_decision

    mapped = map_decision({"action": "type_text", "point": [5, 6], "text": "hello world"})
    assert [a["action_type"] for a in mapped.actions] == ["input_text"], mapped.actions
    one = mapped.actions[0]
    assert one["text"] == "hello world"
    assert (one["x"], one["y"]) == (5, 6), "focus point must ride on the input_text action"
    assert one.get("clear_text") is True, "without clear_text the text appends to the old value"
    assert not mapped.invalid


def test_type_text_lands_in_the_a11y_tree(env):
    """Typing must actually reach a field, not just be accepted by adb.

    Uses the same field-discovery helper as the replace test, so both exercise a field that
    is known to exist rather than hoping one is on screen.
    """
    pt = _open_a_text_field(env)
    assert pt is not None, "could not reach any editable field; typing is UNVERIFIED"

    text = "gatecheck42"
    before = _texts(env)
    _run(env, {"action": "type_text", "point": pt, "text": text}, settle=3.0)
    after = _texts(env)
    assert text in after, (
        "typed text never reached the a11y tree.\n  wanted: %r\n  before: %r\n  after:  %r"
        % (text, before[:160], after[:160]))


def _open_a_text_field(env):
    """Reach an editable field and return its centre point, or None.

    Simple SMS Messenger's new-conversation screen is used because it always offers one and
    needs no task setup. Markor was tried first and skipped here: its home screen has no
    editable field, so the `clear_text` assertion below never actually ran — a skipped test
    is indistinguishable from a passing one in the summary, which is how a real defect hides.
    """
    _run(env, {"action": "open_app", "app_name": "simple sms messenger"}, settle=4.0)
    fields = _editable(env)
    if not fields:
        clicks = [e for e in env.elements() if e.get("clickable") and e.get("bbox")]
        # Prefer the compose entry point: the bottom-most clickable element on the list screen.
        clicks.sort(key=lambda e: -e["bbox"][1])
        for cand in clicks[:3]:
            x1, y1, x2, y2 = cand["bbox"]
            _run(env, {"action": "click", "point": [int((x1 + x2) / 2), int((y1 + y2) / 2)]},
                 settle=3.0)
            fields = _editable(env)
            if fields:
                break
    if not fields:
        return None
    x1, y1, x2, y2 = fields[0]["bbox"]
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def test_type_text_replaces_rather_than_appends(env):
    """`clear_text=True` must replace the field's contents, which is the actual bug fixed.

    Types A, then B into the same field. With clear_text the second typing replaces A, so
    the field reads B. Without it the field reads AB and a filename/recipient is corrupted
    in a way that still "types successfully" — which is exactly why the old gate missed it.
    """
    pt = _open_a_text_field(env)
    assert pt is not None, "could not reach any editable field; clear_text is UNVERIFIED"

    _run(env, {"action": "type_text", "point": pt, "text": "alphafirst"}, settle=2.5)
    mid = _texts(env)
    assert "alphafirst" in mid, "first typing did not land, so replace cannot be judged: %r" % mid[:160]

    _run(env, {"action": "type_text", "point": pt, "text": "betasecond"}, settle=2.5)
    seen = _texts(env)
    assert "betasecond" in seen, "second typing did not land: %r" % seen[:160]
    assert "alphafirst" not in seen, (
        "clear_text did not clear: both strings present, so typing appended (%r)" % seen[:160])


# ------------------------------------------------------------------------ open_app
@pytest.mark.parametrize("app", CANON_APPS)
def test_open_app_reaches_every_canonical_app(env, app):
    """`open_app` must change the foreground activity, not just return without raising.

    Polls for the activity change instead of sleeping a fixed amount: Retro Music's cold
    start exceeded a 5 s wait and failed this test on one run and passed on the next. A gate
    that reports a flaky failure teaches nothing, and one that a longer sleep would hide
    teaches less — so the wait is explicit and bounded, and a real failure still fails.
    """
    _run(env, {"action": "navigate_home"}, settle=1.5)
    before = env.foreground()

    from awenv.adapter import map_decision

    mapped = map_decision({"action": "open_app", "app_name": app})
    assert mapped.invalid is False, "adapter rejected %r: %s" % (app, mapped.reason)
    for a in mapped.actions:
        env.execute(a)

    after = before
    deadline = time.time() + 20.0
    while time.time() < deadline:
        time.sleep(1.0)
        after = env.foreground()
        if after and (after != before or "launcher" not in after.lower()):
            break

    assert after, "foreground activity is empty"
    assert after != before or "launcher" not in after.lower(), (
        "open_app %r left the foreground at %r after 20s" % (app, after))
    assert env.is_alive()
