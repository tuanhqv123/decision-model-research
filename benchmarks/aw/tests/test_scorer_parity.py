"""Scoring parity with AndroidWorld's own checker.

Runs only where AndroidWorld is installed (the emulator host). Locally it is skipped
rather than faked: a fake checker would defeat the purpose of the test.
"""
import os

import pytest

ANDROID_WORLD = os.environ.get("ANDROID_WORLD_PATH") or "/opt/android_world"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ANDROID_WORLD),
    reason="AndroidWorld not installed; run on the emulator host with ANDROID_WORLD_PATH set",
)


def test_official_checker_module_is_importable():
    import android_world  # noqa: F401


def test_success_is_read_from_device_state_not_from_the_agent():
    """The harness must never accept the agent's own claim of success.

    On a no-op episode the official checker must return False even if the agent
    emitted ``status``; that is the regression that produced the 1/6 confusion.
    """
    from android_world.env import env_launcher  # noqa: F401
