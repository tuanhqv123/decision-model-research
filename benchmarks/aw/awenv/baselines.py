"""Control policies that share the real agent path.

``random`` and ``wait`` answer the question "does the task complete by itself?", which is
the question the 2026-11-02 run could not answer: ``wait`` scored 0/6 and ``random`` 0/6,
and without those two the model's 1/6 could have been read as progress.

Each policy implements ``decide(goal, history, pixels, **kw)`` — the same signature as
``ModelClient.decide`` — so ``AgentCore``, the adapter, the recorder and the metrics treat a
baseline exactly as they treat the model.

``m3a`` is deliberately *not* one of these policies: AndroidWorld's M3A is a full agent with
its own loop, so it is built by ``build_m3a`` and runs through its own integration. Fitting
it into ``decide()`` would compare a policy to an agent.
"""

import random
from typing import Any, Dict, List, Optional

from .adapter import CANON_APPS, EMITTABLE

POLICIES = ("wait", "random", "status_now")

#: ``status`` is excluded: a control that declares done cannot be told from one that gave up.
RANDOM_ACTIONS = tuple(a for a in EMITTABLE if a != "status")

#: Screen space the emulator reports; a random point outside it is a guaranteed invalid action.
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 2400


class WaitPolicy:
    """The null agent: no-ops until the step budget runs out."""

    name = "wait"

    def decide(self, goal: str, history: Optional[List[str]] = None, pixels: Any = None, **kw) -> Dict[str, Any]:
        return {"action": "wait", "confidence": 1.0, "latency_ms": {"model": 0.0}}


class RandomPolicy:
    """Uniform over the actions the checkpoint can emit, with valid payloads per action.

    Never emits ``status``: a control that terminates early cannot be told apart from one
    that gave up, and the point of this baseline is to spend the whole budget.
    """

    name = "random"

    def __init__(self, seed: int = 0, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.rng = random.Random(seed)
        self.width = width
        self.height = height

    def decide(self, goal: str, history: Optional[List[str]] = None, pixels: Any = None, **kw) -> Dict[str, Any]:
        action = self.rng.choice(RANDOM_ACTIONS)
        dec: Dict[str, Any] = {
            "action": action,
            "confidence": round(self.rng.random(), 3),
            "latency_ms": {"model": 0.0},
        }
        if action in ("click", "long_press"):
            dec["point"] = [self.rng.randrange(self.width), self.rng.randrange(self.height)]
        elif action == "type_text":
            dec["point"] = [self.rng.randrange(self.width), self.rng.randrange(self.height)]
            dec["text"] = self.rng.choice(["test", "1234", "hello"])
        elif action == "scroll":
            dec["direction"] = self.rng.choice(["up", "down", "left", "right"])
        elif action == "open_app":
            dec["app_name"] = self.rng.choice(CANON_APPS)
        return dec


class StatusNowPolicy:
    """The floor of the official metric (plan Q11): declare `status` at step 0.

    `suite_utils` scores `task_successful if done else 0`, so this policy scores exactly the tasks
    whose initial state already satisfies the checker. Whatever it gets, every arm gets for free.
    """

    name = "status_now"

    def decide(self, goal: str, history: Optional[List[str]] = None, pixels: Any = None, **kw) -> Dict[str, Any]:
        return {"action": "status", "confidence": 1.0, "latency_ms": {"model": 0.0}}


def policy_names():
    return POLICIES


def build_policy(name: str, **kw):
    """Registry for the controls that fit the decide() interface."""
    if name == "wait":
        return WaitPolicy()
    if name == "random":
        return RandomPolicy(seed=kw.get("seed", 0))
    if name == "status_now":
        return StatusNowPolicy()
    raise KeyError("unknown policy %r (m3a is built by build_m3a, not a policy)" % (name,))


def build_m3a(env: Any):
    """AndroidWorld's own minimal agent — the honest in-suite reference.

    Imported lazily: the package exists only on the emulator host, and a fake here would
    make the baseline meaningless.
    """
    from android_world.agents import m3a

    return m3a.M3A(env, name="m3a")
