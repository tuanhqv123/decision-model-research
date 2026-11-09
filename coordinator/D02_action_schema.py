"""D02 — the frozen action schema (Step 3 of the guide), with validation and tests.

One normalized representation; benchmark-specific translation happens in adapters. Coordinates are normalized
[0,1] so a resize never silently moves a click.

Run: python3 coordinator/D02_action_schema.py   (runs the unit tests and prints PASS/FAIL)
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

# The action types AndroidWorld accepts. GUI-only adapter: no MCP_CALL / ASK_USER unless a track enables them.
AW_ACTIONS = {
    "TAP":        {"needs_point": True,  "needs_args": ()},
    "LONG_PRESS": {"needs_point": True,  "needs_args": ()},
    "SWIPE":      {"needs_point": True,  "needs_args": ("direction",)},
    "BACK":       {"needs_point": False, "needs_args": ()},
    "HOME":       {"needs_point": False, "needs_args": ()},
    "TYPE_TEXT":  {"needs_point": False, "needs_args": ("text",)},
    "WAIT":       {"needs_point": False, "needs_args": ()},
    "FINISH":     {"needs_point": False, "needs_args": ()},
}
OPTIONAL_TRACKS = {"ASK_USER": ("question",), "MCP_CALL": ("tool", "arguments")}


@dataclass
class Action:
    action_id: str
    type: str
    target: Optional[Dict[str, float]] = None      # {"x":..,"y":..,"coordinate_system":"normalized"}
    arguments: Dict[str, str] = field(default_factory=dict)
    source_observation_id: str = ""
    candidate_id: Optional[str] = None

    def validate(self, allow_tracks=()) -> None:
        legal = set(AW_ACTIONS) | (set(OPTIONAL_TRACKS) if allow_tracks else set())
        if self.type not in legal:
            raise ValueError(f"unsupported action type {self.type!r}; legal: {sorted(legal)}")
        spec = AW_ACTIONS.get(self.type)
        if spec is None:                               # optional track
            for k in OPTIONAL_TRACKS[self.type]:
                if k not in self.arguments:
                    raise ValueError(f"{self.type} requires argument {k!r}")
            return
        if spec["needs_point"]:
            if not self.target:
                raise ValueError(f"{self.type} requires a target point")
            x, y = self.target.get("x"), self.target.get("y")
            if x is None or y is None:
                raise ValueError("target needs x and y")
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"target ({x},{y}) outside normalized [0,1]")
            if self.target.get("coordinate_system") != "normalized":
                raise ValueError("coordinate_system must be 'normalized'")
        for k in spec["needs_args"]:
            if not (self.arguments.get(k) or "").strip():
                raise ValueError(f"{self.type} requires a non-empty {k!r}")
        if not self.source_observation_id:
            raise ValueError("source_observation_id is required (stale-observation guard)")

    def to_androidworld(self, W: int, H: int) -> Dict:
        """Translate to the environment command, in PIXELS of the observation it came from."""
        self.validate()
        if self.type in ("TAP", "LONG_PRESS", "SWIPE"):
            px = round(self.target["x"] * W), round(self.target["y"] * H)
            out = {"action_type": self.type.lower(), "x": px[0], "y": px[1]}
            if self.type == "SWIPE":
                out["direction"] = self.arguments["direction"]
            return out
        if self.type == "TYPE_TEXT":
            return {"action_type": "input_text", "text": self.arguments["text"]}
        if self.type == "BACK":
            return {"action_type": "navigate_back"}
        if self.type == "HOME":
            return {"action_type": "navigate_home"}
        if self.type == "WAIT":
            return {"action_type": "wait"}
        if self.type == "FINISH":
            return {"action_type": "status", "goal_status": "successful"}
        return {"action_type": self.type.lower(), **self.arguments}


def _t():
    ok = 0
    def raises(fn, why):
        nonlocal ok
        try:
            fn(); raise AssertionError(f"expected failure: {why}")
        except ValueError:
            ok += 1
    A = Action
    # happy path
    a = A("a1", "TAP", {"x": .5, "y": .5, "coordinate_system": "normalized"}, source_observation_id="obs1")
    a.validate(); assert a.to_androidworld(1080, 2400) == {"action_type": "tap", "x": 540, "y": 1200}; ok += 1
    # failures the guide names explicitly
    raises(lambda: A("a", "MCP_CALL", source_observation_id="o").validate(), "unsupported type")
    raises(lambda: A("a", "TYPE_TEXT", source_observation_id="o").validate(), "missing text")
    raises(lambda: A("a", "TAP", {"x": 1.5, "y": .5, "coordinate_system": "normalized"},
                     source_observation_id="o").validate(), "out of bounds")
    raises(lambda: A("a", "TAP", {"x": .5, "y": .5, "coordinate_system": "normalized"}).validate(),
           "stale observation (empty id)")
    raises(lambda: A("a", "TAP", {"x": .5, "y": .5, "coordinate_system": "pixels"},
                     source_observation_id="o").validate(), "wrong coordinate system")
    raises(lambda: A("a", "SWIPE", {"x": .5, "y": .5, "coordinate_system": "normalized"},
                     source_observation_id="o").validate(), "swipe without direction")
    # MCP allowed only when a track enables it
    m = A("a", "MCP_CALL", arguments={"tool": "x", "arguments": "{}"}, source_observation_id="o")
    raises(lambda: m.validate(), "mcp without track"); m.validate(allow_tracks=("MCP_CALL",)); ok += 1
    print(f"PASS — {ok} checks (including every failure mode the guide lists)")


if __name__ == "__main__":
    _t()
