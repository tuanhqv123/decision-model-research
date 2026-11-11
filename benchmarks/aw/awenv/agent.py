"""One decision + one execution per step, recorded so failures stay attributable.

The rule that shapes this module: every ambiguity of the 2026-11-02 smoke run has to be
impossible to reconstruct. Nothing is inferred from a step counter, no action degrades
silently, and the history the model sees is built from *what executed*, not from what the
model asked for.

``AgentCore`` is deliberately free of AndroidWorld imports so it can be tested against
fakes; ``make_android_world_agent`` is the thin integration shim loaded on the host.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

from .adapter import MappedAction, map_decision
from .liveness import image_hash
from .masking import NoEffectMask, action_signature, screen_key

# The history renderer is the MODEL's contract, so it lives in `train/ac_common.py` and is imported
# from there — the repo and the deployed harness both keep `train/` next to `aw/`. A second copy here
# is exactly how train and serve history drifted apart three times on 2026-11-10.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "train")))
from ac_common import render_history_line  # noqa: E402

LOOP_BREAK_ACTION = {"action_type": "navigate_back"}
LOOP_BREAK_TAG = "loop_break"

# ----------------------------------------------------------------------------- Q3 measurement
# PLAN-2026-11-10 Q3: of the no-effect clicks, how many land on NO interactive element (grounding)
# and how many on a present but wrong or unresponsive one (semantics)? These helpers only READ the
# pre-action element list; nothing they compute reaches a decision or an executed action.

# One letter per set flag, always in this order: c(lickable) e(ditable) s(crollable) k(checkable)
# x(checked) l(ong_clickable) f(ocused) n(selected).
FLAG_LETTERS = (
    ("is_clickable", "c"), ("is_editable", "e"), ("is_scrollable", "s"), ("is_checkable", "k"),
    ("is_checked", "x"), ("is_long_clickable", "l"), ("is_focused", "f"), ("is_selected", "n"),
)
# A tap can reach a handler on any of these; a tap on none of them can only land by accident.
INTERACTIVE_FLAGS = ("is_clickable", "is_editable", "is_long_clickable", "is_checkable")
POINT_ACTIONS = ("click", "long_press", "input_text")


def _has(e: Dict[str, Any], key: str) -> bool:
    if key == "is_clickable":  # the bridge sends both spellings; older traces only `clickable`
        return bool(e.get("is_clickable") or e.get("clickable"))
    return bool(e.get(key))


def element_flags(e: Dict[str, Any]) -> str:
    return "".join(letter for key, letter in FLAG_LETTERS if _has(e, key))


def _label(e: Dict[str, Any]) -> str:
    for key in ("text", "content_description", "hint_text"):
        value = " ".join(str(e.get(key) or "").split())
        if value:
            return value[:40]
    return ""


def _class_short(e: Dict[str, Any]) -> str:
    return str(e.get("class") or "").rsplit(".", 1)[-1]


def _bbox(e: Dict[str, Any]) -> Optional[List[int]]:
    flat = e.get("bbox")
    if isinstance(flat, (list, tuple)) and len(flat) == 4:
        return [int(v) for v in flat]
    b = e.get("bbox_pixels")
    if isinstance(b, dict) and all(k in b for k in ("x_min", "y_min", "x_max", "y_max")):
        return [int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])]
    return None


def compact_element(e: Dict[str, Any]) -> List[Any]:
    """`[label, class_short, [x_min, y_min, x_max, y_max], flags]` — one trace entry per element."""
    return [_label(e), _class_short(e), _bbox(e), element_flags(e)]


def _containers(els: List[Dict[str, Any]], x: float, y: float):
    for idx, e in enumerate(els):
        b = _bbox(e)
        if b is not None and b[0] <= x <= b[2] and b[1] <= y <= b[3]:
            yield idx, e, b


def element_under(els: List[Dict[str, Any]], x: float, y: float) -> Optional[Dict[str, Any]]:
    """The SMALLEST element whose bbox contains (x, y), or None; ties go to the earlier element."""
    best = None
    for idx, e, b in _containers(els, x, y):
        area = (b[2] - b[0]) * (b[3] - b[1])
        if best is None or area < best[0]:
            best = (area, idx, e, b)
    if best is None:
        return None
    _, idx, e, b = best
    return {"idx": idx, "label": _label(e), "class_short": _class_short(e),
            "flags": element_flags(e), "bbox": b}


def on_interactive(els: List[Dict[str, Any]], x: float, y: float) -> bool:
    """True iff ANY element containing (x, y) is clickable, editable, long-clickable or checkable.

    Any, not only the smallest: a tap on a static label inside a clickable row reaches the row.
    """
    return any(any(_has(e, k) for k in INTERACTIVE_FLAGS) for _, e, _ in _containers(els, x, y))


def ui_fingerprint(els: List[Dict[str, Any]]) -> str:
    """The UI STRUCTURE of one observation: (class, text, description, bbox, state flags) per element.

    The state flags are part of it because a switch that flips `is_checked`, or a field that takes
    focus, is an effect of the action even though no text or rectangle moved. Without them every
    toggle counted as "the action did nothing" (F2's toggle blindness).
    """
    parts = []
    for e in els:
        b = e.get("bbox") or [0, 0, 0, 0]
        parts.append("%s|%s|%s|%s,%s,%s,%s|%s%s" % (
            e.get("class") or "", (e.get("text") or "")[:24],
            (e.get("content_description") or "")[:24], b[0], b[1], b[2], b[3],
            element_flags(e), "|disabled" if e.get("is_enabled") is False else ""))
    parts.sort()
    return "\n".join(parts)


def _action_point(actions: List[Dict[str, Any]]):
    """The point of the first click / long_press / input_text action, or None."""
    for a in actions or []:
        if a.get("action_type") in POINT_ACTIONS and a.get("x") is not None and a.get("y") is not None:
            return a["x"], a["y"]
    return None


def q3_fields(before_els: Optional[List[Dict[str, Any]]], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """`elements_compact`, `click_target`, `click_on_interactive` for one step.

    `before_els` is the PRE-action list from the same `State` as the screenshot the model saw (the
    bridge caches one observation per step), or None when the environment cannot list elements —
    in which case every field is None ("unknown"), never an empty list or a False.
    """
    compact = None if before_els is None else [compact_element(e) for e in before_els]
    pt = _action_point(actions)
    if pt is None or before_els is None:
        return {"elements_compact": compact, "click_target": None, "click_on_interactive": None}
    return {
        "elements_compact": compact,
        "click_target": element_under(before_els, pt[0], pt[1]),
        "click_on_interactive": on_interactive(before_els, pt[0], pt[1]),
    }


def _signature(mapped: MappedAction) -> str:
    """Identity of an EXECUTED action, for loop detection.

    Every argument must be part of the action, or two genuinely different decisions
    collapse into one signature. `text` and `direction` were missing: two `type_text`
    calls with different strings, and two `scroll`s in opposite directions, both looked
    identical. That grew the list of cases where a real, working action was mistaken for
    the model spinning.
    """
    return "|".join(
        "%s:%s:%s:%s:%s" % (
            a.get("action_type"), a.get("x"), a.get("y"),
            (a.get("text") or "")[:40], a.get("direction") or a.get("app_name") or "",
        )
        for a in mapped.actions
    )


def _raw_of(dec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": dec.get("action"),
        "point": dec.get("point"),
        "text": dec.get("text"),
        "app_name": dec.get("app_name"),
        "direction": dec.get("direction"),
        "confidence": dec.get("confidence"),
        "gate_score": dec.get("gate_score"),
        "is_done": dec.get("is_done"),
        # The VA readout: which numbered element the model chose, and its label. Without this in
        # the trace there is no way to tell "picked SAVE confidently" from "guessed a point near
        # SAVE", and those need opposite fixes. `point` is the chosen element's centre when the
        # server honoured the element, so the two fields must be read together.
        "element": dec.get("element"),
        "box_confidence": dec.get("box_confidence"),
        "track": dec.get("track"),
        # Generative control arm (Q1): the generated text, why it failed to parse, and its length.
        # Absent (None) for the one-pass model, which generates nothing.
        "raw": dec.get("raw"),
        "parse_error": dec.get("parse_error"),
        "tokens_generated": dec.get("tokens_generated"),
    }


def _history_line(dec: Dict[str, Any], mapped: MappedAction, ok: bool) -> str:
    """The history line for one step: what EXECUTED, rendered by the training renderer.

    When the adapter substituted the decision (invalid -> `wait`, refused `status` -> `wait`, loop
    break -> `navigate_back`) the executed action is rendered — the model must never be told an action
    ran that did not. Otherwise the decision is rendered with the arguments the model chose, which is
    what training renders from the gold action (the app name as written, not our canonical alias).
    No serve-only text ("-> ok", "(rejected: ...)", "(failed)", coordinates) ever enters the history:
    training contains none of it (decision K3). `ok`/`reason` stay in the step record.
    """
    action = (dec.get("action") or "").strip().lower()
    substituted = mapped.invalid or (mapped.reason is not None and not mapped.done)
    if substituted and mapped.actions:
        return render_history_line(mapped.actions[0])
    if mapped.done:
        return render_history_line({"action_type": "status"})
    return render_history_line({"action_type": action or "wait", "text": dec.get("text"),
                                "direction": dec.get("direction"), "app_name": dec.get("app_name")})


def history_entry_v3_for_step(dec: Dict[str, Any], mapped: MappedAction, before_els: Optional[List[Dict[str, Any]]],
                              after_els: Optional[List[Dict[str, Any]]]) -> str:
    """The contract-v3 history line for one executed step — the SAME renderer and effect test training uses
    (`ac_common.history_entry_with_label`, `ac_common.effect_changed`), so the line is identical to the one the
    model learned from (`add_history_v3`). What EXECUTED is rendered: a substituted action (invalid -> `wait`)
    names no element. The element is named by the server's `history_label` of the element the model chose."""
    from ac_common import SPATIAL, canonical_element, effect_changed, history_entry_with_label
    substituted = mapped.invalid or (mapped.reason is not None and not mapped.done)
    if substituted and mapped.actions:
        action, label = dict(mapped.actions[0]), None
    else:
        action = {"action_type": (dec.get("action") or "wait").strip().lower(), "text": dec.get("text"),
                  "direction": dec.get("direction"), "app_name": dec.get("app_name")}
        label = (dec.get("element") or {}).get("history_label") if action["action_type"] in SPATIAL else None
    changed = None
    if before_els is not None and after_els is not None:
        changed = effect_changed([canonical_element(e) for e in before_els], [canonical_element(e) for e in after_els])
    return history_entry_with_label(action, label, changed)


class AgentCore:
    """One model, one environment, one step at a time."""

    name = "d1_2b"

    def __init__(
        self,
        model: Any,
        env: Any,
        allow_app_fallback: bool = False,
        stuck_k: int = 3,
        track: str = "V",
        return_readouts: bool = False,
        done_threshold: float = 0.0,
        loop_break: bool = False,
        mask: bool = False,
        history_format: str = "v1",
    ):
        self.model = model
        self.env = env
        self.allow_app_fallback = allow_app_fallback
        self.stuck_k = stuck_k
        self.track = track
        self.return_readouts = return_readouts
        self.done_threshold = done_threshold
        # Detection (`stuck_event`) is always recorded; OVERRIDING the model with a back-press is a
        # harness policy that is off unless a config asks for it (it destroyed a correct screen on
        # MarkorCreateNote, 2026-11-10, and upstream agents have no such rule).
        self.loop_break = loop_break
        # Verified-failure masking (PLAN-2026-11-10 Q2-ii, `awenv/masking.py`): OFF unless a config
        # asks for it. It needs the readouts of the same forward pass, so it requests them; with the
        # flag off the request, the decision and the execution are exactly what they were.
        self.mask = bool(mask)
        self._mask = NoEffectMask() if self.mask else None
        # "v3": history entries say which element each action hit and whether the screen changed
        # (`history_entry_v3_for_step`), for checkpoints of contract v3; "v1": the bare renderer.
        if history_format not in ("v1", "v3"):
            raise ValueError("history_format must be 'v1' or 'v3', not %r" % (history_format,))
        self.history_format = history_format
        self.history: List[str] = []
        self._seen: List[Any] = []
        self._va_empty = 0

    def _elements(self) -> Optional[List[Dict[str, Any]]]:
        """The a11y element list, or None on the V track.

        A VA run that silently sends no elements is indistinguishable from a V run, and that is
        exactly what happened on 2026-11-10: the config said V, the list was never sent, and the
        `target_el` head the checkpoint was trained with was never used. `self._va_empty` records
        the first time VA produced nothing so the report can say so instead of hiding it.
        """
        if self.track != "VA":
            return None
        getter = getattr(self.env, "elements", None)
        if not callable(getter):
            self._va_empty += 1
            return None
        els = list(getter() or [])
        if not els:
            self._va_empty += 1
            return None
        return els

    def _n_elements(self) -> int:
        getter = getattr(self.env, "elements", None)
        if not callable(getter):
            return 0
        try:
            return len(getter() or [])
        except Exception:
            return 0

    def _read_elements(self) -> Optional[List[Dict[str, Any]]]:
        """The current observation's element list, or None when the environment cannot list them.

        On the AndroidWorld bridge this is served from the ONE cached `State` of the current
        observation, so reading it never triggers another `get_state`.
        """
        getter = getattr(self.env, "elements", None)
        if not callable(getter):
            return None
        try:
            return list(getter() or [])
        except Exception:  # noqa: BLE001
            return None

    def _structure(self, els: Optional[List[Dict[str, Any]]] = None) -> str:
        """A cheap fingerprint of the UI STRUCTURE, not of the pixels (see `ui_fingerprint`).

        The set of (class, text, bbox, state flags) is stable while a caret blinks or the IME
        redraws, and changes when the app navigates or a widget toggles. That is what "did my
        action do anything" actually asks. Falls back to an empty string when the environment
        cannot list elements, in which case `ui_changed` carries no information.
        """
        if els is None:
            els = self._read_elements()
        return ui_fingerprint(els or [])

    def step(self, goal: str, i: int) -> Dict[str, Any]:
        """Run one decision and one execution; return the step record."""
        before = self.env.screenshot()
        # Read ONCE, from the observation the screenshot started: the fingerprint and the Q3
        # measurement fields both describe exactly the screen the model saw.
        before_els = self._read_elements()
        self._before_struct = self._structure(before_els)
        before_hash = image_hash(before)

        started = time.time()
        sent = self._elements()
        dec = self.model.decide(
            goal=goal,
            history=list(self.history),
            pixels=before,
            elements=sent,
            track=self.track,
            return_readouts=self.return_readouts or self.mask,
        )
        model_ms = _model_ms(dec, (time.time() - started) * 1000.0)

        map_kw = dict(allow_app_fallback=self.allow_app_fallback, goal=goal,
                      done_threshold=self.done_threshold)
        mapped = map_decision(dec, **map_kw)
        raw = _raw_of(dec)  # always what the MODEL said, even when masking replaces it

        # Masking replaces a decision whose (screen, action) pair was verified to do nothing by the
        # model's own next-best option from this same pass. `executed_dec` is what will run; the
        # history is rendered from it, so the model is never told the masked action ran.
        executed_dec, masked_from, mask_choice = dec, None, None
        key = screen_key(self._before_struct) if self.mask else None
        if self.mask:
            width, height = _screen_size(before, dec)
            executed_dec, mapped, masked_from, mask_choice = self._mask.apply(
                dec, mapped, key, before_els, sent, width, height, **map_kw)

        if mapped.done:
            record = self._record(i, raw, mapped, before_hash, None, None, model_ms, 0.0)
            record.update(q3_fields(before_els, mapped.actions))
            record["masked_from"], record["mask_choice"] = masked_from, mask_choice
            record["screenshot_pixels"] = before  # exactly what the model saw; the runner saves it
            record["action_ok"] = None
            record["done"] = True
            self.history.append(_history_line(executed_dec, mapped, True))
            return record

        # A `status` refused by the is_done gate was never executed, so it says nothing about
        # the screen and must not enter the loop counter. Without this guard, three refusals in
        # a row all map to `wait`, produce the same signature, and trip stuck_k — which then
        # OVERWRITES the model's decision with LOOP_BREAK_ACTION (`navigate_back`).
        # This is exactly what destroyed MarkorCreateNote on 2026-11-10: the note was open and
        # correct on screen, the model said done three times, the gate refused three times, and
        # the third refusal was replaced by a back-press that threw the screen away.
        policy_refusal = mapped.reason == "is_done_below_threshold"
        stuck_event = False if policy_refusal else self._is_loop(_signature(mapped), before_hash)
        if stuck_event and self.loop_break:
            mapped = MappedAction(
                actions=[dict(LOOP_BREAK_ACTION)],
                scaffolding=[LOOP_BREAK_TAG],
                reason="loop_detected",
            )
            self._seen = []

        env_started = time.time()
        ok, error, rejected = True, None, False
        try:
            for action in mapped.actions:
                out = self.env.execute(action) or {}
                if isinstance(out, dict) and out.get("ok") is False:
                    ok, rejected = False, True
                    error = out.get("error") or "executor_rejected"
                    break
        except Exception as exc:  # an execution crash must not end the episode
            ok, error = False, "%s: %s" % (type(exc).__name__, exc)
        env_ms = (time.time() - env_started) * 1000.0

        after_hash = image_hash(self.env.screenshot())
        # `screen_changed` from a pixel hash is nearly always True, so it cannot detect "the
        # action did nothing". On ExpenseAddMultiple (2026-11-10) the model clicked 56 times in
        # the note field's band and every step reported `screen_changed=True` because the text
        # cursor blinks and the IME redraws. Compare the UI STRUCTURE instead: the set of
        # (class, text, bbox) of the a11y elements. A caret blink does not change it; navigating
        # to another screen does.
        before_struct = getattr(self, "_before_struct", None)
        after_els = self._read_elements()
        after_struct = self._structure(after_els)
        struct_changed = (before_struct is None) or (after_struct != before_struct)
        screen_changed = after_hash != before_hash
        ui_changed = struct_changed

        # `action_executed` used to hold only `mapped.actions[0]`, so for a
        # type_text decision (which upstream serves as ONE `input_text`) the trace
        # could not show the text at all, and a two-command mapping looked like a
        # bare click. Record the whole list, and keep `[0]` for readers that expect
        # a single dict.
        record = self._record(
            i, raw, mapped, before_hash, mapped.actions[0] if mapped.actions else None,
            ok, model_ms, env_ms,
        )
        record["actions_executed"] = list(mapped.actions)
        # Q3: the element under the EXECUTED point, on the pre-action list. Measurement only.
        record.update(q3_fields(before_els, mapped.actions))
        record["masked_from"], record["mask_choice"] = masked_from, mask_choice
        if self.mask and ok and not ui_changed:
            # environment-verified: this action, on this screen, changed nothing
            self._mask.add(key, action_signature(mapped, before_els))
        record["typed_landed"] = self._typed_landed(mapped)
        record["screenshot_pixels"] = before
        record.update(
            {
                "screen_changed": screen_changed,
                # structure-level change: the signal that "this action did something" can be
                # judged on. `screen_changed` (pixels) is kept for compatibility but is almost
                # always True and must not be used for loop detection.
                "ui_changed": ui_changed,
                "screen_hash_after": after_hash,
                "action_ok": ok,
                "error": error,
                "env_reject": rejected,
                "stuck_event": stuck_event,
                "done": False,
                "history_line": (history_entry_v3_for_step(executed_dec, mapped, before_els, after_els)
                                 if self.history_format == "v3" else _history_line(executed_dec, mapped, ok)),
            }
        )
        # Exactly ONE history line per executed step, in the format training renders. The
        # 2026-11-10 "Note: the last action did not change the interface." sentence was removed the
        # same day: training never contains it, so it was a serve-only input (decision K3). Whether
        # the action changed the UI is still recorded in the trace as `ui_changed`.
        self.history.append(record["history_line"])
        return record

    def _record(
        self,
        i: int,
        raw: Dict[str, Any],
        mapped: MappedAction,
        before_hash: str,
        executed: Optional[Dict[str, Any]],
        model_ok: Optional[bool],
        model_ms: float,
        env_ms: float,
    ) -> Dict[str, Any]:
        return {
            "i": i,
            "action_raw": raw,
            "action_executed": executed,
            "confidence": raw.get("confidence"),
            "gate_score": raw.get("gate_score"),
            "invalid_action": bool(mapped.invalid),
            "reason": mapped.reason,
            "scaffolding": list(mapped.scaffolding),
            "screen_hash": before_hash,
            "model_ms": model_ms,
            "env_ms": env_ms,
            "a11y_elements_n": self._n_elements(),
            # Proof that the VA path was exercised. Without it a `track: VA` run that silently
            # degraded to V looks identical to a successful one in the trace.
            "track": self.track,
            "elements_sent": self._n_elements() if (self.track == "VA" and not self._va_empty) else 0,
            "emitted": executed is not None,
            "typed_landed": None,
        }

    def _typed_landed(self, mapped: MappedAction) -> Optional[bool]:
        """For an executed `input_text`: is the typed string now in some element's text?

        Measurement only (audit §7, hypothesis "typing failed because of the harness"): the pointer
        read produced the right string 24/24 times on 2026-11-08, but no trace recorded whether it
        reached the field. Read from the post-action observation; None for every other action.
        """
        typed = [a.get("text") for a in mapped.actions if a.get("action_type") == "input_text"]
        if not typed or not typed[-1]:
            return None
        want = str(typed[-1]).strip().lower()
        getter = getattr(self.env, "elements", None)
        if not callable(getter):
            return None
        try:
            els = getter() or []
        except Exception:  # noqa: BLE001
            return None
        return any(want in str(e.get("text") or "").lower() for e in els)

    def _is_loop(self, sig: str, screen_hash: str) -> bool:
        """k identical decisions on an unchanged screen is a loop, not persistence."""
        self._seen.append((sig, screen_hash))
        if len(self._seen) < self.stuck_k:
            return False
        return len(set(self._seen[-self.stuck_k:])) == 1


def _screen_size(pixels: Any, dec: Dict[str, Any]):
    """(width, height) of the screenshot the model saw; the server's `screen` as a fallback."""
    shape = getattr(pixels, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    screen = dec.get("screen")
    if isinstance(screen, (list, tuple)) and len(screen) == 2:
        return int(screen[0]), int(screen[1])
    return 1, 1


def _model_ms(dec: Dict[str, Any], fallback: float) -> float:
    lat = dec.get("latency_ms")
    if isinstance(lat, dict):
        value = lat.get("model")
        if value is not None:
            return float(value)
    if isinstance(lat, (int, float)):
        return float(lat)
    return round(fallback, 1)


def make_android_world_agent(core: AgentCore, env: Any, name: Optional[str] = None):
    """Build the AndroidWorld agent class on the host.

    Kept to the minimum surface: AndroidWorld's ``EnvironmentInteractingAgent`` owns the
    loop and the environment handle; ``core`` owns every decision and every record.
    """
    from android_world.env import interface as aw_interface

    class _D1Agent(aw_interface.EnvironmentInteractingAgent):
        def __init__(self, environment):
            super().__init__(environment)
            self.core = core
            self.name = name or core.name
            self._i = 0

        def step(self, goal):
            record = self.core.step(goal, self._i)
            self._i += 1
            done = bool(record.get("done"))
            return aw_interface.AgentInteractionResult(done=done, data=record)

    return _D1Agent(env)
