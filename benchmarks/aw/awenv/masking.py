"""Verified-failure masking (PLAN-2026-11-10 Q2-ii). Off unless `AgentCore(mask=True)`.

Within ONE episode, keep the set of (screen, action) pairs the environment verified to change
nothing: the action executed and the UI fingerprint (`agent.ui_fingerprint`, state flags included)
was identical afterwards. Before a decision executes, if its (screen, action) pair is in the set,
it is replaced by the model's own next-best alternative FROM THE SAME FORWARD PASS — the `readouts`
the server returns with `return_readouts` — never by a new model call and never by a rule.

Definitions (exactly these):

* ``screen_key`` = hash of the pre-action UI fingerprint. A step whose fingerprint is empty (the
  environment listed no elements) has no key: nothing can be verified there, so nothing is masked
  or added.
* ``action signature`` = action type + its argument. Spatial actions (click, long_press,
  type_text): the element under the point (the smallest pre-action element containing it, named by
  class, label and bbox), or the point quantised to a 40-px grid when no element contains it;
  type_text additionally carries its text. scroll: the direction. open_app: the app name.
  navigate_back / navigate_home: the type alone. ``wait`` and a termination have no signature:
  they are never masked and never recorded.
* A pair is ADDED after an action executed without error and left ``ui_changed=False``.
* Alternatives, in order:
  (a) a VA spatial decision: the next most probable ``target_el`` element, same action type (and
      same text for type_text), whose signature is not masked;
  (b) otherwise the next most probable action TYPE whose argument is available from the readouts —
      scroll -> the ``scroll_dir`` choice; type_text -> the ``type_text`` text; open_app -> the
      ``app_name`` text; navigate_back / navigate_home / status -> no argument — and whose
      signature is not masked. Spatial types need a new point: on VA it comes from ``target_el``
      as in (a); on V there is none, so spatial alternatives are skipped. ``wait`` is not an
      alternative (it is not in the plan's list, and it is itself a no-op).
* Nothing left: the original decision executes, and the record says so (``none_left``).

Scope, stated honestly: this removes period-1 repeats (the same action on an unchanged screen) by
construction. It does not address longer cycles (A -> B -> A -> B), an action that changes the
screen without progress, or the choice of a semantically wrong but effective action.
"""

import hashlib
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .adapter import SCROLL_DIRECTIONS, SPATIAL, MappedAction, map_decision

GRID_PX = 40


def screen_key(fingerprint: str) -> Optional[str]:
    if not fingerprint:
        return None
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]


def action_signature(mapped: MappedAction, els: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Signature of what `mapped` would execute, on the screen described by `els`.

    None for anything that is never masked: a termination, an empty mapping, or a `wait`
    (including an invalid decision the adapter turned into one).
    """
    from .agent import element_under  # agent imports this module; resolve at call time

    if mapped.done or not mapped.actions:
        return None
    a = mapped.actions[0]
    at = {"input_text": "type_text"}.get(a.get("action_type"), a.get("action_type"))
    if at == "wait":
        return None
    if at in SPATIAL:
        x, y = a.get("x"), a.get("y")
        if x is None or y is None:
            return None
        target = element_under(els or [], x, y)
        if target is not None:
            where = "el:%s|%s|%s" % (target["class_short"], target["label"], target["bbox"])
        else:
            where = "pt:%d,%d" % (int(x) // GRID_PX, int(y) // GRID_PX)
        if at == "type_text":
            return "type_text|%s|%s" % (where, a.get("text") or "")
        return "%s|%s" % (at, where)
    if at == "scroll":
        return "scroll|%s" % (a.get("direction") or "")
    if at == "open_app":
        return "open_app|%s" % (a.get("app_name") or "")
    return str(at)


# ----------------------------------------------------------------------------- readouts
def _ranked(probs: Dict[str, Any]) -> List[str]:
    """Keys by descending probability; ties keep the server's order."""
    items = [(k, float(v)) for k, v in (probs or {}).items() if v is not None]
    order = {k: i for i, (k, _) in enumerate(items)}
    return [k for k, _ in sorted(items, key=lambda kv: (-kv[1], order[kv[0]]))]


def action_ranking(dec: Dict[str, Any]) -> List[str]:
    """Action types (harness names) by the model's own probability, from the same forward pass."""
    from ac_common import FROM_OPTION

    ro = ((dec.get("readouts") or {}).get("action") or {}).get("probabilities")
    if ro:
        return [FROM_OPTION.get(k, k) for k in _ranked(ro)]
    return _ranked(dec.get("probabilities") or {})


def server_candidates(elements: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    """The numbered list the server built for VA: `serving/d1_api._normalise` + `pick_candidates`.

    The server cannot be imported here (it loads torch and the model), so its normalisation is
    mirrored; `element_points` refuses to use the result unless the model's own chosen element
    matches it, so a drift between the two shows up as `candidates_mismatch`, never as a wrong tap.
    """
    from ac_common import pick_candidates

    nodes = []
    for n in elements or []:
        b = dict(n.get("bbox_pixels") or {})
        if not b:
            flat = n.get("bbox")
            if isinstance(flat, (list, tuple)) and len(flat) == 4:
                b = {"x_min": flat[0], "y_min": flat[1], "x_max": flat[2], "y_max": flat[3]}
        if not b or not all(k in b for k in ("x_min", "x_max", "y_min", "y_max")):
            continue
        b.setdefault("width", b["x_max"] - b["x_min"])
        b.setdefault("height", b["y_max"] - b["y_min"])
        b.setdefault("center", [(b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2])
        merged = {**n, "bbox_pixels": b}
        merged.setdefault("is_clickable", bool(n.get("clickable")))
        merged.setdefault("is_visible", True)
        nodes.append(merged)
    return pick_candidates(nodes, max(int(width), 1), max(int(height), 1))


def element_points(dec: Dict[str, Any], sent: Optional[List[Dict[str, Any]]],
                   width: int, height: int) -> Tuple[List[Tuple[int, List[float]]], Optional[str]]:
    """[(rank, point)] of the `target_el` candidates by descending probability, and a note.

    Empty unless the decision came from the VA track with a `target_el` distribution AND the
    rebuilt candidate list agrees with the model's own pick (same idx -> same label and centre).
    """
    if (dec.get("track") or "") != "VA" or not sent:
        return [], None
    probs = (((dec.get("readouts") or {}).get("target_el")) or {}).get("probabilities")
    if not probs:
        return [], None
    cands = {str(c["idx"]): c for c in server_candidates(sent, width, height)}
    chosen = dec.get("element") or {}
    if chosen.get("idx") is not None:
        c = cands.get(str(chosen["idx"]))
        if c is None or (chosen.get("label") is not None and chosen["label"] != c["label"]):
            return [], "candidates_mismatch"
        ep = dec.get("element_point")
        if ep is not None and (abs(float(ep[0]) - c["bbox"]["center"][0]) > 0.5
                               or abs(float(ep[1]) - c["bbox"]["center"][1]) > 0.5):
            return [], "candidates_mismatch"
    out = []
    for rank, key in enumerate(_ranked(probs)):
        c = cands.get(str(key))
        if c is not None:
            out.append((rank, list(c["bbox"]["center"])))
    return out, None


def _argument_decision(at: str, dec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A decision of type `at` whose argument comes from the readouts, or None if unavailable."""
    ro = dec.get("readouts") or {}
    if at in ("navigate_back", "navigate_home"):
        return {"action": at}
    if at == "status":
        # carry the is_done readout so a configured done gate applies to the alternative too
        return {"action": "status", "is_done": dec.get("is_done")}
    if at == "scroll":
        d = str(((ro.get("scroll_dir") or {}).get("choice")) or "").strip().lower()
        return {"action": "scroll", "direction": d} if d in SCROLL_DIRECTIONS else None
    if at == "open_app":
        name = str(((ro.get("app_name") or {}).get("text")) or "").strip()
        return {"action": "open_app", "app_name": name} if name else None
    return None


def alternatives(dec: Dict[str, Any], sent: Optional[List[Dict[str, Any]]],
                 width: int, height: int) -> Tuple[Iterator[Tuple[Dict[str, Any], Dict[str, Any]]], Optional[str]]:
    """Candidate replacement decisions in the plan's order, each with its `mask_choice` stub.

    Lazy: the caller stops at the first one whose signature is not masked.
    """
    action = (dec.get("action") or "").strip().lower()
    points, note = element_points(dec, sent, width, height)
    text = str(((((dec.get("readouts") or {}).get("type_text")) or {}).get("text")) or "")

    def spatial(at: str, base_text: str):
        for el_rank, pt in points:
            alt = {"action": at, "point": pt}
            if at == "type_text":
                if not base_text:
                    return
                alt["text"] = base_text
            yield el_rank, alt

    def gen():
        # (a) same action type, next element (VA only)
        if action in SPATIAL:
            for el_rank, alt in spatial(action, dec.get("text") or text):
                yield alt, {"via": "target_el", "rank": el_rank}
        # (b) next action type with an available argument
        for rank, at in enumerate(action_ranking(dec)):
            if at in SPATIAL:
                for el_rank, alt in spatial(at, text if at != action else (dec.get("text") or text)):
                    yield alt, {"via": "action", "rank": rank, "element_rank": el_rank}
                continue
            alt = _argument_decision(at, dec)
            if alt is not None:
                yield alt, {"via": "action", "rank": rank}

    return gen(), note


class NoEffectMask:
    """The per-episode set of verified no-effect (screen_key, signature) pairs."""

    def __init__(self):
        self.pairs = set()

    def __contains__(self, pair) -> bool:
        return pair in self.pairs

    def add(self, key: Optional[str], sig: Optional[str]) -> None:
        if key is not None and sig is not None:
            self.pairs.add((key, sig))

    def apply(self, dec: Dict[str, Any], mapped: MappedAction, key: Optional[str],
              els: Optional[List[Dict[str, Any]]], sent: Optional[List[Dict[str, Any]]],
              width: int, height: int, **map_kw):
        """-> (decision to execute, its mapping, masked_from, mask_choice)."""
        sig = action_signature(mapped, els)
        if key is None or sig is None or (key, sig) not in self.pairs:
            return dec, mapped, None, None
        gen, note = alternatives(dec, sent, width, height)
        for alt, choice in gen:
            alt_mapped = map_decision(alt, **map_kw)
            if alt_mapped.invalid:
                continue
            alt_sig = action_signature(alt_mapped, els)
            if alt_sig is not None and (key, alt_sig) in self.pairs:
                continue
            choice = dict(choice, signature=alt_sig, action=alt.get("action"))
            if note:
                choice["note"] = note
            return alt, alt_mapped, sig, choice
        choice = {"via": "none_left", "rank": None}
        if note:
            choice["note"] = note
        return dec, mapped, sig, choice
