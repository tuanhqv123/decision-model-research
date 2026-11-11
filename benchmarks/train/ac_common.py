"""Shared by train_d1.py and eval_d1.py: ONE definition of the decision model's state, questions and gold
answers, so the trained function is exactly the served one.

Tracks (one model can serve both; a row's track is chosen per sample in training):
  V  : screenshot + goal + previous step instructions (the AC-v2 leaderboard protocol: GUI-Libra's harness
       puts row['previous_actions'] in the prompt as 'Interaction History'; no a11y)
  VA : screenshot + goal + a11y candidate list (choice over K elements; reported as a separate,
       a11y-assisted track — never mixed with V numbers)

Both tracks may carry an `apps` memory hint: app names retrieved from TRAINING-split goals similar to this
goal (leave-own-episode-out in training). This is a memory over training data (cf. Mem-W's memory bank),
disclosed as such, and ablatable with --no-memory.

Questions (answers read in one forward pass, zero generated tokens):
  action      choice(8)   always supervised
  target_box  box         supervised on click/long_press/type_text (edges = smallest gold element)
  target_el   choice(K)   VA only, supervised when the gold element is a candidate
  scroll_dir  choice(4)   supervised on scroll
  app_name    text read   start/end over the state; 'nothing' unless open_app
  type_text   text read   start/end over the state; 'nothing' unless type_text
"""
import math
import os
import re

ACTIONS = {
    "click": "Tap or click a specific UI element",
    "write": "Enter text into an input field",
    "long_press": "Press and hold a UI element",
    "scroll": "Scroll a view in a direction",
    "wait": "Pause to let the UI load or update",
    "navigate_back": "Press the system Back button",
    "navigate_home": "Go to the device home screen",
    "open_app": "Launch an app by its name",
    "terminate": "End the task: the goal is already achieved on this screen",
}
# THE single source of the action vocabulary — the 9 names as they appear in the corpus.
# `navigate_home` is present in the raw corpus (29 rows) and was absent from every compact_*
# derivative — do not drop it again (DATA-SPEC §5).
ACTION_VOCAB = ("click", "type_text", "long_press", "scroll", "wait", "navigate_back",
                "open_app", "navigate_home", "status")
# eval naming <-> option name
TO_OPTION = {"click": "click", "type_text": "write", "long_press": "long_press", "scroll": "scroll",
             "wait": "wait", "navigate_back": "navigate_back", "navigate_home": "navigate_home",
             "open_app": "open_app", "status": "terminate"}
FROM_OPTION = {v: k for k, v in TO_OPTION.items()}
# KNOWN INCONSISTENCY, NOT FIXED YET — read before changing these strings.
#
# `direction` is android_world's own vocabulary (`json_action.JSONAction.direction`) and names
# where the VIEWPORT moves, not which way the finger swipes. Upstream `actuation.py:151-160`
# swipes the finger from the screen centre to the top edge when `direction == 'down'`, so the
# content is dragged up and LATTER list items come into view. Measured on the live device
# 2026-11-10: from Settings, `down` replaced `Notifications / Battery / Storage` with
# `Location / Safety & emergency / Passwords` — it advanced down the page.
#
# The descriptions below say "content moves down (swipe down)", which reads correctly as "the
# viewport moves down" but invites the wrong mental model, and they are shipped as both the
# training criteria and the serving prompt (`build_questions` is used by train_d1 AND d1_api).
#
# They are deliberately NOT rewritten here. The `episodic-s0` checkpoint was trained against
# exactly these strings; editing them now makes the serving prompt differ from the training
# prompt for a fixed set of weights, and the effect of that mismatch is unmeasured. Direction
# accuracy on AC-v2 is already 16/19 (84%), so the current wording is evidently learnable —
# it is a clarity defect, not a demonstrated cause of errors.
# Fix it together with the next retrain, and re-measure `scroll_dir` accuracy on val before
# claiming any gain.
SCROLL = {"up": "content moves up (swipe up)", "down": "content moves down (swipe down)",
          "left": "content moves left", "right": "content moves right"}
SPATIAL = ("click", "long_press", "type_text")


def is_done_gold(is_terminal: bool) -> str:
    """The `noul` gold for 'has the task already been completed on this screen?'.

    Free label: every episode ends with exactly one terminal `status` row and no `status` appears
    mid-trajectory (measured 13,740/13,740 and 0). So `step == n_steps` IS the label.
    """
    return "yes" if is_terminal else "no"



# ----------------------------------------------------------------------------- a11y candidates
def iter_nodes(tree):
    if isinstance(tree, dict):
        yield tree
        for v in tree.values():
            if isinstance(v, (dict, list)):
                yield from iter_nodes(v)
    elif isinstance(tree, list):
        for c in tree:
            yield from iter_nodes(c)


def pick_candidates(tree, W, H):
    """Visible, labeled OR actionable, non-system-UI elements, bbox-deduped, smallest first."""
    seen, cands = set(), []
    for n in iter_nodes(tree):
        if not isinstance(n, dict) or n.get("is_visible") not in (True, "True"):
            continue
        if n.get("package_name") == "com.android.systemui":
            continue
        b = n.get("bbox_pixels")
        if not b or b.get("width", 0) <= 0 or b.get("height", 0) <= 0:
            continue
        key = (b["x_min"], b["y_min"], b["x_max"], b["y_max"])
        if key in seen:
            continue
        label = (n.get("text") or n.get("content_description") or n.get("hint_text") or "")
        actionable = any(n.get(k) in (True, "True") for k in ("is_clickable", "is_editable", "is_scrollable"))
        if not (label or actionable):
            continue
        seen.add(key)
        cands.append({"label": " ".join(str(label).split())[:60], "bbox": b})
    cands.sort(key=lambda c: c["bbox"]["width"] * c["bbox"]["height"])
    for i, c in enumerate(cands, 1):
        b = c["bbox"]
        c["idx"] = i
        c["pos"] = (f"x:{round(100*b['x_min']/W)}-{round(100*b['x_max']/W)}% "
                    f"y:{round(100*b['y_min']/H)}-{round(100*b['y_max']/H)}%")
    return cands


def gold_candidate(cands, target, gx, gy):
    """The candidate whose CENTRE the scorer would accept as the gold element.

    The scorer (run_acv2.score_row) awards grounding only if the predicted point lies inside the SMALLEST a11y
    node containing the gold point. So a label is only usable if the candidate's own centre falls inside that
    node; anything else supervises a point the evaluation can never credit.

    2026-11-07 fix (R4 defect 4). The old version returned "the smallest candidate CONTAINING the gold point"
    when no candidate bbox matched the gold bbox exactly. On the 398 eval rows that fallback fired on 44/212
    spatial rows, and on **12 of them the chosen container's centre is not inside the gold node at all** - the
    supervision was unscoreable by construction, which silently caps the supervised ceiling at 91.5% instead of
    the 97.2% quoted from "point inside some candidate". Order of preference:
      1. exact bbox match with the target (the usual 76.4% case);
      2. else the smallest candidate whose CENTRE is inside the target bbox (scoreable by construction);
      3. else None - no element supervision for this row; the box head carries it.
    """
    if target is None or gx is None:
        return None
    for c in cands:
        b = c["bbox"]
        if (b["x_min"], b["y_min"], b["x_max"], b["y_max"]) == \
                (target["x_min"], target["y_min"], target["x_max"], target["y_max"]):
            return c

    def centre_in(b):
        cx, cy = (b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2
        return (target["x_min"] <= cx <= target["x_max"]) and (target["y_min"] <= cy <= target["y_max"])

    usable = [c for c in cands if centre_in(c["bbox"])]
    if usable:
        return min(usable, key=lambda c: (c["bbox"]["x_max"] - c["bbox"]["x_min"]) *
                                         (c["bbox"]["y_max"] - c["bbox"]["y_min"]))
    return None


# ----------------------------------------------------------------------------- state + questions
GOLD_KEEP = {"idx": None}  # set by the trainer so a candidate cap never drops the gold element


def _goal_history_lines(goal, history):
    """`Goal:` + `Previous actions:` lines — the head of every state, v1 and v2 alike."""
    lines = [f"Goal: {goal}"]
    if history:
        lines.append("Previous actions:")
        for i, h in enumerate(history, 1):
            text = str(h)
            # Tolerate data that already carries the prefix, so a re-mix of corpora cannot
            # double it again. This is a repair for existing files, not a licence to add it.
            if text.lower().lstrip().startswith("step "):
                lines.append(text)
            else:
                lines.append(f"Step {i}: {text}")
    else:
        lines.append("Previous actions: none")
    return lines


def build_state(goal, track, cands=None, apps=None, history=None, trajs=None, app_intent=None):
    """history = the earlier steps' instructions, as the AC-v2 harness gives them (row['previous_actions'];
    GUI-Libra guilibra_plan.py 'Interaction History'). Part of the leaderboard protocol.

    History entries are BARE ("click", "type_text Floor lamp"); the `Step N: ` prefix is added
    HERE and nowhere else. The episodic rebuild used to write the prefix into the data as well,
    so training rows rendered as `Step 1: Step 1: open_app IKEA` — 159,954 of 159,954 entries
    (measured 2026-11-10). The `cur15k` corpus, which E2 was trained on, carries no prefix
    (0 of 57,096), which is why the double prefix went unnoticed until `episodic-s0` scored
    55.78%: that arm was trained on a different string than every eval and every serve.
    """
    lines = _goal_history_lines(goal, history)
    if apps:
        lines.append("Apps used for similar tasks: " + "; ".join(apps))
    if app_intent:
        # One line, identical at train and serve time, that gives the pointer read something to copy when
        # the goal implies an app without naming it ("Turn wifi on" -> settings).
        lines.append("Application for this task: " + app_intent)
    for j, (g, steps) in enumerate(trajs or [], 1):
        lines.append(f"Similar solved task {j}: {g}")
        lines.append("  its steps: " + " | ".join(steps) + " | done")
    if track == "VA":
        lines.append("Elements on screen (id: label, position):")
        shown = cands
        cap = int(os.environ.get("D1_MAX_CANDS", "0") or 0)
        if cap and len(cands) > cap:
            # Keep the list short enough for the state budget without ever dropping the gold candidate.
            # `keep` is set by make_sample/gold_answers for the row being trained; at eval it is absent and the
            # plain smallest-first prefix is used.
            keep = GOLD_KEEP.get("idx")
            head = cands[:cap]
            if keep is not None and keep not in {c["idx"] for c in head}:
                tail = [c for c in cands if c["idx"] == keep]
                shown = head[: max(cap - len(tail), 0)] + tail
            else:
                shown = head
        for c in shown:
            lines.append(f"[{c['idx']}] {c['label'] or '(no label)'}  ({c['pos']})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- contract v2: AW elements
# R2 (PLAN-2026-11-10 §5): the model chooses among the elements AndroidWorld itself serves
# (`forest_to_ui_elements(..., exclude_invisible_elements=True)`; corpus field `elements`, harness
# `awenv_bridge._element_dicts`). Three things must be identical at train, eval and serve, so each exists once:
#   canonical_element  — one dict per element, whichever shape it arrived in;
#   ELEMENT_RULES      — which elements are shown, a pure function of the list and the screen (never the gold);
#   render_elements    — the lines the model reads.
# Measured 2026-11-07: lists above ~40 items destroy grounding (−17 val / −24 test); a short-list BENEFIT is
# not established. The rule is chosen by a rule fixed before measuring (README of
# benchmarks/results/2026-11-10-contract-v2/).
SYSTEM_UI = "com.android.systemui"
ELEMENT_FLAGS = ("clickable", "editable", "scrollable", "checkable", "checked", "selected", "long_clickable",
                 "focused")
INTERACTIVE_FLAGS = ("clickable", "editable", "scrollable", "checkable", "long_clickable")
FLAG_WORDS = {"clickable": "click", "editable": "edit", "scrollable": "scroll", "checkable": "checkable",
              "checked": "checked", "selected": "selected", "long_clickable": "long_click", "focused": "focused"}
LABEL_CHARS = 40
ELEMENT_CAP = 40
NO_LABEL = "(no label)"
V2_ELEMENTS_HEADER = "Elements on screen (id: label | type | state | position):"


def _true(v):
    return v is True or v == 1 or (isinstance(v, str) and v.strip().lower() == "true")


def canonical_element(d):
    """One element in the ONE form contract v2 reads, from any of the shapes that reach it:

      * the R2 corpus (`aw_extract.element_record`): `class_name`, `bbox_pixels` {x_min..}, `is_*`, None for
        absent strings;
      * the harness (`awenv_bridge._element_dicts`): `class`, `bbox_pixels` (+ width/height/center) AND a flat
        `bbox` list, `clickable` AND `is_clickable`, "" for absent strings;
      * an older API request: only `bbox` [x_min, y_min, x_max, y_max] and `clickable`.
    """
    label = ""
    for key in ("text", "content_description", "hint_text"):
        value = " ".join(str(d.get(key) or "").split())
        if value:
            label = value[:LABEL_CHARS]
            break
    cls = str(d.get("class_name") or d.get("class") or "")
    b = d.get("bbox_pixels")
    if isinstance(b, dict) and all(k in b for k in ("x_min", "y_min", "x_max", "y_max")):
        box = {k: int(b[k]) for k in ("x_min", "y_min", "x_max", "y_max")}
    else:
        flat = d.get("bbox")
        box = ({k: int(v) for k, v in zip(("x_min", "y_min", "x_max", "y_max"), flat)}
               if isinstance(flat, (list, tuple)) and len(flat) == 4 else None)
    flags = {f: _true(d.get(f"is_{f}")) for f in ELEMENT_FLAGS}
    flags["clickable"] = flags["clickable"] or _true(d.get("clickable"))
    return {"label": label, "class_short": cls.rsplit(".", 1)[-1], "bbox": box, "flags": flags,
            "package_name": str(d.get("package_name") or "")}


def _app_on_screen(e, W, H):
    b = e["bbox"]
    if b is None or e["package_name"] == SYSTEM_UI:
        return False
    if b["x_max"] <= b["x_min"] or b["y_max"] <= b["y_min"]:      # zero area: nothing to tap
        return False
    return not (b["x_max"] <= 0 or b["y_max"] <= 0 or b["x_min"] >= W or b["y_min"] >= H)


def _interactive(e):
    return any(e["flags"][f] for f in INTERACTIVE_FLAGS)


def _reading(elements, idx):
    """Indices in reading order: top to bottom (y_min), then left to right (x_min), then AndroidWorld order."""
    return sorted(idx, key=lambda i: (elements[i]["bbox"]["y_min"], elements[i]["bbox"]["x_min"], i))


def rule_all_app(elements, W, H):
    """(a) every app element on screen: no system UI, no zero-area box, not fully off-screen. AW order."""
    return [i for i, e in enumerate(elements) if _app_on_screen(e, W, H)]


def rule_interactive_or_labelled(elements, W, H):
    """(b) (a) ∩ (an interactive flag or a non-empty label). AW order."""
    return [i for i in rule_all_app(elements, W, H) if _interactive(elements[i]) or elements[i]["label"]]


def rule_cap40_reading(elements, W, H):
    """(c) (b), the first ELEMENT_CAP in reading order."""
    return _reading(elements, rule_interactive_or_labelled(elements, W, H))[:ELEMENT_CAP]


def rule_cap40_interactive_first(elements, W, H):
    """(d) (b): interactive elements first (reading order), then labelled-only ones (reading order); cap."""
    keep = rule_interactive_or_labelled(elements, W, H)
    inter = [i for i in keep if _interactive(elements[i])]
    labelled = [i for i in keep if not _interactive(elements[i])]
    return (_reading(elements, inter) + _reading(elements, labelled))[:ELEMENT_CAP]


ELEMENT_RULES = {
    "all_app": rule_all_app,
    "interactive_or_labelled": rule_interactive_or_labelled,
    "cap40_reading": rule_cap40_reading,
    "cap40_interactive_first": rule_cap40_interactive_first,
}


def select_elements(elements, W, H, rule):
    """Raw element dicts (any shape) -> (kept canonical elements in kept order, their indices in `elements`)."""
    canon = [canonical_element(d) for d in elements or []]
    idx = ELEMENT_RULES[rule](canon, W, H)
    return [canon[i] for i in idx], idx


def element_label(e):
    return e["label"] or NO_LABEL


def option_texts(kept):
    """`target_el` option texts, DISTINCT: a unique non-empty label stays as it is; an empty or repeated label gets
    the element's centre in pixels ("OK at (150,450)"). Measured before this rule: on 17.2% of spatial train rows
    the gold option's text equalled another option's (mostly "(no label)"), so only list order told them apart."""
    from collections import Counter
    counts = Counter(e["label"] for e in kept)
    out, seen = {}, Counter()
    for i, e in enumerate(kept, 1):
        b = e["bbox"]
        text = e["label"] if (e["label"] and counts[e["label"]] == 1) else (
            "%s at (%d,%d)" % (element_label(e), (b["x_min"] + b["x_max"]) // 2, (b["y_min"] + b["y_max"]) // 2))
        seen[text] += 1
        out[str(i)] = text if seen[text] == 1 else "%s #%d" % (text, seen[text])
    return out


def _pct(v, size):
    return round(100 * min(max(v, 0), size) / size)


def render_elements(kept, W, H):
    """`[i] label | class_short | flags | x:a-b% y:c-d%`, i = 1..n in kept order. The ONE renderer."""
    lines = []
    for i, e in enumerate(kept, 1):
        b = e["bbox"]
        flags = ",".join(FLAG_WORDS[f] for f in ELEMENT_FLAGS if e["flags"][f]) or "-"
        lines.append(f"[{i}] {element_label(e)} | {e['class_short'] or '-'} | {flags} | "
                     f"x:{_pct(b['x_min'], W)}-{_pct(b['x_max'], W)}% y:{_pct(b['y_min'], H)}-{_pct(b['y_max'], H)}%")
    return lines


def build_state_v2(goal, history, kept, W, H):
    """Goal, previous actions, the rendered element list. No memory, trajectory or app-intent lines."""
    return "\n".join(_goal_history_lines(goal, history) + [V2_ELEMENTS_HEADER] + render_elements(kept, W, H))


def element_inputs(goal, history, elements, W, H, rule):
    """(state, kept, kept indices) for one screen under contract v2 — called by train, eval and serve."""
    kept, idx = select_elements(elements, W, H, rule)
    return build_state_v2(goal, history, kept, W, H), kept, idx


def element_centre(e):
    """Where a tap on this element lands: its bbox centre (AndroidWorld taps elements the same way)."""
    b = e["bbox"]
    return [(b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2]


def centre_in_box(b, target):
    """True iff the centre of bbox `b` lies inside `target` (edges inclusive) — a tap on `b` is credited."""
    if b is None or target is None:
        return False
    cx, cy = (b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2
    return target["x_min"] <= cx <= target["x_max"] and target["y_min"] <= cy <= target["y_max"]


def chosen_element(ans, kept):
    """(rendered id, kept element) the `target_el` answer names, or (None, None)."""
    choice = str(((ans or {}).get("target_el") or {}).get("choice") or "")
    if choice.isdigit() and 1 <= int(choice) <= len(kept):
        return int(choice), kept[int(choice) - 1]
    return None, None


# ----------------------------------------------------------------------------- contracts
# A checkpoint is valid only with the questions it was trained on. Measured 2026-11-10: E2 was trained
# with 8 action options and 5 questions; the serving code had since grown to 9 options and 6 questions
# and sent them to every checkpoint. Each contract names the exact question set; `contract.json` in the
# checkpoint directory says which one a checkpoint speaks, and the server refuses a checkpoint without it.
CONTRACTS = {
    # ckpt-E2-final / appint_full: the 2026-10-29 code
    "e2": {"actions": ("click", "write", "long_press", "scroll", "wait", "navigate_back", "open_app",
                       "terminate"), "is_done": False},
    # episodic-s0 (2026-11-09): navigate_home + the withdrawn is_done readout
    "episodic": {"actions": tuple(ACTIONS), "is_done": True},
    # 2026-11-10 AndroidWorld contract: navigate_home kept (an AW action), is_done dropped (its label is
    # a restatement of `action == status`, 137/137 rows — see 2026-11-10-episodic-s0-fails-prereg.md)
    "v1": {"actions": tuple(ACTIONS), "is_done": False},
    # 2026-11-10 R2: v1's 9 actions; the element is CHOSEN from the AndroidWorld element list
    # (`target_el`), there is no `target_box`. VA only. `element_rule` is stamped on the checkpoint and
    # the server applies the stamped rule, so a later change of this default cannot reach old weights.
    "v2": {"actions": tuple(ACTIONS), "is_done": False, "elements": True, "element_rule": "cap40_reading"},
    # 2026-11-11 v2a: v2 + the app is CHOSEN from a list of app names given with the request (AndroidWorld's
    # launchable apps at serving; the gold app + distractors in training), instead of copied out of the goal.
    "v2a": {"actions": tuple(ACTIONS), "is_done": False, "elements": True, "element_rule": "cap40_reading",
            "app_choice": True},
}
DEFAULT_CONTRACT = "v1"
# Contracts that exist for one track only (a v2 state is meaningless without the element list).
CONTRACT_TRACKS = {"v2": ("VA",), "v2a": ("VA",)}


def write_contract(ckpt_dir, name, tracks=("V",)):
    """Record, next to the weights, which question set and which tracks this checkpoint was trained on."""
    if name not in CONTRACTS:
        raise KeyError(name)
    only = CONTRACT_TRACKS.get(name)
    if only is not None and list(tracks) != list(only):
        raise ValueError(f"contract {name!r} is defined for tracks {list(only)} only, not {list(tracks)}")
    import json
    with open(os.path.join(ckpt_dir, "contract.json"), "w") as fh:
        json.dump({"name": name, "tracks": list(tracks),
                   **{k: list(v) if isinstance(v, tuple) else v for k, v in CONTRACTS[name].items()}},
                  fh, indent=1)


def read_contract(ckpt_dir):
    """The full contract document. Missing file -> FileNotFoundError: never guess a checkpoint's contract."""
    import json
    with open(os.path.join(ckpt_dir, "contract.json")) as fh:
        doc = json.load(fh)
    if doc.get("name") not in CONTRACTS:
        raise KeyError(doc.get("name"))
    if CONTRACTS[doc["name"]].get("elements") and doc.get("element_rule") not in ELEMENT_RULES:
        # The element list the model reads depends on the rule; serving it under a guessed rule is a
        # train/serve skew by construction.
        raise KeyError(f"contract {doc['name']!r} needs a known element_rule, got {doc.get('element_rule')!r}")
    doc.setdefault("tracks", ["V"])
    return doc


def load_contract(ckpt_dir):
    return read_contract(ckpt_dir)["name"]


def questions_for(doc, track, cands=None, apps=None):
    """Questions for a request against a checkpoint's contract; refuses a track it was never trained on."""
    if track not in doc["tracks"]:
        raise ValueError(f"track {track!r} was not trained for this checkpoint (trained: {doc['tracks']})")
    return build_questions(track, cands, contract=doc["name"], apps=apps)


def render_history_line(action):
    """ONE history line for one executed action — used to build training history from gold actions and
    serve-time history from executed actions, so the two can never drift apart again.

    Bare action token plus the one argument the action carries (text / direction / app name); never
    coordinates, outcomes or a `Step N:` prefix (`build_state` adds that).
    """
    at = (action or {}).get("action_type") or "wait"
    at = {"input_text": "type_text"}.get(at, at)
    arg = {"type_text": "text", "scroll": "direction", "open_app": "app_name"}.get(at)
    val = str((action or {}).get(arg) or "").strip() if arg else ""
    return f"{at} {val}" if val else at


def build_questions(track, cands=None, contract=DEFAULT_CONTRACT, apps=None):
    """Ordered dict of API-JSON questions for `contract`. Order is fixed; train, eval and serve share it.

    Contract v2: `cands` is the KEPT canonical element list (`element_inputs`), and the element is chosen,
    not boxed — action, target_el, scroll_dir, app_name, type_text."""
    spec = CONTRACTS[contract]
    if spec.get("elements"):
        return _build_questions_v2(track, cands, spec, apps)
    q = {
        "action": {"type": "choice", "instructions": "The next UI action to perform to advance the task.",
                   "criteria": {k: ACTIONS[k] for k in spec["actions"]}},
        "target_box": {"type": "box", "instructions": "The UI element the next action acts on, in the picture."},
    }
    if track == "VA":
        q["target_el"] = {"type": "choice", "instructions": "The element the next action acts on.",
                          "criteria": {str(c["idx"]): (c["label"] or f"element at {c['pos']}") for c in cands}}
    q["scroll_dir"] = {"type": "choice", "instructions": "The direction, if the next action is a scroll.",
                       "criteria": dict(SCROLL)}
    q["app_name"] = {"type": "text", "instructions": "The exact app name, if the next action launches an app."}
    q["type_text"] = {"type": "text", "instructions": "The exact text to type, if the next action enters text."}
    if spec["is_done"]:
        q["is_done"] = {"type": "noul",
                        "instructions": "Has the task already been completed on this screen?"}
    return q


def _build_questions_v2(track, kept, spec, apps=None):
    if track != "VA":
        raise ValueError(f"an element-list contract is asked on the VA track only, not {track!r}")
    q = {"action": {"type": "choice", "instructions": "The next UI action to perform to advance the task.",
                    "criteria": {k: ACTIONS[k] for k in spec["actions"]}}}
    if kept:  # a choice needs at least one option; with none kept, a spatial action has no element
        q["target_el"] = {"type": "choice", "instructions": "The element the next action acts on.",
                          "criteria": option_texts(kept)}
    q["scroll_dir"] = {"type": "choice", "instructions": "The direction, if the next action is a scroll.",
                       "criteria": dict(SCROLL)}
    if spec.get("app_choice"):
        if not apps:
            raise ValueError("contract with app_choice needs the list of app names (`apps`)")
        q["app_name"] = {"type": "choice", "instructions": "The app to launch, if the next action launches an app.",
                         "criteria": {str(i): name for i, name in enumerate(apps, 1)}}
    else:
        q["app_name"] = {"type": "text", "instructions": "The exact app name, if the next action launches an app."}
    q["type_text"] = {"type": "text", "instructions": "The exact text to type, if the next action enters text."}
    return q


def app_options(gold, vocab, k=24, seed=""):
    """The app names offered for one training row: the gold app (if any) plus k distractors from `vocab`,
    in an order fixed by `seed` (same row -> same list, every epoch). Case-insensitive de-duplication."""
    import random as _r
    rng = _r.Random(str(seed))
    gold_l = (gold or "").strip().lower()
    pool = sorted({v for v in vocab if v and v.strip().lower() != gold_l}, key=str.lower)
    picks = rng.sample(pool, min(k, len(pool)))
    out = ([gold.strip()] if gold_l else []) + picks
    rng.shuffle(out)
    return out


def row_app_options(rec, vocab, k=24):
    """The app list one row is trained/evaluated with: its own `apps` (generator rows carry AndroidWorld's list),
    else the gold app (open_app rows) plus k distractors, seeded by (episode, step)."""
    if rec.get("apps"):
        return list(rec["apps"])
    a = rec.get("action") or {}
    gold = a.get("app_name") if a.get("action_type") == "open_app" else None
    return app_options(gold, vocab, k=k, seed="%s:%s" % (rec.get("ep"), rec.get("step")))


def decode_app_choice(answer, apps):
    """App name a `choice` answer names, or "" when the key is not a listed option."""
    key = str((answer or {}).get("choice") or "")
    return apps[int(key) - 1] if key.isdigit() and 1 <= int(key) <= len(apps) else ""


def length_buckets(lengths, micro):
    """Indices grouped into micro-batches of `micro`, by length (shortest first), so a micro-batch pads to a similar
    length — as d1's runner groups requests (`SystemOne._batches`). With micro == 1 the original order is kept."""
    if micro <= 1:
        return [[i] for i in range(len(lengths))]
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    return [order[k:k + micro] for k in range(0, len(order), micro)]


FIRST_STEP_ACTIONS = ("open_app", "status")


def restrict_first_step(probs, history):
    """At the first step of a task (empty history) the agent may only launch an app or declare done: every
    AndroidWorld task starts on the home screen, whose few icons rarely include the task's app. Later steps:
    plain argmax. `probs` maps action names to probabilities."""
    pool = probs if history else ({a: p for a, p in probs.items() if a in FIRST_STEP_ACTIONS} or probs)
    return max(pool, key=pool.get)


def find_span(state, text):
    """Char span of `text` in the state (case-insensitive, first occurrence after the 'Goal:' line start
    preferred); None if absent."""
    if not text:
        return None
    m = re.search(re.escape(text.strip()), state, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None


# ----------------------------------------------------------------------------- app-name memory
_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("the a an to in on of for and app my me i with from at by is be this that it as or open".split())


def toks(s):
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


class AppMemory:
    """TF-IDF over training goals -> app names of their open_app steps (and app mentions)."""

    def __init__(self, items):  # items: [(episode, goal, app_name)]
        self.items = items
        df = {}
        self.docs = []
        for _, g, _ in items:
            ts = set(toks(g))
            self.docs.append(ts)
            for t in ts:
                df[t] = df.get(t, 0) + 1
        n = len(items)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.inv = {}
        for i, ts in enumerate(self.docs):
            for t in ts:
                self.inv.setdefault(t, []).append(i)
        self.norm = [math.sqrt(sum(self.idf[t] ** 2 for t in ts)) or 1.0 for ts in self.docs]

    def query(self, goal, k=3, exclude_ep=None):
        q = set(toks(goal))
        score = {}
        for t in q:
            w = self.idf.get(t, 0) ** 2
            for i in self.inv.get(t, ()):
                score[i] = score.get(i, 0) + w
        ranked = sorted(score, key=lambda i: -score[i] / self.norm[i])
        out = []
        for i in ranked:
            ep, _, app = self.items[i]
            if ep == exclude_ep or app in out:
                continue
            out.append(app)
            if len(out) == k:
                break
        return out


class TrajMemory(AppMemory):
    """Procedural memory (cf. Mem-W, here in-domain): TRAIN-split episodes retrieved by goal similarity,
    shown as their step-instruction sequences. items: [(episode, goal, [step instructions])]."""

    def query_trajs(self, goal, k=2, exclude_ep=None, max_steps=12):
        q = set(toks(goal))
        score = {}
        for t in q:
            w = self.idf.get(t, 0) ** 2
            for i in self.inv.get(t, ()):
                score[i] = score.get(i, 0) + w
        out = []
        for i in sorted(score, key=lambda i: -score[i] / self.norm[i]):
            ep, g, steps = self.items[i]
            if ep == exclude_ep:
                continue
            out.append((g, steps[:max_steps]))
            if len(out) == k:
                break
        return out


# ----------------------------------------------------------------------------- app intent
# LEAK WARNING. APP_INTENT is a HAND-WRITTEN key: goal-keyword -> app name. When it fires, build_state
# appends "Application for this task: <app>" to the state, and the pointer read then copies the answer out
# of the state. That makes open_app accuracy unmeasurable: the model is handed the answer.
#
# Measured, and why this stays OFF by default (ALLOW_APP_INTENT=False):
#   - it fires on the goals that already name no app, which is exactly the 10.5% of open_app rows that
#     require world knowledge;
#   - "pick the nearest clickable node" already scores 61.9% without seeing the image, so any figure that
#     improves when the answer is placed in the state is a lower bound on leakage, not a capability.
# Set ALLOW_APP_INTENT=True only to reproduce the historical (disclosed) scaffolding runs. Never train a
# headline number with it on.
ALLOW_APP_INTENT = False

APP_INTENT = {
    "settings": ["wifi", "wi-fi", "bluetooth", "brightness", "airplane mode", "volume", "flashlight"],
    "contacts": ["new contact", "contact for", "contact draft", "contact screen"],
    "camera": ["take one photo", "take one video", "take a photo", "take a video"],
    "clock": ["stopwatch", "timer", "alarm"],
    "simple gallery pro": ["gallery"],
    "files": ["delete file", "move file", "file manager"],
    "simple sms messenger": ["send a text", "send an sms", "text message", "sms to"],
    "dialer": ["call ", "dial ", "phone number"],
    "markor": ["markor"],
    "pro expense": ["expense", "receipt"],
    "simple calendar pro": ["calendar event", "add an event", "meeting on", "calendar"],
    "retro music": ["play music", "music player", "playlist", "retro music"],
    "vlc": ["vlc", "play the video"],
    "osmand": ["directions to", "navigation", "the map"],
    "tasks": ["to-do", "todo list", "task app"],
    "joplin": ["joplin"],
    "chrome": ["private window", "web search", "browser", "open a website"],
    "audio recorder": ["record audio", "voice recording"],
    "clipper": ["clipboard"],
    "simple draw pro": ["draw", "sketch"],
    "open tracks sports tracker": ["sports tracker", "workout"],
}


def app_intent_for(goal):
    """The app a goal implies, if it implies one WITHOUT naming a launchable app first.

    Order matters: the first table entry whose hint appears wins, and an app the goal already names is
    preferred over any inference (so "Create a note in Markor" never becomes something else).
    """
    g = (goal or "").lower()
    named = [a for a in APP_INTENT if a in g]
    if named:
        return named[0]
    for app, hints in APP_INTENT.items():
        if any(h in g for h in hints):
            return app
    return None


# ----------------------------------------------------------------------------- gold answers
def launcher_icon_target(rec, cands):
    """The gold element for an `open_app` executed on the home screen.

    AndroidControl stores NO target bbox for `open_app` (measured: 986/986 rows have target=None), so the
    icon tap has never been supervised. But when the row's screen IS the launcher and the candidate list
    contains the app's own label, the element is not guessed — it is on screen, and this returns it.

    Measured on 6 shards: of 611 launcher-visible open_app rows, the app label IS a candidate on 43.0%.
    The other 57% are rows where the app was already foregrounded (the annotation artifact described in
    DATA-SPEC §1), and there is nothing to select.

    Returns the candidate dict, or None. Never invents a bbox.
    """
    t = rec.get("target")
    if t:
        return None
    app = (rec.get("action") or {}).get("app_name") or ""
    app = app.strip().lower()
    if not app or not cands:
        return None
    if not is_launcher_screen(rec):
        return None
    for c in cands:
        if app in (c.get("label") or "").lower():
            return c
    return None


def is_launcher_screen(rec):
    """True when this row's screen was the launcher. The rebuilt corpus keeps the flag (`on_launcher`);
    the raw corpus carries the package names, so both are accepted."""
    if rec.get("on_launcher") is not None:
        return bool(rec["on_launcher"])
    pkgs = {(n.get("package_name") or "") for n in (rec.get("nodes") or [])}
    return any("launcher" in p for p in pkgs)


def gold_answers(rec, track, cands, state):
    """Supervision for one record: {question: target}. Box target = edges in [0,1] of the gold element."""
    a = rec["action"]
    at = a["action_type"]
    g = {"action": TO_OPTION[at]}
    W, H = rec["W"], rec["H"]
    t = rec.get("target")
    if at in SPATIAL and t is not None:
        g["target_box"] = (t["y_min"] / H, t["y_max"] / H, t["x_min"] / W, t["x_max"] / W)  # top,bottom,left,right
        if track == "VA":
            c = gold_candidate(cands, t, a.get("x"), a.get("y"))
            if c is not None:
                g["target_el"] = str(c["idx"])
    if at == "open_app" and track == "VA":
        # No stored bbox on `open_app`, but a launcher icon IS an element on screen. Supervise the
        # selection when the app's label is a candidate — that teaches "find the icon", which is a
        # vision problem, instead of handing the model the name through APP_INTENT.
        ic = launcher_icon_target(rec, cands)
        if ic is not None:
            g["target_el"] = str(ic["idx"])
            b = ic["bbox"]
            g["target_box"] = (b["y_min"] / H, b["y_max"] / H, b["x_min"] / W, b["x_max"] / W)
    if at == "scroll":
        g["scroll_dir"] = a["direction"]
    span = find_span(state, a.get("app_name")) if at == "open_app" else None
    if at == "open_app" and span is None and a.get("app_name") and ALLOW_APP_INTENT:
        # Same leak as build_state above, on the gold side. Guarded by the same switch: with the switch
        # off, an app name that is not in the state has NO read supervision, which is the honest state of
        # affairs (the pointer genuinely has nothing to copy).
        app = a["app_name"]
        rec["_app_hint"] = app
        state = state + "\nApplication for this task: " + app
        span = find_span(state, app)
        rec["state_with_app_hint"] = state
    g["app_name"] = span if span is not None else ("nothing" if at != "open_app" else None)
    g["type_text"] = find_span(state, a.get("text")) if at == "type_text" else "nothing"
    # is_done: free label from the corpus invariant (status is always the final row, never mid-trajectory).
    g["is_done"] = is_done_gold(rec.get("step") is not None
                                and rec.get("n_steps") is not None
                                and rec["step"] == rec["n_steps"])
    if g["app_name"] is None:
        del g["app_name"]  # gold not copyable from the state: no read supervision
    if g["type_text"] is None:
        # NOT deleted: a non-copyable string is the case a pointer read cannot serve, and deleting it
        # teaches the model nothing about when to escalate (DATA-SPEC §4). Label it instead.
        g["type_text"] = "escalate"
    return g


def _centre_in(e, t):
    b = e["bbox"]
    cx, cy = (b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2
    return t["x_min"] <= cx <= t["x_max"] and t["y_min"] <= cy <= t["y_max"]


def gold_target_el_v2(rec, keep_idx, kept):
    """(`target_el` gold id or None, case) for one row under contract v2.

    The executed point is the chosen element's CENTRE and the scorer credits a point inside the gold target, so
    the gold must be an element whose centre lies inside `rec["target"]` — supervising anything else teaches a
    point the evaluation can never credit (the R4 defect class, fixed for v1 in `gold_candidate`). Order:
      "idx"        — the element `target_el_idx` names (smallest AW element containing the gold point) survived
                     the filter AND its centre is inside the target;
      "creditable" — else the smallest KEPT element whose centre is inside the target;
      "none"       — else no element supervision for this row (its action is still supervised).
    Revised 2026-11-10 before any v2 training: the first version also accepted a named element whose centre was
    outside the target (198 train rows) and left 498 rows with a creditable kept element unsupervised.
    Non-spatial rows: (None, None).
    """
    if rec["action"]["action_type"] not in SPATIAL:
        return None, None
    t = rec.get("target")
    if not t:
        return None, "none"
    pos = {orig: i for i, orig in enumerate(keep_idx, 1)}
    orig = rec.get("target_el_idx")
    if orig is not None and orig in pos and _centre_in(kept[pos[orig] - 1], t):
        return str(pos[orig]), "idx"
    inside = [i for i, e in enumerate(kept, 1) if _centre_in(e, t)]
    if inside:
        area = lambda i: ((kept[i - 1]["bbox"]["x_max"] - kept[i - 1]["bbox"]["x_min"]) *
                          (kept[i - 1]["bbox"]["y_max"] - kept[i - 1]["bbox"]["y_min"]))
        return str(min(inside, key=lambda i: (area(i), i))), "creditable"
    return None, "none"


def gold_answers_v2(rec, kept, keep_idx, state, apps=None, contract="v2"):
    """Supervision for one row under contract v2 -> (gold, target_el case). `kept`/`keep_idx` come from
    `element_inputs` on this row; the action, scroll, app and text golds are v1's (`gold_answers`)."""
    g = gold_answers(rec, "V", None, state)
    g.pop("target_box", None)
    g.pop("is_done", None)
    if CONTRACTS[contract].get("app_choice"):
        g.pop("app_name", None)
        a = rec.get("action") or {}
        if a.get("action_type") == "open_app" and apps:
            want = (a.get("app_name") or "").strip().lower()
            hit = [i for i, name in enumerate(apps, 1) if name.strip().lower() == want]
            if hit:
                g["app_name"] = str(hit[0])
    el, case = gold_target_el_v2(rec, keep_idx, kept)
    if el is not None:
        g["target_el"] = el
    return g, case
