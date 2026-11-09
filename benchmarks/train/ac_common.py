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
    "open_app": "Launch an app by its name",
    "terminate": "End the task: the goal is already achieved on this screen",
}
# eval naming <-> option name
TO_OPTION = {"click": "click", "type_text": "write", "long_press": "long_press", "scroll": "scroll",
             "wait": "wait", "navigate_back": "navigate_back", "open_app": "open_app", "status": "terminate"}
FROM_OPTION = {v: k for k, v in TO_OPTION.items()}
SCROLL = {"up": "content moves up (swipe up)", "down": "content moves down (swipe down)",
          "left": "content moves left", "right": "content moves right"}
SPATIAL = ("click", "long_press", "type_text")


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


def build_state(goal, track, cands=None, apps=None, history=None, trajs=None, app_intent=None):
    """history = the earlier steps' instructions, as the AC-v2 harness gives them (row['previous_actions'];
    GUI-Libra guilibra_plan.py 'Interaction History'). Part of the leaderboard protocol."""
    lines = [f"Goal: {goal}"]
    if history:
        lines.append("Previous actions:")
        lines.extend(f"Step {i}: {h}" for i, h in enumerate(history, 1))
    else:
        lines.append("Previous actions: none")
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


def build_questions(track, cands=None):
    """Ordered dict of API-JSON questions. Order is fixed; train and eval share it."""
    q = {
        "action": {"type": "choice", "instructions": "The next UI action to perform to advance the task.",
                   "criteria": dict(ACTIONS)},
        "target_box": {"type": "box", "instructions": "The UI element the next action acts on, in the picture."},
    }
    if track == "VA":
        q["target_el"] = {"type": "choice", "instructions": "The element the next action acts on.",
                          "criteria": {str(c["idx"]): (c["label"] or f"element at {c['pos']}") for c in cands}}
    q["scroll_dir"] = {"type": "choice", "instructions": "The direction, if the next action is a scroll.",
                       "criteria": dict(SCROLL)}
    q["app_name"] = {"type": "text", "instructions": "The exact app name, if the next action launches an app."}
    q["type_text"] = {"type": "text", "instructions": "The exact text to type, if the next action enters text."}
    return q


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
# When a goal implies an app without naming it, the pointer read has nothing to copy. The fix is not to
# retrain the read slot (it is a copy operation, verified: it emits "Markor"/"Joplin" it never saw) but to
# put the app name into the state. This table is the single source of truth for train AND serve, so the
# rendered string is identical on both paths.
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
    if at == "scroll":
        g["scroll_dir"] = a["direction"]
    span = find_span(state, a.get("app_name")) if at == "open_app" else None
    if at == "open_app" and span is None and a.get("app_name"):
        # The app name is not literally in the state, so the pointer read has nothing to copy. Two options:
        # drop the supervision (what this code did, and the model then copied the goal noun instead of the
        # app: "Turn wifi on" -> app_name="wifi"), or append the name to the state so the read has a target.
        # We append, because the served state can carry the same line (see build_state apps= / APP_INTENT).
        app = a["app_name"]
        rec["_app_hint"] = app
        state = state + "\nApplication for this task: " + app
        span = find_span(state, app)
        rec["state_with_app_hint"] = state
    g["app_name"] = span if span is not None else ("nothing" if at != "open_app" else None)
    g["type_text"] = find_span(state, a.get("text")) if at == "type_text" else "nothing"
    if g["app_name"] is None:
        del g["app_name"]  # gold not copyable from the state: no read supervision
    if g["type_text"] is None:
        del g["type_text"]
    return g
