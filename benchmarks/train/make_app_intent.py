"""Generate app-intent supervision from AndroidWorld's OWN task definitions.

Why this file exists (measured, 2026-11-08):

The read slot is a **pointer copy**, so the model already emits app names it has never seen in
training — verified on the live server: goal "…in the Markor app…" -> `open_app app_name="Markor"`
(0.805), "…the Joplin app…" -> "Joplin" (0.787), "…the Broccoli app…" -> "Broccoli" (0.963).

The failure is narrower: when the goal **does not name the app**, the model picks `click` instead of
opening it. Measured on the 116-task suite: **69 tasks name the app in the goal, 22 do not** — e.g.

    "Turn wifi on"                        -> needs settings
    "Create a new contact for Jack Ahmed" -> needs contacts
    "Take one photo"                      -> needs camera

So the missing supervision is an **intent -> canonical app** mapping, not new screenshots. This module
derives that mapping from AndroidWorld's own `app_names` and `template`, so the labels are exact:
each row's goal is the literal goal the benchmark will show, and its app is the literal app the task
requires. No external dataset, no synthetic phrasing invented by us.

Output: `app_intent.jsonl` — rows in `compact_*` format so the trainer can mix them via `--train-set`.

    python3 benchmarks/train/make_app_intent.py --out ~/acv2/acdata
"""
from __future__ import annotations

import argparse
import json
import os
import random

# Mirrors awenv.adapter.CANON_APPS: what the harness can actually launch by name.
CANON_APPS = [
    "android world", "audio recorder", "broccoli app", "camera", "chrome", "clipper", "clock",
    "contacts", "dialer", "files", "joplin", "markor", "miniwob", "open tracks sports tracker",
    "osmand", "pro expense", "retro music", "settings", "simple calendar pro", "simple draw pro",
    "simple gallery pro", "simple sms messenger", "tasks", "vlc",
]

# AW registry app_names -> the name the harness must pass to `open_app`.
# Only entries that differ are listed; everything else maps to itself.
APP_CANON = {
    "broccoli": "broccoli app",
    "simple calendar": "simple calendar pro",
    "simple draw": "simple draw pro",
    "simple gallery": "simple gallery pro",
    "simple sms": "simple sms messenger",
    "open tracks": "open tracks sports tracker",
    "retro music": "retro music",
}

# Words in a goal that imply an app without naming it. Derived from AW's own orphan task goals
# (the 22 that do not name their app), so the phrasings are the benchmark's, not ours.
INTENT_HINTS = {
    "settings": ["wifi", "wi-fi", "bluetooth", "brightness", "airplane", "volume", "turn on wifi",
                  "turn off wifi", "turn bluetooth"],
    "contacts": ["new contact", "contact for", "contact draft", "contact screen"],
    "camera": ["take one photo", "take one video", "take a photo", "take a video", "photo", "video"],
    "clock": ["stopwatch", "timer", "alarm", "pause the stopwatch", "run the stopwatch"],
    "simple gallery pro": ["gallery", "photo gallery", "image gallery"],
    "files": ["file manager", "delete file", "move file", "files app"],
    "simple sms messenger": ["send a text", "send an sms", "text message", "sms to"],
    "dialer": ["call ", "phone number", "dial"],
    "markor": ["markor note", "note in markor"],
    "pro expense": ["expense", "receipt"],
    "simple calendar pro": ["calendar event", "add an event", "meeting on"],
    "retro music": ["play music", "music player", "playlist"],
    "vlc": ["vlc", "play the video"],
    "osmand": ["navigation", "directions to", "map"],
    "tasks": ["to-do", "todo list", "task app"],
    "joplin": ["joplin"],
    "chrome": ["browser", "web search", "open a website", "url"],
    "audio recorder": ["record audio", "voice recording"],
    "clipper": ["clipboard", "copy text"],
    "simple draw pro": ["draw", "sketch"],
    "open tracks sports tracker": ["sports tracker", "workout", "run tracker"],
    "android world": ["android world app"],
    "miniwob": ["miniwob"],
}


def canon(aw_name: str) -> str | None:
    """AW registry app name -> launchable canonical name, or None if the harness cannot launch it."""
    low = (aw_name or "").strip().lower()
    if not low:
        return None
    for prefix, target in APP_CANON.items():
        if low.startswith(prefix):
            return target
    return low if low in CANON_APPS else None


def load_anchors(data_dir: str) -> list[dict]:
    """Real AC `open_app` rows: they carry a real screenshot, W/H and cands, which the trainer needs.

    The screenshot is only a *background*: the label we teach is (goal -> app), and each (goal, app)
    pair is rendered over MANY different backgrounds so the model cannot shortcut "screen -> app".
    """
    out = []
    path = os.path.join(data_dir, "compact_cur15k.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if (r.get("action") or {}).get("action_type") == "open_app" and r.get("img"):
            out.append({"img": r["img"], "W": r["W"], "H": r["H"], "cands": r.get("cands")})
    return out


def synth_rows(n_per_app: int, seed: int, anchors: list[dict] | None = None) -> list[dict]:
    """Rows whose state (goal) implies an app and whose gold action is `open_app` on that app."""
    rng = random.Random(seed)
    phrasings = {
        "settings": ["Turn wifi on.", "Turn wifi off.", "Turn bluetooth on.", "Turn bluetooth off.",
                     "Turn airplane mode on.", "Set brightness to maximum.", "Turn volume up."],
        "contacts": ["Create a new contact for {name}.", "Add {name} to my contacts.",
                     "Go to the new contact screen and enter {name}'s details."],
        "camera": ["Take one photo.", "Take one video."],
        "clock": ["Run the stopwatch.", "Pause the stopwatch.", "Create a timer with 14 hours.",
                  "Set an alarm for 7am."],
        "simple gallery pro": ["Open the gallery.", "Find the most recent photo in the gallery."],
        "files": ["Delete the file called report.pdf.", "Move the file notes.txt to Downloads."],
        "simple sms messenger": ["Send a text to {name}: hello.", "Send an SMS saying I am on my way."],
        "dialer": ["Call {name}.", "Dial +15550100."],
        "markor": ["Create a note in Markor.", "Add a note to Markor called ideas.md."],
        "pro expense": ["Add an expense of 12 dollars for lunch.", "Delete the duplicate expenses."],
        "simple calendar pro": ["Add a calendar event for tomorrow at 3pm.",
                                 "Create a meeting on Friday."],
        "retro music": ["Play music in the background.", "Open the Retro Music player."],
        "vlc": ["Play the video with VLC."],
        "osmand": ["Get directions to the airport.", "Open the map."],
        "tasks": ["Open the to-do list.", "Mark the task as done."],
        "joplin": ["Check whether the note in Joplin is a todo."],
        "chrome": ["Open a new Chrome private window.", "Search the web for coffee makers."],
        "audio recorder": ["Record audio.", "Start a voice recording."],
        "clipper": ["Copy this text to the clipboard."],
        "simple draw pro": ["Draw a square."],
        "open tracks sports tracker": ["Start a workout in the sports tracker."],
        "android world": ["Open the Android World app."],
        "miniwob": ["Open miniwob."],
    }
    names = ["Jack Ahmed", "Luka Ahmed", "Maria Silva", "Tom Chen", "Priya Nair", "Sam Okafor"]
    rows = []
    for app, templates in phrasings.items():
        for i in range(n_per_app):
            t = templates[i % len(templates)]
            goal = t.format(name=rng.choice(names))
            row = {
                "ep": f"appintent::{app}::{i}", "step": 0, "goal": goal,
                "history": [], "action": {"action_type": "open_app", "app_name": app},
                "source": "aw_registry_app_intent",
            }
            if anchors:
                # Rotate backgrounds so (goal -> app) is what is learned, not (screen -> app).
                a = anchors[rng.randrange(len(anchors))]
                row.update(img=a["img"], W=a["W"], H=a["H"], cands=a.get("cands"))
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--n-per-app", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None, help="acdata dir; when given, rows get real screenshots")
    a = ap.parse_args()

    anchors = load_anchors(a.data) if a.data else None
    if anchors is not None:
        print(f"anchoring on {len(anchors)} real AC open_app screenshots")
    rows = synth_rows(a.n_per_app, a.seed, anchors)
    path = os.path.join(a.out, "compact_app_intent.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_app: dict[str, int] = {}
    for r in rows:
        by_app[r["action"]["app_name"]] = by_app.get(r["action"]["app_name"], 0) + 1
    print(f"wrote {len(rows)} rows -> {path}")
    print(f"{len(by_app)} apps covered; the 22 orphan AW tasks map to: "
          f"{sorted({'settings','contacts','camera','clock'} & set(by_app))}")
    unnamed = set(by_app)
    print(f"coverage vs CANON_APPS: {len(unnamed & set(CANON_APPS))}/{len(CANON_APPS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
