"""Scenario suite for DMAgent: tasks on real Android apps, each with a setup and an objective check over ADB.

A task counts as completed only when its check passes on the device after the run (not when the agent says so).
Categories: system settings (toggles and values), notes (create / edit / delete a file), contacts, and information
questions (the answer is compared with the value read from the device).

Two sets. TUNED: the first 12 tasks; the harness fixes of 2026-11-16 were found by running them, so results on them are
optimistic. HELD_OUT: 8 tasks written afterwards and never used to change the agent; report them separately.
Every setup makes the task NOT complete (checked by check_setups.py), and `teardown` restores a state that would affect
later tasks.
"""
import re
from typing import Callable, List, Optional

MARKOR = "/sdcard/Documents/markor"


class Scenario:
    def __init__(self, name: str, task: str, category: str, setup: List[str], check: Callable, apps=(),
                 held_out: bool = False, teardown: List[str] = ()):
        self.name, self.task, self.category, self.setup, self.check, self.apps = name, task, category, setup, check, apps
        self.held_out, self.teardown = held_out, list(teardown)


def sh(device, cmd: str) -> str:
    return device.shell(cmd).strip()


def setting(ns, key, want):
    return lambda d, ans: sh(d, "settings get %s %s" % (ns, key)) == want


def file_contains(path, *words):
    def check(d, ans):
        txt = sh(d, "cat %s 2>/dev/null" % path).lower()
        return bool(txt) and all(w.lower() in txt for w in words)
    return check


def file_absent(path):
    return lambda d, ans: sh(d, "ls %s 2>/dev/null" % path) == ""


def answer_equals(cmd, pattern=r"\d+"):
    def check(d, ans):
        want = re.search(pattern, sh(d, cmd))
        got = re.search(pattern, ans or "")
        return bool(want and got and want.group(0) == got.group(0))
    return check


def answer_contains(cmd=None, *words):
    """The answer contains every word (and, with `cmd`, the value that command prints on the device)."""
    def check(d, ans):
        want = [w.lower() for w in words] + ([sh(d, cmd).lower()] if cmd else [])
        return bool(ans) and all(w in ans.lower() for w in want)
    return check


def contact_absent(name):
    return lambda d, ans: name.lower() not in sh(
        d, "content query --uri content://com.android.contacts/data --projection display_name").lower()


def font_larger(d, ans):
    try:
        return float(sh(d, "settings get system font_scale")) > 1.0
    except ValueError:
        return False


CLEAR_CONTACTS = "pm clear com.android.providers.contacts"
OLD_FRIEND = [CLEAR_CONTACTS,
              "content insert --uri content://com.android.contacts/raw_contacts --bind account_type:n: "
              "--bind account_name:n:",
              "content insert --uri content://com.android.contacts/data --bind raw_contact_id:i:1 "
              "--bind mimetype:s:vnd.android.cursor.item/name --bind data1:s:'Old Friend'",
              "content insert --uri content://com.android.contacts/data --bind raw_contact_id:i:1 "
              "--bind mimetype:s:vnd.android.cursor.item/phone_v2 --bind data1:s:0987654321 --bind data2:i:2"]


def contact_exists(name, number):
    def check(d, ans):
        rows = sh(d, "content query --uri content://com.android.contacts/data --projection display_name:data1")
        digits = re.sub(r"\D", "", number)
        return any(name.lower() in r.lower() and digits in re.sub(r"\D", "", r) for r in rows.splitlines()) or \
            (name.lower() in rows.lower() and digits in re.sub(r"\D", "", rows))
    return check


SCENARIOS: List[Scenario] = [
    Scenario("wifi_off", "Turn off Wi-Fi.", "settings", ["svc wifi enable"], setting("global", "wifi_on", "0")),
    Scenario("wifi_on", "Turn on Wi-Fi.", "settings", ["svc wifi disable"], setting("global", "wifi_on", "1")),
    Scenario("bluetooth_off", "Turn off Bluetooth.", "settings", ["cmd bluetooth_manager enable"],
             setting("global", "bluetooth_on", "0")),
    Scenario("dark_theme_on", "Turn on dark theme.", "settings", ["cmd uimode night no"],
             lambda d, a: "yes" in sh(d, "cmd uimode night").lower()),
    Scenario("dnd_on", "Turn on Do Not Disturb.", "settings", ["cmd notification set_dnd off"],
             lambda d, a: sh(d, "settings get global zen_mode") not in ("0", "", "null")),
    Scenario("screen_timeout", "Set the screen timeout to 10 minutes.", "settings",
             ["settings put system screen_off_timeout 30000"], setting("system", "screen_off_timeout", "600000")),
    Scenario("note_create", "Create a new note in Markor named groceries with the text: bread and butter.", "notes",
             ["rm -f %s/groceries.md %s/groceries.txt" % (MARKOR, MARKOR)],
             lambda d, a: file_contains("%s/groceries.md" % MARKOR, "bread", "butter")(d, a) or
             file_contains("%s/groceries.txt" % MARKOR, "bread", "butter")(d, a), apps=("net.gsantner.markor",)),
    Scenario("note_append", "In Markor, open the note todo and add a new line: call mom.", "notes",
             ["mkdir -p %s" % MARKOR, "echo 'buy stamps' > %s/todo.md" % MARKOR],
             file_contains("%s/todo.md" % MARKOR, "buy stamps", "call mom"), apps=("net.gsantner.markor",)),
    Scenario("note_delete", "Delete the note old_ideas in Markor.", "notes",
             ["mkdir -p %s" % MARKOR, "echo 'nothing important' > %s/old_ideas.md" % MARKOR],
             file_absent("%s/old_ideas.md" % MARKOR), apps=("net.gsantner.markor",)),
    # The setup clears the contacts first. Until 2026-11-16 it was empty, so the contact added by the first run stayed on
    # the device and later runs of this task passed the check whatever the agent did.
    Scenario("contact_add", "Add a new contact named Alex Nguyen with the phone number 0912345678.", "contacts",
             [CLEAR_CONTACTS], contact_exists("Alex Nguyen", "0912345678"), apps=("com.google.android.contacts",)),
    Scenario("battery_level", "What is the current battery level of this phone? Answer with just the number.",
             "question", [], answer_equals("dumpsys battery | grep level")),
    Scenario("android_version", "Which Android version is this phone running? Answer with just the number.",
             "question", [], answer_equals("getprop ro.build.version.release")),
    # ---- held out: written after the harness fixes, never used to change the agent
    Scenario("airplane_on", "Turn on airplane mode.", "settings", ["cmd connectivity airplane-mode disable"],
             setting("global", "airplane_mode_on", "1"), held_out=True,
             teardown=["cmd connectivity airplane-mode disable"]),
    Scenario("auto_rotate_on", "Turn on auto-rotate screen.", "settings",
             ["settings put system accelerometer_rotation 0"], setting("system", "accelerometer_rotation", "1"),
             held_out=True),
    Scenario("location_off", "Turn off location.", "settings", ["cmd location set-location-enabled true"],
             lambda d, a: sh(d, "cmd location is-location-enabled") == "false", held_out=True,
             teardown=["cmd location set-location-enabled true"]),
    Scenario("font_larger", "Make the font size larger.", "settings", ["settings put system font_scale 1.0"],
             font_larger, held_out=True, teardown=["settings put system font_scale 1.0"]),
    Scenario("note_rename", "In Markor, rename the note draft to final.", "notes",
             ["mkdir -p %s" % MARKOR, "rm -f %s/final.md %s/final" % (MARKOR, MARKOR),
              "echo 'first version' > %s/draft.md" % MARKOR],
             lambda d, a: file_absent("%s/draft.md" % MARKOR)(d, a) and
             (file_contains("%s/final.md" % MARKOR, "first version")(d, a) or
              file_contains("%s/final" % MARKOR, "first version")(d, a)),
             apps=("net.gsantner.markor",), held_out=True),
    Scenario("contact_delete", "Delete the contact Old Friend.", "contacts", OLD_FRIEND, contact_absent("Old Friend"),
             apps=("com.google.android.contacts",), held_out=True),
    Scenario("note_question", "In Markor, what is written in the note shopping?", "question",
             ["mkdir -p %s" % MARKOR, "echo 'milk and eggs' > %s/shopping.md" % MARKOR],
             answer_contains(None, "milk", "eggs"), apps=("net.gsantner.markor",), held_out=True),
    Scenario("device_name", "What is the device name of this phone? Answer with just the name.", "question", [],
             answer_contains("settings get global device_name"), held_out=True),
]


def by_name(names: Optional[List[str]] = None, held_out: Optional[bool] = None) -> List[Scenario]:
    return [s for s in SCENARIOS if (not names or s.name in names) and (held_out is None or s.held_out == held_out)]
