"""Scenario suite for DMAgent: tasks on real Android apps, each with a setup and an objective check over ADB.

A task counts as completed only when its check passes on the device after the run (not when the agent says so).
Categories: system settings (toggles and values), notes (create / edit / delete a file), contacts, and information
questions (the answer is compared with the value read from the device).
"""
import re
from typing import Callable, List, Optional

MARKOR = "/sdcard/Documents/markor"


class Scenario:
    def __init__(self, name: str, task: str, category: str, setup: List[str], check: Callable, apps=()):
        self.name, self.task, self.category, self.setup, self.check, self.apps = name, task, category, setup, check, apps


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
    Scenario("contact_add", "Add a new contact named Alex Nguyen with the phone number 0912345678.", "contacts",
             [], contact_exists("Alex Nguyen", "0912345678"), apps=("com.google.android.contacts",)),
    Scenario("battery_level", "What is the current battery level of this phone? Answer with just the number.",
             "question", [], answer_equals("dumpsys battery | grep level")),
    Scenario("android_version", "Which Android version is this phone running? Answer with just the number.",
             "question", [], answer_equals("getprop ro.build.version.release")),
]


def by_name(names: Optional[List[str]] = None) -> List[Scenario]:
    return [s for s in SCENARIOS if not names or s.name in names]
