#!/usr/bin/env python3
"""Rows that teach the FIRST action of an AndroidWorld task: on the home screen, goal -> open_app(<the task's app>).

Runs inside the emulator container (it imports android_world). No task is run: goals come from AndroidWorld's own
parameter generator, the app from each template's `app_names`, the screen from home screens captured once with the
harness (`/tmp/home_<k>.png` + `.json`, the element dicts the agent sends).

    python3 make_aw_app_rows.py <out_dir> [--per-template 40] [--min-copies 10]

Writes to <out_dir>:
  aw_apps.json             [{"name", "aliases", "package"}] — every installed app AndroidWorld can launch by name
  compact_awgen_v2a.jsonl  practice rows (never the official test instances: instance seed 30 is refused)
  compact_awgen_heldout.jsonl rows of the held-out templates (one third per app; never trained on)
  heldout_templates.json   the held-out template names (fixed: every third template per app, by name)
  imgs/awhome/home_<k>.jpg the home screens at width 720 (the stored width of the AndroidControl training images)

Only single-app templates are used (a multi-app goal's first app is ambiguous); information-retrieval templates have
no `app_names` and are skipped.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import subprocess

from PIL import Image

TEST_SEED = 30  # run.py `task_random_seed` default: the official test instances
# Templates whose `app_names` is not the FIRST app the goal asks for (read from the goals, 2026-11-11): the Browser
# goals say "Open the file task.html in Downloads in the file manager; when prompted open it with Chrome".
FIRST_APP_OVERRIDE = {"BrowserDraw": "files", "BrowserMaze": "files", "BrowserMultiply": "files"}
ADB = "/opt/android/platform-tools/adb"


def instance_seed(seed, name, i=0):
    """suite_utils.create_suite._get_instance_seed, verbatim formula."""
    return int(hashlib.sha256(f"{seed}_{name}_{i}".encode()).hexdigest(), 16) % (2 ** 32)


def launchable_apps():
    from android_world.env import adb_utils as A
    out = subprocess.run([ADB, "-P", "5037", "-s", "emulator-5554", "shell", "pm", "list", "packages"],
                         capture_output=True, text=True).stdout
    pk = set(out.replace("package:", "").split())
    if not pk:
        raise SystemExit("no packages listed: is the emulator up?")
    apps = []
    for pattern, activity in A._PATTERN_TO_ACTIVITY.items():
        package = activity.split("/")[0]
        if package in pk:
            aliases = pattern.split("|")
            assert A.get_adb_activity(aliases[0]) == activity, aliases[0]
            apps.append({"name": aliases[0], "aliases": aliases, "package": package})
    return apps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-template", type=int, default=40)
    ap.add_argument("--min-copies", type=int, default=10)
    a = ap.parse_args()
    from android_world import registry
    from android_world.env import adb_utils as A

    apps = launchable_apps()
    names = [x["name"] for x in apps]
    os.makedirs(os.path.join(a.out, "imgs", "awhome"), exist_ok=True)
    json.dump(apps, open(os.path.join(a.out, "aw_apps.json"), "w"), indent=1)

    homes = []
    for k in range(3):
        png = "/tmp/home_%d.png" % k
        if not os.path.exists(png):
            continue
        im = Image.open(png).convert("RGB")
        W, H = im.size
        rel = "imgs/awhome/home_%d.jpg" % k
        im.resize((720, round(H * 720 / W)), Image.BICUBIC).save(os.path.join(a.out, rel), quality=90)
        homes.append((rel, W, H, json.load(open("/tmp/home_%d.json" % k))))
    if not homes:
        raise SystemExit("no captured home screen in /tmp/home_<k>.png")

    reg = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)

    def option_for(app):  # the template's app name -> the option the model is shown
        app_l = app.lower()
        for x in apps:
            if app_l in [s.lower() for s in x["aliases"]] or re.match(x["aliases"][0], app_l):
                return x["name"]
        return None

    by_app = collections.defaultdict(list)
    single = {}
    for name, cls in sorted(reg.items()):
        app_names = tuple(getattr(cls, "app_names", ()) or ())
        if len(app_names) != 1:
            continue
        opt = option_for(FIRST_APP_OVERRIDE.get(name, app_names[0]))
        if opt is None:
            print("skip (app not launchable):", name, app_names)
            continue
        single[name] = opt
        by_app[opt].append(name)
    heldout = sorted(n for names_ in by_app.values() for i, n in enumerate(sorted(names_)) if i % 3 == 2)
    json.dump(heldout, open(os.path.join(a.out, "heldout_templates.json"), "w"), indent=1)

    test_goals = set()
    for name in single:
        random.seed(instance_seed(TEST_SEED, name))
        test_goals.add(reg[name](reg[name].generate_random_params()).goal)

    counts = collections.Counter()
    with open(os.path.join(a.out, "compact_awgen_v2a.jsonl"), "w") as tr, \
            open(os.path.join(a.out, "compact_awgen_heldout.jsonl"), "w") as ho:
        for name, opt in sorted(single.items()):
            cls, goals, s = reg[name], [], 31
            while len(goals) < a.per_template and s < 31 + 20 * a.per_template:
                random.seed(instance_seed(s, name))
                s += 1
                try:
                    g = cls(cls.generate_random_params()).goal
                except Exception as e:  # noqa: BLE001
                    print("param error", name, type(e).__name__)
                    break
                if g not in goals:
                    goals.append(g)
            novel = [g for g in goals if g not in test_goals] or goals  # parameter-free templates repeat the test goal text
            while len(novel) < a.min_copies:
                novel = novel + novel[: a.min_copies - len(novel)]
            fh = ho if name in heldout else tr
            for j, g in enumerate(novel):
                rel, W, H, els = homes[j % len(homes)]
                fh.write(json.dumps({"ep": "awgen_%s_%d" % (name, j), "step": 0, "goal": g, "history": [],
                                     "img": rel, "W": W, "H": H,
                                     "action": {"action_type": "open_app", "app_name": opt}, "target": None,
                                     "elements": els, "apps": names, "source": "aw_generator",
                                     "template": name}) + "\n")
                counts["heldout" if name in heldout else "train"] += 1
    print("apps", len(names), names)
    print("single-app templates", len(single), "held out", len(heldout), dict(counts))
    print("per app (train templates):", {k: len([n for n in v if n not in heldout]) for k, v in by_app.items()})


if __name__ == "__main__":
    main()
