#!/usr/bin/env python3
"""Per-container app-state check (2026-11-12), run inside the emulator container before a chain starts.

Why: on one emulator (aw-emu3) Markor opened on its intro ('Main View', NEXT x4, DONE) and a storage-permission
dialog in 6/6 Markor episodes of run aw116_v4g_s0, while the other emulators opened on the file list. Those episodes
did not start in AndroidWorld's set-up state. The repair is upstream's own `MarkorApp.setup` (clear data, NEXT x4,
DONE, OK, all-files access), applied only when the app shows its intro, and read through AndroidWorld's a11y tree
(never `uiautomator dump`, which kills AndroidWorld's accessibility forwarder).

Also (2026-11-12): OsmAnd's map_markers table, which upstream OsmAndMarker's initialize needs and OsmAnd creates
only on a full start (3 of 4 runs lost OsmAndMarker to 'no such table: map_markers').

    python3 app_setup.py [--check-only]      (cwd = the harness's aw/ directory inside the container)
Prints APP_OK / APP_FIXED / APP_FAIL per app; exit 1 if any app is still not set up.
"""
import sys
import time

INTRO = {
    "markor": ("net.gsantner.markor", ("Main View", "NEXT", "Storage permission required")),
}


def needs_setup(app, elements):
    """True only when the app's own intro or permission screen is visible (an empty read is not evidence)."""
    pkg, marks = INTRO[app]
    labels = {(e.get("text") or e.get("content_description") or "").strip()
              for e in elements if e.get("package_name") == pkg}
    return any(m in labels for m in marks)


OSMAND_DB = "/data/data/net.osmand/databases/map_markers_db"


def has_table(sqlite_out, table="map_markers"):
    """True when `sqlite3 DB .tables` listed the table (names are whitespace-separated)."""
    return table in (sqlite_out or "").split()


def file_size(ls_line):
    """Size column of one `ls -l` line, or None when the file does not exist."""
    parts = (ls_line or "").split()
    return int(parts[4]) if len(parts) >= 8 and parts[0][:1] in "-l" and parts[4].isdigit() else None


def ensure_osmand_markers(adb_utils, controller, check_only=False, wait_s=40):
    """OsmAndMarker's upstream initialize (_clear_db) needs map_markers; OsmAnd creates it on a full start.
    `sqlite3` is only ever run on an existing non-empty file: run as root on a missing DB it leaves a 0-byte root-owned
    file that OsmAnd can never open (found on 3 of 6 emulators, 2026-11-12); such a file is removed first."""
    run = lambda args: adb_utils.issue_generic_request(args, controller).generic.output.decode(errors="ignore")  # noqa

    def size():
        try:
            return file_size(run(["shell", "ls", "-l", OSMAND_DB]))
        except Exception:
            return None

    def out():
        if not size():
            return ""
        try:
            return run(["shell", "sqlite3", OSMAND_DB, ".tables"])   # no spaces: adb shell re-splits arguments
        except Exception:
            return ""
    if has_table(out()):
        print("APP_OK osmand_markers")
        return 0
    if check_only:
        print("APP_FAIL osmand_markers (table missing; --check-only)")
        return 1
    if size() == 0:
        run(["shell", "rm", "-f", OSMAND_DB, OSMAND_DB + "-journal"])
        print("APP_NOTE osmand_markers removed an empty root-owned map_markers_db")
    adb_utils.launch_app("osmand", controller)
    for _ in range(wait_s // 4):
        time.sleep(4.0)
        if has_table(out()):
            break
    adb_utils.close_app("osmand", controller)
    ok = has_table(out())
    print("APP_FIXED osmand_markers" if ok else "APP_FAIL osmand_markers (table still missing)")
    return 0 if ok else 1


def main(check_only=False):
    sys.path.insert(0, ".")
    from android_world.env import adb_utils
    from android_world.env.setup_device import apps
    from awenv.awenv_bridge import build
    env = build(console_port=5554, adb_path="/opt/android/platform-tools/adb", emulator_setup=False)
    raw = env._env
    setups = {"markor": apps.MarkorApp}
    bad = 0
    for app in INTRO:
        adb_utils.launch_app(app, raw.controller)
        time.sleep(3.0)
        before = needs_setup(app, env.elements())
        adb_utils.close_app(app, raw.controller)
        if not before:
            print("APP_OK %s" % app)
            continue
        if check_only:
            print("APP_FAIL %s (intro visible; --check-only)" % app)
            bad += 1
            continue
        try:
            setups[app].setup(raw)
        except ValueError as e:   # e.g. all-files access already granted: its toggle text is absent; the re-check decides
            print("APP_NOTE %s upstream setup stopped: %s" % (app, e))
        adb_utils.launch_app(app, raw.controller)
        time.sleep(3.0)
        after = needs_setup(app, env.elements())
        adb_utils.close_app(app, raw.controller)
        print(("APP_FAIL %s (still on intro after upstream setup)" if after else "APP_FIXED %s") % app)
        bad += int(after)
    bad += ensure_osmand_markers(adb_utils, raw.controller, check_only)
    env.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(check_only="--check-only" in sys.argv[1:]))
