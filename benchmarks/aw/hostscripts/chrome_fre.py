#!/usr/bin/env python3
"""Click through Chrome's first-run screens once, before any episode (2026-11-12 error analysis).

Upstream `ChromeApp.setup` (android_world/env/setup_device/apps.py) clicks "Accept & continue" and "No thanks", but this
image's Chrome shows "Use without an account" first, so setup left the first-run flow open and BrowserDraw /
BrowserMaze / BrowserMultiply started on it. Taps only buttons in KNOWN; an unknown screen is never touched.

    python3 chrome_fre.py [ADB] [chrome|markor]   (inside the emulator container; launches the app, clicks through, closes it)
"""
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

KNOWN = ("Use without an account", "Accept & continue", "No thanks", "Got it", "Skip")
# Markor's intro, exactly the buttons upstream MarkorApp.setup clicks (setup_device/apps.py). 2026-11-12: one emulator
# (aw-emu3) showed this intro in 6/6 Markor episodes while the others opened on the file list.
MARKOR_KNOWN = ("NEXT", "DONE", "OK")


def buttons(xml_text, known=KNOWN):
    """[(text, (cx, cy))] of nodes whose text or content-desc is in `known`."""
    out = []
    for n in ET.fromstring(xml_text).iter("node"):
        label = (n.get("text") or n.get("content-desc") or "").strip()
        if label in known:
            m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", n.get("bounds") or "")
            if m:
                x0, y0, x1, y1 = map(int, m.groups())
                out.append((label, ((x0 + x1) // 2, (y0 + y1) // 2)))
    return out


def click_through(dump, tap, max_screens=8, settle=0.0, known=KNOWN):
    """Tap the first known button on each screen until none is left; returns the number of taps."""
    taps = 0
    for _ in range(max_screens):
        found = buttons(dump(), known)
        if not found:
            break
        tap(*found[0][1])
        taps += 1
        if settle:
            time.sleep(settle)
    return taps


def main(adb="adb", app="chrome"):
    sh = lambda *a: subprocess.run([adb, "shell", *a], capture_output=True, text=True, timeout=30).stdout
    if app == "markor":
        pkg, known = "net.gsantner.markor", MARKOR_KNOWN
        sh("monkey", "-p", pkg, "-c", "android.intent.category.LAUNCHER", "1")
    else:
        pkg, known = "com.android.chrome", KNOWN
        sh("am", "start", "-n", "com.android.chrome/com.google.android.apps.chrome.Main")
    time.sleep(4)

    def dump():
        sh("uiautomator", "dump", "/sdcard/fre.xml")
        return sh("cat", "/sdcard/fre.xml") or "<hierarchy/>"

    n = click_through(dump, lambda x, y: sh("input", "tap", str(x), str(y)), settle=2.0, known=known)
    sh("am", "force-stop", pkg)
    print("%s_FRE taps=%d" % (app.upper(), n))


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["adb"]))
