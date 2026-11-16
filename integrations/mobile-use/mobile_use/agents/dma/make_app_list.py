"""Build the app list (`app_list.json`) for a connected Android device.

DMA-2B chooses which app to open from a list of names, and the agent maps the chosen name to a package. The list must
match the apps installed on the device. This script reads the launchable packages over ADB and names each one by its
label, read from the APK when `apkutils` is installed (`pip install apkutils`); otherwise, or when the label cannot be
read, it uses the last word of the package name. Edit the result to add aliases people use ("play store").

  python -m mobile_use.agents.dma.make_app_list --serial emulator-5554 --out app_list.json

Give the same file to the DMA-2B API (`D1_APPS`) and to the agent (`--apps-json`).
"""
import argparse
import json
import os
import re
import tempfile
from typing import Callable, Dict, List, Optional

GENERIC = {"android", "app", "apps", "mobile", "free", "pro", "lite", "beta", "debug"}


def parse_launchables(query_output: str) -> List[str]:
    """Packages from `cmd package query-activities --brief -a MAIN -c LAUNCHER`, in order, without duplicates."""
    out: List[str] = []
    for line in query_output.splitlines():
        m = re.match(r"\s*([A-Za-z][\w.]*)/[\w.$]+\s*$", line)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def name_from_package(pkg: str) -> str:
    parts = [p for p in pkg.split(".") if p.lower() not in GENERIC] or pkg.split(".")
    word = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", parts[-1]).replace("_", " ")
    return word.lower()


SKIP = {"com.github.uiautomator"}       # uiautomator2's own helper app, installed by the agent


def clean_label(label: Optional[str]) -> str:
    """A usable app label, or "" (labels read from an APK are sometimes the wrong string resource)."""
    label = " ".join(str(label or "").split()).lower()
    return label if 0 < len(label) <= 30 and len(label.split()) <= 4 and "." not in label else ""


def build_app_list(packages: List[str], label_of: Callable[[str], Optional[str]]) -> List[Dict[str, object]]:
    rows = []
    for pkg in packages:
        if pkg in SKIP:
            continue
        rows.append((pkg, clean_label(label_of(pkg)) or name_from_package(pkg)))
    counts: Dict[str, int] = {}
    for _, name in rows:
        counts[name] = counts.get(name, 0) + 1
    apps = []
    for pkg, base in rows:
        # two apps with one label ("Calendar") get the publisher's word, so the model can tell them apart
        name = base if counts[base] == 1 else "%s (%s)" % (base, (pkg.split(".") + [""])[1].lower())
        aliases = [name] + [a for a in (base, name_from_package(pkg)) if a != name]
        apps.append({"name": name, "aliases": list(dict.fromkeys(aliases)), "package": pkg})
    return apps


def apk_label(device, pkg: str) -> Optional[str]:
    try:
        import apkutils
    except ImportError:
        return None
    try:
        path = device.shell("pm path %s" % pkg).split()[0].split(":", 1)[1]
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "base.apk")
            device.sync.pull(path, local)
            with apkutils.APK.from_file(local) as apk:
                apk.parse_resource()
                return apk.app_name
    except Exception:  # noqa: BLE001 - a missing label falls back to the package name
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None)
    ap.add_argument("--out", default="app_list.json")
    a = ap.parse_args()
    import adbutils
    device = adbutils.adb.device(serial=a.serial)
    pkgs = parse_launchables(device.shell(
        "cmd package query-activities --brief -a android.intent.action.MAIN -c android.intent.category.LAUNCHER"))
    apps = build_app_list(pkgs, lambda p: apk_label(device, p))
    json.dump(apps, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("%d apps -> %s" % (len(apps), a.out))
    for x in apps:
        print("  %-28s %s" % (x["name"], x["package"]))


if __name__ == "__main__":
    main()
