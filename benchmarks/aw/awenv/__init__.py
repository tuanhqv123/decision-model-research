"""AndroidWorld evaluation harness (in-process environment, remote model).

Protocol string for every reported number: see PROTOCOL.
"""

PROTOCOL = "aw-main-v1"


def harness_version() -> str:
    """A stable id for this harness build, usable when git is not present in the runtime.

    Runs happen inside the android_world image, which has no checkout; recording "unknown"
    there makes a number unattributable, so the package's own contents become the version.
    """
    import hashlib
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha1()
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(here, name)
        digest.update(name.encode())
        digest.update(str(os.path.getsize(path)).encode())
    return "awenv-" + digest.hexdigest()[:10]


__all__ = ["PROTOCOL", "harness_version"]
