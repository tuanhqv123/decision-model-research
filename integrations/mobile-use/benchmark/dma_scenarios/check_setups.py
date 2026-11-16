"""Sanity check of the suite: after each setup the task must NOT already be complete (questions excluded)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adbutils
from scenarios import SCENARIOS
d = adbutils.AdbClient(host="127.0.0.1", port=5037).device(sys.argv[1] if len(sys.argv) > 1 else "emulator-5554")
for sc in SCENARIOS:
    for cmd in sc.setup:
        d.shell(cmd)
    done = sc.check(d, None)
    print("%-16s setup ok: %s" % (sc.name, "yes" if not done else "NO - already complete"))
