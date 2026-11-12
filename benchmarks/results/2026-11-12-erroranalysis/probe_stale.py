"""Diagnosing-bugs loop for the stale a11y tree. One command, N repetitions, per repetition: click "Internet" on
Settings > Network & internet, then at fixed times read (a) RAW upstream env.get_state(False), (b) RAW get_state(True),
(c) our bridge (screenshot()+elements()). Prints when each reader first shows the new page ("Wi-Fi")."""
import sys, time
sys.path[:0] = ["/tmp/v11/aw", "/tmp/v11/train"]
from awenv.awenv_bridge import build
from android_world.env import adb_utils
env = build(console_port=5554, adb_path="/opt/android/platform-tools/adb", emulator_setup=False)
L = lambda t, c: (t or c or "")
def raw(stab):
    st = env._env.get_state(wait_to_stabilize=stab)
    labs = [L(e.text, e.content_description) for e in st.ui_elements]
    pk = {e.package_name for e in st.ui_elements}
    return labs, pk
def bridge():
    env.screenshot(); els = env.elements()
    return [L(e.get("text"), e.get("content_description")) for e in els], {e.get("package_name") for e in els}
def state_of(labs, pk):
    if "Wi-Fi" in labs: return "NEW"
    if "Calls & SMS" in labs: return "OLD"
    if pk <= {"com.android.systemui"}: return "SYSUI"
    return "OTHER"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for rep in range(N):
    env.reset()
    adb_utils.issue_generic_request(["shell", "am", "force-stop", "com.android.settings"], env.controller); time.sleep(1)
    adb_utils.issue_generic_request(["shell", "am", "start", "-a", "android.settings.WIRELESS_SETTINGS"], env.controller)
    for _ in range(20):
        time.sleep(1); labs, _ = raw(False)
        if "Calls & SMS" in labs: break
    st = env._env.get_state(False); tgt = [e for e in st.ui_elements if L(e.text, e.content_description) == "Internet"][0]
    b = tgt.bbox_pixels
    t0 = time.time(); env.execute({"action_type": "click", "x": int((b.x_min + b.x_max) / 2), "y": int((b.y_min + b.y_max) / 2)})
    line = []
    for when in (0.3, 1.5, 3.0, 4.5, 6.0, 8.0):
        while time.time() - t0 < when: time.sleep(0.05)
        a = state_of(*raw(False)); c = state_of(*bridge()); tb = time.time(); s = state_of(*raw(True)); ts = time.time() - tb
        line.append("t=%.1f raw=%s bridge=%s stab=%s(%.1fs)" % (when, a, c, s, ts))
    print("rep %d: %s" % (rep, " | ".join(line)), flush=True)
