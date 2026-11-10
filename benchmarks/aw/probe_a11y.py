#!/usr/bin/env python3
"""Probe the a11y chain through the AndroidWorld HTTP server, before a run.

`adb shell uiautomator dump` is NOT android_world's a11y path (it argues with
the window-idle state), so it can neither confirm nor refute health. The real
chain is:

    AccessibilityForwarder (in Android) --gRPC 8554--> A11yGrpcWrapper servicer
        --task_extras--> AndroidEnv --get_state--> State(pixels, forest, ui_elements)

`POST /reset` then `GET /screenshot` exercises exactly that, because the server's
`get_state()` builds `ui_elements` from the a11y forest. If the forwarder has
stopped delivering protos this is where it surfaces — as
`RuntimeError: Could not get a11y tree.` and later as
`setup_error` with `n_steps: 0` on every episode.

Instantiating our own controller instead would be wrong twice over: it opens a
second A11yGrpcWrapper (a new local gRPC port, and it re-broadcasts SET_GRPC to
the forwarder, hijacking the server's channel), and it needs the forwarder APK
over the network.

Exit 0 = chain healthy. Exit 1 = chain broken, do not start a run.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _post(url, timeout):
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read(400).decode("utf-8", "replace")


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def probe(base, attempts=3, gap=5.0):
    ok = 0
    last = ""
    for i in range(attempts):
        t0 = time.time()
        try:
            st, body = _post("%s/reset?go_home=true" % base, 120)
            if st != 200:
                last = "reset HTTP %s %s" % (st, body[:120])
                print("attempt %d: FAIL reset HTTP %s" % (i, st))
                continue
            st, raw = _get("%s/screenshot?wait_to_stabilize=false" % base, 120)
            doc = json.loads(raw.decode("utf-8"))
            px = len(doc.get("pixels") or [])
            if st != 200 or px == 0:
                last = "screenshot HTTP %s pixels=%d" % (st, px)
                print("attempt %d: FAIL screenshot HTTP %s pixels=%d" % (i, st, px))
                continue
            print("attempt %d: OK  %.1fs  pixels=%d  (state built => a11y forest present)"
                  % (i, time.time() - t0, px))
            ok += 1
        except urllib.error.HTTPError as exc:
            last = "HTTP %s: %s" % (exc.code, exc.read(300).decode("utf-8", "replace"))
            print("attempt %d: FAIL %s" % (i, last[:200]))
        except Exception as exc:  # noqa: BLE001
            last = "%s: %s" % (type(exc).__name__, str(exc)[:160])
            print("attempt %d: FAIL %s" % (i, last))
        if i + 1 < attempts:
            time.sleep(gap)
    verdict = "PASS" if ok == attempts else "FAIL"
    print("A11Y_PROBE %s (%d/%d)%s" % (verdict, ok, attempts,
                                        "" if verdict == "PASS" else "  last=" + last[:200]))
    return 0 if ok == attempts else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args()
    return probe(a.base, a.attempts)


if __name__ == "__main__":
    sys.exit(main())
