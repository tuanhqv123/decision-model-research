"""Paired reading of probe_v3.json (2026-11-12): per probe, mean paired difference, 95% bootstrap interval (states resampled),
and how many states moved in the expected direction by more than 0.01.
    python3 probe_paired.py probe_v3.json"""
import json, random, sys
R = json.load(open(sys.argv[1])); random.seed(0)
def pair(a, b, expect):
    x, y = R[a], R[b]; d = [j - i for i, j in zip(x, y)]
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(4000))
    good = sum((v < -0.01) if expect == "down" else (v > 0.01) if expect == "up" else abs(v) <= 0.01 for v in d)
    print("%-70s n=%3d  diff %+.3f  95%% [%+.3f, %+.3f]  states in expected direction: %d/%d (expect %s)" % (
        "%s -> %s" % (a.split(" |")[0], b.split(" |")[0]) + " | " + a.split("| ")[1], len(d), sum(d) / len(d), bs[100], bs[3899], good, len(d), expect))
pair("P1 screen changed | P(same element)", "P1 no change | P(same element)", "down")
pair("P1 screen changed | P(same type)", "P1 no change | P(same type)", "down")
pair("P2 failed 1x | P(click same element)", "P2 failed 4x | P(click same element)", "down")
pair("P3 +0 wait lines at start | P(status)", "P3 +10 wait lines at start | P(status)", "flat")
pair("P4 done state, goal as is | P(status)", "P4 done state, goal + unfinished clause | P(status)", "down")
pair("P5 as served (no focus) | P(type_text)", "P5 field marked focused | P(type_text)", "up")
