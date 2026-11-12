"""Compare AndroidWorld runs (official SR, buckets, loop statistics, paired wins/losses).

    python3 aw_compare.py NAME=run_dir[,run_dir...] NAME=...   (first NAME is the reference for pairing)

Reads `episodes/*.jsonl` (not `.abandoned.`), last `result` row per task; SR via awenv.metrics.official_sr.
SMS tasks are reported separately: the emulator has no working SIM (modem simulator unresponsive, 2026-11-12)."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "aw"))
from awenv.metrics import official_sr  # noqa: E402

ANSWER = set(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "aw", "hostscripts",
                               "aw116_answer_only.txt")).read().split())


def is_sms(t):
    return "sms" in t.lower()


def load(dirs):
    res, steps, setups = {}, {}, set()
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "episodes", "*.jsonl"))):
            if ".abandoned." in f:
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            r = [x for x in rows if x.get("type") == "result"]
            if not r:
                continue
            t = r[-1]["task_name"]
            res[t] = r[-1]
            head = rows[0] if rows and rows[0].get("type") == "episode" else {}
            # protocol + agent policy: a number carries the setup that produced it (AW_PROTOCOL; 2026-11-12 review)
            setups.add(json.dumps({"protocol": head.get("protocol"), "policy": head.get("policy")}, sort_keys=True))
            steps[t] = [x for x in rows if x.get("type") == "step"]
    return res, steps, setups


def stats(res, steps, setups):
    o = official_sr(list(res.values()))
    succ = sorted(t for t, s in o["per_task"].items() if s >= 1.0)
    no_sms = {t: s for t, s in o["per_task"].items() if not is_sms(t)}
    fail = [t for t, s in o["per_task"].items() if s < 1.0]
    prem = sum(1 for t in fail if res[t].get("declared_done"))
    nstep = sum(len(v) for v in steps.values())
    nochg = [s for v in steps.values() for s in v if s.get("ui_changed") is False or s.get("screen_changed") is False]
    rep = tot = 0
    for v in steps.values():
        for a, b in zip(v, v[1:]):
            if a.get("screen_changed") is False:
                tot += 1
                rep += (a.get("action_executed") == b.get("action_executed"))
    longrun = 0
    for t in fail:
        acts = [json.dumps(s.get("action_executed"), sort_keys=True) for s in steps[t]]
        run = best = 0
        for i, a in enumerate(acts):
            run = run + 1 if i and a == acts[i - 1] else 1
            best = max(best, run)
        longrun += best >= 5
    sc_idx = [s for v in steps.values() for s in v if (s.get("action_executed") or {}).get("action_type") == "scroll"]
    return dict(sr=o["sr"], n=o["n_tasks_scored"], excl=o["excluded"], succ=succ,
                sr_nosms=sum(no_sms.values()) / len(no_sms), n_nosms=len(no_sms), n_succ_nosms=sum(1 for s in no_sms.values() if s >= 1),
                fail=len(fail), premature=prem, budget=len(fail) - prem, steps=nstep, nochange=len(nochg),
                rep_after_nochange=(rep, tot), fail_long_repeat=longrun,
                scroll=len(sc_idx), scroll_with_index=sum(1 for s in sc_idx if s["action_executed"].get("index") is not None),
                answer_succ=sum(1 for t in succ if t in ANSWER), per_task=o["per_task"], setups=sorted(setups))


def main():
    runs = []
    for a in sys.argv[1:]:
        name, dirs = a.split("=", 1)
        runs.append((name, stats(*load(dirs.split(",")))))
    for name, s in runs:
        print("== %s" % name)
        for st in s["setups"]:
            print("  setup: %s" % st)
        if len(s["setups"]) > 1:
            print("  WARNING: this run mixes %d setups; its rate is not one configuration's" % len(s["setups"]))
        # the official SR averages scores, and composite tasks score 0.5: print the SUM of scores, not the count of
        # full successes (review 2026-11-12: "13.5% (15/115)" mixed 15.5/115 with 15 full successes)
        print("  official SR %.2f%% (score sum %g/%d scored; %d full successes; excluded %s)" % (
            100 * s["sr"], sum(s["per_task"].values()), s["n"], len(s["succ"]), s["excl"]))
        print("  without SMS tasks: %.1f%% (%d/%d)" % (100 * s["sr_nosms"], s["n_succ_nosms"], s["n_nosms"]))
        print("  failed %d: declared done but not done %d, step budget %d" % (s["fail"], s["premature"], s["budget"]))
        print("  steps %d, no change %d (%.0f%%); same action after no change %d/%d; failed eps with >=5 identical in a row %d"
              % (s["steps"], s["nochange"], 100 * s["nochange"] / max(1, s["steps"]), *s["rep_after_nochange"], s["fail_long_repeat"]))
        print("  scrolls %d, with index %d; answer-task successes %d" % (s["scroll"], s["scroll_with_index"], s["answer_succ"]))
        print("  successes: %s" % ", ".join(s["succ"]))
    if len(runs) > 1:
        ref_name, ref = runs[0]
        for name, s in runs[1:]:
            common = set(ref["per_task"]) & set(s["per_task"])
            w = sorted(t for t in common if s["per_task"][t] > ref["per_task"][t])
            l = sorted(t for t in common if s["per_task"][t] < ref["per_task"][t])
            from math import comb
            n = len(w) + len(l)
            p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(len(w), len(l)) + 1)) / 2 ** n) if n else 1.0
            if s["setups"] != ref["setups"]:
                print("  NOTE: %s and %s differ in protocol/policy (see setup lines); the paired difference is the whole setup" % (name, ref_name))
            print("== paired %s vs %s on %d common tasks: wins %d, losses %d, sign test p=%.3f" % (name, ref_name, len(common), len(w), len(l), p))
            print("  wins: %s" % ", ".join(w))
            print("  losses: %s" % ", ".join(l))


if __name__ == "__main__":
    main()
