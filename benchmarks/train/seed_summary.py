#!/usr/bin/env python3
"""Step and episode success of each training seed of the final recipe on AndroidControl, scored in AndroidControl's
action space (no "done"), from the per-step files of score_actest.py. Seed 0 is the released DMA-2B.

    python benchmarks/train/seed_summary.py  ->  benchmarks/results/2026-11-16-ac-noterm/seeds_summary.json
"""
import json
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "benchmarks/results/2026-11-16-ac-noterm")
FILES = {0: {"low": "perstep-ours_actest_low_noterm.jsonl", "high": "perstep-ours_actest_noterm.jsonl"}}
for s in (1, 2):
    FILES[s] = {sp: f"seeds/perstep-exec2_s{s}_{sp}_noterm.jsonl" for sp in ("low", "high")}


def score(path):
    rows = [json.loads(l) for l in open(path)]
    ep = {}
    for r in rows:
        ep.setdefault(r["episode_id"], []).append(r["exact_match"])
    return {"steps": len(rows), "exact": sum(r["exact_match"] for r in rows),
            "sr": round(100 * sum(r["exact_match"] for r in rows) / len(rows), 2),
            "type": round(100 * sum(r["type_match"] for r in rows) / len(rows), 2),
            "episodes": len(ep), "episodes_all_right": sum(all(v) for v in ep.values()),
            "episode_sr": round(100 * sum(all(v) for v in ep.values()) / len(ep), 2)}  # as paper_ac_final.py


def summary():
    out = {"seeds": {}}
    for s, f in FILES.items():
        if all(os.path.exists(os.path.join(R, p)) for p in f.values()):
            out["seeds"][str(s)] = {sp: score(os.path.join(R, p)) for sp, p in f.items()}
    for sp in ("low", "high"):
        for k in ("sr", "episode_sr"):
            v = [x[sp][k] for x in out["seeds"].values()]
            out[f"{sp}_{k}"] = {"n": len(v), "mean": statistics.mean(v),
                                "sd": statistics.stdev(v) if len(v) > 1 else None,
                                "min": min(v), "max": max(v)}
    return out


if __name__ == "__main__":
    s = summary()
    json.dump(s, open(os.path.join(R, "seeds_summary.json"), "w"), indent=1)
    for sp in ("low", "high"):
        x = s[f"{sp}_sr"]
        print(sp, {k: s["seeds"][k][sp]["sr"] for k in s["seeds"]}, "mean %.2f sd %s" % (x["mean"], x["sd"]))
