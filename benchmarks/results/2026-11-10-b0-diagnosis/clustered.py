"""Episode-clustered diagnosis statistics (pre-registration amendment A2-bis). FROZEN 2026-11-10 15:30 UTC,
before any trace of the confirmation set was opened.

Usage:
  python clustered.py <b0_export.json> --set inspected|confirmation|all [--inspected inspected_tasks.txt]

Inputs come from `export_steps.py`. A step's "no visible effect" = the next image the model received differs
from this step's image in <0.2% of pixels (270x600 grey, status bar excluded, >24 grey levels).

Prints, with 95% bootstrap intervals over EPISODES (1000 resamples, seed 0):
  - pooled no-effect rate and per-episode median no-effect rate (A2-bis threshold: median >= 35%)
  - first-encounter repeat rate: of the decisions taken on a screen the previous action left unchanged
    (the FIRST such decision in a run), the share that repeat that previous action (threshold >= 50%)
  - AUROC(action confidence -> step has a visible effect) (threshold < 0.65)
  - typing share of decisions (threshold <= 5%)
"""
import argparse
import json
import os
import random
import statistics as st


def same_action(a, b):
    if a["action"] != b["action"]:
        return False
    p, q = a.get("point"), b.get("point")
    if p and q:
        return abs(p[0] - q[0]) <= 40 and abs(p[1] - q[1]) <= 40
    return (a.get("app"), a.get("text")) == (b.get("app"), b.get("text"))


def episode_stats(e):
    steps = e["steps"]
    ex = [s for s in steps if not s["done"] and s["next_diff"] is not None]
    noeff = [s for s in ex if s["next_diff"] < 0.002]
    fe_tot = fe_rep = 0
    prev_same = False
    for i in range(len(steps) - 1):
        s, n = steps[i], steps[i + 1]
        if s["next_diff"] is None:
            prev_same = False
            continue
        same = s["next_diff"] < 0.002
        if same and not prev_same:
            fe_tot += 1
            fe_rep += same_action(s, n)
        prev_same = same and same_action(s, n)
    return {
        "task": e["task"], "n": len(ex), "noeff": len(noeff), "fe_tot": fe_tot, "fe_rep": fe_rep,
        "ce": [float(s["conf"]) for s in ex if s["next_diff"] >= 0.002 and s["conf"] is not None],
        "cn": [float(s["conf"]) for s in noeff if s["conf"] is not None],
        "decisions": len(steps), "typing": sum(1 for s in steps if s["action"] == "type_text"),
    }


def auroc(p, n):
    if not p or not n:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in p for b in n) / (len(p) * len(n))


def pooled(sample):
    n = sum(x["n"] for x in sample)
    return (sum(x["noeff"] for x in sample) / max(1, n),
            sum(x["fe_rep"] for x in sample) / max(1, sum(x["fe_tot"] for x in sample)),
            auroc([c for x in sample for c in x["ce"]], [c for x in sample for c in x["cn"]]),
            sum(x["typing"] for x in sample) / max(1, sum(x["decisions"] for x in sample)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--set", choices=["inspected", "confirmation", "all"], default="all")
    ap.add_argument("--inspected", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspected_tasks.txt"))
    a = ap.parse_args()
    inspected = set(open(a.inspected).read().split())
    eps = [e for e in json.load(open(a.export)) if not e.get("setup_error") and not e.get("unscored")]
    if a.set == "inspected":
        eps = [e for e in eps if e["task"] in inspected]
    elif a.set == "confirmation":
        eps = [e for e in eps if e["task"] not in inspected]
    per = [episode_stats(e) for e in eps if e["steps"]]
    rng = random.Random(0)
    boot = [pooled([rng.choice(per) for _ in per]) for _ in range(1000)]

    def ci(k):
        v = sorted(b[k] for b in boot if b[k] == b[k])
        return (v[25], v[974]) if len(v) >= 1000 else (float("nan"), float("nan"))

    pt = pooled(per)
    med = st.median([x["noeff"] / x["n"] for x in per if x["n"]])
    out = {"set": a.set, "episodes": len(per), "executed_steps": sum(x["n"] for x in per),
           "no_effect_pooled": round(pt[0], 4), "no_effect_pooled_ci": [round(x, 4) for x in ci(0)],
           "no_effect_episode_median": round(med, 4),
           "first_encounter_repeat": round(pt[1], 4), "first_encounter_repeat_ci": [round(x, 4) for x in ci(1)],
           "first_encounter_n": sum(x["fe_tot"] for x in per),
           "auroc_conf_effect": round(pt[2], 4), "auroc_conf_effect_ci": [round(x, 4) for x in ci(2)],
           "typing_share": round(pt[3], 4)}
    out["a2bis_pass"] = {"no_effect_median>=0.35": med >= 0.35, "first_encounter_repeat>=0.50": pt[1] >= 0.50,
                         "auroc<0.65": pt[2] < 0.65, "typing<=0.05": pt[3] <= 0.05}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
