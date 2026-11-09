"""Build candidate-selection supervision from data already on disk — zero GPU, zero new annotation.

WHY. The model's grounding failure is element *selection* (71 of 143 errors; median miss 437 px, and only 4.4%
of misses land within 50 px of the right element). The a11y tree already contains the right element on 97.2% of
eval rows and on 82.0% of training target rows. So the supervision for a selection head exists in the data we
already have; nothing is being bought here.

WHAT. For every spatial training row, find which candidate (if any) contains the gold point, and write a compact
record:

    {ep, step, goal, img, W, H, n, gold_idx, gold_type, overlap_top1, gold_text}

`gold_idx` is the index into the SAME candidate list the trainer already stores, so a choice head can be trained
without rebuilding candidates. Rows whose gold point is inside no candidate are recorded with gold_idx = -1 and
are NOT silently dropped (dropping them would inflate accuracy exactly like the earlier ensemble bug).

USAGE
    python3 benchmarks/train/build_candidate_supervision.py \
      --data backup-2026-10-29/s4090-1x-100/acwork/acdata \
      --train compact_cur15k.jsonl --val compact_validation.jsonl \
      --out benchmark/results/2026-11-07-candidate-supervision

Outputs: `<out>.jsonl` (one line per spatial row), `<out>.summary.json`.
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPATIAL = {"click", "long_press", "type_text"}
TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an to in on of for and app my me i with from at by is be this that it as or open click tap".split())


def toks(s):
    return {t for t in TOK.findall((s or "").lower()) if t not in STOP and len(t) > 1}


def inside(bbox, x, y):
    return bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]


def gold_point(rec):
    """A point for every spatial type. type_text has no point in AndroidControl; use the target centre."""
    a = rec["action"]
    if a["action_type"] in ("click", "long_press"):
        return a["x"], a["y"]
    t = rec.get("target")
    if t:
        return (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2
    return None


def process(path, out_fp, stats):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["action"]["action_type"] not in SPATIAL:
            continue
        stats["spatial"] += 1
        pt = gold_point(r)
        cands = r.get("cands") or []
        if pt is None or not cands:
            stats["no_point_or_cands"] += 1
            continue
        g = toks(r.get("goal", ""))
        scores = [len(g & toks(c.get("label"))) for c in cands]
        best = max(scores) if scores else 0
        # Use the SAME label rule the trainer uses, so supervision and scoring agree by construction.
        # (R4 defect 4: a container whose centre is outside the gold node supervises a point the scorer can
        # never award. gold_candidate() now requires centre-in-node.)
        from ac_common import gold_candidate
        tgt = r.get("target")
        gc = gold_candidate(cands, tgt, pt[0], pt[1]) if tgt else None
        gold_idx = next((i for i, c in enumerate(cands)
                         if gc is not None and c["idx"] == gc["idx"]), -1)
        if gold_idx < 0:
            stats["gold_not_in_candidates"] += 1
        else:
            stats["gold_in_candidates"] += 1
            # BUG FIXED: `rank = #(score > gold)` counts a gold score of 0 as rank 0 when nothing scores
            # above it, i.e. every tie-at-zero row looked like a top-1/top-5 hit. That inflated the
            # "top-5 = 82.5%" number to a meaningless 82.5%. Use the actual top-k SET instead.
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            pos = order.index(gold_idx)
            stats["text_overlap_top1"] += int(pos == 0)
            stats["text_overlap_top5"] += int(pos < 5)
            stats["text_overlap_top10"] += int(pos < 10)
            stats["gold_has_text"] += int(bool((cands[gold_idx].get("label") or "").strip()))
        stats["n_cands"].append(len(cands))
        out_fp.write(json.dumps({
            "ep": r["ep"], "step": r["step"], "goal": r.get("goal", "")[:300],
            "img": r.get("img"), "W": r.get("W"), "H": r.get("H"),
            "n": len(cands), "gold_idx": gold_idx,
            "gold_type": r["action"]["action_type"],
            "text_overlap_best": best,
        }) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train", default="compact_cur15k.jsonl")
    ap.add_argument("--val", default="compact_validation.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    summary = {}
    for split, fname in (("train", a.train), ("val", a.val)):
        p = os.path.join(a.data, fname)
        if not os.path.exists(p):
            print(f"[skip] {split}: {p} not found")
            continue
        stats = collections.Counter()
        stats["n_cands"] = []
        with open(f"{a.out}-{split}.jsonl", "w", encoding="utf-8") as fp:
            process(p, fp, stats)
        nc = stats.pop("n_cands")
        s = dict(stats)
        s["n_cands_median"] = statistics.median(nc) if nc else 0
        s["n_cands_p90"] = sorted(nc)[int(0.9 * len(nc))] if nc else 0
        if s["spatial"]:
            s["ceiling_pct"] = round(100 * s["gold_in_candidates"] / s["spatial"], 2)
            s["text_top1_pct"] = round(100 * s["text_overlap_top1"] / max(s["gold_in_candidates"], 1), 2)
            s["gold_has_text_pct"] = round(100 * s["gold_has_text"] / max(s["gold_in_candidates"], 1), 2)
        summary[split] = s
        print(f"[{split}] {json.dumps(s)}")
    json.dump(summary, open(f"{a.out}.summary.json", "w"), indent=1)
    print(f"wrote {a.out}-{{train,val}}.jsonl + .summary.json")


if __name__ == "__main__":
    main()
