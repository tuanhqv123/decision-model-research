#!/usr/bin/env python3
"""Zero-shot probe of two TypeSafe cookbook patterns with the CURRENT checkpoint, no training (2026-11-12).

M1  re-ranking (cookbook rerank_typesafe: "one System One question per query-candidate pair"): on validation rows whose
    gold has a target element, take the top-K elements of the trained `target_el` choice, then ask one noul per
    candidate in the same pass ("speculative fan-out"). Compare element accuracy of: the choice argmax, the noul
    argmax, and their product.
M3  progress verification on the NEXT screen: noul "did the last action move the task closer to the goal?" asked on
    (a) the screen after the demonstrator's own action (label 1, consecutive validation rows) and (b) the screen a
    WRONG action leads to (label 0, cross-episode branches built by build_recovery_data from validation rows only).
    Report AUROC.

The questions are new to the checkpoint: this measures whether the patterns carry signal before any training.

    python probe_patterns.py --ckpt CKPT --data DATA --val-set validation_v4 [--k 5] [--limit 0] --out OUT.json
"""
import argparse
import collections
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ac_common import CONTRACTS, build_questions, element_inputs_v3, gold_answers_v2, read_contract  # noqa: E402
from eval_d1 import load_rows  # noqa: E402

RERANK_Q = ("Is element [{i}] ({label}) the one the next UI action should act on to advance the goal?")
PROGRESS_Q = ("Did the last action in 'Previous actions' move the task closer to the goal (rather than to a screen "
              "that does not help, or to no change)?")


def noul_p(ans):
    """P(yes) of a noul answer, whatever key the engine uses."""
    if not isinstance(ans, dict):
        return None
    if isinstance(ans.get("noul"), (int, float)):     # d1 engine: {"type": "noul", "noul": P(yes)}
        return float(ans["noul"])
    pr = ans.get("probabilities") or {}
    for k in ("yes", "Yes", "true", "True", True, "1"):
        if k in pr:
            return float(pr[k])
    for k in ("probability", "p", "confidence"):
        if k in ans and isinstance(ans[k], (int, float)):
            v = float(ans[k])
            return v if str(ans.get("value", ans.get("answer", "yes"))).lower() in ("yes", "true", "1") else 1 - v
    return None


def auroc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, rsum, i = 1, 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (rank + rank + (j - i) - 1) / 2.0
        rsum += avg * sum(1 for x in allv[i:j] if x[1] == 1)
        rank += j - i
        i = j
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-set", default="validation_v4")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModel
    from train_d1 import load_image
    import build_recovery_data as R
    import ac_common
    R.A = ac_common
    contract = read_contract(a.ckpt)["name"]
    spec = CONTRACTS[contract]
    rule = spec["element_rule"]
    on_off = bool(spec.get("on_off"))
    model = AutoModel.from_pretrained(a.ckpt, trust_remote_code=True, dtype=torch.bfloat16).cuda()
    model.engine.calibration = None
    rows = load_rows(None, None, a.data, True, a.limit, a.val_set)
    print("[probe] contract", contract, "rows", len(rows), flush=True)

    def run(reqs):
        out = []
        for b in range(0, len(reqs), a.batch):
            with torch.no_grad():
                out += model.system_one_batch(reqs[b:b + a.batch])
        return out

    # ---- M1 -------------------------------------------------------------------------------------------------------
    m1, sample = [], None
    base_reqs, metas = [], []
    for r in rows:
        W, H = r["screenshot_width"], r["screenshot_height"]
        state, kept, idx = element_inputs_v3(r["goal"], r["history_v3"], r["elements"], W, H, rule, None, on_off=on_off)
        gold, _ = gold_answers_v2(r["rec"], kept, idx, state, apps=None, contract=contract)
        if "target_el" not in gold:
            continue
        qs = build_questions("VA", kept, contract=contract, apps=["Settings", "Clock", "Chrome"])
        qs.pop("app_name", None)   # only target_el is read here; the app list does not matter
        base_reqs.append((state, qs, [load_image(r["screenshot"])]))
        metas.append((r, kept, state, str(gold["target_el"])))
    base = run(base_reqs)
    rr_reqs = []
    for (r, kept, state, g), res, (_, _, img) in zip(metas, base, base_reqs):
        pr = res["answers"]["target_el"]["probabilities"]
        top = sorted(pr, key=lambda k: -pr[k])[:a.k]
        qs = {"cand_%s" % c: {"type": "noul", "instructions": RERANK_Q.format(i=c, label=(kept[int(c) - 1]["label"]
                                                                                        or "no label")[:60])}
              for c in top}
        rr_reqs.append((state, qs, img))
        m1.append({"gold": g, "top": top, "choice_p": {c: pr[c] for c in top}})
    rr = run(rr_reqs)
    for m, res in zip(m1, rr):
        m["noul_p"] = {c: noul_p(res["answers"].get("cand_%s" % c)) for c in m["top"]}
        if sample is None:
            sample = res["answers"]
    n = len(m1)
    in_top = sum(m["gold"] in m["top"] for m in m1)
    acc_choice = sum(m["top"][0] == m["gold"] for m in m1)
    acc_noul = sum(max(m["top"], key=lambda c: (m["noul_p"][c] or 0)) == m["gold"] for m in m1)
    acc_prod = sum(max(m["top"], key=lambda c: (m["noul_p"][c] or 0) * m["choice_p"][c]) == m["gold"] for m in m1)

    # ---- M3 -------------------------------------------------------------------------------------------------------
    by_ep = collections.defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r)
    pos_reqs = []
    for ep, rs in by_ep.items():
        rs.sort(key=lambda r: r["step"])
        for t in range(1, len(rs)):
            prev, cur = rs[t - 1], rs[t]
            if (prev["rec"].get("action") or {}).get("action_type") in ("status", "wait") or not cur["history_v3"]:
                continue
            W, H = cur["screenshot_width"], cur["screenshot_height"]
            st, _, _ = element_inputs_v3(cur["goal"], cur["history_v3"], cur["elements"], W, H, rule, None, on_off=on_off)
            pos_reqs.append((st, {"progress": {"type": "noul", "instructions": PROGRESS_Q}}, [load_image(cur["screenshot"])]))
    recs = collections.defaultdict(list)
    for r in rows:
        recs[r["rec"]["ep"]].append(r["rec"])
    for v in recs.values():
        v.sort(key=lambda x: x["step"])
    neg_rows = R.recover_back_rows(recs, rule, random.Random(0), cap=len(pos_reqs))
    shot_dir = os.path.dirname(rows[0]["screenshot"])
    neg_reqs = []
    for x in neg_rows:
        st, _, _ = element_inputs_v3(x["goal"], x["history_v3"], x["elements"], x["W"], x["H"], rule, None, on_off=on_off)
        img = os.path.join(a.data, x["img"]) if not os.path.isabs(x["img"]) else x["img"]
        neg_reqs.append((st, {"progress": {"type": "noul", "instructions": PROGRESS_Q}}, [load_image(img)]))
    pos = [noul_p(res["answers"]["progress"]) for res in run(pos_reqs)]
    neg = [noul_p(res["answers"]["progress"]) for res in run(neg_reqs)]
    out = {"contract": contract, "k": a.k,
           "m1": {"n": n, "gold_in_top_k": in_top, "acc_choice_argmax": acc_choice, "acc_noul_rerank": acc_noul,
                  "acc_choice_x_noul": acc_prod},
           "m3": {"n_pos": len(pos), "n_neg": len(neg), "auroc": auroc([p for p in pos if p is not None],
                                                                       [p for p in neg if p is not None]),
                  "mean_p_pos": sum(p or 0 for p in pos) / max(1, len(pos)),
                  "mean_p_neg": sum(p or 0 for p in neg) / max(1, len(neg))},
           "noul_answer_sample": {k: v for k, v in list((sample or {}).items())[:1]}}
    json.dump({"summary": out, "m1_rows": m1}, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
