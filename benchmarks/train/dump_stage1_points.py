"""Step 0 for the two-stage design: dump stage-1 predictions for the TRAINING rows.

WHY THIS EXISTS. The two-stage selector must be trained on shortlists built the same way they are built at
inference — filtered by **stage 1's own predicted point**. Training rows do not have that point, only the gold
one, and filtering by gold would produce lists that inference never sees (the R4 train/infer skew class). So we
run the frozen E2 checkpoint once over every training row and save its predicted point.

Output: one JSON line per row: {ep, step, pred, pt, conf, box_conf, correct}

    python3 benchmarks/train/dump_stage1_points.py \
        --ckpt ~/acv2/ckpt/ckpt-E2-final --data ~/acv2/acdata --train-set cur15k \
        --out ~/acv2/runs/stage1_train_points.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ac_common import FROM_OPTION, SPATIAL, build_questions, build_state  # noqa: E402
from ac_common import gold_answers  # noqa: E402


def load_train_rows(data, name, limit, seed=0):
    import random
    path = os.path.join(data, f"compact_{name}.jsonl")
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    if limit:
        random.Random(seed).shuffle(recs)
        recs = recs[:limit]
    return recs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-set", default="cur15k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModel
    from train_d1 import load_image

    model = AutoModel.from_pretrained(a.ckpt, trust_remote_code=True, dtype=torch.bfloat16).cuda()
    model.engine.calibration = None
    recs = load_train_rows(a.data, a.train_set, a.limit)
    print(f"[stage1] {len(recs)} rows", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n, t0 = 0, time.time()
    with open(a.out, "w", encoding="utf-8") as fp:
        for b0 in range(0, len(recs), a.batch):
            chunk = recs[b0:b0 + a.batch]
            reqs, metas = [], []
            for r in chunk:
                state = build_state(r["goal"], "V", None, None, r.get("history"))
                reqs.append((state, build_questions("V", None), [load_image(os.path.join(a.data, r["img"]))]))
                metas.append(r)
            out = model.system_one_batch(reqs)
            for r, res in zip(metas, out):
                ans = res["answers"]
                at = FROM_OPTION[ans["action"]["choice"]]
                W, H = r["W"], r["H"]
                box = ans["target_box"].get("box")
                pt = [((box["left"] + box["right"]) / 2) * W,
                      ((box["top"] + box["bottom"]) / 2) * H] if box else None
                gold = r["action"]
                correct = (at == gold["action_type"])
                if correct and at in SPATIAL and gold.get("x") is not None and pt:
                    tg = r.get("target")
                    if tg:
                        correct = (tg["x_min"] <= pt[0] <= tg["x_max"] and tg["y_min"] <= pt[1] <= tg["y_max"])
                fp.write(json.dumps({
                    "ep": r["ep"], "step": r["step"], "pred": at, "pt": pt,
                    "conf": float(ans["action"].get("confidence") or 0.0),
                    "box_conf": ans["target_box"].get("confidence"),
                    "correct": bool(correct),
                }) + "\n")
                n += 1
            if (b0 // a.batch) % 50 == 0:
                el = time.time() - t0
                print(f"  [progress] {n}/{len(recs)} {el/60:.1f}m", flush=True)
    print(f"[stage1] wrote {n} rows to {a.out} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
