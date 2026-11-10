"""Evaluate a generative (gen_v1) checkpoint with the SAME rows, state and scorer path as eval_d1.

Rows   : eval_d1.load_rows (`--val`: compact_<val-set>.jsonl through eval_d1.val_row; otherwise the AC-v2 398 file).
State  : gen_common.state_text = ac_common.build_state(goal, "V", None, None, previous_actions).
Answer : greedy decoding, at most --max-new-tokens, every logits processor off (gen_model.GenEngine).
Score  : run_acv2.score_row(row, action_type, value, point) with the arguments built exactly as eval_d1 builds them
         (no point for a non-spatial prediction; status -> "successful"). A parse failure is scored as action "".

Reports step accuracy, type accuracy, grounding, per-type counts, parse-failure rate (+ reasons), generated tokens,
latency, and the trivial baselines on the same rows (always-click type share, always-status step accuracy).

Usage: python benchmarks/gen/eval_gen.py --ckpt <run>/ckpt --data <acdata> --val --val-set validation_v1 --tag gen_s0
"""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from gen_common import action_type_confidence, check_contract, parse_action, scorer_args  # noqa: E402


def _q(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


def run_eval(engine, rows):
    """Generate, parse and score every row. `engine.run(goal, history, image)` -> GenEngine.run's dict."""
    from run_acv2 import score_row
    from train_d1 import load_image
    out, t_start = [], time.time()
    for r in rows:
        W, H = r["screenshot_width"], r["screenshot_height"]
        g = engine.run(r["goal"], r.get("previous_actions") or [], load_image(r["screenshot"]))
        parsed, err, flags = parse_action(g["text"])
        at, value, pt = scorer_args(parsed, W, H)
        correct, grounding = score_row(r, at, value, pt)
        out.append({"episode_id": r["episode_id"], "step": r["step"], "gold": r["action"], "pred": at,
                    "value": value, "pt": list(pt) if pt else None, "raw": g["text"], "parse_error": err,
                    "flags": flags, "conf": action_type_confidence(g["pieces"], g["probs"]) if parsed else None,
                    "tokens_generated": g["tokens_generated"], "hit_eos": g["hit_eos"],
                    "greedy_consistent": g.get("greedy_consistent"), "latency_ms": g["model_ms"],
                    "correct": bool(correct), "grounding": grounding})
    return out, summarize(out, time.time() - t_start)


def summarize(out, secs=0.0):
    n = len(out)
    types = sorted(set(o["gold"]["action_type"] for o in out))
    g = [o for o in out if o["grounding"] is not None]
    fails = [o for o in out if o["parse_error"]]
    confs = [(o["conf"], o["correct"]) for o in out if o["conf"] is not None]
    s = {"track": "V", "n": n,
         "step_acc": round(sum(o["correct"] for o in out) / n, 4),
         "type_acc": round(sum(o["pred"] == o["gold"]["action_type"] for o in out) / n, 4),
         "grounding": round(sum(1 for o in g if o["grounding"]) / max(1, len(g)), 4),
         "per_type": {t: f"{sum(o['correct'] for o in out if o['gold']['action_type'] == t)}/"
                         f"{sum(1 for o in out if o['gold']['action_type'] == t)}" for t in types},
         "parse_fail_rate": round(len(fails) / n, 4),
         "parse_errors": dict(collections.Counter(o["parse_error"] for o in fails)),
         "parse_flags": dict(collections.Counter(f for o in out for f in o["flags"])),
         "tokens_generated": {"mean": round(sum(o["tokens_generated"] for o in out) / n, 2),
                              "p50": _q([o["tokens_generated"] for o in out], .5),
                              "max": max(o["tokens_generated"] for o in out)},
         "no_eos_within_budget": sum(not o["hit_eos"] for o in out),
         "greedy_inconsistent": sum(o.get("greedy_consistent") is False for o in out),
         "latency_ms": {"mean": round(sum(o["latency_ms"] for o in out) / n, 1),
                        "p50": _q([o["latency_ms"] for o in out], .5), "p95": _q([o["latency_ms"] for o in out], .95)},
         "confidence": {"source": "product of the action_type value tokens' probabilities (raw logits, T=1)",
                        "rows_with_confidence": len(confs)},
         # Trivial baselines on the same rows (AGENTS verification gate; PRE-REG §4 kill rule).
         "always_click_type": round(sum(o["gold"]["action_type"] == "click" for o in out) / n, 4),
         "always_status": round(sum(o["gold"]["action_type"] == "status" for o in out) / n, 4),
         "eval_sec": round(secs)}
    if confs:
        from eval_d1 import ece
        s["ece_action"] = round(ece([c for c, _ in confs], [y for _, y in confs]), 4)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a gen_v1 checkpoint dir (or the base id with --zero-shot)")
    ap.add_argument("--rows", default="evalset/500_steps_filtered.json")
    ap.add_argument("--shots", default="evalset/images")
    ap.add_argument("--data", default="acdata")
    ap.add_argument("--val", action="store_true", help="score compact_<val-set>.jsonl (eval_d1 --val)")
    ap.add_argument("--val-set", default="validation_v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0 = gen_model.DEFAULT_MAX_NEW_TOKENS")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--zero-shot", action="store_true", help="allow a checkpoint without the gen_v1 contract")
    a = ap.parse_args(argv)

    import torch
    from eval_d1 import load_rows
    from gen_model import DEFAULT_MAX_NEW_TOKENS, GenEngine, load
    contract = None
    if not a.zero_shot:
        contract = check_contract(a.ckpt)["name"]
    model, processor = load(a.ckpt, a.device, torch.bfloat16 if a.device == "cuda" else torch.float32)
    model.eval()
    engine = GenEngine(model, processor, a.max_new_tokens or DEFAULT_MAX_NEW_TOKENS)
    rows = load_rows(a.rows, a.shots, a.data, a.val, a.limit, a.val_set)
    print(f"[eval] {len(rows)} rows | contract {contract} | max_new_tokens {engine.max_new_tokens}", flush=True)
    out, summ = run_eval(engine, rows)
    summ.update(tag=a.tag, contract=contract, max_new_tokens=engine.max_new_tokens, ckpt=os.path.basename(
        os.path.normpath(a.ckpt)))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"eval-{a.tag}-V{'-val' if a.val else ''}.jsonl")
    with open(path, "w") as fh:
        for o in out:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(path.replace(".jsonl", ".summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    print("SUMMARY " + json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
