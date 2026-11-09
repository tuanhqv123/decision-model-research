"""Run a decision-model e2e on RAW AndroidControl next-action selection using
an on-device LFM (Liquid) model via mlx_lm, then score through the
calibration-reliability harness (coverage-risk, ECE, Brier, cascade).

Decision framing (built from real data, nothing fabricated):
  state  = goal (high-level) + step_instruction (the current step)
  K      = candidate actions (the CORRECT ground-truth action + K-1 real
           distractors drawn from other episodes in the same sample)
  label  = index of the correct action

The model scores each candidate via LLM2Jev-style bracket-numeric decoding
from next-token logits (read directly from the mlx model's logits).

Usage:
  python3 benchmarks/run_android_decisions.py --max_rows 300 --k 4 --seed 0
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import calib_metrics as cm
from benchmarks import cascade as cg

MODEL = "mlx-community/LFM2.5-1.2B-Instruct-4bit"
DATA = "benchmarks/data/android_control_test.jsonl"


def action_str(a):
    t = (a.get("text") or "").strip()
    typ = (a.get("action_type") or "click").lower()
    if typ in ("text", "input", "type"):
        return f'TYPE text "{t}"' if t else "TYPE text"
    if typ == "scroll":
        return f"SCROLL {a.get('direction') or 'down'}"
    if typ in ("tap", "click") and t:
        return f'TAP "{t}"'
    if t:
        return f'TAP "{t}"'
    return f"TAP element ({typ})"


def build_decisions(rows, k=4, seed=0, max_rows=300):
    rng = random.Random(seed)
    # Flatten a pool of real candidate actions for distractors.
    pool = []
    for r in rows:
        for a in r.get("actions", []):
            pool.append(action_str(a))

    decisions = []
    for r in rows:
        steps = r.get("step_instructions", [])
        acts = r.get("actions", [])
        n = min(len(steps), len(acts))
        if n == 0:
            continue
        s = rng.randrange(n)
        correct = action_str(acts[s])
        distractors = rng.sample(pool, k - 1) if len(pool) >= k - 1 else pool[: k - 1]
        # Avoid colliding with the correct answer text.
        distractors = [d for d in distractors if d != correct][: k - 1]
        while len(distractors) < k - 1:
            distractors.append("DO NOTHING")
        cands = distractors + [correct]
        rng.shuffle(cands)
        label = cands.index(correct)
        state = (
            f"Goal: {r.get('goal') or ''}\n"
            f"Current instruction: {steps[s]}"
        )
        decisions.append((state, cands, label))
        if max_rows and len(decisions) >= max_rows:
            break
    return decisions


def load_rows(path):
    if not os.path.exists(path):
        print(f"!! data file missing: {path} (run prep_android_control.py first)")
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_option_logprobs(model, tok, prompt, option_ids):
    import mlx.core as mx
    import mlx.nn as nn

    ids = tok.encode(prompt)
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)  # [1, seq, vocab]
    lp = nn.log_softmax(logits[:, -1, :].astype(mx.float32))[0]  # [vocab]
    vals = [float(lp[wid]) for wid in option_ids]
    import math
    # softmax over the option set (normalize the K logprobs)
    logits_k = vals
    m = max(logits_k)
    exps = [math.exp(v - m) for v in logits_k]
    return [e / sum(exps) for e in exps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--max_rows", type=int, default=300)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.data)
    if not rows:
        return

    import mlx_lm
    print(f"Loading {args.model} ...")
    model, tok = mlx_lm.load(args.model)
    option_labels = [str(i + 1) for i in range(args.k)]
    option_ids = [tok.encode(lab, add_special_tokens=False)[0] for lab in option_labels]

    decisions = build_decisions(rows, k=args.k, seed=args.seed, max_rows=args.max_rows)
    print(f"Built {len(decisions)} decisions (k={args.k}, seed={args.seed})")

    probs, labels = [], []
    for i, (state, cands, label) in enumerate(decisions):
        prompt = (
            "You are a mobile agent choosing the next action on an Android screen.\n"
            f"{state}\n"
            "Choose the correct action:\n"
            + "\n".join(f"[{j+1}] {c}" for j, c in enumerate(cands))
            + "\nBest answer: ["
        )
        p = extract_option_logprobs(model, tok, prompt, option_ids)
        probs.append(p)
        labels.append(label)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(decisions)}")

    # ---- metrics ----
    import numpy as np
    confs = [cm.max_prob(p) for p in probs]
    preds = [cm.argmax(p) for p in probs]
    acc = [1.0 if a == b else 0.0 for a, b in zip(preds, labels)]
    ece, detail = cm.ece(probs, labels, n_bins=5)
    brier, (rel, res, unc) = cm.brier(probs, labels, k=args.k)
    nconf = [cm.n_confidence(p) for p in probs]
    curve = cm.coverage_risk(confs, acc, thresholds=[0.3, 0.5, 0.7, 0.9])

    n = len(acc)
    mean_conf = sum(confs) / n
    mean_acc = sum(acc) / n

    print("\n===== C3 BASELINE (LFM2.5 on raw AndroidControl, text) =====")
    print(f"  rows           : {n}  (k={args.k})")
    print(f"  mean accuracy  : {mean_acc:.3f}")
    print(f"  mean confidence: {mean_conf:.3f}   (max_prob)")
    print(f"  mean n_conf    : {sum(nconf)/n:.3f}")
    print(f"  ECE (n_bins=5) : {ece:.4f}")
    print(f"  Brier          : {brier:.4f} (rel={rel:.4f}, res={res:.4f}, unc={unc:.4f})")
    def fmt(x):
        return "NA" if x is None else f"{x:.3f}"
    print("  --- coverage-risk ---")
    for pt in curve:
        print(f"    thresh={pt['threshold']} cov={pt['coverage']:.3f} acc={fmt(pt['accuracy'])} risk={fmt(pt['risk'])} n_act={pt['n_act']}")

    # Aggregate cascade decision counts
    acts = [cg.gate_utility(p)[0] for p in probs]
    from collections import Counter
    print("  --- cascade distribution ---")
    for g, c in Counter(acts).most_common():
        print(f"    {g}: {c}")


if __name__ == "__main__":
    main()
