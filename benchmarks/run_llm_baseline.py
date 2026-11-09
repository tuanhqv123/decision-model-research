"""Run 'THE OLD' baseline: a strong LLM (Qwen3-14B Q4 via llama-server)
zero-shot on the IDENTICAL AndroidControl K-candidate selection protocol.

Protocol (must match run_android_decisions.py exactly):
  - every (step, action) pair in the test split = one decision row
  - K=4 candidates: the correct action + 3 real distractors from the same split
  - prompt ends with "Best answer: [" -> the LLM must emit digit 1..K
  - confidence = softmax of the K digit-token logprobs at the answer position
  - latency measured per request (wall ms)

Output: benchmarks/results/<stamp>-llm-baseline.jsonl + console metrics.

Usage:
  # server must already be listening (see run_llm_baseline.sh)
  python3 benchmarks/run_llm_baseline.py --base_url http://127.0.0.1:8080/v1 \
      --data benchmarks/data/android_control_test_full.jsonl --k 4 --seed 0
"""
import argparse
import json
import math
import os
import random
import time
import urllib.request

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import calib_metrics as cm
from benchmarks import run_android_decisions as rad


def build_all_step_decisions(rows, k=4, seed=0, max_rows=0):
    """Every step in every episode = one decision. Same candidate logic as C3."""
    rng = random.Random(seed)
    pool = []
    for r in rows:
        for a in r.get("actions", []):
            pool.append(rad.action_str(a))
    decisions = []
    for r in rows:
        steps = r.get("step_instructions", [])
        acts = r.get("actions", [])
        n = min(len(steps), len(acts))
        for s in range(n):
            correct = rad.action_str(acts[s])
            distractors = rng.sample(pool, k - 1) if len(pool) >= k - 1 else pool[: k - 1]
            distractors = [d for d in distractors if d != correct][: k - 1]
            while len(distractors) < k - 1:
                distractors.append("DO NOTHING")
            cands = distractors + [correct]
            rng.shuffle(cands)
            label = cands.index(correct)
            state = f"Goal: {r.get('goal') or ''}\nCurrent instruction: {steps[s]}"
            decisions.append((state, cands, label))
    if max_rows:
        decisions = decisions[:max_rows]
    return decisions


def api_post(base_url, payload):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def digit_probs_from_logprobs(result, k):
    """Map top_logprobs of the FINAL token to the K digit strings."""
    lp_list = result.get("choices", [{}])[0].get("logprobs", {}).get("tokens", [])
    if not lp_list:
        return None
    last = lp_list[-1]  # {token, logprob, top_logprobs:[{token,logprob}...]}
    top = last.get("top_logprobs", [])
    vals = {}
    for t in top:
        vals[str(t.get("token", "")).strip()] = float(t.get("logprob"))
    # also include the sampled token itself
    t = str(last.get("token", "")).strip()
    if t in [str(i) for i in range(1, k + 1)]:
        vals[t] = max(vals.get(t, -99.0), float(last.get("logprob", -99.0)))
    need = [str(i) for i in range(1, k + 1)]
    if any(d not in vals for d in need):
        return None
    logits_k = [vals[d] for d in need]
    m = max(logits_k)
    exps = [math.exp(v - m) for v in logits_k]
    z = sum(exps)
    return [e / z for e in exps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--data", default="benchmarks/data/android_control_test_full.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = rad.load_rows(args.data)
    if not rows:
        print("!! data missing; run prep_android_control_full.py first")
        return
    decisions = build_all_step_decisions(rows, k=args.k, seed=args.seed, max_rows=args.max_rows)
    print(f"Decisions: {len(decisions)} (k={args.k}, seed={args.seed})")

    out = args.out or f"benchmarks/results/llm-baseline-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    probs, labels, lat_ms = [], [], []
    with open(out, "w") as f:
        for i, (state, cands, label) in enumerate(decisions):
            prompt = (
                "You are a mobile agent choosing the next action on an Android screen.\n"
                f"{state}\n"
                "Choose the correct action:\n"
                + "\n".join(f"[{j+1}] {c}" for j, c in enumerate(cands))
                + "\nBest answer: ["
            )
            payload = {
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 32,
            }
            t0 = time.perf_counter()
            try:
                res = api_post(args.base_url, payload)
                dt = (time.perf_counter() - t0) * 1000.0
                p = digit_probs_from_logprobs(res, args.k)
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000.0
                p = None
                print(f"  !! row {i}: {type(e).__name__} {e}")
            if p is None:
                continue
            probs.append(p)
            labels.append(label)
            lat_ms.append(dt)
            f.write(json.dumps({
                "i": i, "state": state, "cands": cands, "label": label,
                "probs": p, "latency_ms": round(dt, 1),
            }) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(decisions)}  (p50 lat {sorted(lat_ms)[len(lat_ms)//2]:.0f} ms)", flush=True)

    n = len(labels)
    if n == 0:
        print("no completed rows")
        return
    confs = [cm.max_prob(p) for p in probs]
    preds = [cm.argmax(p) for p in probs]
    acc = [1.0 if a == b else 0.0 for a, b in zip(preds, labels)]
    ece, _ = cm.ece(probs, labels, n_bins=5)
    brier, _ = cm.brier(probs, labels, k=args.k)
    curve = cm.coverage_risk(confs, acc, thresholds=[0.3, 0.5, 0.7, 0.9, 0.95])
    lat_sorted = sorted(lat_ms)
    print("\n===== THE OLD: LLM zero-shot on identical protocol =====")
    print(f"  rows            : {n}")
    print(f"  M1_sel accuracy : {sum(acc)/n:.3f}   (chance {1/args.k:.3f})")
    print(f"  mean confidence : {sum(confs)/n:.3f}")
    print(f"  ECE (5 bins)    : {ece:.4f}")
    print(f"  Brier           : {brier:.4f}")
    print(f"  latency p50/p95 : {lat_sorted[n//2]:.0f} / {lat_sorted[int(n*0.95)]:.0f} ms")
    print("  --- coverage-risk ---")
    for pt in curve:
        r = pt.get("risk")
        print(f"    t={pt['threshold']:.2f} cov={pt['coverage']:.3f} acc={pt['accuracy'] if pt['accuracy'] is not None else 'NA':>6} risk={r if r is not None else 'NA':>6} n_act={pt['n_act']}")
    print(f"  rows file: {out}")


if __name__ == "__main__":
    main()