"""Temperature scaling (post-hoc calibration) for the LFM2.5 decision head.

Recomputes raw option logits for the real AndroidControl decisions, fits a
single temperature T on a calibration split (minimize NLL over the correct
option), then compares UNCALIBRATED vs CALIBRATED probabilities on a held-out
split: ECE, Brier, and the coverage-risk curve.

Runs entirely on-device (MLX, text-only LFM2.5). Optimal for this machine;
no GPU, no long training.

Usage: python3 benchmarks/run_calibration.py [--max_rows 300] [--k 4] [--seed 0]
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import calib_metrics as cm

MODEL = "mlx-community/LFM2.5-1.2B-Instruct-4bit"
DATA = "benchmarks/data/android_control_test.jsonl"


def action_str(a):
    t = (a.get("text") or "").strip()
    typ = (a.get("action_type") or "click").lower()
    if typ in ("text", "input", "type"):
        return f'TYPE text "{t}"' if t else "TYPE text"
    if typ == "scroll":
        return f"SCROLL {a.get('direction') or 'down'}"
    if t:
        return f'TAP "{t}"'
    return f"TAP element ({typ})"


def load_rows(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} (run prep_android_control.py)")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_decisions(rows, k=4, seed=0, max_rows=300):
    rng = random.Random(seed)
    pool = [action_str(a) for r in rows for a in r.get("actions", [])]
    out = []
    for r in rows:
        steps = r.get("step_instructions", [])
        acts = r.get("actions", [])
        n = min(len(steps), len(acts))
        if n == 0:
            continue
        s = rng.randrange(n)
        correct = action_str(acts[s])
        dist = [d for d in rng.sample(pool, k - 1) if d != correct][: k - 1]
        while len(dist) < k - 1:
            dist.append("DO NOTHING")
        cands = dist + [correct]
        rng.shuffle(cands)
        state = f"Goal: {r.get('goal') or ''}\nCurrent instruction: {steps[s]}"
        out.append((state, cands, cands.index(correct)))
        if max_rows and len(out) >= max_rows:
            break
    return out


def option_logits(model, tok, prompt, option_ids):
    """Return raw logits (before softmax) over the K option tokens."""
    import mlx.core as mx
    logits = model(mx.array([tok.encode(prompt)]), cache=model.make_cache())
    last = logits[0, -1, :].astype(mx.float32)
    raw = [float(last[wid]) for wid in option_ids]
    m = max(raw)
    return raw, [math.exp(v - m) for v in raw]  # logits, unnormalized probs


def softmax(vals):
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def scaled_probs(logits, T):
    return softmax([v / T for v in logits])


def fit_temperature(logits_list, labels, grid=None):
    if grid is None:
        grid = [0.5 + 0.05 * i for i in range(31)]  # 0.5..2.0
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nll = 0.0
        for logits, y in zip(logits_list, labels):
            p = scaled_probs(logits, T)
            nll += -math.log(max(p[y], 1e-12))
        nll /= len(labels)
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T, best_nll


def summarize(name, probs, labels):
    preds = [cm.argmax(p) for p in probs]
    confs = [cm.max_prob(p) for p in probs]
    acc = [1.0 if a == b else 0.0 for a, b in zip(preds, labels)]
    ece, _ = cm.ece(probs, labels, n_bins=5)
    brier, (rel, res, unc) = cm.brier(probs, labels, k=len(probs[0]))
    curve = cm.coverage_risk(confs, acc, thresholds=[0.5, 0.7, 0.9])
    print(f"\n[{name}]  acc={sum(acc)/len(acc):.3f}  ECE5={ece:.4f}  Brier={brier:.4f}")
    print(f"    conf_mean={sum(confs)/len(confs):.4f}")
    print("    coverage-risk:")
    for pt in curve:
        a = "NA" if pt["accuracy"] is None else f"{pt['accuracy']:.3f}"
        r = "NA" if pt["risk"] is None else f"{pt['risk']:.3f}"
        print(f"      t={pt['threshold']} cov={pt['coverage']:.3f} acc={a} risk={r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--max_rows", type=int, default=300)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_frac", type=float, default=0.4)
    args = ap.parse_args()

    import mlx_lm
    print("Loading", args.model, "...")
    model, tok = mlx_lm.load(args.model)
    opt_ids = [tok.encode(str(i + 1), add_special_tokens=False)[0] for i in range(args.k)]
    rows = load_rows(args.data)
    decisions = build_decisions(rows, k=args.k, seed=args.seed, max_rows=args.max_rows)
    print(f"built {len(decisions)} decisions")

    all_logits, all_labels = [], []
    for i, (state, cands, label) in enumerate(decisions):
        prompt = (
            "You are a mobile agent choosing the next action on an Android screen.\n"
            f"{state}\nChoose the correct action:\n"
            + "\n".join(f"[{j+1}] {c}" for j, c in enumerate(cands))
            + "\nBest answer: ["
        )
        raw, _ = option_logits(model, tok, prompt, opt_ids)
        all_logits.append(raw)
        all_labels.append(label)
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(decisions)}")

    # uncalibrated probs (T=1)
    probs_unc = [softmax(l) for l in all_logits]

    # calibration split
    idx = list(range(len(all_logits)))
    random.Random(1).shuffle(idx)
    cut = int(len(idx) * args.calib_frac)
    cal_idx, eval_idx = idx[:cut], idx[cut:]
    T, nll = fit_temperature([all_logits[i] for i in cal_idx], [all_labels[i] for i in cal_idx])
    print(f"\nFitted temperature T = {T:.2f}  (NLL on calibration split = {nll:.4f})")

    # calibrated probs on eval split
    probs_cal = [scaled_probs(all_logits[i], T) for i in eval_idx]
    labels_eval = [all_labels[i] for i in eval_idx]

    print("\n===== POST-HOC CALIBRATION (temperature scaling) =====")
    print(f"  calibration split: {len(cal_idx)}  eval split: {len(eval_idx)}")
    # Uncalibrated on the SAME eval split for a fair comparison
    probs_unc_eval = [softmax(all_logits[i]) for i in eval_idx]
    summarize("UNCALIBRATED (T=1.0)", probs_unc_eval, labels_eval)
    summarize("CALIBRATED   (T=%.2f)" % T, probs_cal, labels_eval)


if __name__ == "__main__":
    main()
