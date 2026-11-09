"""D05 — supervised coordinator, trained properly: episode-split, leak-free features, full metrics.

WHY NOT d1-2B DIRECTLY (stated, not hidden). d1-2B's served entry point (`system_one_batch`) returns decoded
typed answers and is **not differentiable**, so a head cannot be trained through it. Training the model's own
readout for this question requires the internal forward path that `benchmarks/train/train_d1.py` uses
(`runner._sequences` + slot losses). That is the upgrade step, and it is listed as such in
coordinator/D05_STATUS.md rather than simulated here.

WHAT THIS IS. The guide's A1 ("replace the learned coordinator with a smaller classifier") is a required
ablation *and* a legitimate first supervised coordinator. It is trained on exactly the legal inputs the decision
service is allowed to see, with the split fixed by episode, and every metric the guide asks for (§47).

Run: python3 coordinator/D05_train_coordinator.py
"""
import json
import os
import re
import sys

import numpy as np

TOK = re.compile(r"[a-z0-9]+")
ACTIONS = ["click", "status", "scroll", "type_text", "wait", "open_app", "navigate_back", "long_press"]


def feats(r):
    """Leak-free features only: goal, subgoals, history length, previous action type."""
    g = set(TOK.findall((r["goal"] or "").lower()))
    a = set(TOK.findall((r["active_subgoal"] or "").lower()))
    p = set(TOK.findall((r["prev_subgoal"] or "").lower()))
    j = lambda x, y: (len(x & y) / len(x | y)) if (x | y) else 0.0
    v = [
        min(r["hist_len"], 40) / 40.0,
        j(a, g), j(a, p), j(p, g),
        1.0 if not a else 0.0,
        1.0 if not p else 0.0,
        float(len(a)), float(len(p)),
    ]
    v += [1.0 if r["action_type"] == t else 0.0 for t in ACTIONS]
    return np.array(v, dtype=np.float64)


def fit_logistic(X, y, l2=1.0, iters=3000, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * float((p - y).mean())
    return mu, sd, w, b


def predict(mu, sd, w, b, X):
    z = ((X - mu) / sd) @ w + b
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def metrics(p, y, thr=0.5):
    yhat = (p >= thr).astype(float)
    tp = float(((yhat == 1) & (y == 1)).sum()); fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum()); tn = float(((yhat == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / len(y)
    # ECE on the positive class
    ece = 0.0
    for lo in np.linspace(0, 1, 10, endpoint=False):
        m = (p > lo) & (p <= lo + 1 / 10)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return dict(acc=float(acc), precision=float(prec), recall=float(rec), f1=float(f1),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), ece=float(ece))


def main():
    P = json.load(open("coordinator/split.json"))
    out = {"features": "hist_len, goal∩active, active∩prev, prev∩goal, no_active, no_prev, len_active, len_prev, one-hot action",
           "split": "by episode (70/15/15), fixed before training"}
    for part in ("train", "dev", "test"):
        rows = P[part]
        X = np.stack([feats(r) for r in rows])
        y = np.array([1.0 if r["label"] == "CONTINUE_PLAN" else 0.0 for r in rows])
        out.setdefault("data", {})[part] = {"n": len(rows), "continue_pct": round(100 * float(y.mean()), 2)}
        if part == "train":
            mu, sd, w, b = fit_logistic(X, y)
            out["train_fit"] = metrics(predict(mu, sd, w, b, X), y)
        else:
            p = predict(mu, sd, w, b, X)
            m = metrics(p, y)
            m["trivial_always_continue_acc"] = round(100 * float(y.mean()), 2)
            m["lift_over_trivial_pts"] = round(100 * (m["acc"] - float(y.mean())), 2)
            m["auroc"] = float(((p[y == 1][:, None] > p[y == 0][None, :]).mean()
                                + 0.5 * (p[y == 1][:, None] == p[y == 0][None, :]).mean()))
            out[part] = m
    os.makedirs("coordinator", exist_ok=True)
    json.dump(out, open("coordinator/D05_metrics.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
