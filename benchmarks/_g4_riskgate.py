"""G4: risk-gated ACT / ASK / REFUSE on the calibrated single-pass decision head.

Locks the Q2/G3 result (proper-scoring -> calibrated p_success) and asks the G4
question: does a confidence threshold turn calibrated confidence into a
coverage vs risk tradeoff, and is it on-device feasible (Mac MPS, no GPU)?

Reads the frozen embeddinggemma-2 text-state embeddings on the LOCKED curated
split (train 6,805 / test 1,630), trains a tiny decision head with the
proper-scoring objective (calibrated p_success), then sweeps tau:
    p_success >= tau  -> ACT  (take argmax action)
    p_success <  tau  -> ASK / REFUSE (no action)
Reports, per tau: coverage (fraction acted), risk (error rate among acted),
and the risk-budget operating point. Also reports on-device feasibility knobs
(latency proxy, model size, dims).

Usage:
  python3 benchmarks/_g4_riskgate.py                 # proper (calibrated)
  python3 benchmarks/_g4_riskgate.py --objective ce   # uncalibrated baseline
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PREFIX = "task: classification | query: "
TRAIN = "benchmarks/data/curated_actions_train.jsonl"
TEST  = "benchmarks/data/curated_actions_test.jsonl"
ACTIONS = ["click", "swipe", "open_app", "wait", "input_text", "navigate_back", "long_press"]


def load():
    t0 = time.time()
    m = SentenceTransformer("google/embeddinggemma-2",
                            model_kwargs={"torch_dtype": torch.bfloat16},
                            config_kwargs={"audio_config": None})
    if DEV != "cpu":
        m.eval().to(DEV)
    print(f"[model] {time.time()-t0:.0f}s | {DEV}", flush=True)
    return m


def encode(m, rows):
    texts = [PREFIX + (r["goal"] or "") + " [step: " + (r["step_instruction"] or "") + "]" for r in rows]
    return np.asarray(m.encode(texts, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)


class Model(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.action = nn.Linear(dim, K)
        self.calib = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, S):
        logits = self.action(S)
        probs = torch.softmax(logits, dim=1)
        top2 = torch.topk(probs, 2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
        conf_in = torch.cat([probs.max(1, keepdim=True).values, margin], dim=1)
        p_success = torch.sigmoid(self.calib(conf_in)).squeeze(1)
        return logits, p_success


def risk_gate(p_success, correct, taus):
    """For each tau, coverage = fraction ACTed; risk = error rate among ACTed."""
    rows = []
    for tau in taus:
        act = p_success >= tau
        n_act = int(act.sum())
        cov = n_act / len(p_success)
        if n_act > 0:
            risk = float((~correct[act]).mean())
        else:
            risk = 0.0
        rows.append(dict(tau=tau, coverage=round(cov, 4), n_act=n_act,
                         risk=round(risk, 4), risk_gated_acc=None))
    return rows


def run(objective, epochs=60, lr=2e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = load()
    train = [json.loads(l) for l in open(TRAIN)]
    test  = [json.loads(l) for l in open(TEST)]
    print("[encode] text train/test...", flush=True)
    S_tr = encode(m, train); S_te = encode(m, test)
    y_tr = np.array([ACTIONS.index(r["action"]) if r["action"] in ACTIONS else ACTIONS.index("click") for r in train], np.int64)
    y_te = np.array([ACTIONS.index(r["action"]) if r["action"] in ACTIONS else ACTIONS.index("click") for r in test], np.int64)
    K = len(ACTIONS); dim = S_tr.shape[1]
    model = Model(dim, K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    St = torch.from_numpy(S_tr).to(DEV); yt = torch.from_numpy(y_tr).to(torch.long).to(DEV)
    Se = torch.from_numpy(S_te).to(DEV); ye = torch.from_numpy(y_te).to(torch.long).to(DEV)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        logits, psucc = model(St)
        correct = (logits.argmax(1) == yt).float()
        loss = ce(logits, yt)
        if objective in ("proper", "joint"):
            w = 1.0
            if objective == "joint":
                p = torch.softmax(logits, 1)
                mar = torch.topk(p, 2, 1).values[:, 0] - torch.topk(p, 2, 1).values[:, 1]
                w = 1.0 + (1.0 - mar.detach())
            loss = loss + 0.5 * torch.mean(w * (psucc - correct) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[train] {objective} {epochs} epochs in {time.time()-t0:.0f}s", flush=True)

    model.eval()
    with torch.no_grad():
        logits, psucc = model(Se)
        pred = logits.argmax(1)
        acc = (pred == ye).float().mean().item()
        correct = (pred == ye).cpu().numpy().astype(bool)
    # calibrated vs uncalibrated confidence source
    if objective in ("proper", "joint"):
        conf = psucc.cpu().numpy()
    else:
        conf = torch.softmax(logits, 1).max(1).values.cpu().numpy()
    print(f"=== base acc (no gate) = {acc:.4f}", flush=True)
    print(f"=== calibrated p_success range [{conf.min():.3f}, {conf.max():.3f}] mean {conf.mean():.3f}", flush=True)

    taus = np.round(np.arange(0.40, 1.0001, 0.05), 3)
    rows = risk_gate(conf, correct, taus)
    print(f"\n=== RISK GATE ({objective}) ===\n tau   coverage   n_act   risk", flush=True)
    for r in rows:
        print(f" {r['tau']:.2f}   {r['coverage']:.3f}   {r['n_act']:5d}   {r['risk']:.3f}", flush=True)

    # pick operating point: highest tau still covering >85% (high-confidence ACT)
    op = [r for r in rows if r["coverage"] >= 0.85]
    pick = op[0] if op else rows[0]
    print(f"\noperating-point (max tau, coverage>=0.85): tau={pick['tau']} cov={pick['coverage']} risk={pick['risk']}", flush=True)

    # on-device feasibility knobs
    n_param = sum(p.numel() for p in model.parameters())
    print(f"\n[on-device] head params={n_param} (head only). Embedding dim={dim} (MRL trim to 256 optional). "
          f"Encoder = frozen embeddinggemma-2 440M bf16 (~0.9GB); single forward, zero output tokens.", flush=True)
    return acc, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="proper", choices=["ce", "proper", "joint"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.objective, a.epochs, seed=a.seed)
