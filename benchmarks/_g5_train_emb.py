"""G2/G3 (multimodal) from PRE-ENCODED embeddings (T4 output), train-only.

Uses the cached embeddinggemma-2 text + multimodal embeddings already produced by the
T4 kernel (S_*_t.npy / S_*_m.npy), plus the curated action labels. Trains a tiny
single-pass decision head per mode (text vs multimodal) x objective (ce/proper/joint)
and reports acc / ECE / Brier. No encoder, no 6.4GB, no GPU needed.

Usage:
  python3 benchmarks/_g5_train_emb.py
"""
import json, os, collections
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
EMB = "benchmarks/data/curated/emb"
TRAIN = "benchmarks/data/curated_actions_train.jsonl"
TEST  = "benchmarks/data/curated_actions_test.jsonl"
ACTIONS = ["click", "swipe", "open_app", "wait", "input_text", "navigate_back", "long_press"]


def load_y(path):
    rows = [json.loads(l) for l in open(path)]
    return np.array([ACTIONS.index(r["action"]) if r["action"] in ACTIONS else ACTIONS.index("click")
                     for r in rows], dtype=np.int64)


def ece_binary(conf, correct, n_bins=10):
    n = len(conf)
    if n == 0: return 0.0
    counts = [0]*n_bins; csum=[0.0]*n_bins; asum=[0.0]*n_bins
    for c, lab in zip(conf, correct):
        b = min(int(c*n_bins), n_bins-1)
        counts[b]+=1; csum[b]+=c; asum[b]+=1.0 if lab else 0.0
    tot=0.0
    for b in range(n_bins):
        if counts[b]==0: continue
        tot += (counts[b]/n)*abs(asum[b]/counts[b]-csum[b]/counts[b])
    return tot


def brier_binary(conf, correct):
    n=len(conf)
    if n==0: return 0.0
    return sum((c-(1.0 if lab else 0.0))**2 for c,lab in zip(conf,correct))/n


class Model(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.action = nn.Linear(dim, K)
        self.calib = nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,1))
    def forward(self, S):
        logits = self.action(S)
        probs = torch.softmax(logits, 1)
        t2 = torch.topk(probs,2,1).values
        margin = (t2[:,0]-t2[:,1]).unsqueeze(1)
        ci = torch.cat([probs.max(1,keepdim=True).values, margin], 1)
        p_success = torch.sigmoid(self.calib(ci)).squeeze(1)
        return logits, p_success


def run(S_tr, S_te, y_tr, y_te, objective, epochs=60, lr=2e-3, label='', seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    K = len(ACTIONS)
    model = Model(S_tr.shape[1], K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    St = torch.from_numpy(S_tr).to(DEV); yt = torch.from_numpy(y_tr).to(torch.long).to(DEV)
    Se = torch.from_numpy(S_te).to(DEV); ye = torch.from_numpy(y_te).to(torch.long).to(DEV)
    for ep in range(epochs):
        model.train()
        logits, psucc = model(St)
        correct = (logits.argmax(1) == yt).float()
        loss = ce(logits, yt)
        if objective in ("proper","joint"):
            w = 1.0
            if objective == "joint":
                p = torch.softmax(logits,1)
                mar = torch.topk(p,2,1).values[:,0]-torch.topk(p,2,1).values[:,1]
                w = 1.0+(1.0-mar.detach())
            loss = loss + 0.5*torch.mean(w*(psucc-correct)**2)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, psucc = model(Se)
        pred = logits.argmax(1)
        acc = (pred==ye).float().mean().item()
        correct = (pred==ye).cpu().numpy().astype(np.float64)
    conf = psucc.cpu().numpy() if objective in ("proper","joint") else torch.softmax(logits,1).max(1).values.cpu().numpy()
    ece = ece_binary(conf, correct); bri = brier_binary(conf, correct)
    print(f"  [{label}|{objective}] acc={acc:.4f} ECE={ece:.4f} Brier={bri:.4f} conf={conf.mean():.3f}", flush=True)
    return dict(mode=label, objective=objective, acc=round(acc,4), ece=round(ece,4), brier=round(bri,4),
                conf_mean=round(float(conf.mean()),4), conf_min=round(float(conf.min()),4), conf_max=round(float(conf.max()),4))


if __name__ == "__main__":
    S_tr_t = np.load(os.path.join(EMB,'S_tr_t.npy')); S_te_t = np.load(os.path.join(EMB,'S_te_t.npy'))
    S_tr_m = np.load(os.path.join(EMB,'S_tr_m.npy')); S_te_m = np.load(os.path.join(EMB,'S_te_m.npy'))
    y_tr = load_y(TRAIN); y_te = load_y(TEST)
    print(f"[data] text tr {S_tr_t.shape} te {S_te_t.shape} | mm tr {S_tr_m.shape} te {S_te_m.shape} | dev={DEV}", flush=True)
    results = []
    for mode, Str, Ste in [("text",S_tr_t,S_te_t), ("multimodal",S_tr_m,S_te_m)]:
        for obj in ("ce","proper","joint"):
            results.append(run(Str, Ste, y_tr, y_te, obj, label=mode))
    print("\n=== SUMMARY ===")
    import pandas as pd
    df = pd.DataFrame(results)
    os.makedirs("benchmarks/results", exist_ok=True)
    df.to_csv("benchmarks/results/2026-10-16-g2g3-text-vs-multimodal.csv", index=False)
    print(df.to_string(index=False))
