"""L1: calibrated single-pass ACTION-TYPE decision on AndroidControl-Curated (bbox-corrected).

LOCKED benchmark: state = goal + step_instruction (+ option screenshot, frozen
google/embeddinggemma-2), decision = action type over K classes, a calibrated
probability vector in one forward pass (zero output tokens).

Q2 objective comparison (G3): CE-only vs CE+Brier(p_success) (proper-scoring) vs
success-adaptive joint. Reports accuracy + ECE + Brier on a held-out episode split.

Usage:
  python3 benchmarks/_g3_train_actions.py --objective ce
  python3 benchmarks/_g3_train_actions.py --objective proper --image
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sentence_transformers import SentenceTransformer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PREFIX = "task: classification | query: "
TRAIN = "benchmarks/data/curated_actions_train.jsonl"
TEST  = "benchmarks/data/curated_actions_test.jsonl"
CLASSES = ["click", "swipe", "open_app", "wait", "input_text", "navigate_back", "long_press", "other"]
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


def encode(m, rows, use_image, max_rows=None):
    if max_rows: rows = rows[:max_rows]
    texts = [PREFIX + (r["goal"] or "") + " [step: " + (r["step_instruction"] or "") + "]" for r in rows]
    S = np.asarray(m.encode(texts, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    if use_image:
        Simg = []
        for r in rows:
            if not os.path.exists(r["image"]):
                Simg.append(np.zeros(S.shape[1], np.float32)); continue
            img = Image.open(r["image"]).convert("RGB")
            with torch.no_grad():
                e = m.encode({"text": PREFIX + (r["goal"] or ""), "image": [img]},
                             convert_to_numpy=True)
            e = np.asarray(e, np.float32)
            if e.ndim == 2 and e.shape[0] == 1: e = e[0]
            e = e / (np.linalg.norm(e) + 1e-9)
            Simg.append(e)
        S = np.concatenate([S, np.asarray(Simg, np.float32)], axis=1)
    labels = np.array([ACTIONS.index(r["action"]) if r["action"] in ACTIONS else ACTIONS.index("click")
                       for r in rows], dtype=np.int64)
    return S, labels


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
        tot += (counts[b]/n)*abs(asum[b]/counts[b] - csum[b]/counts[b])
    return tot


def brier_binary(conf, correct):
    n=len(conf)
    if n==0: return 0.0
    return sum((c-(1.0 if lab else 0.0))**2 for c,lab in zip(conf,correct))/n


class Model(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.action = nn.Linear(dim, K)
        self.calib = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, S):
        logits = self.action(S)                 # (N, K)
        probs = torch.softmax(logits, dim=1)
        top2 = torch.topk(probs, 2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
        conf_in = torch.cat([probs.max(1, keepdim=True).values, margin], dim=1)
        p_success = torch.sigmoid(self.calib(conf_in)).squeeze(1)
        return logits, p_success


def run(objective, use_image, max_rows=0, epochs=60, lr=2e-3):
    m = load()
    train = [json.loads(l) for l in open(TRAIN)]
    test  = [json.loads(l) for l in open(TEST)]
    S_tr, y_tr = encode(m, train, use_image, max_rows)
    S_te, y_te = encode(m, test, use_image, max_rows if max_rows else 0)
    K = len(ACTIONS)
    dim = S_tr.shape[1]
    model = Model(dim, K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    S_tr = torch.from_numpy(S_tr).to(DEV); y_tr = torch.from_numpy(y_tr).to(torch.long).to(DEV)
    S_te = torch.from_numpy(S_te).to(DEV); y_te = torch.from_numpy(y_te).to(torch.long).to(DEV)
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        logits, psucc = model(S_tr)
        correct = (logits.argmax(1) == y_tr).float()
        loss = ce(logits, y_tr)
        if objective in ("proper", "joint"):
            w = 1.0
            if objective == "joint":
                mar = (torch.topk(torch.softmax(logits, 1), 2, 1).values[:, 0] - torch.topk(torch.softmax(logits, 1), 2, 1).values[:, 1])
                w = 1.0 + (1.0 - mar.detach())
            loss = loss + 0.5 * torch.mean(w * (psucc - correct) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 20 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                tl, tp = model(S_tr); vl, vp = model(S_te)
                ta = (tl.argmax(1) == y_tr).float().mean().item()
                va = (vl.argmax(1) == y_te).float().mean().item()
                print(f"  ep{ep+1:03d} loss={loss.item():.4f} train_acc={ta:.3f} test_acc={va:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits, psucc = model(S_te)
        pred = logits.argmax(1)
        acc = (pred == y_te).float().mean().item()
        correct = (pred == y_te).cpu().numpy().astype(np.float64)
        conf_src = psucc.cpu().numpy() if objective in ("proper", "joint") else torch.softmax(logits,1).max(1).values.cpu().numpy()
        ece = ece_binary(conf_src, correct)
        bri = brier_binary(conf_src, correct)
        print(f"=== {objective} (image={use_image}) === acc={acc:.3f} ECE={ece:.4f} Brier={bri:.4f} n={len(y_te)}")
    return acc, ece, bri


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="ce", choices=["ce", "proper", "joint"])
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()
    run(args.objective, args.image, args.max_rows)
