"""G2 de-risk: minimal single-pass choice-over-K decision head on EmbeddingGemma 2.

Q: Does a TRAINED head (not zero-shot) actually separate correct from wrong
   candidate actions, given the (weak) G1 zero-shot result? And does the image
   help once the head is trained?

Design (encode-once-decide, QUAN lessons):
  * google/embeddinggemma-2 (440M text+image) FROZEN.
  * Precompute state embedding S = encode(goal [+image]) once per row.
  * Precompute each candidate text embedding C_k [K, 768].
  * Head (tiny, trainable): logit_k = MLP([S; C_k; S*C_k; |S-C_k|]) -> softmax(K).
  * Train head with CE only, encoder frozen (fast, no backprop through encoder).

Also emits a p_success head (Brier-calibrated) for the Q2/G3 comparison hook.
Eval: top-1 accuracy, ECE, Brier. Compares text-only vs multimodal state.

Usage:
  python3 benchmarks/_g2_train.py --mode text [--mode multimodal]
  python3 benchmarks/_g2_train.py --mode multimodal --max-rows 60
"""
import argparse, json, os, time, random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sentence_transformers import SentenceTransformer

random.seed(0); torch.manual_seed(0)
np.random.seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PREFIX = "task: classification | query: "

TRAIN = "benchmarks/data/acv2_train2.jsonl"
TEST  = "benchmarks/data/acv2_test2.jsonl"
CACHE = "benchmarks/data/_g2_cache"

def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    return x / (np.linalg.norm(x) + 1e-9)

def load_model():
    t0 = time.time()
    m = SentenceTransformer("google/embeddinggemma-2",
                            model_kwargs={"torch_dtype": torch.bfloat16},
                            config_kwargs={"audio_config": None})
    if DEV != "cpu":
        m.eval().to(DEV)
    print(f"  [model] {time.time()-t0:.0f}s | {DEV}", flush=True)
    return m


def encode_rows(m, rows, mode, idx=None):
    """Return S (n, d) state embs, C (list of [K,d]) candidate embs, gold (n,)."""
    # state text = prefix + goal (+ step_instruction)
    state_texts = [PREFIX + (r["goal"] or "") for r in rows]
    S = np.asarray(m.encode(state_texts, convert_to_numpy=True, normalize_embeddings=True),
                   dtype=np.float32)
    cand_list = [r["candidates"] for r in rows]
    golds = np.array([r["gold"] for r in rows], dtype=np.int64)
    C = []
    for cands in cand_list:
        ce = np.asarray(m.encode([PREFIX + c for c in cands], convert_to_numpy=True,
                                 normalize_embeddings=True), dtype=np.float32)
        C.append(ce)
    if mode == "multimodal":
        S_img = []
        for r in rows:
            if not os.path.exists(r["image"]):
                S_img.append(np.zeros(S.shape[1], dtype=np.float32)); continue
            img = Image.open(r["image"]).convert("RGB")
            with torch.no_grad():
                e = m.encode({"text": PREFIX + (r["goal"] or ""), "image": [img]},
                             convert_to_numpy=True)
            S_img.append(_norm(e))
        S_img = np.asarray(S_img, dtype=np.float32)
        S = np.concatenate([S, S_img], axis=1)
    return S, C, golds


def build_head(dim_s, dim_c, hid=96, dropout=0.2):
    return nn.Sequential(
        nn.Linear(dim_s + dim_c + dim_c + dim_c, hid),
        nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hid, 1),
    )


def feats(S, C):
    """Padded per-candidate features: [S; C_k; S*C_k; |S-C_k|] for each candidate.
    Handles variable K per row via padding + mask."""
    n, d = S.shape
    Kmax = max(len(c) for c in C)
    F = np.zeros((n, Kmax, d + d + d + d), dtype=np.float32)
    mask = np.zeros((n, Kmax), dtype=np.float32)
    for i, cands in enumerate(C):
        Ki = len(cands)
        Ck = np.asarray(cands, dtype=np.float32)   # (Ki, d)
        Sk = np.tile(S[i].astype(np.float32), (Ki, 1))  # (Ki, d)
        Fi = np.concatenate([Sk, Ck, Sk * Ck, np.abs(Sk - Ck)], axis=1)  # (Ki, 4d)
        F[i, :Ki] = Fi
        mask[i, :Ki] = 1.0
    return F, mask


def train_eval(m, train, test, mode, epochs=200, lr=1e-3, max_rows=None):
    if max_rows:
        train = train[:max_rows]; test = test[:max(max_rows // 3, 8)]
    print(f"  [{mode}] rows train={len(train)} test={len(test)}", flush=True)
    S_tr, C_tr, g_tr = encode_rows(m, train, mode)
    S_te, C_te, g_te = encode_rows(m, test, mode)
    K_tr, K_te = len(C_tr[0]), len(C_te[0])
    dim_s, dim_c = S_tr.shape[1], C_tr[0].shape[1]
    head = build_head(dim_s, dim_c)
    if DEV != "cpu":
        head = head.to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    ft_tr, mk_tr = feats(S_tr, C_tr)
    ft_te, mk_te = feats(S_te, C_te)
    ft_tr = torch.from_numpy(ft_tr).to(DEV)
    mk_tr = torch.from_numpy(mk_tr).to(DEV)
    ft_te = torch.from_numpy(ft_te).to(DEV)
    mk_te = torch.from_numpy(mk_te).to(DEV)
    y_tr = torch.from_numpy(g_tr).to(torch.long).to(DEV)
    y_te = torch.from_numpy(g_te).to(torch.long).to(DEV)

    def score(feats, mask):
        # feats (n, K, 4d), mask (n, K)
        logits = head(feats).squeeze(-1)          # (n, K)
        logits = logits.masked_fill(mask <= 0, -1e9)
        return logits

    for ep in range(epochs):
        head.train()
        logits = score(ft_tr, mk_tr)
        loss = lossf(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            head.eval()
            with torch.no_grad():
                te = score(ft_te, mk_te)
                ta = score(ft_tr, mk_tr)
                at = (ta.argmax(1) == y_tr).float().mean().item()
                ae = (te.argmax(1) == y_te).float().mean().item()
                print(f"    ep{ep+1:03d} loss={loss.item():.4f} train_acc={at:.3f} test_acc={ae:.3f}", flush=True)

    head.eval()
    with torch.no_grad():
        te = score(ft_te, mk_te)
        probs = torch.softmax(te, dim=1).cpu().numpy()
        pred = probs.argmax(1)
        acc = float((pred == g_te).mean())
        conf = probs.max(1)  # max prob as confidence
        correct = (pred == g_te)
        from calib_metrics import ece_binary, brier_binary
        ece = ece_binary(conf, correct)
        brier = brier_binary(conf, correct)
        print(f"  === {mode} === test_acc={acc:.3f} ECE={ece:.4f} Brier={brier:.4f} "
              f"mean_conf={conf.mean():.3f} n={len(g_te)}", flush=True)
    return acc, ece, brier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="text", choices=["text", "multimodal"])
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()
    train = [json.loads(l) for l in open(TRAIN)]
    test  = [json.loads(l) for l in open(TEST)]
    m = load_model()
    train_eval(m, train, test, args.mode, max_rows=args.max_rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    main()
