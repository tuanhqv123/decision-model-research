"""Local logic smoke for the Step 3 training pipeline (no model download).

Exercises the SAME head/logic as benchmarks/train_decision_head.py with MOCK
feature tensors (H=2048) so shape/loss bugs surface on the Mac before burning
Kaggle GPU. Validates: find_data, load_rows, LayerMix (bf16 in), the two heads
(action logit + p_succ on winning latent+margin), the three objectives
(ce / properscoring / joint), and eval ECE/Brier.

Run:  python3 benchmarks/_local_train_smoke.py
"""
import os, sys, random
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_decision_head import LayerMix, ece_binary, brier_binary, find_data, load_rows  # noqa

H = 2048   # LFM2.5 hidden size (smoke test)
K = 6      # choice-over-K candidate count


def make_head(device):
    # A) action logit: (K,H)->(K,1).   B) p_succ: (H+1)->1 (winning latent + margin).
    return nn.ModuleDict({
        "logit": nn.Linear(H, 1),
        "psucc": nn.Sequential(nn.Linear(H + 1, H // 2), nn.GELU(), nn.Linear(H // 2, 1)),
    }).to(device)


def step_loss(head, z, gold, objective):
    """Same objective logic as train_decision_head.py main()."""
    logits = head["logit"](z).squeeze(-1)            # (K,)
    probs = torch.softmax(logits, dim=-1)            # (K,)
    ce = F.cross_entropy(logits.unsqueeze(0), gold.unsqueeze(0))
    loss = ce
    if objective in ("properscoring", "joint"):
        win = int(probs.argmax())
        margin = probs.max() - probs.mean()
        feat = torch.cat([z[win], margin.unsqueeze(0)])          # (H+1,)
        ps = torch.sigmoid(head["psucc"](feat.unsqueeze(0)).squeeze())
        ps_t = (probs.argmax() == gold).float()
        brier = F.mse_loss(ps, ps_t)
        loss = ce + ((1.0 - margin).clamp(0.05, 1.0) * brier) if objective == "joint" else ce + brier
    return loss


def run_training(X, golds, objective, device, epochs=2):
    head = make_head(device)
    opt = torch.optim.AdamW(list(head.parameters()), lr=2e-4)
    y = torch.tensor(golds, device=device)
    losses = []
    for ep in range(epochs):
        idx = torch.randperm(len(X), device=device); tot = 0.0
        for i in idx:
            loss = step_loss(head, X[i], y[i], objective)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        losses.append(tot / len(idx))
    # eval (confidence = psucc head for proper/joint, else softmax max proxy)
    conf, yy = [], []
    with torch.no_grad():
        for i in range(len(X)):
            logits = head["logit"](X[i]).squeeze(-1)
            probs = torch.softmax(logits, dim=-1)
            if objective in ("properscoring", "joint"):
                win = int(probs.argmax()); margin = probs.max() - probs.mean()
                feat = torch.cat([X[i][win], margin.unsqueeze(0)])
                ci = torch.sigmoid(head["psucc"](feat.unsqueeze(0)).squeeze()).item()
            else:
                ci = probs.max().item()
            conf.append(ci); yy.append(int(probs.argmax().item() == y[i].item()))
    return losses, sum(yy) / len(yy), ece_binary(conf, yy), brier_binary(conf, yy)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # 1) find_data resolves the real AC-v2 data (no Kaggle mount on Mac)
    train_p = find_data("acv2_train.jsonl"); test_p = find_data("acv2_test.jsonl")
    print(f"find_data OK -> train={train_p} | test={test_p}")
    train = load_rows(train_p, 8); test = load_rows(test_p, 8)
    print(f"train rows={len(train)} test rows={len(test)}")
    r = train[0]
    assert "candidates" in r and "gold" in r, f"row schema bad: {list(r)}"
    assert 0 <= r["gold"] < len(r["candidates"]), "gold out of range"
    print(f"row0: K={len(r['candidates'])} | gold={r['gold']} | goal={r['goal'][:40]!r}")

    # 2) LayerMix must handle bf16 layer states + int64 attention_mask (real encoder out)
    torch.manual_seed(0)
    lmx = LayerMix(H, 4).to(device)
    layers = torch.randn(4, 1, 17, H, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, 17, device=device, dtype=torch.int64)
    zmix = lmx(layers, mask)
    assert zmix.shape == (1, H), f"LayerMix bad shape {zmix.shape}"
    assert zmix.dtype == torch.bfloat16, f"LayerMix dtype {zmix.dtype} (must stay bf16)"
    zf = zmix.float()
    logit = make_head(device)["logit"](zf)
    assert logit.shape == (1, 1), f"head logit bad shape {logit.shape}"
    print(f"LayerMix OK (bf16 in) -> {tuple(zmix.shape)} dtype={zmix.dtype}; head logit {tuple(logit.shape)}")

    # 3) all three objectives on mock features
    N, golds = 8, [random.randrange(K) for _ in range(8)]
    X = torch.randn(N, K, H, device=device)
    for obj in ["ce", "properscoring", "joint"]:
        losses, acc, ece, brier = run_training(X, golds, obj, device)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), f"{obj}: non-finite {losses}"
        assert 0.0 <= ece <= 1.0 and 0.0 <= brier <= 1.0, f"{obj}: bad metric {ece},{brier}"
        print(f"objective={obj}: losses={[round(l,4) for l in losses]} acc={acc:.3f} ECE={ece:.4f} Brier={brier:.4f}")

    print("\nLOCAL SMOKE PASS: data + LayerMix(bf16) + two heads + ce/proper/joint + ECE/Brier OK.")


if __name__ == "__main__":
    main()
