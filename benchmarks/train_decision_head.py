"""Train the calibrated single-pass decision head (Step 3) on AC-v2.

Text-state path (GOAL.md: base = LFM2.5, text-only; vision = stretch).
Data: benchmarks/data/acv2_train.jsonl + acv2_test.jsonl (choice-over-K, gold idx).
Spec: notes/decision-head-step3.md.

Design (notes/quan-transcribe-lessons.md L1/L2/L3/L6/L7):
  frozen bf16 LFM2.5 encoder -> LayerMix over top layers -> light decision head.
  Two heads: (A) action = per-candidate scalar logit -> softmax over K (CE);
             (B) p_success = head on winning-candidate latent + margin (Brier).
  Single forward pass per (goal+candidate), classification, never generation.

Objectives (Q2): --objective / --objectives ce,properscoring,joint
  ce           : CE(action) only. p_success NOT trained; eval conf = softmax max.
  properscoring: CE(action) + Brier(p_success, correctness). eval conf = psucc head.
  joint        : CE(action) + success-adaptive Brier(p_success, correctness).
                 Low-margin (hard) samples weighted higher. eval conf = psucc head.

Usage:
  python3 benchmarks/train_decision_head.py --objectives ce,properscoring,joint \
      --limit 100 --limit-eval 46 --epochs 3
"""
import argparse, json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)


class LayerMix(nn.Module):
    """Learned weighted sum (L3) of top encoder layer states, then masked mean-pool."""
    def __init__(self, dim, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers) / n_layers)
    def forward(self, layer_states, mask=None):
        # layer_states dtype (e.g. bf16 from the encoder) drives weights/mask dtype
        w = torch.softmax(self.weights, dim=0).to(layer_states.dtype)
        z = torch.einsum('l,lbtd->btd', w, layer_states)
        if mask is not None:
            mask = mask.to(z.dtype)
            z = z * mask.unsqueeze(-1)
            z = z.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            z = z.mean(dim=1)
        return z


def ece_binary(conf, labels, n_bins=10, lo=0.0, hi=1.0):
    """ECE for scalar confidence vs binary correctness (0/1)."""
    n = len(conf)
    if n == 0 or n_bins <= 0:
        return 0.0
    counts = [0] * n_bins; csum = [0.0] * n_bins; asum = [0.0] * n_bins
    for c, lab in zip(conf, labels):
        b = min(int(min(max(c, lo), hi) * n_bins), n_bins - 1)
        counts[b] += 1; csum[b] += c; asum[b] += 1.0 if lab else 0.0
    total = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        cm = csum[b] / counts[b]; am = asum[b] / counts[b]
        total += (counts[b] / n) * abs(am - cm)
    return total


def brier_binary(conf, labels):
    """Brier for scalar confidence vs binary correctness: mean((conf-y)^2)."""
    n = len(conf)
    if n == 0:
        return 0.0
    return sum((c - (1.0 if lab else 0.0)) ** 2 for c, lab in zip(conf, labels)) / n


DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"


def set_seed(s, cuda=True):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s) if cuda else None


def load_rows(path, limit=0):
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def find_data(name):
    """Locate a data file. Robust: glob the dataset mount wherever Kaggle puts it."""
    import glob
    hits = glob.glob("/kaggle/input/**/" + name, recursive=True)
    if hits:
        print(f"[find_data] {name} -> {hits[0]}")
        return hits[0]
    cands = [name, os.path.join(HERE, name), os.path.join(HERE, "data", name),
             "/kaggle/working/" + name, "benchmarks/data/" + name]
    for c in cands:
        if os.path.exists(c):
            print(f"[find_data] {name} -> {c}")
            return c
    # diagnostic: dump /kaggle/input tree so we can see the real mount path
    print("[find_data] ERROR: /kaggle/input tree:")
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            print("    ", os.path.join(root, f))
    raise FileNotFoundError(f"cannot find {name}; tried dataset glob + {cands}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--objective", default=os.environ.get("TRAIN_OBJECTIVE", "properscoring"),
                    choices=["ce", "properscoring", "joint"], help="single objective")
    ap.add_argument("--objectives", default=None,
                    help="comma-separated list of objectives to run (overrides --objective)")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("TRAIN_LIMIT", 8)),
                    help="train rows (0=all)")
    ap.add_argument("--limit-eval", type=int, default=None, help="eval rows (default = --limit)")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("TRAIN_EPOCHS", 2)))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--load-4bit", action="store_true", default=False,
                    help="4-bit quantization (requires bitsandbytes). Default off: bf16 fits on T4.")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", None))
    ap.add_argument("--out", default=os.environ.get("TRAIN_OUT", "/kaggle/working"))
    ap.add_argument("--n-mix", type=int, default=4, help="top layers to LayerMix")
    ap.add_argument("--mock-encoder", action="store_true", default=False,
                    help="TEST-ONLY: skip real model load; use deterministic synthetic features (runs full main() on cpu/mps).")
    args = ap.parse_args()
    args.data = args.data or find_data("acv2_train.jsonl")
    args.eval = args.eval or find_data("acv2_test.jsonl")
    args.limit_eval = args.limit_eval if args.limit_eval is not None else args.limit
    objectives = [o.strip() for o in args.objectives.split(",")] if args.objectives else [args.objective]
    for o in objectives:
        assert o in ("ce", "properscoring", "joint"), f"bad objective: {o}"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    set_seed(args.seed)
    dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported() or torch.backends.mps.is_available()) else torch.float32
    if not args.mock_encoder:
        assert torch.cuda.is_available(), "FAIL-FAST: real training needs CUDA (Kaggle T4)"

    if args.mock_encoder:
        H = 2048            # mimic LFM2.5 hidden size; no model load for e2e test
        print(f"[mock-encoder] H={H} | objectives {objectives} (NO model load)")
        def encode(goal, step, cand, lmx):
            seed = abs(hash((goal, step, cand))) % (2 ** 31)
            g = torch.Generator(device="cpu").manual_seed(seed)
            z = torch.randn(H, generator=g, dtype=torch.float32)   # CPU, then move to device
            return z.to(device)
    else:
        # Load model ONCE (frozen encoder), reuse across all objectives.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {"torch_dtype": dtype, "token": args.hf_token}
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
        tok = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
        enc = AutoModelForCausalLM.from_pretrained(args.model, **kw).to(device)
        enc.eval()
        H = enc.config.hidden_size
        print(f"model {args.model} | hidden {H} | dtype {dtype} | objectives {objectives}")

        PROMPT = "Goal: {goal}\nStep: {step}\nCandidate action: {cand}\nIs this the best action?"

        def encode(goal, step, cand, lmx):
            text = PROMPT.format(goal=goal, step=step, cand=cand)
            e = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                o = enc(**e, output_hidden_states=True)
            hs = o.hidden_states[-args.n_mix:]
            z = lmx(torch.stack(hs), e["attention_mask"])      # (1, H) bf16
            return z[0].float().detach()                        # (H,) float32 for fp32 heads

    def build_features(rows, lmx):
        """Encode rows -> (X, mask, golds). Rows may have variable K, so pad to maxK + a mask."""
        feats, golds = [], []
        for r in rows:
            cands = r["candidates"]; g = r["gold"]
            if not (0 <= g < len(cands)):
                continue
            zs = torch.stack([encode(r["goal"], r.get("step_instruction", ""), c, lmx) for c in cands])  # (Ki, H)
            feats.append(zs); golds.append(g)
        if not feats:
            return None, None, None
        maxK = max(f.shape[0] for f in feats)
        X = torch.zeros(len(feats), maxK, H, device=device)
        M = torch.zeros(len(feats), maxK, device=device)
        for i, f in enumerate(feats):
            X[i, :f.shape[0]] = f
            M[i, :f.shape[0]] = 1.0
        return X, M, torch.tensor(golds, device=device)

    train = load_rows(args.data, args.limit)
    test = load_rows(args.eval, args.limit_eval)
    print(f"train rows {len(train)} | eval rows {len(test)}")

    for obj in objectives:
        print(f"\n===== objective: {obj} =====")
        lmx = LayerMix(H, args.n_mix).to(device)
        head = nn.ModuleDict({
            "logit": nn.Linear(H, 1),
            "psucc": nn.Sequential(nn.Linear(H + 1, H // 2), nn.GELU(), nn.Linear(H // 2, 1)),
        }).to(device)
        opt = torch.optim.AdamW(list(head.parameters()) + list(lmx.parameters()), lr=args.lr)

        def psucc_from(z, valid=None):
            """Given (K,H) latents: p_success scalar, margin, action probs (masked if valid given)."""
            logits = head["logit"](z).squeeze(-1)          # (K,)
            if valid is not None:
                logits = logits.masked_fill(~valid, float("-inf"))
            probs = torch.softmax(logits, dim=-1)          # (K,)
            win = int(probs.argmax())
            margin = probs.max() - probs.mean()            # scalar
            feat = torch.cat([z[win], margin.unsqueeze(0)])  # (H+1,)
            ps = torch.sigmoid(head["psucc"](feat.unsqueeze(0)).squeeze())
            return ps, margin, probs

        t0 = time.time()
        X, M, y = build_features(train, lmx)
        print(f"encoded {tuple(X.shape)} in {time.time() - t0:.0f}s")
        N, K, _ = X.shape
        for ep in range(args.epochs):
            idx = torch.randperm(N, device=device); tot = 0.0
            for i in idx:
                z = X[i]; valid = M[i].bool()                # (K,)
                logits = head["logit"](z).squeeze(-1)        # (K,)
                logits = logits.masked_fill(~valid, float("-inf"))  # ignore padded candidates
                probs = torch.softmax(logits, dim=-1)        # (K,)
                ce = F.cross_entropy(logits.unsqueeze(0), y[i].unsqueeze(0))
                loss = ce
                if obj in ("properscoring", "joint"):
                    ps, margin, _ = psucc_from(z, valid)
                    ps_t = (probs.argmax() == y[i]).float()
                    brier = F.mse_loss(ps, ps_t)
                    if obj == "joint":
                        alpha = (1.0 - margin).clamp(0.05, 1.0)
                        loss = ce + alpha * brier
                    else:
                        loss = ce + brier
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            print(f"epoch {ep + 1}: loss {tot / len(idx):.4f}")

        # eval (confidence = psucc head for proper/joint, else softmax max proxy)
        conf, yy = [], []
        with torch.no_grad():
            for r in test:
                cands = r["candidates"]; g = r["gold"]
                if not (0 <= g < len(cands)):
                    continue
                zs = torch.stack([encode(r["goal"], r.get("step_instruction", ""), c, lmx) for c in cands])
                if obj in ("properscoring", "joint"):
                    ps, _, probs = psucc_from(zs)
                    conf.append(ps.item()); yy.append(int(probs.argmax().item() == g))
                else:
                    logits = head["logit"](zs).squeeze(-1)
                    p = torch.softmax(logits, dim=-1)
                    conf.append(p.max().item()); yy.append(int(p.argmax().item() == g))
        n = len(yy); acc = sum(yy) / n if n else 0.0
        ece = ece_binary(conf, yy); brier = brier_binary(conf, yy)
        print(f"| objective={obj} eval n={n} | acc={acc:.4f} | ECE={ece:.4f} | Brier={brier:.4f} |")

        os.makedirs(args.out, exist_ok=True)
        torch.save({"head": {k: v.state_dict() for k, v in head.items()}, "lmx": lmx.state_dict(),
                    "objective": obj, "model": args.model, "acc": acc, "ece": ece, "brier": brier},
                   f"{args.out}/head_{obj}.pt")
        print(f"saved -> {args.out}/head_{obj}.pt")


if __name__ == "__main__":
    main()
