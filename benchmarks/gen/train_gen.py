"""Q1 generative control arm: supervised fine-tuning of LiquidAI/LFM3-VL-2B to GENERATE the action as one JSON object.

Matched with the one-pass arm R1 (train_d1.py, contract v1, track V, --no-memory) in everything but the output:
  * rows      : the same compact_<train-set>.jsonl lines, shuffled and drawn by train_d1.load_records and the SAME
                cursor/reshuffle loop with the same seed — so step k sees the same rows as R1's step k;
  * state     : ac_common.build_state(goal, "V", None, None, history) (gen_common.state_text);
  * picture   : train_d1.load_image (720 px wide);
  * recipe    : optimizer steps, micro x accum, lr, warmup, cosine schedule, AdamW8bit (torchao, weight decay 0,
                bf16 stochastic rounding), grad-norm clip 1.0, bf16 weights, gradient checkpointing, the tied
                embedding / LM head frozen (--freeze-embed), torch.manual_seed(seed);
  * loss      : token cross-entropy on the answer tokens only (target JSON + <|im_end|>), mean over those tokens.

Writes <out>/ckpt (weights, processor, the base's remote code + LICENSE, NOTICE_MODIFIED.txt, args.json,
gen_prompt.json, contract.json = {"name": "gen_v1", "tracks": ["V"]}).

Usage (smoke): python benchmarks/gen/train_gen.py --data <acdata> --out <runs>/gen_smoke --steps 30 \
                   --micro 1 --accum 16 --grad-ckpt --freeze-embed --mem-frac 0.45
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from gen_common import CONTRACT_NAME, INSTRUCTION_SHA256, target_text, write_contract  # noqa: E402
from gen_model import BASE, BASE_REV, answer_loss, build_example, load, to_device  # noqa: E402
from train_d1 import load_image, load_records  # noqa: E402  (R1's own row order and picture)


def make_example(rec, data, processor):
    """One compact row (a JSON line) -> (model inputs, labels, n_prompt)."""
    if isinstance(rec, str):
        rec = json.loads(rec)
    img = load_image(os.path.join(data, rec["img"]))
    return build_example(processor, rec["goal"], rec.get("history"), img, target_text(rec))


def lr_lambda(warmup, total):
    """train_d1's schedule, verbatim: linear warmup to 1, then cosine to 0 over `total` steps."""
    return lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))


def draw_order(n, total_draws, micro, seed):
    """The indices train_d1's loop draws, in order (cursor over range(n); on exhaustion rng.shuffle and restart).
    `rng` = random.Random(seed), as in train_d1 (track V consumes no other random numbers from it)."""
    rng = random.Random(seed)
    order = list(range(n))
    cursor, out = 0, []
    while len(out) < total_draws:
        if cursor + micro > len(order):
            rng.shuffle(order)
            cursor = 0
        out.extend(order[cursor:cursor + micro])
        cursor += micro
    return out


def freeze_embeddings(model):
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(False)
    out_w = model.get_output_embeddings().weight
    out_w.requires_grad_(False)
    return out_w is emb


def save(model, processor, out, base_dir, args):
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    # Architecture unchanged: keep the release's own config files (train_d1 does the same) and its remote code.
    for f in os.listdir(base_dir):
        src = os.path.realpath(os.path.join(base_dir, f))
        if f.startswith(".") or os.path.isdir(src) or f == "README.md":
            continue
        if f in ("config.json", "generation_config.json") or not os.path.exists(os.path.join(out, f)):
            shutil.copy(src, os.path.join(out, f))
    with open(os.path.join(out, "NOTICE_MODIFIED.txt"), "w") as fh:  # LFM Open License v1.0 §4(b)
        fh.write(f"The weights in this directory were modified: supervised fine-tuning of {BASE} (revision "
                 f"{BASE_REV}) by benchmarks/gen/train_gen.py (generative control arm, contract {CONTRACT_NAME}). "
                 f"All other files are copied unmodified from the base release. LICENSE applies.\n")
    with open(os.path.join(out, "args.json"), "w") as fh:
        json.dump(args, fh, indent=1)
    write_contract(out)  # LAST: a contract inherited from --init can never survive under new weights


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--train-set", default="cur15k_v1", help="compact_<name>.jsonl")
    ap.add_argument("--steps", type=int, default=938, help="optimizer steps (R1: 938)")
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.0, help="cap this process's share of GPU memory")
    ap.add_argument("--save-every", type=int, default=0, help="also save <out>/ckpt every N steps (0 = end only)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--freeze-embed", action="store_true", help="freeze the tied embedding / LM head (R1 does)")
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw8bit", "adamw"],
                    help="adamw8bit = R1's torchao AdamW8bit with bf16 stochastic rounding; adamw = torch (tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    if a.mem_frac and a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    os.makedirs(a.out, exist_ok=True)
    args = {**vars(a), "contract": CONTRACT_NAME, "instruction_sha256": INSTRUCTION_SHA256, "base_rev": BASE_REV}
    with open(os.path.join(a.out, "args.json"), "w") as fh:
        json.dump(args, fh, indent=1)
    t0 = time.time()
    dtype = getattr(torch, a.dtype)
    model, processor = load(a.init, a.device, dtype)
    if os.path.isdir(a.init):
        base_dir = a.init
    else:
        from huggingface_hub import snapshot_download
        base_dir = snapshot_download(BASE, revision=BASE_REV)
    model.config.use_cache = False
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[load] {a.init} in {time.time() - t0:.0f}s  params={n_all / 1e9:.3f}B", flush=True)
    if a.freeze_embed:
        tied = freeze_embeddings(model)
        print(f"[freeze] embeddings/LM head frozen (tied={tied})", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] trainable params {sum(p.numel() for p in params) / 1e9:.3f}B", flush=True)

    recs = load_records(a.data, a.train_set, a.limit, a.seed)
    per_step = a.micro * a.accum
    total = a.steps
    draws = draw_order(len(recs), total * per_step, a.micro, a.seed)
    print(f"[data] {len(recs)} records ({a.train_set}) | {total} steps x {per_step} = {len(draws)} draws "
          f"({len(draws) / max(1, len(recs)):.3f} epochs)", flush=True)

    if a.optim == "adamw8bit":
        from torchao.optim import AdamW8bit
        opt = AdamW8bit(params, lr=a.lr, weight_decay=0.0, bf16_stochastic_round=True)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.warmup, total))

    log = open(os.path.join(a.out, "train.log"), "a")
    agg, t_last, k = {}, time.time(), 0
    for step in range(total):
        for _ in range(a.accum):
            for _ in range(a.micro):
                ex, labels, n_prompt = make_example(recs[draws[k]], a.data, processor)
                k += 1
                ex = to_device(ex, a.device, dtype)
                loss, st = answer_loss(model, ex, labels, n_prompt)
                (loss / per_step).backward()
                st["loss"] = float(loss.detach())
                for key, v in st.items():
                    agg.setdefault(key, []).append(v)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0 or step == 0:
            dt = time.time() - t_last
            n = (10 if step else 1) * per_step
            msg = {key: round(sum(v) / len(v), 4) for key, v in sorted(agg.items())}
            msg.update(step=step + 1, total=total, lr=f"{sched.get_last_lr()[0]:.2e}", gnorm=round(float(gn), 3),
                       sps=round(n / dt, 2), eta_h=round((total - step - 1) * per_step / max(n / dt, 1e-6) / 3600, 2))
            if a.device == "cuda":
                msg["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            line = json.dumps(msg)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            agg, t_last = {}, time.time()
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 != total:
            save(model, processor, os.path.join(a.out, "ckpt"), base_dir, {**args, "saved_at_step": step + 1})
            print(f"[save] step {step + 1} -> {a.out}/ckpt", flush=True)
    model.config.use_cache = True
    save(model, processor, os.path.join(a.out, "ckpt"), base_dir, {**args, "saved_at_step": total})
    print(f"[save] step {total} -> {a.out}/ckpt", flush=True)
    print("TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
