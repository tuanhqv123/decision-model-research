"""Fine-tune d1-2B (single-pass decision model) on AndroidControl train for AC-v2.

The forward is the checkpoint's OWN serving path (runner._sequences -> slots/pointer readouts, `set`
attention packing), run with gradients: the trained function is the served one. Losses:
  choice/noul slots : CE over the option token groups (max-pooled forms), as `prompt.readout` reads them
  box edges         : CE against a Gaussian-smoothed target on the 0-999 thousandths grid
  text reads        : CE on the pointer's start/end positions (index 0 = nothing)

Full fine-tune, bf16 weights + torchao AdamW8bit with bf16 stochastic rounding (pure-bf16 updates of ~1e-5
otherwise round away). Memory capped with --mem-frac (shared/prod GPUs).

Usage (smoke): python train_d1.py --data ~/acwork/acdata --out ~/acwork/runs/smoke --steps 30 --limit 2000
"""
import argparse
import glob
import importlib
import io
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ac_common import (CONTRACTS, DEFAULT_CONTRACT, AppMemory, TrajMemory, build_questions,  # noqa: E402
                       build_state, element_inputs, element_inputs_v3, gold_answers, gold_answers_v2, length_buckets, row_app_options,
                       write_contract)

# Run-wide settings that every sample must agree on. `contract` names the question set (ac_common.CONTRACTS)
# and is stamped on every saved checkpoint as contract.json, which the server requires.
RUN = {"contract": DEFAULT_CONTRACT, "app_vocab": []}


def tracks_of(track):
    """The serving tracks a `--track` training flag actually supervises."""
    return {"V": ["V"], "VA": ["VA"], "mix": ["V", "VA"]}[track]


def stamp_contract(ckpt_dir, track):
    write_contract(ckpt_dir, RUN["contract"], tracks=tracks_of(track))

BASE = "LiquidAI/d1-2B"
BASE_REV = "cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41"  # pinned: train and eval run the same remote code
W_LOSS = {"hit": 0.0, "action": 1.0, "target_el": 1.0, "scroll_dir": 0.5, "presence": 0.2, "edge": 0.25,
          "read": 0.5, "read_nothing": 0.05, "is_done": 0.5}

# The candidate cap below this discards most of the distractors the contrastive signal comes from.
# The VA arm trained at 14 and threw away 58.6% of all candidates (DATA-SPEC §8) — that result is a
# lower bound, not a verdict. Median candidate count with the production filter is 37.
MIN_CAND_CAP = 40


def check_cand_cap(track):
    """Refuse to start a candidate-track run whose cap is below the median candidate count.

    D1_MAX_CANDS=0 means 'no cap' in ac_common.build_state, which is fine. A cap between 1 and
    MIN_CAND_CAP-1 is the memory workaround that produced a false negative; fail loudly instead.
    """
    cap = int(os.environ.get("D1_MAX_CANDS", "0") or 0)
    if track in ("VA", "mix") and 0 < cap < MIN_CAND_CAP:
        raise SystemExit(
            f"D1_MAX_CANDS={cap} is below {MIN_CAND_CAP}. The median candidate count with the "
            f"production filter is 37, so a cap of {cap} discards most distractors and any result "
            f"is a lower bound, not a measurement. Set D1_MAX_CANDS>={MIN_CAND_CAP} (or 0 for no "
            f"cap) and see docs/research/DATA-SPEC-2026-11-09.md §8."
        )

def check_args(a):
    """Refuse, before any model is loaded, a run whose flags the contract does not define."""
    if a.contract not in CONTRACTS:
        raise SystemExit(f"--contract {a.contract!r} is not one of {sorted(CONTRACTS)}")
    if not CONTRACTS[a.contract].get("elements"):
        return
    if a.track != "VA":
        raise SystemExit(f"--contract {a.contract} is trained on the VA track only (--track VA), not {a.track!r}")
    if a.eval_every:
        raise SystemExit("--eval-every scores the AC-v2 398 test rows, which carry no AndroidWorld element list; "
                         f"evaluate {a.contract} with eval_d1.py --val --val-set validation_v2 --contract {a.contract}")
    if a.traj_k:
        raise SystemExit(f"--traj-k adds trajectory lines that a {a.contract} state does not have")


OBJECTIVE = "ce"          # "ce" (baseline) | "brier" (proper score) | "joint" (ce + lambda*brier)
BRIER_LAMBDA = 0.5        # only used by "joint"


def d1_modules(eng):
    pkg = sys.modules[type(eng).__module__].__package__
    m = {n: importlib.import_module(f"{pkg}.{n}") for n in ("prompt", "slots", "attention", "pointer", "media", "box")}
    return m


def load_model(path, device="cuda"):
    from transformers import AutoModel
    rev = None if os.path.isdir(path) else BASE_REV
    model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, revision=rev).to(device)
    model.fuse_projections = lambda: None  # serving fuses linears; training keeps them separate (lfm3.py)
    eng = model.engine
    del model.fuse_projections
    eng.calibration = None  # raw readouts; temperatures are refit after training
    return model, eng


# --------------------------------------------------------------------------------------------- data
def load_records(data, split, limit=0, seed=0):
    """compact_<split>.jsonl lines (compact_ac.py), kept as strings and parsed per sample: RAM stays small."""
    recs = []
    for name in split.split(","):  # e.g. "cur15k,aux7k": decisions + auxiliary grounding samples
        recs += open(f"{data}/compact_{name}.jsonl").read().splitlines()
    random.Random(seed).shuffle(recs)
    return recs[:limit] if limit else recs


def build_memory(data):
    items = []
    for line in open(f"{data}/compact_train.jsonl"):
        if '"open_app"' in line:
            r = json.loads(line)
            if r["action"]["action_type"] == "open_app":
                items.append((r["ep"], r["goal"], r["action"]["app_name"]))
    return AppMemory(items)


TRAJ = {"mem": None, "k": 0}  # procedural memory (--traj-k), set in main / by eval


def build_traj_memory(data):
    """Every TRAIN episode's goal + full step-instruction list (its status row carries the whole history)."""
    items = []
    for line in open(f"{data}/compact_train.jsonl"):
        if '"status"' in line:
            r = json.loads(line)
            if r["action"]["action_type"] == "status":
                items.append((r["ep"], r["goal"], r["history"]))
    return TrajMemory(items)


AUX_STATE = {"aux_instr": "Step to perform: {}", "aux_label": "Find the element: {}"}


def make_sample(rec, data, track, memory, rng):
    if isinstance(rec, str):
        rec = json.loads(rec)
    spec = CONTRACTS[RUN["contract"]]
    if spec.get("elements"):
        return make_sample_elements(rec, data, track, spec)
    if "kind" in rec:  # auxiliary grounding sample (make_aux.py): same box question as the decision's target_box
        t, W, H = rec["target"], rec["W"], rec["H"]
        qs = {"target_box": build_questions("V", contract=RUN["contract"])["target_box"]}
        gold = {"target_box": (t["y_min"] / H, t["y_max"] / H, t["x_min"] / W, t["x_max"] / W)}
        return AUX_STATE[rec["kind"]].format(rec["text"]), qs, gold, load_image(f"{data}/{rec['img']}")
    if track == "mix":
        track = "VA" if rng.random() < 0.5 else "V"
    cands = rec.get("cands") if track == "VA" else None
    if track == "VA" and not cands:  # empty a11y list: a choice needs options -> the V form of this row
        track, cands = "V", None
    apps = memory.query(rec["goal"], exclude_ep=rec["ep"]) if memory else None
    trajs = TRAJ["mem"].query_trajs(rec["goal"], TRAJ["k"], exclude_ep=rec["ep"]) if TRAJ["k"] else None
    # Tell build_state which candidate must survive any length cap, BEFORE the state is rendered.
    from ac_common import GOLD_KEEP, gold_candidate
    GOLD_KEEP["idx"] = (gold_candidate(cands, rec.get("target"), rec["action"].get("x"), rec["action"].get("y")) or {}).get("idx") if cands else None
    from ac_common import app_intent_for, ALLOW_APP_INTENT
    state = build_state(rec["goal"], track, cands, apps, rec.get("history"), trajs,
                        app_intent=app_intent_for(rec["goal"]) if ALLOW_APP_INTENT else None)
    qs = build_questions(track, cands, contract=RUN["contract"])
    gold = gold_answers(rec, track, cands, state)
    img = load_image(f"{data}/{rec['img']}")
    return state, qs, gold, img


def make_sample_elements(rec, data, track, spec):
    """Contract v2 (R2): state, questions and gold from the row's AndroidWorld element list (`rec["elements"]`,
    compact_*_v2.jsonl), through `ac_common.element_inputs` / `build_questions` — the functions `d1_api` calls
    on the harness's elements. No app memory, trajectory or app-intent line: the served v2 state has none, so
    `memory` is not an argument here. Anything v2 does not define is refused rather than approximated."""
    if "kind" in rec:
        raise ValueError("an auxiliary box sample has no question under an element contract (v2 has no target_box)")
    if track != "VA":
        raise ValueError(f"contract {RUN['contract']!r} is trained on the VA track only, not {track!r}")
    if "elements" not in rec:
        raise ValueError(f"contract {RUN['contract']!r} needs rows with an `elements` list (compact_*_v2.jsonl)")
    if spec.get("history") == "v3":
        if "history_v3" not in rec:
            raise ValueError(f"contract {RUN['contract']!r} needs rows with `history_v3` (add_history_v3.py)")
        state, kept, keep_idx = element_inputs_v3(rec["goal"], rec["history_v3"], rec["elements"], rec["W"],
                                                  rec["H"], spec["element_rule"])
    else:
        state, kept, keep_idx = element_inputs(rec["goal"], rec.get("history"), rec["elements"], rec["W"], rec["H"],
                                               spec["element_rule"])
    apps = row_app_options(rec, RUN["app_vocab"]) if spec.get("app_choice") else None
    qs = build_questions("VA", kept, contract=RUN["contract"], apps=apps)
    gold, _ = gold_answers_v2(rec, kept, keep_idx, state, apps=apps, contract=RUN["contract"])
    return state, qs, gold, load_image(f"{data}/{rec['img']}")


def app_vocab(recs, aw_apps_path):
    """Distractor pool for app choice: every gold app name of the training rows + AndroidWorld's app names."""
    names = set()
    for line in recs:
        if '"open_app"' in line:
            a = json.loads(line).get("action") or {}
            if a.get("action_type") == "open_app" and a.get("app_name"):
                names.add(a["app_name"].strip())
    if os.path.exists(aw_apps_path):
        names |= {x["name"] for x in json.load(open(aw_apps_path))}
    return sorted(names, key=lambda n: (n.lower(), n))  # case ties in a fixed order, not hash order


def load_image(path, width=None):
    """Train and eval share this: RGB, resized to width D1_IMG_W (default 720; 858 = d1's own 1.64MP cap for a
    1080x2400 screen), bicubic, JPEG q90 round-trip."""
    from PIL import Image
    width = width or int(os.environ.get("D1_IMG_W", "720"))
    im = Image.open(path).convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return im


def center_in_range_logprob(lp_a, lp_b, lo, hi):
    """log P(lo <= (A+B)/2 <= hi) for two independent edges on the 0.000-0.999 grid (log-probs lp_a, lp_b over n
    bins); a center (i+j)/2 bins = (i+j)/(2n) of the picture. Exact, in log space (logsumexp over the i+j band),
    so the gradient survives even when the box is far from the element (P ~ 1e-30): the decision model's box
    readout is a distribution, so the expected hit of its click needs no sampling."""
    n = lp_a.shape[0]
    k0 = max(0, math.ceil(lo * 2 * n - 1e-9))
    k1 = min(2 * n - 2, math.floor(hi * 2 * n + 1e-9))
    idx = torch.arange(n, device=lp_a.device)
    band = (idx[:, None] + idx[None, :] - k0).clamp_min(0) <= (k1 - k0)
    band &= (idx[:, None] + idx[None, :]) >= k0
    if k1 < k0 or not band.any():
        return lp_a.sum() * 0 - 60.0
    return torch.logsumexp((lp_a[:, None] + lp_b[None, :]).masked_fill(~band, float("-inf")).view(-1), 0)


# --------------------------------------------------------------------------------------------- forward
def forward(model, eng, M, samples, backward_scale=None):
    """One forward over a batch of (state, questions, gold, img). Returns (loss, stats).

    With `backward_scale`, also runs (loss * scale).backward() INSIDE the state_attention context: under
    gradient checkpointing the layers are recomputed during backward, and they must see the same `set`
    attention plan as the forward (outside the context they would silently recompute with another mask)."""
    P, PT, A, S = M["prompt"], M["pointer"], M["attention"], M["slots"]
    tok = eng.tokenizer
    requests, names, golds = [], [], []
    for state, qs, gold, img in samples:
        n = list(qs)
        requests.append((state, [P.as_question(qs[k]) for k in n], M["media"].media_list([img])))
        names.append(n)
        golds.append(gold)
    position_maps = {}
    seqs = eng._sequences(requests, position_maps=position_maps)
    batch = list(range(len(seqs)))
    slots = []  # (row, (req, qi), slot question, end, slot index within its question)
    for r, b in enumerate(batch):
        i, first = seqs[b][:2]
        for k, ends in enumerate(seqs[b][5]):
            key = (i, first + k)
            q = requests[i][1][first + k]
            slots.extend((r, key, sq, end, s) for s, (sq, end) in enumerate(zip(P.question_slots(q, tok), ends)))
    reads = [j for j, s in enumerate(slots) if isinstance(s[2], P.Position)]
    pointer = None
    if reads:
        masks = [torch.tensor(position_maps[batch[slots[j][0]]].pointer_mask) for j in reads]
        pointer = PT.query_plan([slots[j][0] for j in reads], masks, reads, len(slots))
    ids, lengths, media = eng._rows(seqs, batch)
    dev = ids.device
    width = ids.shape[1]
    mask = (torch.arange(width, device=dev) < torch.tensor(lengths, device=dev)[:, None]).long()
    state_len = torch.tensor([seqs[b][3] for b in batch])
    question = torch.tensor([seqs[b][4] + [-1] * (width - len(seqs[b][4])) for b in batch])
    with A.state_attention(model, state_len, question):
        out = A.text_backbone(model)(input_ids=ids, attention_mask=mask, use_cache=False, **eng._on_device(media))
        return _losses(model, eng, M, samples, out, slots, reads, pointer, position_maps, batch, requests, names,
                       golds, dev, backward_scale, ids.shape)


def _losses(model, eng, M, samples, out, slots, reads, pointer, position_maps, batch, requests, names, golds, dev,
            backward_scale, shape):
    P, S = M["prompt"], M["slots"]
    tok = eng.tokenizer
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    row = torch.tensor([s[0] for s in slots], device=dev)
    last = torch.tensor([s[3] for s in slots], device=dev)
    sl = hidden[row, last]
    states = hidden[pointer["rows"].to(dev), : pointer["width"]] if pointer is not None else None
    res = S.split(sl, states, pointer, S.output_head(model))
    codes = res.codes.float()
    logz = codes - torch.logsumexp(codes, -1, keepdim=True)
    code_idx = pointer["code_slots"].tolist() if pointer is not None else list(range(len(slots)))

    terms, stats = [], {}

    def add(name, w, value):
        if w:
            terms.append(w * value)
        stats.setdefault(name, []).append(float(value.detach()))

    edge_lp = {}  # sample -> [top, bottom, left, right] log-probs, for the expected-hit loss
    for ci, j in enumerate(code_idx):
        r, (i, qi), sq, _, s_idx = slots[j]
        name, gold = names[i][qi], golds[i]
        if name not in gold:
            continue
        tokens, forms, _ = P.readout_table(tok, sq)
        tt = torch.as_tensor(tokens, device=dev)
        ff = torch.as_tensor(forms, device=dev)
        v = logz[ci, tt][ff].max(-1).values
        logp = torch.log_softmax(v, -1)
        if name == "target_box":
            if s_idx == 0:
                add("presence", W_LOSS["presence"], -logp[0])
            else:
                frac = gold["target_box"][s_idx - 1]
                grid = torch.arange(logp.shape[0], device=dev, dtype=torch.float32)
                center = min(logp.shape[0] - 1, max(0.0, frac * 1000))
                tgt = torch.exp(-0.5 * ((grid - center) / 4.0) ** 2)
                tgt = tgt / tgt.sum()
                add("edge", W_LOSS["edge"], -(tgt * logp).sum())
                edge_lp.setdefault(i, [None] * 4)[s_idx - 1] = logp
                if s_idx == 1:  # top edge: track the argmax error in thousandths
                    stats.setdefault("edge_err", []).append(abs(int(logp.argmax()) - center))
        else:
            if name == "is_done":
                # noul: the two option groups are the readout's own yes/no forms, not a criteria map.
                gi = 0 if gold[name] == "yes" else 1
                labels = None
            else:
                labels = list(requests[i][1][qi].criteria.keys())
                gi = labels.index(gold[name])
            ce = -logp[gi]
            if OBJECTIVE == "ce":
                value = ce
            else:
                p = logp.exp()
                tgt = torch.zeros_like(p)
                tgt[gi] = 1.0
                brier = ((p - tgt) ** 2).sum()
                value = brier if OBJECTIVE == "brier" else ce + BRIER_LAMBDA * brier
            add(name, W_LOSS.get(name, 1.0), value)
            stats.setdefault(f"acc_{name}", []).append(float(int(logp.argmax()) == gi))

    for i, lps in edge_lp.items():  # expected step accuracy of the click: P(box center inside the gold element)
        if any(lp is None for lp in lps):
            continue
        top, bottom, left, right = golds[i]["target_box"]
        lhit = center_in_range_logprob(lps[0], lps[1], top, bottom) + center_in_range_logprob(lps[2], lps[3], left, right)
        add("hit", W_LOSS["hit"], -lhit)
        stats.setdefault("p_hit", []).append(float(lhit.detach().exp()))

    if pointer is not None:
        # res.positions is [R, 2, T]: one row per READ (a Read contributes two slots: start and end),
        # in query_plan's order. `reads` is the flat list of those slots, so read i owns reads[2i:2i+2].
        # The old code walked reads[::2] and indexed pos[k]; that pairing breaks the moment a slot that
        # is not a Read enters `reads`, and it silently unpacked the string "yes" as a span.
        pos = res.positions.reshape(-1, 2, res.positions.shape[-1])
        read_pairs = [reads[i:i + 2] for i in range(0, len(reads), 2)]
        for k, pair in enumerate(read_pairs):
            if k >= pos.shape[0] or len(pair) < 2:
                continue
            j = pair[0]
            r, (i, qi), _, _, _ = slots[j]
            name, gold = names[i][qi], golds[i]
            if name not in gold:
                continue
            gv = gold[name]
            if isinstance(gv, str):
                if gv != "nothing":
                    # "escalate": not copyable from the state, so there is no span to supervise.
                    continue
                first = last_t = 0
                w = W_LOSS["read_nothing"]
            else:
                a, b = gv
                pm = position_maps[batch[r]]
                idx = [t for t, o in enumerate(pm.offsets) if o and o[0] < b and o[1] > a]
                if not idx:
                    continue
                first, last_t = min(idx), max(idx)
                got = pm.read(first, last_t)["text"] or ""
                if got.strip().lower() != pm.text[a:b].strip().lower():
                    continue
                w = W_LOSS["read"]
            st, en = pos[k, 0], pos[k, 1]
            add("read" if w == W_LOSS["read"] else "read_nothing", w,
                F.cross_entropy(st[None], torch.tensor([first], device=st.device)) +
                F.cross_entropy(en[None], torch.tensor([last_t], device=en.device)))
    loss = (torch.stack(terms).sum() if terms else hidden.sum() * 0) / len(samples)
    stats["tokens"] = [shape[1]]
    if backward_scale is not None:
        (loss * backward_scale).backward()
    return loss.detach(), stats


# --------------------------------------------------------------------------------------------- periodic eval
def periodic_eval(model, a, memory, step, log, base_dir, best):
    """Serve + score the AC-v2 398 set in-process; keep the checkpoint with the best V-track step accuracy."""
    from eval_d1 import load_rows, run_eval
    model.eval()
    rows = load_rows(f"{a.evalset}/500_steps_filtered.json", f"{a.evalset}/images", a.data)
    score_v = None
    for track in a.eval_tracks.split(","):
        out, summ = run_eval(model, rows, track, memory, a.eval_batch, contract=RUN["contract"])
        summ["step"] = step
        line = "EVAL " + json.dumps(summ)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        with open(f"{a.out}/eval-step{step}-{track}.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        if track == "V":
            score_v = summ["step_acc"]
    model.train()
    torch.cuda.empty_cache()
    if score_v is not None and score_v > best and a.save_every:
        save(model, f"{a.out}/best", base_dir)
        print(f"[best] V step_acc {score_v:.4f} at step {step} -> {a.out}/best", flush=True)
        return score_v
    return max(best, score_v or -1.0)


# --------------------------------------------------------------------------------------------- local SGD
def sync_round(model, sync, k, log):
    """Publish this replica's weights for round k, wait for the cross-replica mean, load it in place
    (optimizer state stays local, as in DiLoCo with outer lr 1)."""
    from safetensors.torch import load_file, save_file
    os.makedirs(sync, exist_ok=True)
    t0 = time.time()
    sd = {n: p.detach().to("cpu", copy=True).contiguous() for n, p in model.named_parameters()}
    save_file(sd, f"{sync}/rep_r{k}.safetensors")
    open(f"{sync}/rep_r{k}.done", "w").close()
    print(f"[sync] round {k}: published, waiting for avg", flush=True)
    while not os.path.exists(f"{sync}/avg_r{k}.done"):
        time.sleep(5)
    avg = load_file(f"{sync}/avg_r{k}.safetensors")
    with torch.no_grad():
        drift = 0.0
        for n, p in model.named_parameters():
            new = avg[n].to(p.device, p.dtype)
            drift = max(drift, float((p.float() - new.float()).norm() / (new.float().norm() + 1e-12)))
            p.copy_(new)
    msg = f"[sync] round {k}: loaded avg (max rel drift {drift:.2e}) in {time.time()-t0:.0f}s"
    print(msg, flush=True)
    log.write(json.dumps({"sync_round": k, "max_rel_drift": drift, "sec": round(time.time() - t0)}) + "\n")
    os.remove(f"{sync}/rep_r{k}.safetensors")


# --------------------------------------------------------------------------------------------- main
def save(model, out, base_dir):
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    if RUN.get("app_vocab"):  # app-choice contracts: the distractor pool travels with the weights (eval reads it)
        with open(os.path.join(out, "app_vocab.json"), "w") as fh:
            json.dump(RUN["app_vocab"], fh)
    os.remove(f"{out}/config.json")  # architecture unchanged: keep the release's own config (its auto_map/d1_* keys)
    for f in os.listdir(base_dir):  # remote code, tokenizer, processor, calibration
        src = os.path.realpath(f"{base_dir}/{f}")
        if os.path.exists(f"{out}/{f}"):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, f"{out}/{f}")
        else:
            shutil.copy(src, f"{out}/{f}")
    # Stamped LAST so a contract.json inherited from --init can never survive under the new weights.
    stamp_contract(out, RUN["track"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--track", default="mix", choices=["V", "VA", "mix"])
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps (0 = epochs)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--bucket", action="store_true",
                    help="group each optimizer step's micro*accum samples by length into micro-batches (less padding)")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: train on the i-th of n data shards (soup replicas)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.0, help="cap this process's share of GPU memory")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-ckpts", action="store_true",
                    help="save each checkpoint to its own ckpt_<step> directory (choose the best on validation)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--sync-dir", default="", help="local-SGD/DiLoCo: write rep_r<k>.safetensors every "
                    "--round-steps, wait for avg_r<k>.safetensors (made by sync_avg.sh), load it, continue")
    ap.add_argument("--round-steps", type=int, default=300)
    ap.add_argument("--traj-k", type=int, default=0, help="procedural memory: k similar TRAIN episodes in the state")
    ap.add_argument("--loss-weights", default="", help="override W_LOSS, e.g. 'read=0,read_nothing=0'")
    ap.add_argument("--contract", default=DEFAULT_CONTRACT,
                    help="question set (ac_common.CONTRACTS); stamped on the checkpoint as contract.json")
    ap.add_argument("--objective", default="ce", choices=["ce", "brier", "joint"],
                    help="readout objective: ce (released recipe) | brier (proper score) | joint (ce+lambda*brier)")
    ap.add_argument("--brier-lambda", type=float, default=0.5, help="weight of the Brier term when --objective=joint")
    ap.add_argument("--freeze-embed", action="store_true", help="freeze the (tied) embedding / LM head: the "
                    "readout rows every option, number and yes/no answer is scored on")
    ap.add_argument("--train-set", default="train", help="compact_<name>.jsonl (e.g. cur15k from curate_ac.py)")
    ap.add_argument("--eval-every", type=int, default=0, help="serve+score the AC-v2 398 set every N steps")
    ap.add_argument("--eval-tracks", default="V,VA")
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--evalset", default="evalset")
    ap.add_argument("--task-complete-weight", type=float, default=0.5,
                    help="loss weight for the is_done noul readout (default 0.5)")
    a = ap.parse_args()

    check_args(a)
    check_cand_cap(a.track)
    W_LOSS["is_done"] = float(a.task_complete_weight)
    for kv in filter(None, a.loss_weights.split(",")):
        k, v = kv.split("=")
        assert k in W_LOSS, k
        W_LOSS[k] = float(v)
    global OBJECTIVE, BRIER_LAMBDA
    OBJECTIVE, BRIER_LAMBDA = a.objective, a.brier_lambda
    print(f"[loss] weights {W_LOSS} objective={OBJECTIVE} lambda={BRIER_LAMBDA}", flush=True)
    torch.manual_seed(a.seed)
    if a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    RUN["contract"], RUN["track"] = a.contract, a.track
    print(f"[contract] {a.contract} tracks={tracks_of(a.track)}"
          + (f" element_rule={CONTRACTS[a.contract]['element_rule']}" if CONTRACTS[a.contract].get("elements")
             else ""), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    t0 = time.time()
    model, eng = load_model(a.init)
    M = d1_modules(eng)
    from huggingface_hub import snapshot_download
    base_dir = a.init if os.path.isdir(a.init) else snapshot_download(BASE, revision=BASE_REV)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    print(f"[load] {a.init} in {time.time()-t0:.0f}s  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B",
          flush=True)

    recs = load_records(a.data, a.train_set, a.limit, a.seed)
    if CONTRACTS[a.contract].get("app_choice"):
        RUN["app_vocab"] = app_vocab(recs, f"{a.data}/aw_apps.json")
        print(f"[apps] app-choice vocabulary: {len(RUN['app_vocab'])} names", flush=True)
    si, sn = map(int, a.shard.split("/"))
    recs = recs[si::sn]
    # An element contract's state has no memory line (the server never adds one), so none is built.
    memory = None if (a.no_memory or CONTRACTS[a.contract].get("elements")) else build_memory(a.data)
    if a.traj_k:
        TRAJ["mem"], TRAJ["k"] = build_traj_memory(a.data), a.traj_k
        print(f"[traj] procedural memory over {len(TRAJ['mem'].items)} train episodes, k={a.traj_k}", flush=True)
    per_step = a.micro * a.accum
    total = a.steps or int(a.epochs * len(recs) / per_step)
    print(f"[data] {len(recs)} train records (shard {a.shard}) | {total} steps x {per_step} | track={a.track} "
          f"| memory={'off' if a.no_memory else len(memory.items)}", flush=True)

    if a.freeze_embed:
        model.get_input_embeddings().weight.requires_grad_(False)
        out_w = model.get_output_embeddings().weight
        out_w.requires_grad_(False)
        print(f"[freeze] embeddings/LM head frozen (tied={out_w is model.get_input_embeddings().weight})", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] trainable params {sum(p.numel() for p in params)/1e9:.2f}B", flush=True)
    from torchao.optim import AdamW8bit
    opt = AdamW8bit(params, lr=a.lr, weight_decay=0.0, bf16_stochastic_round=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))

    best = -1.0
    rng = random.Random(a.seed)
    order = list(range(len(recs)))
    cursor, epoch = 0, 0
    log = open(f"{a.out}/train.log", "a")
    agg, t_last = {}, time.time()
    for step in range(total):
        if a.bucket:
            # The same per_step samples per optimizer step, grouped by length into micro-batches (`length_buckets`):
            # every sample keeps weight 1/per_step, only the padding inside a micro-batch shrinks.
            if cursor + per_step > len(order):
                rng.shuffle(order)
                cursor, epoch = 0, epoch + 1
            step_samples = [make_sample(recs[k], a.data, a.track, memory, rng) for k in order[cursor:cursor + per_step]]
            cursor += per_step
            micro_batches = [[step_samples[j] for j in g]
                             for g in length_buckets([len(x[0]) for x in step_samples], a.micro)]
        else:
            micro_batches = None
        for mb in range(a.accum):
            if micro_batches is not None:
                samples = micro_batches[mb]
            else:
                if cursor + a.micro > len(order):
                    rng.shuffle(order)
                    cursor, epoch = 0, epoch + 1
                samples = [make_sample(recs[k], a.data, a.track, memory, rng) for k in order[cursor:cursor + a.micro]]
                cursor += a.micro
            loss, stats = forward(model, eng, M, samples, backward_scale=1.0 / a.accum)
            stats["loss"] = [float(loss.detach())]
            for k, v in stats.items():
                agg.setdefault(k, []).extend(v)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0 or step == 0:
            dt = time.time() - t_last
            n = (10 if step else 1) * per_step
            msg = {k: round(sum(v) / len(v), 4) for k, v in sorted(agg.items())}
            msg.update(step=step + 1, total=total, epoch=epoch, lr=f"{sched.get_last_lr()[0]:.2e}",
                       gnorm=round(float(gn), 3), sps=round(n / dt, 2),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                       eta_h=round((total - step - 1) * per_step / max(n / dt, 1e-6) / 3600, 2))
            line = json.dumps(msg)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            agg, t_last = {}, time.time()
        if a.sync_dir and ((step + 1) % a.round_steps == 0 or step + 1 == total):
            sync_round(model, a.sync_dir, (step + 1 + a.round_steps - 1) // a.round_steps, log)
        if a.eval_every and ((step + 1) % a.eval_every == 0 or step + 1 == total):
            best = periodic_eval(model, a, memory, step + 1, log, base_dir, best)
        if a.save_every and ((step + 1) % a.save_every == 0 or step + 1 == total):
            # --keep-ckpts: one directory per save, so the best one can be chosen on validation afterwards
            dest = f"{a.out}/ckpt_{step + 1}" if a.keep_ckpts else f"{a.out}/ckpt"
            save(model, dest, base_dir)
            print(f"[save] step {step+1} -> {dest}", flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
