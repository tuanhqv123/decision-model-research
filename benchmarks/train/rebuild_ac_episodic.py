#!/usr/bin/env python3
"""Rebuild the AndroidControl compact corpus so episodes keep their step sequence.

The defect this fixes: `compact_cur15k.jsonl` keeps 17% of ROWS but 66% of EPISODES — it took most
episodes and kept 1-2 steps of each (median 1 step/episode). That is what makes the trained model a
one-click agent. This script samples by EPISODE and keeps every step of an episode it takes.

Drop rules are the 311-row list in docs/research/DATA-SPEC-2026-11-09.md §6 and nothing else.

Usage:
  python3 benchmarks/train/rebuild_ac_episodic.py \
      --root <acdata> --out <dir> --rows-budget 39000 --seed 0 \
      --drop-ledger <dir>/drop_ledger.json
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ac_common import pick_candidates  # noqa: E402  the ONE production filter (spec §3)

TARGET_MISSING_ACTIONS = ("click", "type_text")


def is_dropped(rec):
    """Return a drop reason, or None to keep. Only the spec's §6 list is dropped."""
    a = rec.get("action") or {}
    at = a.get("action_type")
    t = rec.get("target")
    if t:
        x0, x1 = t.get("x_min"), t.get("x_max")
        y0, y1 = t.get("y_min"), t.get("y_max")
        if x0 is None or x1 is None or y0 is None or y1 is None:
            return "inverted_or_zero_area"
        if x0 >= x1 or y0 >= y1 or t.get("area", 0) <= 0:
            return "inverted_or_zero_area"
        W, H = rec.get("W"), rec.get("H")
        if W and H and (x0 < 0 or y0 < 0 or x1 > W or y1 > H):
            return "out_of_bounds"
    if at in TARGET_MISSING_ACTIONS and t is None:
        return f"{at}_without_target"
    return None


def load_episodes(root):
    """(split, ep) -> list of rows sorted by step. Streams shards; no sampling."""
    files = sorted(glob.glob(os.path.join(root, "records", "*.jsonl")))
    if not files:
        raise SystemExit(f"no records/*.jsonl under {root}")
    eps = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                eps.setdefault((rec.get("split"), rec["ep"]), []).append(rec)
    for k in eps:
        eps[k].sort(key=lambda r: r["step"])
    return eps


def sample_episodes(episodes, rows_budget, seed):
    """Whole-episode sampling. Never truncates an episode: skips and keeps scanning."""
    keys = list(episodes)
    random.Random(seed).shuffle(keys)
    picked, total = [], 0
    for k in keys:
        n = len(episodes[k])
        if total + n <= rows_budget:
            picked.append(k)
            total += n
    return picked


def _label_of(action):
    a = action or {}
    for k in ("app_name", "text", "direction"):
        v = a.get(k)
        if v:
            return str(v).strip()
    return ""


def build_compact(rec, prev=None, idx_map=None):
    """One output row. History is TEXT only (never screenshots).

    Entries are BARE ("click", "type_text Floor lamp"); `build_state` adds the "Step N: "
    prefix. Writing the prefix here as well produced `Step 1: Step 1: open_app IKEA` for
    159,954 of 159,954 entries (measured 2026-11-10). The `cur15k` corpus carries no prefix
    (0 of 57,096), so this was a corpus-mixing bug introduced by the episodic rebuild, and it
    meant `episodic-s0` was trained on a string no eval and no serve ever produces.
    """
    history = []
    for i, p in enumerate(prev or [], 1):
        a = p.get("action") or {}
        lab = _label_of(a)
        line = f"{a.get('action_type')}"
        if lab:
            line += f" {lab}"
        history.append(line)
    out = {
        "ep": rec["ep"],
        "step": rec["step"],
        "n_steps": rec.get("n_steps"),
        "split": rec.get("split"),
        "goal": rec.get("goal"),
        "step_instruction": rec.get("step_instruction"),
        "history": history,
        "img": rec.get("img"),
        "W": rec.get("W"),
        "H": rec.get("H"),
        "action": rec.get("action"),
        "target": rec.get("target"),
    }
    if rec.get("cands") is not None:
        out["cands"] = rec["cands"]
    elif rec.get("nodes") and rec.get("W") and rec.get("H"):
        # Convert the raw a11y tree to the production candidate list ONCE, here, so the 89k-row
        # corpus does not carry 836 nodes per row. pick_candidates covers the gold point 99.30%.
        out["cands"] = pick_candidates({"root": rec["nodes"]}, rec["W"], rec["H"])
    else:
        # ALWAYS emit the key, even with no tree to derive it from: a missing key crashes the trainer
        # at whatever step first samples the row, and that failure is not reproducible from the data.
        out["cands"] = []
    # The launcher flag is needed downstream to supervise "tap the app icon" (open_app carries no target
    # bbox in this corpus: 986/986 rows). Keep the boolean, drop the 836-node tree.
    pkgs = {(n.get("package_name") or "") for n in (rec.get("nodes") or [])}
    out["on_launcher"] = any("launcher" in p for p in pkgs)
    out.pop("nodes", None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="acdata dir containing records/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-budget", type=int, default=39000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-ledger", default="")
    ap.add_argument("--holdout-first", type=int, default=0,
                    help="seal the first N validation episodes for later use")
    a = ap.parse_args(argv)

    eps = load_episodes(a.root)
    val_keys = [k for k in eps if k[0] == "validation"]
    tr_keys = [k for k in eps if k[0] != "validation"]
    tr_eps = {k: eps[k] for k in tr_keys}
    va_eps = {k: eps[k] for k in val_keys}

    picked = sample_episodes(tr_eps, a.rows_budget, a.seed)
    os.makedirs(a.out, exist_ok=True)

    ledger = {}
    n_in = n_out = 0
    steps_per_ep = []

    def dump(keys, source, name):
        nonlocal n_in, n_out
        path = os.path.join(a.out, name)
        kept = 0
        steps_list = []
        with open(path, "w") as fh:
            for k in keys:
                rs = source[k]
                steps_list.append(len(rs))
                prev = []
                for rec in rs:
                    n_in += 1
                    reason = is_dropped(rec)
                    if reason:
                        ledger[f"{rec.get('split')}/{rec['ep']}/{rec['step']}"] = reason
                        prev.append(rec)
                        continue
                    out = build_compact(rec, prev=prev)
                    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n_out += 1
                    kept += 1
                    prev.append(rec)
        steps_per_ep.extend(steps_list)
        return kept, len(keys)

    kept_tr, n_tr = dump(picked, tr_eps, "compact_train.jsonl")
    kept_va, n_va = dump(sorted(va_eps), va_eps, "compact_validation.jsonl")

    if a.drop_ledger:
        os.makedirs(os.path.dirname(a.drop_ledger) or ".", exist_ok=True)
        with open(a.drop_ledger, "w") as fh:
            json.dump(ledger, fh, indent=1, sort_keys=True)

    med = statistics.median(steps_per_ep) if steps_per_ep else 0
    print(f"ROWS_IN={n_in} ROWS_OUT={n_out} DROPPED={n_in - n_out} "
          f"EPISODES={len(picked) + len(va_eps)} MEDIAN_STEPS={med:.0f}")
    print(f"  train: {kept_tr} rows / {n_tr} episodes   val: {kept_va} rows / {n_va} episodes")
    if med <= 1 and a.rows_budget > 2000:
        print("  WARNING: MEDIAN_STEPS<=1 — the sampler is truncating (spec §9)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
