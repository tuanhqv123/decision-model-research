#!/usr/bin/env python3
"""EVERY step of EVERY AndroidControl train episode, in the v2 row schema (AndroidWorld element list).

Why (2026-11-11): `compact_cur15k_v2` keeps 14,965 of the 74,714 train actions spread over 8,984 episodes (every step
of only 0.6% of them), so most decisions of a task are never taught. The raw records hold every episode whole
(13,594 train episodes with actions, one more screen than actions).

Reuses, unchanged, what built the v2 rows on 2026-11-10: `rebuild_ac_v2.v1_oracle` (action, gold target, history with
the ONE history renderer, from the ORIGINAL preprocess code) and `rebuild_ac_v2.build_episode` (AndroidWorld element
list, target element, effect labels, local goal). Images are not re-extracted: `imgs/<ep>/<step>.jpg` (720 px) already
exist for every step on the training host.

    python build_ac_full_v2.py --raw RAW --out OUT.jsonl --work WORK --workers 6 [--split train]
    python build_ac_full_v2.py --check OUT.jsonl --against compact_cur15k_v2.jsonl   (differential: must be 0)
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPLIT = {}
IMG_DIR = ""  # --eval-test: where the test screenshots are written (imgs/<ep>/<step>.jpg)
COMPARED = ("action", "history", "target", "target_el_idx", "n_steps", "goal", "W", "H")


def assemble_rows(ep, split, goal, W, H, oracle, built):
    """One v2-schema row per step: the oracle's v1 fields + the element-list fields of `build_episode`."""
    rows = []
    for t in sorted(oracle):
        b = built.get(t) or {}
        if "v2" not in b:
            continue
        o = oracle[t]
        row = {"ep": ep, "step": t, "split": split, "goal": goal, "history": o["history"],
               "img": f"imgs/{ep}/{t}.jpg", "W": W[t], "H": H[t], "action": o["action"], "target": o["target"]}
        row.update(b["v2"])
        rows.append(row)
    return rows


def _need(oracle):
    """`build_episode`'s per-step input: the executed point and the gold target, as the v1 rows carried them."""
    need = {}
    for t, o in oracle.items():
        a = o["action"]
        pt = [a["x"], a["y"]] if a.get("x") is not None and a.get("y") is not None else None
        need[t] = {"pt": pt, "tgt": o["target"]}
    return need


def do_shard(args):
    path, work, want = args
    import rebuild_ac_v2 as R
    import aw_extract as AX
    from preprocess_ac import flat_nodes, records, save_screenshot
    from tfrecord import example_pb2
    name = os.path.basename(path)
    out_path = os.path.join(work, name + ".jsonl")
    if os.path.exists(out_path + ".done"):
        return name, "skip", 0, 0
    t0, n_ep, n_rows, bad = time.time(), 0, 0, 0
    with open(out_path + ".tmp", "w") as fo:
        for raw in records(path):
            ex = example_pb2.Example()
            ex.ParseFromString(raw)
            f = ex.features.feature
            ep = int(f["episode_id"].int64_list.value[0])
            split = SPLIT.get(ep)
            if split not in want:
                continue
            goal = f["goal"].bytes_list.value[0].decode()
            shots = list(f["screenshots"].bytes_list.value)
            trees = list(f["accessibility_trees"].bytes_list.value)
            W = list(f["screenshot_widths"].int64_list.value)
            H = list(f["screenshot_heights"].int64_list.value)
            acts = [json.loads(x) for x in f["actions"].bytes_list.value]
            instrs = [x.decode() for x in f["step_instructions"].bytes_list.value]
            T = len(acts)
            if T == 0 or any(len(x) != T + 1 for x in (shots, trees, W, H)):
                bad += 1
                fo.write(json.dumps({"ep": ep, "episode_error": {"n_actions": T, "shots": len(shots),
                                                                  "trees": len(trees)}}) + "\n")
                continue
            if split == "test" and IMG_DIR:
                os.makedirs(os.path.join(IMG_DIR, "imgs", str(ep)), exist_ok=True)
                for i, png in enumerate(shots):
                    p = os.path.join(IMG_DIR, "imgs", str(ep), f"{i}.jpg")
                    if not os.path.exists(p):
                        save_screenshot(png, p)
            old = {i: flat_nodes(trees[i]) for i in range(T + 1)}
            oracle = R.v1_oracle(acts, old, W, H, set(range(T + 1)))
            forests = [AX.parse_forest(b) for b in trees]
            built = R.build_episode(ep, goal, acts, instrs, forests, W, H, _need(oracle),
                                    grey=lambda i: AX.grey_small(shots[i]), old_nodes=None)
            for row in assemble_rows(ep, split, goal, W, H, oracle, built):
                fo.write(json.dumps(row, separators=(",", ":")) + "\n")
                n_rows += 1
            n_ep += 1
    os.replace(out_path + ".tmp", out_path)
    open(out_path + ".done", "w").close()
    return name, f"episodes={n_ep} rows={n_rows} bad={bad} {time.time() - t0:.0f}s", n_rows, bad


def check_split_request(want, eval_test, out):
    """AndroidControl test episodes are built only for SCORING: alone, in eval mode, into a file named `actest`.
    Every other build must not contain them (they are never trained on)."""
    if "test" not in want:
        return
    if not eval_test:
        raise SystemExit("AndroidControl test episodes are never built for training (use --eval-test to score)")
    if want != {"test"} or "actest" not in os.path.basename(out):
        raise SystemExit("--eval-test builds the test split alone, into a file whose name contains 'actest'")


def _init(split, img_dir=""):
    global IMG_DIR
    SPLIT.update(split)
    IMG_DIR = img_dir


def cmd_build(a):
    from multiprocessing import Pool
    splits = json.load(open(os.path.join(a.raw, "splits.json")))
    split = {e: s for s, eps in splits.items() for e in eps}
    want = set(a.split.split(","))
    check_split_request(want, a.eval_test, a.out)
    os.makedirs(a.work, exist_ok=True)
    shards = sorted(p for p in glob.glob(os.path.join(a.raw, "android_control-*-of-*"))
                    if not p.endswith(".done") and os.path.exists(p + ".done"))
    print(f"[build] shards={len(shards)} splits={sorted(want)} workers={a.workers}", flush=True)
    if a.episodes:
        keep = set(json.load(open(a.episodes)))
        split = {e: s for e, s in split.items() if e in keep}
        print(f"[build] restricted to {len(split)} listed episodes", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(split, a.img_dir if a.eval_test else "")) as pool:
        for name, msg, _, _ in pool.imap_unordered(do_shard, [(p, a.work, want) for p in shards]):
            print(f"[build] {name}: {msg}", flush=True)
    rows, errors = [], 0
    for p in sorted(glob.glob(os.path.join(a.work, "*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            if "episode_error" in r:
                errors += 1
            else:
                rows.append(r)
    rows.sort(key=lambda r: (r["ep"], r["step"]))
    leak = [r["ep"] for r in rows if split.get(r["ep"]) == "test"]
    if leak and not a.eval_test:
        raise SystemExit(f"LEAK: {len(leak)} rows from test episodes")
    with open(a.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"BUILD_DONE rows={len(rows)} episodes={len({r['ep'] for r in rows})} episode_errors={errors} "
          f"leak_check=0 test rows", flush=True)


def cmd_check(a):
    built_eps, got = set(), {}
    all_rows = [r for r in map(json.loads, open(a.check)) if "episode_error" not in r]
    for r in all_rows:
        built_eps.add(r["ep"])
    # the reference rows of the episodes this build covers (a partial build checks only what it built)
    ref = {(r["ep"], r["step"]): r for r in map(json.loads, open(a.against)) if r["ep"] in built_eps}
    for r in all_rows:
        if (r["ep"], r["step"]) in ref:
            got[(r["ep"], r["step"])] = r
    missing = len(ref) - len(got)
    mism = {k: 0 for k in COMPARED}
    els = 0
    for key, r in got.items():
        for k in COMPARED:
            if r.get(k) != ref[key].get(k):
                mism[k] += 1
        if r.get("elements") != ref[key].get("elements"):
            els += 1
    print(f"CHECK rows_in_reference={len(ref)} found={len(got)} missing={missing} mismatches={mism} "
          f"element_list_mismatches={els}", flush=True)
    return 0 if missing == 0 and not any(mism.values()) and els == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", default="", help="JSON list of episode ids to build (default: all of --split)")
    ap.add_argument("--eval-test", action="store_true", help="build the TEST split for scoring only (actest file)")
    ap.add_argument("--img-dir", default="", help="with --eval-test: write imgs/<ep>/<step>.jpg under this dir")
    ap.add_argument("--check")
    ap.add_argument("--against")
    a = ap.parse_args(argv)
    if a.check:
        return cmd_check(a)
    return cmd_build(a)


if __name__ == "__main__":
    raise SystemExit(main())
