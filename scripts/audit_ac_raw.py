#!/usr/bin/env python3
"""Stage 1 — corpus integrity of the AndroidControl raw records.

Streams all 20 shards of records/*.jsonl and reports, with NO sampling:
  rows, episodes, shard coverage, duplicate detection, step-index continuity,
  split integrity, and cross-split leakage on episodes and images.

Run: python3 scripts/audit_ac_raw.py --stage 1 [--root <acdata>]
"""
import argparse
import collections
import glob
import json
import os
import statistics as st
import sys


def default_root():
    cands = glob.glob("backup-*/prd-survey-*/acwork/acdata")
    cands += glob.glob("backup-*/GPU_HOST_A/acwork/acdata")
    for c in cands:
        if os.path.isdir(os.path.join(c, "records")):
            return c
    raise SystemExit("no acdata root with records/ found")


def iter_rows(root):
    files = sorted(glob.glob(os.path.join(root, "records", "*.jsonl")))
    for f in files:
        shard = os.path.basename(f)
        with open(f) as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    yield None, shard, ln
                    continue
                yield d, shard, ln


def stage1(root):
    print("=" * 78)
    print("STAGE 1 — corpus integrity (no sampling)")
    print("root:", root)
    print("=" * 78)

    files = sorted(glob.glob(os.path.join(root, "records", "*.jsonl")))
    print(f"\nshards found: {len(files)}")

    rows = 0
    bad_json = 0
    ep_split = {}                      # ep -> split
    ep_rows = collections.Counter()    # ep -> n rows
    ep_steps = collections.defaultdict(set)
    ep_nsteps = {}
    split_rows = collections.Counter()
    shard_eps = collections.defaultdict(set)
    ep_shards = collections.defaultdict(set)
    img_paths = collections.defaultdict(set)   # split -> set of img
    ep_split_conflict = []
    dup_full = collections.Counter()

    for d, shard, ln in iter_rows(root):
        if d is None:
            bad_json += 1
            continue
        rows += 1
        ep, split = d.get("ep"), d.get("split")
        split_rows[split] += 1
        ep_rows[ep] += 1
        ep_steps[ep].add(d.get("step"))
        ep_shards[ep].add(shard)
        shard_eps[shard].add(ep)
        if ep in ep_split and ep_split[ep] != split:
            ep_split_conflict.append(ep)
        ep_split[ep] = split
        if d.get("step") == 0:
            ep_nsteps[ep] = d.get("n_steps")
        im = d.get("img")
        if im:
            img_paths[split].add(im)
        # cheap exact-duplicate fingerprint (ep, step) — should be unique
        dup_full[(ep, d.get("step"))] += 1

    print(f"rows parsed: {rows}   bad JSON lines: {bad_json}")
    print(f"unique episodes: {len(ep_rows)}")
    print(f"rows by split: {dict(split_rows)}")

    print("\n--- episode/shard layout ---")
    multi_shard = sum(1 for v in ep_shards.values() if len(v) > 1)
    print(f"episodes spanning >1 shard file: {multi_shard}")
    print(f"episodes per shard: median {st.median([len(v) for v in shard_eps.values()]):.0f} "
          f"min {min(len(v) for v in shard_eps.values())} "
          f"max {max(len(v) for v in shard_eps.values())}")

    print("\n--- duplicate (ep, step) keys ---")
    dups = {k: v for k, v in dup_full.items() if v > 1}
    print(f"(ep, step) keys appearing more than once: {len(dups)}")
    if dups:
        for k, v in list(dups.items())[:5]:
            print(f"   ep={k[0]} step={k[1]} -> {v} rows")

    print("\n--- step index continuity ---")
    gaps = 0
    contig = 0
    offsets = 0
    for ep, steps in ep_steps.items():
        s = sorted(steps)
        n = ep_nsteps.get(ep)
        if s and s[0] != 0:
            offsets += 1
        if len(s) != s[-1] - s[0] + 1:
            gaps += 1
        else:
            contig += 1
    print(f"episodes with contiguous step range: {contig} "
          f"({100*contig/len(ep_steps):.1f}%)")
    print(f"episodes with GAPS in step index: {gaps} ({100*gaps/len(ep_steps):.1f}%)")
    print(f"episodes not starting at step 0: {offsets}")
    if ep_nsteps:
        diffs = collections.Counter()
        for ep, steps in ep_steps.items():
            n = ep_nsteps.get(ep)
            if n is not None:
                diffs[len(steps) - n] += 1
        print(f"declared n_steps vs observed rows, (len(rows)-n_steps) distribution:")
        for k, v in sorted(diffs.items())[:6]:
            print(f"   {k:+d}: {v} episodes")

    print("\n--- split integrity ---")
    print(f"episodes whose split label changes across rows: {len(set(ep_split_conflict))}")

    tr = {ep for ep, s in ep_split.items() if s == "train"}
    va = {ep for ep, s in ep_split.items() if s == "validation"}
    print(f"train episodes: {len(tr)}   validation episodes: {len(va)}")
    print(f"EPISODE overlap train∩val: {len(tr & va)}  <-- must be 0")
    itr, iva = img_paths.get("train", set()), img_paths.get("validation", set())
    print(f"train images: {len(itr)}   validation images: {len(iva)}")
    print(f"IMAGE overlap train∩val: {len(itr & iva)}  <-- must be 0")

    print("\n--- steps per episode ---")
    c = sorted(ep_rows.values())
    print(f"  n episodes {len(c)} | median {st.median(c):.0f} | mean {st.mean(c):.1f} "
          f"| p90 {c[int(.9*len(c))]} | max {max(c)}")
    dist = collections.Counter(c)
    print("  top episode lengths:", dict(sorted(dist.items(), key=lambda kv: -kv[1])[:8]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--stage", type=int, default=1)
    a = ap.parse_args()
    root = a.root or default_root()
    if a.stage == 1:
        stage1(root)
