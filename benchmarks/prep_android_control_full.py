"""Download the FULL text-only AndroidControl dataset (train + test).

Reads parquet shards directly with pyarrow, keeping ONLY the text columns
(goal / step_instructions / actions) so the ~67 GB base64 image column is
never materialized.

Outputs:
  benchmarks/data/android_control_train.jsonl   (12,232 episodes)
  benchmarks/data/android_control_test_full.jsonl (3,051 episodes)

Usage: python3 benchmarks/prep_android_control_full.py [split|both]
"""
import json
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO = "smolagents/android-control"
COLS = ["goal", "step_instructions", "actions"]
OUT_DIR = "benchmarks/data"


def keep_action(a):
    if not isinstance(a, dict):
        return None
    return {
        "action_type": a.get("action_type"),
        "text": a.get("text"),
        "direction": a.get("direction"),
        "app_name": a.get("app_name"),
    }


def episode_out(ex):
    acts = ex.get("actions") or []
    steps = ex.get("step_instructions") or []
    if not acts or not steps:
        return None
    return {
        "goal": ex.get("goal"),
        "step_instructions": [str(s) for s in steps if s is not None],
        "actions": [ka for ka in (keep_action(a) for a in acts) if ka],
    }


def run_split(name, fname, out_path):
    writes = 0
    with open(out_path, "w") as f:
        path = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
        tbl = pq_read(path)
        for ex in tbl.to_pylist():
            rec = episode_out(ex)
            if rec is None:
                continue
            f.write(json.dumps(rec) + "\n")
            writes += 1
        print(f"[{name}] {fname}: {tbl.num_rows} rows -> {writes} episodes", flush=True)
    return writes


def pq_read(path):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=COLS)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.makedirs(OUT_DIR, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")

    for split in (["train", "test"] if which == "both" else [which]):
        pattern = f"data/{split}-"
        shards = sorted(f for f in files if f.startswith(pattern) and f.endswith(".parquet"))
        print(f"[{split}] {len(shards)} shards", flush=True)
        total = 0
        for i, sh in enumerate(shards):
            out = f"{OUT_DIR}/android_control_{split}.jsonl"
            with open(out, "a") as f:
                path = hf_hub_download(repo_id=REPO, filename=sh, repo_type="dataset")
                tbl = pq_read(path)
                n = 0
                for ex in tbl.to_pylist():
                    rec = episode_out(ex)
                    if rec is None:
                        continue
                    f.write(json.dumps(rec) + "\n")
                    n += 1
                total += n
                print(f"[{split}] shard {i+1}/{len(shards)}: +{n} (cum {total})", flush=True)
        print(f"[{split}] DONE: {total} episodes -> {out}", flush=True)


if __name__ == "__main__":
    main()