"""Prep a bounded, TEXT-ONLY sample of the raw AndroidControl dataset.

Reads the parquet shards directly with pyarrow (not datasets.load_dataset,
which insists all declared splits be present), keeping ONLY the text columns
(goal / step_instructions / actions) so the ~67 GB image column is never
materialized in memory or JSON.

Usage: python3 benchmarks/prep_android_control.py [N_SHARDS] [OUT]
"""
import json
import os
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

N_SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/data/android_control_test.jsonl"

REPO = "smolagents/android-control"
COLS = ["goal", "step_instructions", "actions"]
NUM_SHARDS = 67  # test split

os.makedirs(os.path.dirname(OUT), exist_ok=True)


def keep_action(a):
    if not isinstance(a, dict):
        return None
    return {
        "action_type": a.get("action_type"),
        "text": a.get("text"),
        "direction": a.get("direction"),
        "app_name": a.get("app_name"),
    }


def main():
    writes = 0
    with open(OUT, "w") as f:
        for i in range(N_SHARDS):
            fname = f"data/test-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
            path = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
            tbl = pq.read_table(path, columns=COLS)
            print(f"  shard {i}: {tbl.num_rows} rows")
            for ex in tbl.to_pylist():
                acts = ex.get("actions") or []
                steps = ex.get("step_instructions") or []
                if not acts or not steps:
                    continue
                rec = {
                    "goal": ex.get("goal"),
                    "step_instructions": [str(s) for s in steps if s is not None],
                    "actions": [ka for ka in (keep_action(a) for a in acts) if ka],
                }
                f.write(json.dumps(rec) + "\n")
                writes += 1
    print(f"Done. Wrote {writes} episodes -> {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
