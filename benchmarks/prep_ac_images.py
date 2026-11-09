"""Download AndroidControl TEST shards (WITH screenshots) and extract images.

First N rows -> benchmarks/data/ac_test_imgs/<row_id>_step<k>.png
Plus a manifest jsonl with (row_id, step, goal, step_instruction, action, img).

Usage: python3 benchmarks/prep_ac_images.py [N_ROWS] [N_SHARDS]
"""
import base64
import io
import json
import os
import sys

from huggingface_hub import hf_hub_download

REPO = "smolagents/android-control"
NUM_SHARDS = 67
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = "benchmarks/data/ac_test_imgs"
MANIFEST = "benchmarks/data/ac_test_manifest.jsonl"

os.makedirs(OUT, exist_ok=True)


def main():
    from PIL import Image

    manifest = open(MANIFEST, "w")
    written = 0
    for i in range(N_SHARDS):
        fname = f"data/test-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
        path = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)  # full table incl. images (needed here)
        cols = tbl.to_pylist()
        for row in cols:
            shots = row.get("screenshots_b64") or []
            steps = row.get("step_instructions") or []
            acts = row.get("actions") or []
            n = min(len(shots), len(steps), len(acts))
            for s in range(n):
                if written >= N_ROWS:
                    break
                try:
                    img = Image.open(io.BytesIO(base64.b64decode(shots[s])))
                    rid = f"ep{row.get('episode_id', i)}"
                    img_path = f"{OUT}/{rid}_s{s}.png"
                    img.save(img_path)
                    a = acts[s] or {}
                    manifest.write(json.dumps({
                        "row_id": rid, "step": s,
                        "goal": row.get("goal"),
                        "step_instruction": str(steps[s]),
                        "action": a,
                        "img": img_path,
                        "img_size": [img.width, img.height],
                    }) + "\n")
                    written += 1
                except Exception as e:
                    print(f"  skip {rid} s{s}: {e}")
            if written >= N_ROWS:
                break
        print(f"shard {i}: total {written}", flush=True)
        if written >= N_ROWS:
            break
    manifest.close()
    print(f"DONE: {written} images -> {OUT}, manifest {MANIFEST}")


if __name__ == "__main__":
    main()