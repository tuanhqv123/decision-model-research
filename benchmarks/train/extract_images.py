"""Extract only the screenshots a training set needs, at a chosen width, straight from the public GCS
AndroidControl shards (download -> extract -> delete, shard by shard, so disk stays small).

Needed = every (episode, step) in the given compact_*.jsonl files; written to <out>/imgs/<ep>/<step>.jpg,
the same paths the compact records reference. Default width 858 = d1-2B's own 1.64MP picture cap for a
1080x2400 screen (it trained at that budget; 720 px under-used it).

Usage: python extract_images.py --out acdata --width 858 acdata/compact_cur15k.jsonl acdata/compact_validation.jsonl
"""
import argparse
import gzip
import io
import json
import os
import struct
import subprocess
from multiprocessing import Pool

B = "https://storage.googleapis.com/gresearch/android_control"


def records(path):
    with gzip.open(path, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                return
            n = struct.unpack("<Q", h)[0]
            f.read(4)
            d = f.read(n)
            f.read(4)
            yield d


def do_shard(args):
    k, out, width, need, tmp = args
    from PIL import Image
    from tfrecord import example_pb2
    name = f"android_control-{k:05d}-of-00020"
    done = f"{out}/.extract_{name}.done"
    if os.path.exists(done):
        return name, "skip"
    path = f"{tmp}/{name}"
    subprocess.run(["curl", "-sfL", "--retry", "5", "-o", path, f"{B}/{name}"], check=True)
    n = 0
    for raw in records(path):
        ex = example_pb2.Example()
        ex.ParseFromString(raw)
        f = ex.features.feature
        ep = f["episode_id"].int64_list.value[0]
        steps = need.get(ep)
        if not steps:
            continue
        shots = list(f["screenshots"].bytes_list.value)
        os.makedirs(f"{out}/imgs/{ep}", exist_ok=True)
        for i in steps:
            if i < len(shots):
                im = Image.open(io.BytesIO(shots[i])).convert("RGB")
                im.resize((width, round(im.height * width / im.width)), Image.BICUBIC).save(
                    f"{out}/imgs/{ep}/{i}.jpg", quality=90)
                n += 1
    os.remove(path)
    open(done, "w").close()
    return name, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=858)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("--shards", default="0-19", help="shard range to download, e.g. 0-6 (data curated from those)")
    ap.add_argument("compact", nargs="+")
    a = ap.parse_args()
    need = {}
    for p in a.compact:
        for line in open(p):
            r = json.loads(line)
            need.setdefault(r["ep"], set()).add(r["step"])
    total = sum(len(v) for v in need.values())
    print(f"[extract] {total} screenshots from {len(need)} episodes at width {a.width}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    got = 0
    with Pool(a.workers) as pool:
        lo, hi = map(int, a.shards.split("-"))
        for name, n in pool.imap_unordered(do_shard, [(k, a.out, a.width, need, a.tmp) for k in range(lo, hi + 1)]):
            got += n if isinstance(n, int) else 0
            print(f"[extract] {name}: {n}", flush=True)
    print(f"EXTRACT_DONE {got}/{total}", flush=True)
    if got != total:  # every training/validation screenshot must exist, or the run would silently train on less
        raise SystemExit(f"EXTRACT_INCOMPLETE {got}/{total}")


if __name__ == "__main__":
    main()
