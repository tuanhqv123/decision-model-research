#!/usr/bin/env python3
"""Extra copies of the executor rows that CONTINUE a multi-step step instruction (2026-11-14).

A `span` row continues an instruction when the row before it in the file is a `span` row of the same episode whose
`local_done` is False (the instruction was not finished there). exec1 (build_exec_data rows, one copy each) says "done"
on 34% (530/1,547) of the AndroidControl-Low steps whose instruction repeats the previous step's, vs 2.8% elsewhere.
Each continuation row is written `k` times (copies marked `aug="span_cont"`); every other row once, order kept.
    python3 upweight_exec.py IN.jsonl OUT.jsonl [--k 3] [--extra-only]
--extra-only writes only the k-1 added copies of each continuation row (train with --train-set IN,OUT).
"""
import json
import os
import sys


def upweight(rows, k=3):
    prev = None
    for r in rows:
        yield r
        cont = (r["aug"] == "span" and prev is not None and prev["aug"] == "span"
                and str(prev["ep"]) == str(r["ep"]) and prev.get("local_done") is False)
        if cont:
            for _ in range(k - 1):
                yield dict(r, aug="span_cont")
        prev = r


def extras(rows, k=3):
    """Only the added copies: the trainer reads several files, so IN + these = the upweighted data."""
    return (r for r in upweight(rows, k) if r["aug"] == "span_cont")


def main(src, dst, k=3, extra_only=False):
    n = c = 0
    gen = extras if extra_only else upweight
    with open(dst + ".tmp", "w") as fo:
        for r in gen((json.loads(l) for l in open(src)), k):
            fo.write(json.dumps(r, separators=(",", ":")) + "\n")
            n += 1
            c += r["aug"] == "span_cont"
    os.replace(dst + ".tmp", dst)
    print(json.dumps({"rows": n, "extra continuation copies": c, "k": k}))


if __name__ == "__main__":
    a = sys.argv[1:]
    k = 3
    if "--k" in a:
        i = a.index("--k")
        k = int(a[i + 1])
        del a[i:i + 2]
    extra_only = "--extra-only" in a
    a = [x for x in a if x != "--extra-only"]
    main(a[0], a[1], k, extra_only)
