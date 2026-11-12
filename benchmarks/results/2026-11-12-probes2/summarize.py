"""Summarise probe_aw.py output: per probe and variant, mean P(option of interest), top-1 share, and the paired change
against the first variant with a 95% bootstrap interval resampled BY EPISODE (task), plus the top-1 flip rate.
    python3 summarize.py OUT.jsonl [OUT2.jsonl ...]"""
import collections as C, json, random, sys

KEY = {"P1": "same", "P2": "same_el", "P3": "terminate", "P4": "terminate", "P5": "write"}


def val(r):
    if r["probe"] == "P1":
        return r["p"].get(r["same"], 0.0)
    if r["probe"] == "P2":
        return r["p"].get("click", 0.0) * (r.get("p_same_el") or 0.0)
    return r["p"].get(KEY[r["probe"]], 0.0)


def top_is(r):
    if r["probe"] == "P1":
        return r["top1"] == r["same"]
    if r["probe"] == "P2":
        return r["top1"] == "click" and (r.get("p_same_el") or 0) > 0.5
    return r["top1"] == KEY[r["probe"]]


for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    print("==", f, len(rows), "rows")
    by = C.defaultdict(list)
    for r in rows:
        by[r["probe"]].append(r)
    for pr in sorted(by):
        rs = by[pr]
        variants = list(dict.fromkeys(r["variant"] for r in rs))
        # pair rows: consecutive groups of len(variants) belong to one state
        groups, cur = [], []
        for r in rs:
            cur.append(r)
            if len(cur) == len(variants):
                groups.append(cur)
                cur = []
        for vi, v in enumerate(variants):
            xs = [g[vi] for g in groups]
            line = "  %s %-16s n=%3d (eps %2d)  mean %.3f  top1 %.0f%%" % (
                pr, v, len(xs), len({x["ep"] for x in xs}), sum(map(val, xs)) / len(xs), 100 * sum(map(top_is, xs)) / len(xs))
            if vi:
                d = C.defaultdict(list)
                for g in groups:
                    d[g[0]["ep"]].append(val(g[vi]) - val(g[0]))
                flips = sum(top_is(g[vi]) != top_is(g[0]) for g in groups)
                eps = list(d)
                rng = random.Random(0)
                bs = []
                for _ in range(2000):
                    s = [x for e in (rng.choice(eps) for _ in eps) for x in d[e]]
                    bs.append(sum(s) / len(s))
                bs.sort()
                m = sum(x for e in eps for x in d[e]) / sum(len(d[e]) for e in eps)
                line += "  Δ %+.3f [%+.3f, %+.3f]  top1 flips %d/%d" % (m, bs[50], bs[1949], flips, len(groups))
            print(line)
