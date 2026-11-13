"""Compact AndroidControl rows -> generative examples (prompt text, image path, target JSON).

The trainer converts rows on the fly with the same functions (gen_common), so `--out` is for inspection only;
`--check` is the gate before any GPU time:
  * action-type counts and the history source of every row;
  * every target is RE-PARSED by the serving parser and must come back identical;
  * every target, parsed and scored through eval_d1's own path (val_row -> run_acv2.score_row): the rows it still
    loses are the grid representation's own ceiling (reported, not a failure);
  * the rows where the target string is NOT copyable from the state (the one-pass arm's pointer read has no
    supervision there; generation does) — a disclosed asymmetry, counted here.

Usage:
  python benchmarks/gen/make_gen_data.py --data <acdata> --split cur15k_v1 --check
  python benchmarks/gen/make_gen_data.py --data <acdata> --split validation_v1 --check --out /tmp/gen_val.jsonl
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from ac_common import SPATIAL, find_span  # noqa: E402
from gen_common import (from_grid, parse_action, scorer_args, state_text, target_obj,  # noqa: E402
                        target_text, user_text)


def convert(rec):
    """One compact row -> the example the trainer builds: {ep, step, img, W, H, prompt, target}."""
    return {"ep": rec["ep"], "step": rec["step"], "img": rec["img"], "W": rec["W"], "H": rec["H"],
            "prompt": user_text(rec["goal"], rec.get("history")), "target": target_text(rec)}


def _pct(a, b):
    return f"{a}/{b} = {100.0 * a / max(1, b):.2f}%"


def check(rows, data):
    """Returns (report dict, number of hard failures)."""
    from eval_d1 import val_row          # eval_d1's own row shape, so the scorer path is literally the same
    from run_acv2 import score_row
    types, src, errors = collections.Counter(), collections.Counter(), collections.Counter()
    reparse_bad, scored_bad, in_box, n_spatial = 0, collections.Counter(), 0, 0
    scored_bad_rows = []
    not_copyable = collections.Counter()
    ans_len, prompt_len = [], []
    examples = {}
    for rec in rows:
        at = rec["action"]["action_type"]
        types[at] += 1
        src[rec.get("history_source")] += 1
        try:
            obj = target_obj(rec)
            txt = target_text(rec)
        except ValueError as e:
            errors[str(e)] += 1
            continue
        examples.setdefault(at, txt)
        ans_len.append(len(txt))
        prompt_len.append(len(user_text(rec["goal"], rec.get("history"))))
        parsed, err, flags = parse_action(txt)
        if err or parsed != obj or flags:
            reparse_bad += 1
            continue
        if at in SPATIAL:
            n_spatial += 1
            t = rec["target"]
            px, py = from_grid(parsed["x"], rec["W"]), from_grid(parsed["y"], rec["H"])
            in_box += (t["x_min"] <= px <= t["x_max"]) and (t["y_min"] <= py <= t["y_max"])
        if rec.get("target") is not None or at not in SPATIAL:
            ok, _ = score_row(val_row(rec, data), *scorer_args(parsed, rec["W"], rec["H"]))
            if not ok:
                scored_bad[at] += 1
                scored_bad_rows.append([rec["ep"], rec["step"]])
        st = state_text(rec["goal"], rec.get("history"))
        if at == "open_app" and find_span(st, rec["action"].get("app_name")) is None:
            not_copyable["open_app"] += 1
        if at == "type_text" and find_span(st, rec["action"].get("text")) is None:
            not_copyable["type_text"] += 1
    n = len(rows)
    ans_len.sort()
    prompt_len.sort()
    rep = {
        "rows": n,
        "action_types": dict(types.most_common()),
        "history_source": dict(src),
        "conversion_errors": dict(errors),
        "reparse_mismatches": reparse_bad,
        # The grid's own ceiling: gold answers the scorer still rejects (zero-width/height gold boxes).
        "gold_scored_wrong_by_type": dict(scored_bad),
        "gold_scored_wrong_rows (ep, step)": scored_bad_rows[:20],
        "decoded_gold_point_inside_target_bbox": _pct(in_box, n_spatial),
        "not_copyable_from_state (one-pass pointer read unsupervised there)": {
            k: _pct(v, types[k]) for k, v in not_copyable.items()},
        "answer_chars": {"p50": ans_len[len(ans_len) // 2], "max": ans_len[-1]} if ans_len else None,
        "prompt_chars_excl_image": {"p50": prompt_len[len(prompt_len) // 2], "max": prompt_len[-1]}
        if prompt_len else None,
        "example_target_per_type": examples,
    }
    # Hard failures = the conversion cannot say the gold action, or the parser does not give it back.
    # A gold answer the scorer rejects is a measured property of the grid, reported above, not a failure.
    hard = sum(errors.values()) + reparse_bad
    return rep, hard


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="acdata dir holding compact_<split>.jsonl and imgs/")
    ap.add_argument("--split", default="cur15k_v1")
    ap.add_argument("--out", default="", help="write the converted examples here (inspection only)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    with open(os.path.join(a.data, f"compact_{a.split}.jsonl")) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if a.limit:
        rows = rows[:a.limit]
    if a.out:
        with open(a.out, "w") as fh:
            for rec in rows:
                fh.write(json.dumps(convert(rec), ensure_ascii=False) + "\n")
        print(f"[out] {len(rows)} examples -> {a.out}", flush=True)
    if a.check:
        rep, hard = check(rows, a.data)
        print("CHECK " + json.dumps(rep, indent=1, ensure_ascii=False), flush=True)
        print(f"CHECK_{'FAIL' if hard else 'PASS'} hard_failures={hard}", flush=True)
        return 1 if hard else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
