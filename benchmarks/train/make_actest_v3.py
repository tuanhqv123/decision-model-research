#!/usr/bin/env python3
"""AndroidControl-High test rows in contract v3, for SCORING ONLY (never trained on).

Input: `build_ac_full_v2.py --eval-test` rows of the official test split (every step, plus our terminal `status` row).
Steps: v3 history (`add_history_v3.episode_entries`, the training renderer) -> drop the terminal `status` rows (the
benchmark has none) -> one fixed app list on every row -> check every benchmark step is present once with the same
action type as InfiGUI's `step_check_pams`.

    python make_actest_v3.py RAW.jsonl INFIGUI_TEST.json APP_VOCAB.json OUT_DIR
      -> OUT_DIR/compact_actest_v3.jsonl         app list = model's known apps + every app the test phones need
      -> OUT_DIR/compact_actest_v3_strict.jsonl  app list = model's known apps only (no test information)
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_OPTIONS = 702  # d1-2B option codes (d1_codes 2): A..Z then AA..ZZ; longer lists are silently cut by zip
TYPE_OF = {"click": "click", "long_press": "long_press", "scroll": "swipe", "input_text": "type", "type_text": "type",
           "open_app": "open", "navigate_back": "system_button", "navigate_home": "system_button", "wait": "wait"}


def app_list(known, extra, cap=MAX_OPTIONS):
    """One fixed, alphabetical, case-insensitively de-duplicated list (first spelling kept)."""
    seen, out = set(), []
    for name in list(known) + list(extra):
        k = name.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(name.strip())
    out.sort(key=lambda n: (n.lower(), n))
    if len(out) > cap:
        raise SystemExit(f"app list of {len(out)} > {cap}: d1 cannot code that many options")
    return out


def alignment(rows, test):
    """Every benchmark (episode, step) present exactly once with the same action type."""
    have = {(r["ep"], r["step"]): r for r in rows}
    bench = {(x["episode"]["episode_id"], i): c for x in test for i, c in enumerate(x["step_check_pams"])}
    matched = mism = missing = 0
    for key, c in bench.items():
        r = have.get(key)
        if r is None:
            missing += 1
        elif TYPE_OF.get(r["action"]["action_type"]) == c["action"]:
            matched += 1
        else:
            mism += 1
    return {"benchmark_steps": len(bench), "matched": matched, "type_mismatch": mism, "missing": missing,
            "extra": sum(1 for k in have if k not in bench)}


def main(raw_path, test_path, vocab_path, out_dir):
    from add_history_v3 import episode_entries
    rows = [json.loads(line) for line in open(raw_path) if line.strip()]
    by_ep = collections.defaultdict(list)
    for r in rows:
        by_ep[r["ep"]].append(r)
    for ep_rows in by_ep.values():
        ep_rows.sort(key=lambda r: r["step"])
        for r, h in zip(ep_rows, episode_entries(ep_rows)):
            r["history_v3"] = h
    kept = [r for r in rows if r["action"]["action_type"] != "status"]
    test = json.load(open(test_path))
    a = alignment(kept, test)
    print("ALIGN " + json.dumps(a), flush=True)
    if a["missing"] or a["type_mismatch"] or a["extra"]:
        raise SystemExit("the test rows do not line up with the benchmark: fix before scoring")
    known = json.load(open(vocab_path))
    needed = [c["text"] for x in test for c in x["step_check_pams"] if c["action"] == "open"]
    for name, apps in (("compact_actest_v3", app_list(known, needed)), ("compact_actest_v3_strict", app_list(known, []))):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as fh:
            for r in kept:
                fh.write(json.dumps(dict(r, apps=apps), separators=(",", ":")) + "\n")
        print(f"WROTE {name}: {len(kept)} rows, app list {len(apps)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
