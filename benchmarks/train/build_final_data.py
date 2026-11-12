#!/usr/bin/env python3
"""The consolidated final training mix (2026-11-12), built from the v5 curated rows (`compact_cur_v6`, stop pairs
protected) and the base AndroidControl episodes of `compact_awlike_full_v5`. Every change is one finding of the
2026-11-12 trace analysis of the 91 failed AndroidWorld tasks or of the ckpt_250 early warning
(docs/research/2026-11-12-FINAL-PLAN-vi.md):

  * cut_recovery      recovery rows 3,468 -> CAP pairs: ckpt_250 misused navigate_back 44/798 (v4: 16/798),
                      the second time recovery rows did this (pilot 1: ~10%);
  * long_press_repeats every AndroidControl long_press row and the step after it (the menu choice) x K: Markor
                      delete/move, Files, SMS resend start with a long press the model gives P 0.0002-0.04;
  * satisfied_rows    a screen already in the goal state is `status` without flipping it: AndroidWorld's *Verify
                      tasks start satisfied and the model flipped the switch back;
  * extra answer episodes (build_v4_data.answer_rows on unused episodes): the model emitted `answer` in only 6 of
                      21 failed information-retrieval tasks.

No AndroidWorld data: every row is an AndroidControl screen.

    python3 build_final_data.py CODE_DIR DATA_DIR  ->  DATA_DIR/compact_cur_final.jsonl
"""
import collections
import json
import random
import sys

A = None
RECOVERY_CAP = 600
LONG_PRESS_K = 4
EXTRA_ANSWER = 350
# Satisfied-state rows are OFF (2026-11-12): the second final's ckpt_250 raised false `status` on validation to 108/690
# (first final 67, v4 64); 34% of the 53 new false stops sit on screens with a checkable element vs 16% of all rows.
# The rows teach "the screen looks done -> status" without the completing action, which is the premature-done failure.
INCLUDE_SATISFIED = False


def cut_recovery(rows, rng, cap=RECOVERY_CAP):
    """Keep `cap` recover_back rows that still have their twin, and exactly those twins; other rows untouched."""
    have_twin = {(r["ep"], r["step"]) for r in rows if r.get("aug") == "recover_twin"}
    backs = [r for r in rows if r.get("aug") == "recover_back" and tuple(r["branch_from"]) in have_twin]
    rng.shuffle(backs)
    keep = backs[:cap]
    want = {tuple(r["branch_from"]) for r in keep}
    out = [r for r in rows if r.get("aug") not in ("recover_back", "recover_twin")]
    seen = set()
    for r in rows:
        if r.get("aug") == "recover_twin" and (r["ep"], r["step"]) in want and (r["ep"], r["step"]) not in seen:
            out.append(r)
            seen.add((r["ep"], r["step"]))
    return out + keep


def drop_id_keyed_answers(rows):
    """Drop answer episodes whose question names a contract-v5 id label ('[id: ...]'), which is not screen text
    (8 of the 250 v5 answer episodes; key_value_pairs no longer picks such keys)."""
    bad = {r["ep"] for r in rows if r.get("aug") == "answer" and "[id:" in (r.get("goal") or "")}
    return [r for r in rows if r["ep"] not in bad]


def long_press_repeats(eps, k=LONG_PRESS_K):
    """k copies of every long_press row and of the row right after it in the same episode."""
    out = []
    for rows in eps.values():
        by_step = {r["step"]: r for r in rows}
        for r in rows:
            if (r.get("action") or {}).get("action_type") == "long_press":
                for x in (r, by_step.get(r["step"] + 1)):
                    if x is not None:
                        out += [dict(x, aug="lp_rep") for _ in range(k)]
    return out


def _checkable_target(r):
    i = r.get("target_el_idx")
    els = r.get("elements") or []
    return i is not None and 0 <= i < len(els) and bool(els[i].get("is_checkable"))


def satisfied_rows(eps):
    """For an episode whose last real action clicks a checkable element: its `status` screen, with the history
    before that click, labelled `status` (the goal already holds; flipping it would undo it)."""
    out = []
    for rows in eps.values():
        by_step = {r["step"]: r for r in rows}
        n = max(r["n_steps"] for r in rows)
        done, flip = by_step.get(n), by_step.get(n - 1)
        if not done or not flip or (done.get("action") or {}).get("action_type") != "status":
            continue
        if (flip.get("action") or {}).get("action_type") != "click" or not _checkable_target(flip):
            continue
        out.append(dict(done, history_v3=list(done["history_v3"])[:-1], aug="satisfied", ep="sat:%s" % done["ep"],
                        local_goal=None))
    return out


def main(code_dir, data_dir, seed=0):
    global A
    sys.path.insert(0, code_dir)
    import ac_common as _A
    import build_v4_data as B
    A = _A
    B.A = _A
    rng = random.Random(seed)
    cur = [json.loads(l) for l in open("%s/compact_cur_v6.jsonl" % data_dir)]
    full = [json.loads(l) for l in open("%s/compact_awlike_full_v5.jsonl" % data_dir)]
    eps = collections.defaultdict(list)
    for r in full:
        if r.get("aug") == "base":
            eps[r["ep"]].append(r)
    for rows in eps.values():
        rows.sort(key=lambda r: r["step"])
    out = drop_id_keyed_answers(cut_recovery(cur, rng))
    lp = long_press_repeats(eps)
    sat = satisfied_rows(eps) if INCLUDE_SATISFIED else []
    used = {r["ep"].split(":", 1)[1] for r in cur if r.get("aug") == "answer"}
    B.CAPS["answer"] = EXTRA_ANSWER
    rule = A.CONTRACTS["v5"]["element_rule"]
    extra = [r for ep_rows in B.answer_rows({e: v for e, v in eps.items() if e not in used}, rule, rng) for r in ep_rows]
    out += lp + sat + extra
    rng.shuffle(out)
    with open("%s/compact_cur_final.jsonl" % data_dir, "w") as f:
        for r in out:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    acts = collections.Counter((r.get("action") or {}).get("action_type", "avoid") for r in out)
    print(json.dumps({"rows": len(out), "long_press_rep": len(lp), "satisfied": len(sat),
                      "extra_answer_rows": len(extra), "extra_answer_episodes": len({r["ep"] for r in extra}),
                      "recover_back": sum(r.get("aug") == "recover_back" for r in out),
                      "recover_twin": sum(r.get("aug") == "recover_twin" for r in out),
                      "actions": {k: "%d (%.1f%%)" % (v, 100 * v / len(out)) for k, v in acts.most_common()}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
