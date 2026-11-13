#!/usr/bin/env python3
"""Executor training data for the LLM + decision-model harness (2026-11-13). AndroidControl train split only; no emulator,
no AndroidWorld rows, no external model. Design and evidence: docs/research/2026-11-13-LLM-PLUS-DECISION-HARNESS-vi.md.

Roles: the LLM plans, writes text, answers and decides when the whole task is finished; the decision model executes a
LOCAL GOAL (one or a few steps) and says when that local goal is done (`status`). So:

  goal       a fixed share of episodes (`--p-goal`) as AndroidControl-High: goal only, every action row, and NO `status`
             row (the AC test has no termination step; finishing the task is the LLM's call)
  span       the other episodes: the AndroidControl step instructions (`local_goal`; one instruction can cover several
             steps until `local_done`) are grouped into runs of 1..MAX_SPAN consecutive instructions. Every action row of a
             run is shown "Step instruction: <the run's instructions joined>"; rows inside a run teach "not done yet"
  span_done  after each run, one row on the screen after its last action (that step's own screen, elements and history)
             with the run's text and gold `status` = "this local goal is done"

The contract stays v4 (same questions, same state renderer); a row says whether its instruction is shown
(`show_instruction`, read by `ac_common.instruction_included`). No BLOCKED label: AndroidControl has no failed local
goal; the harness reads BLOCKED from low confidence, an unchanged screen, or a step budget per local goal.

    python3 build_exec_data.py CODE_DIR IN_V2.jsonl OUT.jsonl [--seed 0] [--p-goal 0.34] [--open-end]
    (evaluation: --p-goal 0 --open-end on the AndroidControl test rows, which have no final screen)
IN_V2 = rows of build_ac_full_v2.py (complete episodes, the final `status` row included).
"""
import collections
import itertools
import json
import os
import random
import sys

A = None  # ac_common, set in main() from CODE_DIR (tests set it directly)
MAX_SPAN = 4
SPAN_WEIGHTS = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
P_GOAL = 0.34
STATUS = {"action_type": "status", "goal_status": "successful"}
NO_TARGET = dict(target=None, target_el_idx=None, target_el_exact=None, target_el_hits_gold=None,
                 target_el_centre_idx=None)


def _clean(s):
    return " ".join((s or "").split()).rstrip(" .")


def units(rows):
    """Action rows of one episode grouped into instructions: consecutive steps with the same step instruction, a group
    closing when `local_done` is True or the instruction changes."""
    out, cur = [], []
    for r in rows:
        if r["action"]["action_type"] == "status":
            continue
        if cur and _clean(cur[-1]["local_goal"]) != _clean(r["local_goal"]):
            out.append(cur)
            cur = []
        cur.append(r)
        if r.get("local_done"):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def join_instructions(texts):
    parts = [_clean(t) for t in texts]
    return ", then ".join([parts[0]] + [p[:1].lower() + p[1:] if p[1:2].islower() else p for p in parts[1:]])


def draw_spans(n, rng):
    spans = []
    while sum(spans) < n:
        k = rng.choices(list(SPAN_WEIGHTS), weights=list(SPAN_WEIGHTS.values()))[0]
        spans.append(min(k, n - sum(spans)))
    return spans


def choose_mode(rows, rng, p_goal=P_GOAL):
    if any(not _clean(r.get("local_goal")) for r in rows if r["action"]["action_type"] != "status"):
        return "goal"
    return "goal" if rng.random() < p_goal else "span"


def episode_rows(rows, mode, spans, stats=None):
    """Training rows of one prepared episode (rows sorted by step, the final `status` row last; an evaluation file may
    end at the last action, and then the last run has no done row)."""
    if mode == "goal":
        return [dict(r, show_instruction=False, aug="goal") for r in rows if r["action"]["action_type"] != "status"]
    by_step = {r["step"]: r for r in rows}
    us, out, i = units(rows), [], 0
    for k in spans:
        run = us[i:i + k]
        i += k
        text = join_instructions([u[0]["local_goal"] for u in run])
        for u in run:
            out += [dict(r, local_goal=text, show_instruction=True, aug="span") for r in u]
        nxt = by_step.get(run[-1][-1]["step"] + 1)
        if nxt is None:
            if stats is not None:
                stats["done rows without a next screen"] += 1
            continue
        out.append(dict(nxt, ep="done:%s" % nxt["ep"], action=dict(STATUS), local_goal=text, local_done=None,
                        show_instruction=True, aug="span_done", **NO_TARGET))
    return out


def iter_build(ac_rows, rule, stats, seed=0, p_goal=P_GOAL, open_end=False):
    """Rows of `ac_rows` (sorted by episode; each episode's rows adjacent), one episode at a time: a 3 GB input never
    has to fit in memory. `open_end`: evaluation files (the AndroidControl test rows end at the last action) are
    accepted without the final `status` row; training files must be complete."""
    import build_v4_data as B
    B.A = A
    seen = set()
    for ep, group in itertools.groupby(ac_rows, key=lambda r: r["ep"]):
        if ep in seen:
            raise ValueError("episode %s is not contiguous in the input" % ep)
        seen.add(ep)
        rows = sorted((dict(r, elements=B.unfocus(r["elements"])) for r in group), key=lambda r: r["step"])
        ends = rows[-1]["action"]["action_type"] == "status" or open_end
        if [r["step"] for r in rows] != list(range(len(rows))) or not ends:
            stats["episodes skipped (incomplete)"] += 1
            continue
        for r, h in zip(rows, B.entries_v4(rows, rule)):
            r["history_v3"] = h
        rng = random.Random("%s:%s" % (seed, ep))
        mode = choose_mode(rows, rng, p_goal)
        spans = draw_spans(len(units(rows)), rng) if mode == "span" else []
        stats["episodes " + mode] += 1
        stats.update("span length %d" % k for k in spans)
        for r in episode_rows(rows, mode, spans, stats):
            stats["rows"] += 1
            stats["rows " + r["aug"]] += 1
            stats["status rows"] += r["action"]["action_type"] == "status"
            yield r


def build(ac_rows, rule, seed=0, p_goal=P_GOAL, open_end=False):
    stats = collections.Counter()
    return list(iter_build(ac_rows, rule, stats, seed, p_goal, open_end)), stats


def main(code_dir, src, dst, seed=0, p_goal=P_GOAL, open_end=False):
    global A
    sys.path.insert(0, code_dir)
    import ac_common as _A
    A = _A
    stats = collections.Counter()
    with open(dst + ".tmp", "w") as fo:
        for r in iter_build((json.loads(l) for l in open(src)), A.CONTRACTS["v4"]["element_rule"], stats, seed, p_goal,
                            open_end):
            fo.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(dst + ".tmp", dst)
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = {}
    for flag in ("--seed", "--p-goal"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            del args[i:i + 2]
    open_end = "--open-end" in args
    args = [a for a in args if a != "--open-end"]
    main(*args[:3], seed=int(opts.get("--seed", 0)), p_goal=float(opts.get("--p-goal", P_GOAL)), open_end=open_end)
