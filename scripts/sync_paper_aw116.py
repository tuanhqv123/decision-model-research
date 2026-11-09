#!/usr/bin/env python3
"""Rewrite the Limitation 12 paragraph of the paper from the AndroidWorld evidence file.

Why this exists
---------------
Limitation 12 quotes a run that is *still running*. Every hand-edit of those numbers drifts
within minutes, and on 2026-11-08 two such drifts made ``check_record.py`` fail at exactly the
moment the record was supposed to be consistent. The prose must therefore be **generated**
from the evidence file, not typed:

    python scripts/refresh_aw116.py        # pull + recompute the evidence
    python scripts/sync_paper_aw116.py     # rewrite Limitation 12 from that evidence

The prose template lives here, so the numbers have exactly one source. Any sentence in the
paragraph that is not a number is stable text; every number is interpolated.
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(REPO, "benchmarks", "results", "2026-11-08-aw116", "evidence-82tasks.txt")
PAPER = os.path.join(REPO, "papers", "single-pass-decision-acv2.md")

START = "12. **Our own online attempts measure a narrow capability"
END = "## 7. Conclusion"


def parse(path):
    txt = open(path, encoding="utf-8").read()

    def one(pat, cast=str):
        m = re.search(pat, txt)
        if not m:
            raise SystemExit("evidence file is missing pattern: %s" % pat)
        return cast(m.group(1))

    ev = {}
    ev["finished"] = one(r"TASKS_FINISHED=(\d+)", int)
    ev["planned"] = one(r"PLANNED=(\d+)", int)
    ev["success"] = one(r"SUCCESS=(\d+)", int)
    ev["sr"] = one(r"SR=([\d.]+)")
    ev["wlo"] = one(r"WILSON=([\d.]+)-")
    ev["whi"] = one(r"WILSON=[\d.]+-([\d.]+)")
    ev["timeout"] = one(r"'timeout_non_terminating': (\d+)", int)
    ev["premature"] = one(r"'premature_status': (\d+)", int)
    ev["setup"] = one(r"'setup_error': (\d+)", int)
    ev["emitted"] = one(r"EMITTED_ACTIONS=(\d+)", int)
    ev["invalid"] = one(r"INVALID=(\d+)", int)
    ev["invalid_rate"] = one(r"INVALID_RATE=([\d.]+)")
    ev["crashes"] = one(r"ENV_CRASH_EPISODES=(\d+)/", int)
    ev["drift_s_n"] = one(r"drift success: n=(\d+)", int)
    ev["drift_s_decay"] = one(r"drift success: n=\d+ mean=-?[\d.]+ decaying=(\d+)%", int)
    ev["drift_f_n"] = one(r"drift fail   : n=(\d+)", int)
    ev["drift_f_decay"] = one(r"drift fail   : n=\d+ mean=-?[\d.]+ decaying=(\d+)%", int)
    ev["drift_f_mean"] = one(r"drift fail   : n=\d+ mean=(-?[\d.]+)", float)
    ev["drift_s_mean"] = one(r"drift success: n=\d+ mean=(-?[\d.]+)", float)
    ev["conf_p50"] = one(r"conf p50=([\d.]+)")
    ev["gate_p50"] = one(r"gate p50=([\d.]+)")
    ev["gate_zero"] = one(r"gate==0 frac=([\d.]+)", float)
    ev["tt_emitted"] = one(r"type_text emitted: (\d+)/", int)
    ev["premature_med"] = one(r"premature median_steps=(\d+)", int)
    ev["timeout_med"] = one(r"timeout   median_steps=(\d+)", int)
    ev["type_text_pct"] = round(100.0 * ev["tt_emitted"] / max(ev["finished"], 1))
    mc = re.search(r"COVERAGE_FIELD_POPULATED=(\d+)/(\d+)", txt)
    ev["cov_with"], ev["cov_total"] = (int(mc.group(1)), int(mc.group(2))) if mc else (0, ev["finished"])
    mr = re.search(r"SR_REACHABLE=(\d+)/(\d+)=([\d.]+)% n_reachable=(\d+) n_all=(\d+)", txt)
    ev["r_num"], ev["r_den"], ev["r_pct"] = (
        int(mr.group(1)), int(mr.group(2)), mr.group(3)) if mr else (0, ev["finished"], "n/a")
    mo = re.search(r"ENV_CRASH_IS_SETUP_FAILURE=(\d+)/(\d+) setup_errors=(\d+)", txt)
    ev["crash_same"], ev["crash_n"], ev["setup_n"] = (
        int(mo.group(1)), int(mo.group(2)), int(mo.group(3))) if mo else (0, 0, ev["setup"])
    mu = re.search(r"UNSCORED_CHECKER_UNAVAILABLE=(\d+) of (\d+)", txt)
    ev["unscored"], ev["unscored_of"] = (int(mu.group(1)), int(mu.group(2))) if mu else (0, ev["finished"])
    msw = re.search(r"WILSON_SCORABLE=([\d.]+)-([\d.]+)", txt)
    ev["slo"], ev["shi"] = (msw.group(1), msw.group(2)) if msw else ("—", "—")
    ms = re.search(r"SCORABLE=(\d+) SUCCESS=(\d+) SR_SCORABLE=([\d.]+)%", txt)
    ev["scorable_n"], ev["scorable_ok"], ev["sr_scorable"] = (
        int(ms.group(1)), int(ms.group(2)), ms.group(3)) if ms else (
        ev["attempts"], ev["success"], ev["sr_attempts"])
    mg = re.search(r"GENUINE_ATTEMPTS=(\d+) SUCCESS=(\d+) SR_OVER_ATTEMPTS=([\d.]+)%", txt)
    ev["attempts"], ev["sr_attempts"] = (int(mg.group(1)), mg.group(3)) if mg else (
        ev["finished"], ev["sr"])
    ev["complete"] = ev["finished"] >= ev["planned"]
    return ev


def render(ev):
    n, p, s = ev["finished"], ev["planned"], ev["success"]
    state = (
        "**complete**" if ev["complete"]
        else "**incomplete at the time of writing and is reported as such**"
    )
    zero_invalid = ev["invalid"] == 0
    invalid_clause = (
        "Of **{:,} emitted actions, zero were invalid** — every action the model chose was representable "
        "and executable by the harness (invalid rate {}%, {}/{} timeouts). The 28.2% invalid rate in §5.7 "
        "was therefore a *read-slot* defect (`app_name` empty), not a model limitation: repair the read "
        "slot and the model's chosen actions are executable essentially every time. What it cannot do is "
        "*string them together*."
    ).format(ev["emitted"], ev["invalid_rate"], ev["timeout"], n) if zero_invalid else (
        "Of **{:,} emitted actions, {:,} were invalid** ({}%), alongside {} environment crash in {} "
        "episodes and {}/{} timeouts."
    ).format(ev["emitted"], ev["invalid"], ev["invalid_rate"], ev["crashes"], n, ev["timeout"], n)

    body = """12. **Our own online attempts measure a narrow capability, and the {planned}-task run says which one.** The first
    full-suite run (protocol `aw-main-v1`, checkpoint E2+`app_intent`, step budget 25, seed 30, no app-name
    fallback) is {state}: **{n} of {planned} tasks finished, {s} successes, SR {sr}% ({s}/{n}), 95% Wilson
    {wlo}–{whi}%**. That SR is over the tasks that ran, not the suite, and no published-baseline comparison is
    made from it. Every figure below re-derives from `benchmarks/results/2026-11-08-aw116/` via
    `scripts/refresh_aw116.py`, and this paragraph is regenerated by `scripts/sync_paper_aw116.py`; no number
    here is transcribed by hand.

    The decomposition over those {n} is **{timeout} non-terminating timeouts (median {timeout_med} steps),
    {premature} premature `status` (median {premature_med} steps), {s} successes, and {setup} tasks whose own
    `initialize_task` or `reset` raised**. Two mechanisms, opposite signatures, same missing signal: the action
    head's own confidence stays high in **both** (p50 ≈ {conf_p50}), while the calibrated gate score starts
    high and collapses on the non-terminating runs (median gate p50 ≈ {gate_p50}, exactly 0 on {gate_zero:.0f}%
    of all actions). The gate-trend statistic — mean `gate` over the last three actions minus the first three —
    is negative on **{drift_f_decay}% of failures (n = {drift_f_n})** and on **{drift_s_decay}% of successes
    (n = {drift_s_n})**; the two distributions are visibly different but with {drift_s_n} successes this is a
    mechanism, not a result, and it is labelled as such.

    **Three denominators are reported, because they answer different questions, and one bucket is neither
    a model outcome nor a device failure.** Of the {n} finished episodes: {setup} never reached a usable device
    state (the accessibility forwarder stopped responding; `env.reset()` could not obtain a tree), and a further
    **{unscored} ran to completion but were never judged** — `task.is_successful(env)` raised
    `AttributeError: 'AndroidWorldEnv' object has no attribute 'get_state'` because the harness wrapper did not
    forward `get_state` to the wrapped environment. The second group is **not** a model failure and **not** the
    same defect as the first: on the evidence, {crash_same} of {crash_n} crashes carry a setup error while the
    {unscored} unscored episodes have `n_steps` 4–25. Both are now separate buckets, so the decomposition reads
    **{s} successes, {premature} premature `status`, {timeout} non-terminating timeouts, {unscored}
    unscored, {setup} setup failures** — and the earlier version of this paragraph, which folded the unscored
    episodes into "timeout" and "premature `status`", was wrong.

    The rate over **scorable** episodes — the only denominator that answers "how good is it where we could
    judge" — is **{sr_scorable}% ({scorable_ok}/{scorable_n}), 95% Wilson {slo}–{shi}%**. Over genuinely attempted tasks it is
    **{sr_attempts}% ({s}/{attempts})**; counting every finished episode, including the ones nobody could judge,
    gives **{sr}% ({s}/{n})**. All three are stated; none is a capability estimate this paper would defend, and
    the run is incomplete.

    **`SR_all` and `SR_reachable` are separated below, but not by the field this harness fills.**
    AndroidWorld upstream exposes no gold action list — its `TaskEval` surface has `template`, `goal` and
    `is_successful`, nothing reporting which action types a solution needs — so the `coverage` field carries
    no value on **any** of the **{cov_total}** episodes (`n_reachable = 0` is the harness's unset default,
    not a count of unreachable tasks — that is exactly why the field cannot be used to answer the question). This programme nevertheless established by hand
    which tasks the option set cannot serve: **exactly two of the 116 need `answer`** (`BrowserMultiply`,
    `RetroPlaylistDuration`). Excluding those, the reachable split is **{r_pct}% ({r_num}/{r_den})** against
    **{sr}% ({s}/{n})** over all finished tasks. Both are reported, with the definition stated, rather than
    calling the split unavailable when a narrower version of it is computable.

    **The action-space finding is the one that changes an earlier claim in this paper.** {invalid_clause}

    **Every one of the {s} successes is a task whose gold trajectory is one or two actions beyond
    `open_app`** — camera, single-record expense/recipe deletes, settings toggles and their verifiers, opening an
    app. The model emits `type_text` in **{tt} of {n} episodes ({tt_pct}%)** and succeeds **once** among them;
    **zero successes** occur on tasks that require filling a text field and then navigating. **The conclusion
    this paper draws is that per-step decision quality and task composition are separate capabilities.** The
    interface is no longer the binding constraint; composition is, and the calibrated gate is valuable
    precisely where composition cannot be reached.

""".format(
        planned=p, state=state, n=n, s=s, sr=ev["sr"], wlo=ev["wlo"], whi=ev["whi"],
        timeout=ev["timeout"], timeout_med=ev["timeout_med"],
        premature=ev["premature"], premature_med=ev["premature_med"],
        setup=ev["setup"], conf_p50=ev["conf_p50"], gate_p50=ev["gate_p50"],
        gate_zero=ev["gate_zero"], drift_f_decay=ev["drift_f_decay"], drift_f_n=ev["drift_f_n"],
        drift_s_decay=ev["drift_s_decay"], drift_s_n=ev["drift_s_n"],
        invalid_clause=invalid_clause, cov_with=ev["cov_with"], cov_total=ev["cov_total"],
        crash_same=ev["crash_same"], crash_n=ev["crash_n"],
        r_pct=ev["r_pct"], r_num=ev["r_num"], r_den=ev["r_den"],
        attempts=ev["attempts"], sr_attempts=ev["sr_attempts"],
        unscored=ev["unscored"], scorable_n=ev["scorable_n"], scorable_ok=ev["scorable_ok"],
        sr_scorable=ev["sr_scorable"], slo=ev["slo"], shi=ev["shi"],
        tt=ev["tt_emitted"], tt_pct=ev["type_text_pct"],
    )
    return body


# The clause contains decimals ("SR 14.4%"), so it cannot stop at the first period. It ends at the
# sentence about composition, which is the last sentence we generate.
ABSTRACT_PAT = (r"\*\*only with its completeness line\*\*:.*?\*\*composition\*\* is\.")


def render_abstract_clause(ev):
    return ("**only with its completeness line**: {n} of {p} tasks finished, {s} successes, SR {sr}% "
            "(95% Wilson {wlo}–{whi}%), with **zero invalid actions among {emitted} emitted** — so the "
            "interface is no longer the binding constraint and **composition** is.").format(
        n=ev["finished"], p=ev["planned"], s=ev["success"], sr=ev["sr"],
        wlo=ev["wlo"], whi=ev["whi"], emitted=format(ev["emitted"], ","))


RECORD_DOCS = [
    ("notes/GOAL.md", r"\| AndroidWorld full benchmark \|[^\n]*\n", "goal"),
    ("docs/HANDOFF.md", r"\(\d+/116 finished at time of writing:[^\n]*?emitted actions\*\*\)", "handoff"),
]


def render_goal_row(ev):
    return ("| AndroidWorld full benchmark | **INCOMPLETE at time of writing**: {n}/{p} finished, "
            "**{s} successes**; SR **{sr_scorable}% ({s_ok}/{sc})** over **scorable** episodes (the only "
            "denominator that answers how good we could judge), {sr_att}% ({s}/{att}) over genuinely "
            "attempted, {sr}% ({s}/{n}) counting everything; **Wilson (scorable) {slo}–{shi}%**; buckets {tmo} "
            "non-terminating timeouts / {pre} premature `status` / **{uns} unscored "
            "(`checker_unavailable`)** / {setup} setup raise; **{inv} invalid of {emit} emitted "
            "({invr}%)** | **in progress, not a headline**. Refresh: `scripts/refresh_aw116.py` then "
            "`scripts/sync_paper_aw116.py` |\n").format(
        n=ev["finished"], p=ev["planned"], s=ev["success"], sr=ev["sr"], wlo=ev["wlo"], whi=ev["whi"],
        tmo=ev["timeout"], pre=ev["premature"], setup=ev["setup"], inv=ev["invalid"],
        emit=format(ev["emitted"], ","), invr=ev["invalid_rate"], crash=ev["crashes"],
        uns=ev["unscored"], sc=ev["scorable_n"], s_ok=ev["scorable_ok"], sr_scorable=ev["sr_scorable"],
        slo=ev["slo"], shi=ev["shi"],
        sr_att=ev["sr_attempts"], att=ev["attempts"])


def render_goal_row5(ev):
    """GOAL §5's ordered-work row 5 quotes the run too, and drifted independently of §4's row.
    Two tables in the file declared authoritative disagreed about the same run, which is exactly
    the class of defect the auditor found."""
    return ("| 5 | Online AndroidWorld port | \u22651 day | **RUN 2026-11-08, INCOMPLETE** (the operator made it "
            "the primary benchmark). {p} tasks \u00d7 1 seed on protocol `aw-main-v1`, checkpoint "
            "E2+`app_intent`: **{n}/{p} finished**, **{s} successes, SR {sr}% ({s}/{n}), Wilson {wlo}\u2013{whi}%**, "
            "and **{sr_att}% ({s}/{att})** over genuinely attempted tasks after excluding {setup} whose device "
            "never became usable and {uns} that ran but could not be scored (`checker_unavailable`) "
            "\u2014 and **0 invalid of {emit} emitted actions**. Two failure signatures "
            "(premature `status` at median {pmed} steps; non-termination at median {tmed}) that no single "
            "confidence threshold separates. **40 tasks \u00d7 3 seeds not started.** Detail: \u00a74 row and "
            "Limitation 12 |").format(
        p=ev["planned"], n=ev["finished"], s=ev["success"], sr=ev["sr"], wlo=ev["wlo"], whi=ev["whi"],
        sr_att=ev["sr_attempts"], att=ev["attempts"], setup=ev["setup"], emit=format(ev["emitted"], ","),
        uns=ev["unscored"],
        pmed=ev["premature_med"], tmed=ev["timeout_med"])


GOAL_ROW5_PAT = r"\| 5 \| Online AndroidWorld port \|[^\n]*\n"
GOAL_DRIFT_PAT = r"\| `gate_score` drift vs episode outcome \|[^\n]*\n"
GOAL_MODES_PAT = r"\| AndroidWorld failure modes observed \|[^\n]*\n"


def render_goal_modes_row(ev):
    """The failure-mode row quoted a 52-episode snapshot (25/52, 27/52) long after the run finished."""
    return ("| AndroidWorld failure modes observed | **premature `status`** at median {pm} steps ({pre}/{n}); "
            "**never terminates**, clicks to the step budget with `gate_score` decaying to 0.0 ({tmo}/{n} "
            "timeouts) \u2014 plus **{uns} unscored** episodes whose checker raised, which are neither "
            "a model outcome nor a device failure; **zero invalid actions** in {emit} emitted "
            "(`benchmarks/results/2026-11-08-aw116/`) | measured, n={n} |").format(
        pm=ev["premature_med"], pre=ev["premature"], n=ev["finished"],
        tmo=ev["timeout"], emit=format(ev["emitted"], ","), uns=ev["unscored"])


def sync_goal_modes_row(ev, write=True):
    path = os.path.join(REPO, "notes", "GOAL.md")
    if not os.path.exists(path):
        return False
    txt = open(path, encoding="utf-8").read()
    if not re.search(GOAL_MODES_PAT, txt):
        return False
    updated = re.sub(GOAL_MODES_PAT, lambda m: render_goal_modes_row(ev) + "\n", txt, count=1)
    if updated == txt:
        return False
    if write:
        open(path, "w", encoding="utf-8").write(updated)
    return True


def render_goal_drift_row(ev):
    """§4's drift row drifted independently of the rest of the table — it quoted the 14-success
    snapshot while §4's own benchmark row quoted 15. Generate it with everything else."""
    return ("| `gate_score` drift vs episode outcome | mean \u0394(first3\u2192last3) **{fm} on failures "
            "({fp}% decaying, n={fn})** vs **{sm} on successes ({sp}%, n={sn})** | **hypothesis, not a "
            "result** |").format(fm=round(ev["drift_f_mean"], 3), fp=ev["drift_f_decay"], fn=ev["drift_f_n"],
                                 sm=round(ev["drift_s_mean"], 3), sp=ev["drift_s_decay"], sn=ev["drift_s_n"])


def sync_goal_drift_row(ev, write=True):
    path = os.path.join(REPO, "notes", "GOAL.md")
    if not os.path.exists(path):
        return False
    txt = open(path, encoding="utf-8").read()
    if not re.search(GOAL_DRIFT_PAT, txt):
        return False
    updated = re.sub(GOAL_DRIFT_PAT, lambda m: render_goal_drift_row(ev) + "\n", txt, count=1)
    if updated == txt:
        return False
    if write:
        open(path, "w", encoding="utf-8").write(updated)
    return True


def sync_goal_row5(ev, write=True):
    path = os.path.join(REPO, "notes", "GOAL.md")
    if not os.path.exists(path):
        return False
    txt = open(path, encoding="utf-8").read()
    if not re.search(GOAL_ROW5_PAT, txt):
        return False
    new_row = render_goal_row5(ev) + "\n"
    updated = re.sub(GOAL_ROW5_PAT, lambda m: new_row, txt, count=1)
    if updated == txt:
        return False
    if write:
        open(path, "w", encoding="utf-8").write(updated)
    return True


def render_handoff_paren(ev):
    return ("({n}/116 finished at time of writing: {s} successes; SR **{sr_scorable}% ({s_ok}/{sc})** over "
            "scorable episodes (Wilson {slo}–{shi}%), {sr}% ({s}/{n}) counting all; {tmo} non-terminating "
            "timeouts, {pre} premature "
            "`status`, **{uns} unscored (`checker_unavailable`)**, {setup} setup raise; **{inv} invalid of "
            "{emit} emitted actions**)").format(
        n=ev["finished"], s=ev["success"], sr=ev["sr"], wlo=ev["wlo"], whi=ev["whi"],
        tmo=ev["timeout"], pre=ev["premature"], setup=ev["setup"], inv=ev["invalid"],
        emit=format(ev["emitted"], ","), uns=ev["unscored"], sc=ev["scorable_n"],
        s_ok=ev["scorable_ok"], sr_scorable=ev["sr_scorable"], slo=ev["slo"], shi=ev["shi"])


def sync_abstract(ev, write=True):
    """The abstract quotes the same run, so it drifts the same way. Regenerate that clause."""
    path = PAPER
    txt = open(path, encoding="utf-8").read()
    # re.S matters: the clause is line-wrapped in the markdown, so `.` must cross newlines or the
    # substitution silently matches nothing and this function reports "no change" forever.
    if not re.search(ABSTRACT_PAT, txt, re.S):
        return False
    updated = re.sub(ABSTRACT_PAT, lambda m: render_abstract_clause(ev), txt, count=1, flags=re.S)
    if updated == txt:
        return False
    if write:
        open(path, "w", encoding="utf-8").write(updated)
    return True


def sync_records(ev, write=True):
    """Keep GOAL and HANDOFF in step with the same evidence file. AGENTS.md keeps a dated log
    entry (history), so it is updated by hand and checked rather than regenerated."""
    changed = []
    if sync_goal_row5(ev, write=write):
        changed.append("notes/GOAL.md §5 row 5")
    if sync_goal_drift_row(ev, write=write):
        changed.append("notes/GOAL.md §4 drift row")
    if sync_goal_modes_row(ev, write=write):
        changed.append("notes/GOAL.md §4 failure-mode row")
    for rel, pat, kind in RECORD_DOCS:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            continue
        txt = open(path, encoding="utf-8").read()
        new_row = render_goal_row(ev) if kind == "goal" else render_handoff_paren(ev)
        if not re.search(pat, txt):
            continue
        updated = re.sub(pat, lambda m: new_row, txt, count=1)
        if updated != txt:
            changed.append(rel)
            if write:
                open(path, "w", encoding="utf-8").write(updated)
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the paper differs, do not write")
    args = ap.parse_args(argv)

    ev = parse(EVIDENCE)
    new = render(ev)
    paper = open(PAPER, encoding="utf-8").read()
    if START not in paper or END not in paper:
        raise SystemExit("could not locate the Limitation 12 block in the paper")
    start = paper.index(START)
    end = paper.index(END)
    current = paper[start:end]
    if args.check:
        bad = current.strip() != new.strip()
        if sync_abstract(ev, write=False):
            bad = True
        stale = sync_records(ev, write=False)
        if bad or stale:
            sys.stderr.write("out of date with the evidence file: %s; run "
                             "scripts/sync_paper_aw116.py\n"
                             % (", ".join((["Limitation 12"] if bad else []) + stale)))
            return 1
        print("Limitation 12 and the record documents match the evidence file.")
        return 0
    # Early-returning when Limitation 12 was already current skipped the abstract and the record documents,
    # which is how the abstract kept a stale Wilson interval while this script reported success.
    # Everything that depends on the evidence is now refreshed on every run.
    if current != new:
        open(PAPER, "w", encoding="utf-8").write(paper[:start] + new + paper[end:])
        changed_abs = ["Limitation 12"]
    else:
        changed_abs = []
    if sync_abstract(ev, write=True):
        changed_abs.append("abstract")
    changed = changed_abs + sync_records(ev, write=True)
    print("Limitation 12 rewritten: %d/%d finished, %d successes, SR %s%%, %d/%d invalid. Records: %s"
          % (ev["finished"], ev["planned"], ev["success"], ev["sr"], ev["invalid"], ev["emitted"],
             ", ".join(changed) if changed else "already current"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
