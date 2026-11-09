#!/usr/bin/env python3
"""Keep the record documents from drifting away from the artifacts.

This exists because the program has twice shipped a paper whose numbers no longer matched disk, and twice
had a record document contradict a sibling. Run it before any claim, any commit that touches a number, and
before requesting completion.

    python3 scripts/check_record.py          # exit 0 = consistent, 1 = drift

What it does NOT do: verify the numbers themselves (that is `benchmarks/train/verify_paper.py` and
`benchmarks/repro_manifest.py`). This checks that every *document* says the same thing the artifacts say.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

fails = []
passes = 0


def check(name, ok, detail=""):
    global passes
    if ok:
        passes += 1
        print("PASS  " + name)
    else:
        fails.append(name)
        print("FAIL  " + name + ("  :: " + detail if detail else ""))


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------ artifacts
PURE = "backup-2026-10-29/Model_Training_Server/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl"
SERVED = "backup-2026-11-06/offline/e2-pure-hist.jsonl"
GROUNDB = "backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl"
VAL = "benchmarks/results/2026-10-29-evals/E2-val.jsonl"
ARM_VAL = "benchmarks/results/2026-11-07-arm/eval-arm20-V-val.jsonl"
ARM_TEST = "benchmarks/results/2026-11-07-arm/eval-arm20test-V.jsonl"


def score(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return sum(1 for r in rows if r["correct"]), len(rows)


has = {p: os.path.exists(p) for p in (PURE, SERVED, GROUNDB, VAL, ARM_VAL, ARM_TEST)}
if all(has.values()):
    pure, pure_n = score(PURE)
    served, served_n = score(SERVED)
    gb, gb_n = score(GROUNDB)
    arm_v, arm_vn = score(ARM_VAL)
    arm_t, arm_tn = score(ARM_TEST)

    # The headline must be what the artifact says, wherever it is quoted.
    expected = {
        "64.07": pure,          # pure in-process
        "63.82": served,        # pure served
        "64.82": gb,            # groundB
        "63.85": arm_v,         # stage-2 arm, val
        "60.80": arm_t,         # stage-2 arm, test
    }
    paper = read("papers/single-pass-decision-acv2.md")
    goal = read("notes/GOAL.md")
    handoff = read("docs/HANDOFF.md")
    agents = read("AGENTS.md")

    check("pure artifact is 255/398", pure == 255 and pure_n == 398, f"{pure}/{pure_n}")
    check("served artifact is 254/398", served == 254 and served_n == 398, f"{served}/{served_n}")
    check("groundB artifact is 258/398", gb == 258 and gb_n == 398, f"{gb}/{gb_n}")
    check("arm val is 528/827", arm_v == 528 and arm_vn == 827, f"{arm_v}/{arm_vn}")
    check("arm test is 242/398", arm_t == 242 and arm_tn == 398, f"{arm_t}/{arm_tn}")

    # No document may still present the retracted headline as a live claim.
    retired = re.compile(r"64\.32")
    allowed = ("corrected", "correction", "superseded", "no artifact", "exists in no", "retract",
               "retired", "pre-audit", "before the 2026-11-07", "written as", "at the time",
               "does not exist", "audit", "said 64.32", "Bản gốc", "resolved 2026-11-07")
    for name, text in (("paper", paper), ("GOAL", goal), ("HANDOFF", handoff), ("AGENTS", agents)):
        bad = []
        for m in retired.finditer(text):
            window = text[max(0, m.start() - 160): m.start() + 200].lower()
            if not any(a.lower() in window for a in allowed):
                bad.append(text[max(0, m.start() - 60): m.start() + 40].replace("\n", " "))
        check(f"{name}: no live 64.32 claim", not bad, str(bad[:2]))

    # Section numbering is the paper's, and every pointer in every record doc must match it.
    # (Auditor 2026-11-08: the §5.8 insertion shifted the cost window to §5.12 and the pointers were stale.
    # This block existed but encoded the OLD mapping, so it could not catch its own drift.)
    SECTIONS = {
        "5.8": "Why grounding plateaus",
        "5.9": "A training-procedure lever",
        "5.10": "The type bucket",
        "5.11": "Reading the same decision under CE",
        "5.12": "What the gate is worth",
        "5.13": "The protocol that would upgrade",
    }
    for num, title in SECTIONS.items():
        check(f"paper §{num} = {title}", f"### {num} {title}" in paper)

    # Each pointer claim must point at the section that actually says that thing.
    POINTER_SUBJECT = {
        "cost window": "12",
        "cost-window": "12",
        "type bucket": "10",
        "re-weighting": "10",
        "negative arm": "10",
        "stage-2 arm": "10",
        "CE, Brier": "11",
        "CE/Brier": "11",
    }
    for name, text in (("AGENTS", agents), ("HANDOFF", handoff), ("GOAL", goal)):
        bad = []
        for line in text.splitlines():
            # Only judge a pointer against the subject named on ITS OWN line: a +/-200-char window
            # reaches into the next paragraph's subject and produces false alarms (found 2026-11-08).
            # Pair each pointer with the subject NEAREST to it on the same line. A line may correctly
            # name two subjects and two pointers ("the stage-2 arm (§5.10) and the cost window (§5.12)"),
            # so a whole-line match is wrong (found 2026-11-08: it produced a false alarm on GOAL).
            ptrs = [(m.start(), m.group(1)) for m in re.finditer(r"§5\.(\d+)", line)]
            low = line.lower()
            subs = [(low.find(ph.lower()), ph, want) for ph, want in POINTER_SUBJECT.items()]
            for pos, num in ptrs:
                # A subject only governs pointers within ~70 chars; beyond that the line is naming a
                # different thing (GOAL legitimately lists the arm (§5.10) then the window (§5.12)).
                cands = [(abs(pos - sp), ph, want) for sp, ph, want in subs
                         if sp != -1 and abs(pos - sp) <= 70]
                if not cands:
                    continue
                _, ph, want = min(cands)
                if num != want:
                    bad.append(f"{ph!r} -> §5.{num}, expected §5.{want}: {line.strip()[:70]}")
        check(f"{name}: section pointers match subjects", not bad, str(bad[:3]))

    # §5.7 reporting obligations
    check("paper 5.7 has a Timeout column", "Timeout | Model p50" in paper)
    check("paper 5.7 states the reachable split is undefined", "reachable split is undefined" in paper)

    # Online story must agree everywhere.
    check("GOAL online is 1/8 in both", "**1/8 in both conditions**" in goal)
    check("HANDOFF does not call the harness unbuilt", "Not built yet" not in handoff)
    # The intent is "HANDOFF states a test count", not "it states a particular count" — pinning the
    # literal made the check fail the moment the suite grew, which is the drift this file exists to
    # prevent, not to re-introduce. The specific number is verified against the live suite below.
    check("HANDOFF states a test count", re.search(r"\d+\s+passed", handoff) is not None)

    # Paper precedence must be unambiguous.
    check("GOAL names single-pass as the authority",
          "the audit-corrected version, and the authority" in goal)
    check("GOAL marks the older draft subordinate", "subordinate" in goal)

    # Stage-2 arm is a recorded negative, everywhere it is mentioned.
    check("HANDOFF reports the arm", "63.85" in handoff and "60.80" in handoff)
    check("paper reports the arm as a failure", "**It fails, and the" in paper or "It fails" in paper)

    # Files whose deltas the pre-registered seed rule withdrew must say so, in the file itself.
    for f in ("benchmarks/results/2026-11-07-best-recipe.md",
              "benchmarks/results/2026-11-07-grounding-recipe-result.md",
              "benchmarks/results/2026-11-07-aux7k-is-data-volume.md"):
        if os.path.exists(f):
            check(f"withdrawn deltas are bannered: {os.path.basename(f)}",
                  "RETRACTED DELTAS" in read(f))

    # Arithmetic: any "(N/398)" next to a percent must compute to that percent.
    bad_arith = []
    for f in ("benchmarks/results/2026-11-02-d1-acv2-groundB.md",
              "benchmarks/results/2026-10-29-d1-acv2-sft.md", "papers/single-pass-decision-acv2.md"):
        if not os.path.exists(f):
            continue
        for i, line in enumerate(read(f).splitlines(), 1):
            for m in re.finditer(r"(\d{2}\.\d{2})%\s*\((\d+)/398\)", line):
                pct, num = float(m.group(1)), int(m.group(2))
                if abs(round(num / 398 * 100, 2) - pct) > 0.02:
                    bad_arith.append(f"{os.path.basename(f)}:{i} {pct}% vs {num}/398={round(num/398*100,2)}")
    check("no arithmetic-contradictory N/398 percentages", not bad_arith, str(bad_arith[:3]))

    # Same for the 223 spatial denominator: 56.05 = 125/223 and 56.50 = 126/223 are DIFFERENT checkpoints,
    # and confusing them was one of the auditor's findings. The pair must stay paired.
    bad_gr = []
    for f in ("benchmarks/results/2026-11-02-d1-acv2-groundB.md",
              "benchmarks/results/2026-10-29-d1-acv2-sft.md",
              "papers/single-pass-decision-acv2.md", "docs/HANDOFF.md", "notes/GOAL.md"):
        if not os.path.exists(f):
            continue
        for i, line in enumerate(read(f).splitlines(), 1):
            for m in re.finditer(r"(\d{2}\.\d{2})%\s*\((\d+)/223\)", line):
                pct, num = float(m.group(1)), int(m.group(2))
                if abs(round(num / 223 * 100, 2) - pct) > 0.02:
                    bad_gr.append(f"{os.path.basename(f)}:{i} {pct}% vs {num}/223={round(num/223*100,2)}")
    check("no arithmetic-contradictory N/223 percentages", not bad_gr, str(bad_gr[:3]))

    # The @938 grounding must never be quoted as 56.50 (that is step 705) without saying so on the same line.
    bad_ck = []
    for f in ("docs/HANDOFF.md", "notes/GOAL.md", "papers/single-pass-decision-acv2.md"):
        if not os.path.exists(f):
            continue
        for i, line in enumerate(read(f).splitlines(), 1):
            if "56.50" in line and not re.search(r"705|step.?705", line, re.I):
                bad_ck.append(f"{os.path.basename(f)}:{i}: 56.50 quoted without a checkpoint qualifier")
    check("56.50 always carries its step-705 qualifier", not bad_ck, str(bad_ck[:3]))

    # §5.9 is the only surviving positive claim, so it must be re-derived from artifacts, not trusted.
    # (Added 2026-11-08: the paper said +2.69 / t=3.34 / df=4 while its own table's n implied df=5 and the
    # artifacts give +2.62 / t=3.54 / df=5. Means matched; the test statistic did not.)
    import subprocess
    seed_script = "benchmarks/results/2026-11-08_seed_table.py"
    if os.path.exists(seed_script):
        r = subprocess.run([sys.executable, seed_script], capture_output=True, text=True)
        out = r.stdout + r.stderr
        check("§5.9 seed table re-derives from artifacts", r.returncode == 0, out.strip()[-160:])
        check("§5.9 means reproduce", "MEANS REPRODUCE: YES" in out, out.strip()[-160:])
        check("§5.9 diff/t/df reproduce", "DIFF/t/df REPRODUCE: YES" in out, out.strip()[-160:])
        # §5.9 must report all three defensible tests and must NOT claim significance: the variance ratio
        # is 8.3x, so the pooled Student test's assumption is violated and Welch gives p = 0.063.
        check("paper §5.9 reports the Welch test", "Welch" in paper)
        check("paper §5.9 reports the permutation test", "permutation" in paper)
        check("paper §5.9 does not claim significance",
              "We therefore do not claim significance" in paper)
        check("paper §5.9 names the variance-ratio problem", "8.3" in paper)

    # §5.7's online numbers are the last objective item-2 claim without a mechanical check.
    # (Added 2026-11-08: 1/8 in both conditions, Wilson 2.2-47.1%, invalid 20/71 vs 10/68.)
    online_script = "benchmarks/results/2026-11-08_online_sr.py"
    if os.path.exists(online_script):
        r = subprocess.run([sys.executable, online_script], capture_output=True, text=True)
        out = r.stdout + r.stderr
        check("§5.7 online SR re-derives from episode JSONLs", r.returncode == 0, out.strip()[-200:])
        check("§5.7 online SR is 1/8 in both conditions", "SR REPRODUCES: YES" in out)
        check("§5.7 Wilson interval reproduces", "2.2-47.1" in out)
        check("§5.7 invalid-action counts reproduce", "20/71 vs 10/68" in out)
        check("paper §5.7 states the unmatched-instances caveat",
              "different task instances" in paper or "unmatched" in paper)

    # §5.2's raw-confidence ECE/AUROC come from the @705 and @938 leaderboard artifacts, NOT from the
    # pure E2-nomem run (which gives 0.201/0.642). Quoting the wrong artifact is an easy, silent error.
    def _ece_auroc(path):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        pts = [(r.get("conf"), 1 if r.get("correct") else 0)
               for r in rows if isinstance(r.get("conf"), (int, float))]
        c = [x[0] for x in pts]; y = [x[1] for x in pts]; n = len(c)
        e = 0.0
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            idx = [k for k in range(n) if (lo < c[k] <= hi) or (i == 0 and c[k] == 0)]
            if idx:
                e += len(idx) / n * abs(sum(c[k] for k in idx) / len(idx)
                                        - sum(y[k] for k in idx) / len(idx))
        order = sorted(range(n), key=lambda k: c[k]); ranks = [0.0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(c[order[j + 1]] - c[order[i]]) < 1e-12:
                j += 1
            avg = (i + 1 + j + 1) / 2
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        n1 = sum(y); n0 = n - n1
        s1 = sum(ranks[k] for k in range(n) if y[k] == 1)
        return e, (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)

    E938 = "benchmarks/results/2026-10-29-evals/E2-938-V.jsonl"
    E705 = "benchmarks/results/2026-10-29-evals/E2-705-V.jsonl"
    if os.path.exists(E938) and os.path.exists(E705):
        e938, a938 = _ece_auroc(E938)
        e705, a705 = _ece_auroc(E705)
        check("§5.2 ECE @938 = 0.204", abs(e938 - 0.204) < 0.002, f"{e938:.4f}")
        check("§5.2 ECE @705 = 0.231", abs(e705 - 0.231) < 0.002, f"{e705:.4f}")
        check("§5.2 AUROC @938 = 0.631", abs(a938 - 0.631) < 0.002, f"{a938:.4f}")
        check("§5.2 AUROC @705 = 0.626", abs(a705 - 0.626) < 0.002, f"{a705:.4f}")

    # The pre-registration must record the executed-recipe deviation.
    prereg = read("docs/research/PRE-REG-2026-11-07-status-downweight.md")
    check("pre-reg records the micro/accum deviation", "Execution deviation" in prereg)

    # Limitation 12 is GENERATED from the evidence file. Re-render it and require byte equality, so a
    # hand-edit cannot survive and the numbers cannot drift from the run they describe.
    ev_path = "benchmarks/results/2026-11-08-aw116/evidence-82tasks.txt"
    sync = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_paper_aw116.py")
    if os.path.exists(ev_path) and os.path.exists(sync):
        import subprocess
        r = subprocess.run([sys.executable, sync, "--check"], capture_output=True, text=True)
        check("Limitation 12 is generated from the evidence file (no hand edits)",
              r.returncode == 0, (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else "")

    # A test count quoted as current must be the real count. Three documents claimed 110/123 while the
    # suite was at 132, which is the same drift class the auditor found in HANDOFF.
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       cwd=os.path.join(ROOT, "benchmarks", "aw"),
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "benchmarks", "aw")})
    m = re.search(r"(\d+) passed", r.stdout)
    actual = int(m.group(1)) if m else -1
    check("the test suite runs", actual > 0, str(actual))
    for doc in ("AGENTS.md", "docs/HANDOFF.md", "notes/GOAL.md"):
        body = read(doc)
        # Any count presented as current (i.e. not inside an "as of"/dated-history clause) must match.
        for cm in re.finditer(r"(\d+)\s+passed(?:,?\s*/?\s*\d+\s*skipped)?", body):
            # The qualifier can sit on either side of the number, so look at a window around it.
            ctx = body[max(0, cm.start() - 260):cm.end() + 260].lower()
            historical = ("as of" in ctx or "when this list was written" in ctx
                          or "2026-11-0" in ctx or "current:" in ctx
                          or " then" in ctx or "was " in ctx or "now 13" in ctx)
            if historical:
                continue
            check("%s test count %s matches the suite" % (doc, cm.group(1)),
                  int(cm.group(1)) == actual, "doc=%s actual=%d" % (cm.group(1), actual))

    # The Gates section must describe what it actually is. It once claimed dated entries carry no
    # counts while they visibly do, which is the kind of small false statement that makes a reader
    # distrust the block above it.
    agents_body = read("AGENTS.md")
    if "GATES:END" in agents_body:
        # Normalise whitespace: the prose is line-wrapped, so a phrase can straddle a newline.
        tail = re.sub(r"\s+", " ", agents_body.split("GATES:END")[1][:900])
        check("AGENTS §3.5 describes dated entries honestly",
              "describe what was true" in tail and "not about today" in tail)
        check("AGENTS §3.5 does not claim dated entries carry no counts",
              "carry **no counts**" not in tail)

    # §5.12.1 is arithmetic on published numbers, which is exactly the kind of claim that survives
    # review when nobody re-runs it. Re-run it from the captured primary table.
    import subprocess
    vca = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_compound_accuracy.py")
    if os.path.exists(vca):
        r = subprocess.run([sys.executable, vca], capture_output=True, text=True)
        check("§5.12.1 re-derives from the captured MAI-UI table", r.returncode == 0,
              (r.stdout.strip().splitlines() or [""])[-1])

    # --- the six defects the third-round auditor found, each now checked -------------------
    _ev_txt = open(ev_path, encoding="utf-8").read() if os.path.exists(ev_path) else ""
    buckets_tmo = (re.search(r"'timeout_non_terminating': (\d+)", _ev_txt) or
                   re.search(r"TIMEOUT_EPISODES=(\d+)", _ev_txt))
    buckets_tmo = buckets_tmo.group(1) if buckets_tmo else "?"
    buckets_pre = (re.search(r"'premature_status': (\d+)", _ev_txt) or
                   re.search(r"PREMATURE=(\d+)", _ev_txt))
    buckets_pre = buckets_pre.group(1) if buckets_pre else "?"

    _agents = read("AGENTS.md")
    _handoff = read("docs/HANDOFF.md")
    _goal = read("notes/GOAL.md")

    # 1. AGENTS' newest entry must carry the unscored bucket, not the pre-fix split.
    _newest = _agents[:_agents.index("- **2026-11-08 (2nd)")] if "- **2026-11-08 (2nd)" in _agents else _agents
    check("AGENTS newest entry names the unscored bucket", "unscored" in _newest)
    # The entry also *quotes* the retracted claim in order to correct it, so strip that quotation
    # before asserting the claim does not appear as an assertion.
    _newest_unquoted = re.sub(r'"[^"]*"', "", _newest)
    check("AGENTS newest entry does not assert a single device defect",
          "not a second problem" not in _newest_unquoted)

    # 2. The HANDOFF parenthetical must appear exactly once and be well-formed. It was concatenated
    #    three times because the sync regex matched the replacement it had just written.
    check("HANDOFF parenthetical appears exactly once",
          _handoff.count("/116 finished at time of writing:") == 1,
          str(_handoff.count("/116 finished at time of writing:")))
    check("HANDOFF has no duplicated 'over scorable episodes' fragment",
          _handoff.count("over scorable episodes") == 1,
          str(_handoff.count("over scorable episodes")))
    check("HANDOFF has no stray '**)' run-on",
          "actions**)**" not in _handoff)

    # 3. No document may emit the dangling section reference; it is Limitation 12.
    for _doc, _body in (("papers/single-pass-decision-acv2.md", read("papers/single-pass-decision-acv2.md")),
                        ("notes/GOAL.md", _goal), ("docs/HANDOFF.md", _handoff), ("AGENTS.md", _agents)):
        check("%s has no dangling section 6.12" % _doc, "\u00a76.12" not in _body)

    # 4. The aw116 note's FINAL STATE table must agree with the generated buckets.
    _note = read("notes/2026-11-08-aw116-first-results.md")
    _mbt = re.search(r"\| non-terminating timeout \| \*\*(\d+)\*\*", _note)
    _mbp = re.search(r"\| premature `status` \| \*\*(\d+)\*\*", _note)
    if _mbt and _mbp:
        check("aw116 note timeout bucket matches the evidence", _mbt.group(1) == buckets_tmo,
              "%s vs %s" % (_mbt.group(1), buckets_tmo))
        check("aw116 note premature bucket matches the evidence", _mbp.group(1) == buckets_pre,
              "%s vs %s" % (_mbp.group(1), buckets_pre))

    # 5. The scorable denominator must carry a Wilson interval wherever the rate is quoted.
    _msw2 = re.search(r"WILSON_SCORABLE=([\d.]+)-([\d.]+)", _ev_txt)
    if _msw2:
        _iv = "%s\u2013%s%%" % (_msw2.group(1), _msw2.group(2))
        check("paper gives the scorable Wilson interval", _iv in read("papers/single-pass-decision-acv2.md"))
        check("GOAL gives the scorable Wilson interval", _iv in _goal)
        check("HANDOFF gives the scorable Wilson interval", _iv in _handoff)
        check("GOAL does not print an empty Wilson for the scorable rate",
              "Wilson (scorable) \u2014" not in _goal)

    # 6. A dated snapshot whose categories do not sum must say so rather than look like arithmetic.
    check("AGENTS' 82/116 snapshot either sums to 82 or says it does not",
          ("do not sum to 82" in _agents) or ("8+39+37" not in _agents))

    # "0 invalid of 1,394 emitted" must not be an artefact of a missing field. Assert the field is
    # present on every step row and that the count is a real 0 over a real denominator.
    _eps = os.path.join(ROOT, "benchmarks", "results", "2026-11-08-aw116", "episodes", "episodes")
    if os.path.isdir(_eps):
        _rows = 0
        _with_field = 0
        _inv_true = 0
        for _f in glob.glob(os.path.join(_eps, "*.jsonl")):
            for _l in open(_f, encoding="utf-8"):
                if not _l.strip():
                    continue
                _r = json.loads(_l)
                if _r.get("type") != "step":
                    continue
                _rows += 1
                if "invalid_action" in _r:
                    _with_field += 1
                if _r.get("invalid_action"):
                    _inv_true += 1
        check("every step row carries invalid_action (0 is real, not missing)",
              _rows > 0 and _with_field == _rows, "%d/%d" % (_with_field, _rows))
        check("the invalid-action count is genuinely zero", _inv_true == 0, str(_inv_true))

    # The unscored bucket is a second, independent defect and must be visible everywhere the run is
    # described. It was previously folded into timeout/premature — a harness that turns an unscorable
    # episode into a model failure produces a number that looks conservative and is simply wrong.
    # This block sits outside the evidence `if`, so it reads the file itself.
    mau = re.search(r"UNSCORED_CHECKER_UNAVAILABLE=(\d+) of (\d+)", _ev_txt)
    if mau:
        n_uns = mau.group(1)
        for doc in ("papers/single-pass-decision-acv2.md", "notes/GOAL.md", "docs/HANDOFF.md"):
            db = read(doc)
            check("%s names the unscored bucket" % doc, "unscored" in db, doc)
            check("%s states the unscored count" % doc, n_uns in db, n_uns)
        msc = re.search(r"SR_SCORABLE=([\d.]+)%", _ev_txt)
        if msc:
            for doc in ("papers/single-pass-decision-acv2.md", "notes/GOAL.md", "docs/HANDOFF.md"):
                check("%s reports the scorable-episode SR" % doc,
                      ("%s%%" % msc.group(1)) in read(doc), msc.group(1))
    notes_aw2 = read("notes/2026-11-08-aw116-first-results.md")
    check("aw116 note states there are two defects, not one",
          "second, independent defect" in notes_aw2)
    check("aw116 note names the get_state root cause",
          "does not forward `get_state`" in notes_aw2 or "did not forward `get_state`" in notes_aw2)

    # The 12 tasks that never ran are an infrastructure limit with a measured cause; the notes must
    # say so, or a future session will read "104/116" as missing work rather than a degraded device.
    notes_aw = read("notes/2026-11-08-aw116-first-results.md")
    check("aw116 notes name the accessibility-forwarder cause",
          "accessibility forwarder degrades" in notes_aw)
    check("aw116 notes state the suite's uptime requirement",
          "2.5 hours of continuous emulator uptime" in notes_aw)
    check("aw116 notes give the operational fix (chunked restarts)",
          "chunking the suite into resumable segments" in notes_aw)

    # The provenance header must state the number of sources the brief actually contains, and the
    # gate count it quotes must be this script's own count. Both drifted (36 vs 39; 84 vs 120).
    brief_body = read("outputs/calibrated-decision-value.md")
    src_sec = brief_body[brief_body.index("## Sources"):]
    src_sec = src_sec[:src_sec.index("## Revision record")] if "## Revision record" in src_sec else src_sec
    n_sources = len(re.findall(r"^(\d+)\.\s", src_sec, re.M))
    prov = read("outputs/calibrated-decision-value.provenance.md")
    m_prov = re.search(r"Sources consulted:\*\*\s*(\d+) numbered sources", prov)
    check("provenance states the real source count",
          m_prov is not None and int(m_prov.group(1)) == n_sources,
          "prov=%s actual=%d" % (m_prov.group(1) if m_prov else "?", n_sources))
    check("no source-numbering gaps",
          n_sources == max(int(x) for x in re.findall(r"^(\d+)\.\s", src_sec, re.M)),
          str(n_sources))
    # Derive the external/internal split from the Sources list instead of trusting the header, and
    # require the header to state it. The old header said "36 … 23 external / 13 internal" for a list
    # of 39 whose actual split is 31/8 — three inconsistent statements of one fact (found 2026-11-09).
    entries = re.findall(r"^(\d+)\.\s+(.*?)(?=\n\d+\.\s|\Z)", src_sec, re.M | re.S)
    internal_nums = []
    for num, body in entries:
        b = re.sub(r"\s+", " ", body)
        if re.search(r"`(?:notes|benchmarks|docs|outputs|backup|references|scripts|papers)/", b) or \
           re.search(r"(?:I|i)nternal note|artifact:", b):
            internal_nums.append(int(num))
    ext = len(entries) - len(internal_nums)
    check("provenance states the real external/internal split",
          ("%d external, %d internal" % (ext, len(internal_nums))) in prov,
          "derived %d/%d" % (ext, len(internal_nums)))
    check("provenance acceptance count equals the source count",
          ("%d/%d after verification" % (n_sources, n_sources)) in prov, str(n_sources))
    # A source listed but never cited in the body is provenance theatre, not provenance.
    body_only = brief_body[:brief_body.index("## Sources")]
    uncited = [n for n, _ in entries if ("[%s]" % n) not in body_only]
    check("every numbered source is cited in the body", not uncited, "uncited=%s" % uncited)

    # The value brief's §5.1 must carry the same per-step arithmetic as the paper, from the same
    # primary source, or the two documents can disagree about what the program's gap is.
    brief = read("outputs/calibrated-decision-value.md")
    check("value brief §5.1 exists", "### 5.1 The 2026-11-08 measurement" in brief)
    for v in ("92.3%", "45.1%", "49.1%", "97.4%", "+0.8 per-step"):
        check(f"value brief §5.1 states {v}", v in brief)
    check("value brief distinguishes SFT from RL",
          "SFT only" in brief and "after online RL" in brief)
    check("value brief §5.1 cites the MAI-UI primary source", "[37]" in brief)
    check("value brief has a source entry for [37]",
          "37. MAI-UI Technical Report" in brief)
    check("value brief has a source entry for the derived arithmetic",
          "38. Derived arithmetic" in brief)
    check("value brief warns the aw116 run is incomplete",
          "incomplete" in brief.split("### 5.1")[1][:4000])

    # The app_intent guard: the paper's read-slot numbers must equal the saved summary.
    gp = "benchmarks/results/2026-11-08-appint/eval-guard_appint_full-V.summary.json"
    if os.path.exists(gp):
        g = json.load(open(gp, encoding="utf-8"))
        paper = read("papers/single-pass-decision-acv2.md")
        check("§5.7 read-slot guard step accuracy = 63.07%",
              abs(g["step_acc"] - 0.6307) < 0.0001, f'{g["step_acc"]:.4f}')
        check("§5.7 read-slot guard grounding = 66.49%",
              abs(g["grounding"] - 0.6649) < 0.0001, f'{g["grounding"]:.4f}')
        check("paper prints the guard step accuracy",
              ("%.2f%%" % (100 * g["step_acc"])) in paper, "%.2f" % (100 * g["step_acc"]))
        check("paper prints the guard grounding",
              ("%.2f%%" % (100 * g["grounding"])) in paper, "%.2f" % (100 * g["grounding"]))
        check("paper states mixing is mandatory (catastrophic forgetting)",
              "mixing is mandatory" in paper)
        # Every row of the read-slot table must match its artifact.
        import glob as _glob
        import json as _json
        for fp in sorted(_glob.glob("benchmarks/results/2026-11-08-appint/*.summary.json")):
            d = _json.load(open(fp, encoding="utf-8"))
            tag = d.get("tag", "")
            check("read-slot %s step acc printed" % tag,
                  ("%.2f%%" % (100 * d["step_acc"])) in paper, "%.2f" % (100 * d["step_acc"]))
            check("read-slot %s grounding printed" % tag,
                  ("%.2f%%" % (100 * d["grounding"])) in paper, "%.2f" % (100 * d["grounding"]))

    # The 116-task run evidence must be internally consistent with what the paper prints.
    ev = "benchmarks/results/2026-11-08-aw116/evidence-82tasks.txt"
    if os.path.exists(ev):
        txt = open(ev, encoding="utf-8").read()
        m = re.search(r"TASKS_FINISHED=(\d+)(?: PLANNED=\d+)? SUCCESS=(\d+) SR=([\d.]+)%", txt)
        n_fin, n_succ = int(m.group(1)), int(m.group(2))
        check("aw116 evidence: success <= finished", n_succ <= n_fin, f"{n_succ} <= {n_fin}")
        check("aw116 evidence: SR matches the counts",
              abs(float(m.group(3)) - 100.0 * n_succ / n_fin) < 0.15, m.group(3))
        # The paper must not print a completeness-free number.
        paper = read("papers/single-pass-decision-acv2.md")
        check("Limitation 12 states the completeness of the 116-task run",
              "incomplete at the time of writing" in paper)
        check("Limitation 12 says the SR is over tasks that ran, not the suite",
              "over the tasks that ran, not the suite" in paper)
        # Collapse whitespace: the paper wraps lines, so "non-terminating\ntimeouts" must still match.
        # The whole Limitation 12 block, not a fixed prefix: a clause appended to Limitation 12 would
        # otherwise fall outside the window and the check would fail for the wrong reason.
        sec = re.sub(r"\s+", " ", paper.split("12. **Our own online")[1].split("## 7. Conclusion")[0])
        check("Limitation 12 gives the Wilson interval", "Wilson" in sec)
        for bucket in ("non-terminating timeouts", "premature `status`", "initialize_task"):
            check(f"Limitation 12 names the {bucket} bucket", bucket in sec)
        # The objective requires invalid/crash/timeout with their own denominators, never folded into SR.
        mi = re.search(r"EMITTED_ACTIONS=(\d+) INVALID=(\d+) INVALID_RATE=([\d.]+)%", txt)
        if mi:
            check("Limitation 12 states the emitted-action denominator",
                  ("%s emitted" % format(int(mi.group(1)), ",")) in sec or mi.group(1) in sec,
                  mi.group(1))
            check("Limitation 12 states zero invalid actions when the evidence says zero",
                  (mi.group(2) == "0") == ("zero were invalid" in sec), mi.group(2))
            check("Limitation 12 states the invalid rate", mi.group(3) in sec, mi.group(3))
        mc = re.search(r"ENV_CRASH_EPISODES=(\d+)/(\d+)", txt)
        mcs = re.search(r"ENV_CRASH_IS_SETUP_FAILURE=(\d+)/(\d+) setup_errors=(\d+)", txt)
        if mc and mcs:
            # The previous version of this check was tautological: `... or ("environment crash" in sec)`
            # passed on any occurrence of the phrase, so it could never fail. What the paper must
            # actually say is that the crashes ARE the setup failures, with the right counts.
            check("Limitation 12 states the crash/setup identity with counts",
                  ("%s of %s crashes carry a setup error" % (mcs.group(1), mcs.group(2))) in sec,
                  "%s/%s" % (mcs.group(1), mcs.group(2)))
            # The claim changed: the crashes ARE the setup failures, but the 12 unscored episodes are
            # a SECOND, disjoint defect. What must hold is that neither is folded into a model bucket.
            check("Limitation 12 separates unscored from model outcomes",
                  "ran to completion but were never judged" in sec)
            check("Limitation 12 does not fold unscored into timeout/premature",
                  "folded the unscored episodes" in sec)
        # Every number the paper prints must equal the evidence file, so a refresh cannot drift
        # away from the prose. This is the check that would have caught 64.32 in 2026-11-07.
        check("Limitation 12 finished count matches the evidence",
              ("%d of 116 tasks finished" % n_fin) in sec, f"evidence={n_fin}")
        check("Limitation 12 success count matches the evidence",
              ("%d successes" % n_succ) in sec, f"evidence={n_succ}")
        check("Limitation 12 SR matches the evidence",
              ("SR %.1f%%" % float(m.group(3))) in sec, m.group(3))
        # A degraded emulator must not be readable as a model failure: both denominators must be
        # in the paper, and the genuine-attempt one must match the evidence.
        mg = re.search(r"GENUINE_ATTEMPTS=(\d+) SUCCESS=(\d+) SR_OVER_ATTEMPTS=([\d.]+)%", txt)
        if mg:
            check("Limitation 12 reports the genuine-attempt denominator",
                  ("%s/%s" % (mg.group(2), mg.group(1))) in sec, "%s/%s" % (mg.group(2), mg.group(1)))
            check("Limitation 12 reports the SR over genuine attempts",
                  ("%s%%" % mg.group(3)) in sec, mg.group(3))
            check("Limitation 12 explains the setup failures (a11y forwarder)",
                  "accessibility forwarder" in sec)
        # The objective asks for SR_all vs SR_reachable explicitly, so the answer must be explicit.
        mev = re.search(r"SR_REACHABLE=(\w+) episodes_with_coverage=(\d+)/(\d+)", txt)
        if mev:
            check("aw116 evidence answers the reachable split",
                  mev.group(1) == "UNDEFINED" or mev.group(1).isdigit(), mev.group(1))
            check("Limitation 12 explains why the reachable split is undefined",
                  "exposes no gold action list" in sec)
            check("Limitation 12 states how many episodes carry coverage",
                  ("%s of %s" % (mev.group(2), mev.group(3))) in sec,
                  "%s/%s" % (mev.group(2), mev.group(3)))
        mb = re.search(r"BUCKETS (\{.*?\})", txt)
        if mb:
            b = eval(mb.group(1), {"__builtins__": {}}, {})  # the file is written by our own script
            for key, label in (("timeout_non_terminating", "non-terminating"),
                               ("premature_status", "premature"),
                               ("unscored_checker_error", "unscored")):
                if key in b:
                    shown = {"non-terminating": "non-terminating timeouts",
                             "premature": "premature `status`",
                             "unscored": "unscored"}[label]
                    check(f"Limitation 12 {label} count matches the evidence",
                          ("%d %s" % (b[key], shown)) in sec, str(b[key]))

        # A number can be stale in a document the generated-block checks never look at. The auditor
        # found exactly that: GOAL §5 row 5 and a paper cross-reference both carried a superseded
        # emitted-action count while check_record was green. Assert the run's load-bearing numbers
        # EVERYWHERE they appear in live prose, not only inside the blocks the sync script rewrites.
        mei = re.search(r"EMITTED_ACTIONS=(\d+) INVALID=(\d+)", txt)
        if mei:
            emitted_truth = mei.group(1)
            live_docs = ["papers/single-pass-decision-acv2.md", "notes/GOAL.md", "docs/HANDOFF.md",
                         "AGENTS.md", "outputs/calibrated-decision-value.md",
                         "outputs/calibrated-decision-value.provenance.md"]
            for doc in live_docs:
                body = read(doc)
                for cm in re.finditer(r"([\d,]{3,})\s+(?:actions|emitted)", body):
                    n = cm.group(1).replace(",", "")
                    check("%s emitted-action count %s is the artifact's" % (doc, cm.group(1)),
                          n == emitted_truth, "truth=%s" % emitted_truth)
                for stale in ("1,344", "1344 "):
                    if stale in body:
                        idx = body.index(stale)
                        ctx = re.sub(r"\s+", " ", body[max(0, idx - 200):idx + 200]).lower()
                        historical = ("2026-11-08" in ctx or "as of" in ctx or "superseded" in ctx)
                        check("%s has no unqualified stale emitted count" % doc, historical,
                              ctx[:120])

        # GOAL has TWO tables describing this run (§4 measured, §5 ordered-work row 5) and they
        # drifted apart: §4 said 15/104 while §5 said 14/104 and 15.6% (14/90). Both are generated now,
        # and both must equal the artifact, and each other.
        goal_body = read("notes/GOAL.md")
        row5 = re.search(r"\| 5 \| Online AndroidWorld port \|([^\n]*)", goal_body)
        if row5:
            r5 = row5.group(1)
            check("GOAL §5 row 5 success count matches", ("%d successes" % n_succ) in r5, str(n_succ))
            check("GOAL §5 row 5 finished count matches",
                  ("%d/%d finished" % (n_fin, 116)) in r5, "%d/%d" % (n_fin, 116))
            check("GOAL §5 row 5 SR matches", ("SR %.1f%%" % float(m.group(3))) in r5, m.group(3))
        drift_row = re.search(r"\| `gate_score` drift vs episode outcome \|([^\n]*)", goal_body)
        mdf = re.search(r"drift fail   : n=(\d+) mean=(-?[\d.]+) decaying=(\d+)%", txt)
        if drift_row and mdf:
            dr = drift_row.group(1)
            check("GOAL drift row failure n matches", ("n=%s" % mdf.group(1)) in dr, mdf.group(1))
            check("GOAL drift row failure decay matches", ("%s%%" % mdf.group(3)) in dr, mdf.group(3))
        row4 = re.search(r"\| AndroidWorld full benchmark \|([^\n]*)", goal_body)
        if row4:
            check("GOAL §4 and §5 agree on the success count",
                  ("%d successes" % n_succ) in row4.group(1), str(n_succ))
else:
    missing = [p for p, ok in has.items() if not ok]
    check("all prediction artifacts present", False, str(missing))


# The gate counts themselves are numbers the record quotes, and they drifted three times
# (105/105 -> 120/120 -> ...). Write the current values into the two places that state them,
# so the documents cannot claim a count this script does not produce.
def _sync_gate_counts():
    total = passes + len(fails)
    # ONLY the clause explicitly introduced as "Current:" may be rewritten. The first version also
    # matched the bare pattern and overwrote the "as of that day" dated-history figure with today's
    # count, which destroys the history this log exists to keep.
    # ONE machine-refreshed block in AGENTS.md, plus the provenance line. Dated log entries carry no
    # counts at all, because a dated entry cannot know a later commit's test count.
    # Measure every line of this block instead of typing it. A hardcoded "69 assertions" and
    # "132 passed" are exactly the numbers that went stale three times before.
    def _run(cmd, cwd, extra_env=None):
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)
        return subprocess.run([sys.executable] + cmd, cwd=cwd, capture_output=True, text=True, env=env)

    vp_n = "?"
    # verify_paper.py resolves its inputs relative to the CWD, so it must run from the repo root.
    rv = _run([os.path.join(ROOT, "benchmarks", "train", "verify_paper.py")],
              cwd=ROOT,
              extra_env={"PYTHONPATH": os.path.join(ROOT, "benchmarks", "train")})
    mv = re.search(r"ALL CHECKS PASS", rv.stdout)
    npass = len(re.findall(r"^\s*\[PASS\]", rv.stdout, re.M))
    vp_n = "%d assertions, exit %d" % (npass, rv.returncode) if npass else (
        "ALL CHECKS PASS" if mv else "see output")

    ca_n = "ALL CHECKS PASS"
    rc2 = _run([os.path.join(ROOT, "scripts", "verify_compound_accuracy.py")], cwd=ROOT)
    if rc2.returncode != 0:
        ca_n = "FAILING"

    ao_n = "ALL OBJECTIONS CLOSED"
    ra = _run([os.path.join(ROOT, "scripts", "verify_audit_objections.py")], cwd=ROOT)
    if ra.returncode != 0:
        ao_n = "SOME OBJECTION OPEN"

    tests_n = "?"
    rt = _run(["-m", "pytest", "tests/", "-q"], cwd=os.path.join(ROOT, "benchmarks", "aw"),
              extra_env={"PYTHONPATH": os.path.join(ROOT, "benchmarks", "aw")})
    mt = re.search(r"(\d+) passed(?:, (\d+) skipped)?", rt.stdout)
    if mt:
        tests_n = "%s passed%s" % (mt.group(1), ", %s skipped" % mt.group(2) if mt.group(2) else "")

    gates = (
        "<!-- GATES:BEGIN (machine-refreshed by scripts/check_record.py; do not edit by hand) -->\n"
        "```\n"
        "check_record.py            %d/%d\n"
        "verify_paper.py            %s\n"
        "verify_audit_objections.py %s\n"
        "verify_compound_accuracy.py %s\n"
        "benchmarks/aw/tests        %s\n"
        "```\n"
        "<!-- GATES:END -->" % (total, total, vp_n, ao_n, ca_n, tests_n)
    )
    stale = [
        ("AGENTS.md", r"<!-- GATES:BEGIN.*?<!-- GATES:END -->", gates),
        ("outputs/calibrated-decision-value.provenance.md", r"\d+ checks pass", "%d checks pass" % total),
    ]
    for rel, pat, repl in stale:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        body = open(path, encoding="utf-8").read()
        new_body = re.sub(pat, repl, body, flags=re.S)
        if new_body != body:
            open(path, "w", encoding="utf-8").write(new_body)


if not fails:
    _sync_gate_counts()

print()
if fails:
    print(f"{len(fails)} FAILED, {passes} passed — the record does not match the artifacts.")
    sys.exit(1)
print(f"ALL {passes} CHECKS PASS — the record documents agree with the artifacts.")
