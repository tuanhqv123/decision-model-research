"""Verify the auditor's four objections + the fifth defect found during rework.

Each check is mechanical against the current tree. Pointer checks pair a pointer to the NEAREST
subject on its own line (a line may legitimately name two subjects with two pointers).
"""
import os
import sys, re, sys

ROOT = "~/Workspace/research"
os.chdir(ROOT)
ok = True

def chk(label, cond, detail=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  :: " + detail) if not cond and detail else ""))
    if not cond:
        ok = False

def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

def line(p, n):
    return read(p).splitlines()[n - 1]

print("O1 — docs/HANDOFF.md contradicts the repository")
for f in ("benchmarks/aw/awenv/agent.py", "benchmarks/aw/awenv/runner.py",
          "benchmarks/aw/awenv/baselines.py", "benchmarks/aw/run_nightly.sh"):
    chk(f"{os.path.basename(f)} exists (auditor: 'not built yet')", os.path.exists(f))
h = read("docs/HANDOFF.md")
chk("no 'Not built yet'", "Not built yet" not in h)
chk("no stale '48 tests'", "48 tests" not in h)
# The auditor's point was that HANDOFF quoted a stale count ("48 tests, 2 skipped"). Pinning a new
# literal re-creates the same defect the moment the suite grows, so assert the count HANDOFF quotes
# is the one the suite actually reports.
import re as _re, subprocess as _sp, os as _os
_r = _sp.run([sys.executable, "-m", "pytest", "tests/", "-q"],
             cwd="benchmarks/aw", capture_output=True, text=True,
             env={**_os.environ, "PYTHONPATH": "benchmarks/aw"})
_m = _re.search(r"(\d+) passed", _r.stdout)
_actual = _m.group(1) if _m else "?"
# HANDOFF may either quote the live count or point at the machine-refreshed block. What it must NOT
# do is quote a count that is not the real one — that was the auditor's original complaint.
# Only counts presented as CURRENT matter. "110 passed ... when this list was written" is dated
# history and must stay as it was; a count asserted as today's must equal today's.
_bad = []
for _m2 in _re.finditer(r"(\d+)\s+passed", h):
    _ctx = _re.sub(r"\s+", " ", h[max(0, _m2.start() - 240):_m2.end() + 240]).lower()
    if any(k in _ctx for k in ("when this list was written", "as of", "then", "was ", "2026-11-0")):
        continue
    if _m2.group(1) != _actual:
        _bad.append(_m2.group(1))
chk(f"no current count in HANDOFF contradicts the suite ({_actual})",
    _actual != "?" and not _bad, "stale=%s" % _bad)
chk("stage-2 arm result present", "63.85" in h)
chk("cost-window content present", "2.11–5.79" in h)
chk("body no longer dated 2026-11-05", "Updated: 2026-11-05" not in h)
chk("body carries a current date", "2026-11-09" in h or "2026-11-08" in h)

print("\nO2 — notes/GOAL.md wrong online number + inverted paper precedence")
g = read("notes/GOAL.md")
glines = g.splitlines()
chk("online is 1/8 in both conditions", "1/8 in both conditions" in g)
chk("no unmarked '1/6 vs 0/6'",
    all("superseded" in l.lower() for l in glines if "1/6 vs 0/6" in l))
chk("crash root cause recorded", "Clock-task" in g)
chk("cost window recorded", "Cost window" in g or "cost window" in g)
chk("value brief referenced", "calibrated-decision-value" in g)
chk("single-pass named the authority", "the audit-corrected version, and the authority" in g)
chk("interface draft named subordinate", "subordinate" in g)
chk("§10 table row marks interface subordinate",
    any("| Paper |" in l and "subordinate" in l for l in glines))

print("\nO3 — no live 64.32 / 56.50 outside dated history")
ip = read("papers/calibrated-decision-interface-2026.md")
chk("interface abstract says 64.07", "64.07" in ip.splitlines()[22])
chk("interface abstract says 56.05", "56.05" in ip.splitlines()[23])
chk("interface §3.4 says 56.05", "our 56.05% grounding" in ip)
# content-based, not line-based: an edit elsewhere must not break this check
sft = read("benchmarks/results/2026-10-29-d1-acv2-sft.md")
chk("sft.md pure-protocol line says 255/398", "64.07% (255/398)" in sft)
chk("sft.md no longer claims 56.50 for @938", "Grounding 56.05% (125/223)" in sft)
chk("sft.md retracts 'gap is grounding, not type'", "retracted" in sft and "comparable" in sft)
chk("groundB:19 says 255/398", "255/398" in line("benchmarks/results/2026-11-02-d1-acv2-groundB.md", 19))
chk("groundB:21 says 56.05", "56.05" in line("benchmarks/results/2026-11-02-d1-acv2-groundB.md", 21))
chk("REPRO:17 is MATCH not N/A", "MATCH" in line("docs/compute/REPRO-2026-11-06.md", 17))
chk("REPRO:31 states the pure artifacts exist", "255/398" in line("docs/compute/REPRO-2026-11-06.md", 32))
chk("D1_API_SERVER:3 qualifies 64.32",
    "before the 2026-11-07 audit" in line("docs/compute/D1_API_SERVER.md", 3))

print("\nO4 — section pointers match the paper's actual sections")
paper = read("papers/single-pass-decision-acv2.md")
SECTIONS = {"5.8": "Why grounding plateaus", "5.9": "A training-procedure lever",
            "5.10": "The type bucket", "5.11": "Reading the same decision under CE",
            "5.12": "What the gate is worth", "5.13": "The protocol that would upgrade"}
for num, title in SECTIONS.items():
    chk(f"paper §{num} = {title}", f"### {num} {title}" in paper)
# pair each pointer with the nearest subject on its own line (<=70 chars)
SUBJ = {"cost window": "12", "type bucket": "10", "negative arm": "10",
        "stage-2 arm": "10", "CE, Brier": "11", "CE/Brier": "11"}
for f in ("AGENTS.md", "docs/HANDOFF.md", "notes/GOAL.md", "notes/2026-11-07-session.md"):
    bad = []
    for ln in read(f).splitlines():
        low = ln.lower()
        ptrs = [(m.start(), m.group(1)) for m in re.finditer(r"§5\.(\d+)", ln)]
        subs = [(low.find(p), p, w) for p, w in SUBJ.items() if low.find(p) != -1]
        for pos, num in ptrs:
            cands = [(abs(pos - sp), p, w) for sp, p, w in subs if abs(pos - sp) <= 70]
            if not cands:
                continue
            _, p, want = min(cands)
            if num != want:
                bad.append(f"{p!r}->§5.{num} want §5.{want}: {ln.strip()[:60]}")
    chk(f"{f}: every pointer matches its nearest subject", not bad, str(bad[:2]))

print("\n(5) — §5.9 significance retracted (found during rework, not by the auditor)")
chk("Welch test reported", "Welch" in paper)
chk("exact permutation reported", "permutation" in paper)
chk("no significance claim", "do not claim significance" in paper)
chk("variance ratio named (8.3x)", "8.3" in paper)
chk("n=3 and n=4 stated", "**3, 56.80 ± 1.29**" in paper and "**4, 59.42 ± 0.45**" in paper)
chk("pre-registered bar stated as unmet", "never\nreached and must not be quoted as met" in paper)

print()
print("ALL OBJECTIONS CLOSED" if ok else "SOMETHING STILL FAILS")
sys.exit(0 if ok else 1)
