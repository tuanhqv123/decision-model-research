# Deep research plan — `calibrated-decision-value`

**Status:** PLAN ONLY. No evidence has been gathered. Nothing else in `outputs/` or `papers/` may be
created until the user approves this plan.
**Date:** 2026-11-07
**Owner:** lead (this session) — synthesis is not delegated.
**Workflow:** Feynman `deep-research` skill + the pasted 7-step recipe (plan → scale → evidence → draft →
cite → review → deliver). No separate `/recipe` command exists in this workspace; the pasted recipe is the
contract.

---

## 0. Why this topic (the real question, stated plainly)

The program can already produce numbers. The failure mode now is that the numbers are **correct but
inert**: a 64.32% tie, a 0.729 AUROC, a 20%-risk certificate at 40.7% coverage. None of those, by itself,
tells a reviewer why the work should exist, tells a builder when to ship it, or tells a user why their
phone should ask instead of act.

So the research question is not "is calibration interesting". It is:

> **For which real decision, for which person, does a calibrated single-pass act / ask / refuse gate
> change an outcome — and what evidence would make a skeptical CCF-A reviewer and a real user care?**

This is a *value and evidence* investigation, not a benchmark re-run. It must end with a defensible
answer to "do we have anything", including the possibility that the answer is "not yet, and here is the
one measurement that would change it".

### Feynman framing (the explanation we must be able to give a non-specialist)

A decision model is a component that answers a fixed question with a probability instead of writing a
sentence. "Calibrated" means: when it says 0.8, it is right about 80% of the time. A "gate" uses that
probability to choose one of three actions — act, ask the human, or refuse. The value claim is that
**asking at the right moment is worth more than acting confidently at the wrong moment**. That claim is
only true when a wrong action is expensive and an ask is cheap. The research must say where that
condition actually holds in real products and real life, and measure whether our model's confidence is
good enough to find those moments.

---

## 1. Key questions

Each question is written so it can come back **negative**.

**Q1 — Definition and measurability.** What exactly is being claimed when a paper says a small model's
confidence is "usable"? Which quantities are measured (ECE, Brier, AUROC of confidence vs correctness,
selective risk at coverage, AURC, risk–coverage area), on which splits, and which of them actually license
a deployment decision? *Falsifier:* if the literature only reports ECE and never a coverage-conditioned
risk, then "usable confidence" is mostly unmeasured and our contribution is larger than assumed.

**Q2 — Outcome evidence.** Is there any **peer-reviewed** evidence that abstention / ask-for-help
improves *task outcome* (success rate, harm avoided, user time saved) rather than only ranking metrics —
for GUI agents, and in adjacent domains (medical triage, content moderation, robotics, driving)? What are
the effect sizes and the baselines? *Falsifier:* if every result is AUROC-only, the field has a
metric-to-outcome gap and we must say so rather than imply deployment readiness.

**Q3 — Who cares, and about what.** Three audiences with different currencies:
(a) a CCF-A reviewer — what makes a reliability contribution publishable when accuracy is a tie;
(b) a builder — what threshold and latency make a gate shippable, and what failure is unacceptable;
(c) an end user — which concrete harms are avoided (irreversible tap, wrong purchase, message sent to the
wrong person, medical/automotive action), and is "ask" acceptable in that moment?
*Falsifier:* if no audience has a decision where the gate changes the outcome, the value claim fails.

**Q4 — Our own evidence, audited.** After the audit: which of our headline numbers survive re-derivation
from artifacts (64.32% pure, 64.82% groundB, Type 84.17, Grounding 56.50, routed AUROC 0.729, 20% risk @
40.7% coverage, risk@50% 19.1%, online SR 1/8 with 25% env crashes)? Which are underpowered, which are
protocol-bound, and which must be retracted? *Falsifier:* if the certificate does not reproduce from the
saved artifacts, the central claim is not yet a result.

**Q5 — The failure that kills it in production.** Under what conditions does a calibrated gate silently
become unsafe — distribution shift, threshold non-transfer across checkpoints (already measured: 17.3% →
23.9%), base-rate change, correlated errors, adversarial/ambiguous screens? What monitoring or
re-certification is required, and has anyone published a method that survives it?
*Falsifier:* if no published gate survives a checkpoint or population change, then "certified" must be
reported as per-checkpoint only, never as a product property.

**Q6 — The single highest-value next measurement.** Given Q1–Q5, what one experiment has the best
evidence-per-cost for this program, what effect size is predicted, and what result would make us stop?
This question must produce a number, a cost, and a kill criterion — not a roadmap.

---

## 2. Evidence needed

| Q | Evidence type | Source class | Minimum acceptance |
|---|---|---|---|
| Q1 | metric definitions + reported values | peer-reviewed papers, arXiv, official benchmark docs | 3+ independent sources; exact metric names and splits recorded |
| Q2 | outcome results with baselines and effect sizes | peer-reviewed first; preprints clearly labelled | at least 1 peer-reviewed outcome result per domain claimed, or an explicit "none found" |
| Q3 | venue/reviewer standards + product/regulatory requirements | ICML/TMLR/NeurIPS reviewer guidance, official product docs, standards (e.g. EU AI Act, FDA/IMDRF if cited) | primary documents only; no blog summaries as sole support |
| Q4 | our artifacts | local JSONL/JSON/MD + `verify_paper.py` output | every number re-derived or marked unreproducible |
| Q5 | shift/transfer failures and re-certification methods | peer-reviewed + preprints | at least one measured transfer failure (we already have one) and one proposed remedy with evidence |
| Q6 | synthesis of Q1–Q5 | lead inference | explicitly labelled as inference, with predictions and kill criteria |

**Excluded from evidence:** vendor marketing claims presented as measurements; Twitter/X threads;
aggregator sites; any number whose protocol (split, scorer, prompt) cannot be stated.

---

## 3. Scale decision (made before assigning owners)

**Chosen scale: 4 researcher subagents + lead synthesis.**

Why not direct search: the brief must connect four evidence domains that do not overlap (metric
definitions / outcome evidence / audience requirements / internal reproducibility). A direct pass would be
≥40 tool calls across heterogeneous sources and would still lack the internal audit. Why not 6: the
marginal domain (e.g. legal regulation) is not load-bearing for the decision this brief must support.

Why not fewer than 4: dropping the internal audit would leave Q4 as opinion; dropping the audience strand
would leave the brief with no answer to "for whom".

Guardrails: each researcher may consult at most ~15 primary sources, must record dead/unverifiable links,
and must mark PDF-parser failures and continue from metadata + abstracts rather than stall.

**Deliverable type:** `outputs/calibrated-decision-value.md` (research/positioning brief), **not**
`papers/` — this is not a paper draft. It is allowed to become the paper's Introduction/Related Work
seed later, but it is judged as a brief.

---

## 4. Task ledger

| ID | Owner | Task | Output |
|---|---|---|---|
| T1 | `researcher` | Q1 + Q2: What is measured about selective prediction / calibration for small action models; is there peer-reviewed *outcome* evidence (GUI agents and adjacent domains)? Record exact metrics, splits, effect sizes, baselines. | `outputs/.drafts/calibrated-decision-value-research-metrics-outcomes.md` |
| T2 | `researcher` | Q3: Audience requirements — CCF-A review standards for reliability papers; builder/shipping thresholds; end-user harm cases where act/ask/refuse changes the outcome. Primary docs only. | `outputs/.drafts/calibrated-decision-value-research-audience.md` |
| T3 | `researcher` | Q5: Production failure modes — shift, threshold non-transfer, correlated errors, base-rate change; published re-certification / monitoring methods and their evidence. | `outputs/.drafts/calibrated-decision-value-research-failure-modes.md` |
| T4 | `researcher` | Q4: Internal audit — re-derive every headline number from `benchmarks/results/` and `backup-2026-11-06/`; run `benchmarks/train/verify_paper.py`; produce a claim-status table (supported / underpowered / retracted). | `outputs/.drafts/calibrated-decision-value-research-internal-audit.md` |
| T5 | lead | Write draft from the four notes + repo state; no delegation of synthesis. | `outputs/.drafts/calibrated-decision-value-draft.md` |
| T6 | `verifier` | Inline citations, verify every URL, write cited brief. | `outputs/.drafts/calibrated-decision-value-cited.md` |
| T7 | `reviewer` | Verification pass: unsupported claims, single-source critical claims, overstatement. | `outputs/.drafts/calibrated-decision-value-verification.md` |
| T8 | lead | Fix FATAL findings (revised file if >3 fixes), deliver + provenance. | `outputs/calibrated-decision-value.md`, `outputs/calibrated-decision-value.provenance.md` |

Researcher briefs will be written to `outputs/.plans/calibrated-decision-value-T{1..4}.md` before launch,
and launches will go through one workflow script
`outputs/.plans/calibrated-decision-value-workflow.js` using `runs.all(...)` with one stable key per task.

---

## 5. Verification log

| When | Check | Result |
|---|---|---|
| 2026-11-07 | Plan file created; no drafts/outputs/provenance created | PASS (plan only) |
| — | All four research notes exist on disk with recorded search terms | pending |
| — | Every critical claim maps to a source URL or local artifact path | pending |
| — | Every URL fetched/verified by verifier | pending |
| — | Internal numbers re-derived or marked unreproducible | pending |
| — | Reviewer pass completed; FATAL count = 0 before delivery | pending |
| — | Final + provenance exist; provenance `Verification:` set honestly | pending |

---

## 6. Decision log

| # | Decision | Why |
|---|---|---|
| D1 | Topic = real-world/audience value of calibrated single-pass decision models | User's explicit ask: "make sure our research value for audience also other people and in life"; the program's risk is inert correctness, not missing accuracy |
| D2 | Slug `calibrated-decision-value` | ≤5 words, hyphenated, no filler |
| D3 | Scale = 4 researchers + lead | Four non-overlapping evidence domains; internal audit cannot be delegated to synthesis |
| D4 | Deliverable = `outputs/` brief, not `papers/` | Positioning/evidence brief; paper integration is a separate step |
| D5 | Pre-registered value test (below) | Prevents a polished but useless deliverable |
| D6 | No GPU, no host access, no new training in this run | Pure evidence run; the user asked for value before effort |

### Pre-registered value test (what makes this worth doing)

The brief is worth delivering only if it establishes **at least two** of:

1. a peer-reviewed, outcome-level result where abstention changes success/harm (not AUROC alone);
2. a named, concrete user harm that an act/ask/refuse gate prevents, with the cost of asking stated;
3. a reviewer-legible contribution shape that our *existing* artifacts can support without new training;
4. a single next measurement with a predicted effect size, a cost, and a kill criterion;
5. a documented reason our current headline cannot support a CCF-A claim yet (a negative that saves months).

If none hold, the brief says so plainly and recommends stopping or re-scoping. That is an acceptable
outcome and will be reported as such.

---

## 7. Open questions for the user (before approval)

- Is this the topic you want, or should the brief target one narrower decision (e.g. "should the phone
  ask before an irreversible tap")?
- Should the brief be written for a CCF-A reviewer, a product builder, or a general audience first?
- Is using 4 researchers acceptable, or do you want a direct-search run to keep it cheap?
