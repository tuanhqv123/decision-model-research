You are reviewing a research plan. Your job is to find what is WRONG, not to praise it.

Read these files:
- docs/research/PLAN-2026-11-07-better.md   (the plan)
- benchmarks/results/2026-11-07-fusion-headroom.md

Then inspect the local artifacts with bash/python:
- benchmarks/results/2026-10-29-evals/*.jsonl
- backup-2026-10-29/**/eval-*.jsonl  (10 different runs exist)
- benchmarks/data/acv2/500_steps_filtered.json  (the 398 test rows: gold actions + a11y trees)

## Your assigned angle: DATA LEAKAGE AND ENSEMBLE VALIDITY

Attack these specifically:
1. Are the 10 checkpoints actually independent? Check which training data each used. `cur24k` is a superset of
   `cur15k`; `cont500`/`cont250` are continuations of E2; `E2_705` is an earlier step of the SAME run as E2.
   Using an earlier step of the same run as an "ensemble member" is not an ensemble — quantify how much of the
   reported union/oracle headroom comes from near-duplicate members.
2. Recompute the oracle union using ONLY genuinely different runs (exclude same-run checkpoints and superset
   datasets). Does 80.90% survive? Report the corrected number.
3. The plan's work #4 proposes "train a checkpoint that is different". Check whether a decorrelated checkpoint
   already exists on disk and was overlooked.
4. Is the majority-vote scheme even well-posed? When 4 runs say click at (100,200) and 3 say click at (100,208),
   the rounded-point signature splits them into two groups and the "majority" may be a minority of the 10. Count
   how often the winning signature has < 6 of 10 votes.
5. The gold labels: are there rows where the gold itself is ambiguous (multiple valid elements, per
   AndroidControl-Curated's known 24% unclear rate)? Estimate how many of the 398 rows that could be, and what
   it does to any "improvement" claim at this n.

Write your findings to benchmarks/results/2026-11-07-review-leakage.md
Format: ## Finding N — [SEVERITY] one-line claim, then evidence with actual numbers you computed, then what
the plan should say instead. Severity ∈ {FATAL, MAJOR, MINOR}. Be concrete and quote real numbers.
Do not edit the plan. Do not be agreeable. If the plan is right, say so briefly and move on.