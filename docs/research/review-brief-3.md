You are reviewing a research plan. Your job is to find what is WRONG, not to praise it.

Read:
- docs/research/PLAN-2026-11-07-better.md   (the plan)
- docs/HANDOFF.md   (programme context)
- notes/GOAL.md    (the stated goal)

## Your assigned angle: STRATEGY — is this the right work at all?

The programme has spent months on AC-v2 step accuracy and is stuck at 64.07% (a tie with an 8B baseline).
This plan proposes more ensembling work. Attack that choice.

Answer these:
1. **Does anyone care about AC-v2 in 2026?** Search the web/arXiv for the current state of AndroidControl-v2
   as a benchmark: is it still used, is it superseded, do the 2026 SOTA papers still report it? Cite URLs and
   dates. If it is stale, say so bluntly.
2. **Is a step-accuracy ensemble a publishable contribution?** Compare against what a reviewer at TMLR/ICML
   would need (see the criteria quoted in outputs/calibrated-decision-value.md §3). Is "+2.01 pts on a
   correlated ensemble, p=0.18" publishable? What WOULD be?
3. **The plan's work #3 (marked-click verifier) is the only genuinely new idea.** Read the AdaGround paper
   summary at https://arxiv.org/html/2610.36874 and judge: does a training-free verifier fit this programme's
   strengths (small model, single pass, calibration), or does it break the "zero output tokens / single forward
   pass" thesis by requiring extra queries? Be specific about the latency and token cost implications.
4. **Rank the five work items by expected value per GPU-hour** and say which you would delete. Justify with the
   numbers already in the artifacts, not with ambition.
5. **What is the strongest paper this programme could actually publish in 3 months?** Give a concrete title,
   one-sentence claim, and the minimum evidence needed. If the honest answer is "none, pivot", say that.

Write your findings to benchmarks/results/2026-11-07-review-strategy.md
Format: ## Finding N — [SEVERITY] one-line claim, then evidence, then recommendation.
Severity ∈ {FATAL, MAJOR, MINOR}. Be blunt. Do not be agreeable. Cite URLs for external claims.