You are reviewing a research plan. Your job is to find what is WRONG, not to praise it.

Read these files:
- docs/research/PLAN-2026-11-07-better.md   (the plan)
- benchmarks/results/2026-11-07-fusion-headroom.md
- benchmarks/results/2026-11-07-status-downweight-arm.md
- benchmarks/results/2026-11-07-what-the-arm-bought.md

Then re-derive, yourself, using bash/python on the local prediction JSONLs:
- backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl  (255/398)
- backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl  (258/398)
- benchmarks/results/2026-11-07-arm/eval-arm20test-V.jsonl  (242/398)
- benchmarks/results/2026-10-29-evals/E2-705-V.jsonl  (244/398)

## Your assigned angle: STATISTICAL VALIDITY

Attack these specifically:
1. The plan claims "+2.01 pts, p=0.18" from 10-checkpoint majority vote. Is the McNemar test valid here? The
   398 rows span 350 episodes, so rows are not independent. Recompute with an episode-clustered test and say
   whether p changes materially.
2. The plan says the val confound (3 checkpoints on val, 10 on test) makes the number uninterpretable. Is that
   right, or is it an overstatement? What exactly does it bias?
3. "Oracle union 80.90%" — is that a legitimate ceiling to motivate work, or is it misleading because 228/398
   rows are all-correct and the 16-pt headroom is concentrated in a few rows? Quantify: how many rows would
   actually have to flip?
4. Multiple comparisons: the plan tried px ∈ {0,4,8,16,32,64} × 2 subsets = 12 configurations before picking
   one. Does +2.01 survive any correction? 
5. Is n=398 even capable of detecting +1.5 pts? Compute the minimum detectable effect with episode clustering.

Write your findings to benchmarks/results/2026-11-07-review-statistics.md
Format: ## Finding N — [SEVERITY] one-line claim, then evidence with actual numbers you computed, then what
the plan should say instead. Severity ∈ {FATAL, MAJOR, MINOR}. Be concrete and quote real numbers.
Do not edit the plan. Do not be agreeable. If the plan is right, say so briefly and move on.