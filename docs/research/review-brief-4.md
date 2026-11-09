You are reviewing a research plan. Your job is to find what is WRONG, not to praise it.

Read:
- docs/research/PLAN-2026-11-07-better.md   (the plan)
- papers/single-pass-decision-acv2.md §5.8, §5.10  (the current paper text)

Then inspect with bash/python:
- benchmarks/results/2026-11-07-arm/*.jsonl
- benchmarks/results/2026-10-29-evals/*.jsonl
- benchmarks/train/{paired_eval.py,error_budget.py,fusion_arbiter.py}

## Your assigned angle: ENGINEERING / REPRODUCIBILITY — would this work actually run?

Attack these specifically:
1. Read benchmarks/train/fusion_arbiter.py. It reported 2/398 accuracy in the plan's summary. Find the actual
   bug(s): the label scheme (`y = 1 if arm correct and E2 not` gives -1 for ~57% of rows and those rows are
   dropped), the standardisation being fitted on disagreement rows but applied to all rows, the `Xv_s if False
   else` dead branch. List each bug with the line, and say what a correct implementation must do.
2. Work #1 says "regenerate val for all ten checkpoints, ~20 min GPU". Verify that claim: read
   benchmarks/train/eval_d1.py and estimate the real cost (827 rows, batch 8, ~120 ms/step). Does 2 min per
   checkpoint hold? What else is needed (checkpoint files on disk, image paths, acdata)?
3. Which of the ten checkpoints still have their WEIGHTS on disk? Check backup-2026-10-29/ for model.safetensors
   per run. If a checkpoint's predictions exist but its weights do not, work #1 is impossible for that run — say
   which ones, with file paths and sizes.
4. Is the "10 checkpoints on disk" claim even correct? List them with the artifact path and the sha of their
   correctness vector, and identify which are byte-identical duplicates (the same run evaluated twice).
5. The plan has no rollback/cleanup section. Given this programme's history (five separate teardown audits, one
   of which missed 3.2 GB), what must be added before any GPU work starts?

Write your findings to benchmarks/results/2026-11-07-review-engineering.md
Format: ## Finding N — [SEVERITY] one-line claim, then evidence (file:line, real numbers), then the fix.
Severity ∈ {FATAL, MAJOR, MINOR}. Be concrete. Do not edit the plan. Do not be agreeable.