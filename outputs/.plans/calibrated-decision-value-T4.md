# T4 brief — internal audit (local artifacts, read-only)

Read this, then write `outputs/.drafts/calibrated-decision-value-research-internal-audit.md`.
You may READ and RUN read-only analysis, but do NOT edit, delete, or train anything.

## Question

Which of our own headline numbers actually reproduce from artifacts, which are underpowered, and which
must be retracted?

## Audit targets

- AC-v2 pure **64.32%** (256/398) and groundB **64.82%** (258/398)
- Type **84.17%**, Grounding **56.50%** (126/223), groundB grounding 60.09%
- Trivial baselines: always-`status` 21.86%, always-`click` 53.27%, uniform ~14.3%
- Type-routed gate AUROC **0.729** test / 0.760 val / 0.709 second checkpoint
- Certified **20% selective risk @ 40.7% coverage** (CP upper 21.7%), empirical 17.3%
- Learned-fusion risk@50% **19.1%** (and its non-significant Δ +0.013 [−0.020, +0.043])
- Conformal non-transfer **17.3% → 23.9%** across checkpoints
- Online AndroidWorld: SR **1/8** both conditions, invalid 28.2% → 14.7%, crash **2/8**, model p50 122.9–124.3 ms

## Where to look

- `benchmarks/results/2026-11-03-stage0/`, `benchmarks/results/2026-10-29-evals/`,
  `backup-2026-11-06/` (gate_runs, reports, offline), `benchmarks/results/*.jsonl`
- `benchmarks/train/verify_paper.py` — run it and paste the actual output
- `benchmarks/train/{error_budget,paired_eval,gate_routed,grounding_audit,score_fusion}.py`
- `papers/single-pass-decision-acv2.md` §5.2, §5.7 and `notes/GOAL.md`
- `docs/compute/AW_PROTOCOL.md` for what the online numbers must carry

## Rules

- Report the command you ran and its real output. No paraphrased results.
- If an artifact is missing, say exactly which file and what that blocks.
- Distinguish: reproduces / reproduces only with a different protocol / cannot reproduce / not on disk.
- Do not recompute anything with a GPU or the network. CPU/numpy only.
- Do not "fix" a number to match the paper; report the discrepancy.

## Output format

- Claim-status table: Claim | Paper value | Re-derived value | Command/artifact | Status | Action.
- A "must retract or restate" list and a "safe to keep" list.
- The exact `verify_paper.py` output.
- What is blocked by a missing artifact, and the cheapest command that would unblock it.
