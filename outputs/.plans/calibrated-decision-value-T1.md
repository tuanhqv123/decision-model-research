# T1 brief — metrics and outcome evidence

Read this, then write `outputs/.drafts/calibrated-decision-value-research-metrics-outcomes.md`.
Do not modify anything else in the repo.

## Question

When a paper claims a small model's confidence is "usable", what is actually measured — and is there any
peer-reviewed evidence that abstention (act / ask / refuse) improves a real outcome rather than only a
ranking metric?

## Find

1. **Definitions and metrics actually reported** in selective prediction / calibration work: ECE, Brier,
   AUROC(conf, correct), selective risk at coverage, AURC, risk–coverage curves, E-AURC, CRC/Clopper–Pearson.
   For each: what decision it licenses and what it does not.
2. **GUI / mobile agent evidence**: AndroidControl(-v2), AndroidWorld, ScreenSpot, GUI-Odyssey, MobileRL,
   GUI-Libra, UI-Venus, Mem-W, Jev-Mobile, GUI-JEPA, decider, THX-01, TypedBench, AgentAbstain.
   Record exact protocol, split, n, and whether the claim is peer-reviewed or a preprint/vendor number.
3. **Outcome-level evidence in adjacent domains** where abstention is studied with a real cost of error:
   medical triage/imaging, content moderation, robotics, autonomous driving, LLM routing/guarding.
   Prefer results with a baseline and an effect size (e.g. risk at fixed coverage, harm avoided, cost).
4. **The gap**: any paper that explicitly states confidence ranking ≠ deployment safety, or that measures
   calibration→outcome transfer.

## Rules

- Primary sources first. Peer-reviewed beats preprint beats vendor card. Label each.
- Every number gets: source URL, venue/date, protocol, split, n, baselines.
- If only an abstract is available, say so; do not infer numbers from prose.
- Mark dead links. If PDF parsing fails, continue from metadata/abstract and record the failure.
- Do not invent citations, benchmarks, or effect sizes.

## Output format

- Section per item above.
- A table: Claim | Value | Source | Peer-reviewed? | Protocol/split/n | Caveat.
- A short "what is NOT measured anywhere" list.
- Exact search terms used, in a final section.
