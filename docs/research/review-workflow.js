return await runs.all([
  {
    key: "statistics",
    agent: "researcher",
    task: "Read docs/research/review-brief-1.md and follow it exactly. Write benchmarks/results/2026-11-07-review-statistics.md.",
    output: "benchmarks/results/2026-11-07-review-statistics.md",
  },
  {
    key: "leakage",
    agent: "researcher",
    task: "Read docs/research/review-brief-2.md and follow it exactly. Write benchmarks/results/2026-11-07-review-leakage.md.",
    output: "benchmarks/results/2026-11-07-review-leakage.md",
  },
  {
    key: "strategy",
    agent: "researcher",
    task: "Read docs/research/review-brief-3.md and follow it exactly. Write benchmarks/results/2026-11-07-review-strategy.md.",
    output: "benchmarks/results/2026-11-07-review-strategy.md",
  },
  {
    key: "engineering",
    agent: "researcher",
    task: "Read docs/research/review-brief-4.md and follow it exactly. Write benchmarks/results/2026-11-07-review-engineering.md.",
    output: "benchmarks/results/2026-11-07-review-engineering.md",
  },
]);
