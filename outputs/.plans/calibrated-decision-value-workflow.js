return await runs.all([
  {
    key: "metrics-outcomes",
    agent: "researcher",
    task: "Read outputs/.plans/calibrated-decision-value-T1.md and write outputs/.drafts/calibrated-decision-value-research-metrics-outcomes.md.",
    output: "outputs/.drafts/calibrated-decision-value-research-metrics-outcomes.md",
  },
  {
    key: "audience",
    agent: "researcher",
    task: "Read outputs/.plans/calibrated-decision-value-T2.md and write outputs/.drafts/calibrated-decision-value-research-audience.md.",
    output: "outputs/.drafts/calibrated-decision-value-research-audience.md",
  },
  {
    key: "failure-modes",
    agent: "researcher",
    task: "Read outputs/.plans/calibrated-decision-value-T3.md and write outputs/.drafts/calibrated-decision-value-research-failure-modes.md.",
    output: "outputs/.drafts/calibrated-decision-value-research-failure-modes.md",
  },
  {
    key: "internal-audit",
    agent: "researcher",
    task: "Read outputs/.plans/calibrated-decision-value-T4.md and write outputs/.drafts/calibrated-decision-value-research-internal-audit.md.",
    output: "outputs/.drafts/calibrated-decision-value-research-internal-audit.md",
  },
]);
