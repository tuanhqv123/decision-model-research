return await runs.all([
  { key: "v1", agent: "researcher", task: "Read docs/research/V1-brief.md and execute it. Write docs/research/V1-verify-grounding-fix.md.", output: "docs/research/V1-verify-grounding-fix.md" },
  { key: "v2", agent: "researcher", task: "Read docs/research/V2-brief.md and execute it. Write docs/research/V2-verify-design.md.", output: "docs/research/V2-verify-design.md" },
]);
