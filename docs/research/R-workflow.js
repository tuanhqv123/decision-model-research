return await runs.all([
  { key: "r1", agent: "researcher", task: "Read docs/research/R1-brief.md and execute it. Write docs/research/R1-small-model-grounding-recipes.md.", output: "docs/research/R1-small-model-grounding-recipes.md" },
  { key: "r2", agent: "researcher", task: "Read docs/research/R2-brief.md and execute it. Write docs/research/R2-grounding-loss-design.md.", output: "docs/research/R2-grounding-loss-design.md" },
  { key: "r3", agent: "researcher", task: "Read docs/research/R3-brief.md and execute it. Write docs/research/R3-grounding-data-design.md.", output: "docs/research/R3-grounding-data-design.md" },
  { key: "r4", agent: "researcher", task: "Read docs/research/R4-brief.md and execute it. Write docs/research/R4-candidate-interface-postmortem.md.", output: "docs/research/R4-candidate-interface-postmortem.md" },
]);
