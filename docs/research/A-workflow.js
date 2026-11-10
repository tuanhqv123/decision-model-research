// A-lane research fanout: four independent issues, four independent outputs.
// One writer per file, no shared scope. A4 additionally writes benchmarks/train/audit_labels.py.
return await runs.all([
  {
    key: "a1",
    agent: "researcher",
    task:
      "Read docs/research/ISSUES-2026-11-09.md, then docs/research/A1-brief.md, and execute A1-brief.md exactly. " +
      "Deliverable: docs/research/A1-gui-odyssey-usability.md. " +
      "You must inspect the REAL annotation JSON of OpenGVLab/GUI-Odyssey (and the newer hflqf88888/GUIOdyssey the card recommends), " +
      "not just the dataset card. Count actions per type over the corpus. Answer the one question that decides the plan: " +
      "can this corpus supervise 'which field do I type into', which is the only competence gap our AndroidWorld run exposed. " +
      "Cite every number to a URL or a file path you actually read. Mark what you could not verify. Write no other files.",
  },
  {
    key: "a2",
    agent: "researcher",
    task:
      "Read docs/research/ISSUES-2026-11-09.md, then docs/research/A2-brief.md, and execute A2-brief.md exactly. " +
      "Deliverable: docs/research/A2-trajectory-sft-evidence.md. " +
      "Core question: is there MEASURED evidence that multi-step trajectory SFT raises END-TO-END task success for a <=4B GUI model, " +
      "versus only per-step accuracy. If no qualifying study exists at 2B, say so loudly. " +
      "Also cover compounding error when the starting model is only 64% per-step, and include the steelman counter-case. " +
      "Never mix per-step and end-to-end numbers in one column. Cite every number to a URL. Write no other files.",
  },
  {
    key: "a3",
    agent: "researcher",
    task:
      "Read docs/research/ISSUES-2026-11-09.md, then docs/research/A3-brief.md, and execute A3-brief.md exactly. " +
      "Deliverable: docs/research/A3-typing-and-termination.md. " +
      "Diagnose three candidate mechanisms (rare-action supervision imbalance, readout-slot collapse, no progress/state model) against " +
      "published evidence AND against our own measured negatives (status down-weighting already failed: -27 premature, -26 correct). " +
      "Rank remedies by expected end-to-end delta per GPU-hour, marking SFT-only vs needing RL/planner infrastructure. " +
      "Include the Android text-entry mechanics section. Cite every number to a URL. Write no other files.",
  },
  {
    key: "a4",
    agent: "researcher",
    task:
      "Read docs/research/ISSUES-2026-11-09.md, then docs/research/A4-brief.md, and execute A4-brief.md exactly. " +
      "Deliverables: docs/research/A4-label-audit.md AND benchmarks/train/audit_labels.py. " +
      "You must actually write and RUN the audit script on compact_cur15k.jsonl + compact_validation.jsonl under " +
      "backup-2026-10-29/GPU_HOST_C/acwork/acdata/ and report the numbers it produces, not the numbers in the brief. " +
      "Also verify the provenance of the quoted 37.63% AndroidControl-Curated wrong-GT figure from its primary arXiv source. " +
      "If the data path is missing, say so and stop rather than inventing numbers. Do not train anything. Write no other files.",
  },
]);
