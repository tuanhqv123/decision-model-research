// Second fan-out: the two lanes that died (a3, a5). One writer per file.
return await runs.all([
  {
    key: "a5",
    agent: "researcher",
    task:
      "Read docs/research/ISSUES-2026-11-09.md, then AGENTS.md section 3 'MISTAKE LEDGER' (14 real errors, do not repeat), " +
      "then docs/research/A5-brief.md, and execute A5-brief.md exactly. " +
      "Deliverable: docs/research/A5-benchmark-protocol.md. " +
      "This is the MOST IMPORTANT brief of the set: it decides whether any future training run can produce a publishable claim. " +
      "Key facts already established that you must USE, not re-derive: " +
      "(1) our AndroidWorld run = 104/116 finished, 15 successes, 18.8% over 80 scorable, Wilson 11.7-28.7; " +
      "(2) published 2B bar = MAI-UI-2B 45.1% SFT / 49.1% post-RL (END-TO-END ONLY; MAI-UI reports no per-step metric); " +
      "(3) a power calculation already run (scripts/power_aw116.py, artifacts benchmarks/results/2026-11-09-power-aw116.txt) shows " +
      "MDE at n=116 paired is ~12.9 points (psi=0.25) and ~10.0 at best; a +4-point claim needs ~1224 paired tasks, and seeds do NOT buy " +
      "independent n. Confirm or challenge that arithmetic with the proper formula and state the consequence plainly. " +
      "Also answer: is our benchmark protocol adequate at all; what is the single change that matters most; " +
      "and the training-ingredient checklist (present/absent/effect size/closable-on-one-4090) for the 'don't miss a concept' question. " +
      "Cite every number to a URL. Mark abstract-only numbers. Never mix per-step and end-to-end. Write no other files.",
  },
  {
    key: "a3",
    agent: "researcher",
    task:
      "Read docs/research/ISSUES-2026-11-09.md (issue I4), then docs/research/A3-brief.md, and execute A3-brief.md exactly. " +
      "Deliverable: docs/research/A3-typing-and-termination.md. Do not edit any other file. Be efficient: budget your searches, " +
      "you have a limited wall clock, so prefer 4-6 targeted searches with fetch_content on the key arXiv HTML pages over broad crawling. " +
      "Diagnose three mechanisms (rare-action supervision imbalance; readout-slot collapse; no progress/state model) against published " +
      "evidence AND our own negatives. Our measured facts: type_text = 24/1394 emitted (1.7%), 3.0% of training labels; " +
      "19/104 episodes emitted type_text -> 1 success at median 25 steps (budget); 85/104 never typed -> 14 successes at median 12 steps; " +
      "34 timeouts + 31 premature status declarations; a val-fitted p(status) gate was monotone-harmful at every tau (2026-11-07), " +
      "and a status down-weighting arm traded -27 premature terminations for -26 correct ones (val +1.09 p=0.44, test -3.27 p=0.15). " +
      "So do NOT recommend re-weighting. " +
      "NEW EVIDENCE you must incorporate: GUI-Odyssey has 86.7% of episodes containing text entry and localises the field via the " +
      "preceding CLICK on 99.27% of TEXT steps (see docs/research/A1-gui-odyssey-usability.md) - so mechanism M1 is directly addressable " +
      "by data mixing, and you should say what mixing ratio the literature supports. " +
      "Rank remedies by expected end-to-end delta per GPU-hour, marking SFT-only vs needing RL/planner infrastructure. " +
      "Include the Android text-entry mechanics section (ADBKeyboard/IME, editable flag, click-then-type vs direct setText). " +
      "Cite every number to a URL. Write no other files.",
  },
]);
