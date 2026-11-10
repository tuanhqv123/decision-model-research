# A5 — What is the CORRECT way to benchmark this model? (the protocol brief)

Read `docs/research/ISSUES-2026-11-09.md` first, and AGENTS.md §3 "MISTAKE LEDGER" (twelve real errors; do not
repeat them). Then execute this brief and write `docs/research/A5-benchmark-protocol.md`.

## Why this brief exists

The operator's words: *"deep research explore what will be the correct way that correct our benchmark, don't
make a mistake, miss concept training like the old."* The programme has already burned months on a benchmark
that turned out to be unrankable across papers (AC-v2, 5.18-point protocol spread on one backbone) and then
reported an AndroidWorld number (18.8%) whose denominator nobody had chosen in advance. **This brief is about
getting the measuring instrument right before spending more compute.**

## What we currently have, measured (do not re-derive)

| Fact | Value | Where |
|---|---|---|
| AndroidWorld-116 run | 104 finished / 116 planned; **15 successes** | `benchmarks/results/2026-11-08-aw116/` |
| SR over scorable | 15/80 = **18.8%**, Wilson 11.7–28.7 | same |
| SR over attempted | 15/92 = 16.3% | same |
| SR over all finished | 15/104 = 14.4%, Wilson 8.9–22.4 | same |
| Buckets | 15 success / 34 timeout / 31 premature `status` / 12 unscored / 12 setup_error | same |
| Invalid actions | **0 of 1,394** emitted | same |
| Published 2B bar | MAI-UI-2B **45.1% SFT / 49.1% post-RL**; per-step 92.3/93.1 | arXiv 2512.22047 Table 9 |
| **Never run** | AndroidLab-138; 40 tasks × 3 seeds; on-device latency | — |

## The questions this brief must answer, with evidence

1. **What does AndroidWorld's own protocol actually require?** Read arXiv 2405.14573 and the
   `google-research/android_world` repo (`docs/`, `README`, `run.py`, the `TASK_SET` definition). Extract
   verbatim: the **task count** (is it 116?), the **official success metric**, whether runs are seeded, the
   **step budget** convention, whether a human/automatic checker runs on device state, and **what a published
   number must disclose**. Then state exactly which of these our run did NOT satisfy.
2. **Why do published AndroidWorld numbers disagree by ~5 points on the same backbone?** Find 2–3 concrete
   pairs (same or near-identical model, different reported numbers) and identify the specific protocol
   difference that explains each (step budget, task subset, prompting, a11y tree vs pixels, retry policy,
   averaging over seeds, checker version). This is the single most important item: **if the protocol has a
   5-point tail, a 2-point improvement claim is meaningless.**
3. **What is the MINIMUM protocol a reviewer will accept for a ≤2B single-pass model?** Find published
   small-model AndroidWorld evaluations and record their disclosure: task count, seeds, budget, hardware,
   number of runs. Then write the exact protocol specification we must adopt, in a form a reviewer could
   check line by line. Include the pre-registration requirement.
4. **Statistical power.** With 116 tasks, what is the minimum detectable effect at 80% power? Compute it
   explicitly for the paired design (same tasks, two checkpoints) and for the unpaired case. State how many
   seeds are needed to distinguish a +2, +4 and +6 point claim. **This determines whether our next training
   run can produce a publishable claim at all** — answer that plainly.
5. **What is the correct primary metric for THIS paper?** The programme's surviving thesis is the calibrated
   act/ask/refuse gate, not accuracy. So: (a) which existing benchmark/reporting convention supports a
   risk-controlled abstention claim on mobile GUI; (b) what must be reported for such a claim (risk at fixed
   coverage, coverage at fixed risk, AURC, the conformal/CRC bound and its assumptions); (c) is there a
   published GUI-abstention result to compare against — the recorded claim is that **no GUI paper has an
   outcome-level abstention result**, verify or refute that from primary sources. Also check CORA
   (arXiv 2604.09155) which reportedly already publishes conformal risk-controlled execute/abstain for mobile
   GUI — if so, our gate is not novel and the paper must be reframed around the *price* of an ask.
6. **What concept did we miss in training?** This is the "don't miss a concept like the old" part. Compare the
   published training recipes (MAI-UI arXiv 2512.22047; MobileRL 2509.18119; GUI-Odyssey/OS-Atlas/UGround
   lineage) against what our recipe does, and produce a **checklist of ingredients**, each marked
   present/absent in our runs with the evidence. At minimum cover: multi-step trajectory data; interaction
   history / state rendering; action-space coverage; resolution and aspect-ratio handling; coordinate
   representation (0-999 grid vs pixels vs normalised); loss for grounding; negative/contrastive samples;
   rejection sampling; an MLLM judge; progress/termination supervision; per-task-type balancing; seed
   averaging. For each **absent** item, state the measured or published effect size it is worth, and whether
   it is closable on one 4090.

## Output format

`docs/research/A5-benchmark-protocol.md`:

- **VERDICT first** — one paragraph: is the current benchmark protocol adequate to support the paper's claim;
  if not, what is the single change that matters most.
- §1 AndroidWorld's requirements vs our compliance (a table: requirement | ours | gap).
- §2 The protocol-spread analysis with the pairs and the specific cause of each gap.
- §3 The protocol we must adopt, as a numbered specification (task count, seeds, budget, metric definitions,
  buckets, what is excluded and why, pre-registration text).
- §4 Power analysis with the arithmetic shown (n=116, paired and unpaired, α=0.05, power 0.8).
- §5 The correct primary metric for the gate thesis, and whether the novelty claim survives CORA.
- §6 The training-ingredient checklist (present/absent/effect size/closable-on-one-4090).
- "What I could not verify" — explicit.

## Rules

- Every number to a URL. Mark abstract-only numbers. Never mix per-step and end-to-end.
- Do not propose a training run that cannot be powered by 116 tasks; say so if none can.
- If the honest answer is "our current claim cannot be supported at this n", write that as the verdict.
- Do not edit any file other than your output.
