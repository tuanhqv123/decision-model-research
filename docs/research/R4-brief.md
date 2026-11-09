# R4 — Why did the candidate interface fail, and can it be fixed? (analysis brief)

Read this, then write `docs/research/R4-candidate-interface-postmortem.md`. You may run python on local files.

## The facts

- Interface A (current, "V"): the model predicts a **box** in picture fractions; grounding 56.05%, step 64.07%.
- Interface B ("VA"): the model picks one of N a11y candidates (`target_el` choice). Measured on the same 398
  rows by `benchmarks/results/2026-10-29-evals/E3-300-VA.jsonl`: **element grounding 47.5%, box grounding 43.9%,
  step accuracy 55.5%**. It lost.
- The candidate ceiling is **97.2%** (gold point inside some candidate, 206/212 rows).
- Training data has candidates (median 24/row), and the gold bbox equals a candidate exactly on 83% of rows.
- Prior notes say "combining box and element readouts post hoc does not help (best = element 55.53, oracle
  59.05)" and "mixed V+VA training -2 to -4 points".

## What to determine, with numbers from the artifacts

1. **Compare the two runs row by row.** `E3-300-VA.jsonl` vs `E2-nomem/eval-E2final-nomem-V.jsonl` (both 398
   rows, same episode/step keys). How many rows does each get right, how many are VA-only / V-only, and is the
   union materially better than either? Report the exact counts and a McNemar p.
2. **Was interface B given a fair shot?** E3-300 is **300 steps**; the V baseline is **938 steps**. Check
   `backup-2026-10-29/prd-survey-x1-5090/acwork/runs/E3-cur24k-mix/args.json` and the E3 artifacts for the actual
   schedule, lr, data, and steps. State plainly whether the comparison is confounded by training budget.
3. **Where exactly does the element head fail?** In E3-300-VA, among rows where `el_pt` exists, how often is the
   chosen candidate's bbox the SAME as the gold target bbox, how often is it a parent/child of it, and how often
   unrelated? Use the a11y trees in `benchmarks/data/acv2/500_steps_filtered.json`.
4. **Is the candidate list too long?** Distribution of N per row in eval, and accuracy as a function of N. If
   accuracy collapses above some N, that is a concrete, cheap fix (truncate or two-stage).
5. **Does candidate ORDER matter?** `pick_candidates` sorts smallest-first. Check whether gold candidates sit at
   a systematically different rank than the ones chosen. If the model mostly picks early candidates, the task may
   be biased.
6. **The pairing question**: for the 83% of training rows where the gold bbox IS a candidate, is the candidate
   list at TRAIN time built the same way as at EVAL time? Compare a training record's `cands` to what
   `pick_candidates()` produces from the eval tree for the same row shape. Any train/eval skew is a fatal bug and
   must be reported loudly.

## Output

- Section 1: the row-by-row comparison with exact counts.
- Section 2: a verdict on whether "candidate interface fails" is a scientific finding or a budget artefact.
- Section 3: the failure taxonomy of the element head.
- Section 4: a numbered list of concrete defects found, each with the evidence and the one-line fix.
