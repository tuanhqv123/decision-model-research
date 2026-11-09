# V1 — independent verification of the grounding "fix", before we believe any number

You are a verifier. Your job is to find errors, not to agree. Read and run code.

## Artifacts to check

1. `benchmarks/train/ac_common.py` — function `gold_candidate()` and `build_state()` (the VA branch + the
   `D1_MAX_CANDS` cap + `GOLD_KEEP`).
2. `benchmarks/train/build_candidate_supervision.py` and its outputs
   `benchmarks/results/2026-11-07-candidate-supervision.summary.json`.
3. `benchmarks/results/2026-11-07-candidate-supervision.md`
4. `benchmarks/results/2026-11-07-why-grounding-plateaus.md`
5. `benchmarks/results/2026-11-07-candidate-supervision-{train,val}.jsonl`
6. `benchmarks/data/acv2/500_steps_filtered.json` (398 eval rows with `accessibility_tree`)
7. `backup-2026-10-29/prd-survey-x1-5090/acwork/acdata/compact_cur15k.jsonl` (15k training rows)

## Verify these specific claims, with your own code

1. **"The box loss has zero gradient beyond ~400 bins."** Reproduce it. Then check the claim's *relevance*: the
   loss is per-edge on a 0-999 grid, but is the y-axis really 2400 px mapped to 1000 bins (2.4 px/bin)? Or does
   the model see a resized image (720 px wide in training, `D1_IMG_W`)? If the image is resized before the model,
   the px-per-bin changes and the "400 bins" number may be off by a factor. State the correct px/bin for BOTH
   the training (720-wide) and eval (full-res coordinate space) settings, and say whether the conclusion survives.
2. **"Unscoreable labels 12 → 0 after the fix."** Re-derive this independently on the 398 eval rows. Then check
   the *training* side: does the same fix change how many training rows get a `target_el` label? Compute the old
   and new counts on `cur15k` (spatial rows only) and report both.
3. **"Candidate cap keeps the gold candidate 392/400."** Re-derive on a different slice (e.g. rows 400-800) and
   report the number. Then find the 8 failures and explain what they are — is it a bug in the cap or genuinely
   impossible?
4. **Is the cap itself a confound for the running experiment?** The arm was trained with `D1_MAX_CANDS=14` while
   the E2 baseline was trained with **no cap** (36 candidates median). State plainly whether "VA with 14
   candidates" vs "V with full state" is a clean comparison, and if not, what the correct matched baseline is.
5. **`GOLD_KEEP` is a module-level mutable global.** In `make_sample` it is set per row. Is it possible for it to
   leak stale state across rows, or to be wrong under any data-parallel / worker path? Read the code path and say
   whether this can silently supervise the wrong candidate on some rows.

## Output

Write `docs/research/V1-verify-grounding-fix.md`:
- A table: claim | verdict (CONFIRMED / WRONG / MISLEADING) | your number | what it should say
- The #1 thing in this body of work most likely to be wrong, with evidence.
- An explicit list of anything that would make the running VA experiment uninterpretable.

Be concrete: file:line and numbers. Do not edit any file except your output.
