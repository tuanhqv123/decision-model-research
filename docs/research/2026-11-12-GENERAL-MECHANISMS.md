# Replacing "fixed tools" with general mechanisms: drawn from TypeSafe's cookbook (read directly 2026-11-12)

Source: docs.typesafe.ai, with 11 cookbooks, 2 patterns and the Jev 1.13 limits page. The text version is in the scratchpad; the README version is saved in `references/raw/`.

## The rules TypeSafe wrote for decision models (quoted verbatim)

1. **Re-ranking: one question per (query, candidate) pair.** "uses one System One question per query-candidate pair to raise top-1 accuracy from 5% to 30% and top-10 accuracy from 38% to 65%" (40 queries, 30 candidates per query).
2. **Progressive disclosure, a two-pass check** (skill_suggestion): read all 194 choices cheaply, then "re-reads only the top three, now with each skill's full description … and is free to reject". Result: "cut wrong skill loads by half and needless ones by nearly two thirds".
3. **Let code count and compute; the model only judges each item:** "Jev does not count reliably … When you want to count items matching some criteria, iterate in code over the candidates and ask one question for each, then add up the answers yourself."
4. **Dates:** "Jev reads dates as text, not as ordered quantities … Every part of a date is a small closed set … Code assembles the parts into a real date and owns everything after that, including ordering, duration, offset, and weekday."
5. **Literal understanding:** "Where interpretation is unavoidable, split it into two literal questions and combine them in code."
6. **Fan-out:** "putting all of the questions your system needs in a single request, and then using code to decide what is relevant … adding more questions usually has little effect on response time." Asking together or asking separately gives the same result (parallel_questions: standard deviation 0.0 on most questions).
7. **Confidence is the second axis:** "The score tells you what; confidence tells you whether to act on it." classification_using_confidence: the confident group is right 93% of the time, the rest 50%; this remainder is reported at a higher classification level.
8. **Check each rule with noul** (sde_cascade): "ask yes/no questions … one per rule it must follow", then code decides.

## Three general mechanisms for our agent (replacing fixed tools)

### M1. Two-level element selection: shortlist, then score each candidate (rules 1 + 2)
- **Now:** one choice question picks from a list of 20–70 elements. Clicking the wrong target is **435/1,199** of first errors on AC-High; on AndroidWorld it is "wrong way of operating the app" and "wrong element chosen".
- **Do:**
  - **Level 1** takes the top-k (for example 5) from the existing choice question itself.
  - **Level 2** asks **one noul per candidate**: "Does the next action act on this element?". The level-2 input is a compact state, only that element, its row and its context.
  - Code picks the candidate with the highest probability. If no candidate is confident enough, fall back to the level-1 answer.
- **Labels:** taken directly from AndroidControl (gold element = yes, the other candidates = no). No synthetic data needed.
- **Existing evidence:** measured on 2026-11-07: a shortlist built from the model's own predicted point contains the correct answer in **78.5%** of rows, while current grounding is **56%**. The two-level design was pre-registered then but not run.
- **Can be measured offline right away** (grounding on validation and AC-High) before running AndroidWorld.

### M2. "Map–reduce" for answers and every computation (rules 3 + 4 + 5)
This replaces the date table, calculator, note and duplicate-marking. One general mechanism, not written separately for each kind of task.
- **The model judges each item** (fan-out, in one pass):
  - one noul for each row on the screen: "Does this row meet the condition in the goal?";
  - one choice "what kind of answer does the goal ask for": {one item, count, sum, list, yes/no, date, no question}.
- **Code gathers the results:**
  - accumulate the rows that meet the condition across scrolls (keyed by the row's text, so duplicates drop out by themselves);
  - split numbers, durations and dates with regex (cookbook pre-parsed value);
  - compute counts, sums, unit conversions;
  - join lists with commas (the exact format the AndroidWorld checker requires);
  - match duplicates between records;
  - format and then send `answer`.
- **Dates in forms:** the model picks the parts of the date (relative/absolute, weekday, month, day). Code computes the real date from AndroidWorld's fixed clock (15/10/2023).
- **Error groups targeted:**
  - information questions: 21 tasks (sees it but does not answer, counting, adding, multi-item lists, dates);
  - finding duplicates;
  - date/time form tasks.
- **Labels:** the "does the row qualify" question needs synthetic data from the lists in AndroidControl, using literal conditions built from the row's text. This is the riskiest part.

### M3. Verify progress with the next screen (rules 7 + 8)
- **Do:** after each action, at the next pass ask one more noul: "Did the last action bring the goal closer?". If the model is **confident** that it did not, code presses Back and tries the second candidate.
- **Different from the "is it done yet" question that already failed:** that question shares its label with the `status` action and only looks at the screen *before* acting. This one looks at the screen *after* acting, that is, it uses new information that a single read pass does not have.
- **Labels:**
  - progress = the triple (screen t, gold action, screen t+1) in AndroidControl;
  - no progress = wrong branches taken from other episodes, already built in `build_recovery_data`.
- **Error groups targeted:** looping until the budget is exhausted (40 tasks), choosing wrongly and then moving on.
- **Risk:** going back wrongly. So Back is decided by code only when the model is highly confident (per rule 7), and is not an action label.

## Not used
- **Fixed tools for each kind of task:** date table, note, duplicate-marking as separate items. M2 replaces them all.
- **Clipboard tool:** a privileged channel.
- **OCR:** it is an external model.
- **A threshold on p(status):** measured to be harmful at every threshold (M12).

## Proposed order
1. **Final run 3 (training)** fixes early stopping, the largest error group. Measure it first.
2. **Do M1 first.** The labels come ready from AndroidControl, it can be measured offline right away, and it helps every task.
3. **Do M3 next.** Labels from AndroidControl plus the wrong-branch data we already have, measured offline with AUROC on a held-out set.
4. **Do M2 last,** for the information and computation group. Synthetic data, the highest risk, but it unblocks the most within the blocked group.
5. **Train once for M1 + M3** (both are new questions on the AndroidControl data), with an early check, then pair against final.

## Current plan (2026-11-12, afternoon)

1. **New v4 baseline + harness V14, running.** All 116 tasks on 6 fixed emulators (Markor, OsmAnd, SIM). V14 = answer format + keyboard hiding. Model v4 is served on GPU_HOST_A. This run answers two questions: do the old environment errors remain, and how much does the harness gain when the model stays the same.
2. **Final run 3 (contract v6), training, no more stopping.**
   - The early-stopping rule at ckpt_250 was refuted by our own data. In the v3 run, the mid-run metrics go up and down strongly and do not predict the final result (ledger M17).
   - Only gate 1 on the final checkpoint decides.
   - The trend of the intermediate checkpoints will be re-measured after training, because the checkpoints are kept.
3. **Test the patterns with the current model, no training** (`benchmarks/train/probe_patterns.py`, v4, 827 validation rows):
   - **M1:** re-rank the top-5 elements with one noul per candidate, compared with choosing by choice;
   - **M3:** AUROC of the noul "did the last step get closer to the goal?". Positive samples are the screen after the demonstrator's step, negative samples are the screen after a wrong branch.

   If there is a signal, then put it into training (once, M1 + M3). If not, record it as a negative result.
4. **After that:** run the 13 targeted tasks, then 116 tasks for final with V14, paired against the baseline. The paper uses the best number measured.

## Review 2026-11-12 (on the request "check again carefully")

**What was checked and is correct:**
- **v4 weights:** md5 `fa8ba04b…` is identical on all three machines (GPU_HOST_A `init_v4`, the training machine `v4_s0/ckpt_2482`, the serving machine `v4_s0_ckpt2482`). The v4 server used for the targeted run is exactly the model that ran in v4 + V12.
- **Harness policy recorded in the episode:** `answer_format: true`, `hide_keyboard_after_type: true`, `fresh_wait_s: 20`, `mask: false`.
- **Keyboard hiding really runs:** 19 typing steps, 17 times the keyboard was hidden afterwards. The other two times the keyboard did not show.
- **Environment:** each container reports `APP_OK` for markor and osmand_markers at startup.

**Errors found and fixed in this review:**

| # | Error | Consequence if not fixed | Done |
|---|---|---|---|
| 1 | Preflight hard-codes port 8701 | Cannot run any server other than 8701 | The URL is taken from `model.url` in the config, committed |
| 2 | 6 chains write to the same `/tmp/preflight.out` | The preflight results of the emulators overwrite each other | Each emulator writes its own file |
| 3 | Re-running all 116 tasks for v4 + V14 | About 2.5 hours wasted | Run only the 39 tasks that are really affected; 73 tasks keep the v4 + V12 result (will be declared clearly when reporting the number) |
| 4 | **Final training takes the GPU shared with the customer: 31.9/32.6 GB, only 0.7 GB left** (the customer's service grew its memory, and a new service was added) | The customer's service may go OOM | Stopped training at step 630; move to GPU_HOST_A (24.5 GB, almost empty) after the 39-task run finishes |
| 5 | GPU_HOST_A has 21 GB of disk left, but keeping every checkpoint needs about 30 GB | Disk fills up during training | Deleted my own old pilots and checkpoints (the weight merge is still on the serving machine). Final saves one checkpoint, overwritten every 500 steps |
