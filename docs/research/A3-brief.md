# A3 — Why does the model fail on typing and termination, and what do published agents do about it? (research brief)

Read `docs/research/ISSUES-2026-11-09.md` first (issue I4). Then execute this and write
`docs/research/A3-typing-and-termination.md`. Do not edit any other file.

## Our measured situation (do not re-derive)

1.92B single-pass decision model, 8 typed readout slots (`click, long_press, scroll, type_text, open_app,
navigate_back, wait, status`), zero output tokens, one forward pass. Evaluated on AndroidWorld-116.

**The numbers:**
- `type_text` = **24 of 1,394 emitted actions (1.7%)**.
- 19/104 episodes emitted `type_text` ≥1× → **1 success**, median **25** steps (the budget).
- 85/104 never emitted it → **14 successes**, median **12** steps.
- 34 episodes ran to budget; 31 declared `status` at median step **7**. Same missing signal, opposite symptom.
- `ContactsAddContact`: `open_app` ok → 3 clicks → `type_text` (conf 0.829) → `scroll` → **16 more clicks**,
  never returning to the form. It typed once, then lost the thread.
- Training labels: `type_text` = **450/15,000 = 3.0%**; `status` = **3,300/15,000 = 22.0%**;
  **17.6% of training rows have empty interaction history**.
- A val-fitted threshold on `p(status)` was **monotone-harmful at every τ** (2026-11-07) — a confidence
  threshold cannot manufacture a progress signal.

## Three candidate mechanisms — test each against the literature

**M1 — Supervision imbalance.** `type_text` is 3% of labels while contact/notes/expense tasks need several
consecutive typing steps. Questions: does the field have measured evidence that rare-action under-representation
causes systematic omission in GUI agents? What class-balancing or data-mixing remedies are measured to work —
and, critically, what is measured **not** to work? (We already ran one negative: down-weighting `status` from
22% → 5.3% traded −27 premature terminations for −26 correct ones, val +1.09 p=0.44, test −3.27 p=0.15. Do not
recommend a re-weight; the literature must beat that result or be reported as contradicted by it.)

**M2 — Readout/slot defect.** Before the `app_intent` fix, `open_app app_name` was **empty on 24/24** attempts;
after it, 8/8. A text action needs a field coordinate AND a string. Questions: how do published single-pass /
structured-output GUI models represent text entry — a field coordinate plus a separate string, or an implicit
"type into focused field"? What failure modes are reported for the string slot? Is there published work on
slot-collapse in multi-head readout models? What does the evidence say about typing *without* localising the
field (i.e. click-then-type, or SUBMIT-style implicit focus)?

**M3 — No progress/state model.** The trace shows no memory that the form was already filled. Questions: how do
published agents track task progress — explicit subtask lists, planner/replanner, critic/verifier in the loop,
or learned termination? Which is measured cheapest for a small model? Specifically: what published mechanism
gets reliable "task complete / not complete" at ≤4B, and what are its false-positive and false-negative rates?
(Note: our own measured facts are 34 timeouts and 31 premature declarations on 104 episodes, so termination is
~60% of all failure endings.)

## Also required

4. **Text entry in Android specifically.** Enumerate the real failure surfaces: does the action need the field
   focused first, is `ADBKeyboard`/IME switching needed, does the a11y tree expose an editable flag, does
   setting text directly bypass the keyboard? Ground this in AndroidWorld/android_env documentation and in what
   published Android agents actually do. Our harness must be checked against it — state the check, do not run it.
5. **What would each mechanism predict?** For M1/M2/M3, give the one cheap, decisive measurement that separates
   them, and the predicted outcome under each. If a mechanism is already refuted by our own numbers above, say
   so and drop it.

## Output format

`docs/research/A3-typing-and-termination.md`:
- **Verdict first**: ranked mechanism with the evidence that ranks it, one paragraph.
- Per-mechanism section: literature evidence table (claim | number | source URL | measured or abstract-only),
  and whether our own data already refutes it.
- A "remedies ranked by (expected end-to-end delta) / (GPU-hours)" list, each with the number supporting it and
  the exact change to make. Mark which are SFT-only (feasible on one 4090) vs needing RL/planner infrastructure.
- The Android text-entry mechanics section.
- "What I could not verify."

Rules: cite every number to a URL; mark abstract-only numbers; never mix per-step and end-to-end accuracy;
prefer measured ablations over claims. Semantic Scholar may 429 — fall back to OpenAlex/arXiv/web_search.
