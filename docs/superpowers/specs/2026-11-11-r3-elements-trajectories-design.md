# R3 — element choice + full trajectories + screenshot (design, 2026-11-11)

## Intent (operator, 2026-11-11)

The model must act the way AndroidWorld's own agents do: **choose an element from the accessibility tree**, and the
code taps that element — the model never decides pixels. It must be trained on **whole trajectories** (every step of
every episode), with the **element tree** and the **screenshot** of each step. One decision model only.

Why (measured, `docs/research/2026-11-11-DIAGNOSIS-why-low.md`): R1 gets no element list; 30–37% of its online clicks
hit something not clickable; 29–36% of its `open_app` names cannot be launched because the name is copied out of the
goal and `gold_answers` gives NO supervision when the app is not named in the goal (`ac_common.py:715-732`); it trains
on 15k of ~88k AndroidControl decisions for one epoch with a history line that never says what was clicked.

## Goals

1. **Element choice (contract v3 ⊇ v2):** the `target_el` choice over AndroidWorld's served element list
   (`forest_to_ui_elements`, vendored in `aw_extract.py`), rule `cap40_reading` (measured 2026-11-10); the executed
   tap is the chosen element's centre. No `target_box`.
2. **Full trajectories:** every step of every AndroidControl **train** episode (≈88k decisions incl. the final
   `status`), plus every step of the validation episodes, each with screenshot, element list, gold action, gold
   element, history, effect labels. AndroidControl test episodes stay excluded (leak-check as in `preprocess_ac.py`).
3. **History that says what happened:** each history line names the element acted on and whether the screen changed
   (`click "Save" (changed)`, `type_text "Bob" into "Name" (changed)`, `click "Back" (no change)`). Built by ONE renderer
   from gold+`effect_app` at training time and from executed element + `ui_changed` at serving time.
4. **App choice:** `app_name` becomes a `choice` over a list of launchable app names given in the request
   (serving: AndroidWorld's launchable names; training: the gold app + sampled distractors from the corpus app
   vocabulary), so a goal that does not name the app can still be answered.
5. **Screenshot stays** as the image input (720 px width unless the resolution test in the plan says otherwise).

## Also in scope (operator, 2026-11-11: one plan, no phases)

- `answer` action (span copy from goal/element labels + a number choice) for the 25 information-retrieval tasks.
- Trajectories collected in the AndroidWorld apps (rollout collector, checker-verified), MiniWoB rollouts,
  self-training rounds, and an unlikelihood signal on premature `status` from failed rollouts.
- No generated text, no second model at inference.

## Anti-goals

- Training on AndroidWorld's 116 evaluation tasks or their generated instances.
- A training-time element list or history line produced by different code than the serving one.
- Claiming an R3 result before the pre-registration, 3 seeds and the paired analysis (same rules as R1).

## Constraints

- Training host: one 24 GB 4090 (shared); 45 GB disk free; images for all episodes already on it (178,288 jpg, 11 GB).
- Raw AndroidControl TFRecords (47 GB, 20 shards) are on the emulator host with the v2 extractor and its venv.
- Every claim needs ≥3 seeds and an interval (AGENTS M6); pre-registration before any online number.

## Decided by the builder (my calls; cost if wrong)

- Keep v2's element rule `cap40_reading` and its option texts — measured; cost: a few % gold-not-in-list rows.
- Distractor count for app choice in training: 24 (≈ size of AndroidWorld's app list); cost: a mismatch between the
  number of options seen in training and at serving.
- Rows whose gold element is not in the kept list keep their action supervision and get no element supervision (as
  v2); cost: those rows teach nothing about the target.

## Open decisions (operator)

- **D1 — decided 2026-11-11: yes**, practice instances of the 116 templates (never the 116 test instances), disclosed, with one third of the templates never practised to measure unseen templates.
- **D2 — decided 2026-11-11: no.** No external model at data time either; AndroidWorld-app trajectories come only from our own model's checker-verified rollouts.
- **Typing — decided 2026-11-11:** text is copied by position from the goal (as in v1/v2).
- **D3 — decided 2026-11-11: yes**, the `answer` span may be copied from on-screen element text (IR answers are never in the goal).
- Questions that test this design: `docs/research/2026-11-11-R3-QUESTIONS.md`.
