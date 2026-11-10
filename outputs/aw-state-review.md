# Review — state of the AndroidWorld programme, 2026-11-10 (Feynman /review shape)

*Artifact reviewed: the programme as it stands — decision record `notes/decision-2026-11-10-androidworld-primary.md`,
audit `docs/research/2026-11-10-AUDIT-androidworld-contract.md`, pre-registration
`docs/research/PRE-REG-2026-11-10-contract-v1.md`, branch `aw-official-harness`, and the two running jobs.
Labels: **verified** = checked against a primary file/source in this session; **inferred** = reasoning, not measured.*

## Summary assessment

The **measurement** side is now sound: the harness reproduces AndroidWorld's `suite_utils` (five deviations found
and fixed, each test-pinned), the image runs upstream `55ddaf5` byte-for-byte, the served checkpoint is asked
exactly the questions it was trained on, and the baseline uses a checkpoint that never saw AndroidWorld data.
The **capability** side is not: the one model change in flight (R1, history contract) does not target the failure
classes the data shows, and the published evidence says those classes are exactly what a single-pass decision
model cannot do alone. Verdict: *keep B0 running; let R1 finish as pre-registered; do not spend further
emulator time on history-format variants; the next decision is architectural, not a tuning knob.*

## Live state (14:04 UTC)

| Job | State |
|---|---|
| B0 (E2, contract e2, official protocol) | 15/116 finished: 2 succeeded, 5 declared done, 10 hit the budget. **No rate is reportable yet.** Preflight 15/15 each round. |
| R1 seed 0 (contract v1) | step 480/938, loss 3.10, train action acc ~0.77, ETA ~45 min; seeds 1–2 queued, each then evaluated on `validation_v1`. |

## Strengths (verified)

- Harness parity with upstream, line by line (H1 float score, H2 seeded instances, H3 initialise-then-home,
  H4 instance budget, H5 one `State` per step); the official SR helper reproduces the hand count 12/80 on old data.
- Contracts travel with checkpoints (`contract.json`); the server refuses an untrained contract or track.
- The serve-time history is produced by the training renderer; no serve-only text enters it.
- Contamination found and avoided: `appint_full` trained on 1,380 AndroidWorld goal rows → not used as baseline.

## Critical issues

**C1 — No standalone single-pass model has a published AndroidWorld result; the one 79% is a hybrid.** *(verified, arXiv 2609.30186)*
Jev-Mobile's 79% uses Qwen3.8-Max as planner: it sets local goals, **writes the exact input text**, and
**decides finish/blocked**; Jev only picks an ID among candidates built by a program from the accessibility tree,
sees the tree as text, not the screenshot, and gets a program-built history of actions actually submitted. The
authors did not test replacing Jev with a small VLM. GUI-JEPA (arXiv 2610.36877; 0.5B, candidate scoring, zero
tokens) reports AndroidControl only — no AndroidWorld number. *Consequence:* our standalone 2B agent is attempting
the three jobs (text, termination, answers) that the only working decision-model system hands to a large
generative model. Any AndroidWorld claim must be framed against that.

**C2 — The documented limits of one-pass decision models match our failure classes.** *(verified, arXiv 2610.24929, 2610.25947)*
- State tracking fails from the second update: accuracy 53–80% at 16 updates, while single lookups stay ~100%
  (2610.24929). Long AndroidWorld tasks are state tracking (which fields are filled, which item is next).
- Decisions that need intermediate written steps (counting, arithmetic) fall from 99% to 50% as demand grows
  (2610.24929). 10 of the 25 answer-only tasks are counts, sums or conversions.
- Records outweigh observations: in all 12 model–domain settings the action record moves the commitment decision
  more than the observation; after a claimed-but-failed step models still stop or advance in 53–100% of cases
  (2610.25947). This is a published mechanism for our **premature `status`** bucket.
*Inferred:* R1 (history format) cannot fix C2; at most it removes an input-distribution shift.

## Major issues

**M1 — R1's expected effect is small relative to what 116 tasks can resolve.** *(inferred)* The pre-registered rule
needs a large paired difference; R1 changes no dominant failure class (C2). Recommendation: evaluate seed 0 online;
run seeds 1–2 online **only** if seed 0's paired difference vs B0 is ≥ 6 tasks — amend the pre-registration
*now*, before any R1 online result exists, and record why (compute value gate).

**M2 — Pixel-box grounding while the contract offers element candidates.** *(verified mechanism; benefit for us
unmeasured)* Both decision-model GUI papers act on program-enumerated a11y candidates (Jev-Mobile §3.2, GUI-JEPA
§3: "a click per enumerated node, an input_text per editable one"). Our V track guesses a point.

**M3 — Custom scaffolding still alters actions.** *(verified on B0 trace)* `adapter.APP_ALIASES` maps the copied word
"photo" to Simple Gallery, so CameraTakePhoto opened the wrong app. Any hand-written mapping between the model and
the benchmark is a confound; remove or measure it after B0.

**M4 — 13/20 AndroidWorld apps never appear in training.** *(verified count; effect confounded with task length)*
Published remedy is in-domain synthesized data (OS-Genesis 9.82→17.41 by its own ablation; OpenMobile).

## Minor issues

- The done-step history line is not written to the trace (`?`); trace-only.
- `Thread-6 (_serve)` exception at chunk end; episodes unaffected; traceback truncated by `tail`.
- AGENTS.md still describes Jev-Mobile as owning "single-pass zero-token GUI action selection"; it is a planner +
  executor system (C1).

## Recommendation — what to decide next (ranked by evidence)

1. **Architectural decision (operator):** standalone single-pass agent vs **planner + single-pass executor**.
   Evidence (C1, C2) favours the hybrid for task success; the standalone framing is only defensible as a
   measured limit ("where a 2B decision model breaks on AndroidWorld"), which is itself a publishable negative.
2. If hybrid: our d1 checkpoint becomes the executor over **a11y candidates** (M2), with a generative planner for
   local goals, text and termination — the Jev-Mobile split, with an open 2B executor and our calibrated gate deciding
   when to hand control back (that gate is the programme's existing contribution).
3. If standalone: the next data is AndroidWorld-app trajectories with per-step completion labels from the
   emulator checker (M4, C2-termination); expect hard ceilings on answer and state-tracking tasks.

## Reproducibility and verification

- Paper claims above: read from full text fetched with `feynman alpha ask` (returns the paper text), quoted
  sentence-level. Files: session scratchpad `alpha/{jevmobile,guijepa,claimlaundering,jevlimits}.txt`.
- Harness claims: tests on branch `aw-official-harness` (harness 174 passed, root 105 passed at last run).
- Not verified: OS-Genesis / OpenMobile headline numbers (papers' own reports); B0/R1 outcomes (running).
- `feynman review` via CLI: one run timed out ("Request timed out."), one stopped before finishing; this review
  follows `prompts/review.md` by hand.

## Sources

- arXiv 2609.30186 — Jev-Mobile: Jev as an Executor for Mobile GUI Agents (Table 1; §3.1–3.4; §7).
- arXiv 2610.36877 — GUI-JEPA (§1–3; Tables 1–3).
- arXiv 2610.25947 — Claim Laundering: When Agent Decision Models Trust Records Over Reality (§4, Table 2).
- arXiv 2610.24929 — What Can't Jev Decide? (§3.3, Fig. 3b, Fig. 5).
- arXiv 2412.19723 — OS-Genesis; arXiv 2604.15093 — OpenMobile; arXiv 2509.18119 — MobileRL (App. D.4.1).
- `google-research/android_world@55ddaf5` (suite_utils.py, episode_runner.py, actuation.py, interface.py).
