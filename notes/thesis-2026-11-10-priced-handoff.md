# THESIS PROPOSAL 2026-11-10 — what a paper can claim, and what would prove each claim wrong

*Status: **REJECTED by the operator 2026-11-10** — no external model, no planner; one decision model only. Superseded by `notes/thesis-2026-11-10-learning-when-done.md`. Kept for the record.

Original status: PROPOSAL for the operator to accept or reject. Evidence: `outputs/aw-state-review.md` (sources read in full),
`docs/research/2026-11-10-AUDIT-androidworld-contract.md`. Every claim below is a hypothesis until its test runs.*

## 1. What copying Jev-Mobile alone would give — and why it is not enough

Jev-Mobile (arXiv 2609.30186): Qwen3.8-Max plans (local goal, exact text, finish/blocked); hosted, proprietary Jev
picks an ID among program-built accessibility-tree candidates; 79% on AndroidWorld, 32.7% less time and 73.4% less
API cost per success than a step-wise VLM. Rebuilding that with our open 2B model would answer the authors' own open
question ("we have not tested … Jev replaced by a small VLM") — useful, but a replication, not a thesis.

What Jev-Mobile does **not** do, and we have the means to do:
1. Its handoff is the executor picking a DONE/BLOCKED option — **no confidence, no risk control, no price**. They
   report one operating point (one SR, one cost), not a trade-off.
2. It does not ask **which decisions belong to the one-pass model** — it assumes a split. The limits paper
   (2610.24929) predicts which decisions fail in one pass; nobody has tested that prediction on a real agent benchmark.
3. Its executor is proprietary; the decision model's errors cannot be inspected, retrained or calibrated.

## 2. The claims (each with the result that would refute it)

**C1 — Where a standalone one-pass decision model breaks (motivation, a measured negative).**
On the official AndroidWorld protocol, a 2B single-pass model's failures concentrate in the classes one-pass limits
predict — termination/state tracking, text not copyable from the screen, computed answers — not in element selection.
*Data:* B0 (running) with per-failure attribution. *Refuted if* most failures are wrong-element selections on screens
where the right candidate was present.

**C2 — Splitting by those limits recovers task success.**
Planner (generative) for local goals, text and task termination + our 2B model as executor over accessibility-tree
candidates reaches a task success far above C1's standalone number at a fraction of the planner calls of a step-wise
planner. *Refuted if* the hybrid is not better than the planner alone at equal planner calls, or not better than the
standalone model.

**C3 — The core contribution: a priced, calibrated handoff.**
The executor's own confidence, calibrated and risk-controlled (the programme's existing gate: type-routed AUROC 0.729,
split-conformal certificate), decides *continue vs hand back to the planner*. Claim: this policy **dominates fixed
handoff rules** (executor's DONE/BLOCKED option as in Jev-Mobile; hand back every k steps; never hand back) on the
**success-vs-planner-calls frontier**, and its certified executor-error rate holds out-of-sample on AndroidWorld.
*Refuted if* any fixed rule matches it at every planner-call budget, or the certified risk is violated on held-out tasks.
This turns "the gate is not the contribution, the price is" (AGENTS 2026-11-07 l) into a measured trade-off.

**C4 — Local-goal completion is learnable from existing data (enabler, measured fact + untested use).**
AndroidControl's step instructions are local goals: 61,661 of them in 13,603 train episodes; 8,667 (14.1%) span
several actions (e.g. repeated scrolls until a target is visible), giving screen-dependent *continue* negatives and a
*done* positive after every goal's last action. This is the completion signal AndroidControl lacks at task level.
*Refuted if* a completion head trained on it does not beat "always done after one action" on held-out episodes.

## 3. What we would NOT claim

- Not "first single-pass GUI agent", not a SOTA AndroidWorld number, not a comparison with Jev-Mobile's 79% (different
  planner, harness and executor — not comparable; AGENTS M4). Any baseline we compare with is re-run in our harness.

## 4. Minimal experiment set

| # | Run | Answers |
|---|---|---|
| E0 | B0 standalone (running) + failure attribution | C1 |
| E1 | Planner-only, step-wise, same planner | baseline for C2/C3 |
| E2 | Hybrid with fixed handoffs (DONE/BLOCKED option; every-k; never) | C2 + C3 baselines |
| E3 | Hybrid with calibrated handoff at several risk budgets | C3 frontier |
| E4 | Executor completion head trained on AC local goals, offline | C4 |

Executor training needs: candidates extracted the AndroidWorld way (`forest_to_ui_elements`), local goal as the
instruction, history = executed actions (contract v1). Planner: an API model — **operator decision** (cost and an
external service).

## 5. Risks

- Planner quality may dominate everything (C2 then says little about the executor) → mitigate by holding the planner
  fixed across E1–E3; the comparison of interest is between handoff policies, not planners.
- 116 tasks resolve only large differences → report frontiers with paired per-task counts, not single SR deltas.
- Candidate coverage: targets missing from the tree cannot be selected (Jev-Mobile's stated limit) → measure coverage
  separately, as they do.
