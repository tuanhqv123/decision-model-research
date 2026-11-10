# DECISION 2026-11-10 — AndroidWorld becomes the single primary benchmark; follow its contract exactly

*Decision record for the paper's Method / Experimental-protocol / Limitations sections. Evidence for every
line is in `docs/research/2026-11-10-AUDIT-androidworld-contract.md` (sections cited as §N). Upstream
source = `google-research/android_world` @ `55ddaf5` (2026-10-23).*

## 1. The decisions

| # | Decision | Why (evidence) |
|---|---|---|
| K1 | **AndroidWorld (116 tasks, official protocol) is the only benchmark we optimise.** AC-v2 64.07% is kept as a past, teacher-forced result and not optimised further. | AC-v2 feeds the human-written step instructions of all past steps as history (`previous_actions` = oracle `step_instruction`, 398/398 rows) and is screenshot-only; AndroidWorld gives no such history and does provide the element list. One model cannot be trained to both input contracts. Operator decision, 2026-11-10. |
| K2 | **The harness must reproduce `suite_utils` exactly** before any number is reported: seeded instances, initialise-then-go-home order, float scores, done gate, per-task budget, NaN-excluding average. | Code review §8 below found three deviations that change which tasks are run and how they are scored. |
| K3 | **Train / eval / serve share one input contract that AndroidWorld can supply at run time**: screenshot, goal, the element list as `forest_to_ui_elements(exclude_invisible_elements=True)` produces it, and a history rendered from *executed* actions. | Upstream contract (§1); our training extracted every node (§2 D2) and used human instructions as history (§5.1 T2). |
| K4 | **Nothing is built on an unmeasured hypothesis.** Every model-side change is preceded by an offline count that sizes it, and is changed one at a time. | §7 self-review: four earlier claims were withdrawn or downgraded on a slow re-read. |
| K5 | **Training data on AndroidWorld's own apps is needed, and its origin must be disclosed.** Options: public synthesized sets (OS-Genesis AW, MIT; OpenMobile, data licence unstated) and/or own emulator rollouts on non-test instances. Which one is an open decision. | 13/20 AW apps have zero training goals (§2 D6); AndroidControl has no non-terminal completion labels (§5.1 T1). Effect not yet measured (§7). |

## 2. What the paper may and may not claim from this

- **May** state as facts: the AW action/observation contract, the 25/116 answer-only tasks (ceiling 78.4%
  without `answer`), the done-gate rule, the AndroidControl termination structure (13,603/13,603 terminal
  `successful`), the extractor mismatch, and the harness deviations found and fixed.
- **May not** state, until measured: that unseen apps, bare-click history, the extractor mismatch or the
  double-tap explain the failures (all HYPOTHESIS in §7).
- **Every previously reported AndroidWorld number** (the 116-task run, the 8-task runs) was produced on
  **unseeded, non-official task instances** (§8 H2) and must be labelled as such or re-run.
- The harness deviations found in the line-by-line review (§8): H1 float score collapsed to bool (0.5 → 1),
  H2 unseeded instances, H3 reset-before-initialise instead of initialise-then-home, H4 class-level
  complexity for IR budgets, H5 screenshot and element list read from two different `get_state` calls.
  H6–H8 (denominator, done gate, per-task budget) were checked and **match** upstream.

## 2b. "Where is the plane?" — the cargo-cult check for this programme

Feynman's cargo-cult test applied to ourselves: lots of runway, control tower and ritual; did a plane land?

| Activity we have plenty of | The outcome it is supposed to produce | Has it landed? |
|---|---|---|
| 158 record checks, 69 paper assertions, ~280 tests, many dated notes and handoffs | correct numbers about the model | Partly. They caught drift between documents, but **not** that every AndroidWorld number ran on unseeded instances, that half credit counted as full, or that the served checkpoint was asked questions it was never trained on. The checks verified the records against each other, not the harness against the benchmark. |
| Fixes, gates, pre-registrations | an agent that completes AndroidWorld tasks | **No.** 12/80 scored tasks under the done gate, on non-official instances; every success is a 1–2-action task. |
| AC-v2 64.07% | evidence the model can act on a phone | Weak: AC-v2 feeds the human-written instructions of all past steps; AndroidWorld never does. |

Rules this adds (each is a question to answer before claiming progress):
1. **Name the outcome, not the activity.** The outcome is official AndroidWorld SR (and, for the paper, a gate whose risk holds on it). A passing check, a new document or a new head is activity.
2. **Write down what would prove us wrong before running.** For R1: if the three seeds show no measurable paired difference from B0, the "history contract" explanation of the failures is not supported — the fix is hygiene, not capability, and the next bet must change.
3. **Do not learn from lucky outcomes.** The 1–2-step successes say little about the model; a decision is judged by its process and its pre-registered test, not by one good episode.
4. **A named mechanism is not an explained one.** "Grounding", "termination", "exposure bias" are labels until a count on our own traces shows how many failures each explains.

## 3. Corrections to earlier records made by this review

- HANDOFF-2026-11-10 P4 ("eval_d1 drops history; 64.07 is a lower bound") — **false** (§3).
- A5 §1 "we divide by scorable (80) — wrong denominator" — **false**: upstream averages with pandas
  `mean()`, which skips the `NaN` that `_create_failed_result` assigns to any task that raised
  (`suite_utils.py:544-558, 680-716`). Excluding setup/checker exceptions *is* the official denominator.
- aw116 "15 successes" — **12** under the done gate (3 never declared `status`), i.e. **12/80 = 15.0%** on
  the official definition, still on unofficial instances.
- "episodic-s0 lost −8.29 because `is_done` consumed capacity" — confounded by a history-format change
  (bare actions in training vs human instructions at eval).
