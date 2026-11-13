# PLAN 2026-11-10 (FINAL) — one decision model on AndroidWorld: questions, tests, gates, defences

*Constraints (operator): one decision model; no planner; no external model in the system.
Rule of this document: nothing is planned on a belief. Each branch states what is measured, what is assumed, the
cheapest test that settles it, and the decision it drives. The thesis is chosen at Gate 2 by evidence, not now.
Evidence: `docs/research/2026-11-10-AUDIT-androidworld-contract.md`, `outputs/aw-state-review.md`,
`benchmarks/results/2026-11-10-b0-diagnosis/` (frozen scripts). Status: plan only — nothing below is implemented
unless §1 says so.*

## 0. The outcome ("the plane")

A paper whose central claim survives hostile questions about **one decision model acting alone on a live benchmark**.
Outcome metrics: official AndroidWorld SR, and step-level mechanism metrics **with episode-clustered uncertainty**.
Checks, documents and fixes are activity, not outcome.

## 1. What is established

All B0 numbers are **partial and provisional**: arm *B0-overlay* (see F9), 48 scored episodes / 844 executed steps,
tasks alphabetically up to `OsmAndTrack`; uncertainty = bootstrap over episodes. Steps inside one episode are not
independent: the three longest episodes contribute 50% of all no-effect steps.

| # | Fact | Evidence |
|---|---|---|
| F1 | Harness reproduces upstream `suite_utils` (H1–H5 fixed, test-pinned); image runs `android_world@55ddaf5` byte-for-byte (+4 generated `*_pb2.py`) | audit §8; whole-tree hashes |
| F2 | **No-effect steps:** after 53.9% of executed steps the model saw the same screen again (95% CI 35–66%; per-episode median 47%; 22/48 episodes ≥50%). Measured on the images the model received; independent a11y flag agrees (57%); both are blind to small state toggles, so a slight overestimate is possible | 455/844 |
| F3 | **Repeat after no effect:** the first decision taken on a screen the last action left unchanged repeats that action in 68.2% of cases (60/88; CI 55–80%). Pooled over all pairs the figure is 90.9% vs 25.3% on a changed screen, but that pooled number is dominated by long runs and is not the one to quote. 15/40 failed episodes end in ≥3 identical decisions on an identical screen | first-encounter statistic |
| F4 | Success by required length: ≤3 optimal steps 3/10; more than 3 steps 1/38 | `optimal_steps` |
| F5 | Online, action confidence barely separates effective from no-effect steps: AUROC 0.59 (CI 0.48–0.69). *Not like-for-like* with the offline gate (AUROC 0.729 for step correctness, type-routed): different target, different score | 844 steps |
| F6 | Typing works when chosen (text landed 18/18) but is chosen in 2.0% of decisions | `typed_landed` |
| F7 | Three inspected failures show three different mechanisms: semantic mis-selection (toolbar trash = "delete lines" for "delete the note"); a 108 px near-miss, then clicking a text field repeatedly instead of typing; an app name that cannot be copied from the goal. **Repetition is a shared symptom, not one cause** | screenshots + traces |
| F8 | d1 has typed readouts we never used (`number`, `date`, multi-select `choice`, `score`, `boxes`). The 25 answer-only tasks are therefore *expressible* without generation (13 title lists, 6 numbers, 5 single values, 1 yes/no — classification by reading the prompts). Expressible ≠ solvable | d1 README; IR textproto |
| F9 | **H10:** every official agent hides the pointer-location overlay on reset (`m3a.py:365`, `t3a.py:295`, `seeact.py:132`); we did not. Fixed in code (`8bc2f6b5`), **not deployed**: the running B0 is the "overlay on" arm, not the official baseline | source + screenshots |
| F10 | Prior art read in full: loops/inertia are known in **generative** agents (2610.19376: 57.3% of tasks; 2609.37176); a failure record can *raise* repeats in small LMs (2610.30205; a ban on exact repeats stops them but not bad play); decision-model diagnoses are **offline probes** (2610.24929; 2610.25947; 2610.18579: "the facts are read; the sub-goal is not chosen"); the 79% AndroidWorld decision-model system is a hybrid with a Qwen3.8-Max planner (2609.30186). A standalone decision-model result on AndroidWorld was **not found in three alphaXiv searches; the public leaderboard has not been checked for one** | full texts |

## 2. Assumption ledger — how we could be fooling ourselves

| Assumption | Status | Settled by |
|---|---|---|
| Low SR is caused by one-pass decision-making | **Untested; confounded with data** (15k single AndroidControl steps vs the much larger trajectory data behind 45–49% generative 2B results) | Q1 |
| Repetition is specific to one-pass models | **Weakened** — generative agents loop too (F10). Possibly specific: determinism and no self-written memory | Q1, Q2 |
| Writing the action's effect into the history helps | **Contested** by 2610.30205 | Q4 |
| The offline gate transfers to live use | **Not supported; not yet tested like-for-like** (F5) | Q8 |
| B0 is the official-protocol number | **False until B0′** (F9) | B0′ |
| A difference between two runs is a model difference | **Unmeasured** at fixed seed and weights | Q10 |
| Step-level counts give high power | **Only with episode clustering** (F2: CI 35–66%) | §6 rule 1 |
| The diagnosis generalises beyond the inspected tasks | **Unconfirmed** | A2 confirmation set |
| We did not tune on the test tasks | **False by construction:** AndroidWorld has no dev split and diagnosis used its traces | §6 rule 3 |

## 3. The questions

| Q | Question | Cheapest decisive test | Decision it drives |
|---|---|---|---|
| **Q1** | *Architecture or data?* At the same model family and size, same rows, same inputs, same harness: what does one pass lose or gain versus generating the action? | Control arm: fine-tune public `LiquidAI/LFM3-VL-2B` (the model d1-2B is built on) to emit the AndroidWorld action as text. Precise matching: same training rows and state string; equal tuning budget chosen on validation; one pair per input condition (pixels-only first). Disclosed asymmetry: the one-pass arm starts from d1-2B, which has extra decision post-training. **Needs operator approval** (a control arm, not part of the system; licence terms to be read first) | The spine of the paper (§4) |
| **Q2** | Is repeat-after-no-effect structural (deterministic policy, record barely changes) or incidental? | (i) free: R1's first-encounter repeat rate vs B0′. (ii) with a discrete option set: *verified-failure masking* — within an episode, an (screen, action) pair verified to change nothing is not issued again; the next-best option of the model's own distribution is taken. Scope stated honestly: removes period-1 repeats by construction, not longer cycles; `wait` exempt | Headline or symptom |
| **Q3** | Of the no-effect clicks, how many land on **no interactive element** (grounding) vs on a present but wrong or unresponsive element (semantics)? | Measurement-only trace fields: element under the click; per-step element list with state flags (also repairs the toggle blindness of F2) | Sizes the candidate track before the corpus rebuild is paid for |
| **Q4** | Record design: action only / action + verified effect / effect without the action string | One training + one run per variant; first-encounter repeat rate | Whether the record is a contribution or a hazard |
| **Q5** | Why is typing chosen in 2% of decisions — out-of-distribution history? | Free: R1's typing rate and click→type transitions vs B0′ | Whether the contract fix alone repairs data entry |
| **Q6** | Is termination binding once repeats are reduced? | Re-read the outcome table after R1 / R2 | Whether to build the local-goal completion signal (14.1% of AndroidControl local goals span several actions) |
| **Q7** | How many answer-only tasks are reachable with typed readouts? | Gate Q13 first; then an `answer` action built from multi-select / number / date / yes-no readouts | Lifts the 78.4% ceiling or replaces it with a measured one |
| **Q8** | Can a confidence signal be made valid **online** from free labels (verified effect)? Target stated: *no-effect*, a subset of wrong actions — not step correctness | Offline on B0′ readouts, fitted on half the **tasks**, tested on the other half | Whether a usable gate survives the closed loop |
| **Q9** | How much success is locked behind the 13 unseen apps? | Add OS-Genesis AndroidWorld data (MIT on its card; 0 of 7,329 instructions ≥0.7 token-Jaccard with a test template) to **both** arms; disclosed as in-domain apps | Final competence level; not a novelty claim |
| **Q10** | Flip rate at fixed seed and weights | Re-run 20 tasks of B0′ | The smallest paired difference we may claim |
| **Q11** | What do trivial policies score? | `status` at step 0; `open_app` then `status` (one short 116-task run) | Floor; finds tasks satisfied by doing nothing |
| **Q12** | Does the overlay change behaviour? | B0-overlay vs B0′, read against Q10 | Caveat on older numbers |
| **Q13** | Do `number` / `date` / multi-select work on GUI screens at all? | 50 hand-checkable AndroidControl screens, zero-shot and after SFT | Gate for Q7 |

## 4. Thesis candidates — chosen at Gate 2

| | Claim | Selected if | Dropped if |
|---|---|---|---|
| **T-A** | *Matched comparison.* At equal family, size, data and harness, one pass keeps generation's short-task competence at lower end-to-end latency and loses it beyond a measurable task length | Q1 shows a length-dependent gap | Generation is uniformly far better → the paper reports that cost as a measured negative |
| **T-B** | *Closed-loop dynamics.* Offline probes cannot see what a one-pass policy does when its own action changes nothing; a discrete option set plus verified-failure masking removes period-1 repeats and shortens wasted steps | Q2-ii cuts first-encounter repeats and lifts SR beyond the Q10 floor, and the control arm repeats less or differently | The control arm shows the same repeats and the same fix works equally → incremental over DeLoop; keep as a section, not a claim |
| **T-C** | *Confidence under own-action shift.* Offline-valid confidence does not flag no-effect actions online; environment-verified labels restore a usable signal | Q8 task-split AUROC ≥ 0.70 | Q8 near chance → reported as a documented limit of the offline gate |

- **If Q1 is approved:** spine = T-A (first standalone, official-protocol, matched measurement of a decision model as
  an agent), with T-B and T-C as mechanism sections kept only if their gates pass.
- **If Q1 is refused:** spine = T-B + T-C as a diagnosis-and-remedy paper, with the architecture-versus-data question
  stated as an open limitation; published generative numbers cited only as non-comparable context.
- **Never claimed:** state of the art; "first decision-model GUI agent"; any ranking against numbers from another
  harness; a latency advantage measured on the model alone.

## 5. Stages and gates

**Stage 0 — already running.** B0-overlay completes → Q12 arm and the A2 confirmation set. R1 seeds 0–2 train and are
scored on `validation_v1` (sanity only; not comparable with E2's validation number).

**Stage 1 — about one emulator day, no new training.**
1. Deploy H10; run **B0′** (E2, overlay hidden) = the official-protocol baseline.
2. Q11 trivial policies; Q10 repeat of 20 tasks.
3. Q3 measurement fields.
4. **R1 seed 0** online vs B0′ on the pre-registered mechanism metrics (Q2-i, Q5); seeds 1–2 online only under amendment A1.
5. Offline: Q8 on B0′ readouts; Q13 probe; leaderboard check for F10.

**Gate 1.** Inputs: outcome and mechanism tables for B0′ and R1; Q3 split; Q8; noise floor; A2 confirmation.
A branch (R2, R3, Q7, Q9) is cut if its sized upside is below the noise floor or its premise failed confirmation.

**Stage 2 — the expensive builds, in parallel.** Q1 control arm (if approved). R2: AndroidControl re-extracted the
AndroidWorld way (`forest_to_ui_elements`, state flags) → candidate model → runs with and without masking (Q2-ii).

**Gate 2.** Choose the thesis per §4; cut the rest.

**Stage 3 — only what Gate 2 keeps.** Q4, Q7, Q9 as selected; three training seeds for every surviving claim;
confirmatory runs on held-out task seeds (§6 rule 3); then write.

## 6. Measurement rules (binding)

1. **Clustered statistics.** Step-level metrics are reported with bootstrap intervals over episodes and with
   per-episode summaries; pooled step counts are never quoted alone. First-encounter repeat rate replaces the pooled
   repeat rate.
2. **Pre-registration.** Every online run fixes its metrics, comparison and kill rule before it starts; amendments
   are timestamped and made only before the affected result exists.
3. **No clean dev split exists, so three layers:** (a) every diagnosis is confirmed on tasks not yet inspected
   (amendment A2); (b) final claims are re-run on task seeds never used during development (31 and 32), reported
   beside the official seed 30; (c) anything *fitted* on AndroidWorld data is fitted and tested on disjoint tasks.
   Residual weakness, stated in the paper: the task templates themselves are shared across seeds.
4. **Trivial baselines and exclusions in every table** (Q11; tasks that raised).
5. **One variable per run; contracts travel with checkpoints** (`contract.json`; the server refuses mismatches).
6. **Latency is end to end**, environment time included; a model-only speed-up is not reported as a benefit.
7. **A mechanism counts as explained** only when a clustered count on our traces says how many failures it covers.

## 7. Hostile questions

| Question | Where we stand |
|---|---|
| A generative 2B scores 45–49%. Why one pass? | Unanswered until Q1. No claim about this is made before then. |
| Is repetition just your weak model? | Unknown in part. Measured: 68% repeat at first encounter. Generative agents also loop (F10). Q1 and Q2 separate weak from structural. |
| Is the 54% no-effect figure three bad episodes? | Three episodes give 50% of the steps, which is why the interval is 35–66%; but 22/48 episodes have ≥50% no-effect steps and the per-episode median is 47%. |
| Is masking a harness trick? | It is part of the agent, uses only the model's own distribution and a verified environment fact, removes only period-1 repeats, and is reported with and without. |
| Did you tune on the test tasks? | Yes, unavoidably; §6 rule 3 is the defence and its residual weakness is stated. |
| You give your model the element list; baselines are pixels-only. | Comparisons are made only within an input condition; Q1 pairs are matched per condition. |
| The one-pass arm starts from a checkpoint with extra post-training. | Disclosed; it favours the one-pass arm, so a one-pass deficit is conservative and a one-pass advantage must be read with that caveat. |
| Why did confidence stop working online? | F5 is our own negative and is not yet like-for-like; Q8 is the test; a failure is reported as a limit. |
| Does the latency advantage reach the user? | Only partly: the environment costs about a second per step, so the end-to-end gain is far smaller than the model-only one. Rule 6. |
| One benchmark only? | Yes. External validity is a stated limitation; a second live benchmark is optional work after Gate 2. |
| Are your numbers comparable with anyone's? | No, and we say so; every comparison is between arms run in the audited harness. |

## 8. Decisions needed from the operator

1. **Q1 control arm** — approve fine-tuning LFM3-VL-2B generatively as a baseline (same family, no API)? It decides between the two spines in §4.
2. **Q9** — is public in-domain data (OS-Genesis, synthesized in the AndroidWorld emulator by its authors) acceptable as a training source, disclosed as such?
3. **Emulator time** — Stage 1 needs about one day of sequential runs on the shared emulator host.
