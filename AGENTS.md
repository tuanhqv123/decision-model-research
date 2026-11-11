# AGENTS.md — Decision Model Research Program

This file is the **master map** for this research workspace. It tells any agent (and any human) what we're working on, how the topics relate, what phase we're in, and the rules for keeping citations and references lossless.

---

## 1. The thesis (what this research is about)

We research and build **decision models** (TypeSafe calls this class "System One models"). Unlike an LLM that generates text, a decision model takes unstructured state (text, JSON, images, and eventually audio) and returns a **typed, calibrated decision** — a choice, a score, or a yes/no — with a probability, in one forward pass, with **zero output tokens**.

Core properties to study and reproduce:
- **No text generation** → no parsing, no hallucinated strings, no type errors.
- **Calibrated probabilities** → confidence actually tracks accuracy.
- **Single-pass, parallel sampling** → very low latency (tens of ms on edge).
- **Structured output** → software can branch on it directly (classify / route / score / guardrail).

### Two anchors
| Anchor | Role | Key trait |
|---|---|---|
| **Jev** (TypeSafe AI, proprietary) | The frontier **target** we want to match. | RLCD-trained; 70–500 ms; ~200x faster / ~400x cheaper than LLMs on System One tasks; "can't hallucinate" (schema-guaranteed). |
| **d1 / d1-3B** (Liquid AI, open weights) | The **blueprint** we study and reproduce. | Post-trained from LFM2.5-VL-3B; Decision Index v0.2.1 = 48.57; 8 ms on RTX 4090, 16–50 ms on Jetson; llama.cpp day-one. |

**Our goal:** understand the category → build our own open decision model by post-training/fine-tuning an open foundation model → benchmark it → deploy it to mobile/edge → write a paper.

> **North star (REWRITTEN 2026-11-03):** `notes/GOAL.md` — **build a small, honest, fast decision model a mobile agent can be trusted to act behind, and prove on one fixed public benchmark (AC-v2, 398 steps) both that it is competitive (now **64.07%**, 255/398 — a tie; the served-path cross-check is 63.82%) and that its confidence is usable for a risk-controlled act/ask/refuse gate (now: type-routed AUROC 0.729, 20% selective risk certified on val at 47.6% coverage; test 40.7% @ 17.3%).** Current numeric objective: **risk at 50% coverage < 10%** (today **19.1%** with the learned fusion, 21.5% hand-routed). Paper: `papers/single-pass-decision-acv2.md`, framed as a reliability/mechanism paper (ICML 2027 / TMLR) — *not* an accuracy-SOTA claim. GOAL.md states only measured facts and carries the full retraction list; **Q1/Q2/Q3 from the 2026-10-13 version are superseded** (Q2's CE-vs-proper-scoring mechanism returned nulls; Q3's Apple-silicon venue was never measured). Competitive map: `notes/competitive-2026.md`. If any other file disagrees with GOAL.md, GOAL.md wins.

---

## 2. Research topics and their relations (the concept map)

```
                  ┌─────────────── DECISION MODEL (System One) ───────────────┐
                  │  no text; single forward pass; calibrated typed decision   │
                  └────────────────────────────────────────────────────────────┘
                        │                            │
        ┌───────────────▼─────────────┐      ┌───────▼────────────────┐
        │  Jev  (TypeSafe, 专有)       │      │  d1 / d1-3B (Liquid)    │
        │  frontier target            │◄─────│  open blueprint         │
        └─────────────────────────────┘      └────────────────────────┘
                     │                                    │ (post-train LFM2.5-VL-3B)
                     │                        ┌───────────┼──────────────┐
                     │                        │           │              │
              ┌──────▼──────┐        ┌────────▼──────┐  ┌─▼──────────┐  ┌─▼─────────────┐
              │ LLM2Jev 微调 │        │ 校准评测      │  │ 边缘/移动端  │  │ 智能体决策层   │
              │ → 构建自己   │        │ Decision Index│  │ Jetson/ONNX │  │ → 移动交互     │
              └─────────────┘        └───────────────┘  └────────────┘  └───────────────┘
```

Topic list (each maps to a file under `notes/`):
- `notes/decision-contract.md` — **foundational** — the single-pass input/output contract (state + typed questions → probability vector + confidence, zero tokens). Read this first.
- `notes/jev.md` — Jev: RLCD, architecture, calibration, speed/cost.
- `notes/d1.md` — d1-3B / d1-omni-600M: architecture, training recipe, benchmarks, edge latency.
- `notes/llm2jev.md` — fine-tuning open LLMs into decision models (our build path).
- `notes/calibration.md` — ECE / Brier / reliability diagrams; why calibration is the key metric.
- `notes/benchmark.md` — Decision Index semantics + our custom mobile/edge benchmark.
- `notes/deployment.md` — llama.cpp / CoreML / ONNX / MLX; latency & quantization on mobile.
- `notes/agents.md` — decision models as an agent's action-selection layer (mobile interaction).
- `notes/mobile-agents.md` — mobile-agent SOTA, benchmarks, failure points (where we improve).
- `notes/action-selection-feature.md` — the chosen feature: calibrated on-device next-action selection + confidence gate.
- `notes/EXPLIGNED.md` — PLACEHOLDER
- `notes/GOAL.md` — **north star** — the consolidated goal, why/how, resource check, paper title, risks, decision log. Read this before planning.
- `notes/competitive-2026.md` — 2026 competitive landscape (THX-01, decider, TypedBench, AgentAbstain, CORA, CAAG, Re-TRAC) and the exact gap we own.
- `notes/sota-arch-deep.md` — **architecture deep-read** (2026-10-14): Mem-W / UI-Venus-1.5 / UI-Venus-2 / GUI-Libra — reusable modules, exact training recipes, targetable gaps, and the recommended us-vs-SOTA build order (Steps 0-7 + pass conditions P0-P4). This is the `reuse arch + enhance` plan.

**Relations that matter** (derive from the map):
1. Jev ↔ d1: they define the category's performance frontier and its open blueprint.
2. LLM2Jev ↔ "build our own": the direct engineering route.
3. Calibration ↔ benchmark: Decision Index measures decision quality *and* calibration.
4. Deployment ↔ latency: single-pass + edge = the mobile use case.
5. Agents ↔ application: "use this model to interact on mobile."

---

## 3. Phases (track current phase here)

- [x] **Phase 0 — Setup** (structure, roadmap visual, this map).
- [x] **Phase 1 — Understand** (Jev + d1 + category; notes for Jev/d1/LLM2Jev/calibration/benchmark/deployment written; see `notes/`).
- [ ] **Phase 2 — Build** (pick base open model; post-train / fine-tune into a decision model → `models/`).
- [ ] **Phase 3 — Benchmark** (evaluate on Decision Index + custom mobile bench → `benchmarks/`).
- [ ] **Phase 4 — Deploy** (edge/mobile: llama.cpp / CoreML / ONNX / MLX, latency profile → `mobile/`).
- [ ] **Phase 5 — Paper** (optimize, organize, write → `paper/`).

**Current phase:** Phase 2 (build) → **phase tracker RESET 2026-11-03 by the GOAL.md rewrite. Read `notes/GOAL.md` first; it is authoritative and now states only measured facts.** Position: CAC-v2 offline analysis layer is **DONE and positive** (type-routed gate AUROC 0.649→0.729 held-out, stable across 3 evaluations; split-conformal certifies **20% selective risk on val at 47.6% coverage**; the test operating point is 40.7% coverage at 17.3%), the accuracy number is a **tie** at **64.07% (255/398; no artifact yields 256)**, and the current numeric objective is **risk at 50% coverage < 10% (today 19.1% learned fusion, 21.5% hand-routed)**. The two open limits are measured negatives, not bugs: a 10% budget at ≥50% coverage **does not exist** (risk floor ~12.5%), and the conformal threshold **does not transfer across checkpoints** (17.3% → 23.9%). The 2026-11-02 "pivot to the online AndroidWorld loop" is **superseded** — the online loop is torn down, pixels-only, and **1/8 in both conditions** (Wilson 2.2–47.1%, underpowered); its 2/8 "crashes" were Clock-task checker exceptions on a live device, now fixed; it is out of scope until offline stages 2–3 deliver ≥ +4 pts. **The V2P-σ + ConClick grounding plan remains INVALIDATED — do NOT spend GPU on it.** Also corrected 2026-11-03: the "real gap is element selection, not pixel precision" note is only half right — element errors (68) and type errors (63) are **comparable and the difference is not significant**.

**Historical (pre-2026-11-03, kept for provenance):** [previous text] Phase 2 (build) — PIVOTED 2026-11-02 to the ONLINE AndroidWorld/AndroidLab loop (`docs/HANDOFF.md` §3). AC-v2 offline was judged plateaued at 64.3–64.8%. Box grading penalizes valid alternatives — but note the **scope correction**: `2610.36831`'s 1/5–1/3 out-of-box equivalence is measured on **web** widgets; on **native Android widgets it is ~1%**, so on AC-v2 this is a *small* artifact. **The category framing is TAKEN:** `Jev-Mobile` (`2609.30186`, 79% AndroidWorld) and `GUI-JEPA` (`2610.36877`, decoder-free policy) own "single-pass zero-token GUI action selection". Our surviving core is the **calibrated act/ask/refuse gate** — see `outputs/gap-analysis-calibrated-single-pass-2026.md`. G1–G4 validation runs are COMPLETE on the LOCKED AndroidControl-Curated benchmark (8,435 bbox-corrected rows; base `google/embeddinggemma-2`, frozen, 768-d). Key results in `benchmarks/results/` + `notes/decision-2026-10-15-embedgemma.md`:
  - G1 zero-shot: image adds ~nothing (36.96% vs 34.78%, random 16.7%).
  - G2 single-pass head: acc = 0.583 (text AND multimodal identical → image does NOT boost accuracy).
  - G3 Q2 calibration: multimodal CE ECE = 0.029 (multimodal boosts calibration, not accuracy).
  - G4 risk-gate: proper-scoring collapses confidence range [0.602,0.632] → unusable for risk-gating; CE gives a usable coverage/risk tradeoff (τ=0.55: cover 54% at risk 0.28).
  - **Caveat to surface honestly:** calibration ↔ risk-gating tradeoff; multimodal adds calibration not accuracy on this benchmark.

### Program log (keep this in sync — every background run/result goes here)
- **2026-11-10 (4th) — FIRST PAIRED ANDROIDWORLD RESULT UNDER THE OFFICIAL PROTOCOL (one seed; no claim).**
  B0′ (E2, overlay hidden) and R1 s0 (contract v1) both ran all 116 tasks on the same host and instances.
  Pre-registered analysis (`paired_sr.py`, frozen in `da5d3eb8`): **R1 12.2% (14/115) vs B0′ 7.0% (8/115)**,
  w = 9 / l = 3, sign test p = 0.146 (104-task sensitivity: 12.6% vs 5.8%, p = 0.065). Amendment A1 fires
  (|w−l| = 6, exactly the bar) → seeds 1–2 online (chain8). §4 needs three seeds; nothing is claimed yet.
  Mechanism as predicted for repeats (first-encounter repeat 41.2% vs 65.8%), not for typing (+0.9 pts).
  **Post-hoc: the gain is mostly termination** — 7/9 wins are tasks B0′ finished but never declared done; R1
  declares done 2.4x as often and is right less often (17% vs 24%). **Q11 (no model, `status` at step 0) scores
  6.6% (7/113): B0′'s 7.0% is indistinguishable from doing nothing, and beyond the 7 "state already holds" tasks both
  arms solve exactly 7 — R1's gain is termination, not task skill.** (Seed 1 then scored 14.8% (17/115), w 12 / l 3,
  p = 0.035, with 10 successes beyond the floor tasks — so "termination only" holds for seed 0, not seed 1; the claim
  waits for seed 2.) Q10: a same-host replay is not
  deterministic (6/20 trajectories diverge, 0/20 outcomes flip, upper bound 16.1%). Also: B0′'s 12 previously
  hidden tasks hold 2 of its 8 successes; `run_arm.sh` now counts scored episodes from the index (a killed
  partial episode had read as scored). `benchmarks/results/2026-11-10-stage1/`.
- **2026-11-10 (3rd) — H11: EVERY "116-TASK" ANDROIDWORLD RUN SO FAR SCORED 104 TASKS.**
  `awenv/runner._slug` cut task names at 24 characters (docstring: "collision-free"), so 6 groups of tasks shared
  an episode id (e.g. `SimpleCalendarAddOneEvent` / `…Tomorrow` / `…RelativeDay` / `…InTwoWeeks`) and resumption
  skipped the later members as already finished — silently, with a normal report each chunk. Computed over the
  116-task table: 12 shadowed tasks (Recipe 7, SimpleCalendar 4, AudioRecorder 1; complexity 1.2–6.0), identical to
  the 12 missing from the 2026-11-08/09 run (its "104/116 finished" was this, not infra), B0-overlay, B0′ and Q11.
  **Consequence: every AndroidWorld rate quoted "on 116 tasks" (incl. 15/80, 15/104, 12/80) is on 104 tasks, with
  12 multi-step tasks never attempted; any 116-task claim must be restated.** Fixed with TDD (`_slug` no longer
  truncates; `plan_episodes` refuses duplicate ids; preflight plans the container's real task table and requires
  116 ids / 116 tasks). Harness packages v2f / v4 = v2 / v3 + this fix only. Runs are completed by task name
  (`hostscripts/make_completion_cfg.py`), not re-run. `run_arm.sh` (unversioned until now) also counted abandoned
  partial episodes as finished; fixed and moved into `benchmarks/aw/hostscripts/`. Board: `docs/research/EXEC-2026-11-10.md`.
  Also decided the same day: R1 is NOT moved to the second emulator (paired with B0′ ⇒ same host); the two hosts'
  images were proven byte-identical (layer-list and config hashes), the host itself is unmeasured — a 20-task
  host-equivalence check runs before the second emulator carries any paired arm.
- **2026-11-10 (2nd) — ANDROIDWORLD IS THE SINGLE PRIMARY BENCHMARK; CONTRACT AUDIT; FIVE HARNESS DEVIATIONS.**
  Decision record: `notes/decision-2026-11-10-androidworld-primary.md`. Evidence (every line from upstream
  `android_world@55ddaf5`, our code, or parsed data files): `docs/research/2026-11-10-AUDIT-androidworld-contract.md`.
  (a) **Operator decision:** AndroidWorld (116 tasks, official protocol) is the only benchmark optimised;
  AC-v2 64.07% stays as a past, teacher-forced result (its history is the oracle human step instructions,
  398/398 rows) and is not optimised further.
  (b) **Harness vs `suite_utils`, line by line:** H1 `bool(is_successful)` turns a 0.5 partial credit into a
  success; **H2 task instances were never seeded** (`random.seed(sha256("30_{name}_0"))` missing), so every
  AndroidWorld number so far is on random, non-official instances; H3 reset ran *before* `initialize_task`
  (upstream: initialise, then go home); H4 IR budgets read the class `complexity`, not the instance's;
  H5 screenshot and element list came from two different `get_state` calls. **Checked and correct:** the
  done gate, the per-task budget, and the denominator — upstream averages with pandas `mean()`, which skips
  the `NaN` of tasks that raised, so A5 §1's "wrong denominator" was itself wrong.
  (c) **Data facts:** 25/116 tasks score only via `answer` (ceiling 78.4% without it); AndroidControl's
  `status` is the final row of all 13,603 episodes, always `successful` (no non-terminal negatives);
  training extracted every a11y node while AndroidWorld serves `forest_to_ui_elements` (leaves /
  content-desc / scrollable, invisible excluded); 13/20 AW apps have zero training goals.
  (d) **Withdrawn on a slow re-read (no evidence):** "the model cannot see that wifi is on" (the screenshot
  shows it), "unseen apps are the dominant cause" (confounded with task length: seen-app long tasks 0/6),
  "bare-click history causes the loops", "the double tap caused the typing failures". They are hypotheses
  until an offline count sizes them.
  (e) **Corrections:** HANDOFF-2026-11-10 P4 is false (history is not dropped; 64.07 stands); aw116 is
  **12/80 = 15.0%** under the done gate, not 15 successes — and on unseeded instances.
  (f) Infra: `git` hung because `core.fsmonitor=true` stalls on the 55 GB tree; set to `false` locally.
  (g) **H1–H5 fixed on branch `aw-official-harness`** (plan `docs/superpowers/plans/2026-11-10-aw-official-harness.md`),
  each pinned by a test that cites the upstream line; `official_sr` reproduces 12/80 = 15.0% on the 2026-11-08
  episodes. Baseline config `benchmarks/aw/configs/aw116_official_baseline.yaml` (same `appint_full@938`, V track —
  only the harness changes) is ready; the run itself waits for the operator because it uses the shared hosts.
  (h) **Correction — Jev-Mobile is a hybrid, not a standalone single-pass agent** (read from arXiv 2609.30186):
  its 79% AndroidWorld uses Qwen3.8-Max as planner, which writes the input text and decides finish/blocked; Jev picks
  an ID among program-built a11y candidates. No standalone decision model (single-pass, zero tokens) has a published
  AndroidWorld number in the sources read; GUI-JEPA (2610.36877) reports AndroidControl only. Published limits of
  one-pass decision models (2610.24929 state tracking; 2610.25947 record-over-observation) match our failure classes.
  Review: `outputs/aw-state-review.md`. B0 = `ckpt-E2-final` (appint_full saw 1,380 AndroidWorld goal rows);
  R1 = contract-v1 retrain, 3 seeds; pre-registration amended (A1) before any R1 online result.
  (i) **FINAL PLAN = `docs/research/PLAN-2026-11-10-question-tree.md`** (operator constraint: one decision model, no
  planner, no external model). Partial B0 diagnosis, episode-clustered: no-effect steps 53.9% (CI 35–66%), repeat at
  first encounter of an unchanged screen 68.2% (CI 55–80%), typing lands 18/18 but is chosen in 2.0% of decisions,
  online AUROC(confidence → effect) 0.59 (CI 0.48–0.69). **H10 found:** official agents call `hide_automation_ui()`;
  we did not — fixed in code, so the running B0 is arm *B0-overlay* and the official baseline is B0′. The thesis is
  chosen at Gate 2 by evidence (matched generative control Q1 needs operator approval). Pooled 57% / 83% / 90.9%
  figures quoted earlier the same day are superseded by the clustered ones.
- **2026-11-10 — `episodic-s0` FAILS ITS OWN PRE-REGISTRATION; `is_done` IS NOT A QUESTION IN THIS CORPUS; THE GATE IS OFF.**
  (a) **The eval was already on disk and unread.** `runs/isdone.json` and
  `runs/eval/eval-episodic-s0-V.summary.json` were written 2026-11-09 22:58. The 2026-11-10 work went
  into the AndroidWorld loop instead — the one measurement `PRE-REG-2026-11-09-episodic-isdone.md` §(c)
  explicitly excludes ("a task-success number from this study would be noise read as signal").
  (b) **The mechanism, measured here for the first time:** on the 824 val rows, `step == n_steps` and
  `action == status` are the **same 137 rows** — 137/137, and **0 positives among the 687 non-status
  rows**. So "has the task been completed?" is answered perfectly by "is this action `status`?", which
  the action head already predicts. `auroc` 0.960 is that shortcut; **`auroc_nonterm` is null
  (not computable)**; `risk @ 50% coverage` = **0.670**. PRE-REG §Secondary withdrawal named exactly
  this and said the non-terminal number is the only reportable one — and it does not exist.
  (c) **Conditions: 1 FAIL (not measurable), 2 FAIL (never run — the V-track eval has `el_idx` on
  0/398 rows, so `target_el` was never evaluated), 3 FAIL** — AC-v2 **55.78% (222/398) vs 64.07%
  baseline = −8.29 pts**, `type_acc` 0.7764 vs 0.8417 (−6.53), `status` **64/87 vs 76/87**, while
  **grounding improved to 0.600 (114/218 spatial = 0.5229) vs 0.5605**. The loss is concentrated in
  the class the redundant readout was built from: training it consumed capacity, it added no signal.
  Report: `benchmarks/results/2026-11-10-episodic-s0-fails-prereg.md`.
  (d) **The `is_done` gate is now OFF by default** (`done_threshold: 0.0` in adapter, agent,
  `run_bench`), i.e. upstream behaviour, so SR is comparable again; two tests lock it
  (`test_is_done_gate_is_off_by_default`, `test_is_done_gate_can_still_be_switched_on_explicitly`).
  The gate generated three of its own defects on 2026-11-10 and they are fixed with regression tests:
  (1) three refused terminations all mapped to `wait`, produced **identical signatures**, tripped
  `stuck_k`, and the third was **overwritten with `navigate_back`** — which is what destroyed
  `MarkorCreateNote` with the note correctly typed on screen; (2) history reported a refused
  `status` as `"status -> declare done"`, so every later decision was taken by a model that believed
  the task had ended; (3) an invalid decision was reported as `"click -> ok"`. Also fixed:
  `_signature` now includes `text`/`direction`/`app_name` (two different typings and two opposite
  scrolls used to collide).
  (e) **Two more real harness defects found and fixed the same day, both measured on the device:**
  **`_teardown_task` called `tear_down()` with no env** — upstream signature is `tear_down(self, env)`
  (`task_eval.py:182`; callers `suite_utils.py:288`, `server/android_server.py:183`), so it raised
  `TypeError` every episode and the bare `except` swallowed it: **no episode was ever torn down**.
  Proven on the emulator: old call `TypeError`, `tear_down(env)` returns the device to the launcher.
  And **`goal` was never written to the trace** — the task's identity was unauditable, which is the
  same omission that hid the 2026-11-07 goal-only/history mix-up.
  (f) **`--run-root` moved out of the code tree** (`AW_RUN_ROOT`, default `/tmp/awruns`). It defaulted
  to `benchmarks/aw/runs`, and the harness is reinstalled into `/tmp/awh` on every emulator restart,
  so real episodes were silently deleted while the summary still printed.
  (g) **Gate A rewritten and green: 41/41, 0 skip, 0 fail** (was 40 pass/1 fail/1 skip, and it
  asserted the OLD wrong contract — `["click","input_text"]` for `type_text`, i.e. the double-tap bug
  it was supposed to catch). It now verifies **consequences**, not the absence of exceptions:
  `click`/`scroll` change the image, `navigate_home` reaches the launcher, `navigate_back` changes
  activity or pixels, `open_app` is polled to 24/24 apps (Retro Music's cold start was failing a
  fixed 5 s sleep — a flaky gate is worse than none), and `type_text` is proven to **replace** the
  field rather than append (`clear_text`), which the old gate skipped. Emulator screenshots confirm
  it: `+15195923419` and `The cake is a lie` both land in the right fields.
  (h) **Standing correction:** the earlier 2026-11-10 entry's 1/5 = 20% is **not a benchmark number**
  — it ran with the gate ON (a deviation) and n=5 (Wilson 3.6–62.4%). It is withdrawn as a rate and
  kept only as a mechanism observation.
- **2026-11-09 — THE 116-TASK RUN IS FINISHED AND THE SERVERS ARE CLEAN.**
  (a) **Result, with all four buckets and all three denominators.** 104/116 tasks finished. Buckets: **15
  successes, 34 non-terminating timeouts (median 25 steps), 31 premature `status` (median 7), 12 `unscored`
  (`checker_unavailable`), 12 `setup_error`.** Rates: **18.8% (15/80) over scorable episodes** — the only
  denominator that answers "how good where we could judge" — 16.3% (15/92) over genuinely attempted,
  14.4% (15/104) counting everything. **Zero invalid actions in 1,394 emitted.** Every success is a task
  whose gold trajectory is one or two actions beyond `open_app`; `type_text` appears in 19 episodes and
  succeeds once; no success involves a text field plus navigation. Gate drift negative on 73% of failures
  (n=77) vs 40% of successes (n=15) — mechanism, not result.
  (a2) **There were TWO independent defects, not one, and the second was scoring as a model failure.**
  `awenv_bridge.AndroidWorldEnv` had no `__getattr__` while AndroidWorld's own `task.is_successful(env)`
  calls `env.get_state()`, so every state-reading checker raised `AttributeError`, `success` stayed at its
  initialised `False`, and **12 episodes that ran 4–25 steps were counted as 6 timeouts and 6 premature
  terminations** — model failures no checker ever assessed. That set is disjoint from the 12 setup failures
  except 11 rows. Fixed at three levels: the wrapper forwards `get_state`; the runner records `unscored` as
  its own bucket and excludes it from `timeout`; `refresh_aw116.py` buckets it before timeout/premature and
  reports the scorable denominator. An earlier version of this entry said the crashes and setup failures
  were "one device defect, not a second problem" — that was wrong.
  (b) **The 12 setup failures are an infrastructure limit with a measured cause**, not model failures and not
  a harness bug: the emulator's **accessibility forwarder degrades after hours of continuous use** —
  `dumpsys accessibility` still says enabled+bound, `/health` still says success, `adb devices` still shows
  the device, but `get_a11y_tree` raises so `env.reset()` cannot complete; late in the session `/screenshot`
  returned HTTP `000` too. Two restarts briefly restored it and it degraded again within ~20 episodes.
  **A 116-task suite needs ~2.5 h of continuous emulator uptime and this container does not reliably provide
  it.** The fix is scheduling, not code: chunk the suite and restart every ~20 episodes — the runner is
  already resumable and `scripts/refresh_aw116.py` recomputes from whatever is on disk.
  (c) **Five harness bugs found by running to completion, all with regression tests (135 passed / 35
  skipped):** `wall_clock_budget` was run-wide and silently truncated 116→10 tasks while printing a confident
  `1/10`; a task whose `initialize_task` raises killed the whole run; `env.reset()` raising on a live device
  killed it too; `Recorder.episodes()` scoped to this process's `run_id`, so after a restart the suite
  reported `Completeness: 0/116` with 104 episodes on disk; and the missing `get_state` forwarding above.
  (d) **Every number in Limitation 12, the abstract, GOAL and HANDOFF is GENERATED**, not typed:
  `scripts/refresh_aw116.py` pulls the trajectories and recomputes; `scripts/sync_paper_aw116.py` rewrites
  the prose from that evidence, and `check_record.py` re-renders it and demands byte equality.
  (e) **§5.12.1 and the value brief §5.1 were added**: `P(task) = p^n`; our 64.07% implies 2.8% over 8
  steps; inverting published scores gives the frontier's per-step accuracy — **MAI-UI-2B 92.3% after SFT
  alone, 93.1% after RL, 235B 97.4%** — so **online RL is worth only +0.8 per-step at 2B**.
  ⚠️ **RETRACTED 2026-11-09 (M13/M14):** these are **derived by us**, not reported by MAI-UI (which publishes
  no per-step metric), and the derivation assumes step independence, which **our own data refutes by 6.6x**
  (predicted 2.84% at 8 steps, measured 18.8%). The "28-point per-step gap" is **withdrawn**; the measured
  claim is a **task-success** gap (45.1% vs 18.8%). See paper §5.12.1.1. `scripts/verify_compound_accuracy.py` re-derives all of it from
  `references/raw/maiui2026-androidworld-table9.txt`. **The comparison class is MAI-UI-2B-SFT at 45.1%
  end-to-end, not 76.7% (235B post-RL) and not even 49.1% (2B post-RL) — and we do not reach it.**
  (f) **Read slot**: `app_intent` probe 1/8 → 8/8; five arms tabulated with artifacts (smoke 24.37%,
  hint_only 26.88%, mix60 59.55%, v2 63.07%/68.23% grounding, full 63.07%/66.49%). Training only on the new
  field is catastrophic (64.07 → 24.37); **mixing is mandatory**.
  (g) **Servers clean, measured**: disk 37 G → **56 G** free; containers 13 → **12 (all pre-existing prod)**;
  `android_world` images 1 → **0**; the d1 API process stopped (6,006 MiB → 0); the `docker0:8700` iptables
  rule removed; the prod GPU process `2641764` untouched at 2,166 MiB. Ledger:
  `docs/compute/CLEANUP-2026-11-08.md` §Executed.
  (h) **Gates**: all green at the time of writing — **today's counts are in §3.5 Gates only**, which is
  machine-refreshed; a dated entry must not carry a count a later commit can falsify.
- **2026-11-08 (2nd) — THE COMPOUND-ACCURACY ARITHMETIC; THE FIRST FULL 116-TASK RUN; AND WHY "FIX THE BUG" WAS NEVER THE ANSWER.**
  (a) **`P(task) = p^n` is the governing equation and nobody in this programme had written it down.** At our
  measured per-step 64.07%, an 8-step task succeeds **2.8%** and a 10-step task **1.2%**. Inverting the
  published end-to-end AndroidWorld scores gives the per-step accuracy the frontier *would* need **if steps
  were independent**: **93.1%** (MAI-UI-2B, 49.1% e2e), **96.6%** (MAI-UI-8B, 70.7%), **97.4%**
  (MAI-UI-235B, 76.7%), **88.8%** (M3A, 30.6%). **(a2) THE GAP IS SUPERVISION, NOT RL — from MAI-UI
  Table 9.** Their SFT/RL rows read **2B 45.1 → 49.1**, 8B 64.7 → 70.7, 32B 69.8 → 73.3 (**end-to-end**;
  Table 9 has no per-step column). ⚠️ **2026-11-09 RETRACTION (M13/M14):** the derived per-step pair
  (92.3 → 93.1) and the "28-point per-step gap" are **withdrawn** — they assumed step independence, which
  our own measurement refutes 6.6x over (predicted 2.84% at 8 steps, measured 18.8%). The surviving claim
  is a **task-success** gap. The supervision diagnosis stands and is unchanged: multi-step trajectory corpora
  (GUI-Odyssey's 8,334 cross-app episodes), thousands of expert/synthesised tasks on a live emulator with
  an MLLM-judge rejection loop, a 50-step horizon, and a 12-action space with `ask_user`/`mcp_call`. We
  have SFT on 15k **single-step** decisions, one 4090, 8 actions, no trajectory data, no judge loop.
  **Our comparison class is MAI-UI-2B-SFT at 45.1% end-to-end — not 76.7% (235B post-RL), not even 49.1%
  (2B post-RL) — and we do not reach it.**
  `notes/2026-11-08-compound-accuracy-reframe.md`, `notes/2026-11-08-sota-recipe-and-our-gaps.md`.
  (b) **This retroactively re-explains the 1/8.** The measured **12.5% (Wilson 2.2–47.1%)** is *consistent*
  with 64% step accuracy; it was never evidence that the harness was broken. The last two days of hunting a
  bug kept coming up empty because the arithmetic, not the code, was the binding constraint.
  (c) **The `app_intent` fix is real but small, and the paper must say so.** Serving probe went **1/8 → 8/8**
  on intent-only goals ("Turn wifi on" → `settings`), and the read slot is a pointer copy that works on
  unseen names (Markor 0.805, Joplin 0.787, Broccoli 0.963). But the ceiling moves from ~3% to ~4%. Mixing
  is mandatory: `appint_full` (cur15k+app_intent, 938 steps) guards at **AC-v2 63.07%** with grounding
  **66.49%** (up from 56.05) — accepted as a guard, not a headline.
  (d) **FIRST FULL 116-TASK RUN, as it stood at this entry (`aw116_appint`).** 82/116 finished → **8
  successes, SR 9.8% (Wilson 5.0–18.1%)**, with 39 non-terminating timeouts (median 25 steps) and 37 premature
  `status` (median 7). **Those three categories do not sum to 82** (8+39+37 = 84), because the unscored bucket
  did not exist yet and the per-episode files for that snapshot were later superseded by the final ones, so the
  missing rows cannot now be recovered. Read this entry as a dated observation whose taxonomy was incomplete —
  not as an arithmetic that balances. The corrected, mutually exclusive taxonomy is in (a2).
  **(d2) ZERO INVALID ACTIONS: 0 of 1394 emitted actions were unrepresentable (0.0%)** — the 28.2%
  invalid rate in the 8-task run was purely the empty `app_name` read slot, not a model limit. Repair the read
  slot and every action the model chooses is executable (11 env crashes in 104 episodes; all 11 are setup
  failures — the same device defect — while the 12 `unscored` episodes are a SECOND, disjoint defect: they ran
  but their checker raised. Do not merge the two). What it cannot
  do is compose them. Two failure modes, opposite and both from the same missing signal:
  **premature `status`** at step 3–5 on tasks needing 8–15 (the model is confident and wrong), and
  **never terminates**, emitting 24 consecutive `click`s while `gate_score` decays to **0.0** by step 5
  (ClockTimerEntry: step 0 `open_app clock` conf 0.996 gate 0.996 → step 5 gate 0.000, action head still
  0.92 confident). Action mix at that 54-episode snapshot (**not** the final run, whose action totals are in
  Limitation 12): click 665, open_app 58, scroll 73, wait 42, status 25, type_text 16 — `type_text` was ~2% of actions
  on a suite that needs text entry on most contacts/notes/expense tasks.
  (e) **`gate_score` DRIFT SEPARATES OUTCOMES (hypothesis, n=15 successes — a mechanism, not a result).**
  Mean Δ(first-3 → last-3) is **-0.202 on failures, 73% of them decaying (n=77)** vs **-0.072 on successes,
  40% (n=15)**. Pooled over the final run: gate p50 **0.334**, exactly 0.0 on **11.7%** of actions, while the
  action head's own confidence stays p50 **0.785**. With fifteen successes this is a mechanism, not a result,
  and it is labelled as such everywhere it appears.
  (f) **THREE HARNESS BUGS FOUND BY RUNNING 116 TASKS, all fixed with regression tests** (as of this entry: 128 passed / 35 skipped; the suite is now **132**).
  (1) **`wall_clock_budget` was applied to the whole run, not per episode** → the suite printed a confident
  `SR 10.0% (1/10)` and stopped after ten tasks with no indication that 106 were never attempted. The cap
  now binds inside the episode and the report prints `Completeness: n/N` **before** the SR table, saying
  INCOMPLETE explicitly. (2) **A task whose `initialize_task` raises killed the entire run** —
  `ExpenseDeleteDuplicates` raised `no such table: map_markers` and lost 74 tasks of coverage at episode 42;
  setup failures are now recorded as `setup_error` and the suite continues. (3) **`_build_env` hardcoded
  `emulator_setup=False`**, so a fresh container died on `AUDIORECORDER_DATA`; it is now a config option
  defaulting on. Also: container→model needed **`--network host`-equivalent reachability** — the host's
  `iptables INPUT` policy is DROP, so a `docker0`-scoped ACCEPT on 8700 plus binding uvicorn to `0.0.0.0`
  was required; `--network host` itself breaks the emulator (`TooManyRestartsError`, and it collides with a
  stale host `adb` server on 5037).
  (g) **Standing correction to the plan.** `docs/HANDOFF.md` and `notes/GOAL.md` now carry the arithmetic and
  the in-progress state; GOAL's `AndroidWorld full benchmark: — / not attempted` row is replaced. No headline
  AndroidWorld SR exists yet and the partial 11.1% may not be quoted as one.
- **2026-11-08 — ANDROIDWORLD REBUILT; COORDINATOR D01–D07 BUILT; THE RECORD/ARTIFACT GAP MADE MECHANICAL.**
  (a) **The emulator is back and re-verified.** Image `android_world:latest` 12.7 GB built from upstream @HEAD
  (it must be built — upstream ships **no** prebuilt image; the README only has `docker build`). Container
  `aw-emu` on host A, KVM, 24 GB cap, host port **5051**; `/health` → `{"status":"success"}` after **300 s**
  (boot ~65 s + the lifespan `emulator_setup`). **Gate A: 32 passed, 1 skipped, 0 failed** (the single failure
  was a transient `device offline` on SimpleSMS; the case re-ran **24/24**). Liveness **4/4 layers** ok,
  screenshot 2400×1080×3. Note: `docs/compute/AW_LIVE_2026-11-08.md`.
  (b) **Two harness bugs the live run found.** **`GET /screenshot` requires `wait_to_stabilize`** — a bare
  request returns **422**, so the liveness `noop` layer reported a *dead device while the device was healthy*;
  that is the same false-crash class as the withdrawn 25% claim. Fixed in `awenv/liveness.py` + a regression
  test. And the upstream Dockerfile still needs **patch #3** (Pillow): `PIL` is imported by
  `task_evals/utils/user_data_generation.py` but undeclared, so the container starts and dies on import. The
  rebuild takes ~40 min because `eclipse-temurin:17-jdk-jammy` has no Python and compiles it — **use the layer
  cache**. `adb` on the host cannot see the emulator (separate netns), so Gate A must run **inside** the
  container (command in the note).
  (c) **The coordinator exists as code, not prose.** `benchmarks/aw/awenv/coordinator.py` implements the four
  policies the revision guide specifies (B1 create-once, B2 replan-every-step, B3 rule-based, P1 learned), with
  the output restricted to exactly `CONTINUE_PLAN` / `CREATE_OR_UPDATE_PLAN`, a missing or invalid plan always
  forcing UPDATE, and the decision layer provably unable to execute an action. **12 tests** cover the guide's
  own completion checks (no-plan→planner, CONTINUE preserves version, UPDATE precedes selection, completed
  subgoals survive revision, stale observation does not re-decide, features carry no evaluator/future state).
  `awenv/coordinator_bench.py` + `configs/manifest_v1.json` (12 locked tasks, sha256 `055bda39e1bf8a7e`) run the
  four conditions with per-episode records (completion category, elapsed, planner calls, actions, model p50,
  terminal reason) and a rebuilable summary table; `--dry-run` validates wiring with no environment.
  `coordinator/D01`–`D07` deliver the manifest, the frozen 9-test action schema, the data ledger with an
  episode-id overlap audit, the label file + status, the supervised baseline, the bench, and the error taxonomy.
  **D05's 98.11% is a tautology and is labelled as one**: `no_plan` ≡ `hist_len == 0` (57% of UPDATE rows) and
  `repeat_subgoal` ≡ `active_subgoal == prev_subgoal` (90.2% of UPDATE vs 6.3% of CONTINUE) — the label rule
  created the separation the classifier found. **AndroidControl cannot supply branch-comparison labels**
  (no planner, no reset, no evaluator), which is why P1-vs-B1/B2/B3 needs the live loop.
  (d) **The defect class the auditor found is now checked mechanically.** `scripts/check_record.py` went from
  26 to **32 checks** and they now assert what they only claimed before: every `### 5.N` title against its
  subject, every `§5.N` pointer in AGENTS/GOAL/HANDOFF against the section that actually says that thing (paired
  to the nearest subject on its own line, distance ≤70 chars), arithmetic consistency of every `(N/398)`
  percentage, and a retraction banner in each file whose deltas the pre-registered seed rule withdrew. **This
  was necessary because the previous block encoded the *old* section numbering and so could not catch its own
  drift.** Gates **as of that day**: `check_record.py` **32/32**, `verify_paper.py` exit 0 (69/69, stage-0
  scope), `benchmarks/aw/tests` **123 passed, 35 skipped**. (The counts moved because the harness gained tests
  and the record gained checks. Read the entry as dated history; **today's counts are in §3.5 Gates only** —
  there is deliberately no "current" figure here to go stale.)
- **2026-11-07 — THE 25% "ENV CRASH" WAS A CHECKER BUG; THE PURE ARTIFACT EXISTS; CAPACITY IS THE REAL BLOCKER.**
  (a) Both 8-task "crashes" are the two Clock tasks, and the device was alive: last step `error: null`,
  `action_ok: true`, no `env_crash_candidate`; the next episode in the same suite ran 10 steps immediately
  after (pure8: 20:40:51Z crash → 20:43:08Z clean; 20:45:30Z crash → 20:46:17Z clean); stderr repeats
  `Could not get a11y tree, retrying`; the last screens show a live ticking stopwatch (7 laps) and a timer at
  20 h 00 m 85 s. Cause: `task.is_successful(env)` **raised**, and `runner.py` labelled any checker exception
  `env_crash` without asking the environment. Fixed: `checker_error` is recorded and `env_crash = not
  _alive(env)`; 2 new tests; **110 passed, 35 skipped**. Paper §5.7 corrected (25% crash claim withdrawn;
  Clock failure is behavioural). Note: `notes/2026-11-07-clock-checker-not-crash.md`.
  (b) The pure-protocol predictions **do exist with every readout slot**: `backup-2026-11-06/offline/e2-pure-hist.jsonl`
  = **254/398 (63.82%)**, `readouts` on 398/398 (`action`, `target_box`, `scroll_dir`, `app_name`,
  `type_text`). The goal-only file `e2-pure.jsonl` is 132/398 (33.2%) and must never be used. **Val (827)
  still has no readouts** — that is the only remaining stage-2 blocker.
  (c) GPU: one host reachable from the Mac, **7.0 GB free** (prod, 58% util) → training does not fit; the
  5090 worker behind it has **13.9 GB free**, which fits micro=1 + grad-ckpt (~8.2 GB peak) but not micro=2.
  The AC image corpus is on no reachable disk, so it is being rebuilt from the public GCS TFRecords on the
  5090 now. **No training ETA until the smoke passes.**
  (d) **Zero-GPU post-processing is now closed for the type bucket.** A val-fitted `p(status)` gate that
  switches premature terminations to the best expressible action is monotone-harmful at every τ (best rule =
  never switch; val 62.76%, test 63.82%). The 21 premature `status` rows are 5.3 pts, but they cannot be
  thresholded in — only learned. `benchmarks/results/2026-11-07-status-gate.md`.
  (e) **INTERNAL AUDIT (T4) LANDED; THE ACCURACY HALF DID NOT SURVIVE.** 64.32/256 exists in **no artifact** →
  restated **64.07 (255/398, `E2-nomem`)** with 63.82 as the served-path cross-check. Grounding 56.50 was the
  **step-705** checkpoint → 56.05 pure / 57.40 hinted. Candidate coverage 1.0000 → **216/223 = 0.9686**. §5.7's
  "non-overlapping Wilson" is **false**: intervals overlap on 19.0–25.0 (Fisher p = 0.065), and the two online
  conditions ran **different task instances** (5 of 8 differ), so the invalid-action difference is unmatched and
  is a hypothesis, not an effect. p×box ECE is 0.37–0.40 (not 0.29–0.36); env p50 is 328–332 ms (not 1066).
  Everything in `2026-11-03-stage0` still re-derives byte-identically; `verify_paper.py` is 69/69 but
  **stage-0-scope only**. Paper, GOAL.md and HANDOFF.md restated; commit `2ddc7cc2`.
  (f) **THE DECISION-LEVEL CLAIM (cost window).** At 40.7% coverage / 17.3% risk the gate beats **both**
  always-act and always-ask for `C_err/C_ask` in **(2.11, 5.79)** — (2.25, 4.61) using the CP upper bound;
  learned fusion widens the lower edge to 1.95. Paper §5.12, `benchmarks/results/2026-11-07-cost-window.md`.
  Assumption stated: an ask is answered correctly at a fixed cost (no GUI paper prices the ask).
  (g) **STAGE-2 PRE-REGISTERED; PROVISIONING RUNNING.** Post-processing is closed (see d). Arm 1 = retrain from
  base, E2's exact 938-step recipe, `status` down-sampled 22% → 5.3% (3,300 → 661 rows); PASS = paired val Δ ≥ +1.5
  with no gold type losing >1 row (`docs/research/PRE-REG-2026-11-07-status-downweight.md`). AC corpus is being
  rebuilt from the public GCS TFRecords on the 5090 (~54 GB, 6 parallel shards), then venv → val readouts → smoke.
  **Status 2026-11-07 03:0x:** corpus built (88,288 train / 827 val decisions, 89,144 images), inference path
  smoke-tested (4 val rows, 6 s), 30-step training smoke passed (loss 5.35 → 4.35, `TRAIN_DONE`), full 938-step
  arm launched on the 5090 (`--track V --micro 1 --accum 16 --grad-ckpt --mem-frac 0.38`, ~10.6 GB, 13.9 GB free).
  (h) **VALUE BRIEF DELIVERED** — `outputs/calibrated-decision-value.md` (+ `.provenance.md`), four evidence
  strands, 36 sources, reviewer pass with six MAJOR findings all fixed. Verdict: the act/ask/refuse gate is
  already platform policy at Apple and Google, but no first-party document states a numeric confidence
  threshold, a measured ask rate, or an outcome result; and no GUI-agent paper has an outcome-level abstention
  result *in the sources checked*. The programme's contribution is a measurement plus a certificate boundary.
  (i) **STAGE-2 ARM RESULT — NEGATIVE AND INFORMATIVE (2026-11-07).** Pre-registered `status` down-weighting
  (22% → 5.3%) run to 938 steps from base: val **62.76 → 63.85 (+1.09, McNemar p=0.44)**, test
  **64.07 → 60.80 (−3.27, p=0.15)** — fails both clauses (Δ < +1.5) and the splits disagree in sign. Mechanism:
  **−27 premature terminations bought −26 correct ones** (51→24 premature, 120→94 true `status`). A single class
  weight cannot separate "not finished" from "finished". Grounding improved without being trained
  (**+2.67 val / +2.26 test**) because fewer early terminations mean more acting steps. The oracle bound
  predicted it: a *perfect* progress signal is worth only **+1.26 pts** here. ⇒ **the status bucket cannot be won
  by re-weighting; the next arm must be an explicit `task_complete` readout.** Full note:
  `benchmarks/results/2026-11-07-status-downweight-arm.md`.
  (j) **THE ARM IS A SPATIAL SPECIALIST; ENSEMBLING IT DOES NOT CLEAR THE BAR.** Per type on test: click
  **+9**, scroll **+4**, status **−25**, others ±1. Oracle routing on **gold** type = **66.33 (+2.26)**;
  routing on E2's *predicted* type = 64.82 (+0.76); arm-only-when-types-agree = 65.08 (+1.01, McNemar
  p=0.56, CI [−1.51, +3.51]). **The router's type decision is the bottleneck**, so the oracle ceiling is not
  reachable this way (`benchmarks/results/2026-11-07-what-the-arm-bought.md`).
  (k) **FUSION RESEARCH: +5.83 pts of ORACLE headroom, +1.3 achievable, and the arbiter is the bottleneck.**
  Grounding on the 223 spatial rows: E2 56.05, arm 60.09, **union 65.92 (+5.83)** — the two checkpoints miss
  different elements. The distance between their predicted points is a near-perfect trigger (**147 px** median
  when exactly one is right, **8 px** otherwise) and `box_conf x found` picks the right one **76%** on the 21
  disagreement rows. But the honest val-fitted rule gives **+1.26 (p=0.51)**, and a 20-feature logistic arbiter
  gives **val +2.90 / test +0.00** with 214 of 398 rows switched — 107 fit rows cannot support it. **The arbiter
  is a data problem, not a modelling one.** Also: a first version gated on GOLD type and reported a fake +3.52
  (p=0.0013); recorded in the note as a warning. `benchmarks/results/2026-11-07-fusion-headroom.md`,
  `benchmarks/train/fusion_arbiter.py`. Prior art for the design: AdaGround (arXiv 2610.36874) — two-chain
  portfolio + marked-click verifier, training-free, +7.4 to +20.6 on OSWorld-G / ScreenSpot-Pro.
  (l) **ADVERSARIAL REVIEW OF THE PLAN — FOUR FATAL FINDINGS, ALL ADOPTED** (`docs/research/PLAN-2026-11-07-better.md`
  v2; review in `benchmarks/results/2026-11-07-review-strategy.md`). (1) The plan's premise "type solved,
  placement is not" is a **retracted, do-not-cite** claim (type errors 63 vs grounding 68, z≈1.4) — corrected to
  "both comparable". (2) **The ensemble was a selection artefact:** at matched 3-checkpoint size the vote gives
  **+0.50 on test (p=0.63)**, not +2.01; after a 16-cell correction the +2.01 warrants **p=0.17**; the gain is
  **clicks only (click +14, status −2)**. (3) **AC-v2 is not rankable:** the same backbone scores 49.62 (ECHOES)
  vs 54.80 (Mem-W) — a 5-pt protocol spread twice the deficit; stop quoting cross-paper numbers. (4) **Work #5
  (ConClick/box-σ) is DELETED** — HANDOFF §3.3 marks it invalidated. Also: **CORA (arXiv:2604.09155)** already
  publishes conformal risk-controlled execute/abstain for mobile GUI, so **the gate is not the contribution — the
  price is**; and `V-Droid` (MobiCom 2026) already does verifier-in-the-loop on Android. Pass bars switched from
  accuracy to **precision on switched rows** (need 65% on ≥60 rows; the programme sits at 60–64% on 25–28).
  **Paper reframed: "Priced Interruptions", not ensemble accuracy.**
  (m) **GROUNDING MECHANISM FOUND — THE BOX LOSS HAS ZERO GRADIENT WHERE 86.8% OF ERRORS ARE.** `target_box` is
  four per-edge CE against a Gaussian (σ=4 bins on a 0-999 grid; 1 bin = 1.08×2.40 px). CE saturates at a
  numerically identical **690.78 nats for offsets ≥200 bins**, so ΔCE/bin = **0.0** beyond ~400 px. Our misses are
  median **437 px**, 86.8% >150 px. So the objective is sharp where errors are rare and flat where they are common
  — a sufficient explanation for the 56% plateau with 84% type accuracy.
  `benchmarks/results/2026-11-07-why-grounding-plateaus.md`.
  (n) **"THE CANDIDATE INTERFACE FAILED" WAS WRONG (R4).** On matched weights (same checkpoint, step 300, 16 min
  apart): V 49.50% vs **VA 55.53% (+6.03, McNemar p=1.5e-3)**, grounding **46.20% → 56.38% (+10.2)**; VA-only 39,
  V-only 15. The prior conclusion compared a 300-step `cur24k`/mix run against a 938-step `cur15k`/pure run. The
  **union V+VA = 71.36% (+7.29 vs V, p=5.1e-4)**.
  (o) **`gold_candidate()` SUPERVISED AN UNSCOREABLE POINT ON 12 EVAL ROWS (fixed).** The containment fallback could
  return a container whose centre is outside the gold node, so a perfect imitation of the label was guaranteed
  `grounding=False`. Now requires centre-in-node; unscoreable labels **12 → 0**. Honest supervised ceiling
  **97.99% train / 97.68% val** (the widely quoted 99.99% counted "point inside some candidate", which the scorer
  does not award). `benchmarks/train/build_candidate_supervision.py`.
  (p) **VA ARM RUNNING (pre-registered).** `PRE-REG-2026-11-07-candidate-head.md`: same `cur15k`, same 938 steps,
  same lr/init as E2, ONLY `--track VA`. PASS = grounding ≥62% **and** step acc ≥62%; secondary union ≥70%;
  kill if grounding <58%. Smoke passed (`acc_target_el` 0.55). Full run launched on the 5090.
  **(q) RESULT: VA ARM FAILED, AND THE FAILURE NAMED THE REAL CAUSE.** val **62.76 → 59.01 (−3.75, p=0.019)**;
  test **64.07 → 56.78 (−7.29, p=0.0013)**; grounding test **65.54 → 55.37**. Decisive detail: the arm's
  **box-only path also dropped** (val 60.22, test **59.80**, i.e. **−4.27** vs E2) — so the long candidate state
  damages the coordinate head, not the interface. Per-type: click −29, type_text −6, scroll **+10**, non-spatial
  **+4**. ORACLE union val 70.86 (+8.10) / test 70.10 (+6.03) is **not deployable**: the best free arbiter
  (`box_conf`) gives **+0.60, p=0.68**. Report: `benchmarks/results/2026-11-07-va-arm-result.md`.
  **(r) THE CONSTRUCTIVE OUTPUT — TWO-STAGE SHORTLIST.** Filter candidates by **E2's own predicted point**
  (gold-free) at R=450 px: **median 6 candidates**, gold inside that short list on **78.5%** of the 223 spatial
  rows, vs **56.05%** grounding today. R=300 → 70.9%. This is a **+22-point ceiling** with a ~200-token stage-2
  state, so stage 2 never competes with the image. Pre-registered: PASS grounding ≥68 and step ≥62; kill <62;
  **shortlists must be built from the stage-1 prediction, never from gold** (the R4 skew class).
  `benchmarks/results/2026-11-07-two-stage-design.md`, `docs/research/PRE-REG-2026-11-07-two-stage.md`.
  **(s) Also found and fixed: the 14-candidate cap drops 58.6% of distractors** — so the failed arm was a *lower
  bound* on the interface, not a fair test. And `eval_d1.py` now logs `el_idx` + `box_only_pt` so element-head
  post-mortems stop needing a brittle rebuild.
  (t) **SEED SPREAD MEASURED — AND IT WITHDRAWS THE DAY'S GROUNDING DELTAS.** Three runs, same 398 rows:
  E2 (seed 0) **56.05%** grounding, ctrl (seed 0) **60.09%**, ctrl (seed **2**, identical recipe) **59.19%** —
  **range 4.04 pts, sd 2.12**. The pre-registered rule (sd ≥ 2.0 → withdraw) fired. **Withdrawn:** duplication
  +4.0, aux7k +2.0 (including its "two-seed replication"), 858px +2.79, the +2/+3/+1 decomposition, and the
  additive-combination prediction. **Root cause:** with `--steps 938` and batch 16 the trainer draws 15,008
  samples and logs `epoch: 0` in every run, so a longer data list changes only the *shuffle* — E2 and the
  "duplication control" differ by **8 samples out of 15,008** yet differed by 4 points of grounding.
  **Survives (≫ 2σ or artifact-only):** the loss-gradient arithmetic (CE constant 690.78 nats beyond ~200 bins;
  86.8% of misses beyond 150 px), base-vs-fine-tune (17.4%→64.07% acc, 14.4%→56.1% ground), the >40-candidate
  collapse (−17.3 val / −23.8 test), the placement-vs-verb decomposition (placement +26.0, verb +5.6), and the
  cost window. **Rule going forward: never report an AC-v2 grounding/accuracy delta without ≥3 seeds and an
  interval.** `benchmarks/results/2026-11-07-SEED-SPREAD-VERDICT.md`,
  `docs/research/PRE-REG-2026-11-07-seed-spread.md`.
  **(u) UPDATE 2026-11-08 — THE DUPLICATION "SURVIVOR" WAS NOT SIGNIFICANT EITHER.** §5.9 is the paper's only
  surviving positive claim, so it was re-derived from the eval artifacts (`backup-2026-11-07/seed_evals/`,
  `benchmarks/results/2026-11-08_seed_table.py`). The **means reproduce exactly** (15k n=3, 56.80 ± 1.29; 22.5k
  n=4, 59.42 ± 0.45; diff +2.62), but the **inference did not**: the paper printed `t = 3.54, df = 5`, which is a
  naive average-of-variances pooling, not the standard estimator. Grounding is a **count out of 223**, so each arm
  has only **two distinct values** (125/223 and 130/223 vs 132/223 and 134/223) and its sd is a *floor* from a
  single contrast. The variance ratio is **8.3×**, so the equal-variance assumption fails. The three defensible
  tests disagree: Student pooled t=3.84 (df 5, p=0.012, assumption violated) / **Welch t=3.34 (df 2.4, p=0.063)** /
  exact permutation p=0.029 (which is the *minimum attainable* at n=3 vs 4, so it cannot separate real from lucky).
  The pre-registered bar was t ≥ 2.78 at **5 seeds per recipe**; only 3 and 4 exist. **Verdict: suggestive, not
  established.** Paper §5.9, `2026-11-07-STATUS.md` and `2026-11-07-RESULT-duplication-works.md` all restated;
  `scripts/check_record.py` now re-derives §5.9 from artifacts (5 new checks, **41/41**) so this cannot drift again.
  Consequence: the programme currently has **no certified positive grounding result** — the contribution is the
  loss-mechanism analysis, the base-vs-fine-tune decomposition, the candidate-collapse bound and the cost window.
- **2026-11-06 (4th) — FIRST SUCCESS ONLINE; THE APP-NAME FIX BUYS EXPRESSIVENESS, NOT OUTCOMES; 25% CRASH RATE.**
  8 tasks × 2 conditions, protocol `aw-main-v1`, E2@938, 10-step budget. **Pure: SR 1/8 (12.5%, Wilson
  2.2–47.1%), invalid 20/71 (28.2%, 19.0–39.5%), ASR 71/71.** **Declared fallback: SR 1/8, invalid 10/68
  (14.7%, 8.2–25.0%), ASR 68/68.** The one success is `TurnOnWifiAndOpenApp` in **both** conditions (same task =
  determinism). The invalid-action intervals do not overlap ⇒ the read-slot fix is real; the success intervals
  are identical ⇒ **not** an outcome win at this n. With the fallback the model also emits `type_text` and
  declares `status` for the first time. **Environment crash rate 2/8 (25%) in both conditions, always on the
  same clock task** — the binding infrastructure problem now, and the reason no headline AW number should be
  quoted. A 4-task pair reproduces the direction (0/4 both; invalid 50%→27%). Model p50 **122.9–124.3 ms**,
  env p50 135–294 ms. Paper §5.7 now reports this table with intervals and the explicit "not established" line.
- **2026-11-06 (3rd) — THE HEADLINE REPRODUCES THROUGH THE SERVED PATH ONLY WITH INTERACTION HISTORY;
  THE GOAL-ONLY MISTAKE RECURRED IN OUR OWN TOOLING.** `benchmarks/regen_pure.py` asks the *deployed* API for
  the 398 predictions (so the artifact comes from the same forward pass the latency/calibration numbers do,
  with `--save-readouts` for stage 2). First run: **33.2%** — alarming, because the paper said 64.32 at the time (now 64.07). Root
  cause: the regen sent `history: []`. The AC-v2 protocol is goal + **Interaction History** (`previous_actions`,
  present on 315/398 rows), and `build_state` renders it as `Goal:` + `Previous actions:` lines. With history
  wired in, the **first 40 rows match the artifact exactly (19/40 = 47.5%)** and the full run gives
  **254/398 = 63.82%** vs the artifact's 258 (hinted) and the paper's pure 256. Differential test on all 398:
  **392/398 identical correctness verdicts, 391/398 identical predicted action**; the 6 disagreements are
  borderline box hits (same action, point 1-2 px either side of the edge). ⇒ **the deployed path is faithful**;
  a goal-only run is the bug, and it is the same mistake AGENTS.md already recorded once for training.
  **Also fixed:** `generate_random_params` takes no seed in this AndroidWorld version (pairing now replays
  recorded params), and task params can hold non-JSON objects (`CalendarEvent`) — writing them to the
  trajectory killed a run *after* the episode finished; params are now coerced.
  **Paper:** §5.7 rewritten (Feynman) around the gate results and the `app_name` read-slot finding; one
  unverifiable "first public" claim it introduced was removed.
- **2026-11-06 (2nd) — GATES A AND B PASS; THE ONLINE BLOCKER IS A READ SLOT, NOT GROUNDING.** Live
  measurements on the emulator with E2@938. **Gate A: 32 passed, 1 skipped, 0 failed** (4-layer in-process
  probe, every emittable action round-trips, `open_app` verified by foreground activity for 24/24 apps; the
  skip is the text-landing check with no editable field, skipped with a reason). **Gate B: PASS** — 5/5
  episodes identical across two runs with task params recorded in run 1 and replayed in run 2. First honest
  numbers (5 tasks, 60 actions): model p50 **112–119 ms**, env p50 **1066–1071 ms/step**, ASR **60/60**,
  **invalid 24/60 = 40%**, SR 0/5 (all timeouts), 0 generated tokens. **Finding:** 27 of 60 actions were
  `open_app` and **24 of those carried `app_name=''`** at confidence 0.92. Probes prove the API *does*
  return names (`clock`, `Markor`) on AC-v2 screenshots, so this is **not** a serving bug — the read slot
  collapses on AndroidWorld screens/goals. The action and name slots disagree, and the adapter correctly
  refuses to invent a name. ⇒ the first online number is dominated by **action-space/readout coverage, not
  grounding**; report `SR_all` vs `SR_reachable` or the number is misread. Note: `notes/2026-11-06-online-interface-defect.md`.
  **Infra (all of it removable):** image `aw-emu-snap:20261106`, container `aw-emu` (KVM, 24 GB cap,
  `--network host` because container→host traffic on the bridge is firewalled), model served on the *second*
  4090 (the KVM host's card is 79% occupied by prod, 5 GB free → OOM even at a 0.22 cap), reached from the
  container through a two-hop SSH relay that terminates on my machine. **Harness bugs found by the gates
  themselves:** (a) one failed liveness probe was recorded as `env_crash` → now requires two; (b) the
  gate-B wall clock silently truncated 5 tasks to 3; (c) `generate_random_params` takes no seed, so pairing
  now comes from recorded+replayed params; (d) screenshots were never persisted (agent never handed over the
  frame it saw) → fixed, applies to the next run; (e) `harness_commit` was "unknown" inside the image → now a
  package hash (`awenv-…`).
- **2026-11-06 — HARNESS CODE COMPLETE; TWO SILENT-BUG CLASSES FOUND; EMULATOR PROVISIONED.** Tier A/B code
  is done and tested: `agent.py` (attributable failures, `status`-only termination, loop detection),
  `runner.py` (resumable, paired, timeout≠success, crash≠success), `baselines.py` (wait/random through the
  *real* adapter path), nightly regression with alerts, `run_bench.py` CLI with `--dry-run`, and
  `docs/compute/AW_PROTOCOL.md` (what every number must carry). `pytest benchmarks/aw/tests -q` → **94
  passed, 35 skipped** (the skips are AndroidWorld-only, never faked). Board: `docs/compute/TODO-2026-11-06.md`.
  **Bug 1 (would have corrupted every run):** every episode in a suite got the *same* `episode_id`, because
  the id used `task_idx` (0 for all). Trajectories would have overwritten each other silently. Fixed with a
  task slug + uniqueness test. **Bug 2 (would have inflated SR):** a crashed episode could still be scored
  `success`, because the official checker ran on a dead env. Now `env_crash` forces `success=False` and
  records `checker_unavailable`. **Bug 3 (would have hidden regressions):** `run_nightly.sh` called
  `liveness --layers all`, which the probe never supported — the "probe failed" seen first was an argparse
  error, not a dead device. `--layers` added (+2 tests); the nightly gate can no longer pass silently.
  **Infra:** `android_world:latest` built with the three runbook patches (image `rc=0`), `aw-emu` up with the
  24 GB cap on host port 5051; cold boot took **215 s** under host load 21.8; `boot_completed=1`; the
  FastAPI lifespan (`emulator_setup`, ~17 APKs) is the slow part, not the boot. Second 4090 has **no torch**
  (so the model server goes on the KVM host, `D1_MEM_FRAC`-capped, using the `imagelearner` env's
  torch 2.6.0+cu124 — the conda base env's torch 2.12.1+cu130 **cannot see the GPU**, driver 12.8).
  **Readouts plumbing unblocked:** `serving/d1_api.py` now accepts `return_readouts` and the harness
  `elements` spelling; `run_d12b_acv2.py` gained `--save-readouts` (the stage-2 blocker).
- **2026-11-04 (loop checkpoint) — Stage-2 diagnosis done; THREE hypotheses refuted, zero code changes to claims.** New script `benchmarks/train/class_bias.py`; artifact `benchmarks/results/2026-11-03-stage0/class_bias.json`; provenance now **70 assertions, all PASS**. (a) **`scroll` is a TYPE error, not direction**: 19 errors on 37 gold-scroll rows = **16 type** (`status` 9, `click` 7) + only **3 direction**; scroll type acc 21/37 = 56.8% ⇒ **do not build a direction model**, ceiling +4.0 pts. (b) **Premature `status` is NOT an episode-boundary artifact**: 87.9% of rows are at their episode's last evaluated step vs 85.7% of the 21 premature rows (enrichment **0.97×**); `p(terminate)` does separate them (0.671 vs 0.882) but "suppress status when grounding is strong" flips 21/21 premature **and 76/76 correct** ⇒ net **−55**, refuted. Also: **AC-v2 is ~1 step/episode, so episode-position hypotheses cannot be tested on it.** (c) **Per-class logit adjustment (2007.07314) works on val, fails on test**: type acc 0.8053→0.8210 val but **0.8417→0.8141 test**, 18 regressed/7 fixed, McNemar **p=0.043 significantly worse** — the val/test action-mix mismatch (F5b) causing a concrete practical failure. **BLOCKED:** any step-level evaluation of a type-changing post-processor needs the `scroll_dir`/`type_text`/`app_name` readouts for the switched-to class (the jsonl stores only the emitted action's `value`) ⇒ needs a ~52 s model re-run that saves all readouts.
- **2026-11-03 (2nd pass) — LEARNED FUSION: BETTER POINT ESTIMATE, NOT A CLAIMABLE GAIN; THE BOUND IS WORTH MORE THAN THE MODEL.** Script `benchmarks/train/score_fusion.py`; artifacts `benchmarks/results/2026-11-03-stage0/score_fusion.json`; 63 provenance assertions now. **(F6)** L2 logistic on 8 per-row readout features (top-1, top-2 margin, 8-way entropy, `box_conf`, `found`, spatial, 2 interactions), hyper-params by 5×`StratifiedGroupKFold(5)` **inside** val with **groups = episode**, test scored once: AUROC **0.767 val (OOF 0.758) / 0.743 test** vs routed 0.722/0.730 and global 0.677/0.649; test **risk@50% 0.226 → 0.191**; AUPRC 0.811 → 0.829. It **rediscovers routing by itself** (`box_conf×spatial` +1.08, `p1×nonspatial` +0.50, `spatial` −0.46). **BUT the paired episode-clustered ΔAUROC = +0.013, 95% [−0.020, +0.043] → NOT significant** (MDE ≈ +0.02 at 398 rows / 140 positives). **Report the estimate + interval; claim nothing.** **(F7) CRC ≫ Clopper–Pearson** (Angelopoulos 2208.02814): same val split, same score → δ=20%: CP 38.4% cov @ 19.6% risk vs **CRC 45.5% @ 18.2%**; δ=25%: CP 60.6% vs **CRC 66.1%** ⇒ **+6–7 pp of coverage free from the bound choice alone**. δ=10% is "certifiable" under CRC at **14 steps / 3.5% coverage** — the cleanest proof that the 10% budget is a risk floor, not a modelling target (do NOT report it as a gate). **(F5b) Shift diagnostic:** val→test action-mix **TV 0.106**, ESS after weighting **752/827 (91%)** — cheap; reweighting can move risk ≤~2 pp, and the 6.6-pp cross-checkpoint gap has binomial SE ~4.5 pp (**≈1.5σ**) ⇒ report gap + SE + explained fraction; qualitative claim survives, quantitative one does not. **Research that drove this (2026-11-03):** PACE (2609.24277) — on GUI click correctness **entropy > top-1 prob > top-2 margin**, and uniform aggregation is the failure mode; SelectiveNet (1901.09192) — learned rejection beats thresholding; CRC (2208.02814); SCRC (2512.12844) feasibility `m ≥ ⌈1/δ⌉−1`; MAST (2503.13657) — premature termination is 7.82% of agent failures (ours 5.28%, in band); SwipeGen (2601.18305) — swipe failures are 69% region/scrollable-target, only 6% direction; AndroidControl-Curated's own rates are **24.13% unclear task / 8.12% multiple valid / 37.63% wrong GT** (the "30%" is the intro's prose, and GUI-Libra reports ~20% — cite 20% or the taxonomy, never "30%"); **no sub-3B model exceeds 66% AC-v2 step accuracy in any verifiable source**; **UI-Venus-2 80.2 is a 9B, not 2–3B** (prior note was a category error); AndroidWorld numbers across papers are **not comparable** (GUI-Libra-32B 34.8 vs UI-Venus-2 9B 57.8) — decline to report AW or declare the protocol. **GPU levers if ever run:** unfreeze ViT+projector (+3–6 pts grounding predicted; full AdamW FT is ~27 GB → **does not fit 24 GB**, ViT-only does), ConClick-style hard negatives from our own 68 wrong-element errors (~4k pairs, +2–4), AMEX (1.2M Android element annotations, CC-BY-4.0, 93 GB).
- **2026-11-03 — ZERO-GPU FORENSIC PASS: THE GATE WAS ROUTED WRONG; TWO CLAIMS DIE.** Full note: `notes/research-2026-11-03-calibration-routed-gate.md`; write-up `benchmarks/results/2026-11-03-acv2-error-budget-and-routed-gate.md`; scripts `benchmarks/train/{error_budget,paired_eval,gate_routed,grounding_audit}.py` (numpy only, seconds, no model). **(F1) Error budget is 4 buckets** — on E2 398: 68 element-wrong (48.6%) / 27 spatial type-wrong / 36 non-spatial type-wrong / 9 value-wrong; type errors 63 vs grounding 68 test, 161 vs 137 val. ⚠️ **161-vs-137 is z≈1.4, p≈0.16 — NOT significant**; say "comparable magnitudes", never "type exceeds grounding". The paper + `notes/grounding-precision-2026.md` §2 "gap is grounding, not type" is **wrong** (it conflated high type accuracy 84.17% with few type errors). **(F2) The gate was mis-specified**: `p(action)×box_conf` on ALL rows is **chance on non-spatial** (AUROC 0.504/0.526) because box_conf is meaningless there. Type-routed (box_conf on spatial, p(action) elsewhere, val-fitted per-bucket ECDF): AUROC **0.760 val / 0.729 test938 / 0.709 test705** vs global 0.677/0.649/0.662. `box_conf` on spatial is 0.79–0.80 and **under**-confident. This is the mechanical reason the old 0.723/0.036/55% never reproduced. **(F3) Two claims must die:** (a) a 10% risk budget at ≥50% coverage is **unattainable** (risk floor ~12.5%; conformal certifies only **20% @ 40.7% coverage**, empirical 17.3%, CP UB 21.7%) → **restate GOAL.md C2**; (b) the conformal threshold **does not transfer across checkpoints** (same τ → 17.3% on E2@938 vs 23.9% on E2@705) ⇒ **exchangeability, not the score, is binding**; cite SCRC (arXiv 2512.12844) for the feasibility condition. **(F4) Grounding metric is HONEST — negative result, keep it:** audit of the a11y trees shows **80.6% of element misses are genuinely different elements** (only 17.9% subtree, 1.5% same-label); tolerance curve 66.16%→72.73%. So the AndroidControl-Curated ~30%-defective-GT argument does **NOT** rescue our 56.5% — **do not claim the metric under-reports us**. **Paired testing added** (`paired_eval.py`, McNemar + **episode-clustered** paired bootstrap — 398 rows span 350 episodes, so cluster by episode): E2@705→@938 = **+3.52 pts [0.76, 6.33], p=0.016** on the same 398 rows where ±4.7 independent CIs see a tie; that gain was +12 non-spatial / only **+2 spatial** ⇒ grounding saturates early, late training buys type. **Artifact gap:** `E2-938-V.jsonl` is the 258/398 **hinted** run; the pure predictions were not on disk at the time (~52 s to regenerate). **Resolved 2026-11-07:** they are on disk now — 255/398 in-process (`E2-nomem`) and 254/398 served (`e2-pure-hist.jsonl`); no artifact yields 256, so the pre-audit 64.32 is retired. **Skills:** `kntism/skills` (source of the installed `model-trainer`) is a 1-star unmaintained scrape — use upstream `huggingface/skills` + `K-Dense-AI/scientific-agent-skills`. **Also:** GUI-JEPA (2610.36877) already does top-2-margin confidence routing (AUROC 0.74) — differentiate on the chance-level product score + type routing + the certificate boundary.
- **2026-11-02 — N1 OBJECTIVE ABLATION RUN (pilot) + ALL SERVER ARTIFACTS REMOVED.** Matched 250-step runs, identical data/base/schedule, only the option readout varied: `ce` / `brier` / `joint`. 12,750 train / 2,250 held-out curated decisions, seed 0, frozen embeddings, V-track. **Result = a clean null:** step acc **0.5658 / 0.5653 / 0.5707** (majority-class `click` baseline **0.5218**), ECE10 **0.237 / 0.243 / 0.242**, Brier **0.339 / 0.329 / 0.331**, AUROC(conf,correct) **0.657 / 0.647 / 0.653**, mean conf **0.80** vs accuracy **0.57**. Proper scoring moves the *average* probability but not discrimination; every arm is severely overconfident. Single seed + 250 steps + in-distribution holdout ⇒ cannot settle the mechanism; must be re-run at full schedule with ≥3 seeds and a hard/OOD split. Files: `benchmarks/results/2026-11-02-n1-objective-pilot.md`, paper §5.11, `notes/exp-n1-objective-2026-11-02.md`; trainer flag `--objective` (default `ce`). **Online loop (same day):** 6-task subset at 8 steps → d1 **1/6**, `wait` **0/6**, `random` **0/6** (matched controls); 12-task attempt aborted when the emulator hung under load. **SUPERSEDED 2026-11-07:** the larger 8-task run is **1/8 in both conditions** (Wilson 2.2–47.1%), and its 2/8 "crashes" were Clock-task checker exceptions on a live device, since fixed. **Cleanup verified complete:** `~/awork`, `~/d1-api`, `android_world:latest` image and `aw-emu` container all removed from `GPU_HOST_A`; `GPU_HOST_B` clean; every pre-existing prod container untouched. Backup: `backup-2026-11-02/` (29 files; N1 checkpoints not transferred — regeneration commands in the result file).
- **2026-11-02 — ONLINE LOOP IS RUNNING; first honest numbers are weak and that is the result.** AndroidWorld emulator is up on `GPU_HOST_A` (KVM, container `aw-emu`, 127.0.0.1:5051) with the E2 checkpoint served at 127.0.0.1:8700. Three Dockerfile defects had to be fixed (dead `openjdk:18` base; `COPY . /` + `WORKDIR /` made `uv pip install .` package the whole filesystem — 90 min/21 GB RSS; Pillow imported but undeclared). Full runbook: `docs/compute/AW_DOCKER_RUNBOOK.md`. **Measured (2 tasks, 12-step cap):** d1 **0/2** (earlier attempt **1/2** — `ClockStopWatchRunning` scored 1.0; run-to-run variance is the finding), `wait` **0/2**, `random` **0/2**; model p50 **113–123 ms**. Details: `benchmarks/results/2026-11-02-online-aw-smoke.md`; paper §5.7 + Limitations 7–9 updated. Interface limits: HTTP `/screenshot` is **pixels-only** (no a11y tree) → the V-track checkpoint returns an **empty `app_name`** (harness goal-keyword fallback = scaffolding, must be disclosed); each screenshot is ~7.8M ints (hundreds of MB) and killed the emulator at a 16 GB cap → 24 GB. **N1 (CE vs Brier vs joint) is prepared but not run:** `notes/exp-n1-objective-2026-11-02.md`, trainer flag `--objective` added (default `ce`, behaviour unchanged); blocked only on the 1.8 GB training-image transfer to the server.
- **2026-11-02 — 8h ONLINE LOOP STARTED (user directive).** Objective + step checklist: `docs/compute/LOOP-2026-11-02.md` (S1–S12). Cleanup ledger (MANDATORY at the end): `docs/compute/CLEANUP-2026-11-02.md` — back up results to `backup-2026-11-02/`, then remove every artifact from `GPU_HOST_A` and `GPU_HOST_B`. Loop job `71dbb002` (every 20 min, session-scoped). Compute plan: AndroidWorld emulator container on `GPU_HOST_A` (KVM, host port **5051**; port 5000 is prod) + d1-2B API on the same host's GPU at 127.0.0.1:8700 with `D1_MEM_FRAC=0.35` (21 GB free). Harness: `benchmarks/aw/aw_smoke.py` (policies d1 / wait / random for the baseline gate). Dockerfile base patched `openjdk:18-jdk-slim` → `eclipse-temurin:17-jdk-jammy` (the old base image was deleted from Docker Hub). **Feynman audit corrections (do not re-derive):** (a) **no** AndroidWorld baseline is peer-reviewed — every number is an arXiv preprint or vendor self-report; (b) `GUI-Owl-1.5-2B = 67.9%` has **no primary source** (the real figures are 71.6% for the 8B-Thinking variant; 2B reports OSWorld-Verified 43.5) — do not cite 67.9%; (c) the HTTP server returns **pixels only**, no a11y tree; (d) success requires device-state `is_successful`, so a stalling agent scores 0. Briefings: `outputs/androidworld-online-loop-briefing.md` (+ Feynman paper-gap brief).
- **2026-11-02 — STRATEGIC PIVOT to the online AndroidWorld/AndroidLab loop + HANDOFF rewrite.** AC-v2 offline declared plateaued (box grading = artifact, `2610.36831`) → grounding V2P-σ plan INVALIDATED, no GPU spend. **Machine decision (verified live 2026-11-02): emulator on `GPU_HOST_A`** (only 4090 with `/dev/kvm`, 32 vmx, 16 cores, 51 GB RAM avail, 137 GB disk, docker 28.2.2); model inference on [cloud-gpu] 4090. [cloud-gpu] $2.80 credit **held unspent**. `docs/HANDOFF.md` fully rewritten (13 sections); it is the entry point. ⚠️ [service] container on `GPU_HOST_A` crash-looping (`Restarting (1)`) — do not touch without OK (`docs/compute/STOPPED_CONTAINERS.md`).
- **2026-11-02 — PAPER REWRITTEN (Feynman writer+verifier skill) + CALIBRATION DISCREPANCY FOUND.** `papers/single-pass-decision-acv2.md` rewritten to `papers/single-pass-decision-acv2.md` (4,615 words, all sections + Sources appendix). Deep-verify (re-ran official scorer + analyze_eval.py): E2 pure **64.32%** (256/398 — *retired 2026-11-07: the artifact is 255/398 = 64.07%*), groundB **64.82%**, app-name-hint **64.82%**; per-type (click 121/212, status 76/87, open_app 20/27, navigate_back 13/20, scroll 18/37, type_text 7/11, wait 3/4); val 62.76% (827/519); grounding decomposition verified — 68 rows = 65 click + 3 type_text (**0 open_app**), median 437px / mean 663px / 86.8% >150px / 4.4% ≤50px / y-bias 0.5px; latency median 103ms; p(action) ECE 0.231 (E2@705) / 0.204 (E2@938), AUROC 0.626/0.631. **⚠️ CALIBRATION DISCREPANCY:** the earlier-cited ECE 0.146→**0.036** isotonic, AUROC 0.723, and risk-gate cov 55.0%/36.7% DO **NOT** reproduce from the saved jsonl (recompute gives p×box ECE ~0.29–0.36; isotonic fit-on-val→test collapses to τ=1.0). Paper now reports only the reproducible evidence and marks 0.036/55%/36.7% as **reported-in-run / pending GPU re-fit** (see `docs/HANDOFF.md` §3.4, `notes/grounding-precision-2026.md`). **SUPERSEDED 2026-11-02:** the V2P + ConClick grounding fix is INVALIDATED (`docs/HANDOFF.md` §3.3); next thread = the online AndroidWorld loop.
- **Orchestration rule (teach the system):** gate on GPU before any heavy job; validate cheap (--limit 5) before a full run; run one heavy job in background with auto-notify, never poll it from chat; record every learning in AGENTS.md + `docs/colab/GPU_RUNBOOK.md`. Read the runbook before touching compute.
- **RESEARCHER MODE (user, 2026-10-29 — applies to EVERY session):** act as an expert researcher / AI engineer, not an executor of old plans.
  (a) Think from first principles and read CURRENT papers/SOTA (spawn research agents for 2026 work) — don't be biased by old code, HANDOFF or prior knowledge; do what is actually right.
  (b) The agent is the DECISION MODEL (d1-2B single-pass); target = beat a SOTA point on AC-v2 (a small margin suffices for the paper).
  (c) Trust published baselines; don't spend GPU re-running them.
  (d) Never wait blind: parity/smoke gate before any long run, read logs, fail fast, maximize GPU throughput; fan out across the 3 servers (GPU_HOST_A first).
  (e) Prod servers: stop a container only if ONE safe command restores it; log it in `docs/compute/STOPPED_CONTAINERS.md`; cap VRAM (`--mem-frac`) on shared prod GPUs.
  (f) Data hygiene: the old curated/completion sets contain AndroidControl TEST episodes (e.g. ep 27) — train only on the GCS-rebuilt train split (`benchmarks/train/preprocess_ac.py`, eval episodes asserted absent).
- **DECISION PRINCIPLES (user, hard-taught — apply BEFORE any compute run):**
  0. **Mac = control channel only (user, 2026-10-28, x3).** ALL model inference/training runs on **Colab T4 via `colab` CLI** (`colab new -s <tag> --gpu T4`, `colab exec -f <local.py> --env K=V --timeout N`, `colab upload local remote`). NEVER launch local MPS model runs — even smoke tests. Mac-side work = CLI control, data packaging, result analysis only. Colab CLI gotchas: `colab exec -f` takes a LOCAL file path; `colab upload` to subdirs (`data/...`) → 500, upload to root then move via exec; free T4 quota recovers daily (balance 0 ≠ dead); idle runtimes get reclaimed → chain create+upload+run in one uninterrupted script.
  1. **Do NOT run a long/full job that risks dying.** A 1-2h run can hang and waste the whole turn. Prefer fast, small, resumable over full/long.
  2. **`--limit <N>` small subset FIRST** for anchors/validation (e.g. 100 steps). Scale up only if the small number is trustworthy + actually needed. `--resume` so progress is never lost.
  3. **One short job at a time.** Kaggle caps concurrent GPU sessions at 2 — long/full runs lock the slots and block everything else. No parallel GPU kernels.
  4. **Before every heavy run ask: "is this number actually needed for the goal?"** Save compute; be smart, don't run for run's sake.
  5. Get the anchor from a small subset + published number as cross-check, not from a full 398 run for its own sake.
  6. **Learnings are durable** → write re-usable knowledge into `docs/colab/GPU_RUNBOOK.md` + AGENTS.md so every future session starts already knowing this.

### VERIFICATION GATE — do this BEFORE calling any number a 'result' (HARD, taught 2026-10-16)
Never trust an accuracy/calibration number until you check these. I once reported acc=0.5828 as a REAL
result when it was EXACTLY the majority-class baseline (always-predict 'click') — a degenerate prior-only
solution, not a learned model. Don't repeat this.
1. **Compute the trivial baselines FIRST and show them next to your number.** For any classification task:
   - majority-class (always-predict the most common label), and
   - uniform random (1 / number of classes), and
   - for imbalance, per-class tuned baselines.
   If your model's accuracy is ≈ (or ≤) the majority-class baseline, **your model learned nothing** — the
   task framing/interface is wrong (class imbalance, wrong target, or the input carries no signal). STOP,
   don't report it as a result.
   - G1/G2/G3 reality check: curated K=7 action-type has majority 'click' = 0.5828 (test). My single-pass
     head hit exactly 0.5828 for BOTH text and multimodal → it was just predicting the prior. NOT evidence.
2. **'Trains too fast' / too-easy-to-converge is a red flag, not a win.** A frozen-encoder + tiny linear head
   on 6.8k samples hitting the majority baseline in 60 epochs means the head is degenerate, not that the
   task is solved. Real decision models need much more than a linear probe (see §'How they actually train').
3. **Report BOTH the model accuracy AND the baselines together**, so the reader instantly sees the gap. A
   standalone accuracy number is meaningless without the reference point.
4. **Skeptic's checklist before claiming a meaningful result:** (a) beats all trivial baselines by a real gap;
   (b) not explained by class prior; (c) checked the architecture/method matches the SOTA method (not a
   simplified proxy that can't work).
5. **Ported scorers must be DIFFERENTIAL-TESTED against the original before trusting ANY number**
   (2026-10-28 lesson). The `score_row` port applied the key-subset fallback to SPATIAL steps, so click
   steps counted correct on action-TYPE alone: UI-Venus-1.5 "0.95 step acc" was really type-only accuracy
   (grounding 0.000 was the tell). Fixed `run_acv2.score_row` now mirrors `eval_official.py` and passed a
   312/312-row differential test (0 mismatches) on real predictions. Both contaminated result files
   (2026-10-14 llama smoke, 2026-10-15 UI-Venus baseline) are bannered RETRACTED.

### MISTAKE LEDGER — the errors this programme has actually made, and the rule that prevents each

Taught 2026-11-09 after the operator said: *"remember these things in AGENTS.md that make sure don't make it
again"*. Every row below is a real error from this programme, not a hypothetical. **Read this before starting
any new phase.** If a proposed action matches the "how it happened" column, stop.

| # | The mistake | How it happened | The rule that prevents it |
|---|---|---|---|
| M1 | **Read the dataset card, not the data.** Concluded GUI-Odyssey's `TYPE` action had no coordinates, so the corpus "cannot teach field selection". | Card said `info` is empty for `TYPE`. The real file has action **`TEXT`** (not `TYPE`), and **99.27%** of `TEXT` steps are preceded by a `CLICK` carrying the field coordinates. The wrong conclusion was one step from cancelling the whole training plan. | **Never characterise a dataset from its card or abstract.** Download the annotation file, parse it, and count. A card is marketing copy; the file is the contract. Report counts, not claims. |
| M2 | **Conflated per-step accuracy with task success.** Framed "64% per-step" as the headline achievement. | They differ by ~8x here (64.07% per-step vs 18.8% end-to-end). MAI-UI Table 9 splits them (92.3% per-step, 45.1% e2e) precisely because they are different quantities. | **A per-step number may never be presented as a task-result number, and never in the same column.** Every rate names its unit *and* its denominator. |
| M3 | **Invented a number to close an arithmetic gap** (a spurious "6"). | Wanted the totals to balance. | **Never fabricate a figure.** If totals do not balance, report the discrepancy. A hole in the arithmetic is a finding; a filled hole is a lie. |
| M4 | **Compared numbers across benchmarks/harnesses.** Quoted AC-v2 and AndroidWorld figures against other papers' figures. | Same backbone scores 49.62 (ECHOES) vs 54.80 (Mem-W) — a **5.18-point protocol spread** on identical weights. AC-v2 is not rankable across papers. | **Never compare a number to a number produced by a different harness, protocol or split.** Either reproduce the baseline in our own harness, or decline to rank. Say "not comparable" out loud. |
| M5 | **Scored unscorable episodes as model failures.** 12 episodes whose checker raised were bucketed as timeouts/premature terminations. | `AndroidWorldEnv` had no `__getattr__`, so `task.is_successful(env)` silently returned `False`. | **An infrastructure failure is not a model outcome.** Every bucket must be disjoint, and the harness must prove the environment was alive. `unscored` is its own category. |
| M6 | **Trusted a single seed.** Reported grounding deltas from runs differing by 8 samples out of 15,008. | `--steps 938` never completes an epoch, so "duplication" vs control differed only by shuffle order — yet differed by 4 points. Seed spread measured at **sd 2.12, range 4.04**. | **No GPU delta is reportable without ≥3 seeds and an interval.** Pre-register the withdrawal rule ("sd ≥ 2.0 → withdraw") and honour it. |
| M7 | **Changed a claim to match a hoped-for result**, then had to retract 11 claims. | Iterative "fix-and-re-fix" churn. | **The artifact defines the number, never the reverse.** Recompute from artifacts; if the paper disagrees, the paper is wrong. `check_record.py` enforces byte equality for exactly this reason. |
| M8 | **Read a subagent/skill name and assumed it existed.** Invoked `arxiv-search` on a repo that only ships `textual-screenshot`; it failed with exit 1. | Did not list the available skills first. | **List before you invoke.** `npx skills use <repo>` with no `--skill` lists what is there. Never assume a URL contains the skill you were told about. |
| M9 | **Assumed a branch contained our work.** Branch `vinh` was assumed to hold revisions to this research repo. | It is an *orphan branch of a different project* (a PTIT mobile-agent demo export), 1089 files, no relation to this repo's tree. | **Inspect before merging.** `git ls-tree -r --name-only HEAD` and read the manifest before assuming a branch's contents. Different tree = different project = do not merge, report it. |
| M10 | **Trusted the card's action vocabulary over the file** (same failure class as M1, separate instance): card listed `IMPOSSIBLE`, `HOME`, `BACK`; the data has `INCOMPLETE` and **zero** `HOME`/`BACK`. | Card written for an older/newer schema than the file shipped. | Same as M1. Additionally: **when a card and a file disagree, the file wins and the discrepancy is itself a finding worth recording.** |
| M11 | **Omitted a denominator, then read the number as if it were complete.** "18.8%" silently meant 15/80, not 15/116. | Reported the rate before naming the base. | **A rate always carries its denominator and its interval.** We report three (scorable 80, attempted 92, finished 104) and label which question each answers. |
| M12 | **Built a post-processor on a metric that was already refuted.** Proposed a `p(status)` threshold gate after it had been measured monotone-harmful at every τ. | Did not check the negative-results list first. | **Read the recorded negatives before proposing a fix.** If the literature or our own record already contradicts it, the proposal must beat that result or be reported as contradicted. |
| M13 | **Cited a DERIVED number as if the paper reported it.** AGENTS.md and the paper stated "MAI-UI-2B 92.3% per-step after SFT, 93.1% after RL" and built a "28-point per-step gap" on it. | MAI-UI publishes **no per-step metric at all**. The numbers were produced by *us* inverting `P(task)=p^n` on its task-success figures, with n=10 assumed — then cited as `[maiui2026, Tab. 9]`. The 92.3 string in the paper is an unrelated MMBench-GUI cell. Our own raw capture `references/raw/maiui2026-androidworld-table9.txt` disproved it the whole time. | **A derived number must always be labelled derived, and never carries a table citation to the source it was derived FROM.** Separate "what the paper reports" from "what we computed from it" in every sentence. And test your own assumption before building on it: `P(task)=p^n` predicts 2.84% at 8 steps while we measured 18.8% — a 6.6x refutation that lived in our own artifacts. |
| M14 | **Let a derived assumption stand as a prediction.** Used `P(task)=p^n` to "retro-explain" a 1/8 result and to reassure that a harness bug could not have mattered. | Step independence is false: GUI steps are strongly dependent, and easy screens have far lower error than the 64% average. The 6.6x discrepancy was measurable and unremarked. | **Test the assumption with your own data before reasoning from it.** If a model predicts something we can measure, measure it. A refuted premise invalidates every conclusion drawn from it, including the comforting ones. |
| M15 | **Tested an invariant on hand-picked inputs, not on the real population.** Episode-id uniqueness was tested on 3 unrelated short task names; the real 116 contain 6 groups sharing their first 24 characters, and 12 tasks were silently never run in any 116-task run. | The function's docstring said "collision-free" and the test agreed with the docstring, not with the data. | **Test every invariant over the actual population it must hold for** (here: all 116 registry names), and make the system refuse a violation at plan time rather than absorb it at resume time. A count that never moves (104/116 three runs in a row) is a finding, not noise. |

**The three meta-rules behind all of them:**

1. **Measurement beats assertion.** Every claim in this repo must be reproducible from a file that is on disk.
   If it cannot be, it is a hypothesis and must be labelled one.
2. **The negative result is the deliverable.** A refuted hypothesis, correctly measured, is worth more than a
   confirmed one measured wrongly. Never bury a negative; it is the only thing that stops the next person
   repeating the work.
3. **Read the primary artifact.** Card, abstract, README, manifest, branch name — all are descriptions.
   The data, the artifact and the file are the thing itself. This one rule would have prevented M1, M8, M9, M10.

---

### How advanced agents are ACTUALLY trained (2026-10-16 — MobileRL, arXiv 2509.18119)
The frontier mobile-GUI-agent method is NOT 'frozen encoder + single-pass action classification head'.
MobileRL (Tsinghua/Z.AI, ICLR 2026) trains open VLMs (Qwen2.5-VL-7B, GLM-4.1V-9B) via a 3-stage pipeline:
  1. reasoning-free SFT on expert demos (AndroidControl train + human annotations),
  2. reasoning SFT (add intermediate reasoning traces),
  3. online agentic RL with AdaGRPO (sparse binary terminal reward 0/1 at task completion; difficulty-adaptive:
     AdaPR positive replay, FCF failure curriculum filter, SPA shortest-path reward adjustment).
Results: MobileRL-9B = 80.2% AndroidWorld / 53.6% AndroidLab; 7B = 72.0% / 42.5% (prior SOTA 64.2% / 41.2%).
Key takeaway: the policy is an **autoregressive VLM sampling one action per timestep over a multi-step
trajectory** with RL from sparse terminal reward — NOT a single-pass over-K choice. So a single-pass decision
model (our System One thesis) is a genuinely DIFFERENT, harder-to-justify framing vs the RL-multi-turn SOTA:
any honest comparison must either (a) match the trajectory-level success-rate metric, or (b) explicitly justify
single-pass on a benchmark that rewards calibration/latency, not raw task success.
**So: my 'action-type classification acc=0.58' was the wrong interface AND wrong metric.** The right unit is
per-task success rate over a full episode, computed against the RL SOTA baseline.
- **2026-10-16 — G1–G4 de-risk COMPLETE on the LOCKED curated benchmark.** Multimodal T4 kernel (`embedgemma-curated-t4`) actually ran to COMPLETE and produced text+multimodal embeddings (768-d) cached to `benchmarks/data/curated/emb/` (gitignored). Reused those cached embeddings for the train-only G2/G3 run (`benchmarks/_g5_train_emb.py`) — no re-encode/download needed. G4 risk-gate in `benchmarks/colab/g4_riskgate_t4.ipynb`. Findings: multimodal = calibration (not accuracy); proper-scoring collapses risk-gating confidence. See `benchmarks/results/2026-10-16-g2g3-g4-text-multimodal.md`.
  - **HONEST RETRACTION (2026-10-16, user flagged):** those G2/G3 'accuracy' numbers (0.5828) are NOT a real
    result — they exactly equal the majority-class baseline (always-predict 'click' = 0.5828 on test). The
    frozen-embedding + linear-head single-pass on K=7 was a **degenerate prior-only solution**, and the whole
    'action-type classification' interface was the wrong task framing. The only genuinely useful finding is the
    G4 caveat (proper-scoring collapses confidence spread → unusable for risk gating) and the calibration vs
    accuracy distinction; the accuracy numbers themselves must not be cited as evidence. Next step should be
    re-framed to per-task episode success-rate vs RL SOTA, or the single-pass claim justified on a
    calibration/latency benchmark rather than raw task success.
- **Kaggle cleanup (2026-10-16):** deleted all stale kernels (embedgemma-curated-t4-l1-q2-action-type, ac-v2-decision-train, ac-v2-sota-baseline, lfm2-5-smoke, gpu-test-4, ronaldo-swap-face) and stale datasets (acv2-sota-data, acv2-decision-train-data, sampleimage). Kept active kernel `embedgemma-curated-t4` + new `g4-riskgate-t4`. New dataset `curated-actions-text` (train/test jsonl) for the G4 kernel.
- **2026-10-29 — d1-2B full fine-tune on AC train, 3-GPU local SGD (run `acv2-mix-r1`).** Plan + rationale:
  `notes/plan-2026-10-29-d1-acv2-sota.md`. Key corrections found by verification: (1) the leaderboard harness
  (GUI-Libra `guilibra_plan.py`) DOES give history (`previous_actions`) — our goal-only premise was wrong;
  (2) leaderboard scorer is `eval_rl.py` (≡ our port except no point for non-spatial preds — eval_d1 mirrors it);
  (3) old curated/completion data leaked TEST episodes → rebuilt from GCS (88,288 train decisions).
  Gates passed: parity 24/24, grad-ckpt, ckpt reload, sync smoke. Code: `benchmarks/train/`. Ops rules:
  `docs/colab/GPU_RUNBOOK.md` (2026-10-29 section). Restore [service] container when done
  (`docs/compute/STOPPED_CONTAINERS.md`).
- **2026-10-29 RESULT — d1-2B decision model = 64.32% AC-v2 High Pass@1 (OFFICIAL eval_rl.py, PURE protocol; 95% CI 59.5–68.8) — *superseded 2026-11-07: the artifact is 255/398 = 64.07%, no 256 exists; see the audit entry above*
  (64.82 with an extra train-goal app-name hint) → on par with GUI-Libra-8B 64.3 / Mem-W-4B 63.07 (within noise),
  NOT a SOTA break (Mem-W-8B 68.59).** Final pre-fixed checkpoint of run E2
  (V-only, 15k curated decisions, lr 1e-5, frozen tied embedding, 938 steps, ~1.6 h on one RTX 4090). Run-to-run band
  of V-only finals: 62.3–64.8. Validation split 62.76 (consistent). Type 84.17 (> Mem-W-8B 81.16); gap = grounding
  57.4 vs 71.3. Calibrated gate (isotonic fit on val): test ECE 0.036; risk budgets 25%/20% held on test
  (cov 55.0%/36.7%, risk 23.3%/18.5%). Negative ablations: mixed V+VA training −2–4; TF-IDF procedural memory −6.0;
  more data (24k) / continued training (+8k) no gain. Lessons: overfit-test caught lr 3e-5 destroying box heads.
  Full table: `benchmarks/results/2026-10-29-d1-acv2-sft.md`; ckpt backed up to `backup-2026-10-29/ckpt-E2-final`.
- **2026-10-28 — SCORER BUG FOUND & FIXED (before believing any AC-v2 number).** The `run_acv2.score_row` port counted SPATIAL steps (click/long_press/type_text) correct on action-TYPE alone (key-subset fallback leaked into the spatial branch; pred_action had no x/y so 'click' matched 'click'). Consequences: (a) 2026-10-15 UI-Venus-1.5 baseline "0.95 step acc" was TYPE-only (grounding 0.000 was the tell) — now bannered RETRACTED; (b) 2026-10-14 llama smoke bannered; (c) d1-2B's first full run (0.66 at 160 steps) was inflated — killed, partial quarantined. Fix: score_row now mirrors `benchmarks/data/acv2/eval_official.py` line-by-line (spatial = point-in-gold-element, no fallback; equivalences bidirectional incl. the app_name='' quirk) and passed a **312/312 differential test vs the original process_file (0 mismatches)**. Rule added to VERIFICATION GATE §5. d1-2B vision rerun (fixed scorer) in progress: ~0.20 full-step at 160/398 — its zero-shot strength is action TYPE (~0.66), its weak spot is visual grounding (boxes biased to screen top). The a11y choice-over-K variant (`run_d12b_acv2_ax.py`, candidate coverage 0.97) is the next measurement.
- **2026-10-28 — PIVOT: LFM3 + d1-2B release changes the base-model decision.** Liquid shipped the LFM3 family (Oct 27): one 1.73B tri-hybrid base (13 gated conv k7 + 5 SWA-256 + 6 global attn; head-wise gated attn, partial RoPE, sandwich norm; HW-in-the-loop search over 14 devices) → 6 models incl. **d1-2B** (1.92B decision model on LFM3-VL-2B): single pass, 0 tokens, **NEW `box`/`boxes` grounding primitive**, beats d1-3B everywhere (Decision Index 51.1 vs 49.0; ImajevBench 75.5 vs 64.0), 5.7ms H100 / ~109 states/s batched on M5 Pro mps. No paper — vendor self-scored only; **no AC-v2/AndroidControl number published** → that's the gap. Notes: `notes/lfm3-d1-2b.md`; ledger: `liquidAI2026LFM3`/`liquidAI2026opend1`/`lfm3vl2b2026`. Phase 0 = measure d1-2B on AC-v2 398 (`benchmarks/run_d12b_acv2.py`, official scorer, resumable JSONL); smoke `--limit 5` in progress. Phase 1 base if fine-tuning needed: LFM3-VL-2B/d1-2B (NOT LFM2.5/EmbeddingGemma).
- **2026-10-15 — Step 0 (SOTA anchor) moved to KAGGLE.** Colab T4 free quota exhausted (balance 0, `Service Unavailable`). Switched the AC-v2 SOTA baseline eval to Kaggle free (30h GPU/wk, single account `tuannnnn2222222`), same `run_acv2.py` (High-v2 goal-only protocol) — only the compute engine changed.
  - **Kaggle operational learnings (hard-won, do NOT rediscover):**
    - `kaggle datasets create` **skips directories by default** → must pass `-r zip` (or `-r tar`) to include an `images/` dir; a bare `.tar` file was silently mis-uploaded.
    - Strict macOS: strip `._*`/`.DS_Store` before upload (`find <dir> -name '._*' -delete`).
    - **Dataset mounts nested**: real path is `/kaggle/input/datasets/<owner>/<slug>/` (NOT `/kaggle/input/<slug>/`). Discover it via `glob("/kaggle/input/**/<file>", recursive=True)`.
    - **`enable_gpu`/`enable_internet`/`is_private` must be STRINGS** (`"true"`), not booleans — boolean true silently falls back to CPU + no internet (caused `torch ...+cpu cuda False` and pip DNS failures). GPU type: `machine_shape` in metadata; a bad value (e.g. `NvidiaTeslaT4`) can silently fall back to CPU → 16GB model load "RUNNING" >55min on CPU. **Set `"machine_shape": ""` (let Kaggle allocate default GPU) AND add a fail-fast `assert torch.cuda.is_available()` guard so a missing GPU aborts fast instead of hanging.
    - Kernel dir = `code_file.ipynb` + `kernel-metadata.json`; `kaggle kernels push -p <dir>` pushes AND runs. `kernel_type` lowercase `notebook`. Title must resolve to `id` slug.
    - **Kernel setup is self-contained**: embed `run_acv2.py` + `calib_metrics.py` as strings and write to `/kaggle/working` (companion `.py` files don't reliably land in cwd).
  - **`run_acv2.py` calibration summary already covers Step 1** (ece_binary/brier_binary/mean conf/conf-if-correct-bad/latency median-p95/coverage–risk) — no code change needed.
  - **Status:** validation kernel (`ac-v2-sota-val`, `--limit 5`) was running/loading the 16GB `UI-Venus-1.5-8B`; background monitor auto-triggers on completion. Data+GPU+internet fix applied (string flags + glob discover + machine_shape).
  - **BLOCKER (GPU, 2026-10-15):** Kaggle kernels **never get GPU** via CLI — `torch 2.11.0+cpu cuda False`, `nvidia-smi` missing — across every config tried: `enable_gpu` boolean/string × `machine_shape` `NvidiaTeslaT4`/`""` × `--accelerator` `GPU_T4X2`/**`NvidiaTeslaT4`** (the *correct* accelerator ID per `docs/kernels.md`, = GPU T4 ×2, also the **default** GPU). Fail-fast `torch.cuda.is_available()` guard aborts fast. Colab T4 also `Service Unavailable`. Since even the DEFAULT GPU isn't allocated, this is an **account-level** browser/web GPU-enablement issue, not code. Path: enable GPU via Kaggle web UI (Notebook → Settings → Accelerator → GPU T4 ×2) or browser-run a GPU notebook; OR wait for Colab T4 quota reset.

### Goal (RESOLVED 2026-10-13 — see `notes/GOAL.md` §8 decision log)
- **Paper (primary goal):** *"Calibrated to Be Trusted: Training Single-Pass Decision Models for Risk-Controlled On-Device Action Selection"* — claims C1 (mechanism: proper-scoring training preserves calibration where CE degrades, on hard/OOD split), C2 (system: risk ≤10% act/ask/refuse at ≤100 ms on M1 Pro; big LLM can't meet risk budget on-device), C3 (one-command reproducibility).
- **Superseded positions:** "first calibration-reliability benchmark" → taken by TypedBench (Oct 2026) + AgentAbstain (Jul 2026). "First open calibration recipe" → taken by THX-01 (RLCD, Oct 2026) + decider (Sep 2026). We now *cite* them and own Q1/Q2/Q3 (breakdown regime + mechanism comparison + Apple-silicon venue).
- **Base model:** LFM2.5 1.2B → 2.6B (LFM Open License v1.0 — read full terms before releasing fine-tuned weights). Text-state only on this Mac (vision = stretch).
- **Not the goal:** raw-accuracy SOTA (we lose that axis by design; decider-2b 0.80 / d1-3B / Jev own it).

### Phase 1 key findings (grounding for Phase 2)
- **Definition:** a *decision model* reads state + typed questions once and returns calibrated probabilities over a fixed answer set in one forward pass, zero output tokens. Primitives: `noul` (yes/no), `choice`, `score`.
- **Category frontier:** d1-3B = 48.57 on Decision Index v0.2.1 (better than all <10B, ~par Decider 35B-A3B) — but **Winnow-12B (50.02) is #1**; hosted Jev is #3 (60.1) on DI v0.3, Perplexity Decider v1.1 leads (62.8).
- **Build path (recommended):** you likely **don't need heavy fine-tuning first** — frozen strong LLM + bracket-numeric scoring reaches 81.4% / ECE 0.057 (LLM2Jev). Fine-tuning gives targeted gains (LoRA + KL anchors → 84.0% / ECE 0.050). d1's recipe = weight-average backbones + seed/mixture merge + **long inputs, option shuffling, data-shortcut removal**.
- **Calibration is the differentiator & easiest to break:** RLVR collapses confidence (~0.99, Brier 0.43); RLCD scoring the readout with a proper score is the fix; ECE is definition-sensitive across binning.
- **Deployment path:** only published on-device path is **llama.cpp/GGUF** (`/v1/systemone`); **no Core ML / ONNX / MLX** export for d1. Mobile == custom conversion (CoreML/ONNX/WebGPU). NVFP4 on d1 = **claim, not verified**.

### Key caveats to carry into Phase 2 (skeptical flags)
- **Single-pass decision model is NOT how SOTA mobile agents are built** — frontier methods (MobileRL, Magma,
  UI-Venus, GUI-Libra) train autoregressive VLMs with multi-turn RL from sparse terminal reward. A
  frozen-encoder + single-pass over-K classification head is a fundamentally different (and weaker) framing.
  Any single-pass claim must be justified on a benchmark that rewards calibration/latency, not raw task success,
  and must be compared to the RL-SOTA success-rate metric + baselines (majority/random).
- **Always beat the trivial baselines first** (majority-class / random), or the number is meaningless (see
  VERIFICATION GATE above). My earlier `acc=0.5828` was exactly the always-click baseline — do not repeat.
- d1-3B numbers (`48.57`, latencies) are **vendor self-scored**; no independent replication found.
- Mean benchmark number **drifts** (82.9 = 7 datasets vs 77.1 = 8 incl. HelpSteer2) — pick one and state it.
- **License:** d1 under LFM Open License v1.0 (source-available-style, not OSI) — read full terms before fine-tuning/distribution.
- Jev's RLCD is **unpublished** (only named).

---

## 3.5 Gates — the only place a check count is stated

<!-- GATES:BEGIN (machine-refreshed by scripts/check_record.py; do not edit by hand) -->
```
check_record.py            158/158
verify_paper.py            69 assertions, exit 0
verify_audit_objections.py ALL OBJECTIONS CLOSED
verify_compound_accuracy.py ALL CHECKS PASS
benchmarks/aw/tests        245 passed, 43 skipped
```
<!-- GATES:END -->
Dated program-log entries below describe what was true **on their date** and quote the counts of that moment
(e.g. "41/41", "70 assertions", "32/32"). They are history: a count inside them is a fact about that day, not
about today, and must not be corrected to today's value — that would destroy the record. Three attempts to
keep a dated entry's count "current" produced three stale numbers, which is why the live counts live only in
the block above. Every line of it is measured on each successful `check_record.py` run, and if a checker fails
the block says so rather than repeating a claim.

---

## 4. Citation & reference rules (NEVER lose a citation)

1. Every quantitative claim, architecture detail, or comparison must carry a `bibkey` (see `references/ledger.md`).
2. Primary vendor/paper claims go in `references/ledger.md` with the exact source URL + retrieval date; raw verbatim captures go in `references/raw/`.
3. BibTeX lives in `references/bib.bib` (keep it in sync with the ledger).
4. When writing a note or paper, cite as `[bibkey]` (e.g. `d1-3B = 48.57 on Decision Index v0.2.1 [liquidAI2026opend1]`).
5. Distinguish **primary** (vendor blog / HF card / arXiv) from **secondary** commentary.
6. Flag unverified vendor claims (e.g. Jev's 193.6x faster / 444.6x cheaper) as *claims, not evidence*; prefer independent replications ([jevVsLLMs]).
7. **Backfill rule:** if you add a fact, either it's already sourced or you add a ledger entry first.

---

## 5. Directory structure

| Path | Purpose |
|---|---|
| `AGENTS.md` | This master map. |
| `README.md` | Orientation entry point. |
| `notes/` | One note per topic (Jev, d1, LLM2Jev, calibration, benchmark, deployment, agents). |
| `references/` | `ledger.md` (citation ledger) + `bib.bib` + `raw/` verbatim captures. |
| `papers/` | The 2024→now literature (PDFs + bib), organized by theme. |
| `benchmarks/` | Decision Index definition + evaluator + our custom mobile/edge benchmark. |
| `models/` | Base-model choice, decision-head design, training/fine-tuning recipe. |
| `mobile/` | Edge/mobile deployment: converters, latency profiling, quantization results. |
| `paper/` | Our paper drafts. |
| `docs/visual/` | `answer-me-with-html` visual pages (roadmap, architecture diagrams). |

---

## 6. Tooling (how to do the research)

- **agent-browser** (Chrome automation): `agent-browser open <url>` then `agent-browser read` / `snapshot` / `get text` to pull primary sources verbatim. Save captures to `references/raw/`.
- **subagents** for bounded parallel research (one child per topic/seam; each returns a structured note + led sources).
- **web_search / source_check / fetch_content** for literature discovery (`papers/`).
- **`answer-me-with-html`** (`node ~/.pi/agent/skills/answer-me-with-html/scripts/am.mjs render -`) for one-page visual explainers of relations/architecture; render to `docs/visual/`.
- **`+`todo** for phase tracking.
- **Feynman skills are mandatory for research and writing (user, 2026-11-07).** For any deep-research,
  literature-review, paper-writing, research-review, session-log, pdf-explore or ml-training-recipe task,
  load the matching skill from `~/.pi/agent/npm/node_modules/@companion-ai/feynman/skills/` and follow it.
  Deep research must leave the five artifacts under `outputs/` (plan, draft, cited, final, provenance) and
  must never end chat-only after plan approval. Write for the reader's real-world value — reviewer,
  builder, end user — not for volume. Never skip the Feynman step to save time; it is the step that keeps
  the work from being correct-but-inert.

---

## 7. Ground rules for anyone (agent or human) working here

- **Never repeat private infrastructure details** — host/SSH aliases, IPs, usernames, home/key paths, and
  client or prod service names. Say "the 4090 GPU host", "`[ip]`", "`[user]`". Redact them from any file,
  doc or commit before it leaves the machine.
- **In chat, never echo those details back.** Same rule applies to replies, not just files: refer to hosts as
  GPU_HOST_A/B/C, and to IPs, users, keys and prod services as `[ip]`, `[user]`, `[key]`, `[service]`.

- **Keep the todo list current at all times** — every todo that is done must be marked `completed` immediately; never leave a finished task `in_progress`. A stuck board is a deadlock. Re-check `+todo` before claiming progress.
- **Keep the phase tracker (this file) in sync** with what's actually done and what's open. Update it as work moves; don't let it drift.
- **Proof over prose; a value gate before effort.** Before spending effort (especially GPU), state the
  decision the work changes, the effect size predicted, and the result that would make us stop. A polished
  deliverable that changes no decision is a failure, not a success. Apply the Feynman explain-back test: if
  a claim cannot be taught to a non-specialist in one paragraph without hedging, it is not understood yet.
- Keep the concept map and phase tracker current.
- One topic per note file; link related topics by `bibkey` and cross-reference.
- Don't delete or overwrite raw captures (`references/raw/`) — that's the lossless record.
- Prefer primary sources; mark vendor claims as claims.
- Focus: this is a research program, not a general chat repo. Every file should serve the thesis or a phase.
