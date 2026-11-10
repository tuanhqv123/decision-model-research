# HANDOFF — Decision Model Research Program

*Updated: **2026-11-09**. This is the single entry point for a new session. Read this, then `AGENTS.md`, then
`notes/GOAL.md`. Servers are clean and nothing is running.*

---

## 0. Where this stands, in one paragraph

We fine-tuned the open **d1-2B** decision model (1.92B, single-pass, zero output tokens) on 15k curated
AndroidControl-v2 decisions. It reaches **64.07% AC-v2 step accuracy (255/398)** — a tie with GUI-Libra-8B at
4× fewer parameters — with **84.17% action-type** accuracy and **56.05% grounding**. The calibrated act/ask/refuse
gate is worth a **cost window of 2.11–5.79×** (C_err/C_ask) at its measured operating point. The first full
AndroidWorld 116-task run then produced **104/116 finished, 15 successes**, and — the part that matters more than
the rate — it exposed that **per-step decision quality and task composition are separate capabilities**: at 64%
per-step accuracy, `P(task) = p^n` predicts **2.8% over 8 steps**, and MAI-UI's published table shows their 2B SFT
checkpoint runs at **92.3% per-step** before any RL. The 28-point gap is **supervision** (multi-step trajectories,
a judge-driven data loop, a 50-step horizon, a 12-action space), **not** RL and **not** this model's code.

## 0.1 What is DONE and what is NOT — read this before planning anything

| | status |
|---|---|
| Value brief (`outputs/calibrated-decision-value.md` + `.provenance.md`) | **DONE** — 39 sources (31 external / 8 internal), all cited, pre-registered 5/5 |
| Claim audit — 64.32%→64.07%, grounding 56.50→56.05, coverage 96.9% | **DONE** — 11 retractions bannered; `verify_paper.py` 69 assertions |
| Stage-2 status-downweight arm | **DONE — NEGATIVE**, fails its frozen bar (val +1.09 p=0.44, test −3.27 p=0.15) |
| Online blockers: Clock "crash", empty `app_name` | **DONE** — both root-caused and fixed; read slot probe 1/8 → 8/8 |
| **AndroidWorld 116 tasks** | **PARTIAL — 104/116.** 12 never reached a usable device (accessibility forwarder degrades after hours); 12 ran but their checker raised. **Not a model limitation; a harness defect that is now fixed** |
| **AndroidLab 138 tasks** | **NEVER RUN.** Not started, no harness, no results |
| 40 tasks × 3 seeds (the seed protocol) | **NOT STARTED** |
| On-device (Apple silicon) latency | **NEVER MEASURED** |
| Coordinator B1/B2/B3/P1 on the live loop | **CODE DONE, NOT RUN** |

**The single most misleading thing a new session could do is quote 104/116 as "the 116-task benchmark".** It
is 104 of 116, and the missing 12 have a documented cause. Say both halves.

## 0.2 The defect that was scoring an unscorable episode as a model failure

`awenv_bridge.AndroidWorldEnv` had no `__getattr__` while AndroidWorld's own `task.is_successful(env)` calls
`env.get_state()`. Every state-reading checker raised `AttributeError`, `success` stayed at its initialised
`False`, and **12 episodes that ran 4–25 steps were reported as 6 timeouts and 6 premature terminations** —
model failures no checker ever assessed. This is a **second, independent** defect from the 12 setup failures
(disjoint except 11 rows). Fixed at three levels and covered by three tests. **The rate that answers "how good
where we could judge" is 18.8% (15/80), Wilson 11.7–28.7%** — it appears in the paper, GOAL and this file, and
it was reported nowhere before the fix.

## 0.3 If you resume the AndroidWorld run

The image, the container and the harness copy were all removed at teardown. To continue:

1. `cd ~/aw-build && docker build -t android_world:latest .` (~40 min first time; the Dockerfile in place
   already carries the three required patches: `eclipse-temurin:17-jdk-jammy`, `COPY . /app` + `WORKDIR /app`,
   and `RUN uv pip install --system pillow`).
2. Run with `--device /dev/kvm --security-opt seccomp=unconfined --cpus=6 --memory=24g --memory-swap=24g
   -p 127.0.0.1:5051:5000`. **`--network host` does not work** — the emulator dies with
   `TooManyRestartsError` and collides with a stale host `adb` on 5037.
3. Container→model needs the model bound to `0.0.0.0` **and** an `iptables INPUT -i docker0 -p tcp --dport 8700
   -j ACCEPT` rule (host `INPUT` policy is DROP). Both were reverted at teardown; re-add them.
4. Set `environment.emulator_setup: false` in the config — the APKs install once per container.
5. **Chunk the suite and restart the emulator every ~20 episodes.** This is the fix for the 12 that never ran:
   a 116-task suite needs ~2.5 h of continuous uptime and the forwarder degrades. The runner is resumable and
   `scripts/refresh_aw116.py` recomputes from whatever is on disk, so this is scheduling, not code.
6. **The 12 `unscored` episodes should be re-run with the fixed bridge** — they may succeed. That is the
   cheapest real gain available: 12 episodes × up to 3 potential successes at ~18.8% base rate.

## 0.4 AndroidLab is not started at all

The operator's target was **both** MobileRL benchmarks: AndroidWorld (116) and **AndroidLab (138)**. Only
AndroidWorld was attempted. AndroidLab has no harness, no config, no results. MobileRL's Docker image
(`xuyifan0731/mobilerl-androidlab-eval`) is the intended path and was never pulled.

## 0.5 Gates, and how to trust this file

```
check_record.py             158/158
verify_paper.py             69 assertions, exit 0
verify_audit_objections.py  ALL OBJECTIONS CLOSED
verify_compound_accuracy.py  ALL CHECKS PASS
benchmarks/aw/tests         135 passed, 35 skipped
```

**Four auditor rounds rejected completion, and every objection was real.** They named, in order: stale
numbers in HANDOFF/GOAL; the missing `app_name` repair; the unbuilt-vs-built confusion; the 1,344/1,394 drift;
the `get_state` defect above; and finally a HANDOFF line concatenated three times by a sync regex that matched
its own output. **The lesson to carry forward is not "the checks pass" — it is that each round found something
the previous round's checks could not see.** Every one of those is now asserted, but assume there is a next one.

---

## 1. Number hygiene (unchanged and still binding)

**Every figure in `Limitation 12`, the abstract, GOAL's rows and HANDOFF's parenthetical is GENERATED** by
`scripts/refresh_aw116.py` → `scripts/sync_paper_aw116.py`, and `check_record.py` re-renders them demanding byte
equality. Do not hand-edit those. If a number needs to change, refresh the evidence file first.

**Dated program-log entries in `AGENTS.md` are history.** They quote the counts of their own day. Do not
"correct" them to today's value — that destroys the record. Live counts exist in exactly one place:
`AGENTS.md` §3.5 Gates, which is machine-refreshed.

**Never report an AC-v2 delta without ≥3 seeds and an interval** (seed sd was 2.12 pts, which withdrew a whole
day of grounding deltas on 2026-11-07).

---

**2026-11-07 — read this before quoting any number.** An internal audit re-derived every published figure
from disk. The accuracy half did not survive: **64.32% (256/398) exists in no artifact** — restated to
**64.07% (255/398)** in-process, 63.82% served; the **56.50% grounding is the step-705 checkpoint** (938 pure
is 56.05%); candidate coverage is **96.9%**, not 100%; the online invalid-action intervals **overlap**
(Fisher p = 0.065) and the two conditions ran **different task instances**; the 25% "environment crash" was a
task-checker exception on a **live** device, now fixed. The decision-level claim is a **cost window**
(paper §5.12): the gate beats both always-act and always-ask for `C_err/C_ask` in **2.11–5.79** on the fitted
checkpoint and **2.08–4.19** on the next. Delivery: `outputs/calibrated-decision-value.md` (+ provenance).

**2026-11-08 — the arithmetic that governs everything else.** A multi-step task succeeds only if every
step does: `P(task) = p^n`. At our measured **64.07% per-step**, that is **2.8% over 8 steps and 1.2%
over 10**. Inverting the published end-to-end AndroidWorld scores gives the per-step accuracy the
frontier must have: **93.1%** (MAI-UI-2B, 49.1% e2e), **96.6%** (MAI-UI-8B, 70.7%), **97.4%**
(MAI-UI-235B, 76.7%), **88.8%** (M3A, 30.6%). **The 29-point per-step gap between us and the frontier
is not a bug, a read slot, or a missing dataset — it is a different training regime** (online RL on
512 parallel emulators, multi-step trajectory data, 12-action space). Full derivation:
`notes/2026-11-08-compound-accuracy-reframe.md`. Consequence: the `app_intent` fix is a real repair of a
real defect (serving probe 1/8 → 8/8) but it moves the task ceiling from ~3% to ~4%, and reporting it as
the headline would be dishonest. The first full 116-task run is **finished but incomplete, and reported as such**
(104/116 finished at time of writing: 15 successes; SR **18.8% (15/80)** over scorable episodes (Wilson 11.7–28.7%), 14.4% (15/104) counting all; 34 non-terminating timeouts, 31 premature `status`, **12 unscored (`checker_unavailable`)**, 12 setup raise; **0 invalid of 1,394 emitted actions**). Failure taxonomy and the `gate_score`-drift
observation: `notes/2026-11-08-aw116-first-results.md`; evidence and recomputation script:
`benchmarks/results/2026-11-08-aw116/`.


---

## 0. TL;DR (one paragraph)

We are building an **open decision model** (a "System One" model: reads a state once, returns a calibrated typed decision — choice / score / yes-no — in one forward pass, **zero output tokens**). The **goal is a paper**: *"Calibrated to Be Trusted: Training Single-Pass Decision Models for Risk-Controlled On-Device Action Selection"*.

The d1-2B fine-tune on **AndroidControl (offline, AC-v2, 398-step protocol)** is **COMPLETE**: our 1.92B SFT reaches **64.82%** Pass@1 (groundB) / **64.07%** (pure protocol, 255/398; served path 63.82%) — **on par** with rank-A **GUI-Libra-8B 64.3**, but **NOT** a SOTA break (Mem-W-8B 68.59, ~4.5 pts ahead). The gap is **grounding** (56.05% pure / 60.09% groundB vs **Mem-W-8B 71.30**; 56.50 is the step-705 checkpoint, the 938 pure run is 56.05), while our **Type accuracy 84.17% is the highest reported on the board**. See §2 for the corrected component table.

**STRATEGIC PIVOT (2026-11-02, confirmed by literature):** AC-v2 offline is **plateaued**. Its box-scoring has a **grading artifact** — box-rejected clicks that still reach the intended outcome are real, but the published **1/5–1/3** figure is measured on **web** widgets; on **native Android** (the AC-v2 setting) it is **~1%**, so the artifact is *small here and is not the main justification* (see §3.1). The pivot rests instead on (i) no coordinate headroom, (ii) calibration-under-ambiguity being the differentiator, (iii) outcome-based scoring being the honest metric — and online ambiguity is where calibration is actually worth something. **The value direction = the online AndroidWorld/AndroidLab loop**, with a self-written harness scored by the benchmarks' own env/tasks/checkers.

**Machine decision made (2026-11-02):** run the AndroidWorld **emulator on `GPU_HOST_A`** (RTX 4090, **has KVM** → native emulator speed) in an **isolated, resource-capped Docker container**. Model inference (d1-2B) on **[cloud-gpu] RTX-4090** (~$2.80 credit, ~$0.41/h). The busy **5090** is left alone; **GPU_HOST_B has NO KVM** (skip for emulator).

**Holding all [cloud-gpu] credit ($2.80) unspent** until the direction is confirmed and the emulator is proven. The grounding-fix goal (`mvz1om18-1r2d8c`) is **PAUSED** — it was invalidated by the new measurement (see §3.3).

**2026-11-05 — harness core built and tested; two baseline numbers corrected.** The AndroidWorld harness now has a verified core in `benchmarks/aw/awenv/`: layered liveness probe, no-silent-fallback action adapter, denominator-carrying metrics, JSONL+SQLite trajectory recorder, paired-bootstrap reporting — **110 tests pass, 35 skipped** as of 2026-11-07 (the skips are AndroidWorld-only and never faked). Approved execution plan: `~/.commandcode/plans/androidworld-benchmark-harness.md`.

Three decisions/findings from that build:
1. **The environment runs in-process, only the model is remote.** The HTTP server path is abandoned for benchmarking: `/screenshot` is pixels-only and `ui_elements` is never populated, so `app_name`/`type_text` could never work and a goal-keyword fallback had to be used. In-process `EnvironmentInteractingAgent` gets `state.ui_elements`.
2. **The checkpoint emits 8 of AndroidWorld's 13 action types** → the harness must report `SR_all` *and* `SR_reachable` + coverage, or the first number reads as a grounding failure when it is partly "never trained to press Enter".
3. **Two baseline numbers corrected** (paper commit `19bd001`): GUI-Libra publishes **no** AC-v2 Type/Grounding breakdown — the "76.2" we had been using is its `ASFT-3B` number on **ScreenSpot-v2**. The real AC-v2 grounding comparison is **Mem-W-8B 71.30 / Mem-W-4B 70.85** vs ours 56.05 (pure @938). And our **Type 84.17 is the highest reported on the board** (Mem-W-8B 81.16).

---

## 1. The goal & purpose (why we do this)

### The thesis
LLMs generate **text**. A **decision model** instead takes unstructured state (text / JSON / images) + typed questions and returns a **calibrated probability vector over a fixed answer set** in one forward pass, with **zero output tokens**. Consequences:
- No parsing, no hallucinated strings, no type errors (schema-guaranteed).
- **Calibrated** confidence (probability actually tracks accuracy).
- **Single-pass, fast** (tens of ms on edge) — no autoregressive decode.
- Software can **branch directly** on it (classify / route / score / guardrail).

### Two anchors
- **Jev** (TypeSafe AI, proprietary) = the frontier **target**: RLCD-trained, 70–500ms, ~200× faster / ~400× cheaper than LLMs on System One tasks, "can't hallucinate".
- **d1 / d1-2B** (Liquid AI, open weights) = the **blueprint** we reproduce and extend. d1-2B = 1.92B decision model, single forward pass, zero tokens, has `box`/`boxes` grounding primitives, beats d1-3B (Decision Index 51.1 vs 49.0).

### The three scientific questions (the paper's contribution)
- **Q1 (breakdown):** when and why do single-pass decision models **miscalibrate** (hard/OOD regime)?
- **Q2 (mechanism):** which training objective (plain CE vs proper-scoring vs joint) **preserves reliability** in the decision readout?
- **Q3 (deployment):** can a **risk-controlled act/ask/refuse gate** run **on-device** (Apple silicon, no GPU) at millisecond latency?

### The hard constraint (user, repeated many times)
**The benchmark number is the point. Calibration is the mechanism.** The goal is a **HARNESS** that makes a mobile agent **better on one specific public benchmark**. The paper headline = the benchmark result; calibration = the mechanism.
- **North star (GOAL.md):** build a HARNESS (calibrated single-pass decision head + act/ask/refuse gate) that makes a mobile agent better on **AndroidControl**. **The benchmark number is the point.**
- **User's hard rule (2026-11-02):** must "beat something published at a **CCF-A (rank-A) conference**" — preprints/tech-reports don't count.

### Competitiveness / our honest edge
⚠️ **2026-11-02 correction (Feynman gap analysis, `outputs/gap-analysis-calibrated-single-pass-2026.md`):** the framing **"single-pass zero-token decision model for mobile GUI"** is **taken** — `Jev-Mobile` (`2609.30186`, Sep 2026) is a typed decision model executing mobile GUI actions (79% AndroidWorld, 94.3 s → 5.16 s), and `GUI-JEPA` (`2610.36877`, Oct 2026) calls itself "the first decoder-free GUI agent policy" (0.50B, one batched pass). **Do not claim the category.** The surviving core is the **calibrated act/ask/refuse gate**, which neither paper has.

We do **not** claim raw-accuracy SOTA (8B + memory + RL own that axis), and we do **not** claim "first calibration benchmark" (TypedBench Oct 2026 + AgentAbstain Jul 2026 took that). Our edge is:
1. **Completion-class data recipe** — an entire label class (task completion/`status`) that no public dataset labels. Synthesized by trajectory position.
2. **Two-stage act/ask/refuse** as a first-class decision — calibrated `p_done` gate + natural refusal signals (box-not-found).
3. **Q2 objective comparison** (CE vs proper-scoring) on a decision-model architecture — Liquid's RLCD is named but unpublished.
4. **Weight-merge calibration knob** (backbone averaging) — a cheap ECE-vs-accuracy trade-off curve, unpublished on GUI.
5. **C3 reproducibility** — one consumer-GPU command (vs MobileRL's 4× GPU / 1TB-RAM RL infra).
   Borrowed (attributed): Magma-R1 quality>quantity + stratified sampling; GUI-Libra α token weighting; UI-Venus context augmentation; Liquid option-shuffling + shortcut removal + weight-merge; Mem-W frozen-base + small trainable module.

### NOT the goal
- Raw-accuracy SOTA on GUI (we lose that axis by design; decider-2b 0.80, d1-3B, Jev own it).
- **CoT/reasoning** — REJECTED (against the single-pass thesis).
- **Online RL (MobileRL-style)** — REJECTED as the primary path (risk to calibration; no cluster; needs hundreds of AVDs).
- Reproducing Liquid's decision post-training from scratch (we inherit d1-2B).

---

## 2. The result we have (offline AC-v2) — DONE, keep as the "offline" chapter

**Target frame:** AndroidControl **offline, AC-v2, 398-step, High protocol, goal-only, Pass@1 greedy** (single-step next-action prediction on real Android screenshots).

Official SOTA table (May 2026 leaderboard, for context):
| Rank | System | Backbone | Pass@1 step acc |
|---|---|---|---|
| 1 | Mem-W-8B | UI-Venus-1.5-8B + latent memory | **68.59** |
| 2 | GUI-Libra-8B | (own) | 64.3 |
| 3 | Mem-W-4B | Qwen3-VL-4B + latent memory | **63.07** |
| 4 | GPT-5 + UGround | proprietary | 61.3 |
| 5 | UI-Venus-1.5-8B (no memory) | UI-Venus-1.5-8B | 61.06 |

### The fine-tuned d1-2B result (trustworthy, scorer differential-validated)
Official `eval_rl.py`, AC-v2 398-step, High, goal-only, Pass@1 greedy:

| Run | Pass@1 | Type | Grounding | Notes |
|---|---|---|---|---|
| **E2 pure protocol** | **64.07%** (255/398) | 84.17% | 56.05% (125/223) | CI [59.2, 68.6]; val 62.76%; served 63.82% |
| **groundB** | **64.82%** (258/398) | 82.16% | **60.09%** (134/223) | +858px, hit-loss, aux-grounding |
| E2 + app-name hint | 64.82% (258/398) | 84.17% | 57.40% (128/223) | uses extra input, NOT pure protocol |
| Trivial baseline always-`status` | 21.86% | — | — | uniform random ~14.3% |

**Type breakdown (E2):** click 121/212, status 76/87, open_app 20/27, navigate_back 13/20, scroll 18/37, type_text 7/11, wait 3/4.

**Reading:** single-pass 2B matches **GUI-Libra-8B 64.3** (rank-A) but does **NOT** beat it. Mem-W-8B 68.59 is ~4.5 pts ahead. The gap is **grounding**, and the component split is the informative part.

⚠️ **Correction (2026-11-05, paper commit `19bd001`):** the earlier "gap vs GUI-Libra 76.2%" was **wrong**. GUI-Libra publishes **no** AC-v2 Type/Grounding breakdown at all (its Tables 3–4 are Pass@1/Pass@4 only); **76.2 is `ASFT-3B` grounding measured on ScreenSpot-v2**, a different benchmark and a pre-RL checkpoint. The comparison is withdrawn.

The AC-v2-High component table, recovered from **Mem-W Table 2** (the only public source for these splits):

| Model | Pass@1 | Type | Grounding |
|---|---|---|---|
| Mem-W-8B | 68.59 | 81.16 | 71.30 |
| Mem-W-4B | 63.07 | 74.62 | 70.85 |
| UI-Venus-1.5-8B | 61.06 | 74.62 | 61.06 † |
| Qwen3-VL-8B (base) | 54.80 | 69.35 | 67.71 |
| **d1-2B (ours, pure)** | **64.07** | **84.17** | **56.05** |

† suspicious: Grounding == Pass@1 exactly; likely a transcription error in the source table — do not cite it.

**So: our Type accuracy is the highest on the board; grounding is the entire deficit.** Mem-W does not state its grounding denominator, so cross-paper grounding is indicative, not equivalent (ours = 223 gold-spatial steps, and additionally requires the action type to be correct).

### Grounding error decomposition (measured, E2, 68 wrong-grounding rows)
- 68 rows = 65 `click` + 3 `type_text`, **0 `open_app`** (app-name hint irrelevant).
- **How often:** of the 223 gold-spatial steps, 126 are grounded and **97 are missed = 43.5%** (nearly half of every step that needed a location). Those 97 split into these 68 (verb right, location wrong) + 29 where the verb was wrong too.
- **How far:** on a **50 px button**, only **~3 of 68 misses (4.4%)** land within 50 px — "just outside the edge" is rare. **59 of 68 (86.8%) miss by >150 px** (3+ button-widths); **median miss 437 px ≈ 9 button-widths**.
- Predicted-minus-gold point distance: median **437px**, mean 663px; **86.8% >150px** (far / wrong element); only 4.4% ≤50px.
- Median **y-bias 0.5px** (no systematic spatial bias). Mean conf wrong-ground 0.864 vs right-ground 0.919.
- **→ the failure is element selection + attention dispersion, NOT click precision or bias.** bbox ≠ click point; click point = center of the `box` slot.
- **Independent confirmation (a11y candidate interface, zero-shot):** with goal + candidate list the candidate coverage is **96.9%** (216/223 spatial rows), yet zero-shot grounding is only **50.0%** (n=204) — so candidate generation is not the bottleneck, *selection* is. (Zero-shot checkpoint, so 28.39% step acc there is not comparable to the 64.07% visual headline.)

---

## 3. THE PIVOT (read this — it supersedes the grounding fix)

### 3.1 Why AC-v2 offline is plateaued (the grading artifact) — new, decisive
Literature (via Feynman path-check, free) established:
- **`2610.36831` "Right Click, Wrong Box: Grading GUI Grounding by What a Click Does"** — *"The box is not wrong. The rule is incomplete."* **1/5–1/3 of box-rejected clicks actually reach the intended outcome.** ⚠️ **SCOPE CORRECTION (Feynman deep-read, 2026-11-02):** those 1/5–1/3 figures are measured on **web** widgets (Gitea/Oscar); on **native Android widgets** the out-of-box equivalence is **~1%**. So on AC-v2 the grading artifact is real but **much smaller than the web number implies**. **The pivot is still justified** — by (i) no coordinate headroom, (ii) calibration-under-ambiguity being the differentiator, (iii) outcome-based scoring being the honest metric — but **not** by a large artifact. Do not cite 1/5–1/3 for mobile.
- **`2603.22879` "Confidence Calibration under Ambiguous Ground Truth"** — standard calibration **breaks** under ambiguous GT.
- **`2609.14400` "Policy Ambiguity Masquerades as Agent Error"**; **`2610.34384` AndroidFlux** (failure *recovery*); **`2608.24099` AnTrap** (runtime anomalies); **`2608.07775` AndroidReality** (real-vs-benchmark gap).

**Consequence:** AC-v2's box-grading is an artifact — our ~40% "wrong" clicks were largely **outcome-equivalent**. There is **no headroom** in further coordinate-precision training. Also, AC-v2 **hides** the policy ambiguity that calibration is most valuable for; the **online loop exposes it**. → **Move to the online AndroidWorld/AndroidLab loop.**

### 3.2 Our correctness claim, restated
- **"Outcome-equivalent click"** correctness (per `2610.36831`) — grade by what the click *does*, not by which box it's in.
- **Calibration under ambiguity** (per `2603.22879`) — our calibrated axis is MORE valuable on the online loop than on the offline benchmark.

### 3.3 The grounding V2P-σ plan is INVALIDATED (do NOT spend GPU on it)
The paused goal intended: close grounding with training-time **V2P element-size Gaussian + ConClick logit-level contrastive**. This was **invalidated** by new measurement:
- **d1-2B clicks precisely** (median 0.13 from center) → the failure is **NOT** click precision.
- The failure is (a) **wrong-element / which-button selection** and (b) **~40% of 'errors' are valid-alternative choices** the AC-v2 scorer penalizes.
- **V2P suppression-attention is NOT implementable** on d1-2B (flash-attn — no attention weights exposed); only V2P element-size Gaussian + ConClick logit-level were feasible — **now moot**.

### 3.4 Current phase: the AndroidWorld harness (built 2026-11-05, crash claim withdrawn 2026-11-07)

Full plan: `~/.commandcode/plans/androidworld-benchmark-harness.md`. **Three tiers** — (A) environment correctness,
(B) benchmark correctness, (C) agent evaluation — scored with the benchmark's **own** env / tasks / checkers.

**Decision that shapes everything: the environment runs in-process; only the model is remote.**
`docs/compute/AW_DOCKER_RUNBOOK.md` §4 records that `GET /screenshot` is **pixels only** and
`StateResponse.ui_elements` is **never populated over HTTP**, while an in-process `EnvironmentInteractingAgent`
does receive `state.ui_elements`. The old HTTP path therefore made `app_name` and `type_text` impossible (their
read slots need a11y text), which is why `aw_smoke.py` carried a goal-keyword fallback; it also shipped every
frame as ~7.8M JSON ints. Both problems disappear in-process.

```
[GPU host]      d1-2B API  /v1/decide
                     ▲ HTTP
[Emu host]      AndroidWorld env + D1Agent(EnvironmentInteractingAgent)   ← in-process
                runner + recorder + metrics
                     ▼ ADB
                [emulator container]
```

**Action-space gap (must be reported).** The checkpoint emits **8** of AndroidWorld's **13** action types;
`double_tap / input_text / keyboard_enter / navigate_home / swipe / answer / infeasible` cannot be expressed. The
harness therefore reports `SR_all` **and** `SR_reachable` + coverage — otherwise the first number reads as a
grounding failure when part of it is "the model was never trained to press Enter".

**Built, tested and run on a live device** (110 passed / 35 skipped when this list was written; the suite has since grown and includes the 2026-11-08 regression fixes — **the current count is in `AGENTS.md` §3.5 Gates**, which is machine-refreshed on every `check_record.py` run):
layered liveness probe (`liveness.py`), no-silent-fallback action adapter (`adapter.py`), attributable-failure
agent with loop detection (`agent.py`), resumable/paired runner where timeout ≠ success and crash ≠ success
(`runner.py`), `wait`/`random` baselines through the real adapter path (`baselines.py`), denominator-carrying
metrics (ASR ≠ SR, `metrics.py`), JSONL+SQLite recorder with `.abandoned` recovery (`recorder.py`), paired
bootstrap reporting (`report.py`), a CLI with `--dry-run` (`run_bench.py`), and a nightly gate that refuses to
run when the probe fails (`run_nightly.sh`). Gates A and B passed on the emulator (32 passed / 1 skipped;
5/5 episodes identical across two runs). The 35 skips are AndroidWorld-only tests and are never faked.

**2026-11-07 — the "25% crash rate" is withdrawn.** Both were Clock-task `is_successful()` exceptions on a
**live** device (last step `error: null`, `action_ok: true`; the next episode in the suite ran ten steps
immediately after; stderr `Could not get a11y tree`). `runner.py` now records `checker_error` separately and
requires a failed liveness probe before `env_crash`; two regression tests were added. Note:
`notes/2026-11-07-clock-checker-not-crash.md`.

**2026-11-07 — stage-2 arm run and reported (negative).** Pre-registered before the run
(`docs/research/PRE-REG-2026-11-07-status-downweight.md`): retrain from base with E2's schedule, `status`
down-sampled 22.0% → 5.3%. **Val 62.76% → 63.85% (+1.09, McNemar p=0.44); test 64.07% → 60.80% (−3.27,
p=0.15)** — fails the +1.5 bar and the splits disagree in sign. Mechanism: **−27 premature terminations bought
−26 correct ones** (51→24 and 120→94), because a class weight is one knob on `p(terminate)`. Grounding improved
without being trained (+2.67 val / +2.26 test) since fewer early exits mean more acting steps. A *perfect*
progress oracle is worth only +1.26 here, so the bucket cannot be won by re-weighting. Result:
`benchmarks/results/2026-11-07-status-downweight-arm.md`; paper §5.10. The executed recipe deviated from the
frozen `micro 2/accum 8` to `micro 1/accum 16` (only 13.9 GB free) — recorded as a two-variable change.

**2026-11-07 — the cost window (the decision-level contribution).** At 40.7% test coverage and 17.3% empirical
risk, the gate beats both always-act and always-ask for `C_err/C_ask` in **2.11–5.79** on the fitted checkpoint
and **2.08–4.19** on the next; with a 5% fallible human, 2.36–8.14. Assumption stated: an ask is answered at a
fixed cost; no GUI paper prices the ask. Paper §5.12, `benchmarks/results/2026-11-07-cost-window.md`.

**2026-11-07 — the value brief.** `outputs/calibrated-decision-value.md` (+ `.provenance.md`): four evidence
strands, 36 sources, reviewer pass with six MAJOR findings fixed. Verdict: the act/ask/refuse gate is already
platform policy at Apple and Google, but no first-party document states a numeric confidence threshold, a
measured ask rate, or an outcome result; and no GUI-agent outcome-level abstention result was found in the
sources checked.

---

## 4. THE MACHINE DECISION (2026-11-02, freshly measured)

### Which box hosts the emulator — KVM is the gate (emulator = CPU, NOT GPU)
AndroidWorld emulator uses `-gpu off` + hardware acceleration. **KVM is required for native speed.**

| Host | GPU | KVM? | Free disk | Load | Verdict |
|---|---|---|---|---|---|
| **GPU_HOST_A** | RTX 4090 (24GB) | ✅ `/dev/kvm`, 32× vmx | 137GB | 0.18 | 🏆 **USE for emulator** |
| **GPU_HOST_C** | RTX 5090 (32GB) | ✅ `/dev/kvm`, 32× vmx | — | 0.00 but **live prod** | leave alone (busy) |
| **GPU_HOST_B** | RTX 4090 (24GB) | ❌ **no /dev/kvm, 0× vmx** | 72GB | 0.07 | ❌ skip for emulator |

Both `GPU_HOST_A` and `GPU_HOST_B` are **RTX 4090 boxes**, but **only `GPU_HOST_A` has KVM**. Emulator goes there.

### Resource split
- **Emulator** = `GPU_HOST_A` CPU + KVM (its RTX 4090 GPU is **not** needed for the emulator) + RAM (62GB total, ~51GB available). AndroidWorld is "lightweight footprint (2GB memory, 8GB disk)" for the Python harness; emulator ~2GB each; **32GB RAM ≈ 3 parallel emulators**.
- **Model inference (d1-2B)** = either **[cloud-gpu] RTX-4090** (~$2.80 credit, $0.41/h, ~108ms/step) called over the network, OR `GPU_HOST_A`'s GPU (~21GB free) inside the same isolated container.
- **Caps:** run the container isolated with `--cpus` / `--memory` so prod services are not disturbed.

### Server environment (verified 2026-11-02)
- `GPU_HOST_A`: python 3.10/3.11 (conda `imagelearner`), **docker 28.2.2 + compose 2.37.1**, KVM, 137GB free disk. **No `adb`/`emulator`/`/opt/android` yet** — needs the Android emulator + SDK installed (Docker image `android_world` provides it).
- **prod services on `GPU_HOST_A`** (do not disturb): postgres ×3, grafana, prometheus, loki, ocr-tunnel, cf-tunnel, [service], celery `gpu-4090` workers, imagelearner, streamlit :4200.
- **⚠️ LIVE PROD ISSUE:** the container `display-tracking-[service]` on `GPU_HOST_A` is **`Restarting (1)` / crash-looping**. Documented restore one-liner lives in `docs/compute/STOPPED_CONTAINERS.md` (was previously stopped/started manually to free the 4090). **Do NOT touch without user OK.**

### [cloud-gpu] (verified, credit held)
- Credit = **$2.80**. RTX-4090 = **$0.41/h** (cap $0.60, Vietnam). Storage `acv2-a` 6.7GB (has `data858.tar` 2.5GB cached). `armB`/`vkong.yaml`+`run.sh` = the exact groundB template. Full earlier-style run ~$1.97, smoke ~$0.07, calibration ~$0.07 → fits $2.80.
- **No credit spent. Holding $2.80 unspent** until emulator is proven + direction confirmed.

### Kaggle (assessed, not used for the loop)
- Logged in as `trnanh140999` (different from runbook GPU-blocker accounts), 30h GPU quota. `acv2-decision-train` dataset deleted → not runnable. `train_d1.py` is single-GPU (no DDP) → dual-T4 not usable. T4 3–5× slower than 4090. ~9.9GB image upload blocker.

---

## 5. Calibration — ONE DISCREPANCY TO RESOLVE (2026-11-02)

Reproducible from saved eval data:
- p(action) ECE **0.231** (E2@705) / **0.204** (E2@938); AUROC p(action)→correct **0.626** / **0.631**.
- Selective-prediction curve (act iff conf ≥ τ, E2-938): τ=0.50 cov 96.5%/risk 34.6%; τ=0.70 cov 79.9%/risk 31.1%; τ=0.80 cov 67.6%/risk 29.0%; τ=0.90 cov 51.0%/risk 26.1%; τ=0.95 cov 41.0%/risk 24.5%.

⚠️ **NOT independently reproducible:** the earlier draft/results file claimed ECE 0.146 raw → **0.036** isotonic, AUROC 0.723, risk-gate cov 55.0%/36.7% at risk 23.3%/18.5%. Re-computing from the jsonl does **NOT** produce these — p(action)×box-conf ECE is **~0.29–0.36 (WORSE than p(action) alone)**, and isotonic fit-on-val→apply-test **collapses to τ=1.0** (no usable coverage). The 0.146/0.036/55% numbers likely came from the model's *serving* `calibration.json` (fit+applied live in the run), not from the saved jsonl. **Treat as reported-in-run / pending a GPU re-fit, NOT a proven result.** The paper must not assert them as measured.

---

## 6. Data (we have the real thing)

- **Training raw trajectories:** `benchmarks/data/android_control_train.jsonl` (12,226 trajectories; keys `goal` / `step_instructions` / `actions`; **NO terminate labels**).
- **Curated action rows:** `benchmarks/data/curated/android_control_high_bbox.json` — **8,435 rows** (complete=0), stratified by action type.
- **Completion set (NEW, built this program):** `benchmarks/data/curated/android_control_completion.json` — **1,242 rows** (final-state image + goal, label complete=1).
- **All images local:** `benchmarks/data/acv2_shots/images/` — **6.4GB**.
- **Eval set:** `benchmarks/data/acv2/500_steps_filtered.json` (398 rows) + `eval_official.py` (differential reference).

### CRITICAL data model (verified — do not re-derive)
- A trajectory with **T actions** has **T+1 screenshots** `screenshot_0..screenshot_T` (UGround, 0-indexed).
- `screenshot_i` = state where action i is executed (pre-action state).
- **`screenshot_T` = FINAL state after the last action = the completion screen** (completion label source).
- Curated `step_n.png` (1-indexed) → `screenshot_{n-1}.png`; curated covers only pre-action states.
- Eval status screenshot = state AFTER the last UI action.
- The a11y tree bounding box field is **`bbox_pixels`** (x_min/x_max/y_min/y_max), NOT `bbox`.

### Noise / ambiguity (measured on 398 eval, 212 clicks)
- Gold point inside ≥1 a11y node: **100%** (scorer spatial check is valid).
- Inside labeled node: 79.7%; inside clickable node: 63.7%.
- Inside **≥2 overlapping nodes: 74.5%** (tree-nesting ambiguity).

---

## 7. How d1-2B is read out (verified from model code — no guesswork)

- d1-2B's "decision head" is a **standard LFM3-VL-2B** backbone + a **prompt format** (state + media markup + question suffix with lettered options) + **reading logits at an "answer slot"** (`slot_logits` → `project` = LM head → softmax over code tokens; return log-probs, never `B×T×vocab`) + a **temperature table** (`calibration.json`).
- From `runner.py`: *"answers are read from the logits at the answer slot. Nothing is decoded, so a schema violation is impossible and `tokens generated per decision` is zero."*
- **Fine-tuning = standard SFT** on the same prompt format (no new architecture; full FT or LoRA). **Q2 is directly implementable** at the slot logits (CE / log-score / Brier / joint).
- **Box grounding** uses a pointer / position-logit mechanism (`pointer.py`, `EDGE_READS`, `d1_box_edges`) — fixed σ=4 over a 1000-grid; **logit-level** (flash-attn, no attention map).
- **Temperature:** after training, refit `calibration.json` (per-type temperatures) on held-out.

---

## 8. Compute principles (from AGENTS.md — hard rules)
1. **NO local model runs** — Mac is control channel only. All inference/training on remote GPU.
2. **Smoke test (`--limit 50`) first**, then full. Never "run all and error."
3. **One short job at a time** — don't lock GPU slots.
4. **Ask "is this number actually needed?"** before a heavy run. Get anchors from a small subset + published numbers as cross-checks.
5. **Write learnings durably** to `AGENTS.md` + `docs/colab/GPU_RUNBOOK.md`.
6. **VERIFICATION GATE (HARD):** compute trivial baselines (majority-class / random) and show them next to any number; differential-test ported scorers before trusting ANY number; don't claim "beat" without CI statistical separation.

---

## 9. Key files

| Path | What |
|---|---|
| `AGENTS.md` | Master map (keep in sync with results). |
| `notes/GOAL.md` | North star (claims C1/C2/C3, decision log). |
| `notes/competitive-2026.md` | 2026 competitive landscape + exact gap we own. |
| `notes/plan-androidworld-harness.md` | earlier AndroidWorld plan (emulator facts, tasks, checkers). ⚠️ its `G0` gate cites **GUI-Owl-1.5-2B 67.9%**, which has **no primary source** — superseded, do not gate on it. |
| `~/.commandcode/plans/androidworld-benchmark-harness.md` | **approved execution plan** — API/data contracts, trajectory schema, gates A–C, nightly regression, provisioning (hard stop). |
| `benchmarks/aw/awenv/` | **harness core, tested**: `liveness.py` layered probe · `adapter.py` no-silent-fallback mapping · `metrics.py` denominator-carrying metrics (ASR≠SR) · `recorder.py` JSONL+SQLite + `.abandoned` recovery · `report.py` paired bootstrap · `model_client.py`. |
| `benchmarks/aw/tests/`, `benchmarks/aw/configs/` | 50 tests (**48 pass** locally, 2 need AndroidWorld). Configs: `main_116`, `robust_40x3`, `nightly_smoke`. |
| `benchmarks/aw/aw_smoke.py` | old HTTP-only smoke — **deprecated for benchmarking** (pixels-only, goal-keyword app fallback). |
| `benchmarks/results/d12bax-acv2-d12bax-2026-10-28.md` | a11y choice-over-K **zero-shot** diagnostic: candidate coverage **1.0000**, grounding **0.5000** — proof that selection, not candidate generation, is the bottleneck. |
| `notes/lfm3-d1-2b.md` | d1-2B / LFM3 deep-read. |
| `notes/grounding-precision-2026.md` | grounding decomposition (element-selection, not precision). |
| `notes/training-analysis-2026-10-28.md`, `notes/training-flow-sota.md`, `notes/sota-memw.md`, `notes/mobilerl-magma-r1-methodology.md`, `notes/ui-venus_gui-libra_training_design.md`, `notes/sota-arch-deep.md` | SOTA mechanism deep-reads. |
| `benchmarks/train/train_d1.py` (+ `armB/train_d1.py` ≡, 0 diff) | the proven fine-tune script. |
| `benchmarks/run_acv2.py`, `run_d12b_acv2.py`, `run_d12b_acv2_ax.py` | AC-v2 runners + fixed differential-validated `score_row`. |
| `benchmarks/calib_metrics.py`, `cascade.py` | calibration / risk-gate harness. |
| `benchmarks/results/` | all result files (date-stamped). |
| `docs/compute/GPU_RUNBOOK.md` | compute runbook. |
| `docs/compute/STOPPED_CONTAINERS.md` | how to restore stop/started prod containers (incl. [service]). |
| `docs/compute/D1_API_SERVER.md` | how to serve d1 via `serving/d1_api.py` + SSH tunnel. |
| `references/ledger.md`, `references/bib.bib` | citation ledger (lossless) + BibTeX (108 entries). |

---

## 10. Next actions (gated — get GO, then execute)

**Gate A — environment correctness (P0).** Layered liveness probe + action adapter on a live device: the probe must never pass on `/health` alone (the 2026-11-02 failure was an OOM-killed emulator whose API kept answering 200); `type_text` types non-ASCII and spaces; `open_app` succeeds for all 24 canonical apps; every emittable action type executes. **No benchmark number is produced before Gate A passes.**

**Gate B — benchmark correctness (P0).** Runner resumability (kill mid-episode → no duplicated or lost steps), pairing key `(task_name, task_idx, seed)` in every episode header, trajectory replay from JSONL. **Both halves are done:** the live half passed on 2026-11-06 (5/5 episodes identical across two runs, params recorded then replayed). See `docs/compute/AW_LIVE_2026-11-08.md` for the 2026-11-08 rebuild.

**Provisioning — DONE 2026-11-08** (image rebuilt, container up, Gate A 32 passed / 0 failed; `docs/compute/AW_LIVE_2026-11-08.md`). Kept for the command record: the container was torn down on 2026-11-02 and re-created on 2026-11-08. Commands are drafted in plan §10: build `android_world:latest` with the three Dockerfile patches (`eclipse-temurin:17-jdk-jammy`; `COPY . /app` + `WORKDIR /app`; install Pillow), then run on **port 5051** (`5000` is prod) with `--device /dev/kvm --security-opt seccomp=unconfined --memory=24g --memory-swap=24g`. Before that, confirm: `/dev/kvm` present, 5051 free, ≥24 GB RAM, and a **pinned** GPU for the model (GPU sharing contaminated latency from 113–123 ms to 189–262 ms).

**Status of the ordered list below (updated 2026-11-08):** items 1–3 are **done and exercised**; item 4 is
**done at suite scale** rather than as a 15–20-task pilot; item 5's first half is **run** (116 tasks × 1 seed,
reported with its completeness line in paper Limitation 12 / Limitation 12) and its second half (40 tasks × 3 seeds)
is **not started**; item 6 is **not wired**. The detail is kept because it is the checklist a new session
should re-verify, not because it is all outstanding.

1. Gate A on the live container: `python -m awenv.liveness --url http://127.0.0.1:5051 --layers all`. **done**
2. `agent.py` + `runner.py` — in-process `D1Agent(EnvironmentInteractingAgent)`, **V track first** (screenshot only, the leaderboard-comparable path). **done**
3. `baselines.py` — `random / wait / m3a / d1_zero_shot / d1_sft` on identical instance tuples, same budgets, same adapter. **done** (the `m3a` policy is a stub, not an LLM baseline)
4. **Pilot 15–20 tasks** → error taxonomy with counts: wrong element / unmappable action / env reject / stuck / premature `status` / timeout. **superseded by the full suite**; the taxonomy now comes from all finished episodes
5. **Main run: 116 tasks × 1 seed**, then **40 tasks × 3 seeds** paired. **first half run; second half not started**
6. Wire `run_nightly.sh` once Gate B holds: 8 fixed tasks (~5 min), alert on SR drop ≥2 tasks, any crash, latency regression >25%, invalid-action rise >5 pp. **not wired**

**Parallel track (GPU host, no contention with the emulator):**
- The pure per-row predictions now exist with all readout slots (`e2-pure-hist.jsonl`, 254/398); **stage 2's
  post-processing branch is closed (negative)** and the training arm is the remaining path — this is the documented stage-2 blocker (`GOAL.md:151`); retrofitting readouts later is what stalled it.
- Then the val-fitted two-head selector (box + a11y element, ~6 h) — the grounding lever that also feeds the harness `VA` track.
- Track 2's output checkpoint is what the harness's first real benchmark should measure.

**Paper:** §5.7 of `papers/single-pass-decision-acv2.md` keeps its "preliminary, not a result" framing until Gate C. Do not report an AndroidWorld number that lacks `SR_all`, `SR_reachable`, coverage, crash rate, timeout rate, invalid-action rate, and the proxy grounding rate.

---

## 11. Open questions / risks

- **Is a CCF-A beat achievable?** Offline AC-v2 says "on par, not beating." The online loop's value is calibration/ambiguity/latency, not raw success — **need to confirm a published rank-A online number we can beat** with our calibrated single-pass head. ⚠️ Do **not** use the often-quoted **GUI-Owl-1.5-2B 67.9%** as a bar: it has **no primary source** (the real figures are 71.6% for the 8B-Thinking variant; the 2B reports OSWorld-Verified 43.5).
- **Action-space ceiling on AndroidWorld.** The checkpoint emits **8 of 13** action types, so part of the first SR is a coverage limit, not a model-quality signal. `SR_reachable` vs `SR_all` must be reported together; closing the gap (`answer`, `infeasible`, `keyboard_enter`) is a model change, not a harness change.
- **Cross-paper AndroidWorld numbers are not comparable** (GUI-Libra-32B 34.8 vs UI-Venus-2-9B 57.8 under nominally the same benchmark), and **no peer-reviewed AW baseline exists**. Report our own controls (`random`, `wait`, `m3a`) and a protocol string (`aw-main-v1`) on every table, with no external AW numbers cited.
- **End-to-end latency is dominated by the environment, not the model.** AndroidWorld's `transition_pause` (default 1.0 s) plus emulator I/O dwarf the 103–108 ms forward pass, so a zero-token win will look small per task. Report model p50/p95 **and** ms/step end-to-end separately; never present the model number as the task-speedup.
- **A silent harness is worse than none.** The nightly regression (8 tasks, ~5 min) is not optional once Gate B holds — the whole point is that a harness which fails quietly produces retractable numbers, and this project has already retracted two.
- **Emulator setup risk:** the `android_world` Docker path is **experimental** ("has only been tested on linux devices with KVM enabled" — we have KVM ✅). Apps install is one-time but may be slow.
- **prod safety:** running a container on `GPU_HOST_A` while it hosts postgres/grafana/celery — must cap resources. The **crash-looping `[service]` container** is an existing issue.
- **Latency claim:** must be **measured** on the same machine (d1-2B ~108ms/step on 4090; use p50/p95) before claiming any advantage.
- **License:** LFM Open License v1.0 is source-available, not OSI — read full terms before releasing fine-tuned weights.
- **Scope risk (scoop):** THX-01 + TypedBench shipped Oct 2026; our moats are the on-device venue (Q3) + breakdown-regime study (Q1). Move on the mechanism experiments first.

---

## 12. Environment / tooling

- Working dir: `~/Workspace/research`. Mac = control channel.
- SSH aliases: `GPU_HOST_C`, `GPU_HOST_A` (4090, KVM — **the emulator host**), `GPU_HOST_B` (4090, NO KVM), `HOST_EMU` (DO NOT TOUCH). Keys in `~/Workspace/keys/keys/`.
- [cloud-gpu]: `armB/vkong.yaml` + `armB/run.sh` (validated template); credit $2.80 held.
- Scorers differential-tested against `eval_official.py` (312/312 & 398/398 → 0 mismatches). VERIFICATION GATE is hard.
- Feynman tools (web_search / literature DB) for free literature verification — the pivot was validated this way.
