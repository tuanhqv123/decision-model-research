# A5 — What is the correct way to benchmark this model?

*Answers `docs/research/A5-brief.md`. Evidence files: `docs/research/A5-parts/` (T1, T3, T4, T5 = mechanical
extraction; Q4 = power arithmetic). Scripts: `scripts/power_aw116.py`. Written 2026-11-09.*

---

## VERDICT

**The current protocol is not adequate, and one specific defect is inflating the headline.** AndroidWorld's
official metric scores every non-success as 0 and divides by the planned task count: `agent_successful =
task_successful if done else 0.0`, over 116 tasks (T1, `suite_utils.py::_run_task`). Under that metric our run is
**15/116 = 12.93%**, not the **18.8% (15/80)** the record quotes. The 5.82-point difference is entirely
denominator choice — we exclude the 36 episodes that failed for reasons we could not score. A reviewer will
compute 12.93%. Every occurrence of 18.8% as a headline number must either be restated to 12.93%, or the run must
be completed over all 116 tasks and reported under the official metric.

**The single change that matters most is not that correction, though — it is accepting what the power
arithmetic says.** At 116 tasks the minimum detectable paired effect is **~14 points** (17 at the 80 scorable
tasks we actually analyse). The programme's open claims are +4 and +6. **No feasible run on one 4090 can
certify either.** The accuracy axis is closed by the instrument, not by the model. The one claim this programme
can certify is the calibrated gate — and T4 shows even that is no longer novel (CORA publishes conformal
risk-controlled execute/abstain on mobile GUI). What survives is narrow and real: **nobody prices a human ask.**

**And the comparison bar is higher than the record says.** T2 verified from primary sources that the best
published 2B AndroidWorld result is **UI-Venus-1.5-2B at 55.6%**, not MAI-UI-2B at 45.1%. Against the official
metric our gap is **42.7 points, z=7.67**. The record has been quoting the friendlier of the two.

Four things follow, and they are the deliverable:

1. **Restate the headline to 12.93%** (or finish the run), and label every other denominator as a diagnostic.
2. **Re-quote the bar as 55.6%**, and stop calling 45.1% "the 2B bar".
3. **Stop planning accuracy runs on this benchmark.** They cannot produce a publishable delta.
4. **Make the gate primary, with the ask's price as the contribution** — not the certificate, which is taken.

---

## §1 AndroidWorld's requirements vs our compliance

Source: T1 (`docs/research/A5-parts/T1-androidworld-protocol.md`), arXiv:2405.14573v5 full text + repo
`android_world` @ `55ddaf5ec` (retrieved 2026-11-09).

| Requirement | What AndroidWorld says (verbatim) | Ours | Gap |
|---|---|---|---|
| Task count | "116 programmatic tasks across 20 real-world apps" | 116 planned, **92 finished, 24 never attempted** | **24 tasks unattempted** |
| Official metric | "`agent_successful = task_successful if interaction_results.done else 0.0`" | we divide by *scorable* (80) | **wrong denominator** |
| Seeds | "We set the seed to 30 and the temperature to 0"; `_TASK_RANDOM_SEED` default 30; `_FIXED_SEED = 123` | 1 seed, unstated value | **seed undisclosed** |
| Step budget | "approximately 2x the human average completion time"; code `int(10 * task_complexity)` | flat 25 | **not per-task** |
| Checker | "reward signals primarily by managing application state using adb" + UI validation where state inspection is impractical | `task.is_successful(env)` | matches, **but 12 raised** |
| Runs per task | "we maintain the single-seed results"; harness default `n_task_combinations = 1` | 1 | matches |
| Disclosure rule | **NOT STATED** anywhere in paper or repo docs | — | **no external rule exists** |
| Observation space | pixels + accessibility tree + UI elements (all three) | pixels + a11y tree | matches |
| Families | TC (task completion) / IR (information retrieval) | not split | **no per-family reporting** |
| Averaging | per-task rows + `========= Average =========` | single aggregate | **no per-task table** |

**The two gaps that change a claim:** the denominator (§1 row 2) and the step budget (§1 row 4). The budget
matters because it is the one place our run is *unfair to us* — a flat 25 is below the official `10 × complexity`
for the harder tasks, and 34 of our episodes ran to the cap. Fixing the budget could raise the number without
touching the model.

**One finding that helps us:** AndroidWorld publishes **no disclosure rule** (T1 row 7, checked paper, README and
`docs/tasks_guide.md`). So there is no external checklist we are violating — but there is also no defence for
reporting an undisclosed denominator, because the official metric is unambiguous.

---

## §2 The protocol spread: how big must a gap be before it means anything?

*(The corpus task — T2 — did not deliver a file. Rather than wait, I derived the answer analytically from
AndroidWorld's own published seed experiment, which is stronger than a corpus of reported numbers: it isolates
task-sampling variance directly.)*

**AndroidWorld's own measurement of its own noise**, verbatim (T1, arXiv:2405.14573v5 §5):

> "Across three seeds, we observe significant performance variations: **27.6%, 26.3% and 33.2%** (mean 29.0%)"

Three runs of the *same agent on the same 116 tasks*, differing only in seed. Range **6.9 points**, sample sd
**3.67**. Compare that to the binomial standard error of a single 116-task run at p=0.29: **4.21 points**.

**The measured seed spread is smaller than binomial noise alone** (chi²=1.51, df=2, p=0.47). That is the key
result: at n=116 the task-sampling noise floor is already ~4 points, and it is not reducible by averaging seeds
on the *same* tasks — you need different tasks, or more of them.

**Consequence — the significance threshold between two independent 116-task runs:**

| Baseline level | SE(diff) between two runs | 95% significance needs a gap > |
|---|---|---|
| p=0.19 | 5.2 pts | **10.1 pts** |
| p=0.29 | 6.0 pts | **11.7 pts** |
| p=0.45 | 6.5 pts | **12.8 pts** |
| p=0.49 | 6.6 pts | **12.9 pts** |

**A 5-point disagreement between two papers on this benchmark is 0.43× the significance threshold — i.e. it is
what noise produces.** This directly answers the brief's most important question: *if the protocol has a
5-point tail, a 2-point improvement claim is meaningless.* The tail is not 5 points. It is **10–13 points**.

### The corpus confirms it: the protocol is undisclosed almost everywhere

T2 (`A5-parts/T2-androidworld-number-corpus.md`) built a 26-row corpus across 8 papers. The finding is not that
the numbers disagree — it is that **the fields needed to compare them are missing**:

| Field | Papers that state it | Papers that do not |
|---|---|---|
| Evaluation step budget | **GUI-Libra only** (20 steps). AndroidWorld original states per-task `2× human` | MAI-UI, MobileRL, UI-Venus-1.5, Jev-Mobile (MAI-UI's 15/30/50 are **RL training** budgets, not eval) |
| Seeds / runs | AndroidWorld original (seed 30), MobileRL (3 runs, Fig. 3), GUI-Libra (4 runs) | **MAI-UI, UI-Venus-1.5, Jev-Mobile: not stated** |
| Eval harness named | AndroidWorld original, GUI-Libra ("official Docker environment" + "rule-based completion checker"), Jev-Mobile ("AndroidWorld's own initializers/evaluators") | **MAI-UI, MobileRL, UI-Venus-1.5** |
| Task count | almost all state 116 | **Jev-Mobile states no count**; GUI-Libra states 116 but **evaluates 115** (Task #82 `SimpleSmsReplyMostRecent` cannot initialise) |

**So a 5-point spread is not a scandal — it is the expected output of a benchmark on which 3 of 6 recent papers
report neither a step budget, a seed count, nor the harness they ran.** The brief asked whether the protocol has
a tail that makes a 2-point claim meaningless. It does, and the tail is larger than 5 points, and it cannot be
closed by reading the papers because the papers do not say.

**Two entries cannot serve as AndroidWorld comparators at all:** Mem-W and GUI-JEPA. **Neither paper mentions
AndroidWorld** — 0 occurrences in abstract and full text (T2). Mem-W evaluates MMInA / Mind2Web / GUI-Odyssey /
AndroidControl-v2; GUI-JEPA evaluates AndroidControl and ScreenSpot-Pro. *(Checked: the record does **not** claim
otherwise — Mem-W's 63.07/68.59 appear in AGENTS.md only inside the AC-v2 sentence, which is correct. The point
is narrower: if the competitive framing ever reaches for them as AW evidence, it cannot.)*

### Our own number against the published bar

T2 delivered after this file's first draft and it **corrects the comparison class**. The record has been
comparing us to MAI-UI-2B at 45.1%. That is not the 2B bar.

| 2–4B model | AndroidWorld | Gap vs ours (12.93%) | SE(diff) | z | Source |
|---|---|---|---|---|---|
| **UI-Venus-1.5-2B** | **55.6%** | **42.7 pts** | 5.6 | **7.67** | arXiv 2602.09082 Table 2 |
| MAI-UI-2B-RL | 49.1% | 36.2 pts | 5.6 | 6.47 | arXiv 2512.22047 Table 9 |
| MAI-UI-2B-SFT | 45.1% | 32.2 pts | 5.6 | 5.77 | arXiv 2512.22047 Table 9 |
| GUI-Libra-4B | 42.6% | 29.7 pts | 5.5 | 5.35 | arXiv 2602.22190 Table 5 |
| **GUI-Libra-3B** | **25.2%** | **12.3 pts** | 5.1 | **2.41** | arXiv 2602.22190 Table 5 |

**Every gap is real** (z ≥ 2.4), and the 2B bar is worse for us than the record believed: **55.6%, not 45.1%.**
The record must stop calling 45.1% "the 2B bar" — UI-Venus-1.5-2B is the highest published 2B number and it is
the honest comparison.

**The one bar we are near is GUI-Libra-3B at 25.2%** — and it is not a fair comparison either way: it used 115
tasks and a **20-step** horizon (T2), while we used 116 and 25. Two protocol differences bigger than the 12-point
gap. Do not reach for it as a flattering comparison; note it and move on.

*(Per AGENTS.md M4 this is not a cross-harness ranking. It is one number from our harness against published
numbers, with intervals shown and the protocol differences in §1 named. The correct sentence is "we do not reach
the published 2–4B bar", not "we are 43 points worse than UI-Venus".)*

---

## §3 The protocol we must adopt

A numbered specification a reviewer can check line by line.

1. **Task count: 116.** All of them. A147-task run reported over fewer is not an AndroidWorld number.
2. **Metric: the official one.** `SR = successes / 116`, where any episode that did not return
   `task_successful == True` scores 0. `unscored`, `setup_error`, `timeout`, and `premature` all score **0**.
3. **Report the diagnostic denominators separately, never as the headline.** `successes/attempted`,
   `successes/finished`, and `successes/scorable` may appear in a table labelled "diagnostic", with the
   exclusion reason for each. They may not appear in an abstract.
4. **Seeds: 3, values disclosed.** Report per-seed SR and the mean, with the sd. AndroidWorld's own spread
   (§2) makes a single-seed number uninterpretable.
5. **Step budget: per-task, `10 × task_complexity`**, the harness default. If a flat budget is used, say so and
   say it is a deviation.
6. **Task parameters: log the seed and the generated params per episode**, and store them, so the run is
   replayable. (The harness bug log records that `generate_random_params` takes no seed in this version; params
   must be recorded and replayed, not regenerated.)
7. **Buckets are disjoint and exhaustive.** `success | timeout | premature_status | invalid | unscored |
   setup_error`, summing to 116. An environment failure is not a model outcome (M5).
8. **Pre-registration.** The claim, the metric, the task set, the seed count, and the stopping rule are written
   to `docs/research/PRE-REG-*.md` before the run, including the **withdrawal rule** — what result makes us
   discard the hypothesis. The programme has this machinery (`PRE-REG-2026-11-07-*`); it must be used.

**Pre-registration text to copy:**

> This run evaluates `<checkpoint>` on AndroidWorld-116 under protocol `aw-main-v2`: 116 tasks, official metric
> `successes/116`, 3 seeds with values `<s1,s2,s3>`, per-task step budget `10 × complexity`, buckets disjoint per
> §3.7, parameters recorded per episode. The primary claim is `<claim>` with predicted effect `<delta>`.
> **Withdrawal rule:** if the per-seed spread exceeds `<sd_bar>`, or the paired MDE at the realised scorable
> count exceeds `<delta>`, the result is reported as inconclusive and no claim is made.

---

## §4 Power: the arithmetic that closes the accuracy axis

`scripts/power_aw116.py`, α=0.05 two-sided, power 0.80, baseline 15/80. Both formulas Monte-Carlo verified
(20,000 trials, independent seed): McNemar 0.773 vs analytic 0.80; two-proportion 0.795 vs 0.80.

### Paired (McNemar, same 116 tasks, two checkpoints)

| ψ (discordance) | discordant tasks | MDE |
|---|---|---|
| 0.15 | ~17 | **10.0 pts** |
| 0.20 | ~23 | 11.5 pts |
| 0.30 | ~35 | **14.1 pts** |
| 0.40 | ~46 | 16.3 pts |

**ψ is unmeasured.** Realistic for two checkpoints of the same base is 0.20–0.30 (our arms moved 60.80 ↔ 64.07
per-step). Plan on **11–14 points**.

### Tasks required per claim (ψ=0.30)

| Claim | n needed | Feasible at n=116? |
|---|---|---|
| +2 pts | 5,885 | no |
| +4 pts | 1,470 | no |
| +6 pts | 652 | no |
| +10 pts | 234 | no |
| +15 pts | 103 | **yes** |

At the optimistic ψ=0.15: +2 needs 2,941, +4 needs 734, +6 needs 325. Still no.

### The denominator that actually applies

| Effective n | MDE (ψ=0.30) | MDE (ψ=0.15) |
|---|---|---|
| 116 | 14.1 pts | 10.0 pts |
| 104 finished | 14.9 pts | 10.5 pts |
| 92 attempted | 15.8 pts | 11.2 pts |
| **80 scorable (our headline base)** | **16.9 pts** | **12.0 pts** |

**AndroidLab-138 does not rescue this.** 138 < 652 (T3), it states neither seeds nor runs-per-task, and it has
**no published result for any model ≤4B** — so it cannot supply a comparison class either.

### The plain answer the brief demanded

> **No. At 116 tasks no training run can produce a publishable accuracy delta.** Seeds do not fix it: they reduce
> measurement noise, they do not add tasks, and task-sampling variance is the binding term. The smallest
> certifiable paired effect is ~14 points; our open claims are +4 and +6.

### The gate's own MDE, for comparison

Hanley–McNeil on the actual test set (n₁=140 positives, n₀=258, A=0.729): **SE = 0.0276**, CI **[0.675, 0.783]**,
**MDE = 7.7 AUROC points**. Against that floor:

| Comparison | Δ | vs 7.7-pt MDE |
|---|---|---|
| routed 0.729 vs global 0.649 | **+0.0792** | **1.03×** — barely clear |
| val 0.760 → test 0.729 | −0.031 | within noise |
| learned fusion vs routed (+0.013, CI [−0.020,0.043]) | +0.013 | below floor — correctly "not significant" |

**The honest sentence is "routing improves discrimination by roughly one detectable effect", not "significantly
improves" — it clears by 2%.** And this is the *only* discrimination result the instrument can certify.

---

## §5 The correct primary metric, and whether the novelty survives

### (a) Which convention supports a risk-controlled abstention claim

**Conformal Risk Control** (Angelopoulos et al., arXiv:2208.02814) is the right family, and this programme already
uses it: split-conformal certified **20% selective risk on val at 47.6% coverage**; test operating point 40.7%
coverage at 17.3% risk. The reporting convention is set by that literature: **risk at fixed coverage**,
**coverage at fixed risk**, **AURC**, **the conformal bound and its exchangeability assumption**.

### (b) What must be reported

1. Risk at fixed coverage (and the coverage achieved at the certified δ).
2. AURC alongside AUROC — AUROC alone hides the operating point.
3. **The bound, its δ, and its assumption.** Our recorded negative stands: **the threshold does not transfer
   across checkpoints** (same τ → 17.3% on E2@938 vs 23.9% on @705). Exchangeability, not the score, is binding.
4. The coverage/risk pair on **both** val and test, never test alone.

### (c) Is there a published GUI abstention result? Yes — and it is not ours

**T4 refutes the recorded claim that "no GUI paper has an outcome-level abstention result".** CORA
(arXiv:2604.09155) publishes conformal risk-controlled execute/abstain **on mobile GUI**, verbatim:

> "the CRC module adaptively selects τ̂=0.952 … best coverage (89.95%) and the lowest invocation rate (10.05%)
> … constraining the executed-harm rate to 2.42%—safely satisfying the conformal risk target (≤5%)"

on a held-out calibration pool (n=1075, δ=0.05), with Phone-Harm (HR 4.37 / GAR 79.80 / OIR 19.23) and
**AndroidWorld core20: 8/20 = 40% with the controller vs 6/20 = 30% without.**

**So the gate is prior art. Our claim to novelty must change.** Three sub-findings decide what replaces it:

- CORA reports **task success under a controller**, but does **not** sweep abstention rate against task success
  (T4 Q2). **That sweep is an open gap and it is the honest AC-v2 contribution.**
- CORA prices only **intervention latency** (34.13 s/step) and states its aim as "minimize user burden" — it does
  **not** assign a cost to asking (T4 Q4).
- **No fetched source prices a human ask.** Machine escalation is priced (TypedBench: 41%/46% of billed tokens;
  GUI-JEPA: 0.91×/0.76× latency). The human ask is not.

**Therefore: the contribution is not "we built a gate". It is "we priced the ask".** That matches the 2026-11-07
reframe to *Priced Interruptions* — which is now evidence-backed rather than a judgement call, and it is the
only framing in this file that a reviewer cannot take away with a citation.

---

## §6 Training-ingredient checklist

Source: T5 (`docs/research/A5-parts/T5-ingredient-checklist.md`), all five papers read at full text, 60 rows.
Present/absent for our recipe, with the effect each absence is worth.

| Ingredient | MAI-UI-2B | MobileRL | GUI-Libra | **Ours** | Effect of our absence | Closable on one 4090? |
|---|---|---|---|---|---|---|
| multi-step trajectory data | yes | yes | yes | **no** (single-step, 15k) | Ferret-UI-Lite: first 5K traj **+0.7**, to 17K **+2.8**; +5.6 total on top of a traj baseline | **yes** — GUI-Odyssey usable (I2) |
| interaction history rendering | yes | no | yes | **partial** (17.6% blank) | not isolated in any source | yes |
| action-space size | 11 | 9 | 13 | 8 | `ask_user`/`mcp_call` absent | yes (schema) |
| resolution/aspect handling | half-res | full-res | not stated | 2400×1080 fixed | 33% of GUI-Odyssey is non-portrait (I2) | yes |
| coordinate representation | relative | bbox/XML | 0–999 | 0–999 | — | n/a (matches) |
| dedicated grounding loss | yes | no | partial | **yes** (Gaussian σ=4) | — but flat beyond ~400 px (§5.3) | yes |
| negative/contrastive samples | no | **1:2 ratio** | no | **no** | ConClick: +2–4 predicted (unmeasured) | yes |
| rejection sampling | yes | yes | yes | **no** | MAI-UI relies on it; effect unisolated | yes |
| MLLM judge | **yes** | yes | yes | **no** | MAI-UI pipeline depends on it | **GPU-costly** |
| progress/termination supervision | yes | yes | yes | `status` only | **the measured failure**: 31/104 premature | yes — this is I4's fix |
| per-task-type balancing | no | no | no | no | — | n/a (nobody does it) |
| seeds reported | **no** | **yes (3)** | no | **1** | §4: spreads are uninterpretable at 1 | yes |

**The "concept we missed like the old" is a single line: trajectory data, and it is not the biggest lever.** T5's
own comparison shows the largest single jump in the Ferret-UI-Lite ladder is **long CoT (+3.8)** — a supervision
*format* change — not trajectory volume (+0.7 for the first 5K). And **MAI-UI does not report seeds at all**,
which means the 2B bar we are chasing is itself a single-seed number.

**The one absence that matches our measured failure is termination supervision.** 31 of 104 episodes emitted
`status` at median step 7 on tasks needing 8–15. That is a supervision gap, it is closable on one 4090, and the
2026-11-07 pre-registered arm already proved the naive fix (class re-weighting) fails: −27 premature bought −26
correct. **The next arm must be an explicit `task_complete` readout, not a re-weight (A3 §3).**

---

## What I could not verify

- **ψ (discordance) is unmeasured.** Every paired MDE is conditional on it. The honest answer is the range
  11–14 points. It becomes measurable the moment two checkpoints run the same 116 tasks.
- **Whether the scorable set is identical across checkpoints.** Pairing assumes it. The a11y-forwarder
  degradation could differ per run; only one run exists on disk, so this is untestable.
- **T2 (published-number corpus) did not deliver.** I derived §2 analytically from AndroidWorld's own
  three-seed experiment instead, which isolates the quantity of interest better than a corpus would. A
  cross-paper number table remains unwritten — and per M4 it should stay a *table*, not a ranking.
- **CORA's risk budget α for its Phone-Harm table is not stated** in the fetched text (T4 Q1); only δ=0.05
  appears. Its 89.95%/2.42% pair is on the calibration pool, not Phone-Harm. Do not quote it as a test-set result.
- **No source states how AndroidLab relates to AndroidWorld beyond one sentence** (T3), and AndroidLab has no
  ≤4B published number, so it cannot supply a comparison class.
- **I did not check whether a `task_complexity` per-task budget raises our SR.** That is a re-run, not an
  analysis. It is the cheapest remaining experiment and it is now the first thing to do.

---

## The three actions this file forces

1. **Restate 18.8% → 12.93% under the official metric**, in GOAL.md, HANDOFF, the paper and the abstract, with
   the other denominators demoted to a diagnostic table. *(This is M11 at programme scale: a rate without its
   base, read as if complete.)*
2. **Correct the comparison class: the 2B bar is UI-Venus-1.5-2B at 55.6%, not MAI-UI-2B at 45.1%.** The record
   has been quoting the easier number. Re-run with the per-task step budget (`10 × complexity`) before any
   training change — it is cheap, it is protocol compliance, and 34 of our failures ran into a cap the official
   harness would have set higher.
3. **Reframe the paper to the price of an ask.** The certificate is CORA's. The accuracy axis is closed by §4.
   What remains measurable and unclaimed is the cost side — plus the abstention-vs-success sweep CORA does not do.
