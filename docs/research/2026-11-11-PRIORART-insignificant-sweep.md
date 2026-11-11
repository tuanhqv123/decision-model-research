# Prior-art sweep for "insignificant" sources against the 2026-11-10 plan (2026-11-11)

*Why this note exists.* `PLAN-2026-11-10-question-tree.md` F10 states that a standalone decision-model result on
AndroidWorld "was not found in three alphaXiv searches; the public leaderboard has not been checked for one". The
spine chosen at Gate 2 (T-A: "first standalone, official-protocol, matched measurement of a decision model as an
agent") rests on that absence. An absence is only as good as the source class it was searched in. This note searches
the class the repo had not: **low-visibility work** — no-paper results, leaderboard entries, GitHub repos, vendor
grey literature, workshop/thesis/non-English papers, and 2026 preprints with no citation count.

Rules carried from AGENTS.md: M1 (never characterise from an abstract — open the text), M4 (never compare numbers
across harnesses), M7 (the artifact defines the claim). Every number below is labelled VERIFIED (read in the
source's own text this session), UNVERIFIED (snippet/secondary only) or INFERRED (our reasoning).

## 0. Status

| Slice | File | State |
|---|---|---|
| A — leaderboards + grey literature | `outputs/2026-11-11-priorart-A-leaderboards-grey.md` | running |
| B — decoder-free / candidate-ranking policies | `outputs/2026-11-11-priorart-B-decoder-free.md` | **done** |
| C — in-policy termination readouts | `outputs/2026-11-11-priorart-C-termination.md` | running |
| D — calibration / abstention / conformal gates | `outputs/2026-11-11-priorart-D-calibration.md` | running |
| E — app choice, element lists, unseen apps, 2–3B data | `outputs/2026-11-11-priorart-E-app-element.md` | running |

## 1. Claims under test

| # | Claim (as written) | Repo evidence today | What a low-visibility source could do to it |
|---|---|---|---|
| **F10** | No standalone decision-model result exists on AndroidWorld | `outputs/aw-state-review.md` C1: the one 79% (Jev-Mobile) is a hybrid with a Qwen3.8-Max planner (VERIFIED) | One leaderboard entry or repo with a zero-token agent and an SR. Kills the novelty sentence, not the programme |
| **T-A** | At matched family/size/data/harness, one pass keeps generation's short-task competence at lower end-to-end latency and loses it beyond a measurable task length | Q1 control arm not yet approved or run | An already-published matched pair. Kills the axis as new; leaves the live-benchmark half |
| **T-B** | A discrete option set + verified-failure masking removes period-1 repeats and shortens wasted steps | Own measurement: 68.2% first-encounter repeat (CI 55–80), 53.9% no-effect steps (CI 35–66) | The same mechanism published with an effect size. Turns a claim into a replication |
| **T-C** | Offline-valid confidence does not flag no-effect actions online; env-verified labels restore AUROC ≥ 0.70 | Own negative: online AUROC(confidence → effect) 0.59 (CI 0.48–0.69) | Anyone fitting a gate on verified-effect labels. Kills T-C outright |
| **R3-1** | Element-id choice over a11y candidates beats pixel-box regression for a 2B model | R4 matched-weights: +6.0 step acc / +10.2 grounding at 300 steps, collapses >40 candidates | A small-model element-choice paper with the same comparison |
| **R3-2** | App choice as a classification over launchable apps replaces goal span copying | 29–36% of R1's `open_app` names unresolvable (`appresolve.json`) | A published app-grounding head with a rate |
| **R3-3** | A `status` head trained with constructed negatives is the defensible new cell | `2026-11-11-LIT-termination-one-model.md` §0: no in-policy source with an AndroidWorld effect | Any in-policy done head with an AW number |

## 2. Slice B — decoder-free policies (done)

**The two sources that matter.**

1. **GUI-JEPA** (arXiv 2610.36877, Oct 2026) — a 0.50B decoder-free policy that scores a11y candidates on
   candidate-conditioned JEPA latents, one batched forward pass, no token decoded. It runs T-A's *offline* half
   against a Qwen2.5-VL-7B it SFTs itself — same inputs, same judge, same split — and matches contract v3's shape
   (element-id choice, span copy from the goal, `open_app` by string match). It has no termination head (stated as
   future work), no app-choice head, no online benchmark. Its top1–top2 margin flags its own step errors at
   AUROC 0.74 offline (VERIFIED, §2.1 of slice B).
2. **V-Droid** (MobiCom '26 / arXiv 2503.15937v5) — the only decoder-free *decision* with an AndroidWorld
   116-task number: **59.5%** with a GPT-4-written step memory; Table 4: LLM-written memory 59.5 vs rule-based
   effect extraction 46.1 vs bare action history 40.0, the paper naming "repeated actions" as the failure of the
   bare-history arm (VERIFIED, §2.2 of slice B). It is not single-pass: it prefills one prompt per enumerated
   candidate, and it generates text for `type`/`open_app`/`answer` (12.4% of actions).

**Two exclusions that had to be checked, not assumed.** GUI-JEPA is genuinely *single-pass per step*: "a decision is
one batched forward pass whose cost grows with the number of candidates" (§3), "all candidates of a step form one
batch" (§3.3, VERIFIED) — with a frozen cached encoder, unlike V-Droid's ~50 prefills of an 8B model. Its cost still
scales with the candidate cap (theirs 80), so our `cap40_reading` is a latency parameter, not a format detail.
LAMO (ACL 2026 Findings) is **not** decoder-free: LAMO-3B emits `<tool_call>` tokens (§4.4) and is a PyAutoGUI text
policy; its AndroidWorld Table 4 numbers (60.3 with a Gemini-2.5-Pro planner, 77.6 with GPT-5; the Gemini
end-to-end agent alone 31.0) are a *planner+executor hybrid*, not a zero-token decision — the same architecture class
as Jev-Mobile. Task count, seeds and step budget are not stated (VERIFIED).

**Verdicts.** T-A: DUPLICATES offline / CONFIRMS-NOVEL live + length axis. T-B: DUPLICATES the mechanism, on a
different carrier (VLAA-GUI §3.3 Loop Breaker; Absorbed-in-Inertia Table 2, where a "screen did not change" notice
moves repeats by −4.6% to +16.9% and only a full context reset reaches 0–4%). T-C: CONFIRMS-NOVEL.

**Consequence for the plan.** The architecture is not the contribution. Two columns are still empty across the whole
visible decoder-free population: an in-policy `status` head, and a live 116-task evaluation. Record the distinction
`zero decoded tokens ≠ single forward pass` (V-Droid is the former, GUI-JEPA the latter) — it will be asked.

## 3. Insufficient evidence inside our own plan (independent of the sweep)

A second reading of "insignificant" is *"which of our claims is unsupported"*. Read from the two plan files; none of
these is a literature question, and each names what would settle it.

| # | Assertion in the plan | Status | Settled by |
|---|---|---|---|
| I1 | "Low SR is caused by one-pass decision-making" | Untested and confounded with data (PLAN §2, first row) | Q1 control arm (needs operator approval) |
| I2 | "Writing the action's effect into the history helps" | Contested by 2610.30205 (plan's own ledger); V-Droid Table 4 is evidence *for* effect memory in a different architecture (59.5 vs 40.0) | Q4 variant run |
| I3 | "The offline gate transfers to live use" | Not supported; own online AUROC 0.59 (CI 0.48–0.69) vs offline 0.729 — not like-for-like, so not yet a refutation either | Q8, fitted on half the tasks |
| I4 | A difference between two runs is a model difference | Unmeasured; no flip-rate at fixed seed and weights | Q10 (20-task repeat) |
| I5 | Element list recall at ≤40 on **live AndroidWorld** screens | Measured only offline on AndroidControl with the AW extractor (93.8%); C3 itself says live "measured from rollouts" | R3 Task 6/9 |
| I6 | 858 vs 720 image width | Not measured; the one 858 arm (+2.79 grounding) was withdrawn at seed sd 2.1 (AGENTS M6) | ≥3 seeds per side, or drop the ablation |
| I7 | "24 distractors ≈ the size of AndroidWorld's app list" | The count is a design choice taken before `aw_apps.json` exists (Task 3 builds it); equality is assumed | Task 3's own file count |
| I8 | "~1 day per seed-epoch" and rollout yield at ~15% success | Both INFERRED in the R3 plan; yield determines whether Task 6/10 can close at all (a 15% SR needs ~7 rollouts per success) | Task 8 throughput + first 50 rollouts |
| I9 | **Arithmetic, not evidence:** information-retrieval tasks are "25" (CONTRACT §7, DIAGNOSIS §1) but R3 Task 4 reports answer types STRING 16 + NUMBER 8 + DATE 1 + TIME 1 = **26** | Partially resolved here: `benchmarks/aw/hostscripts/aw116_answer_only.txt` lists exactly **25** registry task classes (counted; derived 2026-11-10 from `TaskRegistry()` by subclassing `InformationRetrieval`). So the 25 is registry-verified and the **26 decomposition is the unexplained one**. One task plausibly carries two answer fields, or one branch of the decomposition is not a registry task. Per AGENTS M3 the discrepancy is reported, not closed | Read `information_retrieval/proto/tasks.textproto` inside the container and count answer fields per task |

Nothing in this table changes a decision by itself; I1, I4 and I8 are the three that can invalidate a *result* rather
than a sentence, and each already has an owner in the plan.

## 4. Slice A — leaderboards and grey literature (done)

### 4.1 F10 is now answered, and the absence survives

The official AndroidWorld leaderboard sheet (Google Research, read via CSV export, 2026-11-11) holds **47 submitted
rows: none is a decision model** (VERIFIED). The sheet's own taxonomy explains why — it defines exactly two
categories, **"Model"** = "a relatively simple prompt involving one LLM / VLM call" and **"AI agent"** = "a multi-agent
architecture involving several LLM calls … or an LLM wrapped into an advanced agent with memory, subgoal planning", and
has no slot for a decision head, classifier or non-generative policy. The one decision-model AndroidWorld result that
exists (Jev-Mobile) was never submitted. So "the public leaderboard has not been checked" is closed: checked, zero.

Two caveats that matter for our own tables: every row is **self-reported, `trials = 1`, "no independent verification"**
(five rows claim 100), and the README confirms task `max_steps` was reset in 2024-11 to ≈2× the human average, so an
AndroidWorld SR quoted without its budget revision is not comparable (INFERRED from the repo's own note). Secondary
aggregators add nothing new: Papers with Code 43 entries, all generative (VERIFIED); BenchLM 10; no HuggingFace
leaderboard Space; AwesomeCUA is a catalogue; AndroidLab is a different 36-task benchmark; AGI-0 ("small action model,
2B–20B+") is a multi-call agent per its own leaderboard row; Taqwright's 97.4% is best-of-repeated-runs.

### 4.2 The most dangerous source in the whole sweep: arXiv 2610.13092

*When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows* (Oct 2026) runs 5 agent
families × 2 benchmarks × 3 runs = **4,500 executions**, with a paired task-cluster bootstrap over the 116 AndroidWorld
tasks (VERIFIED, primary text). Quotes:

- "On AndroidWorld, Jev trails T3A, AutoDroid, and FastAgent by **13.22, 2.30, and 18.10 pp**, respectively."
- "On AndroidWorld, T3A and FastAgent lose **12.5 and 15.3 pp even in the shortest-budget stratum**, with larger
  deficits in the longest-budget stratum."
- Figure 2(b) is literally *"Jev–baseline success-rate differences (pp) by AndroidWorld predefined step budget"*.
- Its 95% task-cluster bootstrap intervals for the difference are **entirely negative** for T3A and FastAgent.
- Time: T3A −10.6%, AutoDroid −5.7%, FastAgent **+0.7%** — the latency win is not uniform either.

Consequences, in order of severity:

1. **T-A's length half is already published on this benchmark, as a figure.** Our version is not "does the deficit
   appear" but "is it the same at matched family, size, data and harness, with a zero-planner policy" — S7's baselines
   are frontier LLMs and its Jev arm inherits Jev-Mobile's VLM planner.
2. **The "keeps short-task competence" half is also challenged.** The deficit is present in the *shortest* stratum. If
   that replicates for us, the defensible claim is a **uniform** deficit, not a crossover. Pre-register the split and
   state in advance that a uniform deficit is the reportable outcome.
3. Its diagnosis — failed recovery, premature completion, "decision models trained for the consequences of sequential
   choices" — is T-B/T-C's remedy space, so both must be cited against it rather than presented as new framings.

### 4.3 Two more collisions the arXiv set had not surfaced

- **Liquid AI has already published the architecture, qualitatively.** `blog/d1-decision-model` and `blog/d1-open`
  demonstrate a zero-token, one-forward-pass d1 operating a flight-search **web** page ("picks the next action among
  everything the page allows"). No AndroidWorld, no mobile GUI, no number. So "a decision model can drive a GUI" is
  prior art; only a **matched, official-protocol mobile measurement** is not.
- **VisCritic** (SOTA2 record, 2026, no arXiv id) is a non-generative pre/post-screenshot change detector used as a
  process reward: P = 87.1 / R = 83.4 / F1 = 85.2. It is T-C's *mechanism* (environment-verified change labels) already
  built, though not used to fit an online confidence gate. Cite it as the label source, then measure the gate.

### 4.4 What slice A does not break

T-B: nothing in this slice masks a **verified no-effect** action and re-takes the next-best from the model's own
distribution (closest: Jev-Mobile's candidate invalidation, and S7's diagnosis). T-C: no source measures AUROC of an
environment-verified gate. F10: the absence holds on the leaderboard specifically.

## 5. Slice E — app choice, element lists, unseen apps, data recipes (done)

### 5.1 The closest single-call 3B baseline on AndroidWorld already exists: AppVLM

**AppVLM** (arXiv 2502.06395, Huawei Noah's Ark) — PaliGemma-3B, **one model call per step, no planner**, Set-of-Mark
boxes over the a11y clickables — reports **37.8% on AndroidWorld (82/116 tasks, 3 seeds, 0.91 s/step)** (VERIFIED).
It is not a decision model (it emits the action as text, so it is not zero-token), but it is the matched *scenario*:
same size band, same one-call-per-step, same input class. Its Table 4 is the data ablation our plan schedules for
Task 9, already run with seeds: **AndroidControl only 11.4 → AndroidWorld-app rollouts only 22.4 → both 37.8.**
Consequences: Q9's premise (in-domain app data is the lever) has a published 3B precedent and must be cited as such;
and R3's result will be read against 37.8 at 3B, not against a generative 45–49% figure from a different scaffold.

### 5.2 What is genuinely unowned in this seam

| Question | State | Note |
|---|---|---|
| Published rate for **app-name span-copy failure** | None found. Nearest is app-name *generation* accuracy in AndroidControl (85.6 high-level / 88.0 low-level at ~1B — the best of all argument types, Table 10) | Our 29–36% unresolvable `open_app` on AndroidWorld is a new measurement, not a new claim needing prior art |
| **App choice as a classification over a launchable-app list** | No prior system passes an app list. M3A emits free-text `app_name` and its own source comments "nothing will happen if the app is not installed" (`m3a.py:75-76`) | Unowned. Keep it — it is also the cheapest of the seven R3 changes |
| **List-size vs element-selection accuracy on Android screens** | No curve found (5 near-misses) | Ours to measure; must be reported as a curve, not one point |
| Element as an **index over an a11y list** (the interface) | **Already known equivalent**: AndroidControl §C.2 reports indices and pixel centres score "equally well" | Do not claim the interface; claim the cap/recall behaviour |
| **Candidate recall vs cap** under 40 | No independent number. AndroidControl averages **222.2 elements/screen**; one cited corpus has median **62 candidates/step** | A 40-cap is *more* aggressive than our 93.8% offline figure suggests. Measure recall on live AndroidWorld before trusting it (I5) |
| **Seen-app vs unseen-app success** split at ≤3B on AndroidWorld | Not published for AndroidWorld. AndroidControl's own split is −2.1 to −4.7 pp low-level and −6.8 to −12.3 pp high-level | Our 13-unseen-app fear is probably smaller than the plan assumes; the AndroidControl split is the better prior |

### 5.3 Three additions our element numbers cannot ship without

1. **A BM25 baseline for element selection.** LEX (Findings of EMNLP 2026) recasts every grounded step as same-screen
   candidate ranking and finds **BM25 R@1 = 0.557 beats all five dense encoders** (VERIFIED in slice B's LEX read). A dense
   element head that does not beat lexical matching on the same screens is not evidence of learning.
2. **Label-type stratification** — element selection on icon-only vs text-labelled nodes behaves differently, and the
   a11y tree supplies the label.
3. **An element-text-masked ablation**, because LEX reports masking element text costs **22–38 R@1 points**: if our
   head collapses when labels are hidden, what it learned is reading, not grounding.

## 6. Slice C — termination readouts, and the correction to F10 (done)

### 6.1 F10 is **false as written**, and this is the most consequential finding of the sweep

`PLAN-2026-11-10` F10 says a standalone decision-model result on AndroidWorld was not found. **A single-pass typed
decision model on the official AndroidWorld protocol is published**: *When Faster Decisions Do Not Make Better
Agents* (arXiv 2610.13092) swaps the LLM decision-maker for a typed "System One" model and reports, on 116 tasks ×
3 runs (VERIFIED, Table 5):

| config | AndroidWorld SR | official score | all-3-runs |
|---|---|---|---|
| T3A-B (LLM decisions) | 57.47 ± 4.75 | 57.47 | 53/116 |
| **T3A-J (typed decision model)** | **44.25 ± 6.05** | 44.83 | 34/116 |
| FastAgent-B | 50.00 | 50.14 | 38/116 |
| **FastAgent-J** | **31.90** | 33.05 | 30/116 |

Paired outcomes (Table 6): T3A 9 gains / **55 losses**; FastAgent 27 / **90**. Decision latency 7.24 s → 1.34 s
(−81.5%) and 5.40 s → 1.41 s (−73.9%), but **per-task time 213.00 s → 190.36 s (−10.6%) and 119.68 s → 120.50 s
(+0.7%)** (Table 8) — the decision-level saving does not survive to end-to-end. That is plan §6 rule 6 already
stated as a measurement.

This forces three edits to the plan, none of them optional:
1. F10 → "a decision model has been measured on the official protocol; no one has run a **matched** (same family,
   size, rows, harness) one-pass-vs-generation pair, and none of the published arms is at our size or scaffold."
2. T-A's spine sentence "first standalone, official-protocol, matched measurement" loses "first".
3. §5 Stage 1 item 5 ("leaderboard check for F10") is done: 47 rows, zero decision models (§4.1).

### 6.2 What the length result does to T-A

The same paper, Fig. 2b and its text: T3A and FastAgent lose **12.5 and 15.3 pp even in the shortest-budget
stratum**, with larger deficits in the longest. So the shape is *uniform deficit plus a gradient*, not
"competent when short". T-A's "keeps generation's short-task competence" half is contradicted by the closest
published analogue. Pre-register the split and state in advance that a **uniform** deficit is the reportable outcome;
a crossover would now be the surprise.

### 6.3 T-B: one sub-claim is already measured — and our contract v3 already implements the winning side

SparkVLA (arXiv 2608.16172, Table 3) measures exactly the design decision our `status` head embodies: a **separate**
stop head stops at ≈80% accuracy for *any* independent head, while letting `Stop` compete inside the same candidate
ranking reaches **96.65%**, with SR **42.53 → 47.12** (VERIFIED). Contract v3 puts `status` inside the `action`
choice — correct by construction, and now citable rather than a preference. The GUI-specific claim (verified no-effect
masking by re-ranking the model's own options) remains unowned.

### 6.4 T-C: the claim is open, the technique is not

No GUI source measures "offline-valid confidence fails to flag no-effect actions online". But a learned readout on the
policy's own features predicting failure/transition validity is routine outside our setting, and two of those results
*independently reproduce our premise*:

| source | venue | what it shows | transfer |
|---|---|---|---|
| SAFE | NeurIPS 2025 | failure head on the VLA's own last-layer features: ROC-AUC **82.26 seen / 77.04 unseen**, <1 ms; **token-level uncertainty ≈ chance (44–58)** | yes — our readout is the same object |
| VLAConf | arXiv 2605.29605 | failure AUROC **0.876** vs logistic 0.731 vs prior 0.500 | yes |
| StepReflect | arXiv 2608.05587 | transition-consistency head from **882 transitions, 41.4% negative → 82.16%** on 1,082 held-out AndroidWorld transitions; online M3A 38.89 → 44.44 | no (separate 8B), but it is the best evidence a small negative-labelled head is learnable |
| AgentStop | CAIS '26 | GBDT on logprobs, 0.6–0.7 AUC; **1.7B logprobs show no separation at all** | yes — supports "small-model probabilities are the weak signal" |
| A3RM | Findings ACL 2026 | essential-state judge; needs **positives oversampled ×4** to avoid all-negative collapse | no (separate judge), but a warning for Task 7's unlikelihood term |
| LearnStop | arXiv 2606.30852 | a learned multi-feature stopper is **≤ a scalar** on all hard settings (−0.022 val-selected on one) | negative evidence: do not over-build the stopper |

Two published rates give our premature-done number context but are **not** comparable: S11 (arXiv 2610.29609) audits
1,384 GUI-agent runs and finds 180 dangerous misreports, **78/151 (51.7%) among AndroidWorld-original audit-target
runs** (generative agents, conditional denominator); and S1 reports premature completion as **14/251 = 5.6%** of an
outcome-selected subset spanning both benchmarks. Neither is a non-generative AndroidWorld premature-done rate — that
number still does not exist, which is the one part of the LIT note's absence that survives.

## 7. Slice D — calibration, abstention, conformal gates (done)

### 7.1 The most damaging result for T-C is not in the GUI literature

*Per-Step Conformal Calibration Does Not Transfer to Agent Trajectories* (EMNLP 2026 main, pp. 41197–41215) states,
**proves** and quantifies what T-C was going to present as a finding: at nominal miscoverage ε = 0.10, per-step split
conformal gives trajectory miscoverage **29.5% (T=3) → 83.5% (T=15), mean 50.7%**, with lower bounds derived for three
regimes, and an α-spending wrapper restoring 8.9–9.7% (VERIFIED, its Table 1). It is agentic-NLP, not GUI, but the
sentence "a threshold calibrated on single steps does not retain its guarantee over a trajectory" is now published
and proved.

**T-C must therefore be narrowed to what is actually open:**
1. the *specific* failure mode is idempotent / **no-effect** actions, not generic compounding;
2. we report **discrimination** (AUROC of a gate on a held-out task split), not coverage of a prediction set;
3. the fix under test is **environment-verified labels**, not an α-spending schedule.

Corroboration that the shift is real, from three independent sources: Argus (NeurIPS 2026 D&B) finds cross-tier
selectivity transfer ρ = **+0.08, 95% CI [−0.219, +0.373]** — indistinguishable from none — and conformal click disks
undercover by 2.4/1.6/0.5 pp under interface mismatch (this is our own "the threshold does not transfer across
checkpoints" finding, independently reproduced); *Last Step Matters* (EMNLP 2026 main) reports early/mid-trajectory
uncertainty **AUROC < 0.60**; and Mobile-Aptus reports confidence trained on one benchmark over-firing on a new UI
(AIF 6.42 → 5.18 on AITZ, with "overconfidence" named).

### 7.2 What is already done, and therefore cannot be claimed

| Thing | Prior source | Consequence |
|---|---|---|
| Ask-for-help policy with numbers and a threshold | OS-Kairos (ACL 2025 Findings, act-vs-ask at γ, TSR +24.6–87.3); Mobile-Aptus (IP 70.75→84.83, AIF 1.59→1.06, +17% SR offline, +26% TSR on device, and an explicit SR-vs-intervention trade-off) | "A calibrated act/ask gate exists" is old. Our claim is the **certificate** and the AndroidWorld measurement |
| Certified selective prediction on a GUI task | SafeGround (ICML 2026): AUROC 0.75–0.82, AUARC ≤ 0.717, Clopper–Pearson FDR control, **feasible α only from 0.34** | Selective *grounding* is solved enough to have a floor; our object is the **agent**, and the risk floor echoes ours |
| Confidence from **environment-verified** labels | CEB (hindsight execution feedback), Outcome-Grounded (episode outcome), **VeriGUI** (`SUCCESS`/`NO_CHANGE` head trained on deterministic unchanged-screen pairs) | The label source is not ours. Ours is fitting a **per-action no-effect gate** from it and measuring AUROC |
| Coverage–risk numbers on GUI | PACE (EMNLP 2026), SafeGround, Argus, HyperClick, CEB | Present on *grounding*; **absent on AndroidWorld** as a curve or certificate (see below) |

### 7.3 The two absences that survive, and one number that does exist

- **No coverage–risk curve or selective-risk certificate on AndroidWorld.** Nearest: VeriCUA publishes a
  decision-rate number — **82% decided, 95.1% decided accuracy, FPR 7.7%** (Table 1) — but from a post-hoc completion
  verifier, not the agent's own gate, and with no coverage axis. CORA's risk–coverage sweep is on
  Phone-Harm/MobileRisk; its AndroidWorld table is success rate only.
- **No numeric platform abstention threshold.** Apple HIG v26 says to "set a confidence threshold below which you
  don't offer results" and gives no number; Android's 2026 Computer Control doc has hand-over, no number. The repo's
  value-brief finding stands.
- **No per-action no-effect gate AUROC is published anywhere.** That is the whole of our T-C residue.

## 8. What changes in the plan (decisions, not observations)

Ordered by what would embarrass a reviewer if left as written.

1. **Rewrite F10.** Replace "a standalone decision-model result on AndroidWorld was not found" with: *a decision model
   has been measured on the official protocol (2610.13092: 44.25% vs 57.47% with LLM decisions, 9 gains / 55 losses);
   no one has published a matched (same family, size, rows, harness) one-pass-vs-generation pair; none of the
   published arms is at our size or scaffold.* Also record §5 Stage 1 item 5 as done: 47 leaderboard rows, zero
   decision models.
2. **Strip "first" from the T-A spine** and pre-register the length split with a **uniform deficit** as the expected
   reportable outcome. The closest analogue loses 12.5–15.3 pp *in the shortest stratum*. A crossover is now the
   surprise, not the hypothesis.
3. **Restate T-C** per §7.1: the claim is the no-effect instantiation, AUROC on a held-out task split, with
   environment-verified labels — never "trajectory non-transfer", which is proved elsewhere.
4. **Cite SparkVLA for the `status` design**, and keep the current wiring: `status` inside the `action` choice
   (96.65% stop accuracy) rather than a separate head (≈80% for any independent head), +4.59 SR. It is now a
   citation, not a preference.
5. **Add three requirements to the element track** (slice E): a **BM25 baseline** (0.557 R@1 beats five dense
   encoders), **label-type stratification**, and an **element-text-masked ablation** (22–38 R@1 points). Without them
   the element numbers are not evidence.
6. **Benchmark the R3 result against AppVLM, not against 45–49%.** 37.8% at 3B, one call per step, no planner,
   82/116, 3 seeds, 0.91 s/step; its Table 4 (AC-only 11.4 → AW-rollouts-only 22.4 → both 37.8) is the data ablation
   our Task 9 schedules.
7. **Design constraints for the `status`/unlikelihood work (Task 7):** StepReflect learned a transition head from
   **882 transitions with 41.4% negatives → 82.16%**; A3RM needed positives oversampled **×4** to avoid all-negative
   collapse; a learned multi-feature stopper is **≤ a scalar** on hard settings (LearnStop). Size Task 7 accordingly;
   do not build a large stopper.
8. **Do not build any table mixing leaderboard, Jev-study and our numbers.** The leaderboard is self-reported at
   `trials = 1` with the 2024-11 step-budget revision; the Jev study is text-only with AutoDroid below 20%; ours is a
   different scaffold. Rule 3 applies to our own paper.
9. **I9 stands as a reported discrepancy** (25 registry-verified IR tasks vs a 26-item answer-type decomposition);
   resolve it from `tasks.textproto` before the `answer` head ships, not by assuming.

### The five claims that survive this sweep

| Claim | Why it survives |
|---|---|
| A **matched** one-pass vs generation pair at equal family/size/rows/harness | GUI-JEPA matches offline on AndroidControl only; 2610.13092 swaps decision layers with no matched generative sibling; AppVLM is one-call but generative |
| **Verified no-effect masking by re-ranking the model's own option distribution**, on a GUI benchmark | No GUI source does it; VLAA-GUI forces a mode change, Inertia resets context, V-Droid pre-filters with a rule |
| A per-action **no-effect gate from deterministic changed/no-change labels**, with a **coverage–risk certificate on AndroidWorld** | No curve or certificate on AndroidWorld; verified-label heads exist but never as an agent gate with AUROC |
| **App choice as a classification over the launchable-app list** | No prior system passes an app list; M3A emits free text and its own source says an uninstalled name does nothing |
| **Candidate-cap recall measured on live AndroidWorld** | No published list-size/recall curve on Android screens; the 40-cap is more aggressive than the offline 93.8% suggests |

**And one correction to our own framing that the sweep forces:** "a decision model cannot drive a GUI" is not
available — Liquid AI demoed a zero-token d1 web agent in 2025-10, and a typed decision model has now been measured on
AndroidWorld. What is untested is whether a decision model *at our size, trained on our data, with an element
interface and a verified-effect record* can be competitive there. That is a narrower and more honest spine.
