# GOAL — the single consolidated objective of this research program

*Last updated: **2026-11-08** (first full 116-task AndroidWorld run launched; compound-accuracy arithmetic added; `app_intent` fix verified as a read-slot repair, not a capability). Previous: 2026-11-07 internal audit restated the headline. This file is the north
star. If any note, plan, or todo disagrees with it, this file wins until it is deliberately updated.*

> **2026-11-10 DECISION — read first.** The operator made **AndroidWorld (116 tasks, official
> `suite_utils` protocol) the single primary benchmark**. The AC-v2 numbers below remain true as measured,
> but AC-v2 is no longer optimised: its history input is the oracle human step instructions, which
> AndroidWorld cannot supply. Every AndroidWorld number recorded before 2026-11-10 was run on **unseeded,
> non-official task instances** and is not comparable to published scores. Record:
> `notes/decision-2026-11-10-androidworld-primary.md`; evidence:
> `docs/research/2026-11-10-AUDIT-androidworld-contract.md`. Where this banner and the sections below
> disagree, this banner wins until §1 is rewritten from new measurements.

> **What changed and why.** The 2026-10-13 version was written before the d1-2B pivot (base model),
> before the AC-v2 protocol was fixed (benchmark), and before any calibration measurement existed. Three
> of its claims (C1, C2, and the on-device venue) have since been tested and **falsified or found
> unattainable**. This rewrite states only what is measured, marks everything else as open, and records
> the retractions in §8. Nothing measured was deleted.

---

## 1. The goal, in one screen

**Build a small, honest, fast decision model that a mobile agent can be trusted to act behind — and
prove, on one fixed public benchmark, both that it is competitive and that its confidence is usable for a
risk-controlled act / ask / refuse gate.**

Two halves, and both must hold:

1. **Competitive on the benchmark.** AC-v2 398-step High Pass@1. Currently **64.07%** (255/398, pure protocol; served cross-check 63.82%,
   official scorer). This is a *tie* with the rank-A 8B baseline, not a win. The benchmark number is the
   entry ticket; without it the calibration result is not interesting.
2. **Usable confidence.** The model's confidence must rank its own errors well enough to certify a
   selective-risk budget. This is the actual contribution, because nobody has measured it for this model
   class, and because a wrong tap on a phone is irreversible.

**The measured position (2026-11-03):** half 1 is done and tied; half 2 is **done and positive** — the
type-routed confidence reaches AUROC 0.729 held-out (0.709–0.760 across three evaluations) and certifies
a **20% selective-risk budget, certified on the val split at 47.6% coverage** (test: 40.7% coverage at 17.3% empirical risk, CP upper 21.7%). What is *not* done is the 10% budget (§3, C2) and any
accuracy break past the tie.

**No longer the goal:** raw-accuracy SOTA (we lose that axis by design), on-device Apple-silicon latency
(never measured; see §7), and "first single-pass zero-token GUI action selection" (taken by Jev-Mobile
2609.30186 and GUI-JEPA 2610.36877).

---

## 2. The benchmark (fixed — do not drift)

| Item | Value |
|---|---|
| Benchmark | **AndroidControl-v2 (AC-v2)**, the 398-step / 350-episode leaderboard subset |
| Protocol | pure: goal + interaction history + screenshot; greedy; Pass@1; **zero generated tokens** |
| Scorer | official element-level (`eval_official.py` ≡ GUI-Libra `eval_rl.py`), ported and differential-tested 398/398 |
| Spatial credit | predicted point must fall inside the **smallest a11y node containing the gold point** |
| Trivial baselines | always-`status` **21.86%**, always-`click` **53.27%** |
| Our result | **64.07%** (255/398), Type 84.17, Grounding 56.05; CI [59.2, 68.6] |
| Bar to beat | Mem-W-8B **68.59** · GUI-Libra-8B **64.30** (rank-A) · Mem-W-4B 63.07 · UI-Venus-1.5-8B 61.06 |
| Model | **d1-2B** (1.92B, LFM3-VL-2B backbone + 164M ViT + 25M projector), slot readouts, not a text head |
| Compute | one RTX 4090 per run; 938 steps ≈ 1.6 h. See `docs/colab/GPU_RUNBOOK.md` |
| Checkpoints | `backup-2026-10-29/ckpt-E2-final` (64.07 path) and `ckpt-groundB` (64.82 hinted / grounding 60.09), 3.6 GB each |

**Superseded:** the 2026-10-13 text-only "K candidate actions" task on 322 test episodes with LFM2.5.
Different task, different data, different model. Do not mix numbers across the two.

---

## 3. The claims — restated to what is measurable

Each claim is stated with its current **status** and its **evidence**. Nothing is aspirational in this
section; aspirational work lives in §5.

### C1 — MEASURED, but not the claim we originally wrote
> **The confidence a single-pass decision model is normally credited with is mis-specified by
> construction, and the repair is free.**

On the 44% of steps that need no coordinates, the box readout carries no information, so the conventional
`p(action) × box_conf` product is **at chance** there (AUROC 0.504 test / 0.577 val). Routing confidence by
decision type — box confidence on spatial steps, action probability elsewhere — lifts
correctness-ranking AUROC from **0.649 → 0.729** held-out (0.760 on the calibration split, 0.709 on a
second checkpoint), at zero inference cost.

**Status: established.** Reproducible from saved predictions, stable across three evaluations, with a
mechanism that predicts the magnitude. Evidence: `benchmarks/results/2026-11-03-acv2-error-budget-and-routed-gate.md`.
**The original C1 is retracted** — see §8.

### C2 — PARTIALLY MEASURED, and the 10% target is unattainable
> ~~At the operating point "risk ≤ 10% of actions taken", the model covers a large share of decisions at
> ≤100 ms on M1 Pro~~

A 10% budget at ≥50% coverage **does not exist** for this checkpoint: the routed score's risk floor is
~12.5%, and at α=0.10 the threshold is not certifiable at any useful coverage. What *is* certified:
**20% selective risk certified on the val split at 47.6% coverage** (val upper bound 19.9%); on test the operating point is 40.7% coverage at 17.3% empirical risk (Clopper–Pearson upper 21.7%), so the budget is not certified on test.

**Status: restated to the measured frontier.** The on-device latency half of C2 is **unmeasured** and
must not be claimed. The replacement engineering objective is numeric and is the program's current
target for C2:

> **Halve the selective risk at 50% coverage: 21.5% → below 10%.**
> This is a property of the score's *discrimination*, not of the threshold, so it cannot be tuned in.

### C3 — DELIVERED for the analysis layer, open for the training layer
> One command regenerates every number in the paper.

**Status: delivered for every published number.** `benchmarks/train/verify_paper.py` performs 69
assertions against `benchmarks/results/2026-11-03-stage0/` and exits nonzero on drift. It already caught
one stale artifact. **Open:** no one-command path regenerates a *checkpoint* from raw data (data
preprocessing → training → eval). The pure per-row predictions ARE on disk now (`E2-nomem` 255/398 and the served-path
  254/398); what is still missing is a one-command path from raw data to checkpoint.

---

## 4. What is measured vs. what is assumed (the most load-bearing section in this file)

| Thing | Value | State |
|---|---|---|
| AC-v2 step accuracy (pure) | **64.07%** (255/398), CI [59.2, 68.6]; served path 63.82% | measured |
| AC-v2 step accuracy (with train-goal app-name hint) | 64.82% (258/398) | measured, **not** the headline |
| Type accuracy | **84.17%** (beats Mem-W-8B 81.16) | measured |
| Grounding | 56.05% (125/223, pure); 57.40% hinted; **56.50% (126/223) is the step-705 checkpoint**; groundB 60.09% | measured |
| Error budget | 68 element-wrong / 27 spatial type-wrong / 36 non-spatial type-wrong / 9 value-wrong | measured |
| Type vs grounding error count | 63 vs 68 (test), 161 vs 137 (val) — **difference not significant (z≈1.4)** | measured |
| Element misses that are genuinely different elements | **80.6%** (17.9% subtree, 1.5% same-label) | measured — **do not claim metric noise** |
| Type-routed gate AUROC | 0.760 val / 0.729 test@938 / 0.709 test@705 | measured |
| Learned fusion AUROC / risk@50% | 0.767 (OOF 0.758) val / **0.743** test; risk@50% **0.191** | measured but **Δ not significant** (+0.013, [−0.020, +0.043]) |
| CRC vs Clopper–Pearson at δ=20% | **45.5%** vs 38.4% coverage at matched budget | measured — the bound is worth +6–7 pp |
| val→test action-mix shift | TV **0.106**, ESS after weighting 752/827 (91%) | measured; reweighting explains ≲2 pp of a 6.6-pp gap (SE 4.5 pp, ≈1.5σ) |
| Certified operating point | **val**: 47.6% coverage @ 20% risk (val CP upper 19.9%). **test**: 40.7% coverage, 17.3% risk, CP upper 21.7% | measured |
| 10% budget @ ≥50% coverage | **does not exist** | **measured negative** |
| Conformal transfer across checkpoints | **fails** (17.3% → 23.9%) | **measured negative** |
| Per-step latency | ~103–130 ms batched, RTX 4090 | measured (desktop) |
| On-device (Apple silicon) latency | — | **never measured** |
| Task-level success rate (episodes) | **1/8 in both conditions** (pure and declared-fallback), Wilson **2.2–47.1%**; `n_reachable = 0` so `SR_all` and `SR_reachable` cannot be split; timeouts 7/8 and 6/8 | **underpowered, not a result** |
| Online "environment crash" | **withdrawn** — both were Clock-task `is_successful()` exceptions on a **live** device (next episode ran 10 steps immediately after); harness fixed, 2 regression tests (suite was 110 passed / 35 skipped then; now 132) | measured, misclassified by the harness, now corrected |
| Cost window (gate vs trivial policies) | beats always-act and always-ask for `C_err/C_ask` in **2.11–5.79** (fitted checkpoint) and **2.08–4.19** (next checkpoint); 2.36–8.14 at 5% fallible delegation | measured from saved artifacts, assumptions stated (`benchmarks/results/2026-11-07-cost-window.md`) |
| Stage-2 status down-weighting arm | val **62.76 → 63.85 (+1.09, p=0.44)**, test **64.07 → 60.80 (−3.27, p=0.15)** | **fails its pre-registered bar**; −27 premature terminations bought −26 correct ones |
| **`is_done` readout (`episodic-s0`)** | **`auroc_nonterm` is NULL — not computable.** On 824 val rows, `step == n_steps` and `action == status` are the **same 137 rows** (137/137; **0 positives among 687 non-status rows**). The `auroc` 0.960 is that shortcut; `risk @ 50% coverage` = **0.670** | **measured negative — the question does not exist in this corpus.** Retracted 2026-11-10: `benchmarks/results/2026-11-10-episodic-s0-fails-prereg.md` |
| **`episodic-s0` on AC-v2** | **55.78% (222/398) vs 64.07% = −8.29 pts**; `type_acc` 0.7764 vs 0.8417 (−6.53); `status` **64/87 vs 76/87**; **grounding 0.600 (114/218 = 0.5229) vs 0.5605, +3.95** | **fails pre-registered condition 3.** The redundant readout consumed capacity in exactly the class it was built from; grounding improved. **Not a benchmark checkpoint — do not quote a number from it** |
| Duplicated-list training lever (15k vs 22.5k) | +2.62 pts grounding (15k 56.80 ± 1.29 n=3; 22.5k 59.42 ± 0.45 n=4), every 22.5k seed beats every 15k seed | **suggestive, NOT certified** — grounding is a count/223 so each arm has 2 distinct values, variance ratio **8.3×**, Student pooled p=0.012 but assumption violated, **Welch p=0.063**, permutation p=0.029 is the minimum attainable at n=3 vs 4. Pre-registered bar (t≥2.78 at 5 seeds) never reached |
| AndroidWorld full benchmark | **INCOMPLETE at time of writing**: 104/116 finished, **15 successes**; SR **18.8% (15/80)** over **scorable** episodes (the only denominator that answers how good we could judge), 16.3% (15/92) over genuinely attempted, 14.4% (15/104) counting everything; **Wilson (scorable) 11.7–28.7%**; buckets 34 non-terminating timeouts / 31 premature `status` / **12 unscored (`checker_unavailable`)** / 12 setup raise; **0 invalid of 1,394 emitted (0.0%)** | **in progress, not a headline**. Refresh: `scripts/refresh_aw116.py` then `scripts/sync_paper_aw116.py` |
| Per-step accuracy → task success | `P(task) = p^n`. Our 64.07% implies **2.8% over 8 steps, 1.2% over 10**. Published end-to-end scores imply per-step **92.3% (MAI-UI-2B SFT)**, 93.1% (2B +RL), 95.7% (8B SFT), 96.6% (8B +RL), 97.4% (235B +RL), 88.8% (M3A 30.6%) | **measured arithmetic** — `notes/2026-11-08-compound-accuracy-reframe.md`. AC-v2 step accuracy and AndroidWorld task success are not the same axis. |
| **Where the 28-point per-step gap comes from** | NOT online RL: MAI-UI Table 9 gives 2B **SFT 45.1 → RL 49.1** e2e, i.e. **92.3% → 93.1% per-step, +0.8 points**. The gap is **supervision** (multi-step trajectories, judge rejection loop, 50-step horizon, 12 actions) | measured from the primary table. **Our comparison class is 45.1% (2B SFT), not 76.7% (235B RL).** |
| AndroidWorld failure modes observed | **premature `status`** at median 7 steps (31/104); **never terminates**, clicks to the step budget with `gate_score` decaying to 0.0 (34/104 timeouts) — plus **12 unscored** episodes whose checker raised, which are neither a model outcome nor a device failure; **zero invalid actions** in 1,394 emitted (`benchmarks/results/2026-11-08-aw116/`) | measured, n=104 |
| `gate_score` drift vs episode outcome | mean Δ(first3→last3) **-0.202 on failures (73% decaying, n=77)** vs **-0.072 on successes (40%, n=15)** | **hypothesis, not a result** |

**Rule:** any number not in the "measured" column may not appear in the paper, in a claim, or in a
rebuttal. If it is needed, measure it first.

---

## 5. The current numeric objective and the ordered work

**Objective:** raise the score's discrimination until **risk at 50% coverage falls below 10%** — today
**19.1%** with the learned fusion (21.5% with the hand-routed score), and the *certified* frontier is 20%
at 45.5% coverage using CRC. Calibration is the mechanism; the benchmark number is the entry ticket.

**Not achievable by post-processing.** Two zero-GPU attempts are now spent: hand routing (0.649 → 0.729)
and a learned fusion of all eight saved readout features (0.743, Δ not significant). Both land at
~19% risk at 50% coverage. The remaining distance is a **representation/data** problem, not a scoring
problem, so the next measurable lever requires GPU (§5's stages 2–3) and the target should be judged
against 19.1%, not 21.5%.

Ordered, cheapest-first, each with a pre-registered pass condition. Full detail:
`~/.commandcode/plans/rl-decision-model-acv2-2026-11-03.md`.

| # | Work | Cost | Status |
|---|---|---|---|
| 0 | Error budget + paired tests + routed gate + grounding audit | 0 GPU | **DONE** |
| 1 | Regenerate the pure-protocol per-row predictions | ~52 s GPU | **DONE 2026-11-07** — `E2-nomem` 255/398 (in-process) and `e2-pure-hist.jsonl` 254/398 (served), both with all readout slots |
| 2 | Type bucket: synthesized-`status` supervision, premature `status`, `scroll` (≈half the headroom) | ~3 h | **diagnosis DONE (2026-11-04)** — scroll is a TYPE error (16/19) not direction (3/19); premature-`status` is *not* an episode-boundary artifact (0.97× enrichment); per-class logit adjustment passes on val and **fails on test** (p=0.043). **Post-processing CLOSED 2026-11-07**: a val-fitted `p(status)` gate is monotone-harmful at every τ (best τ = never switch), so the 21 premature `status` rows need a learned progress readout, not a threshold (`benchmarks/results/2026-11-07-status-gate.md`). **Arm running 2026-11-07**: retrain from base, E2's 938-step recipe, `status` 22.0% → 5.3%; PASS = paired val Δ ≥ +1.5 with no gold type −1 (`docs/research/PRE-REG-2026-11-07-status-downweight.md`). A **perfect progress oracle is worth only +1.26 pts** here (5 of 21 premature rows are rescued by the model's own next-best action), so the status bucket alone cannot clear the bar. **RESULT (2026-11-07): FAILED.** val 62.76 → 63.85 (+1.09, p=0.44), test 64.07 → 60.80 (−3.27, p=0.15); −27 premature terminations bought −26 correct ones; grounding +2.67 val / +2.26 test without being trained. Next arm = explicit `task_complete` readout, not a class weight (`benchmarks/results/2026-11-07-status-downweight-arm.md`). |
| 3 | Grounding: val-fitted two-head selector (box + a11y element); the *hand* combiner already failed, the learned one is untested | ~6 h | open |
| 4 | RLVR refiner (GRPO + SNGS + KL; group rollouts are free for a slot readout) — expectation **+1–2 pts**, not +5 | ~9 h | speculative |
| 5 | Online AndroidWorld port | ≥1 day | **RUN 2026-11-08, INCOMPLETE** (the operator made it the primary benchmark). 116 tasks × 1 seed on protocol `aw-main-v1`, checkpoint E2+`app_intent`: **104/116 finished**, **15 successes, SR 14.4% (15/104), Wilson 8.9–22.4%**, and **16.3% (15/92)** over genuinely attempted tasks after excluding 12 whose device never became usable and 12 that ran but could not be scored (`checker_unavailable`) — and **0 invalid of 1,394 emitted actions**. Two failure signatures (premature `status` at median 7 steps; non-termination at median 25) that no single confidence threshold separates. **40 tasks × 3 seeds not started.** Detail: §4 row and Limitation 12 |

**Pre-registered pass conditions** (do not move them after seeing results): stage 2 and 3 need a paired
Δ ≥ +1.5 pts on the 827-row val split with no single gold type regressing by more than 1 row; stage 4
needs Δ accuracy ≥ +1 **and** Δ routed AUROC ≥ 0.

---

## 6. The paper

- **File: `papers/single-pass-decision-acv2.md`** — the audit-corrected version, and the authority. Every
  number in it now re-derives from disk (64.07% pure, 63.82% served, 56.05% grounding, 96.9% candidate
  coverage), and it carries the stage-2 negative arm (§5.10) and the cost window (§5.12).
- `papers/calibrated-decision-interface-2026.md` is the earlier, narrower draft on the calibration axis. It is
  **subordinate**: its 2026-11-07 banner restates its numbers, and where the two disagree the
  `single-pass-decision-acv2.md` text wins.
- **Title of the authority:** *Calibrated to Be Trusted: Training Single-Pass Decision Models for
  Risk-Controlled On-Device Action Selection.* (The subordinate draft carries the older title *Can a Single-Pass
  Decision Model Tell When It Is Wrong?*, which is the calibration-axis framing only.)
- **Framing:** reliability of a decision interface — not a SOTA claim. Three measured results, two measured
  limits.
- **Target venue:** ICML 2027 main track (abstract ~Jan 2027); **TMLR** as the rolling fallback. Not a
  Position track (NeurIPS 2026 explicitly desk-rejects method-superiority positions), not D&B (this is a
  probabilistic-methods result, not benchmark auditing).
- **Abstract may claim:** (i) the error structure is two-dimensional and the natural confidence conflates
  them; (ii) type routing raises AUROC 0.649→0.729 free at inference; (iii) split-conformal certifies 20%
  at 40.7% coverage, with its boundary measured.
- **Abstract may not claim:** SOTA or "outperforms 8B"; "first single-pass zero-token GUI action
  selection"; deployment readiness; any on-device latency.

---

## 7. Honest risk register

1. **Accuracy is a tie, not a win.** 64.07 vs 64.30 on 398 rows (±4.7). Any reframing that implies
   superiority is a rejection trigger.
2. **The framing is crowded.** GUI-JEPA already routes by top-2 margin (AUROC 0.74) on one checkpoint;
   Jev-Mobile owns the zero-token executor framing. Our differentiation must stay narrow: the *chance-level
   product score*, *routing by decision type*, and the *certificate plus its boundary*.
3. **398 rows.** No ranking, no gate, and no claim smaller than the interval. Always report paired tests
   and episode-clustered bootstrap.
4. **The 10% target is dead.** Do not resurrect it without new discrimination; restating it as met would
   be false.
5. **Conformal holds only within a checkpoint.** A certificate is a property of (score, checkpoint,
   split). Never report a threshold fitted on one distribution and evaluated on another.
6. **On-device latency is unmeasured**, and MLX strips vision towers on this Mac. The "on-device" venue is
   an aspiration, not an asset. `convert_to_gguf.py` is the only concrete route and it is unspent.
7. **Licence:** d1 under the LFM Open License v1.0 (source-available, not OSI). Read the full terms before
   releasing fine-tuned weights; state them in the paper either way.
8. **`kntism/skills` is a stale scrape.** Use `huggingface/skills` and `K-Dense-AI/scientific-agent-skills`.

---

## 8. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-09 | Feature = calibrated on-device next-action selection with act/ask/refuse gate | mobile-agent SOTA gap + decision-model category fit |
| 2026-10-09 | Base = LFM2.5 on MLX, text-only | only on-device-viable lineage with d1 pedigree; Mac-only compute |
| 2026-10-09 | Data = raw downloaded only (AndroidControl) | no synthetic/fabricated data |
| 2026-10-13 | Positioning "(C) first calibration benchmark" **retracted** | TypedBench + AgentAbstain took most of it |
| 2026-10-13 | "First open RLCD recipe" **retracted** | THX-01 + decider published full recipes |
| 2026-10-13 | Goal re-anchored: a HARNESS that makes a mobile agent better on one benchmark; benchmark number = the point | user directive |
| 2026-10-13 | Strategy = get SOTA, run it, then improve it | user directive |
| 2026-10-28 | **Base model changed to d1-2B** (LFM3-VL-2B) | Liquid shipped LFM3; d1-2B is the decision-model lineage, not LFM2.5 |
| 2026-10-28 | Scorer bug fixed; all earlier spatial numbers bannered | a key-subset fallback leaked into the spatial branch |
| 2026-10-29 | Headline = 64.07% pure (255/398), official scorer, differential-tested — *written as 64.32/256 at the time; no artifact yields 256, corrected 2026-11-07* | first defensible number |
| 2026-11-07 | **Stage-2 status down-weighting FAILED** its pre-registered bar (val +1.09, test −3.27); the class weight trades premature for correct terminations 27:26. Next arm must be an explicit progress readout | arm result, same date |
| 2026-11-10 | **`episodic-s0` is NOT a benchmark checkpoint** and the `is_done` gate is OFF by default. `is_done` is a copy of the action label (137/137 terminal rows are `status`; 0 positives among 687 non-status rows), so `auroc_nonterm` is null and no threshold has an operating point. AC-v2 55.78% vs 64.07% (−8.29) fails pre-registered condition 3 | the eval was already on disk from 2026-11-09 22:58 and unread; `benchmarks/results/2026-11-10-episodic-s0-fails-prereg.md` |
| 2026-11-10 | **Any AndroidWorld number from `episodic-s0` is descriptive, never a test** | PRE-REG §(c): task success at n=116 has an MDE of ~14 points |
| 2026-11-07 | **Headline restated to 64.07% (255/398)**; no 256/398 artifact exists. Grounding restated to 56.05% pure (56.50% was step 705). Candidate coverage 96.9%, not 100%. Online invalid-action intervals overlap (p=0.065) and the two conditions ran different instances | internal audit `outputs/.drafts/calibrated-decision-value-research-internal-audit.md` |
| 2026-11-02 | Online AndroidWorld loop tried; 1/6 vs 0/6, emulator unstable — **superseded 2026-11-07 by 1/8 in both conditions** (see the row above) | honesty + infra result, not a benchmark result |
| 2026-11-07 | Online restated to **1/8 in both conditions** (Wilson 2.2–47.1%); the 25% "crash rate" withdrawn as a Clock-task checker bug | audit |
| 2026-11-02 | N1 objective ablation (CE/Brier/joint) = **null** at 250 steps | proper scoring moves the average, not discrimination |
| 2026-11-02 | **C1 as originally written is retracted** | the proper-scoring-vs-CE mechanism did not reproduce; the G2/G3 evidence was invalidated by the majority-class artefact |
| 2026-11-02 | **The 10%-budget / 50%-coverage operating point and all on-device latency claims are withdrawn** | 10% is unattainable (risk floor 12.5%); no Apple-silicon measurement exists |
| 2026-11-03 | **C1 replaced by the score-mis-specification mechanism** (type-routed confidence) | measured, reproducible, stable across 3 evaluations |
| 2026-11-03 | **C2 restated to the measured frontier** 20% @ 40.7%; new numeric target = risk at 50% coverage < 10% | honesty; makes the engineering objective falsifiable |
| 2026-11-03 | **C3 delivered for the analysis layer** (`verify_paper.py`, 69 assertions) | one-command provenance for every published number |
| 2026-11-03 | "The remaining gap is grounding, not type" **corrected to "both, comparable"** | 63 vs 68 (test), 161 vs 137 (val), z≈1.4 — not a ranking |
| 2026-11-03 | Grounding metric ruled **honest**; label-noise argument rejected | audit: 80.6% of misses are genuinely different elements |
| 2026-11-03 | Paper reframed as a reliability/mechanism paper, target ICML 2027 / TMLR | accuracy is a tie; the calibration result is the contribution |
| 2026-11-08 | **The duplicated-list "survivor" is NOT significant** — §5.9's t=3.54/df=5 was a naive average-of-variances pooling. Grounding is a count/223 so each arm has 2 distinct values, variance ratio **8.3×**, pooled Student p=0.012 assumes equal variance (violated), **Welch p=0.063**, permutation p=0.029 is the minimum attainable at n=3 vs 4. Pre-registered bar (t≥2.78 at 5 seeds) never reached. **The programme now has no certified positive grounding result** | re-derivation from `backup-2026-11-07/seed_evals/` (`benchmarks/results/2026-11-08_seed_table.py`, `..._seed_test_choice.py`) |

---

## 9. Do-not-cite list (accumulated; each line is a false claim we once made)

- `acc 0.5828 / 0.583` single-pass head (G2) — exactly the majority-class baseline. INVALIDATED 2026-10-16.
- G3 multimodal ECE 0.029 and all G2/G3 accuracies — same invalidated run.
- G4 gate numbers (conf range [0.602, 0.632], τ=0.55 → cov 54% / risk 0.28).
- UI-Venus-1.5 "0.95 step acc" — type-only accuracy; grounding was 0.000. RETRACTED 2026-10-28.
- 2026-10-14 llama smoke numbers (n=4). RETRACTED 2026-10-28.
- d1-2B "0.66 at 160 steps" — inflated by the scorer bug.
- ECE 0.146 → 0.036 isotonic, AUROC 0.723, gates 55.0% / 36.7% coverage — do not reproduce.
- `GUI-Owl-1.5-2B = 67.9%` — no primary source.
- The 1/5–1/3 out-of-box equivalence as a mobile argument — that figure is web; mobile is ~1%.
- "The remaining gap is grounding, not type" — both, comparable.
- Any on-device latency, any 10%/50% gate, any "first single-pass zero-token" novelty.
- The V2P-σ + ConClick grounding plan as a recommended next step — INVALIDATED (attention suppression is
  not implementable on d1-2B).
- "Duplicating to 22.5k improves grounding, 3-4 seeds, p<0.05" — **NOT significant.** Variance ratio 8.3x;
  Welch p=0.063; each arm has only 2 distinct values (count/223). Suggestive only. RETRACTED 2026-11-08.

---

## 10. Where things live

| Thing | Path |
|---|---|
| This goal | `notes/GOAL.md` |
| Entry point / handoff | `docs/HANDOFF.md` |
| Master map + program log | `AGENTS.md` |
| Active plan | `~/.commandcode/plans/rl-decision-model-acv2-2026-11-03.md` |
| Paper | `papers/single-pass-decision-acv2.md` (authority; `calibrated-decision-interface-2026.md` is subordinate) |
| Value brief | `outputs/calibrated-decision-value.md` + `.provenance.md` |
| Latest research note | `notes/research-2026-11-03-calibration-routed-gate.md`; `notes/2026-11-07-session.md` |
| Latest results | `benchmarks/results/2026-11-07-status-downweight-arm.md` (stage-2 arm), `benchmarks/results/2026-11-07-cost-window.md` |
| Provenance bundle | `benchmarks/results/2026-11-03-stage0/` |
| Analysis scripts | `benchmarks/train/{error_budget,paired_eval,gate_routed,grounding_audit,verify_paper}.py` |
| Training pipeline | `benchmarks/train/train_d1.py` (+ gates `test_*.py`) |
| Compute runbook | `docs/colab/GPU_RUNBOOK.md` |
