# PLAN — make the decision model actually better (v2, corrected after adversarial review)

*Supersedes v1 of this file. v1 was reviewed by four independent adversarial agents; three findings were fatal
and the plan below is rebuilt around them. Every correction is marked **[corrected v1 → v2]**.*

---

## 0. The problem, stated without a retracted premise

A 1.92B model reads a screenshot + goal and answers one typed question in one forward pass: which of 8 actions,
and where. Measured on the 398 public AC-v2 steps:

| | value |
|---|---|
| step accuracy | **64.07%** (255/398) |
| action type | **84.17%** |
| grounding (point inside gold element, 223 steps) | **56.05%** |
| type-routed confidence AUROC | **0.729** |
| certified selective risk | 20% at 47.6% coverage (val) |

**[corrected v1 → v2] Fact 1 was wrong.** v1 said "type is nearly solved; placement is not". That is a claim
this programme already **retracted and put on its do-not-cite list** (`notes/GOAL.md` §8, §9): type errors 63 vs
grounding errors 68 on test, 161 vs 137 on val, **z ≈ 1.4, not a ranking**. The two oracle ceilings (＋17.84 for
placement, ＋15.83 for type) differ by 8 rows — inside the noise band this document quantifies in §3. **There is
no evidence that either axis dominates.**

What survives from v1: the errors are 71 element-wrong / 27 spatial-type-wrong / 36 non-spatial-type-wrong / 9
value-wrong, and **no post-hoc threshold has ever improved any of them** (three attempts, all measured, all
failed).

## 1. What the review changed (the important part)

**[corrected v1 → v2] The ensemble result was a selection artefact, and the matched test proves it.**

| evaluation | Δ | p |
|---|---|---|
| **val 827, 3 checkpoints (the only ensemble val can measure)** | **+0.97** | — |
| **test 398, 3 checkpoints (matched size), px = 64** | **+0.50** | **0.63** |
| test 398, 10 checkpoints (v1's headline) | +2.01 | 0.18 |

Full matched-size sweep (3 voters on both splits), px chosen by the same rule:

| px | val Δ | test Δ | gained/lost | p |
|---|---|---|---|---|
| 8 | +0.24 | −0.25 | 0/1 | 1.00 |
| 16 | +0.73 | +0.25 | 1/0 | 1.00 |
| 32 | +0.73 | +0.25 | 2/1 | 1.00 |
| **64** | **+0.97** | **+0.50** | 3/1 | **0.63** |
| 128 | +1.21 | +0.00 | 3/3 | 1.00 |

The effect **scales with the number of voters**, and val has only three checkpoints, so the 10-voter version was
never validated on anything. After correcting for the 16-cell search family, `P(max ≥ the observed 11 net rows) =
0.033`, but **`P(max ≥ 8, i.e. ≥ +2.01) = 0.17`** — the number v1 quoted as "p = 0.077" or "p = 0.18" in two
places is, after selection, **p = 0.17**. v1's 0.077 row is deleted.

**Where the ensemble gain actually comes from** (per gold type, test, px = 32): **click +14**, status −2,
scroll −1, everything else 0. And on grounding alone the vote moves **125 → 139 (+14 of 223)**. So the vote is a
*click-placing* device, not a general improvement.

**[corrected v1 → v2] The correct pass bar is precision on switched rows, not accuracy.**
Exact McNemar on n = 398 needs, for p < 0.05: 12 net rows out of 28 switched (71% precision, ＋3.0 pts), or 18 of
60 (65%, ＋4.5), or 22 of 100 (61%, ＋5.5). Every combiner the programme has built sits at **60–64% precision on
25–28 switched rows**. Accuracy is the wrong currency because ~65% of rows are fixed (179 all-correct, 82
all-wrong) and dominate any accuracy average.

**[corrected v1 → v2] Work #5 is deleted.** v1 proposed ConClick hard negatives + element-size-scaled box σ
("~6 h GPU"). `docs/HANDOFF.md` §3.3 is literally titled *"The grounding V2P-σ plan is INVALIDATED (do NOT spend
GPU on it)"* and `notes/GOAL.md` §9 lists it as do-not-cite. v1 re-proposed it without noticing. Deleted, with a
cross-reference so it cannot be re-lost.

**[corrected v1 → v2] Work #3 is not novel for mobile.** `V-Droid` (MobiCom 2026, arXiv:2503.15937) already uses
an LLM verifier over candidate actions before acting on Android (4.3 s/step, 59.5% AndroidWorld). And AdaGround's
own ablation says a verifier recovers only **38.8% of the headroom** between the better chain and the oracle —
applied to this programme's measured ＋5.83 grounding headroom that is **≈ +1.2 step pts**, statistically
identical to what the free `box_conf × found` signal already gives (+1.26). It also **breaks "single pass"**:
AdaGround's own cost table is 1.0 call at 0.9 s versus 12.5 calls at 24.9 s. The claim would have to become
"single pass, plus one verification on 16% of steps".

**[corrected v1 → v2] The certificate is no longer unclaimed.** `CORA` (arXiv:2604.09155, Apr 2026) publishes
conformal risk-controlled execute/abstain for mobile GUI agents (δ = 0.05, coverage 89.95%, harm 2.42%, 40.0%
AndroidWorld). `SafeGround` (arXiv:2602.02419) covers the grounding-side certificate. **The gate is not the
contribution any more. The price is** — none of them prices the ask, and v1's cost window does.

**[corrected v1 → v2] AC-v2 is not a rankable board.** The same backbone scores **49.62** (ECHOES, Oct 2026) and
**54.80** (Mem-W aggregate) on "AC-v2-High" — a 5-point protocol spread, twice the deficit being chased. In
ECHOES's own table the whole field sits at 46.9–53.6, so Mem-W's 68.59 is an outlier, not a frontier. And the
name collides with an easier board whose SOTA is 86.1. **Stop quoting cross-paper numbers.**

## 1b. A better headline we already own and are not using

The paper currently leads with "64.07% ties an 8B baseline". The ledger already contains a more informative
comparison, and it is not flattering in the way a tie is not:

| model (AC-v2-High) | Pass@1 | Type | Grounding |
|---|---|---|---|
| **ours (d1-2B fine-tuned)** | **64.07** | **84.17** | **56.05** |
| Qwen3-VL-8B (base, no fine-tune) | 54.80 | 69.35 | **67.71** |
| Qwen3-VL-4B (base, no fine-tune) | 49.30 | 63.57 | **69.06** |
| Qwen3-VL-2B (base, per the May-2026 board) | 45.48 | — | — |

*(all from `references/ledger.md`: `memw2026-breakdown`, `sota2androidcontrol`)*

Two facts fall out, and both are more interesting than "we tied":

1. **+18.6 step accuracy over the same-size general VLM** (64.07 vs 45.48). That is the real effect of the
   fine-tune, and it is the claim a builder cares about.
2. **Our grounding is 11–13 points WORSE than an untuned 4B/8B VLM** (56.05 vs 69.06 / 67.71), while our type
   accuracy is 15–21 points better. **The fine-tune traded placement for verb.** That is a clean, measured,
   mechanistic statement about what this architecture buys and what it spends — and it is a better paper
   sentence than a tie.

Caveat, stated: Mem-W does not publish its grounding denominator, so cross-paper grounding is indicative. The
honest fix is cheap — run Qwen3-VL-2B/8B through **our** scorer on **our** 398 rows (~1 GPU-h, plan item in v1
§4).

## 1c. THE MECHANISM, found 2026-11-07: the box loss has zero gradient where the errors are

The `target_box` loss is four per-edge cross-entropies against a Gaussian target with **σ = 4 bins on a 0–999
grid**. For a 1080×2400 screen one bin is 1.08 × 2.40 px, so:

| offset | in y (px) | CE (nats) | ΔCE/bin |
|---|---|---|---|
| 16 | 38 | 10.2 | 6.0 |
| 57 | 137 | 103.7 | 23.4 |
| 100 | 240 | 314.7 | 200 |
| **200** | **480** | **690.8** | **0.0** |
| **415** | **996** | **690.8** | **0.0** |

**Beyond ~400 bins the CE is a numerically identical constant; the gradient is exactly zero.** Our misses are
median **437 px** with **86.8% beyond 150 px** and only 4.4% within 50 px. So the objective is sharp exactly where
our errors are rare and flat exactly where they are common: it teaches the model to refine a nearly-correct click
and tells it nothing at all about having picked the wrong widget.

That is a sufficient explanation for the plateau, and it makes the ordering obvious:

1. **Make placement a discrete choice** (candidate selection) so a wrong element is a wrong *class* and the
   gradient never vanishes. Independently supported: at matched weights the candidate interface already beats the
   box interface by **+6.03 pts (p = 1.5e-3)**, and its ceiling is 92%.
2. **If a coordinate head is kept, bound the loss** — spatial softmax over patches or a clamped distance term —
   so a far prediction still pulls toward the right region.
3. **Add a contrastive margin** between the gold candidate and the model's own chosen distractor
   (ConClick: **+11.3** ScreenSpot-Pro with 8,584 examples; their error profile — 74–88% of wrong clicks hit
   *other elements* — is ours).

Full note: `benchmarks/results/2026-11-07-why-grounding-plateaus.md`.

## 2. What survives from v1 (still true)

- **Ten checkpoints exist; seven are distinct.** Union = **79.40%** on the seven, 80.90% on all ten. Of 398 rows:
  **179 all-correct, 82 all-wrong, 137 mixed** — the entire decision space is 137 rows.
- **Weights remain for 5 of 7** (`E2-cur15k-V`, `E2c-cont-cur8kB`, `R2b-V-cur24k`, `ckpt-groundB`, `arm20` in
  `backup-2026-11-07/arm_out.tgz`). `E2@705` is prediction-only and can never supply a val row.
- **Distance is a real trigger:** median point separation **147 px** when exactly one is right, **8 px**
  otherwise; `box_conf × found` picks correctly **76%** on 21 disagreement rows.
- **Two of my own bugs were found and fixed:** a mutually-exclusive label scheme that silently dropped 57% of
  rows (107 fit examples for a 20-feature model), and CV folds holding positional indices instead of episode ids
  (CV returned 0.5%). Both produced numbers that looked like findings.

## 3. The plan, ordered by expected value per GPU-hour

### #1 — Give the ensemble a val it can actually be measured on (0.3 GPU-h) — **do this first**

Generate 827-row val dumps for the four checkpoints that lack them (`E2c-cont-cur8kB`, `R2b-V-cur24k`, plus the
two that already have val: `E2-cur15k-V`, `ckpt-groundB`, `arm20`). Then the 10-voter ensemble can be fitted and
evaluated on **matched val**, which is the only thing that makes §1's +2.01 interpretable.

**Pass condition (replace v1's):** val and test must agree in *direction* at matched ensemble size. If they do
not, the ensemble line is closed and written up as a negative.
**Why first:** it is the only item whose absence invalidates a number already quoted in the plan.

### #2 — Fit the pooled arbiter on the correct label, on real val (0.3 GPU-h + 1 h CPU)

`benchmarks/train/fusion_arbiter.py` already contains `build_pooled()` — written, never wired — whose docstring
is exactly the fix v1 proposed. Wire it, fit on **all 827 val rows × N checkpoints**, evaluate once.
**[corrected v1 → v2] Pass bar:** **precision ≥ 65% on ≥ 60 switched rows**, not "CV accuracy > best single".
Accuracy is base-rate dominated and will look flat no matter how good the arbiter is on the rows that matter.

### #3 — Make disagreement, not accuracy (1.5 GPU-h) — **moved ahead of the arbiter**

v1 put this fifth. It belongs third, because the arbiter is being asked to score on a surface of **29 rows**.
A genuinely different checkpoint (different data slice, or `--objective brier` which already exists as a flag)
manufactures the disagreement the arbiter needs. **Pass:** the new checkpoint's error set overlaps **< 90%** with
E2's.

### #4 — One verifier measurement, scoped down (0.2 GPU-h)

One marked-click Yes/No pass on the ~64 rows where checkpoints disagree. **No** zoom portfolio, **no**
re-grounding, **no** existence query (AC-v2 has no absent-target class). **Pass bar:** precision ≥ 65% on ≥ 60
switched rows **and** report the score distribution — if > 50% of decisions sit within 0.01 of 1.0 the signal is
saturated and the item is dead regardless of AUROC. Report p50 **and** p95 passes-per-step, and retire "single
forward pass" from the abstract the moment this ships.

### #5 — ~~Grounding levers (ConClick, box σ)~~ — **DELETED** (invalidated; see §1)

## 4. The paper to write instead of an ensemble paper

**[corrected v1 → v2]** The strongest asset is not "+2 pts from ensembling". It is:

> **Priced Interruptions: Cost-Anchored Risk Control for a Single-Pass GUI Decision Model, and the Boundary of
> Its Certificate.**
>
> A 1.92B model that ties an 8B baseline on step accuracy still carries a usable calibrated signal: its
> type-routed score certifies a 20% selective-risk budget at 40.7% test coverage, and gating on it beats both
> always-act and always-ask for error-to-interruption cost ratios in **2.11–5.79×**, a window that survives a
> checkpoint change (2.08–4.19×) while the *certificate* does not (17.3% → 23.9% at the same threshold).

Evidence status: 6 of 8 pieces already exist. The two gaps are **a measured ask-acceptance rate** (nobody has one
for a GUI gate) and **protocol-matched baselines run by us** (Qwen3-VL-2B and 8B through our own scorer, ~1 GPU-h,
which removes the cross-paper comparison entirely).

Fallback if no human-subjects time exists: **"What a 398-row GUI benchmark can and cannot certify"** — the
protocol spread (49.62 vs 54.80), the power analysis (MDES 3.0–3.8 pts at n = 398), and four measured negative
results. Needs **no GPU**, is defensible from artifacts on disk, and is the least likely to be wrong.

## 5. Guard rails (unchanged, and each one earned)

1. **Never gate on gold.** A version of the fusion analysis used `gold['action_type']` and reported +3.52,
   p = 0.001 — an oracle. Honest version: +1.26, p = 0.51.
2. **Fit on val, score test once.** Three numbers this session (+2.76, +3.52, +2.01) were test-selected.
3. **A confound invalidates a number even when the direction is right.** The 3-val / 10-test mismatch is exactly
   this, and §1 shows the matched test collapses the effect from +2.01 to +0.50.
4. **Report gained / lost / precision**, never a bare average. "+2.01" hides 18 gained, 10 lost.
5. **Multiplicity is real.** Six tests have run on the same 398 rows; Bonferroni at 6 sets α = .008, i.e. ≥ 15
   net rows (+3.8 pts). Any future p = .02 is a re-look, not a result.
6. **A negative result is a deliverable.** Four are recorded (status gate, per-class adjustment, learned fusion,
   status arm) and they are what keeps the next GPU-hour honest.

## 6. Cleanup (missing from v1 — the review caught it)

After any GPU work: back up to `backup-YYYY-MM-DD/`, verify the archive lists, then delete with **the same file
list that created them** (this programme has missed 3.2 GB across 4.6 GB of leftovers by verifying with a
shorter pattern), check `stat` mtimes before removing anything shared, and confirm no pre-existing container was
touched. See `docs/compute/CLEANUP-2026-11-07.md` for the template and the two re-audits that were needed.

## 7. Verification

`verify_paper.py` 69/69 (stage-0 scope, self-declared) · `scripts/check_record.py` 24/24 · `benchmarks/aw/tests`
110 passed / 35 skipped · every number above re-derives from `benchmarks/results/` or `backup-2026-*/`.

## 8. Reviews and their disposition

| review | status | disposition |
|---|---|---|
| strategy (`2026-11-07-review-strategy.md`) | **complete**, 4 FATAL | **all four adopted**: premise corrected (§0), ensemble demoted (§1), #5 deleted, #3 re-scoped |
| statistics | timed out, partial data recovered | **key findings adopted**: family-wise p = 0.17 after 16-cell search; matched-size transfer +0.50 at 3 voters; per-type gain is clicks only |
| leakage | timed out, partial data recovered | **finding adopted**: 3-voter val is the only matched comparison, and it does not replicate the 10-voter gain |
| engineering | timed out, partial data recovered | **finding adopted**: `build_pooled()` exists unwired — #2 is a wiring job, not new code; weights verified for 5 of 7 runs |
