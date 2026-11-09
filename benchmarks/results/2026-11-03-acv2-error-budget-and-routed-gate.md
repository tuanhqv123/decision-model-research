# 2026-11-03 — AC-v2 error budget + type-routed calibrated gate (zero-GPU forensic pass)

> Recomputes everything from **saved prediction jsonl**, no model run. Two deliverables:
> (1) the error budget is four buckets, and **type errors are not smaller than grounding errors** —
> this corrects `notes/grounding-precision-2026.md` §2 and the paper's framing;
> (2) the confidence gate was **routed wrong**, and routing it by action type roughly doubles the
> correctness ranking quality. A split-conformal risk budget then transfers to AC-v2 on the matched
> checkpoint but **fails to transfer on a weaker one** — the binding constraint is exchangeability,
> not the score.

Artifacts: `benchmarks/results/2026-10-29-evals/{E2-938-V,E2-705-V,E2-val}.jsonl`.
Scripts: `benchmarks/train/{error_budget,paired_eval,gate_routed}.py`.

## 0. Artifact caveat (read before citing)

- `E2-938-V.jsonl` is 258/398 = **64.82**, i.e. the **app-name-hinted** run, not the pure run. The pure artifact is **255/398 = 64.07** (`E2-nomem`), 254/398 served; the 256/398 in this line is a pre-audit figure and does not exist on disk.
  The pure per-row predictions are **not** in `2026-10-29-evals/` (E2-705-V is 244 → 61.31); the pure artifact lives in
  `backup-2026-10-29/.../E2-nomem/eval-E2final-nomem-V.jsonl` as **255/398** (a pre-audit line said 64.32/256).
  Numbers below are therefore computed on a 258/398 checkpoint; the error *composition* is what is
  being reported, not a new headline. Regenerating the pure artifact is ~52 s of GPU.
- `E2-val.jsonl` (827 rows) is the validation split of **train** episodes; `E2-*-V` are the AC-v2
  leaderboard episodes. They are different distributions — see §3.

## 1. Error budget (4 buckets, not 1)

`python benchmarks/train/error_budget.py <jsonl>`

| Bucket | test (n=398, 140 err) | val (n=827, 308 err) |
|---|---|---|
| A. spatial gold, **element wrong** | 68 (48.6%) → **+17.09 pts** | 137 (44.5%) → +16.57 |
| B. spatial gold, type wrong | 27 (19.3%) → +6.78 | 52 (16.9%) → +6.29 |
| C1. non-spatial, type wrong | 36 (25.7%) → +9.05 | 109 (35.4%) → **+13.18** |
| C2. non-spatial, value wrong | 9 (6.4%) → +2.26 | 10 (3.2%) → +1.21 |

Oracle ceilings, test: grounding only **81.91%**, type only **80.65%**, both **97.74%**.

**Correction to the repo narrative.** `notes/grounding-precision-2026.md` §2 concludes "the problem is
element SELECTION (semantic) … not type", and the paper repeats it. That conflated *type accuracy is
high (84.17%)* with *type errors are few*. They are not:

| | test 398 | val 827 |
|---|---|---|
| type errors | **63** | **161** |
| grounding errors | **68** | **137** |

Type errors are 45% of test errors and **52% of validation errors** — **both axes are first-order and
comparable**. ⚠️ **Statistical correction:** 63 vs 68 (test) and 161 vs 137 (val) are *not* a ranking.
On val, p₁=161/827, p₂=137/827, multinomial SE of the difference ≈0.021, diff=0.029 ⇒ **z≈1.4,
p≈0.16 — not significant.** Report "comparable magnitudes"; never write "type exceeds grounding".

Any plan that only fixes grounding leaves roughly half the headroom on the table, and vice versa.

Named slices (test / val, worth in points): premature `status` 21 / 51 (+5.28 / +6.17);
gold `scroll` missed 19 / 45 (+4.77 / +5.44); spurious `navigate_back` 8 / 19 (+2.01 / +2.30);
`open_app` wrong value 6 / 4 (+1.51 / +0.48).

Per gold type, test: click 121/212 (57.1), scroll **18/37 (48.6)**, open_app 20/27, navigate_back 13/20,
type_text 7/11, status 76/87, wait 3/4. `scroll` is 9.3% of the eval but 13.6% of the errors.

## 2. Paired testing — what makes a +2 pt gain claimable

`python benchmarks/train/paired_eval.py BASE.jsonl CAND.jsonl`

Self-test (E2 vs E2): Δ=0.00, b=c=0, p=1.0. Validation of the machinery, E2@705 → E2@938:

| | rows | delta | episode-clustered 95% | McNemar |
|---|---|---|---|---|
| E2@705 → E2@938 | 398 (350 eps) | **+3.52 pts** | [+0.76, +6.33] | b=8, c=22, **p=0.016** |

Independent CIs on 398 rows (64.07 → [59.2, 68.6]) cannot resolve this; the paired test can. Also
informative: the 705→938 gain was **+12 non-spatial rows and only +2 spatial rows** (scroll +7 alone) —
late training buys *type* decisions, not grounding. Grounding saturates early.

## 2b. Data-mix audit — two findings that bear on the headline and on §6.2

Action-type rates measured from the local compact jsonl vs the eval gold set:

| type | `cur15k` train | eval | ratio | `validation` | ratio vs eval | full `compact_train` |
|---|---|---|---|---|---|---|
| click | 52.75% | 53.27% | 0.99× | 50.79% | 0.95× | 52.58% |
| status | 22.00% | 21.86% | 1.01× | 16.57% | **0.76×** | 15.41% |
| scroll | 9.00% | 9.30% | 0.97× | 10.40% | 1.12× | 11.14% |
| open_app | 7.00% | 6.78% | 1.03× | 5.44% | 0.80× | 5.71% |
| navigate_back | 5.00% | 5.03% | 0.99× | 3.51% | **0.70×** | 3.02% |
| type_text | 3.00% | 2.76% | 1.09× | 6.41% | **2.32×** | 6.12% |
| wait | 1.00% | 1.01% | 1.00× | 6.77% | **6.74×** | 5.84× |

**(a) The training mixture was stratified to the test action distribution.** Every `cur15k` ratio is
within 0.97–1.09×, while the natural pool (`compact_train`, n=88,288) has `status` 15.41% against the test
21.86%. That is **test-informed data design** and must be disclosed as a limitation; it flatters the
headline by an unmeasured amount. Fix path: re-run without stratification, or stratify every compared
system identically.

**(b) The calibration split is *not* distribution-matched to the test split** — `wait` 6.74×,
`type_text` 2.32×, `navigate_back` 0.70×, `status` 0.76×. This is an **independent mechanistic cause** of
the §6.2 transfer failure: a threshold fitted on val is fitted on a different action mix than it meets at
test. So the exchangeability break is not only "the second checkpoint is weaker" — the splits themselves
are shifted.

## 2d. Learned confidence fusion — better point estimate, not a claimable gain

`score_fusion.py`. An L2 logistic on 8 per-row readout features (action top-1, top-2 margin, 8-way
entropy, `box_conf`, `found`, spatial flag, two routing interactions), hyper-parameters chosen by
5×`StratifiedGroupKFold(5)` **inside** val with groups = episode; test scored once.

| score | AUROC val | AUROC test | test risk @50% | test AUPRC |
|---|---|---|---|---|
| `p(action)×box_conf` | 0.677 | 0.649 | 0.266 | 0.771 |
| entropy only (−H) | 0.667 | 0.634 | 0.256 | 0.736 |
| type-routed (hand) | 0.722 | 0.730 | 0.226 | 0.811 |
| **learned fusion** | **0.767** (OOF 0.758) | **0.743** | **0.191** | 0.829 |

The model rediscovers the routing rule on its own: largest standardised coefficients are
`box_conf×spatial` **+1.08**, `p(action)×nonspatial` **+0.50**, `spatial` **−0.46**.

**Not claimed.** Paired, episode-clustered ΔAUROC on test = **+0.013, 95% [−0.020, +0.043]** — inside the
MDE for 398 rows / 140 positives (≈+0.02). Report the point estimate and the interval; do not claim it.

## 2e. The bound is worth more than the model: CRC vs Clopper–Pearson

CRC (Angelopoulos et al., arXiv 2208.02814): `λ̂ = inf{λ : n/(n+1)·R̂(λ) + 1/(n+1) ≤ δ}`. Same val split,
same fused score, test n=398:

| δ | bound | τ | val cov | test cov | test risk (sd) | held |
|---|---|---|---|---|---|---|
| 20% | CP | 0.713 | 45.1% | 38.4% | 19.6% (3.2) | yes |
| 20% | **CRC** | 0.667 | 51.4% | **45.5%** | 18.2% (2.9) | yes |
| 25% | CP | 0.568 | 63.5% | 60.6% | 22.4% (2.7) | yes |
| 25% | **CRC** | 0.522 | 69.0% | **66.1%** | 23.6% (2.6) | yes |
| 10% | CP | — | — | — | — | **not certifiable** |
| 10% | CRC | 0.913 | 4.0% | **3.5%** | 0.0% (14 steps) | "yes", degenerate |

**+6–7 points of coverage at the same budget** from the bound choice alone. The 10% budget *is*
certifiable under CRC — at 14 accepted steps. That is the arithmetic of the risk floor, not an operating
point, and it is the strongest evidence for §6.1.

## 2f. Action-mix shift is measurable and only partly fixable
TV distance val→test **0.106**; exact likelihood ratios (discrete covariate) 1.43 / 1.32 / 1.25 / 1.05 /
0.89 / 0.43 / 0.15 for navigate_back / status / open_app / click / scroll / type_text / wait. Effective
calibration size after weighting **752 of 827 (90.9%)** — cheap. But the max risk movement available from
reweighting is bounded by `TV × per-type risk spread` ≈ **2 points**, against an observed 6.6-point
cross-checkpoint gap whose binomial SE is **4.5 points (≈1.5σ)**. Report the gap, the SE, and the
explained fraction — the qualitative claim survives, the quantitative one does not.

## 2g. Stage-2 diagnosis — scroll and premature-`status` (2026-11-04 checkpoint)

Goal §5 stage 2 is "the type bucket". Two slices tested directly. **Three hypotheses died, one lever
survived as negative.**

### `scroll` is a TYPE error, not a direction error
Gold `scroll` = 37 rows, 19 errors:
- **type errors (pred ≠ scroll): 16** — `status` 9, `click` 7
- direction errors (pred = scroll, wrong direction): **3** — (up→down) 2, (right→down) 1
- scroll type accuracy 21/37 = **56.8%**

This matches SwipeGen (arXiv 2601.18305): swipe failures are 69% region/scrollable-target and only 6%
direction. **Do not build a direction model.** The lever is type under-prediction, and the ceiling if all
16 were fixed is **+4.0 pts**.

### Premature `status` is NOT an episode-boundary artifact — hypothesis refuted
21 rows where pred = `status`, gold ≠ `status` (gold types: click 12, scroll 9).
- base rate of "at the episode's last evaluated step" over all 398 rows: **87.9%**
- premature-status rows at that step: **85.7%** → enrichment **0.97×** ⇒ **no effect**
- Caveat that invalidates this family of test: the AC-v2 subset is ~1 step per 350 episodes, so
  episode-position hypotheses **cannot be tested on this benchmark**.

Separability (so the signal exists, but not as a rule): median `p(terminate)` **0.671** for premature vs
**0.882** for correct-`status`; median margin +0.432 vs +0.815. `found` is useless here (1.000 both).
A "suppress `status` when grounding evidence is strong" rule flips **21/21** premature rows **and
76/76** correct rows — net **−55**. **Refuted; do not try it.**

### Per-class logit adjustment: passes on val, fails on test (the mix shift bites again)
`class_bias.py`, additive per-class offsets (Menon et al., arXiv 2007.07314) fitted on val to maximise
type accuracy:

| | val type acc | test type acc |
|---|---|---|
| unadjusted | 0.8053 | **0.8417** |
| adjusted | **0.8210** | 0.8141 |

Test: 63 → 74 errors, b=18 regressed / c=7 fixed, **McNemar p = 0.043** — significantly *worse*. Root
cause is exactly §2b(b)/§2f: the bias is fitted on val's action mix (`status` 16.57%, `type_text` 6.41%)
and applied to test's (`status` 21.86%, `type_text` 2.76%). This is the mix mismatch producing a concrete
practical failure, not just a conformal caveat.

### BLOCKED
Any step-level version of the above needs the `scroll_dir` / `type_text` / `app_name` readouts for the
switched-to class. The saved jsonl stores only the *emitted* action's `value`, so a type-changing
post-processor cannot be scored offline. **Needs a model re-run** (~52 s for the 398 rows, plus saving all
readouts).

## 3. Type-routed calibrated gate

`python benchmarks/train/gate_routed.py --val E2-val.jsonl --test E2-938-V.jsonl`

**Mechanism.** On spatial rows the box readout is the honest signal; on non-spatial rows it is
meaningless, and multiplying by it destroys the signal. One global `p(action) × box_conf` gate averages
two incomparable signals. Route by bucket, then map each bucket through an ECDF **fitted on validation**
and applied unchanged to test (no transductive leakage).

### AUROC(score, correct)

| score | val | test 938 | test 705 |
|---|---|---|---|
| global `p(action)×box_conf` (previously reported) | 0.677 | 0.649 | 0.662 |
| `p(action)` only | 0.662 | 0.631 | 0.626 |
| `box_conf` only | 0.665 | 0.646 | 0.656 |
| **type-routed (new)** | **0.760** | **0.729** | **0.709** |

Within bucket (test): spatial `box_conf` **0.793 / 0.783** (938 / 705) vs `p(action)` 0.687 / 0.688;
non-spatial `p(action)` 0.618 / 0.601 vs `p(action)×box_conf` **0.504 / 0.526 (chance)**.

→ The old gate's weakness is fully explained: it multiplies a meaningful signal by a meaningless one on
44% of rows. This is why the earlier gate numbers (ECE 0.146→0.036, AUROC 0.723, coverage 55.0%/36.7%)
never reproduced.

AURC (lower better): routed 0.209 val / 0.201–0.239 test, vs global 0.243 / 0.244–0.270.

`box_conf` on spatial rows is **under**-confident, which is the safe direction: val mean 0.419 vs acc
0.601, ECE 0.183, Brier 0.215 = REL 0.046 − RES 0.070 + UNC 0.240; test mean 0.363 vs acc 0.574,
ECE 0.222, Brier 0.238.

### Risk control — and the honest limit

One-sided Clopper–Pearson upper bound, α=0.10, threshold fitted on val to maximise coverage:

| budget δ | val coverage | val risk | val UB | test coverage (938) | test risk | test UB | held? |
|---|---|---|---|---|---|---|---|
| 5% | — | — | — | — | — | — | **not certifiable** |
| 10% | — | — | — | — | — | — | **not certifiable** |
| 20% | 47.6% | 17.3% | 19.9% | **40.7%** | **17.3%** | 21.7% | YES |

Attainable frontier on val: δ=30% → 78.0% coverage, δ=25% → 66.1%, δ=20% → 47.6%, **δ≤15% not
certifiable**.

**Two hard findings, both of which invalidate a claim currently in the paper:**

1. **A 10% risk budget at ≥50% coverage is not attainable** on this checkpoint. With zero errors,
   certifying 10% needs n_act ≥ 22 (2.7% of val); val's top-20 rows already contain 2 errors (10%).
   The score's risk floor is ~12.5%: test risk is 21.0% at 50.3% coverage, 17.5% at 40.2%, 12.5% at 20.1%.
   To reach 10% at 50% coverage the *ranking* must improve, not the threshold — that is now a concrete
   target for training, and the paper's C2 ("risk ≤10% at coverage ≥50%") must be restated or dropped.
2. **The conformal guarantee does not transfer across checkpoints.** The same τ=0.527 fitted on val
   gives test risk **17.3% on E2@938 (holds)** but **23.9% on E2@705 (violates the 20% budget)**.
   The val split is in-distribution for the 938 checkpoint's quality and out-of-distribution for a
   weaker one; exchangeability fails. This is the mechanism behind "the numbers did not reproduce",
   and it is the paper's own N3 (shift) question showing up in measurement.

## 4. What is claimed / not claimed

**Claimed (reproducible from saved jsonl, stable across 3 evaluations):**
- Type routing raises correctness-ranking AUROC from 0.649 to **0.729** on held-out AC-v2 (0.709–0.729
  across checkpoints, 0.760 on the calibration split); the improvement is localised exactly where the
  errors are (`box_conf` 0.79 on spatial, `p(action)` 0.60–0.62 on non-spatial).
- A split-conformal 20% selective-risk budget holds on AC-v2 test at **40.7% coverage** for the matched
  checkpoint.

**Not claimed:**
- Any 10%-budget / 50%-coverage gate. Any gate number on a checkpoint other than the one the threshold
  was fitted for. Any ECE-improvement claim from isotonic mapping (retracted, unreproduced).
- Nothing here is a new accuracy number. 64.07 (pure, 255/398) / 64.82 (hinted, 258/398) are unchanged.

## 5. Required corrections

1. `notes/grounding-precision-2026.md` §2 and the paper's "remaining gap is grounding, not type" →
   **both** (63 vs 68 test; 161 vs 137 val).
2. Paper §5.2(b): replace the isotonic gate paragraph with the routed-conformal result above, keeping the
   retraction notice.
3. `notes/GOAL.md` C2: "risk ≤10% at coverage ≥50%" is **not attainable at this checkpoint**; restate as
   the measured frontier, or make "reduce risk at 50% coverage from 21% to <10%" the training target.
4. Add the exchangeability finding (§3.2) to Limitations — it is a positive mechanism result, not a
   failure, and it is the honest replacement for the non-reproducible numbers.

## 6. Reproduce

```bash
cd benchmarks/train
python error_budget.py ../results/2026-10-29-evals/E2-938-V.jsonl
python error_budget.py ../results/2026-10-29-evals/E2-val.jsonl
python paired_eval.py  ../results/2026-10-29-evals/E2-705-V.jsonl ../results/2026-10-29-evals/E2-938-V.jsonl
python gate_routed.py  --val ../results/2026-10-29-evals/E2-val.jsonl \
                       --test ../results/2026-10-29-evals/E2-938-V.jsonl --alpha 0.1
```

Dependencies: numpy only. No GPU, no model, no network.
