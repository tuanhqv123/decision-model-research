# 2026-11-03 — Decision-model calibration: error budget, routed gate, grounding audit

> Single research note for the zero-GPU forensic pass over the d1-2B AC-v2 result.
> Everything here is recomputed from **saved prediction jsonl + the eval set's a11y trees**.
> No model run, no GPU, no network. Reproduce in seconds.
>
> Scripts: `benchmarks/train/{error_budget,paired_eval,gate_routed,grounding_audit}.py`
> Full write-up: `benchmarks/results/2026-11-03-acv2-error-budget-and-routed-gate.md`

---

## 1. The four findings, in order of importance

### F1. The error budget is four buckets, and "grounding, not type" is the wrong diagnosis
Saved predictions (`E2-938-V.jsonl`, n=398, 258 correct, 140 errors):

| bucket | rows | share | worth |
|---|---|---|---|
| A spatial gold, **element wrong** | 68 | 48.6% | +17.09 pts |
| B spatial gold, type wrong | 27 | 19.3% | +6.78 |
| C1 non-spatial, type wrong | 36 | 25.7% | +9.05 |
| C2 non-spatial, value wrong | 9 | 6.4% | +2.26 |

Oracle ceilings: grounding only 81.91%, type only 80.65%.

`notes/grounding-precision-2026.md` §2 and the paper both say "the remaining gap is grounding, not
type". **That is wrong**, and it conflated *type accuracy is high (84.17%)* with *type errors are few*.
Type errors = 63 on test, 161 on val.

**⚠️ Statistical correction (do not over-claim this).** 63 vs 68 on test and 161 vs 137 on val are
**comparable magnitudes, not a ranking**. On val: p₁=161/827, p₂=137/827, multinomial SE of the
difference ≈0.021, diff=0.029 ⇒ **z≈1.4, p≈0.16 — not significant.** Say "both axes are first-order
and comparable", never "type exceeds grounding".

### F2. The confidence gate was routed wrong — this is the paper's positive result
On **spatial** rows the box readout is the honest signal; on **non-spatial** rows it is meaningless, and
the repo's `p(action) × box_conf` product destroys the signal there.

| score | val | test(938) | test(705) |
|---|---|---|---|
| global `p(action)×box_conf` (previously reported) | 0.677 | 0.649 | 0.662 |
| `p(action)` only | 0.662 | 0.631 | 0.626 |
| **type-routed (box_conf on spatial, p(action) elsewhere)** | **0.760** | **0.729** | **0.709** |

Within bucket (test): spatial `box_conf` **0.793/0.783**; non-spatial `p(action)×box_conf`
**0.504/0.526 = chance**. That is the mechanical explanation for why the earlier gate claims
(ECE 0.146→0.036, AUROC 0.723, coverage 55.0%/36.7%) never reproduced.

`box_conf` on spatial rows is **under**-confident (val mean 0.419 vs acc 0.601; test 0.363 vs 0.574) —
the safe direction for a risk budget. AURC: routed 0.209/0.201 vs global 0.243/0.244.

### F3. Two hard limits — both kill a claim currently in the paper
1. **A 10% risk budget at ≥50% coverage is NOT attainable.** The routed score's risk floor is ~12.5%
   (test risk 21.0% at 50.3% coverage, 17.5% at 40.2%, 12.5% at 20.1%). Split-conformal certifies only
   **20% at 40.7% coverage** (empirical 17.3%, CP upper 21.7%). Front to certify δ=10% with zero errors
   needs n_act ≥ 22 (≈2.7% of val) — so the budget is unreachable at useful coverage.
   ⇒ `notes/GOAL.md` **C2 must be restated**; do not publish a 10%/50% gate.
2. **The conformal threshold does not transfer across checkpoints.** Same τ=0.527 (fitted on val):
   **17.3% risk on E2@938 (holds)** but **23.9% on E2@705 (violates 20%)**. The binding constraint is
   **exchangeability, not the score**. A conformal threshold is a property of (score, checkpoint, split),
   not of the model family. This is the mechanism behind "the numbers did not reproduce".

### F4. The grounding metric is HONEST — the misses are real, not label noise
`grounding_audit.py` joins predictions with the eval set's accessibility trees and classifies every
element miss (198 joined spatial rows, 67 misses, exact hit 66.16%):

| miss category | count | share of misses |
|---|---|---|
| genuinely **different element** | 54 | **80.6%** |
| same widget, parent/child subtree | 12 | 17.9% |
| same label (dataset ambiguity) | 1 | 1.5% |
| IoU≥0.5 / overlaps gold | 0 | 0.0% |

Tolerance curve: exact 66.16% → +subtree 72.73%. Label ambiguity on 9.6% of rows.
Gold element median area 0.85% of screen; **55.1% of spatial rows have a gold element ≤1% of screen**;
gold area when hit (median 29,106 px²) is ~2× larger than when missed (15,876 px²).

**This is a negative result for an appealing hypothesis and it must be kept.** AndroidControl-Curated
(arXiv:2510.18488) reports ~30% defective GT overall, which suggested our 56.5% grounding might be
partly an artifact. For *our* error set it is not: 80.6% of our misses are genuinely wrong elements.
⇒ **Do not claim the grounding metric under-reports us. Do not use the label-noise argument.**

## 2. Paired testing is the missing instrument
`paired_eval.py`: McNemar exact + **episode-clustered** paired bootstrap + per-gold-type Δ. Self-test
Δ=0, p=1.0. (Cluster by episode: 398 rows span 350 episodes and steps within an episode are correlated,
so a row-level bootstrap understates the interval.)
Demo, E2@705→@938: **+3.52 pts, [0.76, 6.33] clustered, b=8/c=22, p=0.016** on the same 398 rows, where
independent CIs (64.07 → [59.2, 68.6], ±4.7) cannot separate anything. *(Corrected 2026-11-07: the pure artifact is 255/398, not 256.)* Also: that gain was +12 non-spatial
rows and only **+2 spatial** — late training buys type decisions; grounding saturates early.

## 2b. Provenance check
`verify_paper.py` asserts every number published in `papers/calibrated-decision-interface-2026.md`
against `benchmarks/results/2026-11-03-stage0/` (63 assertions, exits nonzero on drift). It caught one
stale artifact during drafting. **Run it after any script change.**

## 2d. F6 — learned fusion helps the point estimate but is NOT claimable
`score_fusion.py`: L2 logistic on 8 per-row readout features (top-1, top-2 margin, 8-way entropy,
`box_conf`, `found`, spatial, 2 routing interactions), hyper-params by 5×`StratifiedGroupKFold(5)`
**inside** val with groups = episode; test scored once.

| score | val | test | test risk@50% | test AUPRC |
|---|---|---|---|---|
| global p×box | 0.677 | 0.649 | 0.266 | 0.771 |
| entropy only | 0.667 | 0.634 | 0.256 | 0.736 |
| routed (hand) | 0.722 | 0.730 | 0.226 | 0.811 |
| **learned fusion** | **0.767** (OOF 0.758) | **0.743** | **0.191** | 0.829 |

It rediscovers the routing rule by itself (`box_conf×spatial` +1.08, `p1×nonspatial` +0.50,
`spatial` −0.46). **Paired episode-clustered ΔAUROC = +0.013, 95% [−0.020, +0.043] → not significant**
(MDE ≈ +0.02 at 398 rows / 140 positives). Report the estimate and the interval; claim nothing.

## 2e. F7 — the BOUND is worth more than the model (CRC vs Clopper–Pearson)
CRC (Angelopoulos et al., arXiv 2208.02814): `λ̂ = inf{λ : n/(n+1)·R̂(λ) + 1/(n+1) ≤ δ}`.

| δ | CP | CRC |
|---|---|---|
| 20% | 38.4% cov @ 19.6% risk | **45.5% cov @ 18.2% risk** |
| 25% | 60.6% @ 22.4% | **66.1% @ 23.6%** |
| 10% | not certifiable | "certified" at **14 steps / 3.5% cov** |

**+6–7 pp of coverage at the same budget from the bound choice alone** — comparable to years of modelling
on this benchmark. The 10% row is the cleanest proof of the risk floor: certifiable, but useless.
Also: val→test action-mix TV distance **0.106**, ESS after weighting **752/827 (91%)**, and reweighting
can move risk by at most ~2 pp against an observed 6.6-pp cross-checkpoint gap whose binomial SE is
~4.5 pp (**≈1.5σ**) — report gap + SE + explained fraction; the qualitative claim survives, the
quantitative one does not.

## 2c. F5 — Data-mix audit (new, 2026-11-03): leakage *and* the cause of the transfer failure
Measured action-type rates from the local compact jsonl vs the eval gold set:

| type | cur15k | eval | ratio | validation | ratio | full pool |
|---|---|---|---|---|---|---|
| click | 52.75% | 53.27% | 0.99× | 50.79% | 0.95× | 52.58% |
| status | 22.00% | 21.86% | 1.01× | 16.57% | **0.76×** | 15.41% |
| scroll | 9.00% | 9.30% | 0.97× | 10.40% | 1.12× | 11.14% |
| open_app | 7.00% | 6.78% | 1.03× | 5.44% | 0.80× | 5.71% |
| navigate_back | 5.00% | 5.03% | 0.99× | 3.51% | **0.70×** | 3.02% |
| type_text | 3.00% | 2.76% | 1.09× | 6.41% | **2.32×** | 6.12% |
| wait | 1.00% | 1.01% | 1.00× | 6.77% | **6.74×** | 5.84% |

**(a) DISCLOSE: `cur15k` is stratified to the *test* action distribution** (all ratios 0.97–1.09×) while
the natural pool has `status` 15.41% vs test 21.86%. That is test-informed data design — a leakage risk
that flatters the headline by an unmeasured amount. Either re-run unstratified or stratify every compared
system identically.
**(b) The calibration split is mismatched to test** (`wait` 6.74×, `type_text` 2.32×), which is an
**independent mechanistic cause** of the §6.2 conformal transfer failure — not just "the second checkpoint
is weaker". Prefer a calibration split drawn from the deployment distribution.

## 2f. F8 — Stage-2 diagnosis: three hypotheses died (2026-11-04)
- **`scroll` is a TYPE error, not direction.** 19 errors on 37 gold-scroll rows: **16 type** (`status` 9,
  `click` 7), only **3 direction**. Scroll type acc 21/37 = 56.8%. Matches SwipeGen (2601.18305: 69%
  region, 6% direction). ⇒ **do not build a direction model**; ceiling +4.0 pts from the type fix.
- **Premature `status` is NOT an episode-boundary artifact.** 87.9% of all rows are at their episode's
  last evaluated step vs 85.7% of the 21 premature rows ⇒ enrichment **0.97×**. `p(terminate)` does
  separate them (0.671 vs 0.882) but the rule "suppress status when grounding is strong" flips 21/21
  premature **and 76/76 correct** ⇒ net **−55**. **Refuted.** Also: AC-v2 is ~1 step/episode, so
  episode-position hypotheses **cannot be tested** on it.
- **Per-class logit adjustment (2007.07314) works on val, fails on test:** type acc 0.8053→0.8210 (val)
  but **0.8417→0.8141 (test)**, 18 regressed / 7 fixed, McNemar **p=0.043 significantly worse**. Cause =
  the val/test action-mix mismatch (F5b) — concrete proof that a post-hoc rule belongs to its own
  distribution.
- **BLOCKED:** any step-level type change needs the `scroll_dir`/`type_text`/`app_name` readouts for the
  switched-to class; the jsonl stores only the emitted action's `value`. Needs a model re-run.

## 3. Artifact gap (unresolved)
`E2-938-V.jsonl` is **258/398 = the app-name-hinted run**, not the pure 64.07 (255/398, `E2-nomem`). *(Corrected 2026-11-07.)* The pure
predictions are **not** on disk. Regeneration ≈52 s of GPU (398 × 130 ms). Two numbers circulate for the
same thing (official 64.07, our port 64.07 — the pre-audit text said 64.32 for the official run) — pick one and label it.

## 4. Literature that changes the plan (researched 2026-11-03)
- **AndroidControl-Curated** (2510.18488): ~30% of AndroidControl is defective (37.6% wrong GT). Real,
  but per F4 it does **not** explain our misses.
- **GUI-JEPA** (2610.36877) already does confidence routing (top-2 margin, AUROC 0.74) on one
  checkpoint. Differentiation must be: theirs is a margin heuristic; ours shows the *standard product
  score is at chance off spatial rows* (0.504), routes by decision type, and attaches a distribution-free
  risk certificate **plus its boundary**.
- **Selective Conformal Risk Control** (2512.12844): gives the formal feasibility condition for the
  10%-budget wall, and states that *selection breaks exchangeability* — exactly F3.2. Cite it there.
- Element-selection levers with 2026 evidence, if GPU is available: candidate-region reranker
  (RankGround 2609.18690, +16–19 at 2B), mask supervision (Seg2Click 2610.16234, +1.5–4.9),
  unfreeze backbone (GUI-Actor 2506.03143, +11.3 at 2B; a verifier on a frozen backbone recovers +8.6).
  AMEX has 25× more Android element annotations than AndroidControl — the biggest under-used mobile data.

## 5. Tooling notes
- `kntism/skills` (source of the `model-trainer` skill) is a **1-star, unmaintained scrape** of
  `huggingface/skills` + `anthropics/skills` + `obra/superpowers`. Use upstream instead:
  `huggingface/skills` (`huggingface-llm-trainer`, `trl-training`, `hf-cli`, `huggingface-trackio`)
  and `K-Dense-AI/scientific-agent-skills` (`scientific-writing`, `venue-templates`,
  `citation-management`, `literature-review`, `statistical-analysis`, `scientific-visualization`).
- Skill spec: `agentskills.io`; only 6 portable frontmatter fields; keep `SKILL.md` < 500 lines and push
  detail to `references/`.

## 6. Reproduction
```bash
cd benchmarks/train
python error_budget.py     ../results/2026-10-29-evals/E2-938-V.jsonl
python error_budget.py     ../results/2026-10-29-evals/E2-val.jsonl
python paired_eval.py      ../results/2026-10-29-evals/E2-705-V.jsonl ../results/2026-10-29-evals/E2-938-V.jsonl
python gate_routed.py      --val ../results/2026-10-29-evals/E2-val.jsonl \
                           --test ../results/2026-10-29-evals/E2-938-V.jsonl --alpha 0.1
python grounding_audit.py  --rows ../data/acv2/500_steps_filtered.json \
                           --pred ../results/2026-10-29-evals/E2-938-V.jsonl
```
Dependencies: numpy only.
