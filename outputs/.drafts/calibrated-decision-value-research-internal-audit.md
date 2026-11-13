# Internal audit — which headline numbers reproduce, which are underpowered, which must be retracted

*Task T4. Read-only audit of local artifacts. No GPU, no network, no edits to any artifact; every
re-derivation below is CPU/numpy and writes only to `/tmp`.*

**Verdict in one paragraph.** The *calibration half* of the paper is in excellent shape: every number in
`benchmarks/results/2026-11-03-stage0/` re-derives **byte-identically** from the three saved prediction
JSONLs, `verify_paper.py` passes 69/69, and the gate AUROC / error-budget / fusion / knockout claims
survive contact with disk. The *accuracy half* does not. `64.32% (256/398)` exists in **no artifact on
disk** — the only pure-protocol artifacts are 255/398 (in-process, `E2-nomem`) and 254/398 (served API,
`e2-pure-hist.jsonl`) — and the `Grounding 56.50% (126/223)` printed in the same table row is the
**step-705** checkpoint's grounding, not the 938 checkpoint's. Two further published numbers are
contradicted outright by their own artifacts: the candidate-interface coverage `1.0000` (artifact: 0.9686)
and §5.7's "**non-overlapping** Wilson intervals" for the online invalid-action rate (they overlap
19.0–25.0, Fisher p = 0.065, and the two conditions were run on **different task instances**).

---

## 0. Method and scope

Commands run (all from repo root `~/Workspace/research`, Python 3 / numpy 2.0.2 /
scikit-learn 1.6.1, no GPU):

| # | Command | Purpose |
|---|---|---|
| C1 | `python3 benchmarks/train/verify_paper.py` | the paper's provenance check (exact output in §5) |
| C2 | `python3 benchmarks/train/error_budget.py …E2-938-V.jsonl …E2-val.jsonl …E2-705-V.jsonl --json /tmp/eb_rerun.json` + `diff` vs stage0 | re-derive `error_budget.json` |
| C3 | `python3 benchmarks/train/gate_routed.py --val …E2-val.jsonl --test …E2-938-V.jsonl --json /tmp/gate938.json` (+ same with `E2-705-V`) | re-derive `gate_test938.json`, `gate_test705.json` |
| C4 | `python3 benchmarks/train/score_fusion.py --val …E2-val.jsonl --test …E2-938-V.jsonl --out /tmp/fus.json` | re-derive `score_fusion.json` |
| C5 | `python3 benchmarks/train/class_bias.py --out /tmp/cb.json` | re-derive `class_bias.json` |
| C6 | `python3 benchmarks/train/grounding_audit.py --pred …E2-938-V.jsonl --json /tmp/ga.json` | re-derive `grounding_audit.json` |
| C7 | `python3 benchmarks/train/paired_eval.py …E2-705-V.jsonl …E2-938-V.jsonl` | re-derive the paired table / CI |
| C8 | `python3 benchmarks/repro_manifest.py --test <artifact> --val …E2-val.jsonl` (run twice: on the hinted and on the pure artifact) | the *second* provenance instrument |
| C9 | ad-hoc numpy/`sqlite3` snippets (reported inline where used) | grounding-flag reconciliation, px-distance re-derivation, Wilson/Fisher/z, AW `index.db` aggregation |

Nothing was recomputed through the model; no file under `benchmarks/`, `papers/`, or `backup-*/` was
modified.

---

## 1. Claim-status table

Status vocabulary: **reproduces** (same number from the artifact) · **reproduces, different protocol**
(same quantity, different plumbing/checkpoint) · **cannot reproduce** (artifact exists, number does not
follow) · **not on disk** (no artifact) · **contradicted** (artifact says otherwise).

| # | Claim | Paper value | Re-derived value | Command / artifact | Status | Action |
|---|---|---|---|---|---|---|
| 1 | AC-v2 pure Pass@1 | **64.32% (256/398)** | **64.07% (255/398)** in-process; **63.82% (254/398)** served API. No 256 anywhere | C8 on `backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl`; C9 over all 41 eval JSONLs on disk | **cannot reproduce** | Restate to 64.07 (255/398); quote 63.82 as the served-API cross-check |
| 2 | AC-v2 groundB Pass@1 | 64.82% (258/398) | 258/398 = 0.6482 | `backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl` | **reproduces** | keep |
| 3 | Type accuracy | 84.17% | 335/398 = 0.8417 — on **both** the mem and the nomem artifact | C2 | **reproduces** | keep (note: identical for pure and hinted) |
| 4 | Grounding (headline row) | **56.50% (126/223)** | @938-mem **128/223 = 57.40%**; nomem **125/223 = 56.05%**; @705 **126/223 = 56.50%** | C2 on the three artifacts | **reproduces, different checkpoint** | 56.50 belongs to step 705, not to the headline checkpoint. Restate the row |
| 5 | Grounding (groundB) | 60.09% (134/223) | 223 − 58 − 31 = 134 → 60.09% | C2 on `ckpt-groundB/eval-step1406-V.jsonl` | **reproduces** | keep |
| 6 | Always-`status` / always-`click` | 21.86% / 53.27% | 0.2186 / 0.5327 | C2 prints both (gold set is identical in all 398-row files) | **reproduces** | keep |
| 7 | Uniform random over 7 types | ~14.3% | 1/7 = 14.29% (7 distinct gold types: click, status, scroll, open_app, navigate_back, type_text, wait) | C9 (`Counter(gold types)`) | **reproduces** | keep; say "7 *gold* types" (the head has 8 outputs incl. `long_press`) |
| 8 | Type-routed gate AUROC | 0.760 val / 0.729 test@938 / 0.709 test@705 | 0.76000 / 0.72860 / 0.70920 | C3; `gate_test938.json`, `gate_test705.json` — **byte-identical diff** | **reproduces** | keep |
| 9 | Certified 20% risk @ 40.7% coverage; CP upper 21.7%; empirical 17.3% | as stated | coverage 0.40704, risk 28/162 = 0.17284, `test_ub` 0.21702, `test_n` 162, `held` True | C3 (`gate` row δ=0.20) | **reproduces** — with a caveat | Test-side CP upper bound **21.7% > 20%**; the certificate is val-side (`val_ub` 0.199). Restate as "certified on val; test empirical 17.3%, CP upper 21.7%" |
| 10 | Learned-fusion risk@50% | 19.1% | 38/199 = 0.19095 | C4 (`test_risk_at_50`) | **reproduces** | keep; add ±5.5 pp (95% binomial) |
| 11 | Fusion ΔAUROC not significant | +0.013 [−0.020, +0.043] | +0.0132, episode-clustered 95% [−0.0199, +0.0425] | C4 | **reproduces** | keep |
| 12 | Conformal non-transfer | 17.3% → 23.9% | 28/162 → 37/155 (0.1728 → 0.2387), `held` False | C3 @705 | **reproduces** | keep; the paper's own 1.5σ caveat is correct (recomputed z = 1.45) |
| 13 | Online SR, both conditions | 1/8 (12.5%), Wilson 2.2–47.1% | sr 1/8 in `task.sr`; Wilson [2.24, 47.09] | `backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/index.db` + `/tmp/awrep/{pure8,fallback8}.json` | **reproduces** | keep (underpowered — §4.3) |
| 14 | Online invalid actions | 20/71 (28.2%) → 10/68 (14.7%), "**non-overlapping Wilson**" | 20/71 = 0.2817, 10/68 = 0.1471. Wilson [19.04, 39.54] and [8.19, 25.00] → **overlap on [19.0, 25.0]**; Fisher exact p = 0.0647 | C9 on `index.db` (steps table) | counts **reproduce**, the interval claim is **contradicted** | Retract "non-overlapping"; report the overlap and Fisher p |
| 15 | Online crash rate | 2/8 (25%) both | 2/8 both | `task.crash_rate` + `episodes.env_crash` | **reproduces** | keep |
| 16 | Online model p50 | 122.9 / 124.3 ms | 122.9 / 124.3 ms | `latency.model_p50` in the two report JSONs | **reproduces** | keep |
| 17 | §5.3 miss partition | "126 grounded, 97 missed = **68** + **29**" | @938-mem: 128 grounded, 95 missed = 68 + 27. @705: 126 grounded, 97 missed = **70** + 27. nomem: 125, 98 = 71 + 27 | C2 buckets A/B vs C9 grounding flags | **contradicted** — no checkpoint yields 97 = 68 + 29 | Restate per checkpoint |
| 18 | §5.3 px distances | median 437 px, mean 663, 86.8% >150 px, 4.4% ≤50 px, median y-bias 0.5 px | 437.0 / 663.1 / 86.8% (59/68) / 4.4% (3/68) / median signed dy = 0.50 | C9 (20-line numpy on `E2-938-V` bucket A) | **reproduces — from the hinted artifact**; pure artifact gives 417.7 / 649.4 / 85.9% / 4.2% / 0.60 | Label the source run, or recompute on the pure artifact |
| 19 | §5.3.1 candidate coverage | **1.0000** (n=398) | 216/223 = **0.9686** (`coverage` field: 216 True, 7 False, 175 null) | C9 on `benchmarks/results/d12bax-acv2-d12bax-2026-10-28.jsonl` | **contradicted** | Retract 1.0000; restate 96.9% (216/223) |
| 20 | §5.3.1 a11y grounding / step acc | 50.0% (n=204) / 28.39% | 102/204 = 0.5000 / 113/398 = 0.2839 | C9 on the same file | **reproduces** | keep |
| 21 | §5.5 procedural-memory ablation | 58.79% (−6.0) | 234/398 = 0.58794 | `backup-2026-10-29/GPU_HOST_A/acwork/runs/R2a-V-traj/eval-step938-V.jsonl` | **reproduces** | keep (`−6.0` is vs 64.82; it is −5.3 vs the pure 64.07) |
| 22 | §5.6 latency | 103–108 ms/step | p50 103.0 (938) / 108.1 (705) | C9 on `latency_ms` | **reproduces** | keep |
| 23 | §5.2 ECE / AUROC / selective table | 0.231 / 0.204; 0.626 / 0.631; 96.5→41.0% cov | 0.2308 / 0.2038; 0.6255 / 0.6315; 96.5/34.6, 79.9/31.1, 67.6/29.0, 51.0/26.1, 41.0/24.5 | C9 on `conf` | **reproduces** | keep |
| 24 | §5.2 "p×box ECE ~0.29–0.36" | 0.29–0.36 | 0.3669 (@705), 0.4047 (@938) | C9 | **cannot reproduce** (understated) | Restate 0.37–0.40 |
| 25 | §5.4 validation split | 62.76% (519/827), 137 episodes | 519/827, 137 unique `episode_id` | C9 | **reproduces** | keep |
| 26 | Type vs grounding error counts | 63 vs 68, z≈1.4, p≈0.16 | 63 vs 68 (@938-mem); multinomial SE 0.0209 → z = 1.39 | C2 | **reproduces** | keep |
| 27 | `verify_paper.py` | "every published number matches the saved artifacts" | 69/69 PASS, exit 0 | C1 | **reproduces** but **overclaims scope** | Reword the final line; see §4.4 |
| 28 | Paired Δ CI (saved `paired_705_vs_938.txt`) | [+1.01, +6.28] | **[+0.76, +6.35]** (deterministic across 4 runs, `PYTHONHASHSEED` 0/1/7) | C7 | stale artifact (point estimate, b/c, p all reproduce) | Regenerate the `.txt`; the paper's [+0.76, +6.33] is also 0.02 off |

---

## 2. Must retract or restate

**R1 — `64.32% (256/398)`, everywhere it appears** (`papers/single-pass-decision-v2` §abstract, §5.1
Table 1, §5.4, §6, §7; `papers/calibrated-decision-interface-2026.md` §2.2 and abstract; `notes/GOAL.md`
§2/§4; `docs/HANDOFF.md`). No artifact on disk yields 256/398. The two pure-protocol artifacts give
255/398 (in-process) and 254/398 (served API). This is a *1–2 row* drift, so no qualitative conclusion
changes — 64.07 vs GUI-Libra's 64.30 is still a statistical tie — but the number is not provenance-backed
and the paper's central integrity claim is that its numbers are. **Restate to 64.07% (255/398)**, with the
served-API cross-check (63.82%, 254/398) named. Also fix "the same checkpoint scores 256/398 on the pure
protocol (a 2-row difference attributable to the app-name hint)": the artifact gap is **3 rows**
(255 → 258).

**R2 — `Grounding 56.50% (126/223)` as the headline row's grounding.** 126/223 is reproducible — from
**step 705** (`E2-705-V.jsonl`: 126 hits, bucket A 70, bucket B 27; the 2026-10-29 results note itself
labels it "E2@705 … Ground 56.50"). The 938 checkpoint it is printed next to is 128/223 (57.40, hinted) or
125/223 (56.05, pure). Table 1 therefore mixes two checkpoints in one row. **Restate**: pure row →
56.05% (125/223); hint row → 57.40% (128/223); if 56.50% is kept, move it to a 705 row.

**R3 — §5.3's `126 grounded, 97 missed = 68 + 29`.** Arithmetic that matches no artifact (95 = 68 + 27
for the hinted 938 run; 97 = **70** + 27 for 705; 98 = 71 + 27 for pure). Looks like the 126/97 pair from
@705 merged with the 68-row element-wrong set from @938. Restate per checkpoint.

**R4 — §5.3.1 `candidate coverage 1.0000`.** The artifact's own `coverage` field is 216 True / 7 False /
175 null over the 398 rows, i.e. **96.86%** of the 223 spatial rows (the companion `-va` file: 209/216 =
96.76%). The argument ("candidate generation is not the bottleneck") survives; the number does not. Note
also that the four-digit precision `1.0000` is not attainable from any denominator in the file.

**R5 — §5.7 "non-overlapping Wilson intervals" for 28.2% → 14.7%.** The printed intervals
(19.0–39.5 and 8.2–25.0) **overlap by 6 points**; Fisher exact p = 0.0647, two-proportion z = 1.96
(p ≈ 0.049 normal, 0.065 exact). The direction is right, the significance claim is not. Restate as
"point estimate roughly halves; intervals overlap; p ≈ 0.06".

**R6 — §5.7 "the two conditions differ in exactly one thing".** Verified from the report headers: the
configs differ only in `policy.allow_app_fallback` (plus the `diagnostic` label), but the **task
instances do not match** — 5 of 8 episodes carry different randomised parameters between `pure8` and
`fallback8` (e.g. ContactsAddContact `Santiago Pereira` vs `Ahmed Zhang`; ClockTimerEntry 22:40:10 vs
14:25:17; MarkorCreateNote `calm_cat_2023_09_08.txt` vs `2023_10_11_hot_horse.md`). The 4-task pair has
the same defect (2/4 differ). This breaks the AW_PROTOCOL rule "baselines must be run on the identical
instance tuples" and confounds the invalid-action comparison. Note the replay machinery *works* —
`gb_run1`/`gb_run2` are 5/5 identical — it just was not used for this ablation.

**R7 — §5.2(b)-adjacent claim `p×box ECE ~0.29–0.36`.** Measured 0.367 (@705) and 0.405 (@938). The
direction of the claim ("worse, not better" than 0.204–0.231) holds; the stated range is understated.

**R8 — the `verify_paper.py` success line.** "ALL CHECKS PASS — every published number matches the saved
artifacts" is false as written: the script checks 69 assertions against
`papers/calibrated-decision-interface-2026.md`, all of them against **derived stage-0 JSON**, and it never
asserts the pure-protocol row (64.32/256, 56.50/126), the trivial baselines, the px-distance table, the
a11y diagnostic, the latency table, or any online number. Reword to "every stage-0 number checks out";
also fix `notes/GOAL.md` C3 ("46 assertions" → 69).

**R9 — `benchmarks/results/2026-11-03-stage0/paired_705_vs_938.txt`.** Its bootstrap CI [+1.01, +6.28]
does not reproduce; the current script deterministically returns [+0.76, +6.35]. Regenerate (free, <1 s)
or annotate as superseded.

**R10 — `benchmarks/repro_manifest.py`'s artifact classifier is circular.** `artifact_kind` is decided by
`correct == 256` / `correct == 258` — i.e. the instrument recognises the artifact by the value it is meant
to verify. Since the real pure artifact is 255, the manifest labels it `unknown` and then emits six
`MISMATCH` lines (§3). Key it on the run's `harness`/`memory` metadata instead.

---

## 3. Safe to keep

- **Everything in `benchmarks/results/2026-11-03-stage0/` re-derives byte-identically** from
  `2026-10-29-evals/{E2-938-V,E2-705-V,E2-val}.jsonl`. Verified by `diff` on all six JSONs (error_budget,
  gate_test938, gate_test705, score_fusion, class_bias, grounding_audit) — "IDENTICAL" in every case.
  The only exception is the human-readable `paired_705_vs_938.txt` CI (R9).
- **groundB row** (64.82 / 82.16 / 60.09) — all three reproduce from `ckpt-groundB/eval-step1406-V.jsonl`.
- **Type 84.17%** — 335/398 on both variants.
- **Gate AUROC trio** 0.760 / 0.729 / 0.709, AURC 0.209 / 0.2006 / 0.2441, the within-bucket AUROCs
  (spatial `box_conf` 0.793, non-spatial `p(action)` 0.618 vs product 0.504), the risk floor (21.0% @
  50.3%, 17.5% @ 40.2%, 12.5% @ 20.1%), "top-20 val rows contain 2 errors", and `cp_upper(0,22,0.10) ≤ 0.10`.
- **The unattainability result**: no certifiable threshold at δ = 0.10 (val frontier `null`), CRC at
  δ=0.10 degenerating to 3.5% coverage / 14 accepted steps. This is the paper's strongest negative and it
  is artifact-solid.
- **Fusion table** (val 0.767 / OOF 0.758 / test 0.743; risk@50% 0.191; AUPRC 0.829) and the
  non-significance verdict.
- **CRC vs CP** (45.5% vs 38.4% at δ=0.20), TV 0.106, ESS 752/827.
- **ECE/AUROC and the whole selective-prediction table** in §5.2(a).
- **Validation split** 62.76% (519/827, 137 episodes).
- **§5.5 ablation** 58.79%, **§5.6 latency** 103–108 ms, **§5.3.1** 50.0% / 28.39% / 0 open-`status`.
- **Online counts**: SR 1/8 both, crash 2/8 both, invalid 20/71 and 10/68, ASR 71/71 and 68/68, model p50
  122.9 / 124.3 ms, the 4-task pair (0/4 both; 50% → 27.0%), and Gate A ("32 passed, 1 skipped", verified
  verbatim in `backup-2026-11-06/reports/aw_reports.tgz:gate_a.log`) and Gate B (5/5 identical episodes,
  params equal across `gb_run1`/`gb_run2`).

---

## 4. Underpowered (survives as a point estimate, cannot carry the claim)

1. **Online SR (1/8).** Wilson [2.2, 47.1]. Includes exactly one success, on `TurnOnWifiAndOpenApp` —
   the **only** task whose instance parameters are identical in both conditions, and the only goal that
   states the app name verbatim. 2 of 8 episodes crashed in both conditions, so the effective denominator
   is 6. Nothing further should be read from it.
2. **The invalid-action comparison.** 13.5 pp on 139 actions, Fisher p = 0.065, on **unmatched
   instances** (R6). It is a hypothesis, not an effect.
3. **The online environment-latency claim (1066–1071 ms/step, §5.6/§5.7 and `notes/2026-11-06-…`).**
   `repro_manifest` cannot be pointed at the 5-task run (no `index.db` was backed up for `gb_run1/2`), and
   the two run reports say **`env_p50` 332.2 and 327.6 ms**. The only artifact near 1066 is `pure4.json`
   (1064.8 ms), a different suite. Also: per-episode env medians span **95.8 → 1466 ms** (pure8), so
   "the environment dominates by ~10×" is task-dependent — on `MarkorAddNoteHeader` the model (122 ms) is
   *slower* than the env (96 ms). Mark as unverified and task-scoped.
4. **"Certified 20% at 40.7% coverage."** The certificate comes from the val split; on test the CP upper
   bound is 21.7%, i.e. it does not certify the budget it is quoted against (R/row 9). And 40.7% of 398
   is 162 accepted steps: the coverage itself has a 95% binomial half-width of ±4.9 pp.
5. **Fusion risk@50% (19.1%).** 38/199, 95% ±5.5 pp → [13.6, 24.6]. The interval contains both the
   hand-routed baseline (22.6) and the programme's 10% target, so it supports neither a win nor a target.
6. **Conformal non-transfer.** z = 1.45 by the script's own arithmetic; the paper already says 1.5σ and
   the project's own note says a gap under ~2.5 sd "is not evidence of a real difference". What survives
   is the *certificate* failure (`held=False`, test CP upper 28.8% > 20%), not a demonstrated risk change.
7. **Eleven online numbers vs AW_PROTOCOL.** The protocol makes `tokens_generated` mandatory beside every
   number; **no field in `pure8.json`/`fallback8.json` records it** (the "0 generated tokens" line is a
   design property, not a measurement here). `sr_reachable`, `coverage` and `grounding_error_proxy` are
   present but all `null`/0-denominator, so `SR_all` vs `SR_reachable` cannot be split. And no matched
   `wait` / `random` / `m3a` / `d1_zero_shot` controls exist for the 8-task suite (the `wait`/`random`
   controls in `benchmarks/results/2026-11-02-online-aw-smoke.md` are a 6-task, 8-step-cap suite).

---

## 5. Exact `verify_paper.py` output

Command: `python3 benchmarks/train/verify_paper.py` (exit code 0)

```
== §2.2 / Table 1 — accuracy and error budget
  [PASS] test n                                                     got=         398 want=398
  [PASS] test accuracy                                              got=0.6482412060301508 want=0.6482
  [PASS] test bucket A (element wrong)                              got=          68 want=68
  [PASS] test bucket B (spatial type wrong)                         got=          27 want=27
  [PASS] test bucket C1 (non-spatial type wrong)                    got=          36 want=36
  [PASS] test bucket C2 (non-spatial value wrong)                   got=           9 want=9
  [PASS] test oracle grounding only                                 got=0.8190954773869347 want=0.8191
  [PASS] test oracle type only                                      got=0.8065326633165829 want=0.8065
  [PASS] val n                                                      got=         827 want=827
  [PASS] val bucket A                                               got=         137 want=137
  [PASS] val bucket B                                               got=          52 want=52
  [PASS] val bucket C1                                              got=         109 want=109
  [PASS] val bucket C2                                              got=          10 want=10

== §3.4 / Table 2 — grounding audit
  [PASS] audit spatial joined                                       got=         198 want=198
  [PASS] audit exact hit rate                                       got=0.6616161616161617 want=0.6616
  [PASS] audit misses differ (different element)                    got=          54 want=54
  [PASS] audit misses subtree                                       got=          12 want=12
  [PASS] audit misses same_label                                    got=           1 want=1
  [PASS] audit tolerant hit rate                                    got=0.7272727272727273 want=0.7273
  [PASS] audit ambiguous rows                                       got=          19 want=19

== §4 / Table 3 — score AUROC
  [PASS] AUROC global val                                           got=0.6769762029877637 want=0.677
  [PASS] AUROC global test@938                                      got=0.6494462901439646 want=0.649
  [PASS] AUROC routed val                                           got=0.7600030027775693 want=0.76
  [PASS] AUROC routed test@938                                      got=0.7285991140642304 want=0.729
  [PASS] AUROC routed test@705                                      got=0.7092026825633383 want=0.709
  [PASS] AURC routed val                                            got=0.20897281397533574 want=0.209
  [PASS] AURC routed test@938                                       got=0.2006101364733357 want=0.2006
  [PASS] AURC global test@938                                       got=0.2441429618693565 want=0.2441

== §5 / Table 4 — conformal risk control
  [PASS] gate tau                                                   got=       0.527 want=0.527
  [PASS] gate val coverage                                          got=0.47642079806529625 want=0.476
  [PASS] gate test coverage                                         got=0.40703517587939697 want=0.407
  [PASS] gate test risk                                             got=0.1728395061728395 want=0.173
  [PASS] gate test CP upper                                         got=0.2170187841482096 want=0.217
  [PASS] gate test n accepted                                       got=         162 want=162
  [PASS] gate held on test@938                                      got=        True want=True
  [PASS] no 10% threshold exists                                    got=        True want=True

== §6.1 — the risk floor forbids the 10% budget
  [PASS] frontier delta=0.20 coverage                               got=0.47642079806529625 want=0.476
  [PASS] frontier delta=0.15 not certifiable                        got=        True want=True
  [PASS] frontier delta=0.10 not certifiable                        got=        True want=True
  [PASS] CP bound at 0 errors needs n>=22 for 10%                   got=        True want=True

== §6.2 — threshold does not transfer across checkpoints
  [PASS] same tau on @938 holds                                     got=        True want=True
  [PASS] same tau on @705 violates                                  got=       False want=False
  [PASS] gate test@705 risk                                         got=0.23870967741935484 want=0.239
  [PASS] gate test@705 coverage                                     got=0.38944723618090454 want=0.389

== §4.4 / Table 3b — learned fusion (reported as NOT claimed)
  [PASS] fusion val AUROC                                           got=0.7672472036633886 want=0.767
  [PASS] fusion test AUROC                                          got=0.7434939091915836 want=0.743
  [PASS] routed test AUROC (for the delta)                          got=0.730343300110742 want=0.73
  [PASS] fusion test risk@50%                                       got=0.19095477386934678 want=0.191
  [PASS] routed test risk@50%                                       got=0.22613065326633164 want=0.226
  [PASS] delta AUROC is NOT significant                             got=       False want=False

== §5.1 / Table 4b — CRC vs Clopper-Pearson
  [PASS] CP 20% test coverage                                       got=0.3844221105527638 want=0.384
  [PASS] CRC 20% test coverage                                      got=0.4547738693467337 want=0.455
  [PASS] CRC 20% holds                                              got=        True want=True
  [PASS] CP 10% not certifiable                                     got=        True want=True
  [PASS] CRC 10% degenerate coverage                                got=0.035175879396984924 want=0.035
  [PASS] CRC 10% accepted steps                                     got=          14 want=14
  [PASS] val/test TV distance                                       got=0.1063479428581845 want=0.106
  [PASS] effective calibration size                                 got=751.6627024219737 want=752

== §3.5 / Table — Stage-2 diagnosis (per-class logit adjustment)
  [PASS] class-bias val unadjusted type acc                         got=0.8053204353083434 want=0.8053
  [PASS] class-bias val adjusted type acc                           got=0.8210399032648126 want=0.821
  [PASS] class-bias test unadjusted type acc                        got=0.8417085427135679 want=0.8417
  [PASS] class-bias test adjusted type acc                          got=0.8140703517587939 want=0.8141
  [PASS] class-bias test regressed rows                             got=          18 want=18
  [PASS] class-bias test fixed rows                                 got=           7 want=7
  [PASS] class-bias McNemar p                                       got=      0.0433 want=0.0433

== §3.2 — paired checkpoint comparison
  [PASS] paired delta (705->938)                                    got=0.035175879396984966 want=0.0352
  [PASS] McNemar b                                                  got=           8 want=8
  [PASS] McNemar c                                                  got=          22 want=22
  [PASS] McNemar p                                                  got=       0.016 want=0.016

==============================================================================
ALL CHECKS PASS — every published number matches the saved artifacts.
```

**Count: 69 PASS, 0 FAIL, exit 0.** `notes/GOAL.md` C3 says "46 assertions" — stale.

**The other instrument, run on the pure artifact, exits 1:**

```
$ python3 benchmarks/repro_manifest.py \
    --test backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl \
    --val  backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-val/eval-E2final-V-val.jsonl
artifact: backup-2026-10-29/.../E2-nomem/eval-E2final-nomem-V.jsonl
  n=398  correct=255  Pass@1=64.07%  95% CI [59.2, 68.6]
  type=84.17%  grounding=56.05% (223 spatial rows)
  ECE=20.1%  AUROC(conf)=0.642  AUROC(routed)=0.737
  risk@50% coverage: routed=20.6%  conf-only=24.6%
  error budget: {'A_spatial_type_ok_element_wrong': 71, 'B_spatial_type_wrong': 27,
                 'C1_nonspatial_type_wrong': 36, 'C2_nonspatial_value_wrong': 9}
  named: {'premature_status (acted as done, should act)': 21, 'spurious_navigate_back': 10,
          'gold_scroll_missed': 19, 'open_app_wrong_value': 6, 'gold_click_missed': 94}

artifact kind: unknown (255/398)

claim                                  paper   recomputed   status
--------------------------------------------------------------------------
Pass@1 (pure protocol)                  64.32      64.07   MISMATCH
Pass@1 (app-name hint)                  64.82      64.07   MISMATCH
Type accuracy                           84.17      84.17   MATCH
Grounding (spatial)                     56.50      56.05   MISMATCH
p(action) ECE                           20.40      20.13   MISMATCH
AUROC(confidence, correct)               0.63       0.64   MISMATCH
Gate AUROC (type-routed, test)           0.73       0.74   MISMATCH
Risk at 50% coverage, %                 21.50      20.60   DEFINITION
Certified coverage at 20% risk, %       40.70      47.00   DEFINITION
EXIT=1
```

(Note `artifact kind: unknown` — R10. Note also that on the *pure* artifact the gate AUROC is 0.737, i.e.
*better* than the 0.729 quoted from the hinted one; the calibration claim is safe either way, but the
exact figure is artifact-dependent.)

---

## 6. What is blocked, and the cheapest unblock

| Blocked claim | Exactly what is missing | Cheapest unblock |
|---|---|---|
| 64.32% (256/398) | no artifact yields 256. Nearest: 255 (`E2-nomem`, in-process) and 254 (served API) | **nothing to run** — re-quote 64.07 (255/398). If 256 must be defended, the original in-process eval of `ckpt-E2-final` with memory off must be re-run; `benchmarks/train/eval_d1.py --ckpt backup-2026-10-29/ckpt-E2-final/ckpt --track V --tag pure` (GPU, ~1 min per the project's own "~52 s" estimate) |
| Grounding for the headline checkpoint | nothing — it is on disk (125/223 pure, 128/223 hinted) | re-quote; no run needed |
| §5.3's 97/68/29 partition | nothing — every needed count is derivable | recompute with `error_budget.py` + a 10-line flag count; no run needed |
| "non-overlapping Wilson" | nothing | recompute Wilson/Fisher; no run needed |
| instance-matched ablation | `fallback8` was run on fresh draws; no run directory with the pure8 params was kept | re-run 8 tasks with the `pure8` params replayed through the Gate-B mechanism: ~450–570 s of emulator time + one model server |
| env p50 1066–1071 ms for the 5-task run | no `index.db` (steps table) for `gb_run1`/`gb_run2`; only the two report JSONs, which say 332.2 / 327.6 | re-run the 5-task suite (~520–545 s), or quote the numbers the two reports actually contain |
| `tokens_generated = 0` per AW_PROTOCOL | the reports have no token field at all | add the field to the report schema (code change, no run) |
| online controls (`wait`, `random`, `m3a`, `d1_zero_shot`) on identical instances | never run on the 8-task suite | 8 tasks × 4 agents ≈ 4 × 570 s emulator time |

---

## 7. Findings (detail)

**7.1 The stage-0 chain is honest end-to-end.** All six derived JSONs in
`benchmarks/results/2026-11-03-stage0/` reproduce **byte-identically** (via `diff` on sorted-key JSON) when
the documented commands are re-run on the three saved prediction files [1][2][3][4][5][6]. `verify_paper.py`
then asserts 69 published values against those JSONs and passes 69/69 [7]. So the pipeline
`jsonl → script → JSON → paper` is verified for the calibration paper — with the crucial exception that the
*input* JSONL is the app-name-hinted run (258/398), not the pure run.

**7.2 The pure headline is not on disk — but a pure artifact is.** Searching every eval JSONL in the repo
(41 files with ≥300 scored rows, including `backup-2026-10-29/` and `backup-2026-11-06/`) finds **no file
with 256 correct** [8]. Three files are close:

| artifact | n correct | acc | type | grounding | bucket A |
|---|---|---|---|---|---|
| `…/E2-nomem/eval-E2final-nomem-V.jsonl` (memory off = no app-name hint) | **255** | 64.07 | 0.8417 | 125/223 = 56.05 | 71 |
| `…/E2-nomem/eval-E2final-mem-V.jsonl` (= the canonical `E2-938-V.jsonl`, md5-identical to `…/E2-cur15k-V/eval-step938-V.jsonl`) | 258 | 64.82 | 0.8417 | 128/223 = 57.40 | 68 |
| `backup-2026-11-06/offline/e2-pure-hist.jsonl` (served-API regeneration, `harness: "served-api"`, `history` set) | **254** | 63.82 | 0.8392 | 125/223 = 56.05 | — |
| `backup-2026-11-06/offline/e2-pure.jsonl` (same regeneration **without** history) | 132 | 33.17 | 0.5000 | 89/223 | — |

The no-history row is a useful warning: dropping the interaction history costs 30 points, which is why
`regen_pure.py`'s docstring insists on `previous_actions`.

The pure/hint gap is therefore **3 rows** (255 → 258), not the 2 rows the paper reports, and the
"grounding gain" attribution in `2026-11-02-d1-acv2-groundB.md` ("wrong-place misses 71 → 58") lines up
exactly with the *nomem* bucket A (71) and groundB's (58) — confirming `E2-nomem` is the run the groundB
comparison was made against, i.e. the pure protocol.

**7.3 Table 1 mixes checkpoints.** Grounding 126/223 is step 705's value (705: bucket A 70 + bucket B 27;
223 − 97 = 126). Step 938's values are 128 or 125 depending on the hint. The same mixture appears in
§5.3 (`97 = 68 + 29`), which is why the §5.3 arithmetic is uncheckable against any one artifact. The
px-distance statistics in §5.3 (437 / 663 / 86.8 / 4.4 / 0.5) *do* reproduce exactly, but only from the
**hinted** 938 artifact's 68 bucket-A rows; the pure artifact gives 417.7 / 649.4 / 85.9 / 4.2 / 0.60.

**7.4 The grounding metric has four denominators in circulation.** For the same underlying behaviour:
`126/223 = 56.50%` (@705, "leaderboard definition"), `128/223 = 57.40%` (@938 hinted), `125/223 = 56.05%`
(@938 pure), `128/198 = 64.65%` (the value `eval_d1.summarize()` itself would print, since it drops rows
where `grounding is None`), and `131/198 = 66.16%` (the audit's independent recomputation of the same
criterion). The audit and the scorer **disagree on 3 of 198 joined rows**: two type-correct spatial rows
with a valid point are stored with `grounding: null` even though the point is inside the gold node
(episodes 19003/step 2 and 13468/step 4), and one row the scorer marks as missed is re-derived as a hit
(15890/step 2). Tiny, but it means "the same number" is not reproducible across the two tools that
produce it, and the 198-vs-223 denominator gap (25 spatial rows where the model emitted a non-spatial
action and therefore no point) must always be stated — the paper states it for §3.4 but not for §5.1.

**7.5 The online loop's numbers are real; the comparison is not matched.** `index.db` and the report JSONs
reproduce every published count (SR 1/8, crash 2/8, invalid 20/71 & 10/68, ASR 71/71 & 68/68, p50 122.9 /
124.3 ms) [9][10]. What does not hold is (a) the "non-overlapping Wilson" sentence, (b) the "exactly one
thing differs" sentence — 5 of 8 instances carry different task parameters (R6), and (c) the
environment-latency figure, which no artifact in the backup contains (reports say 332.2 / 327.6; per-step
env medians swing 96 → 1466 ms by task). Also worth recording positively: the single success in both
conditions is the same task instance (`TurnOnWifiAndOpenApp`) — that is the determinism the section
claims, but it is n = 1 task, not n = 1 sample.

**7.6 The a11y diagnostic's load-bearing number is wrong, its conclusion is not.** `coverage = 1.0000` (R4)
is really 96.9%; 102/204 = 50.0% grounding and 113/398 = 28.39% step accuracy do reproduce [11]. So
"candidate generation is not the bottleneck" survives at 96.9% coverage — but the exact "100% ceiling"
phrasing, and the "third independent confirmation" framing that rests on it, need the corrected number.

**7.7 Two write-ups of the same run disagree with each other, not with disk.** `notes/GOAL.md` §4 records
"Element misses that are genuinely different elements 80.6% (17.9% subtree, 1.5% same-label)" — correct
per the audit JSON — while `benchmarks/results/2026-11-03-…md` §1 correctly says type errors (63) and
grounding errors (68) are comparable (z ≈ 1.4). Both are artifact-supported; the risk is that the paper's
§5.3 narrative ("grounding is *the* axis", "the failure is element selection") reads as more one-sided
than the numbers license on the pure run, where the split is 63 type / 71 grounding with bucket A +13 and
C1 unchanged.

---

## 8. Coverage Status

- **Checked directly (read + executed):** `verify_paper.py` (full output above); re-derivation of all six
  stage-0 JSONs byte-for-byte; `repro_manifest.py` on both the hinted and the pure artifact; the grounding
  audit recomputation and its 3-row disagreement with the saved `grounding` flags; the px-distance table;
  the `d12bax` a11y artifact; 41 eval JSONLs scanned for their (n, correct, type, grounding) signature;
  the AndroidWorld `index.db` databases (episodes + steps tables) and the two report JSONs from
  `aw_reports.tgz`; Wilson / Fisher / two-proportion / binary-SD arithmetic; `gate_a.log`.
- **Uncertain / not resolved:**
  1. *Why* 256/126 appears at all. It is one row above the in-process pure artifact and one row above the
     pure grounding count; it may come from an earlier scorer or harness revision, or from the 858-px
     pipeline. I could not find a command that produces it, and I did not attempt to "fix" it.
  2. Whether `e2-pure-hist.jsonl` is protocol-identical to `eval_d1.run_eval`. It is built by
     `regen_pure.py` against the deployed API (`D1_IMG_W=720`, same `load_image`, same scorer) and
     `serving/d1_api.py` claims to call the model "exactly as eval_d1.run_eval does", but I did not diff
     the two state-construction paths line by line, so I treat 254 vs 255 as an unexplained harness delta.
  3. The `env_p50` 1066–1071 figure: no steps table survives for the 5-task run, so the number is neither
     confirmed nor refuted — only contradicted by the report field.
  4. `§5.2(b)`'s reported-in-run numbers (isotonic ECE 0.036, 55.0%/23.3%, 36.7%/18.5%) remain
     non-reproducible; the paper already labels them as such, so I did not audit them further.
  5. Any number in `papers/*` sourced from competitor tables (GUI-Libra, Mem-W, UI-Venus) — out of scope
     for an internal audit and not re-checked against the external papers.
- **Not attempted:** any GPU reproduction, any model serving, any AndroidWorld re-run.

---

## 9. Evidence table

Locators are local paths (this is an internal audit; no network source was retrieved). Every row was read
or executed directly in this session.

| # | Source | Locator | Key claim (as read/measured) | Type | Confidence |
|---|---|---|---|---|---|
| 1 | `error_budget.json` + `error_budget.py` | `benchmarks/results/2026-11-03-stage0/error_budget.json`; `benchmarks/train/error_budget.py` | Re-run reproduces the JSON byte-identically; test n=398, acc 0.6482, buckets 68/27/36/9 | primary (derived) | high |
| 2 | `gate_test938.json` / `gate_test705.json` + `gate_routed.py` | `benchmarks/results/2026-11-03-stage0/`; `benchmarks/train/gate_routed.py` | Re-run identical; routed AUROC 0.760/0.729/0.709; δ=0.20 → test cov 0.4070, risk 0.1728, CP UB 0.2170, n=162 | primary | high |
| 3 | `score_fusion.json` + `score_fusion.py` | same dir | Re-run identical; fusion test AUROC 0.7435, risk@50% 0.1910, Δ +0.0132 [−0.0199, +0.0425] | primary | high |
| 4 | `class_bias.json`, `grounding_audit.json` | same dir | Re-run identical; type 0.8417→0.8141, McNemar p 0.0433; audit 198 joined / 131 hits / 54 different-element | primary | high |
| 5 | `paired_705_vs_938.txt` | same dir | Point estimate +3.52 pts, b=8, c=22, p=0.0161 reproduce; CI [+1.01,+6.28] does **not** (rerun: [+0.76,+6.35]) | primary, stale | high |
| 6 | `E2-938-V.jsonl`, `E2-705-V.jsonl`, `E2-val.jsonl` | `benchmarks/results/2026-10-29-evals/` | The three input artifacts; 258/398 (md5-identical to the s4090 `eval-step938-V`), 244/398, 519/827 | primary | high |
| 7 | `verify_paper.py` | `benchmarks/train/verify_paper.py` | 69/69 PASS, exit 0; scope limited to stage-0 JSON + the interface paper | primary | high |
| 8 | Pure-protocol artifacts | `backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl` (255/398); `backup-2026-11-06/offline/e2-pure-hist.jsonl` (254/398); `…/e2-pure.jsonl` (132/398, no history) | No artifact on disk equals 256/398; pure = 64.07% or 63.82% | primary | high |
| 9 | groundB artifact | `backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl` | 258/398, type 0.8216, grounding 134/223 = 60.09%, bucket A 58 | primary | high |
| 10 | AndroidWorld run DBs | `backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/index.db` | SR 1/8 both; crash 2/8 both; invalid 20/72 steps and 10/70 steps (20/71 and 10/68 executed); model p50 122.8 / 124.2 | primary | high |
| 11 | AndroidWorld reports | `backup-2026-11-06/reports/aw_reports.tgz` → `pure8.json`, `fallback8.json`, `pure4.json`, `fallback4.json`, `gb_run1.json`, `gb_run2.json`, `gate_a.log` | model p50 122.9/124.3; env_p50 294.2/135.3 (and 1064.8 for `pure4`); instance params differ 5/8; Gate A "32 passed, 1 skipped"; Gate B 5/5 identical | primary | high |
| 12 | a11y diagnostic artifact | `benchmarks/results/d12bax-acv2-d12bax-2026-10-28.jsonl` (+ `.md`) | 113/398 = 28.39%; grounding 102/204 = 50.0%; coverage 216 True / 7 False → 96.86%, **not** 1.0000 | primary | high |
| 13 | Ablation artifact | `backup-2026-10-29/GPU_HOST_A/acwork/runs/R2a-V-traj/eval-step938-V.jsonl` | 234/398 = 58.79% (the §5.5 negative ablation) | primary | high |
| 14 | `repro_manifest.py` | `benchmarks/repro_manifest.py` | Second provenance tool; exits 1 on the pure artifact with 6 MISMATCHes; classifier keys on `correct == 256/258` (circular) | primary | high |
| 15 | Paper under audit | `papers/single-pass-decision-acv2.md` §5.1–§5.7; `papers/calibrated-decision-interface-2026.md` §2.2–§6.2 | The claims being checked | secondary (self-report) | high |
| 16 | Internal bookkeeping | `notes/GOAL.md` §2–§4, C3; `docs/compute/REPRO-2026-11-06.md`; `docs/compute/AW_PROTOCOL.md`; `benchmarks/results/2026-10-29-d1-acv2-sft.md`; `benchmarks/results/2026-11-03-acv2-error-budget-and-routed-gate.md` | Prior disclosures (pure artifact missing; 46-vs-69 assertion count; "E2@705 … Ground 56.50"; mandatory AW fields) | secondary | high |

## 10. Sources

1. `benchmarks/results/2026-11-03-stage0/error_budget.json` (+ `benchmarks/train/error_budget.py`) — path: `benchmarks/results/2026-11-03-stage0/error_budget.json`
2. `benchmarks/results/2026-11-03-stage0/gate_test938.json` and `gate_test705.json` — path: `benchmarks/results/2026-11-03-stage0/`
3. `benchmarks/results/2026-11-03-stage0/score_fusion.json` (+ `benchmarks/train/score_fusion.py`)
4. `benchmarks/results/2026-11-03-stage0/{class_bias,grounding_audit}.json` (+ `grounding_audit.py`, `class_bias.py`)
5. `benchmarks/results/2026-11-03-stage0/paired_705_vs_938.txt` (+ `benchmarks/train/paired_eval.py`)
6. `benchmarks/results/2026-10-29-evals/{E2-938-V,E2-705-V,E2-val}.jsonl`
7. `benchmarks/train/verify_paper.py`
8. `backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl`; `backup-2026-11-06/offline/e2-pure-hist.jsonl`; `backup-2026-11-06/offline/e2-pure.jsonl`; `benchmarks/regen_pure.py`
9. `backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl`
10. `backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/index.db` (tables `runs`, `episodes`, `steps`)
11. `backup-2026-11-06/reports/aw_reports.tgz` (`pure8.json`, `fallback8.json`, `pure4.json`, `fallback4.json`, `gb_run1.json`, `gb_run2.json`, `gate_a.log`)
12. `benchmarks/results/d12bax-acv2-d12bax-2026-10-28.jsonl` (+ `.md`)
13. `backup-2026-10-29/GPU_HOST_A/acwork/runs/R2a-V-traj/eval-step938-V.jsonl`
14. `benchmarks/repro_manifest.py`
15. `papers/single-pass-decision-acv2.md`; `papers/calibrated-decision-interface-2026.md`
16. `notes/GOAL.md`; `docs/compute/REPRO-2026-11-06.md`; `docs/compute/AW_PROTOCOL.md`; `docs/compute/TODO-2026-11-06.md`; `benchmarks/results/2026-10-29-d1-acv2-sft.md`; `benchmarks/results/2026-11-02-d1-acv2-groundB.md`; `benchmarks/results/2026-11-03-acv2-error-budget-and-routed-gate.md`; `notes/2026-11-06-online-interface-defect.md`
