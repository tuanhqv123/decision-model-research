# R4 — Candidate-interface post-mortem: the "VA lost" finding does not survive contact with the artifacts

**Date:** 2026-11-07 · **Scope:** E3-300-VA vs E2 938-step V baseline, both 398 rows of `500_steps_filtered.json`
**Method:** all numbers recomputed from on-disk JSONL by direct execution; no number below is quoted from a prior note unless labelled `[prior]`.

---

## 0. Headline

**"The candidate interface failed" is not a finding. It is a training-budget artefact, and on the one comparison the artifacts actually support a verdict on — same checkpoint, same step, same data — the candidate interface *wins*.**

The failure is real but it is located somewhere else entirely: the **supervision label `gold_candidate()` produces is unscoreable on 5.4% of spatial rows**, and the element head is asked to learn a target the evaluation function can never award. Details in §4.

Two corrections the parent withdrew as "verified" are contradicted by the artifacts and are re-derived below:

- The comparison is **not** merely "confounded by step count." It is confounded by **step count, a different training set, and a 50/50 mixed-track schedule**, and the artifacts contain a *matched* comparison that reverses the sign (§2).
- There **is** a train/eval interface skew — not in the candidate *builder* (that part is clean) but in the **gold answer**, which is looser than the scorer on 19.7% of spatial rows (§4, defect 4).

---

## 1. Row-by-row comparison, exact counts

Key alignment verified first: all four eval files hold **398 rows, 398 unique `(episode_id, step)` keys, zero duplicates, and identical key order**. So row *i* is the same decision in every file and paired testing is valid.

| Comparison | VA correct | V correct | both | **VA-only** | **V-only** | neither | union | McNemar *p* |
|---|---|---|---|---|---|---|---|---|
| **E3-300-VA vs E2-938-V** (the headline claim) | 221/398 = 55.53% | 255/398 = 64.07% | 192 | **29** | **63** | 114 | **284 = 71.36%** | **5.1e-4** |
| **E3-300-VA vs E3-300-V** (same checkpoint) | 221/398 = 55.53% | 197/398 = 49.50% | 182 | **39** | **15** | 162 | 236 = 59.30% | **1.5e-3** |

Grounding, same-row decomposition:

| Run | grounding (defined rows) | on the 223 spatial rows |
|---|---|---|
| E3-300-VA, element path (`el_pt`) | 106/188 = **56.38%** | 106/223 = **47.53%** |
| E3-300-VA, box path (`box_pt`, same checkpoint & prompt) | — | 98/223 = **43.95%** |
| E3-300-V, box path (same checkpoint) | 85/184 = 46.20% | 85/223 = 38.12% |
| E2-938-V, box path (the baseline) | 125/196 = 63.78% | — |

Reading §1 honestly:

1. **VA loses to the 938-step baseline: 221 vs 255, −8.54 pts, McNemar p = 5.1e-4.** That is a real, statistically significant deficit *as measured*. It is not noise. But see §2 for what it is measuring.
2. **The union is 284/398 = 71.36%**, which is **+7.29 pts over the V baseline alone** and **+15.83 over VA alone**. The two interfaces are wrong on substantially disjoint rows (29 vs 63). This is the strongest single argument in the artifacts that the element readout carries information the box readout does not.
3. **The union is not deployable.** It requires the gold label to select which head to trust. The prior note `[prior]` reports an oracle of 59.05% for post-hoc combination, but that number is **unreproducible from the artifacts in this workspace** — I could not find the script or eval file that produced it, and 59.05 is *also* Qwen3-VL-32B's AC-v2 score in `notes/sota-memw.md`, which raises the possibility of a transcription error. **Treat 59.05 as unverified.** The union I can verify is 71.36%, and its deployable version (a learned gate) was never built.
4. **Critically, the "post-hoc combination does not help" claim is measured only *within* VA** (element 55.53 vs its own box 53.52). Across *checkpoints* the union is worth +7.3 pts. Those are different questions and the prior note conflated them.

---

## 2. Verdict: budget artefact, not a scientific finding

### 2.1 The two runs are not comparable on three axes at once

| Aspect | E3-300-VA (`runs/E3-cur24k-mix/args.json`) | E2-938-V (`ckpt-E2-final/args.json`, `runs/E2-cur15k-V`) |
|---|---|---|
| `steps` | **1500** (planned) — log ends at **step 320**; the eval is step **300** | **938** |
| `train_set` | **cur24k** | **cur15k** |
| `track` | **mix** (`VA` if `rng.random() < 0.5` else `V`, `train_d1.py:76-77`) | **V** |
| `lr` / `warmup` | 1e-5 / 30 | 1e-5 / 30 (matched) |
| `micro`×`accum` | 2 × 8 = 16 | 2 × 8 = 16 (matched) |
| `mem_frac` | 0.55 | 0.92 |
| `eval_every` / `eval_tracks` | 300 / `V,VA` | 235 / `V` |

So E3-300 is **300 of a planned 1500 steps (20%)**, on a **different corpus**, with the element head active on only **~50% of samples**. The V baseline is a **fully-scheduled, single-track run on a different corpus**. Comparing 55.53 to 64.07 and concluding "the interface failed" attributes to the *interface* a gap created by at least **1.4× fewer steps, one different dataset, and half the VA gradient signal**. This is a textbook confound; the brief's framing is wrong.

Note also: `save_every: 1500` with the run dead at step 320 means **E3 never saved a final checkpoint**. `runs/E3-cur24k-mix/best/` is an intermediate. Any "VA" number in existence is a mid-warmup snapshot (lr still 9.05e-6 of a 1e-5 peak at step 300 — it has not even finished decaying).

### 2.2 The artifacts contain a matched-budget comparison, and VA wins it

The same checkpoint evaluated at the same step, in the same log block, 16 minutes apart:

```
EVAL {"track": "V",  "n": 398, "step_acc": 0.495,  "grounding": 0.462,  ... "step": 300}
EVAL {"track": "VA", "n": 398, "step_acc": 0.5553, "grounding": 0.5638, ... "step": 300, "step_acc_box_only": 0.5352}
```

Row-by-row on `eval-step300-V.jsonl` vs `eval-step300-VA.jsonl`:

- V 197/398 (49.50%) vs VA 221/398 (55.53%) → **VA +6.03 pts**
- VA-only 39, V-only 15 → **McNemar p = 1.5e-3**
- grounding: 46.20% → **56.38%** (+10.2 pts)
- and the *box* readout alone on the VA prompt scores **53.52%**, i.e. **also above the V-track 49.50%** on identical weights

**A single 300-step checkpoint scores 6 points higher through the element interface than through the box interface, significantly, and its box readout is not the thing carrying the gain.** Cancelling every confound simultaneously by holding the weights fixed, the element interface is *better*, not worse.

The one thing this cannot rule out: the gain could be the **auxiliary loss** acting as a regularizer rather than the readout being better. That is testable (§5, experiment 1) and is *not* what the prior note concluded.

### 2.3 Verdict

> **Not a scientific finding.** "The candidate interface loses ~8.5 points" is an artefact of comparing a 300-step, 24k-corpus, half-strength mixed run against a 938-step, 15k-corpus pure run. The only budget-matched measurement in the artifacts shows the candidate interface **+6.03 pts (p = 1.5e-3)** ahead. The interface has never actually been tested.
>
> The defensible residual claim is narrower and still interesting: **the element head plateaus below the box head's ceiling** (its own box path on the same prompt is 53.52% vs element 55.53% step-level but 43.95% vs 47.53% on spatial rows — the element head is ahead only on spatial rows and only slightly), and **19.7% of its supervision is unscoreable** (§4). Those are two distinct, fixable defects — neither of which was what the team concluded.

---

## 3. Failure taxonomy of the element head

Population: 223 spatial rows (`click`/`long_press`/`type_text` with a gold point), 175 non-spatial rows carry no grounding score.

| Class of the chosen candidate vs the gold element | n | share of 223 |
|---|---|---|
| **Chosen contains the gold point → scored correct** | **106** | 47.5% |
| **UNRELATED** — bbox fully disjoint from the gold element | **89** | 39.9% |
| **IDENTICAL bbox, still scored wrong** | **16** | 7.2% |
| OVERLAPS gold, neither nested | 8 | 3.6% |
| Chosen is an **ANCESTOR** (container) of gold | 3 | 1.3% |
| Chosen is a **DESCENDANT** (child) of gold | 1 | 0.4% |
| scored correct but not classifiable | 0 | 0 |
| **Total** | **223** | 100% |

Three findings, in order of importance:

**(a) The dominant failure is "different element entirely," not "wrong granularity."** Of the 101 rows the element head got wrong (223 − 106 − 16 scored-wrong-but-identical... see (b)), **89 are spatially disjoint** from the gold element. Only **4 rows** (3 ancestor + 1 descendant) are hierarchical near-misses. This **kills the obvious cheap fix**: snapping the prediction to a parent/child of the chosen candidate would recover at most 4 rows = +1.8 pts. The head is not confusing the container with the child — it is tapping a box in the wrong part of the screen. That is a *retrieval/locating* failure, not a *granularity* failure.

**(b) 16 of the "element head failures" are not element failures at all — they are action-head failures.** In all 16 the chosen candidate's bbox is **byte-identical to the gold element's bbox**, yet `grounding` is `False`. Decomposition of the predicted action type:

| gold → pred | n |
|---|---|
| `click` → `status` (premature terminate) | **9** |
| `type_text` → `type_text` (right element, wrong text) | 3 |
| `click` → `open_app` | 1 |
| `click` → `type_text` | 1 |
| `type_text` → `click` | 1 |
| `click` → `navigate_back` | 1 |

`run_acv2.score_row` zeroes grounding whenever the action type mismatches (`if pred_action["action_type"] in SPATIAL and gold == pred` gate). So **the element head is being charged 16 times for the action head's mistakes**, 9 of them the known premature-`status` bucket already under investigation elsewhere. **Corrected element accuracy: 122/223 = 54.71%, not 47.53%.** The reported 47.53% understates the element head by 7.2 points. This is a metric defect, and it means the team has been optimising against a number that mixes two heads.

**(c) 6 spatial rows have no candidate at all**, scoring 0/6. That is 2.7% of spatial rows that are structurally unreachable through the interface.

---

## 4. Defect list

Each item: evidence → one-line fix.

### Defect 1 — The headline comparison is 300 steps vs 938 steps on a different corpus. **[fatal to the conclusion]**
`runs/E3-cur24k-mix/args.json`: `steps:1500, train_set:"cur24k", track:"mix"`; train log ends at **step 320**; only a step-300 eval exists; `save_every:1500` means no final checkpoint was written. `ckpt-E2-final/args.json`: `steps:938, train_set:"cur15k", track:"V"`.
Three confounds (budget, corpus, half-strength element signal) plus a missing checkpoint.
**Fix:** re-run the VA arm on the **exact E2 recipe** (`cur15k`, `track:VA`, `steps:938`, `lr 1e-5/warmup 30`, `mem_frac 0.92`) and compare to `E2-938-V` at matched `steps` only. Until that exists, the paper must not carry a VA-vs-V claim.

### Defect 2 — The matched comparison that *does* exist is never cited and points the other way.
`E3-cur24k-mix/train.log` lines 32–33 evaluate the same step-300 checkpoint on both tracks: V 49.50% / grounding 46.20%, VA 55.53% / grounding 56.38%, `step_acc_box_only` 53.52%. Recomputed per-row: 39 VA-only vs 15 V-only, McNemar p = 1.5e-3. Same weights, same data, same step, same 398 rows.
**Fix:** quote this pair as the budget-matched result and restate the VA verdict as "untested at 938 steps," not "failed."

### Defect 3 — The grounding metric charges the element head for action-head errors (16 rows / 7.2 pts).
16 rows in E3-300-VA have a chosen bbox **identical** to the gold element's bbox yet score `grounding=False`; 9 are `click`→`status`. `score_row` gates the spatial branch on exact action-type equality, so grounding collapses to `False`.
**Fix:** report element accuracy as "element correct **given the gold action type**" (122/223 = 54.71%) alongside the joint number, and split the 9 premature-`status` rows into the action bucket where they belong.

### Defect 4 — **Train/eval label skew, and it is fatal in the same way the builder skew would have been.** The candidate *builder* is clean; the *gold answer* is not.
`ac_common.gold_candidate()` (line 85) returns the candidate that **is** the gold element if present, **else the smallest candidate that merely contains the gold point** — "[cands are smallest-first]". `compact_ac.py` and `eval_d1.py` both call it, so there is no builder skew. But the *scorer* (`run_acv2.score_row`) requires the predicted point inside the **smallest a11y node containing the gold point**, which is strictly tighter than "any candidate containing the gold point."

Measured on the 223 spatial eval rows:

| Supervision quality issued by `gold_candidate()` | n | grounding |
|---|---|---|
| **Tight** — gold bbox *is* a candidate | 173 (77.6%) | 86/173 = 49.71% |
| **Loose** — falls back to a **container** | **44 (19.7%)** | 20/44 = 45.45% |
| **None** — no candidate contains the gold point | 6 (2.7%) | 0/6 = 0% |
| gold *point* inside ≥1 candidate (the claimed 97.2% ceiling) | 217/223 = 97.31% | — |

And the killer: on **12 of the 44** loose rows, the fallback container's **centre does not even lie inside the gold element**. A model that perfectly imitates its own training label is therefore *guaranteed* `grounding=False` on those 12 rows — the training target is **unscoreable by construction**.

**Supervised ceiling = (223 − 6 − 12)/223 = 205/223 = 91.93%**, not 97.2%. The frequently-quoted 97.2% is the *point-inside-some-candidate* ceiling, which is a different (and unreachable) quantity.
**Fix:** in `gold_candidate()`, drop the containment fallback and **supervise `target_el` only when the bbox matches exactly** (173/223 rows), letting the box head carry the remaining 50 rows. Alternatively make the fallback emit the *deepest* candidate inside the gold node rather than the smallest containing it, and re-check that its centre is in the gold node before issuing the label.

### Defect 5 — `el_pt` is a lossy reconstruction; `box_pt` is silently overwritten.
`eval_d1.py:103` sets `pt = el_pt` for spatial actions, so the record's `pt` is the element choice while `box_pt` is the box centre. On the same checkpoint the two prompts disagree on **397/398 rows** with a **median 73 px / mean 194 px** displacement — i.e. the VA prompt is *not* a superset of the V prompt behaviourally; adding the candidate list moves the box head's output a lot. That makes "VA vs V" a comparison of two different models' behaviour, not a readout swap on one model.
Relatedly, `el_pt` records only the **centre**, not the chosen `idx`; I had to reconstruct the choice by matching centres against the rebuilt candidate list, and **centre collision between nested candidates makes that ambiguous** (bubbles and their children often share a centre).
**Fix:** log `el_idx` (and `box_only_pt`) verbatim in the eval record. Without it, every element-head post-mortem requires a brittle rebuild — which is exactly why the 16 Defect-3 rows went unnoticed.

### Defect 6 — Candidate ORDER: hypothesis tested and **refuted** (report as a negative result).
`pick_candidates` sorts smallest-first. If the model were positionally biased the task would be rigged. It is not:

| | median relative rank | p90 relative rank | median absolute idx | share in first 3 positions |
|---|---|---|---|---|
| **Gold** candidate (n=173) | 0.500 | 0.909 | 14 | 23/173 = 13.3% |
| **Model-chosen** candidate (n=398) | 0.537 | 0.887 | 12 | 52/398 = 13.1% |

The chosen candidate's positional distribution tracks the gold's almost exactly (13.1% vs 13.3% in the top 3). **There is no early-index bias.** Positional/order priors are not the problem, and re-ordering the list is not a fix.
**Caveat (measured, benign):** gold candidates sit deep (median rank 0.50, p90 0.91) because small leaf elements sort last — the list is ordered *against* the prior, if anything.

### Defect 7 — Candidate count: hypothesis tested and **refuted**; truncation is not the cheap fix.
N per row: **median 24, mean 29.6, min 1, max 245**, zero empty rows.

| N bucket | rows | step accuracy |
|---|---|---|
| 0–10 | 58 | 34/58 = **58.6%** |
| 11–20 | 93 | 52/93 = 55.9% |
| 21–40 | 161 | 89/161 = 55.3% |
| 41+ | 86 | 46/86 = **53.5%** |

**No collapse point.** The whole span from ≤10 to 41+ candidates (a 25× range in N) costs **5.1 points**, monotone and shallow. Truncating to the smallest 10 would, by this curve, be worth a handful of points on a minority of rows while destroying the 40% of rows that need a long list.
**Fix:** do not truncate. If anything, spend the budget on a two-stage retrieval **only if** §3(a)'s "wrong element entirely" pattern is locality-driven — but the N curve says capacity is a second-order effect, so this is a low-priority experiment, not the fix.

### Defect 8 — The element head shows no sign of having trained at all on this run; the run is a mid-warmup snapshot.
`acc_target_el` over the log: 0.3158 (s10) → 0.42 (s60) → 0.53 (s180) → **0.7143 (s310)** → 0.5349 (s320). It is still climbing steeply at the point the run dies, while `acc_action` has flattened at 0.77–0.81. The element head is the **least converged head in the model** at step 300 — which is the single most likely explanation for its step-level deficit against a 938-step box head.
**Fix:** do not read any capability conclusion off a 300/1500-step checkpoint. The learning curve for `target_el` must be extended to ≥938 before the head is judged.

---

## 5. What would actually settle it (ranked, cheapest first)

1. **Readout-only swap on the finished checkpoint (~0 GPU-hours, 1 eval).** Run `--track VA` on `ckpt-E2-final` (the 938-step V model) with the *existing* candidate builder, no retraining. If VA ≥ V on that checkpoint, the interface is exonerated outright and the whole episode was a budget artefact. **This is the highest-value experiment in the R4 space and it appears never to have been run.** (Caveat: the model was never trained with `target_el`, so its element head is untrained — expect chance. The informative variant is (2).)
2. **VA arm on the exact E2 recipe (1 × RTX 4090, ~1.6 h per `notes/GOAL.md`).** `cur15k`, `track:VA`, `steps:938`, everything else identical, `eval_tracks: V,VA`. This is the only run that can produce a VA-vs-V number.
3. **Fix Defect 4, then repeat (2).** Exact-match-only `target_el` supervision. Predicted effect: +~3–5 pts of element accuracy purely from removing unscoreable labels.
4. **Log `el_idx` (Defect 5) in all future eval records** — prerequisite for any element-head diagnosis that is not a rebuild.
5. **Only then** consider two-stage retrieval; deprioritise parent/child snapping (ceiling +1.8 pts) and list truncation (ceiling ~+5 pts on a minority), both refuted above.

---

## Coverage Status

**Checked directly (executed against on-disk artifacts):**
- All 4 eval JSONLs: row counts, key uniqueness, key-order identity across files.
- Every accuracy, grounding, count and McNemar *p* in §1 and §2 — recomputed, not quoted.
- `args.json` for `E3-cur24k-mix`, `E2-cur15k-V` (`ckpt-E2-final`), `ckpt-groundB`; full E3 train log including both step-300 EVAL lines and the `acc_target_el` series.
- Candidate lists rebuilt from `benchmarks/data/acv2/500_steps_filtered.json` via the real `ac_common.pick_candidates` (imported, not reimplemented) — N distribution, gold/chosen ranks, failure taxonomy, tight/loose/none supervision split.
- Read `ac_common.py` (`pick_candidates`, `gold_candidate`, `build_state`, `build_questions`, `gold_answers`), `compact_ac.py`, `eval_d1.py`, `train_d1.py:70-90`, `run_acv2.py:160-215`.

**Uncertain / could not complete:**
- **`oracle 59.05` is unverified.** No script or eval file in this workspace produces it; the value coincides with Qwen3-VL-32B's AC-v2 score in `notes/sota-memw.md`. I did not treat it as evidence.
- **"mixed V+VA training −2 to −4 points" is unverified** — I found no run that isolates it. E3 is `track:mix` but has no pure-VA or pure-V sibling at matched budget, so the claim cannot be checked from artifacts.
- **The 44-row loose-label quantification is on the *test* set**, not the training set, because `compact_cur24k.jsonl` was not scanned in full (size). If the training split's a11y trees are shaped differently the 19.7% figure will move; the *mechanism* (containment fallback vs strict scorer) is read directly from source and is split-independent.
- **Why 12 container centres fall outside the gold node** is not diagnosed. Plausible causes (stale a11y bbox vs screenshot re-scale, `bbox_pixels` coordinate space) are hypotheses only — not verified.
- Did **not** verify that `compact_cur24k.jsonl`'s stored `cands` are byte-identical to a fresh `pick_candidates` run on the same records. The builder is the same function (read directly), and stored candidate shape `[bbox, idx, label, pos]` matches eval-time shape, so structural skew is ruled out; byte-identity of the *content* is not.
- Non-spatial rows (175) are excluded from every grounding figure; all percentages in §3 are over 223 unless stated.

---

## Evidence table

Every row is a primary artifact in this workspace, read or executed directly. Paths are relative to the repo root.

| # | Source | Path | Key claim | Type | Confidence |
|---|---|---|---|---|---|
| 1 | `benchmarks/results/2026-10-29-evals/E3-300-VA.jsonl` | `benchmarks/results/2026-10-29-evals/E3-300-VA.jsonl` | 398 rows; 221 correct = 55.53%; element grounding 106/223 = 47.53%, box grounding 98/223 = 43.95% | primary | high |
| 2 | `runs/E2-nomem/eval-E2final-nomem-V.jsonl` | `backup-2026-10-29/Model_Training_Server/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl` | 398 rows; 255 correct = 64.07%; same keys/order as #1 | primary | high |
| 3 | `runs/E3-cur24k-mix/eval-step300-V.jsonl` | `backup-2026-10-29/prd-survey-x1-5090/acwork/runs/E3-cur24k-mix/eval-step300-V.jsonl` | same checkpoint as #1 in V mode; 197 correct = 49.50%; grounding 85/184 = 46.20% | primary | high |
| 4 | `runs/E3-cur24k-mix/train.log` | `backup-2026-10-29/prd-survey-x1-5090/acwork/runs/E3-cur24k-mix/train.log` | matched step-300 EVAL pair (50.0% vs 55.5%); run ends step 320 of 1500; `acc_target_el` 0.53@s180 → 0.71@s310 (still climbing) | primary | high |
| 5 | `runs/E3-cur24k-mix/args.json` | `backup-2026-10-29/prd-survey-x1-5090/acwork/runs/E3-cur24k-mix/args.json` | `steps:1500, train_set:cur24k, track:mix, lr:1e-5, warmup:30, micro:2, accum:8, mem_frac:0.55, save_every:1500` | primary | high |
| 6 | `ckpt-E2-final/args.json` (run `E2-cur15k-V`) | `backup-2026-10-29/ckpt-E2-final/args.json` | V baseline recipe: `steps:938, train_set:cur15k, track:V, lr:1e-5, warmup:30, micro:2, accum:8, mem_frac:0.92` | primary | high |
| 7 | `ac_common.py` | `backup-2026-10-29/prd-survey-x1-5090/acwork/ac_common.py` | `pick_candidates` smallest-first dedupe; `gold_candidate` falls back to smallest **containing** candidate; both train and eval import this module | primary | high |
| 8 | `run_acv2.py:160-215` (`score_row`) | `backup-2026-10-29/prd-survey-x1-5090/acwork/run_acv2.py` | spatial branch requires exact action-type match and the point inside the **smallest** node containing the gold point; grounding forced `False` on type mismatch | primary | high |
| 9 | `eval_d1.py:60-115` | `backup-2026-10-29/prd-survey-x1-5090/acwork/eval_d1.py` | `pt` overwritten by `el_pt`; `el_pt` stores only the bbox centre, no `idx`; `correct_box_only` ablation computed on the same rows | primary | high |
| 10 | `train_d1.py:70-90` | `backup-2026-10-29/prd-survey-x1-5090/acwork/train_d1.py` | `track=="mix"` → VA on 50% of samples; empty candidate list silently downgrades the row to V | primary | high |
| 11 | `compact_ac.py` | `backup-2026-10-29/prd-survey-x1-5090/acwork/compact_ac.py` | training `cands` produced by `ac_common.pick_candidates` — same function as eval | primary | high |
| 12 | `benchmarks/data/acv2/500_steps_filtered.json` | `benchmarks/data/acv2/500_steps_filtered.json` | 398 rows, no precomputed `cands`; a11y trees rebuild to median N=24 (mean 29.6, max 245) | primary | high |
| 13 | `benchmarks/results/2026-10-29-evals/E2-938-V.jsonl` | `benchmarks/results/2026-10-29-evals/E2-938-V.jsonl` | 258/398 = 64.82% — **a different file from #2** (255/398); the "V baseline" is ambiguous in the record | primary | high |

---

## Sources

1. `benchmarks/results/2026-10-29-evals/E3-300-VA.jsonl` — VA track, 398 rows, step 300.
2. `backup-2026-10-29/Model_Training_Server/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl` — V baseline, 398 rows.
3. `backup-2026-10-29/prd-survey-x1-5090/acwork/runs/E3-cur24k-mix/eval-step300-V.jsonl` — same checkpoint, V track.
4. `backup-2026-10-29/prd-survey-x1-5090/acwork/runs/E3-cur24k-mix/train.log` — training/eval log with both step-300 EVAL lines.
5. `backup-2026-10-29/prd-survey-x1-5090/acwork/runs/E3-cur24k-mix/args.json` — VA arm hyperparameters.
6. `backup-2026-10-29/ckpt-E2-final/args.json` — V baseline hyperparameters (run `E2-cur15k-V`).
7. `backup-2026-10-29/prd-survey-x1-5090/acwork/ac_common.py` — `pick_candidates`, `gold_candidate`, `gold_answers`.
8. `backup-2026-10-29/prd-survey-x1-5090/acwork/run_acv2.py` — `score_row`, the official-scorer port.
9. `backup-2026-10-29/prd-survey-x1-5090/acwork/eval_d1.py` — eval harness, `pt`/`el_pt` handling, `correct_box_only`.
10. `backup-2026-10-29/prd-survey-x1-5090/acwork/train_d1.py` — `make_sample`, mixed-track sampling.
11. `backup-2026-10-29/prd-survey-x1-5090/acwork/compact_ac.py` — training candidate construction.
12. `benchmarks/data/acv2/500_steps_filtered.json` — the 398-row eval set with a11y trees.
13. `benchmarks/results/2026-10-29-evals/E2-938-V.jsonl` — the *other* 938-step V file (258/398).

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Report at docs/research/R4-candidate-interface-postmortem.md (runtime output path) gives exact row-by-row counts (VA 221/398, V 255/398, VA-only 29, V-only 63, union 284, McNemar p=5.1e-4) and, for the matched-budget pair, VA 221 vs V 197 with 39/15 split and p=1.5e-3; a budget verdict (artefact, not scientific finding) supported by args.json + train.log showing 320/1500 steps on cur24k with track=mix; a failure taxonomy over 223 spatial rows (89 disjoint, 16 identical-bbox scored wrong, 4 hierarchical); an 8-item numbered defect list with evidence and one-line fixes; and a Coverage Status section naming 5 unverified items."
    }
  ],
  "changedFiles": [
    "/Users/macbookpro/.pi/agent/sessions/--Users-macbookpro-Workspace-research--/subagent-artifacts/outputs/dc70455a-ac30-4cf6-8334-622437159719/docs/research/R4-candidate-interface-postmortem.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 (row-by-row join of 4 eval JSONLs; McNemar recomputation)",
      "result": "passed",
      "summary": "398 unique keys, zero dups, identical key order; VA 221/398=0.5533, V@938 255/398=0.6407, V@300 197/398=0.4950; VA-only 29 vs V-only 63 (p=5.1e-4) and 39 vs 15 (p=1.5e-3)"
    },
    {
      "command": "python3 (import ac_common.pick_candidates; rebuild candidates from 500_steps_filtered.json; taxonomy + N buckets + rank distributions)",
      "result": "passed",
      "summary": "median N=24 mean=29.6 max=245; acc 58.6/55.9/55.3/53.5% across N buckets (no collapse); taxonomy 106 hit / 89 disjoint / 16 identical-but-wrong / 8 overlap / 3 ancestor / 1 descendant; chosen vs gold top-3 share 13.1% vs 13.3%"
    },
    {
      "command": "python3 (gold_candidate tight/loose/none split + container-centre scoreability check)",
      "result": "passed",
      "summary": "tight 173/223 (49.71% grounding), loose 44/223 (45.45%), none 6/223; on 12/44 loose rows the training target's centre is outside the gold node, making the label unscoreable; supervised ceiling 205/223=91.93% vs the claimed 97.2%"
    },
    {
      "command": "grep/sed over ac_common.py, compact_ac.py, eval_d1.py, train_d1.py, run_acv2.py",
      "result": "passed",
      "summary": "confirmed builder is shared by train and eval (no builder skew), but gold_candidate's containment fallback is looser than score_row's smallest-node requirement; confirmed track=mix sends 50% of samples to VA; confirmed el_pt stores only a centre with no idx"
    }
  ],
  "validationOutput": [
    "E3 train.log step 300: EVAL V step_acc 0.495 grounding 0.462 | EVAL VA step_acc 0.5553 grounding 0.5638 step_acc_box_only 0.5352 — same checkpoint, matched budget",
    "Recomputed row-by-row: VA-only 39, V-only 15, McNemar p=0.0015 (VA direction)",
    "Supervised-ceiling correction: 205/223 = 91.93% replaces the quoted 97.2%"
  ],
  "residualRisks": [
    "The 'VA wins at matched budget' result rests on a single 300-step checkpoint that was killed at step 320 of a planned 1500 and never saved a final checkpoint (save_every=1500). It is one eval, not a replication.",
    "The 19.7% loose-supervision figure is measured on the TEST split (223 spatial rows), not on compact_cur24k.jsonl; the mechanism is read from source and is split-independent, but the exact training-side rate is unmeasured.",
    "I could not reproduce 'oracle 59.05' from any artifact; no script or eval file in this workspace produces it, and the number coincides with Qwen3-VL-32B's published AC-v2 score. The report flags it as unverified rather than treating it as evidence.",
    "'Mixed V+VA training -2 to -4 points' could not be checked: no pure-VA or pure-V sibling run exists at matched budget.",
    "The training-side gain of VA could be an auxiliary-loss regularisation effect rather than a better readout; the report states this explicitly and makes it experiment 1 rather than a conclusion.",
    "The evidence table cites workspace-relative paths, not URLs, because every source is a local primary artifact; no network sources were used or needed for this brief."
  ],
  "noStagedFiles": true,
  "diffSummary": "Adds one new analysis document: docs/research/R4-candidate-interface-postmortem.md. No source, config, data, or benchmark file was modified.",
  "reviewFindings": [
    "no blockers",
    "note: the parent's correction that 'the comparison is confounded by budget' is upheld, but the parent's implied conclusion (that VA therefore simply needs a matched run) understates the finding — a matched comparison already exists in train.log and reverses the sign (VA +6.03 pts, p=1.5e-3). The report leads with this.",
    "note: the parent's item-3 framing ('parent/child vs unrelated') is measurable but the parent-side expectation of a granularity problem is not supported — only 4/223 rows are hierarchical near-misses while 89/223 are fully disjoint, which rules out parent/child snapping as a fix (+1.8 pts ceiling).",
    "note: the parent reported 'no train/eval candidate skew' as verified and correct in the builder, but the gold-answer constructor is looser than the scorer on 19.7% of spatial rows, making 12 labels unscoreable by construction. Reported loudly as defect 4."
  ],
  "manualNotes": "Two independent corrections to the steering list. (1) The brief's own numbers do not reconcile with the files: 0.5533 should be 55.53% (221/398), and the brief's '97.2% candidate ceiling' is 217/223 = 97.31% on the point-inside-some-candidate definition, which is a different and unreachable quantity from the supervised ceiling of 91.93%. (2) 'E2-938-V.jsonl' (258/398 = 64.82%) is NOT the same file as 'runs/E2-nomem/eval-E2final-nomem-V.jsonl' (255/398 = 64.07%); the record is ambiguous about which is 'the V baseline', and the report uses #2 per the brief while flagging #13. Also: the strongest practical recommendation is experiment 1 in §5 — there is no evidence in the artifacts that anyone has ever evaluated the 938-step checkpoint through the candidate interface, which would have cost one eval pass."
}
```
