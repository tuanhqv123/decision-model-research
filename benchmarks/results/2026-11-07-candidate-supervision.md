# 2026-11-07 — the supervision for element selection is already in the data (ceiling 99.4%)

**The problem, restated.** Grounding is 56.05% and 71 of 143 errors are "right verb, wrong element". Median miss
437 px, 86.8% beyond 150 px, 4.4% within 50 px. This is **selection**, not precision.

**The measurement.** For every spatial training row, find which a11y candidate contains the gold point, using the
same `pick_candidates()` the trainer already stores:

| split | spatial rows | gold point inside a candidate | ceiling | median cands | p90 |
|---|---|---|---|---|---|
| train (`cur15k`) | 8,400 | **8,231** | **97.99%** | 36 | 76 |
| val | 474 | **463** | **97.68%** | 32 | 72 |

169 training rows and 11 val rows have no *usable* label (169+11 come from two causes: no candidate contains the
gold point, or the only containing candidate's centre is outside the gold node and the scorer therefore could
never award it — see R4 defect 4, fixed in `gold_candidate()` the same day).

**The honest number is 97.99%, not 99.99%.** The 99.99% figure counted "gold point lies inside some candidate",
which is not the same as "the supervised candidate's centre is inside the gold node", and only the latter is
scoreable.

Earlier, on the 398 eval rows (built from the eval a11y trees), the same measurement gave **97.2%** (206/212).

**So the headroom is not the question.** A candidate-selection head has a near-perfect recall surface. The
current box head reaches 56.05% against a 99.4% ceiling: **43 points are sitting in the interface choice**, not
in the data, not in the label, not in new annotations.

**Why the earlier candidate attempt (E3-300-VA) failed anyway.** It reached 47.5% element grounding. That run was
**300 steps** against the box baseline's **938**, so the comparison was never matched on training budget. That is
a confound, not a finding — and it means the candidate interface has not actually been tested.

**Two free signals inside the candidate list**, both now measured on the supervision file:

| signal | train | val |
|---|---|---|
| gold candidate carries a text label | **80.9%** | 77.9% |
| gold is top-1 by goal∩label token overlap | **13.0%** | 10.6% |
| gold in the top-5 by that overlap | **38.8%** | 37.4% |
| gold in the top-10 | **52.5%** | 49.5% |
| gold has **zero** goal-token overlap (unreachable by text) | **61.0%** | — |

⚠️ **A first version of this table reported "top-1 35.7% / top-5 82.5%" and both were wrong.** The rank was
computed as `#{candidates scoring strictly higher than gold}`, so when the gold score was 0 and nothing scored
above it, the row counted as a top-1 hit. With ~61% of gold rows scoring zero, that inflated top-5 to 82.5%. The
correct computation — position of gold inside the actual top-k list — gives **38.8%**, not 82.5%. The generator is
fixed (`build_candidate_supervision.py`).

**So text overlap is a weak prior, not a shortcut.** 61% of gold elements have no goal-token overlap at all, which
is exactly why a *learned* selector is needed rather than a lexical rule. Two-stage pruning by text would throw
away the majority of answerable rows.

**Artifacts.**
- `benchmarks/train/build_candidate_supervision.py` (generator; zero GPU)
- `benchmarks/results/2026-11-07-candidate-supervision-{train,val}.jsonl` — one line per spatial row with
  `gold_idx` into the existing candidate list. Rows with no containing candidate are recorded as `gold_idx = -1`
  and kept, **not dropped** (dropping them is the bug class that already produced one fake ensemble result).
- `benchmarks/results/2026-11-07-candidate-supervision.summary.json`

**What this licenses.** Train a choice head over the existing candidate list, with the budget matched to the box
baseline (938 steps, same data, same init). Predicted ceiling 99.4%; the only open question is what a 1.92B model
can do with 36-option selection — which is exactly what the literature says is the hard part (and where a
contrastive/listwise loss is indicated rather than plain CE).
