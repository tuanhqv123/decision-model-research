# A4 — Audit our own training labels (research + measured audit brief)

Read `docs/research/ISSUES-2026-11-09.md` first (issue I1). Then execute this and write
`docs/research/A4-label-audit.md` **plus** a runnable script `benchmarks/train/audit_labels.py`.
Do not edit any other existing file.

## Why this exists

`compact_cur15k.jsonl` (15,000 rows, 9,005 episodes, 98 MB) is the dataset every checkpoint we cite was
trained on. **Its labels have never been audited.** The dataset it derives from (AndroidControl) self-reports
**37.63% wrong-GT / 24.13% unclear-task / 8.12% multiple-valid** on a different split, and we have never
reproduced that on our subset. If a large slice of training is mislabelled, adding trajectory data adds noise
on top of noise.

Path: `backup-2026-10-29/GPU_HOST_C/acwork/acdata/compact_cur15k.jsonl`
Validation: `backup-2026-10-29/GPU_HOST_C/acwork/acdata/compact_validation.jsonl` (827 rows)
Images: `backup-2026-10-29/GPU_HOST_C/acwork/acdata/imgs/<ep>/<step>.jpg` (19 GB total dir)
Code that built it: `benchmarks/train/preprocess_ac.py`, `benchmarks/train/curate_ac.py`,
`benchmarks/train/build_candidate_supervision.py`, and `benchmarks/train/ac_common.py`
(`gold_candidate()` in particular).

## Already known (do not re-derive, but DO verify the counts)

| Fact | Value |
|---|---|
| rows | 15,000 |
| episodes | 9,005 |
| action mix | click 7,913 / status 3,300 / scroll 1,350 / open_app 1,050 / navigate_back 750 / type_text 450 / wait 150 / long_press 37 |
| rows with **empty history** | 2,635 (17.6%) |
| known-unscoreable labels found and fixed | 12 (the `gold_candidate()` centre-in-node bug) |
| has `target` bbox | click 7,913, long_press 37, type_text 450 — nothing else |

## What the audit must measure (GPU-free, structural, reproducible)

Write `benchmarks/train/audit_labels.py` (numpy-free, stdlib only, must run on this Mac in < 3 min) that emits
one JSON summary and a CSV of flagged rows. It must measure **at minimum**:

1. **Gold-point-in-candidate rate per action type.** For every spatial row, is the gold point inside at least one
   a11y candidate? Report counts and % per type (`click`, `long_press`, `type_text`). A row whose gold point is
   inside *no* candidate is unscoreable in our own scorer — those are definitionally bad labels.
2. **`type_text` label consistency.** For each of the 450 `type_text` rows: is `target` present? Is the gold
   point inside the target bbox? Does the bbox correspond to an editable node (check for an `editable`/`class`
   hint in candidates if the field exists)? Report how many `type_text` rows have a target box that lies outside
   any candidate.
3. **Coordinate sanity.** Any gold point outside `[0,W]x[0,H]`; any zero-area `target` bbox; any bbox with
   `x_min > x_max`. Report counts. Also the distribution of bbox areas by type (median, p10, p90).
4. **Duplicates and leakage.** Exact-duplicate `(img, action)` rows; identical `img` appearing with conflicting
   action types (a real ambiguity signal — count them, this is our proxy for "multiple valid"); duplicate
   screenshots across `cur15k` and `compact_validation` (train/val leakage — must be 0).
5. **History validity.** Rows where `len(history)` exceeds `step` (impossible); rows where `step == 0` but
   history is non-empty; rows where the episode's steps are non-contiguous (missing step indices) — report
   per-episode completeness of the 9,005 episodes.
6. **`status` label placement.** Of the 3,300 `status` rows, how many are the last step of their episode? How
   many appear mid-episode (a label that says "done" while more steps follow is suspect — count and list).
7. **Image existence.** Sample 2,000 rows, verify the `img` path resolves on disk. Report the missing rate.
   (Do not read pixels; this is an existence check.)
8. **The 37.63% question.** You cannot do human GT verification. So state precisely what a *structural* audit can
   and cannot rule out, and define the smallest sampling protocol that would measure the true error rate
   (sample size, stratification by action type, decision procedure, CI method). Include the exact sample-size
   arithmetic: to estimate a 10% error rate ±4 pp at 95% confidence, how many rows are needed?

## Also research (short, cited)

- What is the **published label quality** of AndroidControl and of AndroidControl-Curated? Find the primary
  source of the 24.13/8.12/37.63 figures (arXiv 2501.17196, "AndroidControl-Curated" lineage — verify the ID)
  and report verbatim what each percentage is measured over. We currently cite it second-hand.
- Do any published mobile-GUI SFT papers report an audited label-error rate for their training data? If yes,
  how did they measure it, and what did they do about it (filtering, relabelling, rejection sampling)?

## Output format

`docs/research/A4-label-audit.md`:
- **Verdict first**: one paragraph — are our labels good enough to train on, and what is the measured evidence?
- The measured-facts table (every item above, with the actual number produced by the script).
- A "flagged rows" section: how many, of which type, and what the top failure patterns are.
- The 37.63% reconciliation and the sampling protocol.
- "What this audit cannot rule out."
- The exact command to re-run: `python3 benchmarks/train/audit_labels.py --data <dir> --out <dir>`.

Rules: every number must come from running the script, not from this brief. If the data path does not exist,
say so and stop rather than inventing numbers. Do not train anything. Do not read 19 GB of images.
