# Verification pass — `outputs/.drafts/calibrated-decision-value-cited.md`

*Adversarial evidence audit, 2026-11-07. Target: 31 KB research brief, 8 sections, 35 sources, 11 rows of
internal-artifact claims. Method: every internal number re-derived from the artifacts on disk; every cited
path stat-ed; the §7 cost arithmetic recomputed independently; eight external sources re-fetched and read
against the wording of the claim they carry. No edits were made to the draft or to any artifact.*

---

## 0. Checklist result (the four things this pass was asked to check)

| # | Check | Result | Detail |
|---|---|---|---|
| a | Every number in the §5 table against the cited internal artifacts | **PASS (11/11 rows re-derive)**; 2 status labels understate the artifact | §0.1 |
| b | The cost-window break-evens in §7 are arithmetic on the cited values | **PASS on the four stated numbers** (2.0345 / 2.1683 / 5.7857 / 1.8774 reproduce exactly); **FAIL on the derived prediction** — the §7 sentence about the second checkpoint is refuted by the same rows | §0.2 |
| c | The "no GUI outcome evidence" negative is stated as "not found" | **FAIL at 4 of 5 sites.** The hedge survives only in §8 Q3, which contradicts §6 item 5 in the same document | §0.3 |
| d | Internal artifact paths cited actually exist on disk | **PASS (27/27 paths + commit `2d68f68a`)**, with one blanket claim that the audit it cites contradicts | §0.4 |

### 0.1 — (a) The §5 table, row by row, against disk

Each row was re-derived independently (Python over the JSONL artifacts; `sqlite3` over the online
`index.db`; the arithmetic on the published values recomputed by hand).

| §5 row | Draft's number | What the artifact actually says | Verdict |
|---|---|---|---|
| AC-v2 pure Pass@1 | 64.32% (256/398) → 64.07% (255/398) in-process; 63.82% served | `E2-nomem/eval-E2final-nomem-V.jsonl`: 255 `correct:true` / 398 = 64.0704%. `backup-2026-11-06/offline/e2-pure-hist.jsonl`: 254/398 = 63.8191%. No artifact yields 256. | **correct** |
| groundB Pass@1 | 258/398 | `ckpt-groundB/eval-step1406-V.jsonl`: 258/398 = 64.8241% | **correct** |
| Type accuracy | 84.17% → 335/398 | `class_bias.json` `test_type_acc[0]` = 0.8417085427135679 = 335/398 | **correct** (but see W10) |
| Grounding | 56.05 pure / 57.40 hinted; 56.50 is step 705 | `grounding:true` counts over the 223 spatial rows: 938-hinted 128 (57.40%), pure 125 (56.05%), step-705 126 (56.50%) | **correct** — the best catch in the brief |
| Gate AUROC | 0.760 / 0.729 / 0.709 | `gate_test938.json` routed `[0.7600030027775693, 0.7285991140642304]`; `gate_test705.json` routed `[0.7600030027775693, 0.7092026825633383]` | **correct** |
| Certified 20% @ 40.7% | val-side only; test CP upper 21.7% | δ=0.20 row: `val_ub` 0.19948 (≤0.20), `test_coverage` 0.40704, `test_risk` 28/162 = 0.17284, `test_ub` 0.21702, `test_n` 162, `held: true` | **correct** (see W11) |
| Fusion risk@50% | 38/199 = 0.191, ±5.5 pp | `score_fusion.json` `test_risk_at_50["LEARNED FUSION"]` = 0.190954773…; 0.5·398 = 199 retained rows; binomial ±5.5 pp checks out | **correct** — note the ±5.5 pp is binomial and ignores episode clustering (see W3) |
| Candidate coverage | 1.0000 → 216/223 = 0.9686 | `d12bax…jsonl` `coverage` field: 216 `true` / 7 `false` / 175 `null` → 216/223 = 96.861% | **correct** |
| Online invalid actions | 28.2% → 14.7%, overlap 19.0–25.0, Fisher p 0.065 | `steps` table: `invalid_action` 20/71 (emitted actions) and 10/68. Wilson 95% = [19.04, 39.54] and [8.19, 25.00]; Fisher exact p = 0.0647 | **correct** (p truncated at 0.065, acceptable) |
| Online SR | 1/8 both, Wilson 2.2–47.1 | `{pure8,fallback8}.json` `task.sr` = 1/8, `sr_wilson95` = [0.02242, 0.47089] | **correct**, but see **W7** — the same file reports `n_reachable: 0`, `sr_reachable: null`, `coverage: {den: 0}` |
| "25% environment crash" | 2/8 per condition → 2 Clock-checker exceptions on a live device | `episodes.env_crash`: pure8 = 2/8 (`clockstopwatchrunning`, `clocktimerentry`); `crash_rate` = 2/8 in both reports; the note at `notes/2026-11-07-clock-checker-not-crash.md` supports the reclassification and the fix commit `2d68f68a` exists ("fix(aw): a raising task checker is not an environment crash") | **correct** |

Supporting counts also verify: 398 test rows / **350** distinct `episode_id` (§4 M2); **827** validation rows
(`error_budget.json`, `E2-val` n=827, 519 correct); `verify_paper.py` exits 0 with **69/69** PASS and prints
its own scope caveat verbatim ("stage-0 derived JSON only; the pure-protocol row, baselines, px-distances,
the a11y diagnostic, latency and every online number are NOT asserted here").

**Conclusion for (a):** no fabrication, no drift. The table is honest about the two rows it downgrades. The
problems are omissions around the numbers, not the numbers — see W7, W9, W10, W11.

### 0.2 — (b) §7 arithmetic

Independent recomputation from the artifact values (`c` = `test_coverage`, `r` = `test_risk`, `e` = 1 − acc):

| Quantity | Draft | Recompute | Verdict |
|---|---|---|---|
| gate vs always-act, 938 lower break-even | 2.03 | (1−0.407035)/(0.361809 − 0.407035·0.172839) = **2.0345** | ✅ |
| same, using CP upper risk 0.21702 | 2.17 | **2.1683** | ✅ |
| gate vs always-ask, upper break-even | 5.78 | 1/0.172839 = **5.7857** (rounds to 5.79; see W17) | ✅ (truncation) |
| learned fusion, lower break-even | ~1.9 | 0.5/(0.361809 − 0.5·0.190955) = **1.8774** | ✅ |

So the four numbers are exactly what the stated formulas give on the stated inputs. The arithmetic is not
the problem. **The derived prediction is** (W1), and the inputs are not all from the same protocol (W2).

### 0.3 — (c) The negative claim — fails

The draft makes the same negative claim five times and hedges it once, in the wrong place:

| Site | Wording | Hedged? |
|---|---|---|
| Exec summary, claim 1 | "there is **no peer-reviewed outcome-level abstention result at all**" | ❌ "at all" is an addition |
| §2 "The GUI gap" | "**No** GUI benchmark reports selective risk as a headline; no GUI-agent paper prices the *ask*; no paper measures …; no GUI paper reports …" | ❌ four absolutes, scope qualifier deleted |
| §3 closing | "it is **empty** in the GUI literature (T2 §A.5, T1 §6)" | ❌ and neither cited note says "empty" of the GUI literature |
| §6 item 5 | "**No GUI outcome result exists** (T1 §6)" | ❌ strongest form |
| §8 Q3 | "None found; the claim is 'not found', not 'does not exist' (T1 §6)" | ✅ |

The cited source says the opposite of what the draft asserts by omission. T1 §6 is headed
*"What is NOT measured anywhere (in the sources I checked)"* and opens: *"Phrased as 'not found', not 'does
not exist' — a negative claim needs a wider sweep than I ran."* The draft cites "(T1 §6)" at the point where
it deletes exactly that sentence's content — the citation is attached, the qualifier is stripped. Worse, §6
item 5 and §8 Q3 now contradict each other inside one document.

### 0.4 — (d) Cited paths

27/27 exist, including every artifact named in Sources 8–12 and 35, plus all six stage-0 JSONs, all six
`benchmarks/train/*.py` scripts, `repro_manifest.py`, the four prediction JSONLs, both online `index.db`
files, `aw_reports.tgz`, the d12bax diagnostic pair, and `notes/2026-11-07-clock-checker-not-crash.md`.
Commit `2d68f68a` resolves to "fix(aw): a raising task checker is not an environment crash".

One caveat, not a path failure: the draft's blanket claim that *everything* in
`benchmarks/results/2026-11-03-stage0/` re-derives byte-identically was not true at audit time — T4 R9 shows
`paired_705_vs_938.txt`'s bootstrap CI did not reproduce ([+1.01, +6.28] vs [+0.76, +6.35]). That file was
regenerated after the audit (mtime 2026-11-07 02:18, committed in `2ddc7cc2` "fix(paper): restate every
number the internal audit could not reproduce"), and it now reads [+0.76, +6.35]. The claim is true *now* but
the draft should say the file was regenerated, because as printed it contradicts the audit section it cites
two lines earlier. See W9.

---

## Part 1 — Structured review

## Summary

The brief assembles four internal evidence notes (T1 metrics/outcomes, T2 audience, T3 failure modes, T4
internal audit) plus 23 external sources into a single argument: a calibrated act/ask/refuse gate is worth
building because (i) abstention demonstrably changes outcomes in adjacent fields, (ii) the act/ask/refuse
pattern is already platform policy but the calibrated threshold is missing, (iii) no per-checkpoint
certificate survives distribution shift as a product property, and (iv) the program's own artifacts already
support a decision-level claim — expected cost per 1,000 decisions versus two trivial policies — as soon as
it is written down. It also reports, without soft-pedalling, that its own headline accuracy does not
reproduce (256/398 exists nowhere on disk), that its grounding figure belongs to a different checkpoint, and
that two further published numbers are contradicted by their artifacts.

The contribution the brief actually makes is a *reframing*: it converts an accuracy story into a
cost/coverage story and specifies the one measurement that would make the reframing gradeable. That reframing
is defensible, and the internal numbers supporting it check out — I re-derived all eleven §5 rows from disk
and all four §7 break-evens reproduce exactly. The weaknesses are concentrated in the last two pages: the
§7 prediction and kill criterion are logically inconsistent with the §7 arithmetic, the cost model silently
drops a branch that the draft's own cited source (T2 §B.6) includes, and the central negative claim is
stated four times in a form the cited source explicitly disclaims.

## Strengths

- **[S1] The §5 table is real, and I verified it independently rather than accepting T4's word.** Eleven rows,
  zero fabrication: 255/254/258 of 398, 125/128/126 of 223, 216/223, 20/71 and 10/68, 1/8 and 2/8 all
  reproduce from `benchmarks/results/2026-10-29-evals/*.jsonl`, `backup-2026-10-29/*`, `backup-2026-11-06/*`,
  and the two `index.db` files. The two hardest rows (candidate coverage 96.86%, mean-vs-truth `1.0000`) are
  correct and the grounding checkpoint swap is correct.
- **[S2] The brief performs self-surgery in writing.** It names the artifact drift ("the headline accuracy is
  a one-row drift", "56.50 is step 705", "cannot reproduce → restated") instead of repairing it silently. The
  "Removed unsupported claims" section documents a row that was *restored* with new evidence
  (`notes/2026-11-07-clock-checker-not-crash.md`, commit `2d68f68a`) rather than removed to make the table
  look clean — and I confirmed both the note and the commit exist.
- **[S3] §7's four break-evens are exactly what its formulas give** (2.0345, 2.1683, 5.7857, 1.8774), and the
  brief labels them "this brief's arithmetic on the cited rows" rather than dressing them as measurements.
  That labelling is the honest thing to do and it is rare.
- **[S4] The citation pass is documented at the level of individual sentences.** Sources 1, 2, 3, 13, 14, 16,
  17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 33, 34 each carry the sentence that was read. I spot-checked
  eight of them against the live sources (see Sources) and all eight support the quoted wording.
- **[S5] Degraded-access sources are flagged where the flag is needed.** FILTER-SCAD's OUP landing page is
  bot-blocked and says so, with the claim verified from indexed abstract text; the USENIX PDF parse failure is
  disclosed. This is the failure mode most briefs hide.
- **[S6] §1's metric taxonomy is correct and load-bearing rather than decorative.** The four-tier table
  (ECE/Brier vs AUROC/AURC vs risk-at-coverage vs realised cost) with "what it does *not* license" is the
  cleanest one-page justification I have seen for why the program's own AUROC claims cannot carry a
  deployment argument, and it is consistent with the cited sources' abstracts.
- **[S7] The failure-mode bullet in §4 is used against the brief's own programme.** The 17.3% → 23.9% transfer
  result and the val-side-only certificate are cited to *limit* the programme's claim, not to sell it.

## Weaknesses

- **[W1] MAJOR — §7's prediction and kill criterion are contradicted by §7's own arithmetic.** The draft
  predicts the 2–6× window "narrows to nothing on the second checkpoint". The same cited rows give the second
  checkpoint (`gate_test705.json`, δ=0.20: coverage 0.38945, risk 37/155 = 0.23871) a break-even window of
  **≈[2.08, 4.19]** using that checkpoint's own always-act error (1 − 244/398 = 0.3869), or ≈[2.27, 4.19]
  using the served artifact's 0.3618. That is a narrowing, not a vanishing. Separately, the kill criterion
  ("does not dominate always-act at C_err/C_ask ≥ 2 on *both* checkpoints") triggers on the *current*
  checkpoint, because the measured lower break-even is 2.03 > 2.00: at ratio 2.0 the gate loses on 938, and
  on 705 it loses at anything below ≈2.08. A pre-registered falsifier set below its own point estimate is not
  a falsifier.
- **[W2] MAJOR — the cost model makes an assumption its own §2 evidence contradicts, and drops a branch that
  its cited source includes.** The formula `1000·(c·risk(c)·C_err + (1−c)·C_ask)` charges **zero** error cost
  to the (1−c) delegated decisions, i.e. it assumes the human resolves every ask correctly. §2 of the same
  brief reports "under 25% adherence" and override rates of 49–96%. With a delegated-branch error rate *q*,
  the window becomes `[0.593/(0.2915−0.593q), 0.407/(0.0704+0.593q)]`: **q=5% → [2.27, 4.07]; q=10% →
  [2.55, 3.14].** The model also omits the refuse branch entirely, although T2 §B.6 — cited as the
  justification — defines the value unit as
  `1000·[(1−a)·P(err|act)·C_err + a·C_ask + r·C_refuse]` **plus** a dismissal-rate tolerance. A three-way
  gate priced with a two-way model is the kind of simplification that a reviewer reads as the author not
  having used their own source. Finally, the two sides of the same comparison come from **different
  protocols**: always-act error from the served/pure artifact (254/398) but coverage and risk from the
  hinted artifact (`gate_test938.json`'s `test` is `E2-938-V.jsonl`, 258/398). Same-protocol numbers give
  2.107, not 2.0345.
- **[W3] MAJOR — break-evens are reported to three significant figures with no uncertainty, and the interval
  straddles the kill threshold.** The break-even is a ratio of two estimated quantities. Propagating only the
  always-act accuracy CI (254/398 Wilson [0.590, 0.684] → e ∈ [0.316, 0.410]) gives **2.03 ∈ [1.75, 2.41]**;
  propagating only the risk CI (28/162 → [0.122, 0.239]) gives [1.90, 2.24]. T4 §4 itself notes the coverage
  has "a 95% binomial half-width of ±4.9 pp". Nothing is reported. So the headline "2.03" is a point estimate
  whose plausible band contains 2.0 — the number the kill criterion is anchored on.
- **[W4] MAJOR — the central negative is unhedged in four places and self-contradictory in two.** See §0.3.
  §6 item 5 ("No GUI outcome result exists") and §8 Q3 ("the claim is 'not found', not 'does not exist'")
  cannot both stand; the brief's niche argument rests on the negative, so the unhedged form is the one that
  will be quoted.
- **[W5] MAJOR — the flagship outcome study is described with a design term the primary source does not use,
  and its construct inversion is dropped.** The draft: "ran standard double reading in parallel with
  AI-supported screening". The source (fetched): *"This is a prospective paired noninferiority experimental
  study"* … *"Each participant received both reading strategies"* — i.e. within-woman paired, not parallel
  arms. The error is inherited from T1 ("parallel-arm, not randomised between arms"), so this is a
  propagation failure, not invention. Second, T1 explicitly flags the structural inversion — *"here the
  machine abstains from escalating to the human, so the cost being traded is human reading time, not a human
  interruption"* — and the draft deletes it while claiming this is "the exact evidence shape our program
  needs" and then pricing **user** interruptions in §7. A skeptical reviewer's first two questions about that
  trial are exactly these two.
- **[W6] MAJOR — "the accuracy is a tie" has no named comparator and no test, and sits next to a documented
  significant improvement.** The draft says "the headline accuracy is a one-row drift from a tie" (§5) and
  "the accuracy is a tie [8]" (§6.5). T4's tie is against a specific external number (64.07 vs GUI-Libra
  64.30) that the brief never names; and "tie" is asserted without an interval, for a quantity whose 95% CI
  is ±4.7 pp. Meanwhile the brief's own artifact set contains a *significant* paired result in the opposite
  direction — 244/398 → 258/398, McNemar p = 0.0161, episode-clustered CI [+0.76, +6.35] — which the brief
  never cites. As written, "a tie" is a load-bearing premise for §6 item 5 and is not auditable.
- **[W7] MINOR — the online-SR row understates how degenerate that run is.** The cited report JSONs report
  `n_reachable: 0`, `sr_reachable: {den: 0, value: null}` and `coverage: {den: 0, value: null}` in **both**
  conditions, plus `timeout_rate` 7/8 and 6/8. T4 §4.3 flags exactly this ("`SR_all` vs `SR_reachable` cannot
  be split"). "reproduces, underpowered" tells a reader the number is small-n; the artifact says the split is
  undefined and the run is mostly timeouts. The §5 closing sentence partly covers this, but the row does not.
- **[W8] MINOR — the §5 footnote's row numbering is off by one.** The table has 11 rows. The footnote says
  "Rows 1–8: (T4 §1, §2) [8, 9]; candidate coverage: [12] …" but candidate coverage *is* row 8. Should read
  "rows 1–7".
- **[W9] MINOR — "Everything in `benchmarks/results/2026-11-03-stage0/` re-derives byte-identically" is not
  what the cited audit found.** T4 R9 lists `paired_705_vs_938.txt` in that directory as stale. The file has
  since been regenerated and committed (`2ddc7cc2`, mtime 02:18 — after T4 at 02:14), so the sentence is now
  true but unsupported as printed, and the draft never mentions the regeneration. Its scope list is also one
  item short: `verify_paper.py` does not assert the **a11y diagnostic** either, which is the artifact behind
  the 96.9% row in the same table.
- **[W10] MINOR — §5 omits a result in a cited artifact that cuts against the programme's optimism.**
  `class_bias.json` reports the raw type accuracy 0.8417 *and* the bias-adjusted 0.8141 with McNemar
  p = 0.0433 (7 fixed / 18 regressed). The brief's "Type accuracy … reproduces" reports only the flattering
  half. That is itself another instance of "a fix that does not transfer", which is §4's theme.
- **[W11] MINOR — "val-side only" hides the number that makes it legible.** The certified point is
  `val_coverage` 0.4764 with `val_ub` 0.1995; the 40.7% in the same row is the *test* coverage. Without the
  47.6%, a reader cannot tell that the certified coverage and the quoted coverage differ — which matters
  because §7 prices the gate at 40.7%.
- **[W12] MINOR — citation range is one item short.** §5 cites "(T4 §2, R1–R9)" while T4 §2 lists R1–**R10**;
  the brief's own verification note says it dropped the count because it was wrong, then cites a range that
  excludes R10 anyway.
- **[W13] MINOR — two hedges dropped between note and brief.** §3: "every shipped gate is rule-based" (T2:
  "every shipped gate **found**"); §3: "no first-party document gives a numeric confidence threshold" (T2:
  "No first-party document **I found** states"). §4 also drops T3's "no paper **I found** demonstrates that
  combination end-to-end".
- **[W14] MINOR — single-preprint sources are flagged only in the Sources list, not at the point of use, in a
  brief that flags preprints elsewhere.** M6's coverage collapse rests on one preprint (T3 calls it "single-
  preprint evidence; not independently replicated"), and the clustering rationale in §4 rests on another
  (T3 rates it "medium — single very recent preprint"). Contrast §2, which does say "a preprint paired-task
  abstention benchmark".
- **[W15] MINOR — an uncited claim.** "habituation over days [T2 §C.2]" carries no numbered source, and the
  source behind T2 §C.2 is abstract-only (ACM DL blocked per T2's own table).
- **[W16] MINOR — one citation does not support the sentence it is attached to.** §8 Q4: "a single threshold
  cannot express that, and no published gate does it (T1 §6 item 8)". T1 §6 item 8 is about **per-item
  harm-weighted loss**, not per-action-type thresholds. Items 7 and 8 are adjacent but not this claim.
- **[W17] MINOR — one truncation.** 1/(28/162) = 5.7857, which rounds to 5.79, not 5.78. The fusion window's
  upper break-even (1/0.191 = 5.24) is computed but not reported, which makes "the fusion moves the *lower*
  break-even" sound more favourable than the full window.
- **[W18] MINOR — terminology drift inside the brief.** Exec summary: "act / ask / **suppress**"; §3 and the
  title: "act / ask / **refuse**". Apple's third branch is "don't offer results", which is neither, and the
  brief never maps one to the other.
- **[W19] MINOR — two §6 scorings are thinner than "HOLDS" implies.** Condition 4 requires "a predicted effect
  size"; §7 predicts a *cost window*, not an effect size (no "cost falls X%"). Condition 3 says "the
  artifacts support the risk and the baselines [8]", while T2 §A.5 records "no trivial-baseline comparison in
  the current artifacts; no cost ratio" — resolvable (the always-status/always-click baselines do exist in
  `error_budget.json`), but the sentence should say which baselines.
- **[W20] MINOR — two small source-list problems.** The Sources preamble says "Entries 8–12 are internal
  repository artifacts, not web sources", but [35] (`notes/2026-11-07-clock-checker-not-crash.md`) is also
  internal. And "a typed-decision benchmark with a realised-cost experiment on **non-GUI tickets** [19]" —
  the TypedBench abstract supports "realised cost under asymmetric cost matrices" but says nothing about
  tickets or GUI; that detail comes from T1, not the source.
- **[W21] MINOR — the same transfer result is characterised two ways.** §4: "23.9% … is a certificate failure,
  **not a demonstrated risk change**" (correct: z ≈ 1.45). §6 item 3: "the 17.3% → 23.9% **failure is real**".
  A reader who reads only §6 will over-read the gap.

## Questions for Authors

- **[Q1]** With `gate_test705.json` giving coverage 0.38945 and risk 0.23871, the second-checkpoint window is
  ≈[2.08, 4.19], not empty. What did you intend by "narrows to nothing" — a *further* degradation beyond the
  measured 23.9%, or the measured one? If the latter, the sentence needs replacing, not softening.
- **[Q2]** The kill criterion is anchored at ratio 2.0, while the point estimate that it is supposed to test
  is 2.03 and its plausible band is ≈[1.75, 2.41]. What margin (and what CI method — episode-clustered, as in
  `paired_eval.py`?) makes that criterion a test rather than a coin flip?
- **[Q3]** §7 omits `r·C_refuse` from the T2 §B.6 value unit and omits the dismissal-rate tolerance. Was this a
  deliberate narrowing to a two-policy gate, or an oversight? For an act/**ask**/**refuse** gate the refuse
  branch is not obviously negligible (T2 §C.3 treats "don't ask, refuse/undo" as the preferred branch for
  reversible harms).
- **[Q4]** Why is always-act error taken from the served/pure artifact (254/398) while gate coverage and risk
  come from the hinted artifact (`E2-938-V`, 258/398)? Same-protocol inputs give 2.11 rather than 2.03.
- **[Q5]** §5 and §6 say "the accuracy is a tie". Tie against what, tested how? And why is the paired
  improvement over the previous checkpoint (244 → 258, McNemar p = 0.0161, CI [+0.76, +6.35]) not cited,
  given that it is the one accuracy claim in the artifact set that *is* statistically supported?
- **[Q6]** Was the mammography trial's noninferiority margin pre-registered in the protocol (NCT04849776), or
  is "pre-registered" inherited from T1? The fetched text says "noninferior" but not "pre-registered".
- **[Q7]** `{pure8,fallback8}.json` both report `n_reachable: 0` and `timeout_rate` 7/8 and 6/8. Is
  `n_reachable` a live field or a stub? Either way, should "Online SR 1/8 — reproduces, underpowered" stand
  without that context in the same row?
- **[Q8]** §6 scores 5/5 and the plan's threshold was 2/5. Which single condition would you drop first if a
  reviewer attacked the scoring, and what evidence would upgrade it? (My candidate: condition 1, which rests
  on one trial in an inverted construct plus one routing paper.)

## Verdict

**Evidence integrity: high. Claim discipline: uneven, and concentrated in §7 and in the negative.**

Every internal number in the brief survives an independent re-derivation from disk; all 27 cited internal
paths exist; the four break-evens are the exact output of the printed formulas; and the brief is
unusually forthright about the numbers it had to withdraw. That is the part that most often fails in
briefs of this kind, and it did not fail here. On the four checks in scope: (a) passes, (b) passes as
arithmetic but fails as a derived prediction, (c) fails, (d) passes with one resolved caveat.

The revision risk is therefore not "are the numbers real" but "does the last page say what the numbers say".
§7 currently makes a falsifiable prediction that its own rows refute, sets a kill criterion below its own
point estimate, prices a three-way gate with a two-way model that assumes a perfect human, and reports a
three-significant-figure ratio with no interval. The negative claim that justifies the programme's niche is
stated in the strongest possible form in four places and hedged in one — and the hedged one and the strongest
one are in the same document. Fixing these is a few hours of writing and no new computation; leaving them
hands a reviewer four independent, checkable objections, two of which (W1, W4) are internal contradictions.

Revision priority: **high for W1–W6, low for W8–W21.** I would not circulate the §7 section in its current
form because W1 is refutable in one line of arithmetic by anyone who opens `gate_test705.json`.

Confidence in this assessment: **0.88.** I verified every internal number directly; the residual 0.12 is
(a) external sources I spot-checked rather than read end-to-end (8 of 23), (b) the FILTER-SCAD and USENIX
claims whose landing pages are bot-blocked/parse-failed and which I did not independently retrieve, and
(c) the mammography trial's pre-registration status, which I could not confirm from the fetched text.

## Revision Plan

1. **Rewrite the §7 prediction from the rows** (W1). Replace "narrows to nothing" with the computed second
   checkpoint window (≈[2.08, 4.19] on its own accuracy) and state the exact condition under which the window
   *does* close: it is empty iff `risk(c) ≥ 1 − acc`, i.e. iff the gate's selective risk on the acted subset
   reaches the always-act error rate (r ≥ e). At the measured 23.9% that condition is far from met.
2. **Re-anchor the kill criterion with a margin and a CI** (W1, W3). E.g. "fails if the gate does not dominate
   always-act at C_err/C_ask = 2 on both checkpoints, where dominance requires the episode-clustered 95% CI of
   the break-even to lie below 2." Report break-evens as intervals, not 3-s.f. points.
3. **State the cost model's assumptions in one sentence** (W2): perfect delegation, no refuse branch, no
   dismissal term; and either add T2 §B.6's `r·C_refuse` and dismissal tolerance or say explicitly why they
   are dropped. Add the sensitivity line (q = 5% → [2.27, 4.07]).
4. **Use one protocol on both sides of the comparison, or say why not** (W2). Cheapest fix: report the
   break-even twice, once on hinted inputs (2.11) and once on served inputs (2.03).
5. **Hedge the negative everywhere, once, and delete the contradiction** (W4). Use T1 §6's own formulation
   ("not found in the sources checked, which were X") in the exec summary, §2, §3 and §6 item 5; delete
   "exists" from §6 item 5; keep §8 Q3 unchanged as the canonical statement.
6. **Fix the trial description and restore the inversion** (W5): "each woman's screen was read under both
   strategies (paired, within-woman)" + one clause on the inverted direction and why the cost traded is
   reading time, not an interruption. Confirm "pre-registered" against the protocol or drop the word (Q6).
7. **Name the comparator and the test behind "tie", and cite the paired result** (W6).
8. **Tighten the §5 rows** (W7–W11): add the reachability/timeout context to the online-SR row; fix "rows
   1–8"→"rows 1–7"; note the `paired_705_vs_938.txt` regeneration and add the a11y diagnostic to the
   `verify_paper.py` scope list; add the class-bias adjusted 0.8141/p=0.0433 as a second column or footnote;
   add the val coverage 47.6% to the certified row.
9. **Housekeeping** (W12–W21): R1–R10; restore "found" hedges; move the preprint flags inline for [33] and
   [34]; cite habituation properly; fix the Q4 citation; 5.79; unify suppress/refuse; specify which baselines
   §6 item 3 means; add [35] to the "internal artifacts" note; drop "on non-GUI tickets" or cite T1 for it;
   align §6 item 3 with §4's "not a demonstrated risk change".

---

## Part 2 — Inline annotations

> "In GUI agents there is **no peer-reviewed outcome-level abstention result at all** (T1 §6, §7)."
**[W4] MAJOR:** The cited section is titled "What is NOT measured anywhere (in the sources I checked)" and states: "Phrased as 'not found', not 'does not exist' — a negative claim needs a wider sweep than I ran." "at all" is an addition with no support in the cited note. Restore the scope qualifier.

> "**The GUI gap.** No GUI benchmark reports selective risk as a headline; no GUI-agent paper prices the *ask*; no paper measures calibration→outcome transfer for a mobile agent; no GUI paper reports threshold transfer across checkpoints (T1 §6)."
**[W4] MAJOR:** Four absolutes, and the section cited supplies all four as "not found in the sources I checked". The citation number is present but the qualifier that makes it true has been removed. Rewrite as "In the sources checked (T1 §6 lists the search terms), no GUI benchmark reports…".

> "5. **A documented reason the current headline cannot support a CCF-A claim yet — HOLDS.** No GUI outcome result exists (T1 §6), the accuracy is a tie [8], the certificate is per-checkpoint (T3 §6) [7], and there is no cost model or ask rate in any GUI paper (T1 §6)."
**[W4] MAJOR:** "No GUI outcome result exists" is the strongest form of the negative and directly contradicts §8 Q3 ("the claim is 'not found', not 'does not exist'"). Two statements, one document, opposite truth conditions.
**[W6] MAJOR:** "the accuracy is a tie [8]" — no comparator, no interval, no test; and it omits the one accuracy comparison in the artifact set that *is* significant (244 → 258, McNemar p = 0.0161, CI [+0.76, +6.35], `paired_705_vs_938.txt`). Condition 5 is load-bearing for the whole "publishable-but-not-as-a-capability-claim" argument, so it needs an auditable premise.

> "builders price (T2 §B.6), and it is empty in the GUI literature (T2 §A.5, T1 §6)."
**[W4] MAJOR:** Neither cited section says "empty" of the GUI literature. T2 §A.5 describes the sentence a reviewer wants; T1 §6 says "not found". This is a citation attached to a claim its source does not make in that form.

> "A paired noninferiority trial in mammography and digital tomosynthesis ran standard double reading in parallel with AI-supported screening in which low-risk cases were called normal and the rest were double-read with AI support (T1 §3.1) [1]."
**[W5] MAJOR:** The primary source says the opposite of "parallel": *"This is a prospective paired noninferiority experimental study"* and *"Each participant received both reading strategies"* (fetched 2026-11-07). "Paired" and "in parallel" cannot both describe this design. The error originates in T1 ("parallel-arm, not randomised between arms"), so it propagated — but the brief now contradicts its own adjective in the same sentence.

> "That is the exact evidence shape our program needs — and the exact reason a bare AUROC claim is not enough."
**[W5] MAJOR:** T1 flags the structural inversion on the same page: *"here the machine abstains from escalating to the human, so the cost being traded is human reading time, not a human interruption."* An act/ask gate interrupts the *user*, and §7 prices exactly that. "The exact evidence shape" overstates the analogy; the honest version is "the same *form* of evidence, in an inverted direction".

> "With a measured 36.2% error when acting always (1 − 0.6382 on the pure-protocol served artifact) and 17.3% selective risk at 40.7% coverage, the gate dominates **always-act** once a wrong action costs more than **~2.0×** an interruption, and still beats **always-ask** until a wrong action costs **~5.8×** one (T4 §1, rows 1 and 9) [8, 9]"
**[W2] MAJOR:** The two sides of this comparison come from different protocols: 0.6382 is the served artifact (254/398), while 40.7% / 17.3% come from `gate_test938.json`, whose `test` field is the hinted `E2-938-V.jsonl` (258/398). Same-protocol inputs give 2.11. Say so, or report both.
**[W3] MAJOR:** "~2.0×" from a point estimate whose plausible band is ≈[1.75, 2.41] (from the always-act CI alone) and ≈[1.90, 2.24] (from the risk CI alone). Report the interval.

> "- **gate at coverage c**: `1000 · (c · risk(c) · C_err + (1 − c) · C_ask)`"
**[W2] MAJOR:** Charges zero error to the (1−c)·1000 delegated decisions — i.e. assumes a perfect human — while §2 of the same brief reports "under 25% adherence" and 49–96% override. Also omits `r·C_refuse` and the dismissal tolerance from T2 §B.6, the source this section says a builder will price. With 5% delegated-branch error the window becomes [2.27, 4.07]; with 10%, [2.55, 3.14].

> "the gate beats always-act when `C_err/C_ask > 2.03` (2.17 using the upper bound) and beats always-ask when `C_err/C_ask < 5.78`."
**[S3] Verified:** I recomputed all four from the artifact values: 2.0345, 2.1683, 5.7857, 1.8774. The arithmetic is exactly right. [W17] MINOR: 5.7857 rounds to 5.79.

> "**Prediction.** The gate dominates both trivial policies over a roughly 2–6× cost window on the current checkpoint, and that window narrows to nothing on the second checkpoint if the transfer failure is representative [8]."
**[W1] MAJOR:** Refuted by the rows it cites. `gate_test705.json` (same τ = 0.5269, δ = 0.20) gives coverage 0.38945 and risk 37/155 = 0.23871 → lower break-even 2.08 (its own accuracy) or 2.27 (served accuracy); upper break-even 1/0.23871 = 4.19. The window narrows to ≈[2.1, 4.2], not to nothing. A one-line recomputation by any reader falsifies this sentence.

> "**Kill criterion.** If the gate does not dominate always-act at `C_err/C_ask ≥ 2` on *both* checkpoints, the value claim fails for this checkpoint pair and the paper must say so — the certificate, not the cost argument, is then the contribution."
**[W1] MAJOR:** The measured lower break-even is 2.03 on 938 and ≈2.08 on 705, both **above** 2.0. So at ratio 2.0 the gate does not dominate on either checkpoint and the kill criterion fires on the very data that motivated it. A falsifier anchored below its own point estimate cannot fail in a way that is informative. Anchor it on a margin (and see W3 — 2.0 lies inside the confidence band of 2.03).

> "Everything in `benchmarks/results/2026-11-03-stage0/` re-derives byte-identically and `verify_paper.py` passes 69/69 — but that script checks stage-0 derived JSON only, not the pure-protocol row, the baselines, the pixel distances, or any online number (T4 §2 R8) [8, 10]."
**[W9] MINOR:** `verify_paper.py` does pass 69/69 (I ran it; exit 0, same scope line printed). But T4 R9 lists `paired_705_vs_938.txt` **in this directory** as not reproducing ([+1.01, +6.28] vs [+0.76, +6.35]). The file has since been regenerated and committed in `2ddc7cc2`, so the sentence is now true — but as written it contradicts the audit cited two lines earlier, and the regeneration is unmentioned.
**[W9] MINOR:** The script's own scope note also excludes the **a11y diagnostic**, which is the artifact behind the 96.9% row in this very table. Add it to the list.
**[W10] MINOR:** And `repro_manifest.py`, cited as the second provenance instrument [10], is flagged by T4 R10 as circular (its `artifact_kind` classifier decides by `correct == 256` / `== 258` — recognising the artifact by the value it is meant to check).

> "*Rows 1–8: (T4 §1, §2) [8, 9]; candidate coverage: [12]; online rows: [11]; the crash reclassification: [35].*"
**[W8] MINOR:** The table has 11 rows, and row 8 *is* candidate coverage, which this footnote assigns to a different source. Should read "rows 1–7".

> "| Online SR | 1/8 both | 1/8 both, Wilson 2.2–47.1 | reproduces, underpowered |"
**[W7] MINOR:** The numbers are right (I read `task.sr` = {num 1, den 8} and `sr_wilson95` = [0.0224, 0.4709] in both report JSONs). But those same JSONs report `n_reachable: 0`, `sr_reachable: {den: 0, value: null}`, `coverage: {den: 0, value: null}` and `timeout_rate` 7/8 and 6/8. T4 §4.3 records that "`SR_all` vs `SR_reachable` cannot be split". "underpowered" implies a small valid measurement; the artifact says the split is undefined and most episodes timed out.

> "| Type accuracy | 84.17% | 335/398 | reproduces |"
**[W10] MINOR:** `class_bias.json` also reports the bias-adjusted accuracy 0.8141 with 7 fixed / 18 regressed and McNemar p = 0.0433 — i.e. the class-bias correction *significantly degrades* test type accuracy. That is a second transfer failure of exactly the kind §4 argues about, sitting in a cited artifact, reported only in its favourable half.

> "| Certified 20% risk @ 40.7% coverage | as stated | val-side only; test CP upper 21.7% | reproduces with caveat |"
**[W11] MINOR:** The certified point is `val_coverage` 0.4764 / `val_ub` 0.1995; the 40.7% here is the *test* coverage. Give the 47.6% so a reader can see that the certified coverage is not the coverage §7 prices the gate at.

> "| Online invalid actions | 28.2% → 14.7%, "non-overlapping" | intervals overlap 19.0–25.0; Fisher p = 0.065; instances unmatched | **contradicted** |"
**[S1] Verified:** `invalid_action` counts are 20 over 71 emitted actions (28.17%) and 10 over 68 (14.71%); Wilson 95% = [19.04, 39.54] and [8.19, 25.00]; Fisher exact p = 0.0647. The contradiction of "non-overlapping" is correctly stated and is one of the brief's strongest contributions.

> "| Grounding | 56.50% (126/223) | 56.05% pure / 57.40% hinted; **56.50 is step 705** | reproduces, wrong checkpoint |"
**[S1] Verified:** grounding flags over the 223 spatial rows: step-705 = 126 (56.50%), 938-hinted = 128 (57.40%), pure = 125 (56.05%). This row is correct and is the single best piece of forensics in the brief.

> "**What this means for the value claim.** The calibration result survives audit; the headline accuracy is a one-row drift from a tie; the online half is a harness finding, not a capability result."
**[W6] MAJOR:** "a tie" again with no comparator. T4's tie is 64.07 vs an external 64.30 that this brief never names. Name it (and note the ±4.7 pp CI), or delete the word.

> "What they do not yet support is *transfer across checkpoints* — the 17.3% → 23.9% failure is real [8]."
**[W21] MINOR:** §4 says the same gap "is a certificate failure, **not a demonstrated risk change**" (z ≈ 1.45). "the failure is real" invites the reading that the risk change is real. Say "the certificate failure is real".

> "4. **One next measurement with a predicted effect size, a cost, and a kill criterion — HOLDS.** §7."
**[W19] MINOR:** §7 predicts a cost *window*, not an effect size, and its kill criterion is W1. The plan's bar is softer than it reads; state the predicted effect size explicitly (e.g. "expected cost per 1,000 falls from 36.2·C_err to 7.0·C_err + 0.593·C_ask").

> "**Verdict: the brief passes (5/5).**"
**[W19] MINOR:** The threshold was 2/5. Scoring 5/5 is defensible, but conditions 1 and 3 lean on a single trial in an inverted construct and on baselines T2 §A.5 says are absent. A reviewer-facing brief is stronger if it reports the score *and* the weakest two.

> "**The GUI gap.** … no GUI paper reports threshold transfer across checkpoints (T1 §6)."
**[Q-item]:** Also relevant: the nearest-artifact sentence right below lists AgentAbstain, which is a tool-using/MCP-sandbox agent benchmark, not GUI. Keeping it in the "GUI gap" bullet is defensible ("nearest artifacts") but slightly muddies the domain boundary the brief depends on.

> "and no published gate does it (T1 §6 item 8)"
**[W16] MINOR:** T1 §6 item 8 is "No paper reports a per-item harm-weighted loss … for an act/ask/refuse gate." The claim being supported is about per-action-type thresholds. Point at T1 §6 item 7 or state it as this brief's inference.

> "Group-conditional coverage is provably impossible distribution-free (T3 §6) [7]."
**[MINOR]:** T3's own ledger says "**Exact** conditional coverage … **Impossible.**" and separately lists "near-nominal coverage conditional on the validation draw" as achievable. Dropping "exact" turns a width theorem into a blanket impossibility — the first thing a conformal reader will object to.

> "The cost side is measured and unflattering: override 49–96% [22], habituation over days [T2 §C.2], and "are you sure?" producing reflex approval [32] (T2 §C.3–§C.4)."
**[W15] MINOR:** "habituation over days" carries no numbered citation, and T2's own table marks the underlying source "medium (abstract only; ACM DL blocked)". Give it a number and the caveat.

> "Entries **8–12 are internal repository artifacts, not web sources**; all others are live URLs fetched during the citation pass."
**[W20] MINOR:** [35] (`notes/2026-11-07-clock-checker-not-crash.md`) is also internal, and it is the single source for the §5 crash reclassification. Also, "all others are live URLs fetched" is two-thirds true: [20]'s landing page is bot-blocked and [24]'s PDF failed to parse — both disclosed in the verification notes, but not here.

> "Everything URL-resolvable was resolved; the FILTER-SCAD OUP landing page is bot-blocked (DOI resolves; abstract text verified from the indexing layer)"
**[S5] Strength:** This is exactly the disclosure a skeptical reader needs, and it is in the right place. Same for the USENIX parse-failure note.

---

## Verification log (what I ran, and what it returned)

| Check | Command / method | Result |
|---|---|---|
| All 27 cited internal paths | `for p in …; do [ -e "$p" ]` | 27/27 exist |
| Fix commit | `git cat-file -t 2d68f68a`; `git log -1` | resolves — "fix(aw): a raising task checker is not an environment crash" |
| Test-split counts | Python over `E2-938-V.jsonl` | 398 rows, 350 distinct `episode_id`, 398 distinct (episode, step) |
| Pass@1 by artifact | Python `correct is True` over 4 JSONLs | 255/398 (nomem), 254/398 (served), 258/398 (groundB), 258/398 (938-hinted) |
| Grounding by artifact | Python `grounding is True` over the 223 spatial rows | 938-hinted 128 (57.40%), in-process pure 125 (56.05%), served pure 125 (56.05%), groundB 134 (60.09%), step-705 126 (56.50%) |
| Candidate coverage | Python `coverage` field | 216 True / 7 False / 175 null → 216/223 = 96.86% |
| Gate rows | `gate_test938.json`, `gate_test705.json` | coverage 0.40704 / 0.38945; risk 0.17284 / 0.23871; `test_ub` 0.21702 / 0.28824; `val_ub` 0.19948 |
| Fusion | `score_fusion.json` | `test_risk_at_50["LEARNED FUSION"]` = 0.190954773; ΔAUROC +0.01315 [−0.0199, +0.0425] |
| Online invalid actions | `sqlite3 … index.db "select sum(invalid_action), …"` | 20 over 71 emitted; 10 over 68 emitted |
| Online SR / crash | `task` block of `{pure8,fallback8}.json` | sr 1/8 both; `crash_rate` 2/8 both; `n_reachable` 0 both; `timeout_rate` 0.875 / 0.75 |
| Provenance script | `python3 benchmarks/train/verify_paper.py` | exit 0, 69/69 PASS, scope caveat printed |
| Break-even arithmetic | independent Python | 2.0345 / 2.1683 / 5.7857 / 1.8774 — all as drafted |
| Second-checkpoint window | independent Python | [2.077, 4.189] (own accuracy) or [2.271, 4.189] (served accuracy) |
| Break-even uncertainty | Wilson CIs on 254/398 and 28/162, propagated | 2.03 ∈ [1.75, 2.41] and [1.90, 2.24] |
| Delegated-error sensitivity | closed-form, q = 0.05 / 0.10 | [2.27, 4.07] / [2.55, 3.14] |
| Audit item count | `grep -c '^\*\*R[0-9]'` on T4 | 10 items, R1–R10 (draft cites R1–R9) |

---

## Sources

External sources fetched and read against the draft's claims during this pass:

1. *AI-based triage and decision support in mammography and digital tomosynthesis…* (Nature Medicine 2026) — https://www.nature.com/articles/s41591-026-04277-x — confirms 31,301 women, −63.6% workload, CDR +15.2% (6.6–24.4), RR +14.8% (9.0–20.6) "not noninferior"; **and states "prospective paired noninferiority experimental study" / "Each participant received both reading strategies"** — the basis for W5.
2. RouteLLM, ICLR 2025 — https://proceedings.iclr.cc/paper_files/paper/2025/file/5503a7c69d48a2f86fc00b3dc09de686-Paper-Conference.pdf — Table 6 confirms 3.66 (95%), 1.41 (92%), 1.49 (87%); citation is exactly right.
3. Apple HIG, Machine Learning → Confidence — https://developer.apple.com/design/human-interface-guidelines/machine-learning — confirms "Consider changing how you present results based on different confidence thresholds" and "Always balance the benefits of a feature with the effort required to make a correction."
4. SafeGround (arXiv:2602.02419) — https://arxiv.org/abs/2602.02419 — confirms "up to 5.38% percentage points over Gemini-only inference".
5. AgentAbstain (arXiv:2607.10059) — https://arxiv.org/abs/2607.10059 — confirms 263 paired tasks / 42 sandbox environments / 8 scenarios / "17 frontier LLMs in 4 agent harnesses" / 59.5% (Gemini 3.1 Pro).
6. Continual Calibration (arXiv:2604.23987) — https://arxiv.org/abs/2604.23987 — confirms "coverage drops from 0.92 to 0.61, while accuracy remains within three points of baseline"; no venue listed (preprint).
7. *Learning When to Ask for Help* (arXiv:2305.16502) — https://arxiv.org/abs/2305.16502 — confirms ICRA 2024 and "+0.38 … with only a 0.3 expert interaction rate … Habitat … simulated human expert".
8. TypedBench (arXiv:2610.11392) — https://arxiv.org/abs/2610.11392 — confirms "realised cost under asymmetric cost matrices", seven generators / nine suites; **abstract says nothing about tickets or GUI** (basis for W20).

Internal artifacts inspected directly (all local to `~/Workspace/research`):
`benchmarks/results/2026-11-03-stage0/{error_budget,gate_test938,gate_test705,score_fusion,class_bias,grounding_audit}.json`,
`…/paired_705_vs_938.txt`, `benchmarks/results/2026-10-29-evals/{E2-938-V,E2-705-V,E2-val}.jsonl`,
`backup-2026-10-29/GPU_HOST_A/acwork/runs/E2-nomem/eval-E2final-nomem-V.jsonl`,
`backup-2026-11-06/offline/e2-pure-hist.jsonl`, `backup-2026-10-29/ckpt-groundB/eval-step1406-V.jsonl`,
`backup-2026-11-06/gate_runs/aw_runs_20261106/{pure8,fallback8}/index.db`,
`backup-2026-11-06/reports/aw_reports.tgz`, `benchmarks/results/d12bax-acv2-d12bax-2026-10-28.jsonl`,
`benchmarks/train/{verify_paper,class_bias,score_fusion,gate_routed}.py`,
`notes/2026-11-07-clock-checker-not-crash.md`, `outputs/.plans/calibrated-decision-value.md`,
and the four evidence notes T1–T4 in `outputs/.drafts/`.
