# Provenance: what a calibrated act / ask / refuse gate is actually worth

- **Date:** 2026-11-07
- **Rounds:** 2 (four parallel evidence notes → cited draft → adversarial review → revision; the cost-window
  measurement was added between rounds and the second checkpoint was computed during revision)
- **Sources consulted:** 39 numbered sources — **31 external, 8 internal** ([8]–[10], [12], [35], [36], [38],
  [39] cite repo artifacts or internal notes; the rest are external publications, vendor pages and standards).
  All 39 are accepted; one earlier revision of this header said "36 numbered … 23 external / 13 internal",
  which summed to 36 and matched neither the count nor the split. The numbers are now derived by
  `scripts/check_record.py`, which counts the brief's own Sources list rather than trusting this line.
- **Sources accepted:** 39/39 after verification. All external URLs were fetched during the citation pass;
  two are flagged in place — [20] FILTER-SCAD (publisher page bot-blocked; DOI resolves, claim verified from
  indexed abstract text) and [24] USENIX (landing page serves metadata only; figures read from the PDF).
- **Sources rejected:** 0 outright. One §5 row was removed by the citation pass and **restored** with a new
  internal source ([35], `notes/2026-11-07-clock-checker-not-crash.md`, commit `2d68f68a`) after the verifier
  discovered that the cited audit predates the finding. The restoration is recorded in the brief itself.
- **Verification:** **PASS WITH NOTES.** The reviewer re-derived all 11 §5 rows from disk, all four break-evens
  from the stated formulas, and confirmed 27/27 cited internal paths. Six MAJOR findings were raised and all
  six were fixed in the revision (prediction contradicted by its own arithmetic; perfect-human assumption and
  dropped refuse branch; point estimates without intervals; unhedged negative; trial design misdescribed;
  "the tie" with no comparator or test). Residual notes: (a) 8 of 23 external sources were spot-checked against
  the draft rather than read end to end; (b) the cost window is a model under stated assumptions (perfect or
  fallible delegation, fixed ask cost, no refuse branch), not a measured user outcome; (c) the mammography
  study's noninferiority margin could not be confirmed as pre-registered from the fetched text, so the brief
  says "the study's noninferiority criterion" rather than "pre-registered".
- **Plan:** `outputs/.plans/calibrated-decision-value.md`
- **Research files:** `outputs/.drafts/calibrated-decision-value-{draft,cited,revised}.md`;
  `outputs/.drafts/calibrated-decision-value-research-{metrics-outcomes,audience,failure-modes,internal-audit}.md`;
  `outputs/.drafts/calibrated-decision-value-verification.md`; `benchmarks/results/2026-11-07-cost-window.md`;
  `benchmarks/results/2026-11-07-status-gate.md`; `notes/2026-11-07-clock-checker-not-crash.md`.

## Revision 2 — 2026-11-08 (the compound-accuracy correction)

A measurement made after the first verification pass changes what the brief's argument is *about*, so §5.1
was added rather than the earlier text being quietly rewritten.

- **New sources: 3, all read directly.** [37] MAI-UI Technical Report (arXiv:2512.22047) — Table 9 read from
  the arXiv HTML, which is the only place the SFT and RL rows are separated; Table 6 read for the baselines.
  [38] the derived `P(task) = p^n` arithmetic, reproduced in `notes/2026-11-08-compound-accuracy-reframe.md`.
  [39] the first full AndroidWorld run, `benchmarks/results/2026-11-08-aw116/`.
- **What changed in the argument, stated plainly.** The brief previously treated the gate's value as a
  complement to a competitive model. The per-step arithmetic shows the model is **28 per-step points** below
  MAI-UI-2B's *SFT-only* checkpoint and that online RL accounts for only **+0.8** of those points, so the gap
  is supervision and is not closeable here. The gate's value is therefore not a complement to accuracy — under
  a risk budget it is the **substitute** for it, and that is a stronger claim for the brief's premise, not a
  weaker one. §5.1 says so in those words.
- **A claim was narrowed, not strengthened.** "The online 1/8 was a harness finding" is now stated as
  arithmetic: 12.5% is consistent with 64% per-step accuracy, so the 1/8 never carried capability
  information in either direction.
- **A design requirement was added from measurement.** The two online failure signatures (confident-and-wrong
  premature termination vs non-termination with a decaying gate score) cannot be separated by a single
  confidence threshold. The brief now says the gate must be **score-based and type-aware**, which is a
  requirement it did not previously state.
- **Residual note, unchanged in kind:** the AndroidWorld run cited in [39] is **incomplete** — the live
  finished/total counts are in `benchmarks/results/2026-11-08-aw116/evidence-82tasks.txt` (refreshed by
  `scripts/refresh_aw116.py`), not restated here, because a count quoted in a provenance note goes stale the
  moment the run advances and already did. Only its qualitative facts are used — zero invalid actions among
  the emitted total, and the two failure signatures. No success rate from it appears in the brief.
- **Checks:** `scripts/check_record.py` asserts §5.1 exists, states each of the five load-bearing figures,
  distinguishes SFT from RL, cites [37] with a matching source entry, registers [38], and warns that [39] is
  incomplete. 158 checks pass.
