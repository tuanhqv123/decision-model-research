# Paper revision — 2026-11-02 (Feynman CLI, two passes)

**File:** `papers/single-pass-decision-acv2.md`
**Before:** 5,807 words (`backup-2026-11-02/paper-before-feynman.md`) → **After:** ~5,940 words.

## Method
`feynman --cwd <repo> --prompt=…`, two bounded passes. The first attempt ("revise the whole paper in
place") was rejected: the model hit its output-token limit and wrote nothing. The working pattern was to
ask for **only the replacement sections**, under a strict rule set, and splice them in.

Pass A produced §5.7 and §5.8. Pass B produced the "What would make this a contribution" paragraph and
Limitations 7-9. Both were given the verified numbers explicitly and told not to invent any.

## What changed, section by section

| Section | Change |
|---|---|
| §4.5 | (earlier, this session) "Proposed grounding fix" → **"Why the proposed grounding fix was withdrawn"**, with the AC-v2 box-grading argument and the scope caveat. |
| §4.6 | (earlier) "We have no emulator" removed — we now have one; item 3 rewritten to say online RL is out of scope, not impossible. |
| **§5.7** | **Rewritten by Feynman.** Now reports the 6-task result (d1 1/6, `wait` 0/6, `random` 0/6), the 2-task run-to-run variance (1/2, 0/2, 0/2), the Wilson interval (3–56%), latency under exclusive vs shared GPU (113–123 vs 189–262 ms), the pixels-only interface with the empty `app_name`, and the emulator dying under sustained load while `/health` still answers. |
| **§5.8** | **Rewritten by Feynman.** Objective ablation table (CE / Brier / joint / majority), stated as a **null at this budget**, with the observation that Brier improves the average probability while AUROC does not move, and that every arm is overconfident (mean conf ~0.80 vs accuracy ~0.57). |
| **§ "What would make this a contribution"** | **New.** Names the exact required experiment: full schedule, ≥3 seeds, paired bootstrap CIs, hard/OOD split, ~1.9k scored decisions for a 4-point separation, ≥59 accepted actions to certify 0.05. Says plainly none of it is done. |
| §6 Limitations 7–9 | **Replaced.** (7) online signal is preliminary, not a benchmark result; (8) any online number includes harness scaffolding; (9) novelty is narrow — Jev-Mobile and GUI-JEPA own the framing, no peer-reviewed AndroidWorld baseline exists, and the 67.9% GUI-Owl-1.5-2B figure is not cited because it has no primary source. |
| §5.6 | Restored after an editing slip removed it. |

## Claims corrected or removed
- **Removed:** any citation of `GUI-Owl-1.5-2B = 67.9%` (no primary source; real figures are 71.6% for the
  8B-Thinking variant and OSWorld-Verified 43.5 for the 2B).
- **Narrowed:** no novelty claim for "single-pass zero-token GUI action selection" — Jev-Mobile
  (arXiv:2609.30186) and GUI-JEPA (arXiv:2610.36877) own it.
- **Scoped:** the box-grading artifact argument no longer implies a large AC-v2 effect; the 1/5–1/3 figure
  is web-measured (~1% on native Android).
- **Added:** explicit statement that a stalling agent scores 0 online, and that success is device-state
  `is_successful`, not the model's `done`.

## Not changed
Sections 1–4 (except 4.5/4.6), §5.1–§5.5, the abstract (light touch only), the Sources appendix.

## To review
```bash
diff backup-2026-11-02/paper-before-feynman.md papers/single-pass-decision-acv2.md
```
