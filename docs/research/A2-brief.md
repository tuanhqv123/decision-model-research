# A2 — Does trajectory SFT actually raise END-TO-END success for a ≤4B model? (research brief)

Read `docs/research/ISSUES-2026-11-09.md` first (issue I3). Then execute this and write
`docs/research/A2-trajectory-sft-evidence.md`. Do not edit any other file.

## The claim under test

Our plan is: per-step accuracy is 64.07% because we train on single-step decisions; training on multi-step
trajectory data will lift it, and that lift will propagate to AndroidWorld task success. **Nobody has shown
this decomposition for a 2B-class model.** That is the gap this brief must close or declare open.

## Anchor facts (do not re-derive)

- Our AndroidWorld-116 measured: **18.8% (15/80)** scorable, Wilson 11.7–28.7.
- Published 2B bar: **MAI-UI-2B = 45.1% SFT / 49.1% post-RL** end-to-end; **92.3% / 93.1% per-step**
  (MAI-UI Table 9, arXiv 2512.22047). So RL is worth **+0.8 per-step at 2B** — the gap is supervision.
- Our per-step is 64.07% (AC-v2), i.e. **28 points below that SFT checkpoint**.
- `P(task) = p^n` arithmetic: 64% per-step → 10.8% at 5 steps, 2.8% at 8, 1.2% at 10.

## What to find (each with a number AND a URL)

1. **Direct evidence at ≤4B.** Find every paper that trains a ≤4B GUI/mobile model on multi-step trajectory
   data and reports **task success rate** (AndroidWorld, AndroidLab, AITW, MobileMiniWob, or similar) — not
   action/step accuracy. For each: base model + size, data (single-step vs trajectory vs mixed), the measured
   end-to-end delta and what it is measured against, RL or SFT-only, date, URL. If genuinely none exist at 2B,
   say so loudly — that changes the plan from "known effective" to "inference".
2. **The strongest indirect evidence.** Where a paper *does* ablate trajectory vs single-step data at any scale,
   report the delta. Same for ablations of: more data at the same recipe, longer context/history, higher
   resolution, rejection sampling, and judge-in-the-loop. Which ingredient has the largest measured end-to-end
   coefficient?
3. **Does history/state help at all?** Our training data has **17.6% zero-history rows** and an interaction
   history slot. Find measured evidence that conditioning on prior actions improves task success (any scale),
   and specifically whether the *format* of history matters (raw action list vs rendered subgoal progress).
4. **The low-per-step-accuracy regime.** Most trajectory-SFT papers start from a base model already at ~90%
   per-step. Find any evidence of what trajectory SFT does when the starting model is at ~64% per-step — does
   it help, or does it amplify the model's own errors (compounding / DAgger-style covariate shift)? Cite the
   imitation-learning compounding-error literature if that is the honest answer, and say what mitigation it
   prescribes (DAgger, prefix reuse, rejection sampling on rollouts, error-correction data).
5. **The counter-case.** Find the strongest published argument or ablation that single-step/mixed data is
   competitive with trajectories for mobile agents, or that trajectory SFT alone does not move end-to-end.
   We need the steelman against our plan, not only the support.

## Output format

`docs/research/A2-trajectory-sft-evidence.md`:
- **Verdict line first**: is trajectory SFT a supported lever for a 2B model at 64% per-step? supported /
  inferred / unsupported + one sentence.
- The ≤4B evidence table (or an explicit "no qualifying studies found" with the search strings you used).
- An ingredient-effect table for item 2 (ingredient | measured end-to-end delta | scale | URL).
- The compounding-error section with the arithmetic applied to our numbers.
- The counter-case.
- "What I could not verify."

Rules: mark abstract-only numbers as abstract-only. Never mix per-step and end-to-end in the same column.
If a source is a vendor blog, say so. Semantic Scholar may return HTTP 429 — fall back to OpenAlex, arXiv
listing pages, `web_search`, and `fetch_content` on the arXiv abstract pages; record which route you used.
