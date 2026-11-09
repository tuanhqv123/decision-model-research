# V2 — is the VA experiment itself well-posed? (adversarial design review)

You are a reviewer whose job is to kill a badly-designed experiment before it burns another GPU-day.

## Context

An arm is training: `d1-2B`, `--track VA` (candidate choice over an a11y list), same `cur15k` data, 938 steps,
lr 1e-5, init = the E2 box checkpoint, **with `D1_MAX_CANDS=14`** to fit memory. Its eval is running now.

Pre-registration: `docs/research/PRE-REG-2026-11-07-candidate-head.md`.
Design note: `benchmarks/results/2026-11-07-loss-redesign-notes.md`.
The measurement that motivated it: `benchmarks/results/2026-11-07-why-grounding-plateaus.md`.
The matched-weight evidence that VA beats V: `docs/research/R4-candidate-interface-postmortem.md` §2.2.

## Attack these

1. **Is initialising from E2 a mistake?** E2 was trained as a box-only model. The VA arm adds a new readout and a
   much longer state. Does warm-starting from E2 help or handicap? What does the literature say about adding a
   *choice* head to a model trained on *regression* targets — is there catastrophic interference? Cite papers.
2. **Is `D1_MAX_CANDS=14` defensible?** With 36 median candidates, capping to 14 discards real options. Compute:
   on the 398 eval rows, what fraction of gold candidates are inside the first 14 by the cap's ordering? If it is
   well below 97%, the arm is being handicapped by a memory workaround, and the correct experiment needs a longer
   state budget (or a two-stage selector). Give the number.
3. **Is the comparison matched on training tokens?** A VA row costs ~2,164 tokens vs V's ~1,537. So 938 steps of
   VA sees ~41% more tokens and a different loss composition (target_el weight 1.0 added). Say whether the
   pre-registered comparison "same 938 steps" is actually matched, and what the fairer control is.
4. **The pass bar is grounding ≥62%.** Given the arm's box path is still trained (weight 0.25) and its element
   path weight is 1.0, what does the bar actually test? Is there a way for the arm to hit 62% *without* the
   element head being useful (e.g. by the box path improving incidentally)? How would you detect that?
5. **What result would be MOST informative regardless of pass/fail?** Rank the three outcomes
   (grounding >> , grounding ≈, grounding <<) by information value for the paper, and say what each would mean
   for the programme's direction.

## Output

Write `docs/research/V2-verify-design.md`:
- A table: design choice | risk | severity | evidence/number | fix
- The single experiment you would run next, with its predicted number and cost.
- A blunt verdict: is this arm worth the GPU time, yes/no, and why.
Cite external claims with URLs. Be specific. Do not edit anything except your output file.
