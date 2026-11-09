# R3 — DATA: how to manufacture bbox supervision for element selection (research brief)

Read this, then write `docs/research/R3-grounding-data-design.md`. Do not edit anything else.

## Our data situation (measured, do not re-derive)

- 88,288 train decisions from AndroidControl (GCS TFRecords), 15,000 curated (`cur15k`).
- Each record has: goal, history, screenshot (720 px wide), gold action (click x/y or type_text), the **gold
  target bbox** (smallest a11y node containing the gold point), and a **candidate list** built from the a11y tree
  (median 24, p90 64).
- The gold target bbox matches a candidate **exactly** on 83.0% of target rows.
- At eval, the gold point is inside some candidate on **97.2%** of spatial rows.
- A previous attempt at the candidate interface plateaued at 47.5% element grounding / 55.5% step accuracy.
- Known dataset defect: AndroidControl-Curated reports ~24% unclear tasks and ~8% multiple valid actions; ~20%
  of rows have questionable ground truth (GUI-Libra's estimate).

## What to determine

1. **How do the strong teams build bbox supervision cheaply?** Specifically: OS-Atlas (web-synthesis pipeline),
   UGround (synthetic web pages with DOM), SeeClick (auto-labelling), WinDOM (54,425 records by self-family
   distillation), GUI-Actor ("complete screen parsing supervision", ICML 2026), AMEX (our notes say 1.2M Android
   element annotations, CC-BY-4.0, 93 GB). For each: how the labels were produced, the count, the licence, and the
   reported grounding delta from adding it.
2. **Dense vs sparse supervision.** "Complete screen parsing" supervises ALL elements, not just the target. Does
   the paper report how much of its gain comes from densifying the label set versus other ingredients? Give the
   ablation numbers.
3. **Negative/hard-negative mining.** ConClick builds target-distractor pairs from a stronger model's mistakes.
   What is the reported delta, how many pairs, and how were distractors matched to elements? Could we reproduce
   this with OUR OWN 71 wrong-element rows plus the candidate list (i.e. the model's own mistakes as negatives)?
   Estimate how many pairs 71 rows could yield and whether that is plausibly enough.
4. **Multiple-valid-target handling at the data level.** Do any of these papers drop ambiguous rows, expand the
   target, or accept any-of-k? What is the measured effect on reported accuracy?
5. **AMEX specifically**: what is in it, what licence, what size, what grounding numbers does anything report from
   training on it?

## Output

- Table: source | label type | count | licence | reported grounding delta | source URL
- A ranked list of the three cheapest data moves for us, each with an estimated number of new supervision signals
  we could generate from data already on disk, and the exact generation rule.
- An explicit risk list: what could make our own hard-negative mining actively harmful.
