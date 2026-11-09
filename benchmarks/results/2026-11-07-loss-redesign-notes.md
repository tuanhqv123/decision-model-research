# 2026-11-07 — What to change in the loss, given the gradient finding

Companion to `2026-11-07-why-grounding-plateaus.md`. That note shows the box loss has **zero gradient** beyond
~400 bins, which is where 86.8% of our misses are. This note records the concrete options and what each costs,
so the choice is not re-litigated later.

## The three candidate fixes, with their evidence

| # | change | mechanism why it fixes the flat-gradient problem | published delta | cost |
|---|---|---|---|---|
| 1 | **Candidate classification** (`target_el` choice over the a11y list) | a wrong element is a wrong *class*, so the loss never vanishes regardless of distance | matched-weight test on our own data: **+6.03 pts** (p = 1.5e-3), grounding **+10.2** | one training run (running now) |
| 2 | **Bounded / spatial-softmax coordinate loss** | replaces the per-edge Gaussian with a distribution over patches, so a far prediction still has a slope pointing at the right region | no isolated published number found (R1 could not isolate a point→containment delta) | code change in `train_d1.py` |
| 3 | **Contrastive margin vs the model's own chosen distractor** | a pairwise term between gold and the *actually chosen* wrong element has a gradient exactly where the Gaussian is flat | ConClick **+11.3** ScreenSpot-Pro with **8,584** examples; their error profile (74–88% of wrong clicks hit other elements) matches ours | needs negative mining; ~4k pairs from our own 71 wrong-element rows × candidate list |

## Why #1 is first and not #2

#2 is the more "obvious" fix if you look only at the loss, but it keeps the hard problem: the model must still
regress a coordinate in a continuous space where two adjacent 40×40 buttons are a few bins apart. #1 removes the
continuity entirely — the answer is an index. It also has a matched-weight measurement on **our** data (the +6.03),
which #2 does not. The running arm tests #1.

## What #1 does NOT fix, and what that implies

The candidate list is built from the accessibility tree. On Android that is usually available; in the online
AndroidWorld loop it was **not** (the HTTP path was pixels-only, paper §5.7). So a candidate-interface win is an
offline/AC-v2 win unless candidates can be derived from pixels at inference time.

**That is the honest boundary of the whole "selection" line**, and it is worth stating in the paper rather than
discovering in review. If the arm wins, the next question is candidate generation from pixels (a detector, not an
LLM), which is a different research programme.

## Sequence after the arm lands

1. If grounding ≥ 62% and step acc ≥ 62% → the arm passes; run the union check against E2 (bar: ≥ 70%).
2. If grounding < 58% → the line is closed per the pre-registration, and #2/#3 become the only remaining options.
3. Either way, the **loss-gradient finding stands on its own** and belongs in the paper as the mechanism for the
   plateau — it is arithmetic, it is reproducible in four lines, and it explains a number that has been stuck for
   months.


## A design flaw in the running arm, found while verifying it

The arm trains with `D1_MAX_CANDS=14` because the 5090 had only 13.9 GB free beside prod. The cap is **safe for
the ceiling** — `GOLD_KEEP` pins the gold candidate, and verification on 400 real rows shows it survives 392/400 —
but it is **not free**:

| | value |
|---|---|
| candidates per eval row | median 25, p90 65, max 137 |
| candidates **dropped** by cap=14 | **58.6% of all candidates** |

So the arm sees the right answer but **59% fewer wrong answers to learn from**. ConClick's finding is that the
gain comes specifically from *contrast* against distractors (74–88% of wrong clicks hit a non-gold element), and
this cap removes most of that signal. The arm can therefore **understate** what the interface can do.

This must be reported with the result, not after it: a positive result under a 14-candidate cap is a *lower
bound*; a null result does not close the line, because the cap is a memory workaround, not a design choice.

**Fix for the next run (needs memory, not code):** raise `D1_MAX_CANDS` to 40–64 on a card with ≥20 GB free.
At 2,164 tokens for 14 candidates the relationship is roughly linear, so 40 candidates ≈ 3.5–4k tokens; the
trainer's state budget must be checked against that before the run, not discovered by OOM at step 20.
