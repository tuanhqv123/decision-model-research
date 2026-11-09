# R1 — How do top labs make a SMALL model ground well? (research brief)

You are researching a concrete engineering problem for a 1.92B GUI grounding model. Read, extract numbers, and be
specific. Do not summarize papers generically; we need recipes and effect sizes.

## Our measured situation (do not re-derive)

- 1.92B single-pass decision model, 8 typed readout slots, zero output tokens, ~103 ms.
- AC-v2 398 steps: step accuracy 64.07%, action-type 84.17%, **grounding 56.05%** (point must land inside the
  smallest a11y node containing the gold point).
- Grounding given type is correct: **63.8%** (125/196). Type ceiling inside spatial rows: 87.9%.
- 71 of 143 errors are "right verb, wrong element". Median miss **437 px**, 86.8% of misses >150 px, only 4.4%
  within 50 px. **So it is element SELECTION, not click precision.**
- A11y candidate list is available at eval time and the gold point is inside some candidate on **97.2%** of rows
  (median 24 candidates, p90 64). Training data has candidates too: the gold target bbox matches a candidate
  exactly on **83.0%** of target rows.
- A prior VA-track run (choose from candidates) got only 47.5% element grounding and 55.5% step accuracy — so
  simply switching interface does NOT work.

## What to find (each with a number and a source URL)

1. **Apple ML research**: Ferret-UI Lite (3B on-device GUI agent). What exactly did they find about *grounding*
   for small models — data recipe, resolution handling, coordinate representation, loss, and the measured
   grounding numbers? What did they say did NOT work?
2. **Microsoft Research**: GUI-Actor (coordinate-free, attention-based action head; also "moving beyond sparse
   grounding with complete screen parsing supervision", ICML 2026). What is the mechanism, what are the deltas,
   and what does "complete screen parsing supervision" buy versus sparse point supervision? Give the ablation
   numbers.
3. **Google / DeepMind** and **OpenAI** published work on UI element grounding for small models (ScreenAI,
   SeeClick/UGround lineage, OS-Atlas). Extract: (a) how they turn an element into a training target (point vs
   bbox vs region), (b) what data synthesis they use to get bbox supervision cheaply, (c) reported deltas from
   each ingredient.
4. **Small-model specific recipes 2026**: ZonUI-3B (WACV 2026), GoClick, WinDOM (self-family distillation),
   RankGround (reranker-guided crop), PreView (preprocessed views + cues). For each: what is the single
   highest-leverage ingredient, and does it need training or is it inference-time?
5. **The candidate-selection paradigm specifically**: any paper that makes a small model *choose among candidate
   elements* rather than regress coordinates (UGround-style, "element-wise" contrastive such as ConClick, or
   listwise ranking). What accuracy does candidate selection reach, what is the candidate generation recall, and
   what breaks?

## Output

Write `docs/research/R1-small-model-grounding-recipes.md` with:
- A table: method | model size | what it changed | grounding delta | needs training? | source URL | date
- A "top 5 ingredients ranked by expected delta per GPU-hour for OUR situation" list, each with the number that
  supports it and the exact change to make.
- An explicit "does not transfer to us" list with reasons.

Cite every number to a URL. If a number is from an abstract only, say so. Be blunt where evidence is thin.