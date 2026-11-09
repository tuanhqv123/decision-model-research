# R2 — LOSS FUNCTION for element selection (research brief)

Read this, then write `docs/research/R2-grounding-loss-design.md`. Do not edit anything else.

## The failure we must fix, with numbers

Our model emits a **box** readout: it predicts (left, top, right, bottom) as fractions of the picture, trained
with a Gaussian-smoothed CE on a 0-999 grid per edge, sigma = 4.

Measured on 223 spatial eval rows:
- 125 grounded (56.05%); 71 wrong-element with the right verb; miss distance median **437 px**, 86.8% >150 px,
  4.4% <=50 px; median y-bias 0.5 px.
- So the model picks a **different, plausible, often nearby element**. It is not failing to aim.

## What to determine

1. **What loss do the strong open grounders actually use?** For OS-Atlas, UGround, GUI-Actor, ShowUI, SeeClick,
   ZonUI-3B, ConClick: is the target a point, a bbox with Gaussian smoothing, a set of tokens, an attention map,
   or a classification over a fixed grid? Give the exact formulation where you can find it, and the ablation that
   justifies it.
2. **Is Gaussian-smoothed per-edge CE the right choice for a model whose error is far-away element selection?**
   Reason from first principles about what that loss rewards and what it is blind to. Compare against:
   (a) a classification over candidate elements, (b) a soft-argmax / spatial-softmax over patches,
   (c) center-point CE over a grid, (d) region/segmentation supervision (Seg2Click), (e) contrastive against the
   model's own wrong element (ConClick).
3. **Cell size arithmetic.** Our grid is 0-999 over the whole image. For a 1080x2400 screenshot, what is one grid
   cell in pixels horizontally and vertically? With sigma=4, what is the effective tolerance? Compare that to the
   median element size on Android phone screens. State whether the loss can even *express* "select this 40x40
   button out of two adjacent 40x40 buttons".
4. **What does the literature say about ambiguity?** AndroidControl-Curated reports ~24% unclear/multiple-valid
   rows. How do the strong grounders handle multiple valid targets (e.g. GUI-Actor's "coordinate-free" head,
   UGround's region targets)? Give the mechanism.

## Output

- Table: paper | target representation | loss | reported delta vs its own baseline | source URL
- A ranked recommendation for OUR case (small model, box slot, 0-999 grid, selection-not-precision errors), with
  the arithmetic in item 3 shown explicitly.
- The single cheapest change that could plausibly move grounding by >=5 points, and why.
