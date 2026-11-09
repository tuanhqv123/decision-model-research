# 2026-11-07 — Grounding: what we actually know after today's research

Four parallel research agents (Apple/Microsoft/Google recipes, loss design, data design, and a forensic
post-mortem of our own candidate-interface run) plus local re-derivation. Every number below is from an artifact
on disk or a cited paper.

## 0. The finding that changes the plan

**"The candidate interface failed" was wrong. On matched weights it wins, significantly.**

The prior conclusion came from comparing `E3-300-VA` (300 steps, `cur24k`, 50/50 mixed track) against `E2`
(938 steps, `cur15k`, pure V). Three confounds at once. But the artifacts contain a **matched** comparison —
same checkpoint, same step 300, same weights, 16 minutes apart in the same log block:

| | step acc | grounding |
|---|---|---|
| V (box) | **49.50%** (197/398) | 46.20% (85/184) |
| VA (candidate choice) | **55.53%** (221/398) | **56.38%** (106/188) |
| Δ | **+6.03 pts**, McNemar **p = 1.5e-3** | **+10.2 pts** |

VA-only 39, V-only 15. And VA's **own box readout** on the same prompt scores **53.52%**, i.e. also above the
V-track's 49.50% on identical weights — so the gain is not simply "the element path replaces the box path".

**We have never tested the candidate interface at full budget.** That is the single highest-value unrun
experiment in the programme.

## 1. The measurement that sets the ceiling

Gold point is inside some a11y candidate on **8,399/8,400** training rows, **471/474** val rows, and 217/223 eval
rows. But — and this is the correction R4 forced — that is *not* the supervised ceiling, because the scorer
requires the point inside the **smallest a11y node**, which is tighter:

| supervision quality | eval rows | note |
|---|---|---|
| exact bbox match with the gold node | 162 (76.4%) | cleanly scoreable |
| centre-inside-node fallback | 33 (15.6%) | scoreable |
| **no usable element label** | **17 (8.0%)** | box head must carry these |
| unscoreable label (old fallback) | **12 → 0 after fix** | was silently capping the ceiling |

**Supervised ceiling: 92.0%** (was 91.5% with the bug, and was being mis-quoted as 97.2%).

## 2. What the top labs actually do (R1, 17 sources, all fetched)

| ingredient | measured delta | needs training? | transfer to us |
|---|---|---|---|
| **Distractor contrastive selection loss** (ConClick) | ScreenSpot-Pro **54.71 → 66.03 (+11.3)** with **8,584** examples | yes | **high** — their error profile (74–88% of wrong clicks hit *other elements*) is exactly our 71/143 |
| **Dense "every element" supervision** (ScreenParse, ICML 2026) | Qwen3-VL-2B ScreenSpot-Web PixCov **0.292 → 0.720** | yes | **medium-high** — we get ~24 labels/row free from the candidate list |
| **Data deletion** (GoClick) | removed **65%** of data, **+4.0 avg** | no (eval only) | **highest EV/GPU-h** — pure measurement |
| **Containment / multi-positive labels** (GUI-Actor, Ferret-UI Lite) | not isolable; Ferret-UI Lite switches its RL reward to containment | yes, cheap | medium — our metric is already containment-flavoured |
| **Auxiliary preprocessor on a frozen grounder** (PreView) | Qwen3-VL-2B **57.2 → 64.6**, SSPro **43.1 → 59.3** | no model training | medium — breaks single-pass |
| **Apple Ferret-UI Lite** | high-res synthetic data removal costs **−6.80** on ScreenSpot-Pro | — | confirms resolution/data matters more than head design |

Honest gap: **OpenAI has published no UI-grounding recipe with per-ingredient deltas.** Do not attribute one.

## 3. What we fixed today

1. **`gold_candidate()` was supervising an unscoreable point on 12 eval rows.** The containment fallback could
   return a container whose centre is outside the gold node, so a model that perfectly imitated its own label was
   *guaranteed* `grounding=False`. Fixed to require centre-in-node; unscoreable labels 12 → 0.
2. **A text-overlap rank bug** in `build_candidate_supervision.py` inflated "gold in top-5 by goal text" from the
   true **38.8%** to 82.5% (ties at score 0 were all counted as hits). With **61%** of gold candidates sharing no
   token with the goal, lexical matching is a weak prior, not a shortcut.
3. **The `-1` gold index is kept, not dropped** — rows with no containing candidate are recorded so that any later
   accuracy number cannot be inflated by silent row deletion (the bug class that already produced one fake
   ensemble result).

## 4. The experiment that follows, pre-registered

`docs/research/PRE-REG-2026-11-07-candidate-head.md`:

- **Arm:** the same `compact_cur15k`, same 938 steps, same lr/seed/init as E2 — the **only** change is `--track VA`
  (candidate list in the state + `target_el` supervision). This isolates the interface from every other difference
  that confounded E3.
- **PASS:** grounding ≥ **62.0%** (from 56.05%) **and** step accuracy ≥ **62.0%**.
- **Secondary:** the union of the arm and E2 must reach ≥ **70%** step accuracy. On the 300-step pair the union was
  **71.36% (+7.29 over V, p = 0.0005)**, so this bar is calibrated on measured data, not hope.
- **Kill:** grounding < 58% closes the candidate line permanently.

## 5. What is still missing and why it matters

- **R2 (loss design)** and **R3 (data design)** agents were still running at the time of writing; their questions
  — what target representation to use, and how to mine hard negatives — are exactly what the arm above will need
  if it clears the bar.
- **No measured ask-acceptance rate** exists for a GUI gate (unchanged, and still the critical path for the paper).
- The `el_pt` eval record logs only the centre, not the chosen `idx`, so every element-head post-mortem needs a
  brittle rebuild. **Fix before the next VA run**: log `el_idx`.
