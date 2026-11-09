# 2026-11-07 — Grounding: the honest summary of a full day of work

Written so the next session reads one file instead of twelve. Every number here re-derives from an artifact on
disk or a cited paper; the ones that did not replicate are marked as retracted rather than removed.

## Starting point

| | value |
|---|---|
| step accuracy (AC-v2 398, pure) | 64.07% (255/398) |
| action type | 84.17% |
| **grounding** (point inside the smallest gold node, 223 rows) | **56.05%** |

## What is now ESTABLISHED

**1. The box loss has no gradient where our errors are.** `target_box` is four per-edge cross-entropies against a
Gaussian (σ = 4 bins on a 0–999 grid). Cross-entropy is a **numerically identical constant (690.78 nats) for
offsets ≥ 200 bins**, so ΔCE/bin = **exactly 0** beyond ~400 px. Our misses are median **437 px**, 86.8% > 150 px,
4.4% within 50 px. The objective is sharp where errors are rare and flat where they are common. This explains
why 84% type accuracy coexists with 56% grounding, and it is arithmetic — four lines of numpy.
→ `2026-11-07-why-grounding-plateaus.md`

**2. Long candidate lists destroy grounding; this replicates on both splits.** Splitting the failed VA arm by how
many candidates were in the state:

| candidates | val Δ vs E2 | test Δ vs E2 |
|---|---|---|
| 1–10 | +3.8 | −4.0 |
| 11–20 | +10.0 | −7.3 |
| 21–40 | 0.0 | −5.8 |
| **41+** | **−17.3** | **−23.8** |

Only the **41+ collapse** holds on both splits. And the arm's **box-only** path degraded too (val 60.22, test
59.80 vs E2's 62.76/64.07), so the damage is **state length hurting the coordinate readout**, not the element
interface per se.
→ `2026-11-07-list-length-is-causal.md`

**3. Supervision for candidate selection already exists in our data.** The gold point lies inside a candidate on
**97.99%** of training rows and **97.68%** of val rows once the label rule is corrected; 8,231/8,400 train rows
carry a scoreable element label.
→ `2026-11-07-candidate-supervision.md`

**4. A shortlist filtered by the model's own prediction keeps the answer at high rate.** Using stage-1
predictions over 15,000 training rows (gold-free):

| R | median list | gold inside |
|---|---|---|
| 300 | 5 | 87.9% |
| **450** | **8** | **91.3%** |
| 600 | 12 | 94.1% |

On the 398 eval rows the same construction gives 78.5% at R=450 — **the eval rate is lower than the train rate**,
which is itself a warning that the two distributions differ.

## What was RETRACTED today (and why it matters)

| claim | why it died |
|---|---|
| "VA beats V at matched weights: +6.03, p=1.5e-3" | that was a **300-step** checkpoint; at 938 steps VA loses −3.75/−7.29 |
| "the candidate interface fails" | it fails **only with long lists**; at 300 steps it won |
| "short lists improve grounding (+3.8/+10.0)" | **did not replicate on test** (−4.0/−7.3) |
| "candidate coverage ceiling is 99.99%" | that counted *point inside some candidate*; the **scoreable** ceiling is 97.99% |
| "text overlap ranks the gold top-5 at 82.5%" | a tie-handling bug; the real number is 38.8% |
| "the ensemble gives +2.01" | at matched 3-voter size it is **+0.50**; after a 16-cell correction p = 0.17 |
| "+8.10 from combining the two interfaces" | that is an **oracle**; the best free arbiter gives +0.60, p = 0.68 |

## What is NOT established, and what would establish it

| open question | the cheapest experiment | predicted |
|---|---|---|
| Does a **two-stage shortlist** (R≈450, ~8 items) beat 56.05%? | stage-2 selector trained on the 15k shortlists already built | 62–65% if the selector reaches ~80%; **the 80% is unproven** (test says 63.4%) |
| Did the fine-tune **trade grounding for verb**? | run Qwen3-VL-2B/8B through **our scorer** on our 398 rows (~1 GPU-h) | base grounding reported at 59.6–69.06 elsewhere; our 56.05 would be **below** a base 2B |
| Is a **bounded coordinate loss** enough on its own? | retrain with spatial-softmax instead of per-edge Gaussian | unknown; no isolated published delta |

## The one number that would change the paper

Our grounding (56.05%) is **below** the 59.64–69.06 reported for **untuned** Qwen3-VL-2B/4B/8B on the same
protocol (Mem-W aggregate, denominator unstated). If that holds when measured through **our** scorer, the honest
sentence is:

> *Our fine-tune bought +18.6 points of step accuracy over a same-size base model and **gave up ~10 points of
> element grounding** to do it — the trade, not the accuracy, is the finding.*

That is a paper-worthy claim, it needs **one GPU-hour** to verify, and nobody has run it.
