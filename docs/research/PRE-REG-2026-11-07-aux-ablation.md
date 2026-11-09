# PRE-REGISTRATION — is the +6.01 grounding gain from the aux data? (2026-11-07)

*Frozen before the run.*

## Why

`groundB` reached **69.79%** grounding vs E2's **63.78%** — the only measured grounding improvement in this
programme (+6.01, same scorer, same rows). But it changed three variables at once: `aux7k` auxiliary grounding
data, +468 training steps (938 → 1,406), and 858 px images (720 → 858). None of the three is attributable.

## Arm (one change from E2)

| | E2 baseline | arm |
|---|---|---|
| data | `cur15k` | **`cur15k,aux7k`** |
| steps | 938 | **938** (unchanged) |
| image width | 720 | **720** (unchanged) |
| everything else | init, lr 1e-5, warmup 30, seed 0, micro 1/accum 16, grad-ckpt, freeze-embed | identical |

So the only difference is the auxiliary grounding samples: 7,000 extra examples whose state is an element's
instruction or label and whose target is that element's box, on screenshots the model already trains on.

## Pass condition (frozen)

- **PASS:** grounding ≥ **62.0%** on the 223 spatial test rows (halfway to groundB's 69.79), **and** step accuracy
  ≥ **63.5%** (groundB kept 64.82; E2 is 64.07 — the aux data must not buy grounding with accuracy).
- **KILL:** grounding < **58%** → the +6.01 came from the extra steps or the resolution, not the data, and the
  "cheap data lever" story dies. Report as a negative and stop spending on data work.
- **Diagnostic:** report grounding *given type-correct* separately from overall grounding, because aux data
  targets selection, not verb choice. If grounding-given-type rises but overall does not, the aux data helped
  placement while the type head drifted.

## Prediction

**Grounding 62–66%, accuracy 63.5–64.5%.** Reasoning: groundB's +6.01 cannot all be data (the extra 468 steps are
worth +2–3 on their own by the earlier paired measurement of E2@705 → E2@938 = +3.52 on the same rows), so data
alone should be roughly +3 to +4, and the resolution change contributes the remainder.

## Cost

One 938-step run, ~1.6 h on a 4090, plus one eval (827 val + 398 test, ~4 min).

## What invalidates it

- Adding steps or changing image width.
- Using a different base than `LiquidAI/d1-2B`.
- Evaluating test more than once.

---

## ADDENDUM (written after the first result, before the control run)

The first run gave **+2.13 grounding** (66.49% vs 64.36%), not significant (z ≈ 0.58). A design flaw was then
found in my own conclusion: **`aux7k` adds 7,500 rows to 15,000 — a 50% increase in training rows.** So the gain
could be *more gradient on the same task* rather than *different supervision framing*.

**Control run, pre-registered here:** `cur15k` duplicated/shuffled to **22,500 rows** (same total row count as
`cur15k,aux7k`), same init, same 938 steps, same 720 px, same everything else.

- **If the control also reaches ≥66% grounding** → the `aux7k` effect is **data volume**, the "free framing lever"
  claim dies, and the paper must say "more rows of the same task" rather than "re-framing".
- **If the control stays at ≤65%** → the framing claim survives, and `aux7k` is a genuine free lever as the first
  result suggested.

This control is mandatory before any `aux7k` number enters the paper, and it is one run (~1.6 h).
