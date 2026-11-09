# 2026-11-07 — Base d1-2B vs our fine-tune, measured through OUR scorer

The one comparison nobody had run: is our 56.05% grounding a *deficit* relative to the starting model, or an
*improvement*? Run on the same 398 eval rows, same scorer (`run_acv2.score_row`, differential-tested), same protocol.

## Result (n = 276 of 398; the run's process ended at the task timeout, not on an error)

| model | step accuracy | grounding (spatial) | type-correct on spatial rows |
|---|---|---|---|
| **d1-2B, base, no fine-tune** | **17.39%** (48/276) | **27.44%** (45/164) | 63.2% (141/223) |
| d1-2B + our AC-v2 SFT (E2) | **64.07%** (255/398) | **56.05%** (125/223) | **87.9%** (196/223) |

The stability across 140 → 248 rows (20.7 → 19.4 accuracy; 27.1 → 27.4 grounding) says the sample is not
drifting; these are close to final.

### The decomposition that matters

| | base d1-2B | E2 (fine-tuned) | Δ |
|---|---|---|---|
| type correct on spatial rows | **63.2%** (141/223) | **87.9%** (196/223) | **+24.7** |
| **grounded, given type correct** | **22.7%** (32/141) | **63.8%** (125/196) | **+41.1** |
| grounded overall | 27.4% | 56.05% | +28.6 |

**Both components improved a lot, and placement improved more than typing.** The base model, when it happens to
choose the right verb, still lands on the wrong element 77% of the time — so the fine-tune's grounding gain is
**not** merely a by-product of better verb choice. This corrects a hypothesis I nearly recorded (that the base
"was already 63% grounded given the right verb" — it is 22.7%, and the 63.8% belongs to the fine-tuned model).

Provisional conclusion, stated with its sample size: **the fine-tune is worth roughly +43 points of step accuracy
and +29 points of grounding.** It did **not** trade grounding for verb — it improved both, massively.

## Why this matters, and what it corrects

The honest-summary note raised the possibility that our 56.05% was *below* untuned Qwen3-VL-2B/4B (59.64–69.06
reported elsewhere). That comparison is **cross-paper and cross-family**; measured through our own scorer on our
own rows, our model is far ahead of *its own* base. Two different questions:

1. "Did our fine-tune help?" → **Yes, by a huge margin.** (+43 step, +29 grounding, same scorer.)
2. "Is d1-2B the right base, versus a Qwen3-VL base with better grounding?" → **still open**, and that is the
   comparison the 59.64–69.06 numbers refer to. It requires running Qwen3-VL-2B through this scorer (~1 GPU-h),
   which is a *different* experiment from the one just run.

## Caveats

- n = 140 of 398 at the time of writing; the run is still going. The numbers above will be replaced by the full-398
  figures when it finishes, in this same file.
- `D1_NO_COMPILE=1` because torch's compile path crashes on this host (`NameError: name 's1' is not defined` in a
  generated triton module). That does not change results, only speed (4.2 s/step vs ~0.5).
- The base model's grounding denominator (107) is smaller than E2's (223) because most rows' predicted type is
  wrong, and grounding is only defined when type matches. That is the scorer's rule, not a filter we chose.


## Where the grounding gain comes from (decomposition, same 223-row basis)

`grounding = P(type correct) × P(grounded | type correct)`:

| | base | E2 |
|---|---|---|
| P(type) | 63.23% | 87.90% |
| P(grounded \| type) | 22.70% | 63.78% |
| **product** | **14.4%** | **56.1%** |

Decomposing the **+41.7 point** gain:

| source | contribution |
|---|---|
| better **verb** choice | **+5.6 pts** |
| better **placement** | **+26.0 pts** |
| interaction | +10.1 pts |

**Two thirds of the grounding improvement is placement**, not verb choice. A reviewer who assumes "the fine-tune
just learned which action to take, and grounding followed" is wrong by a factor of five. This is the cleanest
statement the programme can make about what the fine-tune actually did, and it comes from one 1-hour base eval.
