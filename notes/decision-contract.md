# Decision Model — the Single-Pass Input/Output Contract

> **This is the core primitive of the whole project.** Everything (Jev, d1, our mobile action-selection head) is an instance of this contract. Read it before any other note.
> Visual explainer: `docs/visual/single-pass-contract.html`.

## 1. Definition
A **decision model** reads an unstructured **state** plus typed **questions** and, in **one forward pass**, returns **calibrated probabilities over a fixed answer set** per question — **zero output tokens**. It does **not** generate text.

- "Single pass / single phase" = the network reads the state **once** and directly scores every option in its last layer → softmax → probabilities. **No autoregressive token-by-token generation.**
- "Zero output tokens" = `usage.output_tokens == 0`; nothing is text, so nothing needs parsing.

## 2. Input (one call)
| Part | What it is | Examples |
|---|---|---|
| **state** | unstructured, any mix | text string, JSON payload, image, app UI tree + screenshot, audio |
| **questions** | typed, fixed answer set | `choice`, `score`, `noul` (see §4) |

For our mobile-agent feature, the input is:
- **state** = the app's UI (accessibility tree + screenshot) + the task/goal,
- **question** = a `choice` over the **K candidate actions**.

## 3. Output (one call)
| Part | What it is | Notes |
|---|---|---|
| **probabilities** | one per option, summing to 1 | e.g. `[0.72, 0.05, 0.11, … 0.02]` |
| **confidence** | a scalar from the distribution | margin `p_max − p_second`, or `(n·p_max − 1)/(n − 1)` |

The confidence is what powers the **gate**: act (high confidence), ask user (ambiguous), refuse/escalate (can't proceed).

## 4. The three primitives
| type | question | output |
|---|---|---|
| `choice` | pick one from K | vector of K probabilities |
| `score` | rate on a rubric (e.g. 2–10) | probability-weighted level |
| `noul` | yes/no | P(yes), P(no) |

## 5. Concrete example (mobile action-selection)
```
Input  state = UI tree of the screen + goal "process this refund"
       question = choice("which action?", options = [Tap 'Refund', Tap 'Revert',
                    Tap 'Contact support', ..., Refuse])
──► ONE FORWARD PASS ──►
Output probabilities = [0.72, 0.05, 0.11, ..., 0.02]
       confidence     = 0.72 (margin)  → gate → ACT on 'Refund'
```
If confidence were ~0.4 across options → **ask user**; if it can't safely proceed → **refuse / escalate**.

## 6. Why single-pass matters (the whole thesis)
1. **No text generation** → no parsing, no hallucinated strings, no type errors (schema-guaranteed).
2. **Calibrated** → confidence tracks accuracy (ECE / Brier defined in `notes/calibration.md`).
3. **Single-pass, parallel** → very low latency (tens of ms on edge; d1-3B: 8 ms RTX 4090 → 50 ms Jetson Orin Nano).
4. **Structured output** → software can branch directly (classify / route / score / guardrail).

## 7. Decision model vs LLM
| | LLM | Decision model |
|---|---|---|
| Output | generated tokens/text | probability vector over fixed set, zero tokens |
| Sampling | sequential, autoregressive | parallel, one pass |
| Parse needed | yes | no (typed by construction) |
| Calibration | overconfident / inconsistent | calibrated by design |
| Latency | seconds | tens of ms |

Sources: `[typesafe2026systemone]`, `[liquidAI2026opend1]`, `notes/d1.md`, `notes/jev.md`. See `references/`.
