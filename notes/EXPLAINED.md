# The Whole Project, Explained One Time (plain language)

*Written 2026-10-13. Read this file first. Every other note, plan, and number
in this repo serves what is written here. If something later sounds different
from this file, this file wins.*

---

## 1. What we are building

A small **"decision brain"** that plugs into a mobile agent.

When a phone agent must choose its next action ("book a taxi"), our brain:

1. looks at the situation (goal + current instruction),
2. looks at a few real candidate actions,
3. does **one** forward pass (no writing, no thinking-out-loud),
4. returns **one honest number per candidate** + a decision: **ACT / ASK / REFUSE**.

It runs on the phone itself (your Mac for now): ~18 ms, ~1.2 GB, offline.

**In one sentence:** we make a tiny model that knows what it knows, so a mobile
agent can be trusted to act, or honestly stop and ask.

## 2. Why (three facts)

1. **Wrong taps are irreversible.** On a phone, "send payment" cannot be undone.
   An agent that is *usually* right but *sometimes* catastrophically wrong is
   not deployable. It must know when to stop.
2. **Big LLMs can't be trusted for that.** They write paragraphs. They don't
   hand software an honest probability it can set a risk budget on. And they're
   slow/expensive on-device (10–100× our numbers).
3. **The new "decision models" claim to be honest — but publish their
   failures.** The best open one (decider-2b) **failed its own calibration
   bar** on hard inputs ("released although it did not pass"). The best of 17
   frontier agents scores only **59.5%** at knowing when *not* to act. The
   benchmark we use (AndroidControl) caps even GPT-5 at ~61%.
   → **Nobody has solved "honest small decision models for on-device agents."
   That is our gap.**

## 3. How it works (the harness = 4 logic layers)

```
situation + K real candidate actions (+ "none of these are right")
   │
   ▼
[1] CANDIDATES — real actions from real data, shuffled (kills position
    bias), plus an explicit "none are right" option.
   ▼
[2] DECISION HEAD — 1.2B model, ONE pass → probability per candidate.
    Trained with PROPER SCORING: it is rewarded for an HONEST distribution
    (saying 90% when it's really 90%), not just for picking the right one.
    (Plain fine-tuning only rewards the right pick → the model lies
    confidently. That's the difference.)
   ▼
[3] RISK-BUDGET GATE — the math:
        act   only if  confidence ≥ threshold  AND acting is worth more
               than asking/ refusing (wrong-action cost matters:
               "send payment" needs a higher bar than "scroll")
        ask   if confidence is low but a human could fix it
        refuse/stop  if nothing fits
    Threshold is SET so that executed actions are wrong ≤ 10% of the time.
   ▼
[4] TWO MODES — (a) standalone on-device agent, or (b) SHIELD: bolted onto
    ANY existing LLM agent (the LLM proposes, our brain scores + gates).
    Mode (b) = "makes an existing mobile agent better" directly.
```

Where each piece of logic comes from (our research, mapped 1:1):
shuffle & no-shortcuts → d1's recipe · proper scoring → THX-01's published
reward · the gate → CORA + CAAG + UniCR · abstain option → decider ·
single-pass readout → Jev/d1/THX-01 shape.

## 4. The calibration idea (the one concept that matters)

*Calibration = the model's confidence number matches reality.*

Our real measurements (LFM2.5 on real Android steps):

| | before | after calibration |
|---|---|---|
| right 57% of the time | says "67%" (lying) | says "55%" (honest) |
| says "70%" → really right | ~74% | ~89% |
| ECE (distance number↔reality) | 0.10 | 0.055 |

Calibration doesn't make the model smarter — it turns the confidence into a
**dial the software can set**: "act only when I'll be wrong ≤10% of the time."
No honest number → no dial → no safe agent.

## 5. The benchmark & what "better" means

**Benchmark: AndroidControl** (Google, official paper NeurIPS 2024; real human
phone steps; 12,232 train / 3,051 test). Context scoreboard (their harder
screenshot protocol, May 2026): SOTA 68.6%, GPT-5 61.3%, a 2B model 45.5%.

**Our protocol (same for everyone, fair):** text situation + 4 real candidates
→ pick the right one, one pass.

**"The old" = B3:** a big LLM (Qwen3-14B) picking blind, same candidates.
**"Us" = H3:** the harness above.

We win if (fixed pass conditions, see the plan):
- **P1** ≥ accuracy of the 14B LLM (fallback: within 3 points + win P2/P3)
- **P2** executed actions wrong ≤10% of the time, while still handling ≥50%
  of the steps — the 14B LLM **cannot do this at all** (no honest dial)
- **P3** ≥10× faster on-device
- **P4** one command reproduces every number

**Value proof (ablations):** each logic layer gets its own number —
proper scoring (V2), the gate (V3), small-head-vs-big-LLM (V1), data scale (V4),
built-in vs patched calibration (V5).

## 6. The paper (what we will announce)

**Title: "Calibrated to Be Trusted: Training Single-Pass Decision Models for
Risk-Controlled On-Device Action Selection"**

Three claims, each testable:
- **C1** — fine-tuning the normal way breaks honest confidence;
  proper-scoring training keeps it (measured on hard inputs).
- **C2** — at "wrong ≤10%", our 1.2B harness safely acts on ≥50% of real
  mobile steps at ≤100 ms on-device; the big LLM can't meet that budget.
- **C3** — everything reproducible: open code, weights, benchmark.

Headline = the benchmark result (us vs the old). Calibration = the mechanism.

## 7. Where we are now (real state, 2026-10-13)

| Piece | Status | Number |
|---|---|---|
| Harness core (metrics, gate math) | ✅ done, tested | 13/13 tests |
| Decision head on this Mac (LFM2.5-1.2B) | ✅ runs | 18 ms/decision |
| Baseline: frozen head | ✅ done | 57.3% acc, ECE 0.10 (lying) |
| Baseline: + post-hoc calibration | ✅ done | ECE 0.055 (honest) |
| Official field benchmark (Decision Index) | ✅ our model runs it | real run done, 2/8 HellaSwag = chance (honest) |
| Full AndroidControl data (train+test) | ⏳ downloading | 15,283 real steps |
| **B3: the 14B "old" baseline** | ⬜ next | the bar we chase |
| H1/H2: trained heads (3 objectives) | ⬜ after B3 | |
| H3: full harness + gate | ⬜ after training | **the headline row** |
| Ablations V1–V5 → final table → paper | ⬜ | |

## 8. Glossary (keep the names straight)

| Term | Plain meaning |
|---|---|
| **Decision model** | small AI that picks from given options in one pass; never writes text |
| **Calibration** | confidence number matches real success rate |
| **ECE** | the measured gap between number and reality (0 = honest) |
| **Coverage** | share of steps we dare to act on |
| **Risk** | share of the actions we took that were wrong |
| **Act / Ask / Refuse** | the three honest moves; the gate picks one |
| **Proper scoring** | training reward that only pays out for honest probabilities |
| **The old (B3)** | big LLM picking blind — our baseline to beat |
| **H3** | our full harness — the headline result |

**That's the whole project.** Every task on the board exists to fill in row H3
and prove it beat row B3.