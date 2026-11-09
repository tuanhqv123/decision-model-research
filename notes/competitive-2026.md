# Competitive landscape 2026 — what exists, and where our gap is

*Retrieved 2026-10-13. Every entry has a bibkey (see `references/bib.bib`).
Rule: before writing any related-work paragraph, re-verify each entry — this
field moves weekly.*

---

## A. System One / decision-model vendors & open reproductions (the model side)

| Work | Org | Base / size | Training | Calibration evidence | Venue | bibkey |
|---|---|---|---|---|---|---|
| Jev 1.13 | TypeSafe (proprietary) | undisclosed | RLCD (**unpublished**) | claims calibrated; 70–500 ms; "can't hallucinate" | hosted | `typesafe2026systemone` |
| d1-3B / d1-omni | Liquid AI (open weights) | LFM2.5-VL-3B / encoder | weight-averaging + seed/mixture merge + SFT (shuffle, shortcut removal) | DI 48.57 self-scored; RLVR-collapse finding | 8 ms RTX 4090, 16–50 ms Jetson | `liquidAI2026opend1` |
| **THX-01** | HAL-X AI (open, Apache-2.0) | mmBERT 322M encoder + 15M head | **8 stages of RLCD**: reward = strictly proper scoring rule (log + spherical + RPS), zero-variance gradient; ~2.1M training decisions | ECE 0.003 (tickets) / 0.066 mean (39 tasks, down from 0.204); **selective automation: ~75% zero-error, 90% at ≥99.5% accuracy** | ~10 ms on one **NVIDIA GPU** | `thx012026` |
| **decider-2b family** (0.8B/2B/4B/35B-A3B/+vision) | Mapika (independent, open code+weights+data registry) | Qwen3.5-2B-Base etc. | (1) CE + proper scoring on ~95 public decision datasets; (2) **PPO + proper log score + KL-to-v8** on live games/browser; (3) LoRA r64 + **per-type temperature** (choice 1.164 / noul 1.624 / score 1.124) | in-task ECE 0.038 / held-out 0.083; JevBench hard 0.577; **hard items ECE 0.156 vs own release bar 0.08 — "released although it did not pass", "still overconfident on hard items"** | 4 ms on one **NVIDIA GPU** | `decider2026` |
| this-that-model-1.0 ("the smart if-statement") | unknown (arXiv preprint) | 2B typed-decision | n/v (not yet read in depth) | 30.9 ms on one laptop GPU | GPU | `thisthat2026` |
| kotoba typed-decisions | kotoba-lang (open) | ModernBERT / DeBERTa / LLaDA-MoE | open recipe, Jev-shape (Choice/Score/Noul) | measured accuracy, calibration, training cost (not yet read in depth) | `kotobadecisions2026` |
| typed-lm (docs) | neurono-ml | — | n/v | documents the class + scoring | `typedlm2026` |

**Takeaway:** the *recipe* race is on (CE → proper scoring → RL with proper
scores → per-type temperature). "First open calibration recipe" is **taken**.
But nobody studies *when calibration breaks* (Q1) nor does the on-device
venue (Q3).

## B. Benchmarks of the decision-model class

| Benchmark | Scope | What it measures | bibkey |
|---|---|---|---|
| Decision Index (TypeSafe, v0.3) | 37 benchmarks, 5 areas, gold ★, chance-corrected, private split | decision accuracy (calibration NOT the focus) | `decisionIndexKit` |
| **TypedBench** (Oct 2026) | System One class: hosted model + open encoder + open decoders 0.8–9B | **calibration error vs finite-sample noise floor**, paraphrase/framing sensitivity, probability quality (selective prediction, proper scoring, asymmetric cost) — benchmarks *existing models*, does not study training recipes | `typedbench2026` |
| JevBench (public) | Jev-class models | accuracy incl. hard tier (decider v11: 0.577 hard) | `jevbench2026` |

## C. Agent abstention / act-when-uncertain (the agent side)

| Work | What | Key number | bibkey |
|---|---|---|---|
| **AgentAbstain** (Jul 2026) | first systematic framework: do tool-using LLM agents know when NOT to act? near-identical act/don't-act pairs | **best of 17 frontier models: 59.5%** | `agentabstain2026` |
| **CORA** (Apr 2026) | conformal risk-controlled mobile GUI agent; post-hoc Guardian shield over VLM policy, execute/abstain threshold | risk-controlled pre-action safety | `cora2026` |
| **CAAG** (May 2026) | Calibrated Act–Ask–Abstain Gating: calibration head (CE + α·ECE loss) + expected-utility gate + memory-triggered reflection | **simulated results only** ("live validation remains essential future work") | `caag2026` |
| SeerGuard (Jul 2026) | consequence-aware pre-execution screening via world-model prediction | safety framework | `seerguard2026` |
| CONFLICTGUI (Sep 2026) | conflict-aware termination (instruction-internal & instruction-GUI conflicts) | execution-bias overcompliance found | `conflictgui2026` |
| UniCR (Sep 2025) | unified framework: heterogeneous uncertainty → calibrated P(correct) → risk-budget refusal | answer-time refusal, not action selection | `unicr2025` |
| Conformal line (ConU 2024, SConU 2025, CIC 2026, conformal abstention 2024) | distribution-free coverage guarantees, post-hoc | statistical guarantees over black-box LLMs | `conu2024`, `conformalabstain2024`, `sconu2025`, `cic2026` |
| ProbeCal | calibration failures in tool-use traces; probing to align probabilities with tool effectiveness | tool-use calibration | `probecal2025` |

**Takeaway:** the agent side is all **post-hoc** (shields, conformal layers,
separate calibration heads over frozen backbones) — nothing trains the
decision model itself. CAAG is closest to us in framing (act/ask/abstain +
calibration loss) but has **no real experiments** and targets generative LLM
agents, not single-pass decision models.

## D. The template (claim structure, not topic)

- **Re-TRAC** (Microsoft, Feb 2026): ReAct linearity → incomplete branches
  (93% of failures) → recursive trajectory compression → small 4B/30B models
  beat 229B–358B on BrowseComp/GAIA/HLE. Structure: *named failure → mechanism →
  "small + mechanism ≥ big" on a defined axis.* Our paper copies this structure
  with axis = **reliability-per-cost on-device**, not raw accuracy. [`retrac2026`]

---

## E. The gap we own (intersection of A+B+C, after 2026-10-13 sweep)

1. **Q1 breakdown regime:** no systematic study of *when/why* single-pass
   decision models miscalibrate (hard/OOD/confusable items). Published
   evidence of breakdown exists (decider 0.156 hard ECE; THX-01 DAIR 0.284;
   AgentAbstain 59.5%) but is unanalyzed.
2. **Q2 mechanism comparison:** CE-only vs proper-scoring vs joint-calibration
   objectives never compared on identical data/models with easy/hard splits.
   (THX-01 = encoder + closed data; decider = full recipe but no regime study;
   CAAG = simulated.)
3. **Q3 on-device venue:** every model above reports NVIDIA GPU latency.
   Apple-silicon / no-GPU reliability + cost for act/ask/refuse action
   selection is uncharted.

**Paper = the intersection of 1×2×3.** See `notes/GOAL.md` §3 for the method
and §1 for the title.
---

## F. UPDATE 2026-10-28 — the field grew (re-verified after user challenge)

**Scale:** JevBench (benchmarkheaven.com/jev-models) ranks **232 open-weights decision systems** (+ Jev 1.13/1.14 refs). **ImageJevBench** (v0.3.0) ranks **64 image decision models**. **Jev Decision Index — Vision** (metatext.io, 39 models, Oct 2026) top rows: amos-1-large 72.4 (ECE 0.024), vLLM-SR d3-27B 71.3, Perplexity Decider v1.1-27B 70.6, JEV-27B-VL 69.6, OneJev1.5-27B 68.9; **small/vision**: d3-mini-4B 63.2, Winnow-12B 58.8, RSI-Jev v6.1 VL-4B 57.4 (ECE 0.041), RSI-Jev v5.0 VL-3B 57.3 (ECE 0.023), Intern-Decision-4B 56.5 (ECE 0.014), **d3-nano-2B 54.7**, **Intern-Decision-2B 47.8** (ECE 0.032). **d1-2B/d1-3B do NOT appear on this vision board** — Liquid has not ranked them there; d1's vision decision capability is unbenchmarked on the independent board.

**New entrants since 2026-10-13 sweep:**
| Work | Org/date | Base/size | Notes |
|---|---|---|---|
| Typical (small 1.7B / medium 4B) | Oz Labs (CSIRO), Sep 24 2026, open weights + paper | text-only decision (state→KV cache→question suffix→scoring head; choice/noul/score/**abstain** built in; label-invariant noul head) | JevBench std: small .792; "4 open + 1 closed ahead of medium"; honest stats (±6–13pt cluster bootstrap). **No vision** (DOOM demo = frames as text). |
| Strands Decider 2B | AWS Strands, Oct 1 2026, Apache-2.0, full data+scripts open | Qwen3.5-2B torso (LM head removed) + **pointer head ~1M params**; torso fine-tuned with **LoRA r16**; v19 of iterations | 115ms RTX3090; JevBench 3rd of 33 in 2B class; **text-only**; use case = agent guardrails (before_tool_call noul checks). Architecture = the "easy" variant (text torso + tiny head). |
| CoPE-B-S1 | Zentropi, Oct 22 2026 | System One spec, safety-tuned, self-hostable | policy decisions, multiple labels one pass. |
| GLiNER2.5-Decide | Fastino, Sep 24 2026 | 340M open weights | schema-defined decisions + span extraction + constraint feasibility. |
| ej | (freshweights) | 11.4MB, Apache-2.0, CPU-only | 214ms warm; few-shot workflow adaptation. |
| decider-2b-vision | Mapika (Oct 2026), Apache-2.0 | Qwen3.5-2B **VLM** (v5 lang weights transplanted) | **image + lettered-option questions → calibrated probs in one pass**; "256x240 game frame = 64 visual tokens"; decider-2b v12 (Oct 20) state may include images; honest: hard-item ECE 0.18–0.31 across versions. **No box/coordinate primitive — choice over options only.** |

**Implications for our base-model choice (2026-10-28):**
1. "Only d1 has vision" (true on 2026-10-13) is now stale — ~39 ranked vision decision models exist, incl. 2B (d3-nano 54.7, Intern-Decision-2B 47.8) and 3B (RSI-Jev v5.0 57.3).
2. What d1-2B uniquely offers for AC-v2 is still the **native `box`/`boxes` coordinate-grounding primitive** (pure-vision click targets). The ranked vision models do image→decision, not element grounding.
3. But our own zero-shot data says the **choice-over-K (a11y) interface is where 2B single-pass is strongest** (d1-2B ax 28.39% > vision 15.83%). Any image-capable decision model (e.g. decider-2b-vision, built exactly for "image + lettered options") is a candidate for that protocol.
4. **De-risking test before fine-tuning (4090, hours):** run decider-2b-vision zero-shot on the SAME 398 ax protocol (same candidate picker + scorer). If ≥ 28.39% → base choice is contestable (and its Apache-2.0 + open recipe + Qwen VLM base may be the better substrate); if clearly lower → d1-2B stands.
5. Paper positioning unaffected/strengthened: 232+ open decision models, none does **risk-controlled on-device GUI action selection with a calibrated completion gate**; Q1/Q2/Q3 gap intact. New citations for Q1: decider hard-ECE 0.18–0.31, Strands/Typical abstinences.
