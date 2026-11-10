# T4 — Abstention prior art (mechanical extraction)

Extraction only. Every claim below is quoted from the fetched text; nothing is inferred.

**Fetch status (2026-11-09):** all four URLs resolved. Full HTML was fetched and read for all four
(no entry is abstract-only).

| Source | What was fetched |
|---|---|
| CORA, arXiv:2604.09155v1 | abstract page + `https://arxiv.org/html/2604.09155v1` (full text) |
| SafeGround, arXiv:2602.02419v2 | abstract page + `https://arxiv.org/html/2602.02419v2` (full text) |
| TypedBench, arXiv:2610.11392v2 | abstract page + `https://arxiv.org/html/2610.11392v2` (full text) |
| GUI-JEPA, arXiv:2610.36877v1 | abstract page + `https://arxiv.org/html/2610.36877v1` (full text) |

---

## Table

| Source (arXiv id) | Publishes execute/abstain or act/ask/refuse? | Mobile / GUI setting? | Metric it reports for abstention | Verbatim evidence (<=25 words) | URL |
|---|---|---|---|---|---|
| **CORA** (2604.09155) | **Yes** — explicit `execute/abstain` boundary, with abstentions routed to Confirm / Reflect / Abort | **Mobile GUI** — Phone-Harm (Harm-150 + Normal-150, 29 apps on Android sandbox), MobileRisk, AndroidWorld core20 | Harm Rate (HR), Macro Harm Rate (mHR), Goal Achievement Rate (GAR, defined as the execution/coverage rate), Intervention F1, Over-Intervention Rate (OIR), Diagnostician invocation rate; plus risk–coverage sweeps | "we leverage Conformal Risk Control to calibrate an execute/abstain boundary that satisfies a user-specified risk budget" | https://arxiv.org/abs/2604.09155 |
| **SafeGround** (2602.02419) | **Yes, but per-click** — accepted vs rejected grounding prediction; rejected ones are abstained from or escalated to a stronger model. No ask/refuse-to-user action | **GUI, not mobile** — ScreenSpot-Pro (professional desktop software screenshots) | FDR among accepted predictions (with Clopper–Pearson upper bound / LTT), power, cascading rate; AUROC and AUARC for the uncertainty score | "predictions with uncertainty ≤τ are executed directly, while high-uncertainty cases are abstained or cascaded" (Fig. 2 caption) | https://arxiv.org/abs/2602.02419 |
| **TypedBench** (2610.11392) | **Partial** — selective prediction / risk–coverage / abstention on out-of-scope items is a reported metric, and abstention is listed as a published adapter capability. No act/ask/refuse policy result | **No** — text-only typed-decision tasks (support tickets, phishing email, RAG relevance, log triage, policy compliance, guardrail intent) | Risk–coverage curves, AURC, coverage at ≤5% risk; abstention rate on out-of-scope items; escalation threshold in a two-model cascade | "Risk–coverage curves, AURC and coverage at ≤5% risk" | https://arxiv.org/abs/2610.11392 |
| **GUI-JEPA** (2610.36877) | **Unclear / deferral only** — a top-2-margin gate routes uncertain steps to a second model. It never refuses or asks; the action is still taken, by the 7B | **Yes** — AndroidControl (mobile GUI), a11y-tree candidates | AUROC of the top-2 margin for its own correctness; accuracy at coverage quantiles; accuracy/latency of the 50 %-coverage cascade | "escalated to the 7B SFT agent otherwise. The margin predicts whether GUI-JEPA is correct with AUROC 0.74–0.77" | https://arxiv.org/abs/2610.36877 |

---

## Q1. Does CORA report a risk-controlled (conformal or otherwise guaranteed) execute/abstain result on mobile GUI? If yes, on which dataset and what risk/coverage numbers?

**Yes.** Mobile GUI, conformal risk control (CRC), and numeric risk/coverage are all present.

Dataset / setting, verbatim:

> "We ablate the four architectural components of CORA (e.g., CRC, action-conditioning of the Guardian, the Generative Diagnostician, and Goal-Lock) on a held-out calibration pool (n=1075, target risk δ=0.05), sweeping the decision threshold τ∈{0.5,0.6,0.7,0.8,0.9}"

https://arxiv.org/html/2604.09155v1

The CRC-selected operating point and its risk/coverage numbers, verbatim:

> "In contrast, the CRC module adaptively selects τ̂=0.952. This optimally calibrated threshold achieves the best coverage (89.95%) and the lowest invocation rate (10.05%) among the evaluated settings, while constraining the executed-harm rate to 2.42%—safely satisfying the conformal risk target (≤5%)."

https://arxiv.org/html/2604.09155v1

The main-table result on Phone-Harm Harm-150, verbatim:

> "CORA (Ours) | 4.37 | 79.80 | 85.29 | 5.34 | 19.23"
> (columns: HR, GAR, IF1, mHR, OIR)

https://arxiv.org/html/2604.09155v1

Two limits of what the fetched text states:

- The risk budget α used for the Table 1 / Table 2 Phone-Harm results is **not stated** in the main text or those table captions. The only numeric budget stated is δ=0.05, in the ablation and in the appendix sweeps.
- The 89.95 % / 2.42 % pair is reported on the held-out calibration pool (n=1075), **not** on Phone-Harm Harm-150. Phone-Harm appears in the guarantee claim only as prose: "meets its conformal risk target on held-out test data".

---

## Q2. Does any fetched source report an OUTCOME-level abstention result (measured task success as a function of when the agent abstains)?

**Partial — one report of task success under a controller, none as a function of the abstention threshold.**

CORA reports AndroidWorld core20 task success with the controller wrapping a base policy, verbatim:

> "Wrapping the unconstrained AutoGLM policy with CORA raises its success rate from 30.0% to 40.0%, recovering 72.7% of the GPT-5 upper bound and surpassing UI-TARS-1.5 by 20 absolute points."

https://arxiv.org/html/2604.09155v1

Table 3 of that paper, verbatim rows:

> "AutoGLM (Prev setting) | 6/20 | 30.0% | 54.5%"
> "CORA (Ours) | 8/20 | 40.0% | 72.7%"

https://arxiv.org/html/2604.09155v1

This is a controller-on vs controller-off success comparison. The fetched text does **not** sweep the abstention rate and plot task success against it.

CORA's own naming for its abstention-side metric is not task success. Verbatim:

> "Goal Achievement Rate (GAR). GAR is the execution (coverage) rate:"

https://arxiv.org/html/2604.09155v1

SafeGround, TypedBench and GUI-JEPA report only per-click, per-item or per-step quantities (FDR, power, cascading rate, coverage, step accuracy). None of them reports measured task success as a function of abstention.

---

## Q3. Does any state a numeric confidence threshold or a measured ask rate?

**Yes — both, in CORA; numeric thresholds also in SafeGround, GUI-JEPA and TypedBench.**

Numeric threshold plus measured invocation ("ask") rate, CORA, verbatim:

> "the CRC module adaptively selects τ̂=0.952. This optimally calibrated threshold achieves the best coverage (89.95%) and the lowest invocation rate (10.05%) among the evaluated settings"

https://arxiv.org/html/2604.09155v1

The swept thresholds and their invocation rates, verbatim:

> "increasing τ yields the expected safety-autonomy trade-off: task coverage improves from 85.77% to 89.02%, while the executed-harm rate rises from 0.56% to 1.86%, and the Diagnostician invocation rate drops from 14.23% to 10.98%."

https://arxiv.org/html/2604.09155v1

CORA's confirmation rate by design, verbatim:

> "The moderate Over-Intervention Rate of 19.23% is an explicit, tunable consequence of the conformal risk budget"

https://arxiv.org/html/2604.09155v1

SafeGround, numeric risk level and deferral fraction, verbatim:

> "with Holo1.5-7B at risk level 0.34, the system achieves 58.66% accuracy, outperforming Gemini-only inference by 5.38%"

https://arxiv.org/html/2602.02419v2

GUI-JEPA, threshold set by coverage quantile, verbatim:

> "Setting τ at coverage quantiles (the share of steps kept by GUI-JEPA) traces the curve of Figure 11"

https://arxiv.org/html/2610.36877v1

TypedBench, an explicit numeric threshold with a measured escalated share:

> "on phishing, escalating a third of the English checkpoint's answers (threshold 0.7) closes half of its accuracy gap to the hosted model."

https://arxiv.org/html/2610.11392v2

---

## Q4. Does any price the cost of an ask (model the trade-off between asking and erring)?

**NOT FOUND IN FETCHED TEXT for a human ask.** No fetched source assigns a cost to asking a user, and none writes a decision rule of the form "ask when expected error cost exceeds ask cost". CORA states the intention, and it prices *interruption* only as a rate, not as a cost, verbatim:

> "route rejected actions to a trainable Diagnostician model, which performs multimodal reasoning over rejected actions to recommend interventions (e.g., confirm, reflect, or abort) to minimize user burden."

https://arxiv.org/abs/2604.09155

CORA's only measured cost of an intervention is latency, verbatim:

> "the latency for that specific step averages 34.13 seconds due to the deeper multimodal reasoning required."

https://arxiv.org/html/2604.09155v1

**Machine escalation is priced** (tokens, latency), in two sources. TypedBench, verbatim:

> "By median billed tokens, a hosted request with one question costs 41% (phishing) and 46% (tickets) of a request with all of them."

https://arxiv.org/html/2610.11392v2

GUI-JEPA, verbatim:

> "At 50% coverage the cascade reaches 69.48 High on average over seeds, 98% of the 7B agent's 70.87, at 0.91× its mean vLLM latency with fp32 GUI-JEPA and 0.76× with bf16"

https://arxiv.org/html/2610.36877v1

SafeGround names the trade-off without pricing it, verbatim:

> "This reflects the inherent trade-off between accuracy and expert invocation cost in uncertainty-aware cascading."

https://arxiv.org/html/2602.02419v2
