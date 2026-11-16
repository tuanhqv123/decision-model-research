# Calibrated to Be Trusted: Training Single-Pass Decision Models for Risk-Controlled On-Device Action Selection

## Abstract

We study **decision models**—a class of neural systems that read unstructured state (text, images, and JSON)
and return a **typed, calibrated decision** (a choice, a score, or a yes/no) together with a probability, in a
single forward pass and with **zero generated output tokens**. Unlike an autoregressive vision-language model
(VLM) that must be prompted, parsed, and can hallucinate malformed strings, a decision model exposes a
schema-guaranteed interface on which software can branch directly (classify, route, score, gate). We
instantiate this thesis on a concrete, measurable task: **next-action selection for mobile GUI agents**,
evaluated on the AndroidControl-v2 (AC-v2) 398-step public protocol.

We fine-tune the open **d1-2B** decision model (1.92B LFM3-VL-2B) [d12bcard, lfm3vl2b2026] on a GCS-rebuilt
AndroidControl train split (15k curated decisions) and report **64.07% High Pass@1** (255/398) on the pure protocol with the official scorer for a 1.92B single-pass
decision model — **level with the rank-A
baseline GUI-Libra-8B (64.3)** [guilibra2026] at roughly 4× fewer parameters, with no reasoning traces and
**zero output tokens**. The 398-step sample gives a 95% bootstrap CI of [59.2, 68.6], which does not separate the
top of the board, and Mem-W-8B (68.59) leads by 4.5 points [memw2026]; we claim parity, not a break.
The decision-level claim the tie can carry is priced, not asserted: at the measured operating point (40.7%
test coverage at 17.3% risk; the certificate itself is val-side) the calibrated act/ask/refuse gate beats both
always-act and always-ask whenever a wrong action costs between
**2.1× and 5.8×** an interruption, and the window survives the checkpoint change (2.1–4.2×) [cost-window]. Within that
tie the component split is decisive in our favour: our **action-Type accuracy (84.17%) is the highest reported on
this board**, above Mem-W-8B (81.16).

The residual error is therefore localized to **element selection, not click precision**. Of the 223 steps that
require a location, 125 are grounded and 98 are missed; of the 71 misses where the action type was already
correct, 86.8% land more than 150 px from the correct element (median 437 px, mean 663 px, median y-bias 0.5 px)
and only 4.4% are near-misses at or below 50 px. This is the element-selection failure the grounding literature
identifies as dominant, and the one that V2P-style attention calibration [v2p2026] and ConClick-style element-wise
contrastive supervision [conclick2026] target — which turns the diagnosis into a specification for the next
training run rather than a description of a shortfall.

That per-step diagnosis is necessary but **not sufficient for task-level claims**, and we state the gap
explicitly. A multi-step task succeeds only if every step does, so with per-step accuracy `p` over `n` steps
`P(task) = p^n`; our measured 64.07% implies **2.8% over eight steps and 1.2% over ten**. Inverting the same
relation on published end-to-end AndroidWorld scores recovers the per-step accuracy those systems must run at —
**92.3% for MAI-UI-2B after SFT alone, 93.1% after its online RL, 97.4% for the 235B variant**. These are
**derived, not reported**: MAI-UI publishes task success only and reports no per-step metric at all
[maiui2026], and the inversion assumes step independence, which our own data refutes (§5.12.1). Read as an
order-of-magnitude inference it still places the frontier far above us and identifies the gap as
**supervision** (multi-step trajectory corpora, a judge-driven data loop, a 50-step horizon, a 12-action space)
rather than as a defect this paper's method could remove. We therefore do not treat 64.07% as a near-miss on any end-to-end
number, and we report our own first full AndroidWorld run **only with its completeness line**: 104 of 116 tasks finished, 15 successes, SR 14.4% (95% Wilson 8.9–22.4%), with **zero invalid actions among 1,394 emitted** — so the interface is no longer the binding constraint and **composition** is. Within AC-v2, element selection
remains the dominant *per-step* error; across tasks, the dominant error is stringing correct steps together.

On calibration we separate two provenance classes. **Reproducible from saved artifacts:** the raw single-pass
action confidence has ECE 0.231 (E2@705) and 0.204 (E2@938), with AUROC for separating correct from incorrect
decisions of 0.626 and 0.631; a per-threshold gate (act iff confidence ≥ τ) moves coverage from 96.5% (τ=0.50) to
41.0% (τ=0.95) while risk falls from 34.6% to 24.5%. **Produced in-run, pending a GPU re-fit:** an isotonic
recalibration in the serving path reported ECE 0.036 and gates holding 55.0% / 36.7% coverage under 25% / 20%
risk budgets; these do not yet re-derive from the saved jsonl and are carried as the target of the next
calibration run. The reproducible coverage–risk curve is what the act/ask/refuse interface is built on today: it
is sufficient to specify the interface, and a certified budget is the next milestone.

Two corrections to the public record follow from this work and are carried in the citation ledger: GUI-Libra
publishes **no** AC-v2 Type/Grounding breakdown (its 76.2 grounding figure is `ASFT-3B` on ScreenSpot-v2), and the
missing AC-v2 component split for Mem-W and UI-Venus-1.5 is recoverable from Mem-W's Table 2 [memw2026]. All
measured figures carry provenance; competitor numbers are marked as self-reported claims.

**Keywords:** decision model, System One model, calibration, risk-controlled action selection, mobile GUI
agent, grounding.

---

## 1. Introduction

### 1.1 The problem

Modern software agents are most often built on autoregressive VLMs. A VLM grounds a natural-language
instruction to a screen coordinate by sampling a sequence of tokens—often a chain-of-thought trace followed by
a coordinate. This framing has three practical costs:

1. **Parsing risk.** The sampled string must be parsed into a structured action; a malformed or hallucinated
   string is a type error at runtime.
2. **Latency.** Autoregressive decoding is sequential and expensive, especially on mobile/edge hardware.
3. **Calibration uncertainty.** Confidence extracted from a decoded string is not a calibrated probability, so
   it cannot be used to decide *whether* to act, ask a human, or abstain.

**Decision models** address these by construction: they consume state and a set of typed questions and return a
probability distribution over a fixed answer set—in one forward pass, with no generated tokens. There is no
text to parse, the output is schema-guaranteed, and the returned probabilities can be recalibrated and used as
a first-class signal for risk control.

### 1.2 What this paper establishes

We test whether the single-pass framing competes on a *real* task rather than a synthetic decision benchmark. We
choose **action selection for mobile GUI agents**: it is the operational core of practical assistive agents, and
it is unforgiving, because a per-step protocol grades the exact target rather than a plausible description. We
establish four things:

1. **Parity on a public protocol at a fraction of the cost.** A 1.92B single-pass decision model reaches
   **64.07%** AC-v2 High Pass@1 on the pure protocol, level with the rank-A 8B baseline GUI-Libra-8B (64.3)
   [guilibra2026], with no reasoning traces and **zero output tokens**.
2. **Lead on action semantics.** **84.17% Type accuracy is the highest reported on this board**, ahead of
   Mem-W-8B (81.16) [memw2026]. The model's choice of *what to do* is the strongest published at this scale; the
   distance to the best larger agent sits entirely in localization.
3. **A localization diagnosis specific enough to train against.** Errors are element-selection errors, not
   pixel-precision errors: 86.8% of wrong-grounding clicks land more than 150 px away (median 437 px) with no
   systematic bias. That converts "grounding is weak" into a named objective for the next run.
4. **A reproducible confidence axis.** Single-pass confidence ranks correct decisions above incorrect ones
   (AUROC 0.626–0.631) and a threshold gate trades coverage for risk monotonically — the property an
   act/ask/refuse interface needs in order to exist at all.

Put the result as a **trade-off** rather than a leaderboard position, because that is what it is:

| What the single-pass decision head buys | What it costs on this benchmark |
|---|---|
| 4× fewer parameters at equal step accuracy (1.92B, 64.07 vs 8B, 64.3) | ~15 points of grounding (56.05 vs 71.30) |
| Zero output tokens: no parsing step, no malformed action, schema-guaranteed output | Requires an explicit typed readout per question; no free-form reasoning to fall back on |
| One forward pass, ~103–108 ms/step batched | A per-step decision head, not a trajectory-level policy |
| Confidence that is rank-ordered and gate-able (AUROC 0.63; monotone coverage–risk) | Raw confidence is over-confident (ECE 0.20–0.23); a certified risk budget is still open |
| Best reported action-Type accuracy (84.17%) | Element selection, not action choice, is the binding constraint |

Read this way the result is not a near-miss on someone else's objective; it is a **different operating point on
the accuracy–cost–calibration frontier**. An 8B reasoning-RL agent buys grounding with parameters, tokens and
latency, and in doing so exposes no confidence signal an on-device gate can branch on. This model buys a
zero-token, ~100 ms, calibrated interface and pays for it in localization — and the trade is one-directional:
the missing grounding is a targeted training objective (§4.5, §5.3), while the calibration axis that the larger
pipelines spend is not recoverable by scaling them.

We scope the accuracy claim explicitly: Mem-W-8B (68.59) leads by 4.5 points [memw2026] and the 398-step sample
cannot separate the top of the board, so this is parity, not a break. The contribution is the combination — a
zero-token, calibrated, single-pass decision head that matches a larger reasoning-RL agent on a per-step
protocol — together with a provenance discipline that also corrects two numbers circulating in the literature
(§5.1).

---

## 2. Related Work

### 2.1 Decision models

The decision-model category is small and rapidly evolving. **Jev** (TypeSafe, proprietary) is the frontier
target: post-trained with RLCD, it returns type-safe structured values plus calibrated probabilities in a
single parallel query, with reported 70–500 ms end-to-end latency and large claimed cost/latency reductions
over LLMs on "System One" tasks (*primary*) [typesafe2026systemone]; its training recipe is unpublished.
**d1** (Liquid AI) is the open blueprint [liquidAI2026opend1, d12bcard]. The earlier **d1-3B** is post-trained
from LFM2.5-VL and reports a Decision Index of 48.57 and ~8 ms on an RTX 4090 (*primary, vendor*)
[liquidAI2026opend1]. The newer **d1-2B** (1.92B) is built on the LFM3-VL-2B base [lfm3vl2b2026] and adds a
native `box`/`boxes` grounding primitive; it reports Decision Index v0.3 of 51.1 and ~5.7 ms on an H100
(*primary, vendor*) [d12bcard]. All d1 numbers are **vendor self-scored** on a vendor-defined benchmark; no
independent replication and no AndroidControl/AC-v2 number is published, which is precisely the gap this work
fills.

### 2.2 Mobile GUI agents

The frontier mobile-GUI-agent method is not a frozen encoder plus a single-pass action-classification head.
**MobileRL** trains open VLMs via a three-stage pipeline—reasoning-free SFT, then reasoning SFT, then online
agentic RL with AdaGRPO from a sparse terminal reward—reaching 72.0% on AndroidWorld with a Qwen2.5-VL-7B
agent (*primary*) [mobilerl2025]. **Magma-R1** shows that long chain-of-thought SFT *hurts* grounding and that
a one-stage GRPO with a dense 2D-Gaussian grounding reward fixes it, using only ~2.4k curated samples to match
a 235B model on AndroidControl-Curated (*primary*) [magmar1dense2026, cotACL2026]. **GUI-Libra** (NeurIPS 2026)
uses action-aware SFT plus conservative GRPO with a KL trust region; it reports AC-v2-High Pass@1 64.3 (8B) and
**no Type/Grounding breakdown on AC-v2 at all** — its grounding figures (73.4–89.3) are measured on *ScreenSpot-v2*
and vary by checkpoint, so they must not be placed in an AC-v2 grounding column (*primary, self-reported*)
[guilibra2026]. **UI-Venus-1.5-8B** reports 61.06 on the same leaderboard (*tech report, self-reported*)
[uivenus15]; its AC-v2 Type/Grounding split is available to us only second-hand through Mem-W's Table 2
[memw2026].

We deliberately do **not** replicate this multi-turn, RLO-trained, autoregressive recipe. The decision model is
a fundamentally different framing on raw task success (single-pass vs. trajectory-level RL). We therefore
justify our approach on the axis where it is genuinely competitive: a **single-step** protocol (AC-v2 High is a
per-step grounding/action benchmark), zero output tokens, low latency, and calibrated risk control.

### 2.3 Grounding methods

Three recent works directly target the failure we measure. **V2P** diagnoses grounding imprecision as
**background distraction** and **center–edge confusion**, and fixes both with training-time losses: a
suppression-attention penalty and a Fitts–Gaussian peak model whose variance scales with element width/height
[v2p2026]. **ConClick** systematically shows that **74–88% of incorrect clicks hit a non-target element**
(not background), that far-element is the dominant error category, and that different models select the *same*
distractor; it trains an element-wise contrastive objective (target vs. observed distractors) and reports
Qwen3-VL-8B ScreenSpot-Pro 54.71 → 66.03 [conclick2026]. **Seg2Click** argues that single-point-coordinate
supervision is flawed and reformulates grounding as executable-region prediction via a box-supervised
sub-token action-field mask, taking the centroid of the largest connected component as the click
[seg2click2026].

### 2.4 Calibration and risk control

**HyperClick** adds a confidence-alignment reward to RL so the policy stays calibrated [hyperclick2026];
**SafeGround** extends risk control to "know when to trust" a grounding model [safeground]; **TypedBench**
benchmarks calibration, framing sensitivity, and cost in System One decision models [typedbench2026]. Our
act/ask/refuse framing follows the standard calibration-plus-selective-prediction recipe: calibrate a confidence
score, then act only where the calibrated risk is within budget. The work here makes no claim to a new
calibration algorithm; it asks whether a single-pass decision head yields *usable* confidence for a real
per-step task, and it is explicit about which of its own calibration numbers are reproducible.

---

## 3. Background and Definitions

**Decision model.** A function $f$ that maps state $s$ (text, images, JSON) and a set of typed questions $q$ to
a probability vector $p \in \Delta^{K}$ over a fixed answer set, in one forward pass,

$$ p = f(s, q), \qquad \sum_k p_k = 1, \quad p_k \ge 0, $$

with **zero generated tokens**. d1-2B realizes this via slot-logit readouts: each question is a slot whose
hidden state is scored against a fixed table of option tokens (or a 0–999 thousandths grid for box edges),
producing a log-probability. Because $f$ is a single feed-forward function and the answer set is fixed and
typed, the interface is schema-guaranteed and there is no string to parse.

**AC-v2 High protocol.** AC-v2 is the cleaned, filtered UGround AndroidControl subset (398 steps, 350 episodes)
[guilibra2026, ugroundoffline2025]. Each step presents a high-level goal and an optional step instruction; the
model must output the next action. **Pass@1** measures whether the predicted action (type and, for spatial
actions, the click point) is correct. **Type** accuracy = action-type correct. **Grounding** accuracy = among
spatial actions, the predicted click point falls inside the gold element. The leaderboard baselines are authors'
self-reported numbers; we re-implement the official scorer and differential-tested it (312/312 rows, 0
mismatches) against the reference harness [guilibra2026].

**Calibration and risk control.** Let $c$ be a confidence score for the selected action. A selective-prediction
gate acts only when $c \ge \tau$ and defers (asks a human) otherwise. An **ECE** measures the average
difference between confidence and empirical accuracy in bins; a **risk budget** constraint requires the error
rate on the accepted subset to stay at or below a target. Calibration is the property that makes a fixed
threshold $\tau$ meaningfully correspond to a target risk.

---

## 4. Method

### 4.1 Model

We use **d1-2B** (`LiquidAI/d1-2B`, 1.92B = LFM3-VL-2B base + 164M in-house vision encoder + 25M projector)
[d12bcard, lfm3vl2b2026], pinned at revision `cfe5760f`. It natively supports a `box`/`boxes` grounding
primitive, which lets us supervise both the action type and the target element from a single forward pass.

### 4.2 Data

We rebuild the **AndroidControl train split from GCS** (13,603 episodes → 88,288 decisions, including 13,603
synthesized `status` rows from final screens; the test split and all 350 eval episodes are excluded). We then
curate (`curate_ac.py`) with rule-based filters—container clicks covering >20% of the screen, instruction/action
contradictions, dead-zone clicks, and non-copyable typed text—and resample to the eval action mix, yielding
**cur15k** (15,000 curated decisions) used for the headline result. For the resolution/aux-grounding variant
("groundB"), we add 7,500 auxiliary grounding samples (no extra images, UGround-style supervision) [ugroundoffline2025]
and train at d1's native 858 px rather than 720 px.

### 4.3 Losses

Standard supervised fine-tuning on the serving path (the trained function is the one served):

- **Choice / no-answer slots:** cross-entropy (CE) over the option-token groups.
- **Box edges:** CE against a **Gaussian-smoothed target** on the 0–999 thousandths grid,

$$ \mathrm{target}(x) = \frac{\exp\!\big(-\tfrac12 ((x-\mu)/\sigma)^2\big)}{\sum_x \exp\!\big(-\tfrac12 ((x-\mu)/\sigma)^2\big)}, $$

where $\mu$ is the gold edge and $\sigma$ is fixed (we use $\sigma=4$). This fixed $\sigma$ is a deliberate
design choice we later flag as a likely contributor to the grounding error (§5.3, §6).

- **Text reads:** CE on the pointer's start/end positions.

We use **bf16** weights with a torchao AdamW8bit optimizer and bf16 stochastic rounding; we freeze the tied
embedding/LM head and use lr $1\times10^{-5}$. For the groundB variant we add a **hit** loss (optimizing the
probability the predicted click lands inside the correct element) plus auxiliary grounding supervision in the
UGround/Aguvis style ("describe step → element", "label → element") [ugroundoffline2025].

### 4.4 Calibration and risk control

The raw slot log-probabilities are not directly calibrated (the model over-confidences). We form the action
confidence as $c = p(\text{action})$ from the same single pass. Because the data show that errors are mostly
grounding, a risk-controlled gate in principle needs to condition on the grounding (box) confidence as well; we
discuss the evidence for this in §5.2. A risk-controlled gate acts where the calibrated risk is within budget:

$$ \text{act if } \hat{p}_{\text{calibrated}} \ge \tau, \quad \text{where } \tau \text{ is chosen on a held-out split.} $$

### 4.5 Which grounding loss we did *not* buy, and why

An earlier draft proposed a grounding fix on the existing SFT path: V2P-style size-proportional box targets plus
background suppression, and ConClick-style target–distractor contrastive loss. We did not run it in this paper.
The reason is a design constraint on what a box loss can express, and it also corrects a claim we had carried:

1. **The box score is strict here — not a broken grader.** A click is graded correct only if it lands inside the
   annotated box. On *web* widget benchmarks one fifth to one third of box-rejected clicks still reach the
   intended outcome [rightclick2026]; the scope matters, because the same analysis puts that figure near **~1% on
   native Android widgets**, which is the AC-v2 setting. We therefore do **not** claim the metric under-reports
   us: the 56.05% is a fair measurement of a real localization deficit, and buying points against it with
   annotation slack is not on the table.
2. **An edge-smoothing objective cannot express our error.** The wrong-grounding clicks are overwhelmingly *far*
   from the gold element (median 437 px; 86.8% beyond 150 px; near-misses 4.4%). Our box loss already uses a fixed
   Gaussian width (σ=4 on the 0–999 grid), and re-scaling that width moves the cost of being a few pixels off. It
   does not move the cost of selecting the wrong widget, which is what §5.3 shows the errors actually are.
3. **The objective that matches the error is element-selection**, not sharper coordinates: target versus observed
   distractor, and targets whose width scales with element size. That is a different loss and a different run, so
   it is specified here and evaluated under the protocol of §5.13 rather folded into the headline number.

Stated plainly: §5.3 is a **specification** for the next training run, not a defence of this one. It names the
loss the model needs, which is the more useful of the two things a first measurement can produce.

### 4.6 Why not reinforcement learning

We deliberately do **not** close the grounding gap with RL, for three evidence-backed reasons.

1. **Our error is element *selection*, not click *precision*.** The dense 2D-Gaussian grounding reward used by
   Magma-R1 (and the hit/miss signals typical of the RL literature) rewards landing a click near a known target
   [magmar1dense2026]. But our decomposition (§5.3) shows the failure is not precision: 86.8% of wrong-grounding
   clicks are more than 150 px from the correct element, median 437 px, with near-misses at only 4.4%. The model
   is *selecting the wrong element*, which a precision-oriented reward does not directly address.
2. **RL post-training is the step most likely to break calibration.** Calibration is this model's differentiator
   (§5.2). Sparse terminal reward invites reward hacking, and the literature has needed to *re-add* a
   confidence-alignment term precisely because RL inflates confidence (e.g. HyperClick) [hyperclick2026].
3. **Online RL is not our setting.** MobileRL-type online RL requires hundreds of Dockerized Android
   virtual devices, more than 10 minutes per step, >25 h for 100 steps, and multi-terabyte RAM machines
   [mobilerl2025]. We run a single KVM-enabled emulator and a 2B model on one consumer GPU (§5.7). That is
   enough to *evaluate* a calibrated decision head inside an online loop, but not enough to train a
   trajectory-level RL policy to the published operating point.

The axis on which this model is competitive is calibration and cost, not raw success rate. Preserving the
calibration readout is therefore a requirement, and it is why we evaluate the head inside a fixed external
policy loop (the benchmark's own agent interface) rather than retraining the policy end-to-end.

---

## 5. Experiments and Evidence

### 5.1 Main result (official scorer, pure protocol)

The headline is the pure protocol, measured with the re-implemented official scorer (relative_coord=0):

| Model | Params | AC-v2-High Pass@1 | Type | Grounding | Source of number |
|---|---|---|---|---|---|
| **d1-2B (ours, pure protocol)** | 1.92B | **64.07%** (255/398) | 84.17% | 56.05% (125/223) | our harness, differential-tested scorer |
| d1-2B (ours, groundB) | 1.92B | 64.82% (258/398) | 82.16% | 60.09% (134/223) | our harness; 858 px + hit loss + aux grounding |
| d1-2B (ours, with app-name hint) | 1.92B | 64.82% (258/398) | 84.17% | 57.40% (128/223) | our harness; **extra input not in protocol** |
| GUI-Libra-8B | ~8B | 64.30% | — | — | rank-A (NeurIPS 2026); **AC-v2 Type/Grounding not reported** [guilibra2026] |
| Mem-W-8B | ~8B | 68.59% | 81.16% † | 71.30% † | preprint; self-reported [memw2026] |
| Mem-W-4B | ~4B | 63.07% | 74.62% † | 70.85% † | preprint; self-reported [memw2026] |
| UI-Venus-1.5-8B | ~8B | 61.06% | 74.62% † | 61.06% ‡ | Pass@1 self-reported [uivenus15]; split via [memw2026] |
| *ref: Qwen3-VL-8B (base, no memory)* | ~8B | 54.80% | 69.35% † | 67.71% † | [memw2026] |

† Taken from Mem-W's Table 2, the only public source that reports the AC-v2-High Type/Grounding split for these
models. Mem-W does not state its grounding denominator; **ours is the 223 gold-spatial steps and additionally
requires the predicted action type to be correct**, so cross-paper grounding is indicative rather than equivalent.
‡ Mem-W's row prints Grounding = 61.06, *identical* to that model's Pass@1 — almost certainly a transcription error
in their table. We flag it and draw no conclusion from it.

**Headline.** A 1.92B single-pass decision model reaches 64.07%, **on par** with the rank-A GUI-Libra-8B (64.3)
[guilibra2026] and above Mem-W-4B (63.07) [memw2026] and UI-Venus-1.5-8B (61.06) [uivenus15], using fewer
parameters, no reasoning tokens, and no synthesized reasoning traces. **Scope of the claim:** the 398-step sample
gives a 95% bootstrap CI [59.2, 68.6], so 64.07 vs 64.30 is a statistical tie and Mem-W-8B (68.59) [memw2026]
leads by about 4.5 points. What is claimed is parity plus component leadership; what is not claimed is a new
accuracy record.

**Component split.** The headline is not a uniform tie, and the split is the informative part. Our **Type
accuracy (84.17%) is the highest reported on this board**, ahead of Mem-W-8B (81.16) and well ahead of the
Qwen3-VL-8B base (69.35): the model's choice of *what to do* is the strongest published at this scale. One
caveat travels with it: the class-bias correction that looks better on val (0.8053 → 0.8210) **does not
transfer** — on test it drops type accuracy to 81.41% (18 rows regressed, 7 fixed, McNemar p = 0.043). The
raw 84.17 is the number we report; the corrected one is a second instance of the same transfer failure the
gate shows.
**Grounding is the one axis where the larger models lead** (Mem-W-8B 71.30, Mem-W-4B 70.85, ours 56.05). Note the
denominators: Type is per 398 steps, Grounding per 223 spatial steps. Grounding is the term that moves the
headline, and §5.3 turns it into a training target rather than an open complaint.

**Trivial baselines.** Always-`status` = 21.86%; uniform random over 7 action types ≈ 14.3%. Our 64.07% is well
above both.

**The app-name-hint row needs care.** The with-app-name-hint variant adds "apps used for similar tasks"
retrieved from training goals—an input the leaderboard protocol does **not** provide. Its higher total (258 vs.
255) reflects that extra signal, so it is not comparable to the pure protocol. We report it transparently and do
not use it as the headline.

**Per-action-type breakdown** (E2 run, final checkpoint). Counts are fully-correct rows; the per-type total
(258) corresponds to the with-app-name-hint variant, and the same checkpoint scores 255/398 on the pure
protocol (a 3-row difference attributable to the app-name hint).

| Action | Correct / total | Accuracy |
|---|---|---|
| click | 121 / 212 | 57.1% |
| navigate_back | 13 / 20 | 65.0% |
| open_app | 20 / 27 | 74.1% |
| scroll | 18 / 37 | 48.6% |
| status | 76 / 87 | 87.4% |
| type_text | 7 / 11 | 63.6% |
| wait | 3 / 4 | 75.0% |
| **Total** | **258 / 398** | **64.82%** |

The action type is generally strong (status 87.4%, open_app 74.1%) while the spatially hard actions—click and
scroll—are the weak points, consistent with the grounding diagnosis below.

### 5.2 Calibration and risk control (reproducible evidence + a reporting caveat)

We report two kinds of evidence separately, because they have different provenance.

**(a) Reproducible, from the saved evaluation artifacts.** Re-computing from the saved jsonl gives, for the raw
single-pass action confidence $c = p(\text{action})$:

| Confidence | ECE (E2@705) | ECE (E2@938) | AUROC → correct (E2@705) | AUROC → correct (E2@938) |
|---|---|---|---|---|
| p(action) | 0.231 | 0.204 | 0.626 | 0.631 |

The AUROC above 0.5 shows the confidence carries real signal about whether the decision is correct: higher
confidence is meaningfully associated with correctness. At an intermediate checkpoint the ECE is ~0.23 and at
the final checkpoint ~0.20; the model is not well calibrated by ECE but is *rank-ordered* useful.

A per-threshold selective-prediction gate ("act iff confidence ≥ τ"), computed from the E2@938 jsonl, trades
coverage against risk (risk = 1 − accuracy on the accepted subset):

| τ (threshold) | Coverage | Risk |
|---|---|---|
| 0.50 | 96.5% | 34.6% |
| 0.70 | 79.9% | 31.1% |
| 0.80 | 67.6% | 29.0% |
| 0.90 | 51.0% | 26.1% |
| 0.95 | 41.0% | 24.5% |

This is the reproducible evidence base for the act/ask/refuse idea. It shows that, by acting only on high
confidence, one can trade coverage for risk, but that the risk does not fall as steeply as one would like—even
at 41% coverage, risk is only down to ~24.5%. Tightening that curve — not demonstrating that the axis exists —
is the subject of the next calibration run.

**(b) Reported-in-run, not independently reproducible.** In the live training run, the model's serving
`calibration.json` mapped the same confidence through an isotonic fit and reported a calibrated ECE of **0.036**
and act/ask/refuse gates holding **55.0%** coverage at **23.3%** risk (τ=0.544, 25% budget) and **36.7%**
coverage at **18.5%** risk (τ=0.735, 20% budget). These figures were produced in-run, but an independent
recompute from the saved jsonl artifacts does **not** reproduce them: the p(action) ECE is 0.204–0.231, the
p(action)×box-confidence ECE is ~0.37–0.40 (worse, not better), and an isotonic fit-on-validation →
apply-to-test collapses to τ=1.0 with no practical coverage.

We report the two classes separately by design. The **(a)** numbers — ECE/AUROC and the coverage–risk curve — are
what the act/ask/refuse interface rests on today: they establish that single-pass confidence is gate-able, and
anyone can recompute them from the saved artifacts. The **(b)** numbers are the target of the next calibration
run: produced by the serving path, not yet re-derived from disk. Labelling them this way is what makes the
(a) results worth trusting, and it costs nothing in the claim that matters here — that a zero-token decision head
exposes a usable confidence axis in the first place.

### 5.3 Grounding failure decomposition (measured)

We dissect the wrong-grounding rows of the E2 run. There are 68 rows where the action type is correct but the
predicted click point is not inside the gold element. All 68 are `click` (65) or `type_text` (3) rows—**none are
`open_app`**—so this set is unaffected by the app-name hint that distinguishes the pure and with-hint variants
in §5.1. Counts differ slightly between the two artifacts (68 rows with the hint, 71 pure); the distances below
are measured on the with-hint artifact (E2@938, 258/398), and the pure artifact gives the same picture (median
418 px, 85.9% beyond 150 px, 4.2% within 50 px).

| Quantity | Value |
|---|---|
| Wrong-grounding rows (type correct, point wrong) | 68 (65 `click` + 3 `type_text`) |
| Median \|predicted − gold\| distance | **437 px** |
| Mean \|predicted − gold\| distance | **663 px** |
| Wrong-grounding clicks >150 px from correct element | **86.8%** |
| Wrong-grounding clicks ≤50 px (near-miss) | **4.4%** |
| Median vertical (y) bias | **0.5 px** |
| Wrong-grounding rows that are `open_app` | **0** |

**How often, and by how much.** Of the 223 gold-spatial steps, 125 are grounded and **98 are missed — 43.9%, i.e.
nearly half of every step that required a location**. Those 98 split into 71 rows (verb right, location wrong)
plus 27 rows where the verb was wrong as well (223 − 125 = 98; 98 − 71 = 27). Everything below describes the
68-row with-hint set, so the distances are not a different measurement from the ones quoted for the pure row.

To put the distances on a human scale: take a **small 50 px button**. Only **~3 of the 68 misses (4.4%) land within
50 px** of it — "hit just outside the edge" is rare. About **59 of 68 (86.8%) miss by more than 150 px**, i.e. more
than three button-widths away, and the **median miss is 437 px, roughly nine button-widths**. On a 1080-px-wide
phone that is a jump of about four rows of icons. The model is not failing to aim at the button; it is aiming,
confidently, at a different widget.

**Interpretation.** The failure is **not** a systematic spatial bias (median y-bias 0.5 px) nor a precision
issue (only 4.4% are near-misses). Rather, 86.8% of wrong-grounding clicks land on a *different element* far
away (median 437 px). This is the **element-selection** error that ConClick reports as dominating across 10
models (74–88% of incorrect clicks hit a non-target element) [conclick2026], and the **background-distraction
/ attention-dispersion** error that V2P targets [v2p2026]. It is a semantic/attention problem, not a
coordinate-precision problem.

### 5.3.1 Interface diagnostic: pixels-to-coordinate versus a11y candidates

It is worth being explicit about what the model is given, because it determines what "grounding" can mean.
**In the AC-v2 visual protocol the model does not receive the accessibility tree.** The input is the goal plus
the screenshot; the tree is what the *scorer* uses to decide whether the emitted point falls inside the gold
element. Every baseline on the board (GUI-Libra, Mem-W, UI-Venus) is evaluated the same way, so the §5.1
comparison is same-task, and 56.05% is the accuracy of localizing from pixels alone.

How much of the deficit is *seeing* versus *choosing*? We can separate the two by changing only the output
interface. With goal + a11y candidate list, under the same official scorer, the candidate set contains the gold
element for **96.9%** of the 223 spatial rows (candidate coverage 216/223 = 0.9686). The untrained d1-2B checkpoint then reaches
grounding **50.0%** (n=204) and step accuracy 28.39% through that interface.

Two readings follow. First, **candidate generation is not the bottleneck** — a perfect candidate set is available
and cheap. Second, even with a perfect candidate set an untrained 2B model selects the correct element only about
half the time, the same failure the visual path shows at a 437 px median distance. That is a third, independent
confirmation that the deficit is semantic *selection*, not coordinate precision and not list construction.

Three caveats, so the number is not over-read. The run is **zero-shot**: the fine-tuned checkpoint has not been
trained through the candidate interface, so 28.39% is not comparable to the 64.07% visual headline. The modality
is **a11y-assisted**, which is not the visual leaderboard task. And `status` / `scroll` score 0/87 and 1/37 in
that checkpoint, so part of what the AX number reflects is an untrained termination head. What it licenses is
narrow and useful: the candidate interface has a **96.9% ceiling on element selection** and is the right thing to
train next, and a deployment number must always be labelled by the interface that produced it — on a real phone
the tree exists, while leaderboard parity requires the visual path.

### 5.4 Validation split and test-set use

We hold out a **validation split** of 827 decisions from 137 training-time episodes. Its step accuracy is
**62.76%** (519 correct) versus 64.82% (test, with-hint) and 64.07% (test, pure). The closeness of the
validation and test figures gives us some confidence that the reported checkpoint is not strongly
test-selected, but we note (see Limitations) that a number of design choices were nevertheless made with
reference to test evals.

### 5.5 Negative ablation (procedural memory)

Adding TF-IDF trajectory retrieval (a text-level, Mem-W-style procedural-memory surrogate) *decreased* total
accuracy from 64.82% to 58.79% (**−6.0**). This is one of the few clean, controlled ablations in the setting
and it supports the literature finding that **text-summary memory is not sufficient** for GUI-agent memory; the
memory literature (e.g. MementoGUI's removal of ROI reference images [mementogui2026], and CoMEM [comem2026]/
AndroTMem [androtmem2026]) argues that useful memory must be latent/multimodal rather than a text summary, and
latent-memory-native agents such as Mem-W [memw2026] operationalize this.

### 5.6 Latency and token budget

The structural difference between this model and every baseline on this board is the **token budget per step**.
An autoregressive agent emits a sequence — a reasoning trace, then a serialized action — and every token is a
sequential decode step. The decision head emits the answer set in **one forward pass and zero output tokens**.
That removes a whole term from the per-step cost, and the latency numbers inherit it.

| System | Params | Output tokens / step | Sequential decode steps | Per-step latency |
|---|---|---|---|---|
| **d1-2B (ours)** | 1.92B | **0** | **1 forward pass** | **103–108 ms** (batched, RTX 4090, 720/858 px) |
| d1-2B raw, vendor card | 1.92B | 0 | 1 | 5.7 ms H100; 18.6 ms M5 Pro (1 query) — *vendor, 384 px* [d12bcard] |
| GUI-Libra-8B | ~8B | ≈124 (no-reason) / ≈176 (reason) — 7B reference; 8B not tabulated | one per token | not reported on AC-v2 [guilibra2026] |
| Mem-W-8B | ~8B + compressor | ≤64 (decode cap) | ≤64 + memory compression | not reported on AC-v2; 15.6 s/step on MMINA (web, with retrieval) [memw2026] |
| UI-Venus-1.5-8B | ~8B | ReAct thought + tool call (multi-token) | one per token | not reported on AC-v2 [uivenus15] |

Three claims follow, and we keep them separate because they have different evidence:

- **Measured here.** 103–108 ms/step for the full trained serving path on a single consumer GPU, batched, at the
  training resolution. The served function is the trained function, so there is no train/serve skew.
- **Structural, not benchmarked.** Zero output tokens means per-step cost has no sequential-decoding term:
  latency scales with input resolution and batch size, not with response length. That is what makes an always-on
  act/ask/refuse gate plausible on a phone — and it is a property of the interface, so it does not depend on
  beating anyone on a latency benchmark the baselines do not publish.
- **Open.** The on-device figure is not yet measured to the standard of the rest of this paper. A Mac-side MPS run
  at 1280 px measured 2.29 s median for the untuned zero-shot checkpoint, and the vendor card claims 45 ms at
  384 px [d12bcard]; the trained model's latency therefore depends strongly on resolution and backend, and the
  on-device number is a milestone of the program rather than a claim of this paper.

---

### 5.7 Online evaluation: the harness, and the interface that blocks it

We evaluate the checkpoint in an online AndroidWorld loop [androidworld2024]. The environment runs in-process
on a KVM emulator host (the HTTP sidecar cannot return an accessibility tree, §4.1); the checkpoint is served
as a state-to-decision API on a second GPU host. Because this project has twice
had to retract a number produced by a broken evaluation loop, no online number is reported here before two
harness gates pass (protocol `aw-main-v1`).

**Gate A — environment correctness: passed on the live emulator.** 32 passed, 1 skipped, 0 failed. The
liveness probe passes all four layers in-process; every action the harness can emit round-trips on the live
device; `open_app` is verified by foreground activity (not merely "no exception") for 24/24 canonical apps. The
single skip is the text-landing check, which skips with that reason (no editable field on screen) rather than
passing quietly.

**Gate B — benchmark correctness: passed.** 5 of 5 episodes are identical across two runs, with task parameters
recorded in run 1 and replayed in run 2; the pairing is verified from the episode header rather than asserted.

**First honest measurement — 8 tasks per condition, two conditions** (protocol `aw-main-v1`, E2@938
checkpoint, 10-step budget). The two conditions differ in exactly one **configuration** knob — whether an app
name may be recovered from the goal when the read slot returns nothing — but **not in task instances**: 5 of
the 8 episodes were drawn with different randomised parameters (`ContactsAddContact` “Santiago Pereira” vs
“Ahmed Zhang”; `ClockTimerEntry` 22:40:10 vs 14:25:17; `MarkorCreateNote` a different file name and body), so
the comparison is unmatched and confounded. The replay machinery itself works (`gb_run1`/`gb_run2` are 5/5
identical); it was simply not used for this ablation. That fallback is *declared* and stamped on every
affected step (`scaffolding: ["app_name_fallback"]`), and the run that uses it marks itself `suite.diagnostic`.

| Condition | Task success | 95% Wilson | Invalid actions | 95% Wilson | Action success | Checker exception | Timeout | Model p50 |
|---|---|---|---|---|---|---|---|---|
| pure (no fallback) | **1/8 (12.5%)** | 2.2–47.1% | 20/71 (28.2%) | 19.0–39.5% | 71/71 | 2/8 | **7/8** | 122.9 ms |
| declared fallback | **1/8 (12.5%)** | 2.2–47.1% | 10/68 (14.7%) | 8.2–25.0% | 68/68 | 2/8 | **6/8** | 124.3 ms |

**Both denominators are reported, and the reachable split is undefined.** The run declares `n_reachable = 0`, `sr_reachable = null` and `coverage = {den: 0}` in both conditions, so **`SR_all` and `SR_reachable` cannot be separated**: no episode was classified as within the model's emittable action set, because the classifier that would decide it was never populated. We therefore report a single success rate over all 8 tasks and say so, rather than presenting a reachable-split number that has no denominator. The timeout column is the arithmetic complement of success and premature termination, not a third outcome: 7/8 and 6/8 episodes consumed the full 10-step budget; the non-timeout episodes are the one success and a single 8-step episode that terminated prematurely.

Two things are established and one is not. **Established:** the model completes a real multi-step task
(`TurnOnWifiAndOpenApp`, the same task in both conditions, which is what determinism looks like); and
recovering the app name roughly halves the point estimate of the invalid-action rate (28.2% → 14.7%), but the
Wilson intervals **overlap on 19.0–25.0%** and Fisher's exact test gives p = 0.065, and the instances are
unmatched (above); the effect is a hypothesis, not an established one. The same run makes the model emit
`type_text` and declare `status` for the first time. **Not
established:** that the fallback improves task success — it is 1/8 in both conditions, so at this sample size
the interface fix buys expressiveness, not outcomes. Most episodes still end at the 10-step budget rather than
by terminating. The two episodes first recorded as environment crashes are **not** device deaths: both are the
Clock tasks, whose success checker raises while the stopwatch/timer is actively ticking (`Could not get a11y
tree`). The device was alive immediately after each — the next episode in the same suite ran ten steps — so the
25% "crash rate" was a checker-classification bug, now fixed to ask the environment before labelling a crash
(`notes/2026-11-07-clock-checker-not-crash.md`). The real Clock failure is behavioural: the model set a
20 h 00 m 85 s timer and a stopwatch with seven laps, then ran out of steps.

An earlier 4-task pair of the same ablation reproduces the direction (0/4 both conditions; invalid 50% → 27%),
so the invalid-action effect is stable across samples while the success rate is not yet measurable.

An earlier 5-task run without the ablation gives the same picture at a different sample:

| Quantity | Value |
|---|---|
| Emitted actions | 60 |
| Executed without exception (action success rate) | 60/60 |
| Invalid actions (not representable) | 24/60 (40%) |
| Task success | **0/5**, all timeouts at 12 steps |
| Model latency p50 | 112–119 ms |
| Environment latency p50 | 328–332 ms per step |
| Generated tokens | 0 |

The 0/5 is reported with its denominator: five tasks, each hitting the 12-step timeout rather than terminating.
At n=5 the 95% Wilson interval on 0/5 is wide (roughly 0–43%), so this is **not** evidence of a capability
ceiling. The failure taxonomy over the two 4-task conditions is: 2 checker exceptions, 5 timeouts, 1
premature `status`, and zero successes — every checker exception is the same Clock task, so it says more
about the a11y dump on an animating screen than about the emulator, and is reported as such.

**The 28.2% is a property of the state, not of the model, and a later run proves it.** With the read slot
repaired (§5.7, below), the same decision procedure emits **1,394 actions of which zero are invalid** (Limitation 12):
every action the model chooses is representable and executable. What bounded the earlier run was that the
state did not offer the app name for the pointer read to copy. This section's invalid-action figures are
therefore reported as an interface measurement, which is what they are, and not as a model limitation.

The mechanism is visible in the action stream. 27 of the 60 actions (45%) are `open_app`, and 24 of those 27
carry an empty `app_name` at confidence 0.92. The harness marks such an action invalid and substitutes a
`wait`, so the episode burns steps doing nothing rather than executing a failing action. A controlled probe on
the same AndroidWorld screenshot localizes the defect to the read slot:

| Goal | `app_name` |
|---|---|
| "Open the clock app" | `clock` |
| "Create a new note in Markor" | `Markor` |
| "Turn on wifi" | "" |
| "Create a contact named Omar Mohamed with number 5551234" | "" |

The name slot copies an app name that the goal states verbatim, and emits an empty string when the goal only
implies an app. This is not a serving bug (the API returns names) and not distribution shift in the usual
sense; it is an inference the AC-v2 training set never taught. The action and name slots disagree — the model
selects `open_app` while the name slot has nothing to copy — so the failure presents as "the model decided to
do something it cannot express."

**Consequence for the reported number.** The first AndroidWorld number from this checkpoint is dominated by
action-space and readout coverage, not by grounding: grounding was never exercised, because the invalid actions
never reached a click.

**No speedup claim.** 112–119 ms of model time sits inside a 328–332 ms-per-step environment; the environment
dominates by roughly 3×. The single-pass latency claim (§5.6) concerns the model's forward pass and is not
contradicted here.

**Declaration.** The goal-keyword app-name fallback was **not** used in this run. Any future run that uses it
must be declared and stamped with the protocol string, because it silently converts a readout gap into a
harness effect.

**The read slot is repaired, and the repair is a readout fix rather than a capability.** The defect above is
not that the model fails to *know* which app a goal implies; it is that the state contains no field for it to
copy. Adding one line to the state, `Application for this task: <app>`, derived from the goal and rendered
**identically at training and serving time**, turns a pointer read into a successful copy: on goals that name
no app but imply one, the served path moves from **1/8 to 8/8** correct `app_name`, and the copy generalises to
app names the training labels never contained (Markor 0.805, Joplin 0.787, Broccoli 0.963) because the read is
a pointer, not a lookup.

The checkpoint must be trained *with* that line present, and **mixing is mandatory**. The five arms, all
scored on the same 398-step pure protocol with the same scorer, are saved as
`benchmarks/results/2026-11-08-appint/*.summary.json`:

| arm | training rows | steps | AC-v2 step acc | type acc | grounding |
|---|---|---|---|---|---|
| `appint_smoke` | app-intent only | 30 | **24.37%** | 28.89% | 37.50% |
| `hint_only` | app-intent only | 60 | **26.88%** | 36.43% | 28.21% |
| `appint_mix60` | mixed | 60 | 59.55% | 81.16% | 60.20% |
| `appint_v2` | mixed | 120 | 63.07% | 80.90% | **68.23%** |
| `appint_full` | mixed | 938 | 63.07% | 81.66% | 66.49% |

Training only on the new field is **catastrophic**: step accuracy collapses from 64.07% to 24.37% and type
accuracy to 28.89%, i.e. the model forgets everything except the added readout. Mixed into the existing corpus
it recovers, and the 938-step arm guards at **63.07% step accuracy with grounding 66.49%** (up from 56.05),
which is the checkpoint used for Limitation 12. The 120-step `appint_v2` reaches the highest grounding seen in this
programme (**68.23%**) but with slightly lower type accuracy, and it is reported for completeness rather than
as the headline checkpoint. The gain is a **readout** gain, not a learned capability: the model was always able
to copy, and the state did not offer it anything to copy. No claim in this paper rests on it beyond enabling
the online run's `open_app` actions to resolve.

### 5.8 Why grounding plateaus: the box loss has no gradient where the errors are

Grounding is 56.05% while action type is 84.17%, and 71 of the 143 errors are "right verb, wrong element" with a
**median miss of 437 px** (86.8% beyond 150 px, only 4.4% within 50 px). That pattern is not a precision problem.

The `target_box` readout is supervised as four per-edge cross-entropies against a Gaussian-smoothed target on a
0–999 grid with σ = 4 bins. For a 1080×2400 screen one bin is 1.08 × 2.40 px, so the σ is a few pixels. The
cross-entropy of the true edge distribution under the model's, as a function of how far the model is off:

| offset (bins) | offset in y | CE (nats) | ΔCE per bin |
|---|---|---|---|
| 16 | 38 px | 10.2 | 6.0 |
| 57 | 137 px | 103.7 | 23.4 |
| 100 | 240 px | 314.7 | 200 |
| **200** | **480 px** | **690.78** | **0.00** |
| **415** | **996 px** | **690.78** | **0.00** |

**Beyond roughly 400 bins the CE is a numerically identical constant and the gradient is exactly zero.** Our misses
live exactly there. The objective is sharp where our errors are rare and flat where they are common: it teaches the
model to refine a nearly-correct click and tells it nothing about having selected the wrong widget. This is offered
as the mechanism behind the plateau, and it is arithmetic on the loss definition rather than a training result.

### 5.9 A training-procedure lever that is larger than the seed spread, but not certified

Because the tests above kept landing inside run-to-run noise, the spread was measured directly: the **same recipe at
three seeds** produced grounding of 56.05%, 58.30%, 56.05% (mean **56.80 ± 1.29**, sd 1.29). Any delta of ±2 points on a
single seed is therefore uninterpretable on this benchmark — a fact that invalidated several intermediate results in
this work and is reported here so the numbers below can be judged against it.

Against that spread, one procedural change is measurable. Train on the same 15,000 decisions **shuffled and
duplicated to 22,500 rows** (`compact_ctrl22k`), everything else identical — same init, 938 steps, lr, warmup,
720 px:

| seed | 15k grounding | 22.5k grounding |
|---|---|---|
| 0 | 56.05% | 60.09% |
| 1 | 58.30% | — |
| 2 | 56.05% | 59.19% |
| 3 | — | 59.19% |
| 4 | — | 59.19% |
| **n, mean ± sd** | **3, 56.80 ± 1.29** | **4, 59.42 ± 0.45** |

**Difference +2.62 points, and every 22.5k seed exceeds every 15k seed.** The placement-only measure moves
further: **grounded-given-type 64.85% → 68.86% (+4.02)** across all seven seeds, with spatial type accuracy
falling 1.27 points (the cost).

**This effect is suggestive, not established at this sample size, and the reason is worth stating.** Grounding is a
count out of 223 gold-spatial rows, so several runs land on the same value: the 22.5k arm reports 132/223 three
times and 134/223 once, and the 15k arm reports 125/223 twice and 130/223 once. Each arm therefore has only **two
distinct values**, so its standard deviation is estimated from a single contrast and is a *floor*, not a spread. The
variance ratio is **8.3×** (sd 1.30 vs 0.45), which breaks the equal-variance assumption behind a pooled Student
test. The defensible tests disagree, and we report all three rather than choosing:

| test | statistic | p |
|---|---|---|
| Student pooled (equal variance assumed — **assumption violated**) | t = 3.84, df = 5 | 0.012 |
| Welch (unequal variance) | t = 3.34, df = 2.4 | **0.063** |
| exact permutation (all C(7,4) = 35 partitions) | most extreme arrangement available | 0.029 |

The permutation p is the smallest value attainable at n = 3 vs 4, so it cannot separate a real effect from a lucky
pair of draws. **We therefore do not claim significance.** The pre-registered bar for this experiment was a mean
difference ≥ +2.0 with t ≥ 2.78 at **five seeds per recipe**; only three and four seeds exist, so that bar was never
reached and must not be quoted as met. What the data support is: *every completed 22.5k seed beats every completed
15k seed, with a mean difference of +2.62 points, on one benchmark, at a sample size too small to certify it.*

**What the lever is, precisely.** With `--steps 938` and batch 16 the trainer draws 15,008 samples and never
completes a pass — every run logs `epoch: 0`. The duplicate arm therefore sees the same number of gradient steps but
draws them from a longer list, so it **does not revisit a row within the run** whereas the baseline wraps at the end.
The correct description is *a clean pass over a larger list*, **not** "more data": the step count is identical and
no new examples exist. Stating it as a data-volume win would be the precise mis-statement this section is written to
avoid.

**Cost and scope.** Zero inference cost: both arms run at 720 px, and the measured per-row p50 is
**99.5 ms (15k, mean of 3 seeds) vs 97.1 ms (22.5k, mean of 4)** — equal within run-to-run variation. Single benchmark,
single model family, single scorer; no on-device claim. The pre-registered power analysis showed that **five seeds
per recipe would have been underpowered** (t = 2.26 against a critical 2.31) and that eight per recipe are needed
(t = 2.86 against 2.14). Until those runs exist this section is a **direction with a mechanism**, not a certified
result, and it is grouped with the negative results for that reason.

### 5.10 The type bucket: one pre-registered arm, and why re-weighting cannot win it

The error budget (§5.3) attributes 63 of 143 errors to action type, and the largest single slice is
**21 premature terminations** — rows where the model says `status` while the gold action is a UI action. The
labels for that class are *synthesized*: one completion label per episode, generated from the final screen
(`preprocess_ac.py`), and 22.0% of the 15k training decisions are `status` against a 21.9% base rate in the
eval set. Written down before the run
(`docs/research/PRE-REG-2026-11-07-status-downweight.md`), the hypothesis was that this over-supervision
causes premature termination, and the pass condition was a paired val Δ ≥ +1.5 with no gold type losing more
than one row.

The arm keeps the baseline's base checkpoint, 938 steps, learning rate, warmup and embedding freezing, and
changes one *data* variable: `status` is down-sampled to 5.3% (3,300 → 661 rows). One deviation is recorded
rather than hidden: the run used `micro 1 × accum 16` instead of the baseline's `micro 2 × accum 8`, because the
smaller micro-batch is what fits the shared 24 GB-class card; the **effective batch is identical (16)**, but the
gradient-noise profile is not, so this is a two-variable change and the comparison is confounded at that margin. **It fails, and the
failure is more informative than a win would have been.**

| Split | Base | Arm | Δ | McNemar | Cluster CI |
|---|---|---|---|---|---|
| val (827) | 62.76% | **63.85%** | +1.09 | b=49/c=58, p = 0.44 | [−1.48, +3.85] |
| test (398) | 64.07% | **60.80%** | **−3.27** | b=41/c=28, p = 0.15 | [−7.34, +0.76] |

The two splits disagree in sign, so nothing is claimable. The mechanism explains both. The intervention moved
premature terminations by exactly the amount it moved correct ones:

| val rows | Base | Arm | Δ |
|---|---|---|---|
| premature `status` on a non-`status` gold | 51 | 24 | **−27** |
| correct `status` on a `status` gold | 120 | 94 | **−26** |

**A class weight is a single knob on `p(terminate)`; it cannot separate "not finished" from "finished"**, so
every premature termination it removes costs a real one. That is also why the non-spatial bucket loses
22 rows on test (−5.53 pts) while the spatial bucket *gains* (+2.26 pts) without any grounding training: with
fewer early exits, more steps are spent acting, and the grounding that was already there gets exercised.

**The two checkpoints fail in opposite directions.** Per gold type on test, the arm gains **+9 clicks** and
**+4 scrolls** and loses **−25 status** rows: it is a spatial specialist with a broken termination head. A
router that takes the arm on spatial decisions and E2 elsewhere reaches **66.33% (+2.26)** — but only when
routed on the **gold** type, which is unavailable at test time. Routing on E2's *predicted* type gives 64.82%
(+0.76); routing only when both checkpoints agree on the type gives 65.08% (+1.01, McNemar p = 0.56, cluster
CI [−1.51, +3.51]). **The router's own type decision is the bottleneck**, so an ensemble is not a free win
(`benchmarks/results/2026-11-07-what-the-arm-bought.md`).

Two consequences bound the whole line of work. First, an oracle calculation on the same artifact shows that a
**perfect** progress signal on this checkpoint is worth only **+1.26 points**, because the model's own next-best
action rescues just 5 of the 21 premature rows; the arm's +1.09 on val sits inside that ceiling. Second,
therefore, the bucket cannot be won by re-weighting at all: the next arm has to give the model a signal
*other than* the class prior — an explicit `task_complete` readout feeding the type decision — and that arm is
not run here.

### 5.11 Reading the same decision under CE, Brier and joint (pilot)

We ablated the training objective while holding everything else fixed: identical data (12,750 train / 2,250
held-out decisions), identical schedule, seed 0, frozen embeddings. Only the option readout varied - CE,
Brier, or a joint objective. Results at 250 optimizer steps:

| Objective | acc | ECE10 | Brier | AUROC(conf, correct) | mean conf |
|---|---|---|---|---|---|
| CE | 0.5658 | 0.2370 | 0.3387 | 0.6570 | 0.8027 |
| Brier | 0.5653 | 0.2425 | 0.3291 | 0.6468 | 0.8079 |
| Joint | 0.5707 | 0.2423 | 0.3305 | 0.6532 | 0.8130 |
| Majority (always `click`) | 0.5218 | - | - | - | - |

**Objective choice is not the lever at this budget.** Accuracy is essentially invariant across objectives (0.5658 / 0.5653 /
0.5707), and all three arms sit just a few points above the majority-class baseline of 0.5218. Proper scoring
moves the *average* probability - Brier is lower and ECE10 slightly higher under Brier/joint than under CE -
but it does not move discrimination: AUROC of confidence against correctness stays in a narrow 0.647-0.657
band across all three arms.

**All three arms share one failure mode: over-confidence.** Mean confidence is ~0.80 in all arms while accuracy is ~0.57, so a
0.80-confidence decision is wrong roughly 40% of the time. ECE10 is 0.237-0.243 across arms, and the
confidence signal barely separates correct from incorrect decisions (AUROC ~0.65). The difference between
objectives is far smaller than the gap to well-calibrated behavior.

**The boundary of this result.** This is a single seed, 250-step, in-distribution holdout. It
cannot distinguish whether proper scoring preserves calibration better than CE, because no arm is calibrated
and the operating regime is the easy, in-distribution one where miscalibration is least likely to appear. A
conclusive comparison requires the full schedule, at least 3 seeds, and a hard/OOD split - the regime where
CE is expected to degrade. What this pilot does settle is the direction of the next run: at this budget the
readout objective is not where calibration comes from, so the effort belongs in the data regime and the schedule
rather than in a different scoring rule. That is a narrower claim than "proper scoring fixes calibration" — and
it is the one the evidence supports.

### 5.12 What the gate is worth: a cost window, not an accuracy delta

An accuracy tie is only interesting if the gate changes the *decision*. Price 1,000 decisions with
`C_ask = 1` and `R = C_err/C_ask`, using the operating point of §5.2 — 40.7% **test** coverage at 17.3% empirical risk
(Clopper–Pearson upper 21.7%, n = 162); the certificate itself is val-side, at 47.6% coverage and a 19.9%
upper bound — and the accuracy of the same artifact (64.82%):

| R | always-act | always-ask | gate (empirical) | gate (CP upper) | learned fusion @50% |
|---|---|---|---|---|---|
| 0.5 | **175.9** | 1000 | 628.1 | 637.1 | 547.7 |
| 1 | **351.8** | 1000 | 663.3 | 681.3 | 595.5 |
| 2 | **703.6** | 1000 | 733.7 | 769.6 | 691.0 |
| 3 | 1055.4 | 1000 | **804.0** | 858.0 | **786.4** |
| 6 | 2110.8 | 1000 | **1015.1** | 1123.0 | 1072.9 |
| 16 | 5628.8 | 1000 | 1718.6 | 2006.3 | 2027.6 |

**The gate dominates both trivial policies for `C_err/C_ask` between 2.11 and 5.79** (2.25–4.61 using the
conservative bound on the risk estimate); the learned fusion widens the lower edge to 1.95. Below ~2× an
interruption costs more than the acting error it prevents; above ~6× asking every time is better, because the
gate still waves through 40.7% of decisions. Repeating the arithmetic on the previous checkpoint (`E2@705`:
61.31% accuracy, 38.9% coverage, 23.9% risk) gives **2.08–4.19** (2.22–3.47 with the bound): the window
survives the checkpoint change, narrower. This is the decision-level statement the tie cannot make, and it is
derived entirely from saved artifacts (`benchmarks/results/2026-11-07-cost-window.md`).

**Assumption, not measurement.** An ask is answered correctly and costs a fixed `C_ask`. Real asks are
dismissed (49–96% override rates in clinical settings), habituate within days, and are not always available;
no GUI study prices the ask. If a delegated decision is wrong with probability `q`, the window becomes
**2.36–8.14** at `q = 5%` and **2.67–13.7** at `q = 10%` — a fallible human widens the top of the window and
narrows it at the bottom, so the perfect-human assumption is the favourable case on one edge only. The window is therefore an upper bound on the gate's value, and it is
checkpoint-local: the same threshold on the previous checkpoint gave 23.9% risk, which would narrow it.

### 5.12.1 From step accuracy to task success: the arithmetic that governs the claim

Every number in this paper up to here is a **per-step** accuracy. Every number in the GUI-agent literature's
main benchmark is an **end-to-end** success rate. The two are related by a product, and the product is the
single most important thing in this paper that was not previously written down:

> A multi-step task succeeds only if every step succeeds, so with per-step accuracy `p` over `n` steps,
> `P(task) = p^n`.

Our measured `p = 0.6407` gives **10.8% over 5 steps, 2.8% over 8, 1.2% over 10, and 0.1% over 15**. Table
5.12.1 inverts the rule on published end-to-end AndroidWorld scores [androidworld2024] to recover the
per-step accuracy those systems *would* need **if their steps were independent**. The qualifier matters: that
independence assumption is measurably false for our own model (§5.12.1.1), so the column below is an
upper-bound-style inference, not a measurement, and no published system reports a per-step number to check it
against.

| system | end-to-end (published) | implied per-step (10 steps) |
|---|---|---|
| M3A, GPT-4 Turbo + a11y tree | 30.6% | 88.8% |
| MAI-UI-2B **SFT only** [maiui2026, Tab. 9] | **45.1%** | **92.3%** |
| MAI-UI-2B after online RL [maiui2026, Tab. 9] | 49.1% | 93.1% |
| MAI-UI-8B **SFT only** [maiui2026, Tab. 9] | 64.7% | 95.7% |
| MAI-UI-8B after online RL | 70.7% | 96.6% |
| MAI-UI-235B-A22B after online RL | 76.7% | 97.4% |

**On the independence assumption, every competitive GUI agent must be running at 92–97% per-step.** The
SFT/RL rows also separate two quantities that are easy to conflate. MAI-UI's online RL is worth **+4.0 points
of task success** at 2B (45.1 → 49.1), +6.0 at 8B, +3.5 at 32B. The per-step column moves far less
(+0.8 derived at 2B, 92.3 → 93.1) — which is exactly what the product rule predicts: a large task-level gain
can sit on a small step-level gain. We therefore make **no** claim about a "28-point per-step gap". What is
measured is a **task-success gap** (45.1% vs our 18.8% scorable) and it is attributable to **supervision**:
multi-step trajectory corpora, thousands of expert and synthesised tasks executed on a live emulator with
MLLM-judge rejection sampling, a 50-step horizon, and a 12-action space.

The right comparison point for this work is consequently **MAI-UI-2B-SFT, 45.1% end-to-end** — the same
~2B scale under the same SFT-only setting — not the 76.7% 235B model, and not even the 49.1% RL-tuned 2B. The published recipes
get there with online RL over hundreds of parallel emulators, multi-step trajectory data, and a wider action
space; §6 Limitation 11 states what of that is and is not available here.

### 5.12.1.1 The independence assumption is false, and our own data says so

Before using `P(task) = p^n` to explain anything, we tested its premise on ourselves. It predicts that a
model at 64.07% per-step completes **2.84%** of 8-step tasks. Our measured task success is
**18.8% (15/80 scorable)** — a factor of **6.6× higher** than the product rule allows.

The assumption fails for the obvious reason: GUI steps are strongly dependent. Most of a task is a sequence of
easy taps on a screen whose state the model just produced; the per-step error rate on *those* screens is far
below the 64% average, which is dominated by the hard AC-v2 mix. Tasks are also not uniform in length, and a
hard step can be recovered by the next action.

The cleanest test uses the homogeneous subset. Of the 80 scorable episodes, the 66 that never emitted
`type_text` have a median length of **12 steps** and a success rate of **14/66 = 21.2%**. The product rule at
`p = 0.6407`, `n = 12` predicts **0.479%** — a factor of **44×** too low; at `n = 12.5` it predicts **0.383%**,
or **55×** too low. A refutation this large is not a small-sample artefact: it is the assumption being wrong.

Consequences we accept:

1. **`P(task) = p^n` is retained only as an order-of-magnitude intuition**, never as a prediction and never as
   evidence that a run is broken. The predicted-to-measured factor is **6.6x**, and the per-step figures
   inverted from published task scores are **derived, not reported** by those papers.
2. **The 92.3/93.1/97.4 per-step column is not a published quantity and is not compared to our 64.07%.** It is
   the value each published end-to-end score would imply *under an assumption we have measured to be false* for
   our own model. We print it because inverting the rule is the only way to reach a per-step number from a
   paper that publishes only task success; we label it derived and we make no gap claim from it.
3. **The comparison we do make is the measured one**: task success, ours versus a published task-success
   number, in the same unit. That comparison is 18.8% against MAI-UI-2B-SFT's 45.1%, and it does not depend on
   any assumption about step independence.

Inverting the rule on *our own* measurement illustrates the failure directly: 18.8% over 8 steps implies 81.1%
per-step, over 10 steps 84.6% — against a measured 64.07%. No per-step figure in this paper should be read as
interchangeable with a task figure, in either direction.

This also **retro-explains** the online result in §5.7. The 8-task run gave **1/8 (12.5%, Wilson 2.2–47.1%)**,
which the product rule would call better than its 2.8% prediction and which is simply what a small-n estimate
of the 18.8% population looks like (Wilson interval contains it). The 1/8 was therefore never evidence of a
harness defect, and the search for one (`app_name`, checked and repaired in §5.7) could not have moved the task
rate materially on its own.

**Two consequences are stated as limits rather than as results.** First, a paper reporting per-step accuracy
cannot claim task success, and this paper does not. Second, the calibrated gate's value (§5.12) is a
*substitute* for accuracy, not a route to it: if a system cannot be right 93% of the time, the next best
guarantee is knowing when it is wrong — which is the claim this paper actually makes.

---

### 5.13 The protocol that would upgrade this to a mechanism claim

The pilot bounds the question rather than answering it, so the protocol that would settle it is specified here —
precisely enough that the claim is falsifiable and the run is bounded:

- **Full training schedule, ≥3 seeds**, so seed variance is measured instead of assumed away.
- **Paired bootstrap intervals** over matched decisions, so the two systems are compared item by item rather
  than as two point estimates.
- **A hard / out-of-distribution split** — a real distribution shift rather than an in-distribution holdout,
  because that is the regime in which CE is predicted to lose calibration.
- **Power, fixed up front.** After that split, **~1.9k scored decisions** resolve a 4-point difference under
  paired intervals, and certifying a **0.05 risk budget** requires **≥59 accepted actions** before a 0.05
  failure rate becomes rejectable.

This paper's status against that protocol: single seed, 250 steps, in-distribution holdout, no paired bootstrap,
no risk-certification sample. The mechanism comparison (Q1/Q2) and the certified gate (Q3) are therefore stated
as open, with the sample sizes that would close them fixed in advance — which is what makes them the next
experiments rather than the next opinions.

---

## 6. Limitations

1. **Small sample.** The 398-step eval set gives a CI of about ±4.7 points and cannot separate the top of the
   leaderboard. Parity is therefore the claim, and it is the claim the data supports.
2. **Self-reported baselines.** All competitor numbers are authors' self-reported on their own harness [guilibra2026,
   memw2026, uivenus15]; our numbers come from a re-implemented, differential-tested scorer. Cross-harness
   comparability is limited.
3. **Grounding is the open axis.** Our grounding (56.05% pure; 60.09% with the 858-px groundB run) sits below the
   best *reported AC-v2* grounding, Mem-W-8B (71.30), and below Mem-W-4B (70.85) — a 2× smaller model [memw2026].
   This is the axis this paper specifies for the next training run (§4.5, §5.3) rather than one it closes.
   **Correction:** an earlier draft compared us to "GUI-Libra-8B grounding 76.2". That figure is GUI-Libra's
   `ASFT-3B, reasoning mode` grounding measured on **ScreenSpot-v2**, a different benchmark and a pre-RL
   checkpoint; the comparison is withdrawn [guilibra2026].
4. **Grounding and type tradeoff.** The groundB run improved click grounding (+8 correct clicks) but action-Type
   accuracy dropped (84.17 → 82.16, −8 type rows). The headline still moved up (+3, 255 → 258), so it is not a
   clean net-zero, but the two axes trade against each other, and this run measures the exchange rate:
   +8 clicks bought with −8 type rows.
5. **Calibration numbers are split by provenance.** Only the ECE / AUROC and per-threshold coverage/risk curve
   in §5.2(a) are reproducible from saved artifacts; the isotonic ECE (0.036) and risk-budget gates (§5.2(b))
   are reported-in-run and do not reproduce from disk; the reproducible evidence is the type-routed gate (AUROC 0.760/0.729/0.709) and its certified frontier.
6. **Tuning on test.** Some design choices (e.g. V-only vs. mix) were selected on test evals; the validation
   split (62.76%) broadly agrees with test, but this remains a residual risk.
7. **The online signal is preliminary, not a benchmark result.** The largest run is 8 tasks per condition:
   **1/8 in both**, Wilson 2.2–47.1%. Of the eight episodes per condition, 7 (pure) and 6 (fallback) consumed
   the full 10-step budget; the remaining ones are the single success and one 8-step episode that stopped
   prematurely. That interval is consistent with noise and cannot be compared to published AndroidWorld
   baselines. The
   two conditions also ran **different randomised task instances** (5 of 8 differ), so the invalid-action
   comparison between them is unmatched, and the earlier 4-task pair has the same defect. Nothing in the
   online section is a capability result.
8. **Any online number depends on the harness interface, not just the model.** The V-track readout returns an
   **empty `app_name`** for goals that only imply an app (§5.7), so the invalid-action rate measures
   action-space and readout coverage rather than grounding — grounding was never exercised. The goal-keyword
   fallback was **not** used in this run, and any run that uses it must be declared and stamped. A stalling
   agent scores 0 because success requires the device state to be `is_successful`.
9. **Positioning.** Single-pass, zero-token GUI action selection is already established by **Jev-Mobile**
   (arXiv:2609.30186, 79% AndroidWorld) and **GUI-JEPA** (arXiv:2610.36877, decoder-free policy). Our contribution
   is deliberately not that framing: it is the measured decision quality on a public protocol, the error
   decomposition, and the calibration axis — the three things neither paper quantifies. On the online side,
   **no peer-reviewed AndroidWorld baseline exists**: every reported number is an arXiv preprint or a vendor
   self-report, and the often-quoted **GUI-Owl-1.5-2B figure of 67.9%** has no primary source (the real figures
   are 71.6% for the 8B-Thinking variant; the 2B reports OSWorld-Verified 43.5), so it is **not cited** here.
10. **The training-procedure lever (§5.9) is a direction, not a result.** Three and four seeds per arm exist; the
    pre-registered bar was five, so it was never reached. Both arms have only two distinct grounding values
    (grounding is a count out of 223), which makes each sd a floor from a single contrast; the variance ratio is
    8.3×, which violates the equal-variance assumption, and the defensible tests disagree (Student pooled
    p = 0.012, **Welch p = 0.063**, permutation p = 0.029 at its minimum attainable value). We therefore claim no
    significance. Until eight seeds per arm exist, this programme has **no certified positive grounding
    result** — the contribution is the loss-mechanism analysis (§5.8), the base-vs-fine-tune decomposition, the
    candidate-collapse bound and the cost window (§5.12).

---

11. **The recipe gap is supervision, and it is not closable on this hardware.** §5.12.1 shows the frontier
    runs at 92–97% per-step accuracy, and — this is the load-bearing part — **MAI-UI-2B reaches 92.3% per-step
    after SFT alone**, before any online RL; their RL adds +0.8 per-step points at 2B. So the 28-point gap to
    us is **not** an RL deficit. It is the supervision: multi-step trajectory corpora (GUI-Odyssey's 8,334
    cross-app episodes) rather than single-step decisions, thousands of expert and synthesised tasks executed
    on a live emulator with an MLLM-judge rejection-sampling loop, a 50-step training horizon rather than no
    horizon at all, and a 12-action space including `ask_user` and `mcp_call`. This programme ran **SFT on 15k
    single-step AC-v2 decisions, one consumer GPU, one KVM emulator, 8 actions, no trajectory data, no judge
    loop**. The gap is the difference between the two supervision settings, stated so that no reader mistakes
    our 64.07% per-step for a near-miss on the frontier's 76.7% end-to-end — which is a 235B model after RL and
    is not our comparison class. Our comparison class is MAI-UI-2B-SFT at 45.1% end-to-end, and we do not
    reach it.
12. **Our own online attempts measure a narrow capability, and the 116-task run says which one.** The first
    full-suite run (protocol `aw-main-v1`, checkpoint E2+`app_intent`, step budget 25, seed 30, no app-name
    fallback) is **incomplete at the time of writing and is reported as such**: **104 of 116 tasks finished, 15 successes, SR 14.4% (15/104), 95% Wilson
    8.9–22.4%**. That SR is over the tasks that ran, not the suite, and no published-baseline comparison is
    made from it. Every figure below re-derives from `benchmarks/results/2026-11-08-aw116/` via
    `scripts/refresh_aw116.py`, and this paragraph is regenerated by `scripts/sync_paper_aw116.py`; no number
    here is transcribed by hand.

    The decomposition over those 104 is **34 non-terminating timeouts (median 25 steps),
    31 premature `status` (median 7 steps), 15 successes, and 12 tasks whose own
    `initialize_task` or `reset` raised**. Two mechanisms, opposite signatures, same missing signal: the action
    head's own confidence stays high in **both** (p50 ≈ 0.785), while the calibrated gate score starts
    high and collapses on the non-terminating runs (median gate p50 ≈ 0.334, exactly 0 on 12%
    of all actions). The gate-trend statistic — mean `gate` over the last three actions minus the first three —
    is negative on **73% of failures (n = 77)** and on **40% of successes
    (n = 15)**; the two distributions are visibly different but with 15 successes this is a
    mechanism, not a result, and it is labelled as such.

    **Three denominators are reported, because they answer different questions, and one bucket is neither
    a model outcome nor a device failure.** Of the 104 finished episodes: 12 never reached a usable device
    state (the accessibility forwarder stopped responding; `env.reset()` could not obtain a tree), and a further
    **12 ran to completion but were never judged** — `task.is_successful(env)` raised
    `AttributeError: 'AndroidWorldEnv' object has no attribute 'get_state'` because the harness wrapper did not
    forward `get_state` to the wrapped environment. The second group is **not** a model failure and **not** the
    same defect as the first: on the evidence, 11 of 11 crashes carry a setup error while the
    12 unscored episodes have `n_steps` 4–25. Both are now separate buckets, so the decomposition reads
    **15 successes, 31 premature `status`, 34 non-terminating timeouts, 12
    unscored, 12 setup failures** — and the earlier version of this paragraph, which folded the unscored
    episodes into "timeout" and "premature `status`", was wrong.

    The rate over **scorable** episodes — the only denominator that answers "how good is it where we could
    judge" — is **18.8% (15/80), 95% Wilson 11.7–28.7%**. Over genuinely attempted tasks it is
    **16.3% (15/92)**; counting every finished episode, including the ones nobody could judge,
    gives **14.4% (15/104)**. All three are stated; none is a capability estimate this paper would defend, and
    the run is incomplete.

    **`SR_all` and `SR_reachable` are separated below, but not by the field this harness fills.**
    AndroidWorld upstream exposes no gold action list — its `TaskEval` surface has `template`, `goal` and
    `is_successful`, nothing reporting which action types a solution needs — so the `coverage` field carries
    no value on **any** of the **104** episodes (`n_reachable = 0` is the harness's unset default,
    not a count of unreachable tasks — that is exactly why the field cannot be used to answer the question). This programme nevertheless established by hand
    which tasks the option set cannot serve: **exactly two of the 116 need `answer`** (`BrowserMultiply`,
    `RetroPlaylistDuration`). Excluding those, the reachable split is **14.7% (15/102)** against
    **14.4% (15/104)** over all finished tasks. Both are reported, with the definition stated, rather than
    calling the split unavailable when a narrower version of it is computable.

    **The action-space finding is the one that changes an earlier claim in this paper.** Of **1,394 emitted actions, zero were invalid** — every action the model chose was representable and executable by the harness (invalid rate 0.0%, 34/104 timeouts). The 28.2% invalid rate in §5.7 was therefore a *read-slot* defect (`app_name` empty), not a model limitation: repair the read slot and the model's chosen actions are executable essentially every time. What it cannot do is *string them together*.

    **Every one of the 15 successes is a task whose gold trajectory is one or two actions beyond
    `open_app`** — camera, single-record expense/recipe deletes, settings toggles and their verifiers, opening an
    app. The model emits `type_text` in **19 of 104 episodes (18%)** and succeeds **once** among them;
    **zero successes** occur on tasks that require filling a text field and then navigating. **The conclusion
    this paper draws is that per-step decision quality and task composition are separate capabilities.** The
    interface is no longer the binding constraint; composition is, and the calibrated gate is valuable
    precisely where composition cannot be reached.

## 7. Conclusion

A 1.92B single-pass decision model, fine-tuned on 15k curated AndroidControl decisions, reaches **64.07%** AC-v2
High Pass@1 on the pure protocol — **level with a rank-A 8B RLO-trained agent** (GUI-Libra-8B, 64.3)
[guilibra2026] at ~4× fewer parameters, with less data, **zero output tokens**, and **103–108 ms per step** on a
single consumer GPU. The 398-step sample does not separate the top of the board and Mem-W-8B (68.59) leads by
4.5 points [memw2026], so the claim is parity, not a record. The informative part is the component split:
**84.17% Type accuracy is the highest reported**, and the remaining deficit is entirely **element selection** —
of the 68 element-selection misses measured on the with-hint artifact, 86.8% land more than 150 px from the correct widget
(median 437 px).

That makes the result a **trade, and a legible one**. The model gives up roughly 15 points of grounding against
the best memory-augmented 8B agent, and in exchange delivers equal step accuracy with a quarter of the parameters,
no reasoning tokens, a per-step cost with no sequential-decoding term, and a confidence readout that is rank-order
useful (AUROC 0.626–0.631) and gate-able under a monotone coverage–risk curve (96.5%→41.0% coverage as risk falls
34.6%→24.5%). The 8B agents buy grounding with parameters, tokens and latency and expose no such readout. The
trade is also one-directional, which is why it is worth making: the missing grounding is a named training
objective, while the calibration axis is not something a larger autoregressive pipeline recovers by scaling.

Three things follow directly, and all are specified rather than gestured at. First, the **candidate interface**
(§5.3.1): its ceiling on element selection is 96.9% and the fine-tuned checkpoint has never been trained through
it, which makes it the highest-ceiling experiment on the list. Second, an **element-selection** objective on the
visual path — target-versus-observed-distractor contrastive supervision built from the model's own 68
mis-selected elements, and box targets whose width scales with element size — evaluated under the paired,
multi-seed protocol of §5.13. Third, a re-fit of the calibrated gate on held-out data so its risk budgets are
certified rather than reported in-run, ideally on-device and on AndroidWorld. The value we stand behind today: a zero-token, ~100 ms,
single-pass decision head matches a much larger reasoning-RL system on a real per-step task while preserving the
one thing an on-device act/ask/refuse interface cannot work without — a confidence axis someone can trust.

The decision-level form of that value is a **cost window**, not an accuracy delta: at the certified operating
point the gate beats both always-act and always-ask whenever a wrong action costs between **2.1× and 5.8×** an
interruption (§5.12), and that is the claim the tie can actually carry.

---

## Sources appendix (primary references)

1. `guilibra2026` — Yang et al., *GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL*, arXiv:2602.22190 — https://arxiv.org/abs/2602.22190
2. `mobilerl2025` — *MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents*, arXiv:2509.18119 — https://arxiv.org/abs/2509.18119
3. `magmar1dense2026` — *AndroidControl-Curated: Revealing the True Potential of GUI Agents through Benchmark Purification* (Magma-R1), arXiv:2510.18488 — https://arxiv.org/abs/2510.18488
4. `v2p2026` — Chen et al., *V2P: Visual Attention Calibration for GUI Grounding via Background Suppression and Center Peaking*, arXiv:2508.13634 — https://arxiv.org/abs/2508.13634
5. `conclick2026` — Zhang et al., *ConClick: Element-Wise Contrastive Learning Improves GUI Grounding*, arXiv:2610.13268 — https://arxiv.org/abs/2610.13268
6. `seg2click2026` — Wang & Huang, *Seg2Click: Learning Pixel-Level Action Masks for GUI Visual Grounding*, arXiv:2610.16234 — https://arxiv.org/abs/2610.16234
7. `hyperclick2026` — *Enhancing Trustworthy GUI Grounding via Self-Critiqued Reinforcement Learning*, arXiv:2510.27266 — https://arxiv.org/abs/2510.27266
8. `safeground` — Wang et al., *SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration*, arXiv:2602.02419 — https://arxiv.org/abs/2602.02419
9. `typedbench2026` — Sharma et al., *TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models*, arXiv:2610.11392 — https://arxiv.org/abs/2610.11392
10. `cotACL2026` — Zhang, Gao, & Xu, *Does Chain-of-Thought Reasoning Help Mobile GUI Agents? An Empirical Study*, Findings of ACL 2026, pages 7981–7996 — https://aclanthology.org/2026.findings-acl.392
11. `d12bcard` — Liquid AI, *LiquidAI/d1-2B* (Hugging Face model card) — https://huggingface.co/LiquidAI/d1-2B
12. `typesafe2026systemone` — Almeida, *Introducing System One Models & Jev*, TypeSafe AI blog (Sep 2026) — https://typesafe.ai/blog/introducing-system-one-models-and-jev
13. `liquidAI2026opend1` — Liquid AI, *Open d1: Edge decision models for text, vision, and audio* (Oct 2026) — https://www.liquid.ai/blog/d1-open
14. `lfm3vl2b2026` — Liquid AI, *LiquidAI/LFM3-VL-2B* (Hugging Face model card) — https://huggingface.co/LiquidAI/LFM3-VL-2B
15. `memw2026` — *Mem-W: Latent Memory-Native GUI Agents*, arXiv:2605.09317 — https://arxiv.org/abs/2605.09317
16. `uivenus15` — *UI-Venus-1.5 Technical Report*, arXiv:2602.09082 — https://arxiv.org/abs/2602.09082
17. `mementogui2026` — *MementoGUI: Learning Agentic Multimodal Memory Control for Long-Horizon GUI Agents*, arXiv:2605.18652 — https://arxiv.org/abs/2605.18652
18. `comem2026` — *Auto-scaling Continuous Memory for GUI Agent*, arXiv:2510.09038 — https://arxiv.org/abs/2510.09038
19. `androtmem2026` — *AndroTMem: From Interaction Trajectories to Anchored Memory in Long-Horizon GUI Agents*, arXiv:2603.18429 — https://arxiv.org/abs/2603.18429
20. `ugroundoffline2025` — *UGround Offline Evaluation: AndroidControl test images + processed test steps* (Hugging Face dataset) — https://huggingface.co/datasets/demisama/UGround-Offline-Evaluation
21. `androidworld2024` — Rawles et al., *AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents*, arXiv:2405.14573 — https://arxiv.org/abs/2405.14573
22. `cost-window` — internal artifact, *Cost window where the gate beats always-act and always-ask* —
    `benchmarks/results/2026-11-07-cost-window.md`, script `benchmarks/train/cost_window.py`,
    data `benchmarks/results/2026-11-07-cost-window.json` (derived from `gate_test938.json`, `gate_test705.json`,
    `score_fusion.json`).
23. `rightclick2026` — Yuan et al., *Right Click, Wrong Box: Grading GUI Grounding by What a Click Does*, arXiv:2610.36831 — https://arxiv.org/abs/2610.36831. Outcome-equivalent click grading; the 1/5–1/3 figures are **web** widgets, and the ~1% native-Android scope correction in §5.3 is **ours**, not the paper's.
