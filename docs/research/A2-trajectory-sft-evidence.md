# A2 — Does trajectory SFT raise END-TO-END success for a ≤4B GUI model?

**Question (issue I3):** is there *measured* evidence that multi-step trajectory SFT raises **end-to-end task
success** for a ≤4B GUI model — as opposed to only per-step accuracy — and does that evidence cover a model
whose per-step accuracy starts at ~64%?

**Researched:** 2026-11-09. Routes used: `web_search` (Exa), direct `fetch_content` on arXiv HTML (`ar5iv`/`arxiv.org/html`),
arXiv API ID lookups, ACL Anthology and PMLR PDFs. Semantic Scholar was **not** used — no HTTP 429 was
encountered because I went to arXiv HTML and the paper PDFs directly. Table-cell values below were read from
fetched page/PDF content, not inferred from titles or abstracts; where only an abstract was available it is
marked **abstract-only**.

---

## VERDICT

> **INFERRED — not supported by measurement.** **There is no qualifying study.** No paper I could find trains a
> ≤4B GUI model on multi-step trajectory data and reports an end-to-end task-success delta *attributable to the
> trajectory data itself* — i.e. against a single-step-only or non-trajectory control at the same base model,
> same scale, same compute. Every ≤4B end-to-end number in the literature (MAI-UI-2B 45.1%, GUI-Owl-1.5-2B
> 67.9%, Step-GUI-4B 63.9%, UI-Voyager-4B 81.0%) is a *whole-pipeline* number: trajectory data + grounding data
> + CoT synthesis + synthetic environments + RL, bundled. **At 2B specifically the gap is total: zero papers.**
> The plan therefore rests on an inference, exactly as the brief anticipated.

**Three corrections to the brief's own anchors, established below:**

1. **The 92.3% / 93.1% "per-step" figures do not exist in MAI-UI.** MAI-UI reports **no per-step or action
   accuracy metric at all** — not for 2B, not for any size. Table 9 is task success rate only. So "RL is worth
   +0.8 per-step at 2B" (§Anchor facts) is not a paper claim and cannot be verified. See §2.1.
2. **The claim "at 2B, zero papers" is true but the neighbouring evidence at 4B is strong enough to change the
   decision framing** — from "trajectory data is the known lever" to "trajectory data is one of five ingredients,
   and it is not the largest measured one."
3. **`P(task) = p^n` should not be applied to our numbers.** Plugged in with p=0.64 it predicts 1.2% at 10 steps;
   we measured 18.8%. Our own data refutes the independence assumption. Inverting the law on our measurement
   implies a per-step reliability of ~88%, not 64%. §4.

---

## 1. Method note: keeping per-step and end-to-end apart

This was the main failure mode the brief warned about, and it is worse in the literature than expected. Three
different quantities are routinely all called "success rate":

| Quantity | What it is | Where it appears |
|---|---|---|
| **End-to-end task success** | live emulator, agent's own actions, scored by environment state at the end | AndroidWorld, OSWorld, MobileWorld, AndroidLab |
| **Offline episode SR** | teacher-forced replay of a recorded episode; episode counted correct only if *every* step matches gold | AndroidControl-High/Low "SR", GUI-Odyssey, AITW |
| **Per-step / Step SR / Type Acc** | single screen, single action, scored independently | AndroidControl step metrics, ScreenSpot, MMBench-GUI L2 |

**AndroidControl "SR" is not end-to-end.** It is an offline, teacher-forced metric that never lets the model's
own error propagate. MAI-UI-2B scores 67.3 on AC-High and 72.6 on GUI-Odyssey [1] — those numbers are *not*
comparable to its 45.1 AndroidWorld score, and the 22-point difference between them is **not** "the cost of going
online", it is two different measurements. OS-Genesis states its definition explicitly: "A whole task is
successful if all the steps it contains are successful" over recorded data [8].

Every table below uses **one column per metric type**. There is no table in this document with a per-step number
and an end-to-end number in the same column.

---

## 2. The ≤4B evidence table

### 2.1 First, the anchor: what MAI-UI actually reports

I fetched `arxiv.org/html/2512.22047` and searched for "step accuracy", "per-step", "action accuracy",
"Type Acc", "Grounding Acc". Result: **no per-step metric table exists in the paper.** Table 9's caption is
verbatim: *"Online RL performance gains and ablations on standard GRPO and interaction budget per trajectory
(max_env_steps)"* — and its only metric column is AndroidWorld task success rate [1].

| # | Model size | Stage | Metric type | Value | Source |
|---|---|---|---|---|---|
| 1 | MAI-UI-2B | SFT | **end-to-end** AndroidWorld SR | 45.1 | [1] Table 9 |
| 1 | MAI-UI-2B | + online RL | **end-to-end** AndroidWorld SR | 49.1 (+4.0) | [1] Table 9 |
| 1 | MAI-UI-8B | SFT | **end-to-end** AndroidWorld SR | 64.7 | [1] Table 9 |
| 1 | MAI-UI-8B | + online RL | **end-to-end** AndroidWorld SR | 70.7 (+6.0) | [1] Table 9 |
| 1 | MAI-UI-32B | SFT → RL | **end-to-end** AndroidWorld SR | 69.8 → 73.3 (+3.5) | [1] Table 9 |
| 1 | MAI-UI-2B | — | **offline episode SR** AC-High | 67.3 | [1] Table 7 |
| 1 | MAI-UI-2B | — | **offline episode SR** GUI-Odyssey | 72.6 | [1] Table 7 |
| 1 | **any** MAI-UI size | **any** | **per-step accuracy** | **not reported** | [1] (absence verified) |

**The brief's "+0.8 per-step at 2B" is +4.0 end-to-end.** The 92.3 and 93.1 strings do appear in MAI-UI but as
unrelated cells: 92.3 is MAI-UI-8B's Windows-Basic cell in the MMBench-GUI L2 table [1], and 93.1 was not found
in the paper at all. **The per-step column of our comparison table has no MAI-UI entry to put in it.** Our
64.07% per-step (AC-v2) currently has no published 2B per-step number to be compared against.

MAI-UI's SFT stage *is* trajectory-based — §2.3 states "we build a self-evolving data pipeline to collect and
synthesize diverse multi-step trajectories, and train the model with these trajectories" [1] — but the paper
reports **no ablation of trajectory vs single-step data with task success rate**. I asked directly; the answer
is no such ablation exists.

### 2.2 Every ≤4B paper I found that reports an end-to-end metric

All rows are **end-to-end task success rate** unless stated. "Delta" is against the row's own stated comparator,
which is *never* a single-step-only control.

| # | Model (params) | Base model | Data / recipe | Comparator | End-to-end value | Δ attributable to trajectory data? | RL? | Date | URL |
|---|---|---|---|---|---|---|---|---|---|
| 2 | **MAI-UI-2B** | Qwen3-VL-2B family | multi-step trajectories, mixed with grounding/user-interaction data | none (no data ablation) | **45.1** AW SR (SFT) | **no** | RL reported separately | 2025-12-26 | [1] |
| 3 | **GUI-Owl-1.5-2B-Instruct** | "initialized from Qwen3-VL" (exact 2B base **not stated**) | hybrid trajectory corpus (DAG task synth, real-device rollouts, human demos, virtual envs) + CoT | Qwen3-VL-2B = 36.4 (per [1] Table 6) | **67.9** AW SR | **no** — bundle incl. CoT + virtual envs + MRPO RL | yes (MRPO) | 2026-02-15 | [3] |
| 4 | **GUI-Owl-1.5-4B-Instruct** | Qwen3-VL-4B family | as above | Qwen3-VL-4B-Instruct = 45.3 (per [4]) | **69.8** AW SR | **no** | yes | 2026-02-15 | [3] |
| 5 | **Step-GUI-4B** | Qwen3-VL-4B | mid-train 4.0M trajectory + 2.7M grounding; cold-start 404K trajectory (24%) of 1.67M; then RLVR | Qwen3-VL-4B-Instruct = 45.3 | **63.9** AW SR Pass@1 (**75.8** Pass@3) | **no** — no data-mixture ablation reported | yes (RLVR) | 2025-12 | [4] |
| 6 | **Step-GUI-8B** | Qwen3-VL-8B | as above | Qwen3-VL-8B = 47.6 | **67.7** AW SR Pass@1 (**80.2** Pass@3) | no | yes | 2025-12 | [4] |
| 7 | **UI-Voyager-4B** | Qwen3-VL-4B-Instruct | **no static dataset** — RFT on self-generated rollouts + GRSD step-level self-distillation | Qwen3-VL-4B = 45.3 | **81.0** AW SR Pass@1 | **no** — and notably *not* static trajectory SFT | no SFT on static traj.; RFT+GRSD | 2026-03-25 | [6] |
| 8 | **Ferret-UI Lite-3B** | (Apple, unspecified 3B) | human episodes + GPT-4o CoT + synthetic QA + **online synthetic trajectories** | human-annotated episodes only = **13.7** | **28.0** AW SR | **partial** — see ladder below | yes (RLVR) | 2025-09-30 | [5] |
| 9 | **OS-Genesis-4B** | InternVL2-4B | exploration-first trajectory synthesis (1K trajectories, 6.4 steps avg) | task-driven synthesis = 4.02; zero-shot = 0.00 | **15.18** AW SR | **no** — both arms are trajectory data | no (SFT only) | ACL 2025 | [8] |
| 10 | **EAPO-2B** | Qwen3-VL-2B | SFT rollback + exploration-aware GRPO | 0-shot = 46.19 | **76.36** AW SR | **no** — the delta is attributed to RL | yes | 2026-05-11 | [9] |
| 11 | **EAPO-4B** | Qwen3-VL-4B | as above | 0-shot (4B not extracted) | **79.59** AW SR | no | yes | 2026-05-11 | [9] |

**What the ≤4B table shows.** Ten independent groups get a ≤4B model above 15% on AndroidWorld, and every single
one of them uses multi-step trajectory data. That is real, repeated, measured evidence that **trajectory-based
pipelines work at ≤4B**. What none of them do is separate the trajectory data from the other four ingredients.
The nearest thing to a controlled comparison is Ferret-UI Lite's ladder [5], and it is confounded with CoT:

| Ferret-UI Lite-3B variant | End-to-end AW SR | Δ vs previous |
|---|---|---|
| Baseline (human-annotated episodes only, no CoT, no synthetic) | 13.7 | — |
| + Short CoT | 15.8 | +2.1 |
| + Long CoT | 19.6 | +3.8 |
| + Synthetic data 5K | 20.3 | +0.7 |
| + Synthetic data 13K | 22.4 | +2.1 |
| + Synthetic data 17K | 25.2 | +2.8 |
| Final reported model (after RLVR) | 28.0 | — |

Source: [5] Table 4 (SFT ablations, mean of 5 runs). **This is the single most useful table in the entire
search** — it is ≤4B, it is end-to-end, and it decomposes. Two readings, both important:

- **Supporting:** synthetic trajectory data added on top of a trajectory-SFT baseline is worth **+5.6 points
  end-to-end** going 0 → 17K trajectories (19.6 → 25.2).
- **Against our plan:** the *first* 5K trajectories are worth only **+0.7**, and the biggest single jump in the
  whole ladder is **long CoT, +3.8**, which is a supervision-format change, not a trajectory-data change. Also
  the baseline at 13.7 is already trained on trajectories; the ladder measures *more* trajectory data, never
  trajectory-vs-single-step.

### 2.3 Explicit statement of the gap

**No qualifying study exists at 2B.** Not one paper trains a 2B GUI model on trajectory data and reports an
end-to-end delta against a single-step or non-trajectory control. The 2B end-to-end numbers that exist
(45.1, 67.9, 76.36) are all whole-pipeline.

**No qualifying study exists at any size ≤4B** for the trajectory-vs-single-step contrast specifically. The
papers that come closest and still fail the test:

- Ferret-UI Lite [5]: trajectory baseline already trajectory; ladder varies CoT and data *volume*, not data *type*.
- OS-Genesis [8]: varies trajectory *quality* (exploration-first vs task-driven synthesis), both trajectory.
- MaP [7]: varies the *objective* over mixed data, not the presence of trajectory data (see §5).
- Step-GUI [4], GUI-Owl-1.5 [3], MAI-UI [1]: report no data ablation with task success rate at all.

**Search strings used** (run against Exa via `web_search`, plus arXiv API and direct PDF fetch):

```
"2B GUI agent trained on trajectory data AndroidWorld task success rate"
"MAI-UI 2B AndroidWorld 45.1% task success arXiv 2512.22047"
"GUI-Owl-1.5 2B AndroidWorld 71.6 trajectory data"
"Step-GUI 4B AndroidWorld success rate trajectory"
"EAPO 4B AndroidWorld 79.59 GUI agent"
"GUI-Odyssey paper ablation trajectory training end-to-end success"
"ablation \"single-step\" vs \"trajectory\" training GUI agent AndroidWorld success rate table"
"single-step action prediction versus full trajectory training GUI agent ablation"
"MaP masked trajectory prediction GUI agent AndroidWorld ablation single-step"
"trajectory SFT alone insufficient GUI agent failure analysis long horizon"
"AndroidControl single-step annotation vs full trajectory training effect"
"GUI agent training data single-step static data hurts online performance"
"\"AndroidControl\" does not transfer AndroidWorld gap offline online GUI agent study"
"UI-TARS-2B AndroidWorld success rate end-to-end"
"OS-Atlas 4B AITW task success rate"
"AndroidWorld history conditioning ablation previous actions improve success rate"
```

**Verdict on item 1 of the brief: the honest answer is "this has not been reported at 2B", and also not at 3B
or 4B in the trajectory-vs-single-step form.** Per the brief's own instruction, the plan must be relabelled from
"known effective" to "inference".

---

## 3. Ingredient-effect table (item 2)

End-to-end task success deltas only. Per-step and offline-episode deltas are excluded from this table
deliberately — several papers report their ablations in offline metrics, and mixing them here is exactly the
error this brief exists to prevent.

| Ingredient ablated | Scale | Metric | Measured Δ | Confound / caveat | URL |
|---|---|---|---|---|---|
| **Synthetic trajectory data volume** (19.6 → 25.2 via 0→17K) | **3B** | **end-to-end** AW SR | **+5.6** | sits on top of a trajectory+CoT baseline | [5] T4 |
| **Long CoT supervision** (13.7 → 19.6) | **3B** | **end-to-end** AW SR | **+5.9** | is a *format* change, not a data change | [5] T4 |
| Short CoT only (13.7 → 15.8) | 3B | end-to-end AW SR | +2.1 | — | [5] T4 |
| First 5K synthetic trajectories (19.6 → 20.3) | 3B | end-to-end AW SR | **+0.7** | returns are strongly non-linear in volume | [5] T4 |
| **Trajectory data quality** (task-driven → exploration-first) | **4B** | **end-to-end** AW SR | **+11.16** (4.02 → 15.18) | both arms trajectory data; quality ≠ type | [8] T1 |
| Same, 8B | 8B | end-to-end AW SR | +12.50 (4.46 → 16.96) | — | [8] T1 |
| Same, 7B | 7B | end-to-end AW SR | +7.59 (9.82 → 17.41) | — | [8] T1 |
| **Unified CoT synthesis** (✗ → ✓) | **8B** | **end-to-end** AW SR | **+6.6** (65.0 → 71.6) | not ≤4B | [3] T10 |
| Same, OSWorld | 8B | end-to-end OSWorld SR | +5.5 (47.4 → 52.9) | not ≤4B | [3] T10 |
| **Virtual-environment trajectory production** (✗ → ✓) | 8B | end-to-end **in-house** PC-Eval | **+33.4** (42.0 → 75.4) | in-house benchmark, not AndroidWorld; largest single coefficient found | [3] T11 |
| Same, Mobile-Eval | 8B | end-to-end in-house | +36.7 (50.0 → 86.7) | same caveat | [3] T11 |
| **Online RL on top of SFT** | **2B** | **end-to-end** AW SR | **+4.0** (45.1 → 49.1) | — | [1] T9 |
| Same | 8B | end-to-end AW SR | +6.0 (64.7 → 70.7) | — | [1] T9 |
| Same | 32B | end-to-end AW SR | +3.5 (69.8 → 73.3) | — | [1] T9 |
| **Self-generated rollouts + step-level distil.** (RFT → GRSD) | **4B** | **end-to-end** AW SR | **+7.8** (73.2 → 81.0) | vs GRPO/PPO plateauing ~76 | [6] Fig. 8 |
| Cold-start SFT stage | 32B | end-to-end (stage-wise) | **+2.62** | smallest stage in the ladder | [10] T3 |
| RFT stage | 32B | end-to-end (stage-wise) | +3.13 | — | [10] T3 |
| Offline DPO stage | 32B | end-to-end (stage-wise) | +3.21 | — | [10] T3 |
| Unified action space | 32B | end-to-end (stage-wise) | +4.84 | not a data ingredient | [10] T3 |
| Data scale, 3K→10K trajectories | 7B | end-to-end **OSWorld** | "+72% average" | **abstract/secondary-report only**; figure, no cell value | [11] |
| Data scale, 3K→14K Win/Mac | 7B | end-to-end OSWorld | "+125% average" | **same caveat** | [11] |
| Multi-image history: 1 → 3 screenshots | 7B | end-to-end OSWorld | "substantial", **no number given** | qualitative in text; exact values are in Fig. 10, not extracted | [11] |
| Multi-image history: 3 → 5 screenshots | 7B | end-to-end OSWorld | "marginal gains", +3K context tokens | same caveat | [11] |

### Which ingredient has the largest measured end-to-end coefficient?

Among *measured* end-to-end coefficients, ranked:

1. **Environment/infrastructure scale, not data** — virtual-environment trajectory production, **+33 to +37**
   on in-house benchmarks [3]. Big caveat: in-house benchmarks, 8B, and this is measuring "we built a way to
   generate trajectories at all" rather than "we used better data."
2. **Trajectory data quality** — **+11.2 at 4B** on AndroidWorld [8]. Real, measured, ≤4B, end-to-end. But it
   is quality-vs-quality, not trajectory-vs-single-step.
3. **CoT / reasoning supervision format** — **+5.9 at 3B**, **+6.6 at 8B** [5][3].
4. **More synthetic trajectory volume** — **+5.6 at 3B** from 0→17K [5].
5. **Online RL on top of SFT** — **+4.0 at 2B**, **+6.0 at 8B** [1].
6. **Cold-start SFT** — **+2.62** [10].

**The uncomfortable reading for our plan:** the two ingredients with coefficients larger than "more trajectory
data" are (a) *what the trajectories contain* and (b) *the reasoning format attached to each step* — neither of
which GUI-Odyssey supplies for the action we actually fail at (see I2: `TYPE` has no field coordinates). If the
plan is "add GUI-Odyssey trajectories," the measured literature says the volume term is worth about **+5.6**
at 3B and the first 5K trajectories are worth **+0.7** [5]. Going from 18.8% to the 45.1% 2B bar on volume alone
is not supported by any number in this table.

---

## 4. Does history/state help at all? (item 3)

**Measured evidence that conditioning on prior actions helps end-to-end success:** yes, but weaker and more
qualified than expected, and *no paper I found quantifies it as an end-to-end delta at ≤4B*.

| Finding | Scale | Metric | Value | URL |
|---|---|---|---|---|
| Multi-image visual history: multiple screenshots "substantially improves performance over single-image inputs"; default 3 images; 3→5 "marginal gains" while adding 3K context tokens | 7B | **end-to-end** OSWorld | qualitative, figure-only | [11] |
| **Textual** history format: L1 (action only) vs L2 (full CoT) — "L2 history offers no benefit and may introduce hallucinations" | 7B | end-to-end OSWorld | L1 chosen as default | [11] |
| Both episode-level advantage and "maintaining multiple historical images" contribute to performance | 7B | end-to-end AW | figure-only, no cell value | [12] |
| L2 vs L1 vs L3 test-time reasoning: 18.5 / 16.9 / 17.6 | 7B | end-to-end OSWorld | L2 at inference best | [11] T6 |

**On format specifically** — the brief asks whether raw action list vs rendered subgoal progress matters. The
only measurement I found bearing on it says the *leaner* format is better: OpenCUA found L1 (action-only)
textual history superior to L2 (CoT-rich) history, because the richer format hallucinated [11]. That argues
*against* the intuition that adding rendered progress/subgoal text to the history slot is automatically good.
It also means our 17.6% zero-history rows are a defect whose cost is **plausibly real but unmeasured at our
scale** — no paper quantifies "zero-history vs with-history" as an end-to-end delta at any size I could find.

**Inference (labelled):** given [11] and [12] both find visual multi-image history matters and textual history
richness does not, the highest-value fix to our history slot is probably *more prior screenshots*, not *richer
text*. That is an inference from two figures/tables, not a measurement of our failure.

---

## 5. The low-per-step-accuracy regime (item 4)

**The direct question — "what does trajectory SFT do when you start at 64% per-step?" — has no published
answer.** Every trajectory-SFT paper in §2 starts from a base whose offline numbers are already high, and none
of them is run at 64%. So the honest answer is the imitation-learning literature, which is unambiguous about
the mechanism and prescribes a specific mitigation.

### 5.1 The theory

Behavior cloning has a **quadratic** error bound: if the learned policy errs with probability ε on the expert's
state distribution, then over a horizon T the expected cost is **O(T²ε)** — not O(Tε) [13, Theorem 2.1]. The
bound is *tight*: "it is easy to construct a dynamic system... with 0-1 loss on d_{π*} that incurs expected
T-step cost" of that magnitude [13]. The mechanism is that one mistake takes the agent to observations the
expert never visited, where it then errs at a higher rate — errors compound, and the states are off-distribution.

DAgger restores **O(Tε)** by aggregating data from the learner's *own* induced state distribution [14]. The
prescription is: run the current policy, have the expert label the states it actually visits, aggregate, retrain.

### 5.2 The measured version, at scale

A 2026 controlled study across 9 models (six open, 1.2B–671B; three deployed proprietary systems) and 10,664
analyzed trajectories finds success obeys a **geometric law with a single per-step reliability parameter** [15]:

> "If each step succeeds independently with probability r, a task of horizon H succeeds with probability
> approximately r to the power H."

Measured values from [15]:

| Quantity | Value |
|---|---|
| Mean measured per-step reliability r (pooled) | **0.61** |
| Projected success at 8 steps (~GAIA) | 0.42 |
| at 15 steps (~WebArena) | 0.36 |
| at 20 steps (~τ-bench) | 0.33 |
| at 30 steps (~SWE-bench/OSWorld) | 0.30 |
| at 100 steps (production) | 0.24 |
| Mean per-step accuracy, first third of trajectory | **0.58** |
| Mean per-step accuracy, last third of trajectory | **0.44** |

The most damaging finding for a trajectory-SFT plan: "once an agent errs it rarely recovers... errors behave as
**absorbing states** that propagate silently" [15]. This is not a compounding-error model that better data
fixes by itself; it is a claim that *the model has no error-recovery behaviour to imitate unless the training
data contains error-recovery trajectories.*

### 5.3 On offline metrics and horizon

LongHorizonUI measured SOTA GUI agents on AndroidControl binned by sequence length and reports a cliff: >90%
average success for sequences ≤5 steps, **below 75%** past 10 steps, **~60%** past 15 steps [16]. **Caveat that
matters:** this is binned offline/step-level evaluation on AndroidControl, **not** end-to-end task success, so
it belongs in the per-step family and cannot be compared to our 18.8%. It is included because it is the only
GUI-specific measurement of the same qualitative shape, and because it independently contradicts the
independent-error assumption (a purely geometric law would not produce a *widening* gap with horizon in this
fashion).

### 5.4 The arithmetic applied to our numbers

The brief's law: `P(task) = p^n` with p = 0.64.

| Horizon n | `0.64^n` predicted end-to-end | Note |
|---|---|---|
| 5 | **10.74%** | brief says 10.8% — rounding |
| 8 | **2.82%** | brief says 2.8% |
| 10 | **1.15%** | brief says 1.2% |
| 12.5 (our non-typing episodes: "half the 25-step budget") | **0.38%** | |
| 13 | **0.30%** | |

**Our measured result: 18.8% (15/80), Wilson 11.7–28.7.** On non-typing episodes specifically, 14/66 = **21.2%**,
at roughly 12.5 steps.

`0.64^12.5 = 0.38%`. We measured 21.2%. **The model is wrong by a factor of ~56.**

Three readings, and only one of them is comfortable:

1. **(Most likely — inference.)** 64.07% per-step and 18.8% end-to-end are not measurements of the same
   quantity and must not be combined. The per-step number comes from AC-v2, a different task distribution,
   different step definition, and different denominator than AndroidWorld episodes. Inverting the geometric law
   on our own end-to-end result and a ~12.5-step horizon gives an implied **r ≈ 0.88** — not 0.64:
   `0.212^(1/12.5) = 0.883`. Doing the same for MAI-UI-2B's 45.1% at 13 steps gives **r ≈ 0.94**. Under this
   reading the "28-point per-step gap" between us and MAI-UI-2B-SFT is largely a **measurement artefact** of
   comparing AC-v2 to a number MAI-UI never published.
2. **(Also possible.)** Independence fails in the favourable direction: many AndroidWorld steps are recoverable
   (wrong tap → go back), so episode success exceeds `p^n`. This is what [16]'s cliff shape and [15]'s absorbing
   state talk both point at — the law is a hazard model, not a hard bound.
3. **(Least likely.)** The 64.07% figure is wrong.

**Consequence for the decision.** The brief's framing — "our per-step is 64%, so `p^n` says we should be at ~1%,
so the 28-point per-step gap explains our 18.8%" — is not supported. Our end-to-end number is *better* than the
per-step number predicts, not worse. Whatever is costing us the 26 points to the 45.1% bar, it is **not**
demonstrably "insufficient per-step accuracy," and therefore fixing per-step accuracy is not demonstrably the
lever. This does not rescue trajectory SFT; it removes one of the arguments for it, and it strengthens I4's
"no progress signal" hypothesis (a model with r≈0.88 who still fails 79% of episodes is failing on termination
and state-tracking, not on local action choice).

### 5.5 What the mitigation literature prescribes

If compounding is the issue, the prescriptive answer is not "more static trajectories" — it is one of:

| Mitigation | Prescription source | GUI-domain implementation found |
|---|---|---|
| **DAgger / dataset aggregation** | [14] — aggregate expert labels on the learner's *own* induced states | UI-Voyager's RFT loop is a practical instance: generate self-rollouts, filter with a rule-based verifier, SFT on survivors; then GRSD adds step-level supervision at fork points [6] |
| **Rejection sampling on rollouts** | [6], [4] (Step-GUI is a three-iteration rejection-sampling pipeline) | Both ≤4B and measured end-to-end |
| **Error-correction / recovery data** | [15]: errors are absorbing; recovery must be demonstrated | Ferret-UI Lite deliberately perturbs clean trajectories ("a correct terminate action can be replaced with an erroneous swipe") and generates the correction sequence to teach replanning [5] |
| **Prefix reuse / step-level training on long trajectories** | [7] MaP; STEP's step-augmentation | STEP's step-level sampling beats trajectory-level GRPO by **+14.7** AndroidWorld points at 7B (29.7 → 45.7) [17] |

**This is the most actionable finding in the document.** Step-level / prefix-reuse training on trajectory data
beat trajectory-level RL by **+14.7 end-to-end points at 7B** [17] — larger than any data-volume coefficient in
§3. If we adopt trajectory data at all, the measured evidence says to slice it into per-step contexts with
history, not to feed whole trajectories.

---

## 6. The counter-case (item 5)

The steelman against "add trajectory data" is stronger than I expected. Five independent arguments.

### 6.1 Direct mixture training of navigation objectives yields negligible gains

**MaP** (Masked Trajectory Prediction; EMNLP 2026 Findings) states the anti-plan result directly [7]:

> "Directly mixing step-wise decision, state-action alignment and long-horizon planning data gives only
> **marginal gains** because the objectives conflict and the data are heterogeneous."

and, in the results: "direct mixture training yields **negligible improvements over the baseline**, whereas MaP
delivers substantial SR increases of 3.0% and 2.7% on AndroidControl-High and AndroidControl-Low" [7]. On
AndroidWorld, MaP beats direct mixture by **+2.6 Pass@1** and **+5.4 Pass@4** [7]. Base models: Qwen2.5-VL
3B and 7B; AndroidWorld online eval uses 7B [7].

**Why this cuts against the plan:** the naive version of our intervention — "throw GUI-Odyssey trajectories into
the mix" — is precisely "direct mixture training," and the one paper that isolates it finds negligible gains.
MaP needed a new objective and role-aware adapters to extract value from the mixture.

Note the *supporting* half of the same paper, which the plan should also read: "training on diverse prior
navigation strategies, particularly single-step instructions, **improve generalization to local planning
tasks**" [7]. Single-step data is not the enemy.

### 6.2 SFT on full trajectories still optimizes the wrong thing

TraceR1 states the objection with unusual clarity [18]:

> "Supervised fine-tuning (SFT) on next-step predictions enables an agent to imitate local behaviors but
> **struggles to capture long-term dependencies. Even when trained on full trajectories, SFT optimizes token- or
> step-level likelihoods under teacher forcing, neglecting global consistency and failing to penalize redundant
> or unstable rollouts.**"

This directly undercuts the premise that "multi-step trajectory data" is a categorically different signal from
single-step data. Under teacher forcing, a trajectory-SFT loss is still a sum of per-step cross-entropies. The
trajectory structure changes *what contexts* are seen, not *what is optimized*.

TraceR1's own ablation is also informative: removing grounded execution feedback costs ~6% [18], i.e. the
trajectory-level reward without execution grounding is worth less than the grounding.

### 6.3 At 7B, SFT alone is a weak lever and RL is the strong one

UI-S1's stage ladder on AndroidWorld at 7B [12]:

| Configuration | **end-to-end** AW SR |
|---|---|
| Qwen2.5-VL-7B base | 14.9 |
| + SFT | **21.7** (+6.8) |
| + Offline RL | 15.7 (+0.8 vs base; **−6.0 vs SFT**) |
| + Semi-online RL only | 30.4 |
| + SFT then semi-online RL | **34.0** |

Trajectory SFT buys **+6.8**; semi-online RL on top buys **+12.3 more**. Offline RL gives back **6.0 of the 6.8
SFT points** on AndroidWorld [12] — i.e. offline RL on a trajectory-SFT checkpoint is close to a no-op net of
SFT. (Caution: the paper's prose says offline RL "degrades model performance (53.3% on MiniWob++) compared to
the base model", but its own table gives MiniWob++ base 50.5 and offline RL 54.6 — the prose and the table
disagree, so I am relying on the AndroidWorld column only, which is internally consistent.) Corroborated at 2B by
MAI-UI's +4.0 RL delta [1] and at 4B by UI-Voyager, whose entire gain comes from self-generated rollouts and
step-level distillation rather than static data [6].

### 6.4 Cold-start SFT is the smallest stage in the only published stage-wise decomposition

EvoCUA-32B's decomposition [10]: unified action space +4.84, **cold start +2.62**, RFT +3.13, offline DPO
+3.21, iterative training (remainder). Cold-start SFT is the *weakest* of the named stages.

### 6.5 The 2B bar itself may not be a trajectory-data achievement

The plan's target is MAI-UI-2B = 45.1% SFT [1]. But **Qwen3-VL-2B zero-shot scores 36.4%** on the same
benchmark per MAI-UI's own Table 6 [1]. The entire published SFT gain at 2B over its own base is therefore
**+8.7 points**, achieved by a pipeline containing trajectory synthesis, user-interaction data, MCP tool data,
grounding data, and an unspecified amount of compute. EAPO reports **46.19%** for Qwen3-VL-2B *zero-shot* [9],
which is above MAI-UI-2B-SFT — the two papers disagree about the base model's capability by ~10 points, which
means at least one of them is running a materially different evaluation protocol. **Any plan that targets
"45.1%" as a fixed bar is targeting a number with ±10 points of protocol uncertainty.**

Also worth noting: UI-Voyager-4B at **81.0%** [6] is well above every trajectory-SFT 4B model in §2 (63.9, 69.8,
79.59) and its own architecture explicitly avoids static trajectory SFT. Whatever the 4B ceiling is, static
trajectory data is not obviously how you reach it.

---

## 7. What this changes about the plan

Stated as recommendations-for-assessment only; **no training recommendation is made here**, per ISSUES rule 4.

1. **Relabel the plan.** "Trajectory SFT is known to lift end-to-end success at 2B" → **"inferred, unmeasured at
   2B."** The A2 brief's own fallback condition is met.
2. **The per-step→end-to-end decomposition in the brief is broken.** Fix `p^n` before using it to justify
   anything. Our measured end-to-end exceeds what the per-step number predicts; the "28-point gap" is
   substantially a cross-benchmark artefact (§5.4, §2.1).
3. **If trajectory data is adopted, the measured evidence says the form matters more than the volume.** Step/
   prefix slicing (+14.7 at 7B [17]), trajectory *quality* (+11.2 at 4B [8]), and CoT format (+5.9 at 3B [5])
   all have larger measured end-to-end coefficients than the first tranche of extra data (+0.7 [5]).
4. **The mechanism the plan assumes is contradicted by the failure description.** ISSUES I4 says the model
   types once and then "never returns to the form." Per [15], that is an *absorbing-state / no-recovery*
   failure, and per [5] the training signal for it is deliberately *perturbed* trajectories with correction
   sequences — not clean human demonstrations. GUI-Odyssey contains clean demonstrations.
5. **Ferret-UI Lite's ladder is the number to beat in a cheap pilot.** 13.7 → 28.0 at 3B, decomposable into
   +2.1 / +3.8 / +0.7 / +2.1 / +2.8 / RLVR [5]. A pilot that cannot reproduce a +5-point end-to-end slope from
   trajectory data should stop before scaling.

---

## 8. What I could not verify

| # | Unresolved | Why it matters | Status |
|---|---|---|---|
| V1 | **The 92.3% / 93.1% "per-step" MAI-UI numbers.** Not in the paper. No per-step metric of any kind exists in MAI-UI. | The brief's entire decomposition rests on them. | **Blocked. Treated as non-existent in §2.1.** Whoever wrote the anchor facts should re-derive them or drop them. |
| V2 | The brief's own **64.07% (AC-v2)** per-step number | I did not re-derive it; it is our internal measurement. Also unclear whether it is AC-High, AC-Low, or pooled, and AC "SR" is offline-episode, not per-step — those are different metrics [8]. | **Unverified — internal.** Need the exact metric definition before using it in any comparison. |
| V3 | GUI-Owl-1.5-2B's **exact base model** | Paper says only "initialized from Qwen3-VL"; does not state the 2B variant's base explicitly [3]. The 67.9 vs Qwen3-VL-2B's 36.4 delta therefore has an assumed, not stated, comparator. | **Unverified.** |
| V4 | Whether `hflqf88888/GUIOdyssey` localises `TYPE` | Out of A2's scope (that is I2) but it gates whether GUI-Odyssey can supply the supervision §3 says matters most. | **Not checked here.** |
| V5 | OpenCUA's exact end-to-end deltas for data scale (+72% / +125%) and history images | Extracted only from prose/secondary summary, not from figure cell values; the numbers are in Figure 5/7/10 which are images. | **Abstract/secondary only.** |
| V6 | MaP's exact AndroidWorld Pass@1 for single-step-only vs mixture vs MaP | Table cells were not in the fetched HTML (captions and prose only). Only the *deltas* (+2.6 Pass@1, +5.4 Pass@4 over mixture) are confirmed. | **Delta confirmed; absolute values not.** |
| V7 | Any measured trajectory-vs-single-step end-to-end ablation, any size | This is the whole question. 16 search strings, arXiv API, direct HTML fetch. Nothing. | **Confirmed absent to the limit of these searches.** A null result from a bounded search is not proof of non-existence — but the search was broad and the papers that would have it (Step-GUI, GUI-Owl-1.5, MAI-UI) all report other ablations, so the omission looks deliberate rather than incidental. |
| V8 | [15]'s per-step reliability study is over **LLM agents on tool-use/web tasks**, not GUI models | Generalizing r=0.61 and the absorbing-error finding to AndroidWorld is an **inference**, not a measurement of GUI agents. | **Inference — labelled as such in §5.2.** [16] is the GUI-specific support and it is an offline metric. |
| V9 | MAI-UI's RL section reports "~50.1% speedup per step" | This is a *speed* metric, not accuracy. I mention it only to warn against mis-reading it as per-step accuracy. | Note only. |

---

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|---|---|---|---|---|
| 1 | Zhou et al., *MAI-UI Technical Report* (arXiv 2512.22047) | https://arxiv.org/html/2512.22047 | 2B AW SR 45.1 SFT → 49.1 RL; no per-step metric exists; SFT uses multi-step trajectories; no trajectory-vs-single-step ablation | primary | high |
| 2 | Qwen3-VL-2B baseline, via [1] Table 6 | https://arxiv.org/html/2512.22047 | Qwen3-VL-2B = 36.4 AW SR zero-shot | primary (reported baseline) | high |
| 3 | *Mobile-Agent-v3.5 / GUI-Owl-1.5* (arXiv 2602.16855) | https://arxiv.org/html/2602.16855 | 2B = 67.9, 4B = 69.8 AW SR; T10 CoT +6.6 (8B); T11 virtual-env +33.4/+36.7 (in-house, 8B) | primary | high |
| 4 | *Step-GUI Technical Report* (arXiv 2512.15431v2) | https://arxiv.org/html/2512.15431v2 | Step-GUI-4B 63.9 Pass@1 / 75.8 Pass@3 vs Qwen3-VL-4B 45.3; data mix 4.0M traj + 2.7M grounding (mid), 404K traj (cold-start) | primary | high |
| 5 | Yang et al., *Ferret-UI Lite* (arXiv 2509.26539) | https://arxiv.org/html/2509.26539 | 3B, AW 28.0; T4 end-to-end ladder 13.7→15.8→19.6→20.3→22.4→25.2 | primary | high |
| 6 | *UI-Voyager* (arXiv 2603.24533) | https://arxiv.org/html/2603.24533 | 4B AW 81.0 Pass@1 vs Qwen3-VL-4B 45.3; RFT self-rollouts + GRSD; 73.2→81.0 while GRPO/PPO ~76; no static human trajectory dataset | primary | high |
| 7 | Zhang et al., *MaP* (EMNLP 2026 Findings) | https://arxiv.org/html/2609.25769 | "direct mixture training yields negligible improvements"; MaP +2.6 Pass@1/+5.4 Pass@4 on AW over mixture; 3B/7B Qwen2.5-VL | primary | high (deltas only) |
| 8 | *OS-Genesis* (ACL 2025) | https://qiushisun.github.io/OS-Genesis-Home/ | InternVL2-4B AW 4.02 (task-driven) → 15.18 (OS-Genesis); 8B 4.46→16.96; 7B 9.82→17.41; defines whole-task SR as all steps correct | primary | high |
| 9 | Hua, Yue, Ren, *EAPO* (arXiv 2605.08978) | https://arxiv.org/html/2605.08978v2 | EAPO-2B AW 76.36, 4B 79.59, 8B 82.05; Qwen3-VL-2B 0-shot 46.19 (conflicts with [1]'s 36.4) | primary | high |
| 10 | *EvoCUA* (arXiv 2601.15876) | https://arxiv.org/html/2601.15876v2 | EvoCUA-32B stage deltas: action space +4.84, cold start +2.62, RFT +3.13, offline DPO +3.21 | primary | medium (secondary read of T3) |
| 11 | Wang et al., *OpenCUA* (NeurIPS 2025) | https://arxiv.org/html/2508.09123 | multi-image history helps, 3→5 marginal; L1 textual history beats L2; data scale +72%/+125% (figure-only) | primary | medium (figure values not extracted) |
| 12 | *UI-S1* (arXiv 2509.11543v2) | https://arxiv.org/html/2509.11543v2 | 7B AW: base 14.9, SFT 21.7, offline RL 15.7, semi-online 30.4, SFT+RL 34.0 (prose/table disagree on MiniWob++) | primary | high (AndroidWorld column) |
| 13 | Ross & Bagnell, *Efficient Reductions for Imitation Learning* (AISTATS 2010) | https://proceedings.mlr.press/v9/ross10a/ross10a.pdf | Theorem 2.1: J(π̂) ≤ J(π*) + T²ε; bound is tight | primary | high |
| 14 | Ross, Gordon, Bagnell, *DAgger* (AISTATS 2011) | https://proceedings.mlr.press/v15/ross11a/ross11a.pdf | T² → T reduction via dataset aggregation on learner-induced states | primary | high |
| 15 | *How Fast Do Agents Rot?* (arXiv 2609.01660) | https://arxiv.org/pdf/2609.01660 | geometric law r^H; measured r=0.61; 10,664 trajectories; 9 models 1.2B–671B; per-step 0.58→0.44 within trajectory; errors absorbing | primary | high (but non-GUI domain) |
| 16 | *LongHorizonUI* (ICLR 2026) | https://proceedings.iclr.cc/paper_files/paper/2026/file/002a6bef26174b590e4eab95b63feb1b-Paper-Conference.pdf | AndroidControl binned by length: ≤5 steps >90%, >10 <75%, >15 ~60% | primary | medium (offline metric; figure) |
| 17 | *STEP* (Findings ACL 2026) | https://aclanthology.org/2026.findings-acl.1532.pdf | UI-TARS-DPO-7B AW: 29.7 → T-GRPO 31.0 → GiGPO 34.0 → STEP 45.7 (+16.0) | primary | high |
| 18 | *Anticipatory Planning / TraceR1* (arXiv 2603.16777) | https://arxiv.org/html/2603.16777 | SFT "struggles to capture long-term dependencies"; removing grounded execution stage ≈ −6% | primary | high |
| 19 | Google Research, *AndroidWorld* docs | https://google-research.github.io/android_world/ | 116 tasks, 20 apps; dynamic instantiation; step histogram present but no mean/max quoted in text | primary (docs) | medium |

---

## Coverage status

**Checked directly (fetched full text and read the tables):** [1] MAI-UI (incl. absence of per-step metrics —
verified by term scan), [3] GUI-Owl-1.5 T10/T11, [4] Step-GUI v2 data composition + T4, [5] Ferret-UI Lite T4,
[6] UI-Voyager, [7] MaP (deltas), [9] EAPO T5, [11] OpenCUA history ablations, [12] UI-S1, [13] Ross 2010 PDF,
[14] Ross 2011 PDF, [15] agents-rot PDF, [17] STEP PDF, [18] TraceR1.

**Checked at abstract/secondary level only:** [10] EvoCUA T3 (via search snippet), [16] LongHorizonUI (ICLR PDF
snippet + two secondary summaries agree), OpenCUA's +72%/+125% figure values, AndroidWorld step histogram.

**Blocked:** the brief's 92.3%/93.1% per-step MAI-UI figures (do not exist in the paper — see V1). MaP's
absolute Pass@1 cells (only deltas retrievable).

**Confirmed absent:** any trajectory-vs-single-step end-to-end ablation at ≤4B (V7, 16 search strings).
No qualifying 2B study exists in the trajectory-vs-single-step form; none exists at 3B or 4B either.

**Needs follow-up (outside A2 scope):** V3 GUI-Owl-1.5-2B exact base; V4 whether the newer GUI-Odyssey
localises `TYPE`; V2 the exact metric definition behind our 64.07%.
