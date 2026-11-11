# Beyond single-step SFT: what raises AndroidWorld success for small GUI agents (≤ ~8B)

*Survey written 2026-11-11. Every number below was read from the primary arXiv page (abstract or HTML full
text) on that date, with the table or figure named. "not reported" means the source was checked and
does not give the number. **INFERRED** marks our own reasoning; no inferred number is cited as a paper's.
Gains are copied from the source table; where we subtracted two of the paper's own numbers to get a gain, that is
written as "(Δ computed from Table X)".*

## 0. Read this first: three caveats that apply to every row

1. **Harnesses differ (M4).** The papers report AndroidWorld (AW) as Pass@1 over 1 run, a 3-run mean (MobileRL,
   OpenMobile, GHD), a 64-seed mean (UI-Voyager) or **Pass@3** (Step-GUI's headline 75.8 / 80.2; its Pass@1 is
   63.9 / 67.7). Prompts, step budgets and emulator images also differ. Use deltas *within* a paper; never
   rank our number against these.
2. **Many strong small-model results train on the AW apps or on AW's own task generator.** MobileRL's RL uses
   "2,000 AndroidWorld tasks"; UI-Voyager uses ">7,000 tasks from AndroidWorld's training sets"; OpenMobile
   synthesises its data in the AW emulator ("shared environment and app suite"; 3.5% of its instructions have
   cosine > 0.7 to an AW test instruction). Our training data has no goals for 13 of the 20 AW apps. That gap
   alone is a big part of the distance to 49.1 (**INFERRED**). Any plan that adopts it must disclose it.
3. **Nearly every listed system generates text** (a `<think>` block, a ReAct thought, or the action as tokens).
   Only GUI-Actor (grounding head), V-Droid (candidate scoring by first-token probability) and Digi-Q (a Q head on
   frozen features) are structurally close to our single-pass, zero-token head. "Fit" below says what each method
   needs from us.

**Fit legend:** **Y** = works with our classification + readout heads as they are · **Y-env** = fits the head
but needs emulator rollouts (single emulator is enough unless noted) · **Gen** = needs a generative head
(reasoning text or free-text output) · **Ext** = needs an external model at inference (forbidden for us; allowed
only at training time if the operator agrees).

---

## 1. Data

### 1.1 Public trajectory corpora (sizes from the dataset papers)

| Corpus | Size (as reported) | Notes for us |
|---|---|---|
| AndroidControl (2406.03679) | 15,283 demos, 14,548 unique tasks, 833 apps; high+low-level instructions | We use ~15k *single-step* decisions out of it. Abstract: OOD high-level performance "scales much more slowly"; "fine-tuning on more data alone may be insufficient". |
| GUI-Odyssey (2406.08451) | 8,334 cross-app episodes, 212 apps, 6 devices | Multi-step, cross-app. (Count from our 2026-11-08 read of the paper.) |
| AMEX (2407.17490) | ~104K screenshots; 3 annotation levels (element grounding, screen/element function, instruction + action chain) | Instruction/trajectory counts not given in the abstract. |
| AITW (2307.10088) | ~715K episodes, ~30K unique instructions, Android 10–13, 8 devices | Old, noisy; most later work uses subsets. |
| AITZ (2403.02713) | 18,643 screen-action pairs with chain-of-action-thought | Abstract: a 1B AUTO-UI fine-tuned on it is "on-par" with CogAgent-18B. |
| AndroidInstruct (AndroidLab, 2410.24024) | 10.5k traces / 94.3k steps collected; 726 traces / 6,208 steps used | Table 1: Qwen2-VL-7B SoM 3.62 → 18.12 SR on **AndroidLab** (not AW). |
| OpenMobile (2604.15093) | ~2.8K tasks, ~34K steps, 20 apps (AW emulator), mean length 12.2 steps, ~129 words of CoT per step | **Release of data and code is stated on the project page; we have not yet confirmed the download.** |

### 1.2 Synthesis and self-evolving data: measured gains

| Method | Model | Measured AW effect (source) | Cost | Fit |
|---|---|---|---|---|
| **OS-Genesis** reverse task synthesis (2412.19723) | InternVL2-4B / 8B, Qwen2-VL-7B | Table 1, AW SR: task-driven → OS-Genesis **4.02 → 15.18** (4B), 4.46 → 16.96 (8B), 6.25 → 17.41 (7B); 1K trajectories each | GPT-4o labels the triples and scores the trajectories | Y as data (its CoT can be dropped) |
| **OpenMobile** decoupled task synthesis + policy-switching rollouts (2604.15093) | Qwen2.5-VL-7B, Qwen3-VL-8B | Table 1: 7B **25.5 → 51.7 ± 1.7**; 8B **47.6 → 64.7 ± 3.2** (SFT only, 3 epochs, lr 1e-5). Table 5b (same 8B, other recipes' data): OS-Genesis 17.4 (1.5K traj), HATS 24.4 (1K), AutoPlay 40.1 (20K), MobileGen 45.7 (0.5K), OpenMobile 64.7 (2.8K). Table 2b at 1.5K trajectories: OS-Genesis 34.1, coupled pipeline 45.3, OpenMobile 48.3 | Gemini-3.1-Pro acts as expert, monitor and instruction writer; GPU hours not reported | Y as data; Ext at *training* time |
| **Ferret-UI Lite** synthetic navigation data (2509.26539) | 3B (in-house) | Table 4, AW: baseline 13.7 → +long CoT 19.6 → +synthetic 5K 20.3 / 13K 22.4 / **17K 25.2**; RL then 25 → 28.0 (Table 3a, 5-run mean) | GPT-4o CoT; multi-agent online rollouts with a VLM judge | data: Y; CoT part: Gen |
| **UI-Voyager** rejection fine-tuning (RFT) on own rollouts (2603.24533) | Qwen3-VL-4B | Pass@1 **37% → 73% over three RFT rounds** (73.2% at round 3); one RFT round = 64.0%, which GRPO/PPO from the base need ~175 steps to match. Final 81.0% after GRSD (§2) | >7,000 AW-train tasks, rule-based verifier through adb, online rollouts | **Y-env** (imitating successful trajectories needs only log-likelihood) |
| **Step-GUI** CSRS data flywheel (2512.15431) | Qwen3-VL 4B / 8B | Fig. 7: AW **44.83% (round 2) → 73.40% (round 3)**; per-model rows are Pass@3 | 11.2M mid-train + 1.67M cold-start samples; verifiers/humans | Y-env (CoT is used in its data) |
| **MAI-UI** self-evolving pipeline (2512.22047) | Qwen3-VL 2B … 235B | **2B-SFT = 45.1** (Table 9). No per-component AW ablation of the pipeline is reported ("rejection-sampled data helps close the gap between pass@1 and pass@N" — qualitative) | Manuals + expert tasks + human emulator annotation + multi-agent rollouts + MLLM judge; failed trajectories kept as their longest correct prefix | Y-env |
| **GUI-Owl / Mobile-Agent-v3** self-evolving trajectories (2508.15144) | Qwen2.5-VL-7B | GUI-Owl-7B **66.4** (Table 7). Fig. 10 shows gains from iterative trajectory rounds; **exact numbers not reported in text** | Step- and trajectory-level critics, guidance generation | Gen (the model reasons; only its "conclusion" goes into history) |
| **UI-Venus-1.5** mid-training (2602.09082) | Qwen3-VL 2B / 8B | Table 7 (AW): mid-training stage alone **2B 39.0**, 8B 57.0. No "without mid-training" row | 10B-token mid-training corpus from >30 sources | Y as data, but the cost is high |
| Hindsight relabeling / backward construction (Learn-by-interact, 2501.10893) | Codestral-22B etc. | AW **not reported** (SWE-bench, WebArena, OSWorld, Spider2-V only) | — | Y as data |

**Takeaway (INFERRED from the rows above):** at small scale the largest *SFT-only* jumps on AW come from
multi-step trajectories **in the AW apps**, not from more single-step demos. OpenMobile 7B +26.2 and OS-Genesis
4B +11.2 are both Δs computed from their own Table 1. Data quality matters more than volume: MobileGen's 0.5K
trajectories (45.7) beat AutoPlay's 20K (40.1) on the same 8B.

---

## 2. Learning beyond SFT

### 2.1 Stage ablations that isolate the post-SFT gain on AW

| Paper (model) | SFT / start | After the method | Method | Fit |
|---|---|---|---|---|
| **MobileRL** (2509.18119), Qwen2.5-VL-7B | base 27.6 → **reasoning-free SFT 50.2** → reasoning SFT 56.8 | **AdaGRPO 72.0** (Fig. 3a) | Online GRPO + AdaPR replay, FCF curriculum, SPA shortest-path reward; 2,000 AW tasks; up to 256 parallel rollouts | Y-env (many envs); reasoning-SFT step = Gen |
| MobileRL, GLM-4.1V-9B | base 7.7 → RF-SFT 48.1 → R-SFT 66.2 | 80.2 | same | same |
| MobileRL ablation (Fig. 3c, 3 runs) | w/o AdaGRPO 56.8 | full 71.1; w/o AdaPR **63.6**; w/o SPA 69.1; w/o AdaPR & SPA 58.5; w/o FCF 64.8 | → **positive replay is the largest component** | — |
| **MAI-UI** (Table 9) | 2B-SFT **45.1** | 2B-RL **49.1 (+4.0)**; 8B 64.7 → 70.7 (+6.0); plain GRPO on 8B 66.5 (+1.8) | GRPO + curriculum + repetition penalty + replay (combined; no per-component numbers); max_env_steps 15/30/50 → 66.4/68.5/70.7; **32 → 512 parallel envs: 65.5 → 70.7** (Fig. 12b, text) | Y-env, **datacenter scale** |
| **UI-Venus-1.5** (Table 7) | 2B mid-train 39.0 → **offline RL 45.3** | **online RL 59.8** → TIES merge 55.6 | GRPO, trajectory-level advantage, length-decayed success reward | Gen (format reward requires `<think>`); Y-env |
| **UI-Voyager** (2603.24533), 4B | RFT round 3: 73.2 | **GRSD 81.0** vs GRPO/PPO plateau ~76 | GRSD = find the step where a failed rollout diverges from a successful one (SSIM on screenshots), then step-level SFT on the successful action there | **Y-env** (it is SFT, so likelihood-only) |
| **UI-S1** semi-online RL (2509.11543), Qwen2.5-VL-7B | base* 14.9 / SFT 21.7 | **semi-online RL only 30.4; UI-S1 34.0** (Table 2); offline RL 15.7 | Rollouts **on offline trajectories**; a "patch module" injects the expert action on mismatch (thought-free patch, ε=1); discounted future returns | **Y, no emulator** (final config injects an empty thought) |
| **GHD** gated hindsight distillation (2608.06065) | Qwen2.5-VL-7B SFT 46.55; Qwen3-VL-8B SFT 59.05 | GRPO 47.13 / 61.35 → **GHD 52.73 / 66.47** (Table 3, 3 runs). Ablation (7B, Table 2): gated teacher *without* the next screenshot 47.84 / 49.56; with it 52.73 | Same-weight teacher additionally sees the **next screenshot**; distil only on steps where the student fails and the teacher recovers the demonstrated action; trained on OpenMobile trajectories | **Y (INFERRED)**: privileged-input distillation is a KL between our heads; the paper does not clearly say whether rollouts touch an emulator |
| **OpenMobile** RL (Table 4) | 8B SFT 64.7 ± 3.2 | 8B-RL 64.1 ± 0.5 | trajectory-level RL on 244 instructions | **no gain** |
| **Ferret-UI Lite** RL (3B) | SFT 25 | 28.0 | RL with designed rewards; action-type-only reward "causes a substantial drop"; dense > sparse grounding reward (Fig. 7b, no numbers) | Y-env |
| **ZeroGUI** (2505.23762), UI-TARS-7B-DPO | AndroidLab 45.7 | 47.5 (+1.8) (Table 2) | GPT-4o task generation + Qwen2.5-VL-32B judge (precision 61.5 / recall 51.1). **AW not reported** | Y-env + Ext judge |
| **DigiRL** (2406.11896), 1.3B | AitW SFT 17.7 | 67.2 | offline → online RL. **AW not reported** | Y-env |
| **Digi-Q** (2502.15760), LLaVA-1.5-7B features | AitW General test: Filtered BC 54.5, DigiRL-offline 59.0 | **Digi-Q 71.2** (DigiRL online 74.5) (Table 1) | TD-trained Q head on frozen VLM features (~1% of params), Best-of-N over policy samples (N up to 16, monotone). **AW not reported** | **Y**: a Q/value head on our features; candidates from our own top-K |
| DistRL (2410.14803) | — | "20% relative" success gain, 3x training efficiency (abstract) | async distributed online RL. **AW not reported** | Y-env |
| VEM (2502.18906) | — | AitW only; no numbers in abstract | offline value environment model | Y |
| UI-R1 / GUI-R1 / InfiGUI-R1 (2503.21620, 2504.10458, 2504.14239) | — | **AW not reported** (AndroidControl / ScreenSpot only) | rule-based GRPO with reasoning | Gen |

### 2.2 Reward / verifier models

- **V-Droid** (2503.15937), the closest analogue to our design. A Llama-3.1-8B verifier plus an MLP head scores
  every candidate action that a rule-based a11y-tree extractor produces. The score comes from the first token
  (Yes/No) in a batched prefill, so it **generates no tokens to choose an action**. AW **59.5%**.
  - Table 3, same 55K pairs: pairwise progress-preference loss **47.4** vs selector-SFT **35.8** vs
    generator-SFT 27.4.
  - Data scaling, AW SR: 9K pairs **15.0**, 27K 37.5, 55K 47.4, 110K **59.5**.
  - Working memory: LLM-written 59.5, **rule-based 46.1**, action history only 40.0.
  - Caveats: text-only; **GPT-4 writes the working memory**; an unnamed LLM writes the text for
    open-app/type/answer (~12.4% of actions). The deployed system is therefore **Ext**. The *training objective* (pairwise
    preference over candidates) is Y for us.
- **UI-Genie-RM** (2505.21496): step-reward F1 79.6, outcome F1 82.1 (Table 5); agent 7B AndroidLab 18.1 → 38.7
  over 3 rounds. **AW not reported.**
- **MLLM judges** are noisy: ZeroGUI's 32B judge has precision 61.5 / recall 51.1. MAI-UI reports 83% agreement with humans for
  its judge. **INFERRED:** for AW-template tasks the programmatic checker is the better reward and needs no judge.

### 2.3 What works without generating reasoning text

- **Measured without reasoning:** MobileRL **reasoning-free SFT** = 50.2 (7B) / 48.1 (9B) on 97.9k steps (62.2k human +
  21.3k AC-Low + 14.3k AC-High; Table 7). V-Droid (candidate scoring). UI-S1's final patch is thought-free. GUI-Actor.
  Digi-Q.
- **Reasoning adds measurable AW points at small scale:** MobileRL reasoning SFT +6.6 (7B) / +18.1 (9B); Ferret-UI
  Lite long CoT +5.9 (Δ computed from Table 4, 13.7 → 19.6; the text says 4.1); GHD 8B reasoning teacher 60.34 vs
  action-only teacher 58.62 (Table 4). **We cannot take these gains without a generative head.**

---

## 3. Grounding

| Technique | Evidence (exact) | Fit |
|---|---|---|
| **Patch-attention action head** (GUI-Actor, 2506.03143) | A single forward pass; target = uniform over patches covered by the gold bbox, KL loss. Same data recipe: Aguvis-7B point-generation **15.6** vs GUI-Actor-7B **40.7** on ScreenSpot-Pro; SS-v2 87.0 vs 89.5. **GUI-Actor-2B 36.7 / 88.6** (SS-Pro / SS-v2); +verifier 41.8 / 89.3. Frozen backbone, 19M trained params (2B): SS-v2 73.9, with verifier 82.3. **AW not reported** | **Y**: a drop-in alternative to our 4 × per-edge CE box head |
| **Dense Gaussian target vs sparse** (GUI-G², 2507.15846, 7B, RL reward) | SS-v2: sparse point 87.4 / sparse IoU 85.8 / **dense Gaussian 93.3** (Table 2); a Gaussian clipped to zero outside the box 88.4; fixed σ 87.8 vs adaptive σ 93.3. Authors: binary rewards give a "cliff" where "gradient information vanishes" | **Y (INFERRED)**: matches our finding that per-edge CE stops giving gradient beyond ~200 bins (`benchmarks/results/2026-11-07-why-grounding-plateaus.md`). Element-size-adaptive σ is the transferable idea |
| **Zoom-in second pass** (MAI-UI §2.2.2) | Crop half-width/half-height around the first prediction and re-predict. **2B: SS-Pro 57.4 → 62.8; OSWorld-G 52.0 → 55.9; UI-Vision 30.3 → 31.9** (Tables 2, 5, 3). AW effect not reported | **Y**: two zero-token passes, ~2x latency. **INFERRED:** smaller gain on phone screens than on desktop benchmarks |
| Grounding verifier (GUI-Actor) | SS-Pro +5.1 (2B), +8.6 for the frozen-backbone 2B | Y (a second head) |
| **Clean grounding data** | UI-Venus: ~40% of open-source grounding data noisy; 627k → 107k after cleaning (Table 1). MAI-UI: 23.3% of instructions have quality issues. Ferret-UI Lite (Table 6, SS-Pro): all data 50.15; w/o OS-Atlas 43.35; w/o Aria-UI 46.93; w/o UGround 48.07 | Y |
| Large grounding corpora | UGround: 10M elements / 1.3M screenshots. GUI-Actor training: ~1M screenshots / 9.6M elements. AW agent results for UGround/Aria-UI use an external planner → not comparable to us | Y as data |
| Resolution | **No AW-specific resolution ablation was found.** Phi-Ground argues image-token count belongs on the cost axis but gives no clean mobile ablation; Ferret-UI Lite uses AnyRes tiling with no ablation | — |

---

## 4. Memory and history

| Evidence | Numbers | Fit |
|---|---|---|
| Past **screenshots** vs actions only (HiconAgent, 2512.01763, Table 6, Qwen2.5-VL-3B, AndroidControl, not AW) | 2 actions only: SR **43.33**, grounding 59.61; + 2 past observations: SR **52.29**, grounding 63.88; FLOPs 13.21 → 35.75 T | Y (extra image input; ~2.7x compute) |
| Working memory (V-Droid, AW) | LLM memory 59.5 / rule-based 46.1 / action history only 40.0 | rule-based: Y; LLM: Ext |
| Next screenshot as privileged training signal (GHD) | 7B: +3.2 over the matched gated teacher without the next screenshot (52.73 vs 49.56, Table 2) | Y (training only) |
| Action-effect verification (VeriGUI, 2604.05477, ACL 2026) | Robust SFT on synthetic failure trajectories + GRPO with verification rewards. Cites a separate study: repeated ineffective actions caused timeouts in 72.3% of failures. **AW not reported**; sizes not in abstract | Gen (TVAE text) |
| Loop breaker (VLAA-GUI, 2604.21375, desktop, frontier backbones) | Gemini 3 Flash: +1.82 at 100 steps, **+4.24 at 50 steps**; wasted steps 4.9% → 2.8%. Sonnet 4.6: +0.04. **AW not reported** | Y as a test-time rule; transfer to a 2B agent unmeasured |

---

## 5. Termination and `answer`

- **No paper found that measures a zero-token `answer` mechanism.** The handling seen in the sources:
  - **MAI-UI:** `answer` is a separate action with free text (Table 1); no retrieval-specific numbers.
  - **MobileRL:** `Finish(optional message)` and no separate answer action (Table 5).
  - **Ferret-UI Lite:** `terminate` with a reason, trained on **synthetic QA**: episode goals rewritten as questions,
    with answers grounded in the final screen.
  - **V-Droid:** "answer" is a default candidate, but an LLM writes the text.
- **Premature / false completion is the dominant failure even for frontier agents.** VLAA-GUI (desktop): with
  its completeness planner, false DONE is still >86% of failed tasks (Sonnet 4.6: 95.5% → 91.9%); removing the
  planner costs −2.86 (Sonnet) and −1.43 (Gemini Flash) at 100 steps but **helps** +2.86 at 50 steps. The effect
  is not uniformly positive.
- **Shortest-path reward (SPA):** −2.0 when removed (MobileRL Fig. 3c: 71.1 vs 69.1).
- **Trajectory-level training teaches when to stop (INFERRED):** RFT and online RL reward the full episode, so a
  premature `status` gets negative signal, which single-step SFT never gives. Our AndroidControl `status` rows are
  always the final row and always `successful`, so SFT never sees a negative example.
- **For us (INFERRED):** a pointer-copy `answer` readout over on-screen text spans (a11y text) would fit the
  zero-token head, but it cannot produce computed answers (counts, sums). We have not checked how many of the 25
  answer tasks need computation.

## 6. Test-time methods that keep one model

| Method | Evidence | Fit |
|---|---|---|
| Best-of-N with a learned Q/value head | Digi-Q: monotone in N ∈ {1, 4, 8, 16}; 71.2 AitW General (AW not reported) | Y (we already expose top-K per head) |
| Candidate verifier over rule-extracted actions | V-Droid as above | Y for the scoring; the memory/text parts are Ext |
| Loop / repetition breaking | VLAA-GUI loop breaker (above). MAI-UI uses a repetition penalty only as an RL reward (no isolated number) | Y |
| Zoom-in re-prediction | MAI-UI 2B SS-Pro +5.4 (Δ computed from Table 2) | Y |
| Model merging (weights, not inference ensembling) | UI-Venus-1.5 TIES merge: **2B 59.8 → 55.6 (hurts)**, 8B 72.7 → 73.7, 30B 75.5 → 77.6 (Table 7) | Y, but negative at 2B |
| Sampling + external judge | GTA1 (K=8 proposals, MLLM judge; desktop, o3/GPT-5 planner); no AW | Ext |

---

## 7. Ranked: the six most promising for our single-pass, zero-token, one-model agent

1. **Multi-step trajectory SFT in the AW app suite (OpenMobile's released data, then GUI-Odyssey/AMEX
   trajectories, history rendered as in training).** Largest SFT-only AW deltas at our scale (OpenMobile 7B
   25.5 → 51.7; OS-Genesis 4B +11.2; Ferret-Lite 3B +5.6 from 17K synthetic on top of long CoT, 19.6 → 25.2 —
   all Δs computed from the papers' own tables). No head change; we must drop its
   CoT, and the CoT-free effect is **unmeasured**.
2. **Rejection fine-tuning on our own successful rollouts from AW-template tasks with the programmatic checkers**
   (UI-Voyager 4B 37 → 73 in three rounds; one round 64.0 ≈ 175 GRPO steps). Likelihood-only, so it fits the
   heads, and it gives the first trajectory-level termination signal. It needs emulator hours, not 512 envs.
   Disclose the in-distribution task templates.
3. **Recovery data via error-intervention / DAgger-style switching** (OpenMobile Table 3 at matched 1.5K:
   expert-only 44.8 → error-intervention 48.3; self-evolution alone 33.8). It targets our no-effect/repeat loops.
   It needs an expert at **training** time only, which is an operator decision.
4. **Emulator-free post-SFT on offline trajectories: semi-online RL (UI-S1: SFT 21.7 → 34.0, 7B) or
   next-screenshot gated hindsight distillation (GHD: +5.6 over GRPO at 7B, +5.1 at 8B).** Both reduce to
   gradients on our head log-probs and need no emulator fleet. GUI-Actor's grounding head (pick 6) is the
   other main option with no emulator.
5. **Past-screenshot history plus a rule-based "screen changed?" signal** (HiconAgent 3B AC-High SR +8.96 for
   2.7x FLOPs; V-Droid rule-based memory 46.1 vs action-history-only 40.0). This attacks the measured
   no-effect/repeat failure classes directly. It is cheap at a 720 px input.
6. **A grounding head with a dense, size-adaptive target** (GUI-Actor patch attention: 2B SS-v2 88.6, and
   +25.1 SS-Pro over point generation at 7B on the same data; GUI-G² dense vs sparse +5.9 SS-v2), plus the
   optional zoom pass (MAI-UI 2B +5.4 SS-Pro). It fixes the zero-gradient regime we measured in the edge-CE head.
   AW transfer is **not reported** anywhere.

**Deprioritised, with reason:**
- **Small-scale online GRPO:** MAI-UI 8B gains only +0.8 with 32 parallel envs (65.5 vs 64.7 SFT) and +1.8 with
  plain GRPO; OpenMobile RL gave −0.6.
- **Model merging at 2B:** −4.2 (UI-Venus-1.5).
- **CoT:** real gains, but it needs a generative head.
- **External judges or planners:** forbidden at inference.
- **V-Droid's full system:** GPT-4 memory. Its pairwise-preference objective (47.4 vs SFT-selector 35.8 at 55K
  pairs) is worth trying as a loss if we score candidates.

## Sources (all retrieved 2026-11-11 unless noted)

- MAI-UI — arXiv:2512.22047 (Table 9, Fig. 12b, Tables 2/3/5; verbatim capture `references/raw/maiui2026-androidworld-table9.txt`, 2026-11-08)
- MobileRL — arXiv:2509.18119 (Fig. 3a/3c, Table 1, Table 5, Table 7)
- UI-Venus-1.5 — arXiv:2602.09082 (Table 2, Table 7); UI-Venus — arXiv:2508.10833 (Table 10, Table 1)
- GUI-Owl / Mobile-Agent-v3 — arXiv:2508.15144 (Table 7, Fig. 8, Fig. 10)
- Step-GUI — arXiv:2512.15431 (Table 4, Fig. 7)
- UI-Voyager — arXiv:2603.24533
- OpenMobile — arXiv:2604.15093 (Tables 1, 2b, 3, 4, 5b)
- GHD — arXiv:2608.06065 (Tables 2, 3, 4)
- UI-S1 — arXiv:2509.11543 (Table 2, Table 9)
- Ferret-UI Lite — arXiv:2509.26539 (Table 3a, Table 4, Table 6, Fig. 7)
- OS-Genesis — arXiv:2412.19723 (Table 1)
- V-Droid — arXiv:2503.15937 (Table 3, data-scaling and memory ablations)
- AndroidLab — arXiv:2410.24024 (Table 1); ZeroGUI — arXiv:2505.23762 (Tables 1–3)
- DigiRL — arXiv:2406.11896; Digi-Q — arXiv:2502.15760 (Table 1, Table 2, Fig. 3); DistRL — arXiv:2410.14803; VEM — arXiv:2502.18906
- UI-Genie — arXiv:2505.21496 (Tables 2, 3, 5)
- GUI-Actor — arXiv:2506.03143 (Tables 4, 5, ablation); GUI-G² — arXiv:2507.15846 (Table 2, §4.4); UGround — arXiv:2410.05243; Aria-UI — arXiv:2412.16256; GTA1 — arXiv:2507.05791; Phi-Ground — arXiv:2507.23779
- HiconAgent — arXiv:2512.01763 (Table 6); VeriGUI — arXiv:2604.05477; VLAA-GUI — arXiv:2604.21375
- Corpora: AndroidControl arXiv:2406.03679; GUI-Odyssey arXiv:2406.08451; AMEX arXiv:2407.17490; AITW arXiv:2307.10088; AITZ arXiv:2403.02713
- UI-R1 arXiv:2503.21620; GUI-R1 arXiv:2504.10458; InfiGUI-R1 arXiv:2504.14239 (no AW numbers found)
- Not verified here: InfiGUIAgent-2B = 9.0 on AW (from the brief; the arXiv:2501.04575 abstract gives no AW number).
