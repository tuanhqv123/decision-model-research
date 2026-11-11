# Data recipes B — how mobile/GUI trajectory data is synthesised or collected (primary-source extraction)

*Written 2026-11-11. Each paper was read in full from its arXiv HTML. The version is named per paper. Release and
licence facts come from the linked GitHub / Hugging Face pages, never from blogs or dataset-card prose.
Conventions: "§/Tab./Fig." are the paper's own numbers. **not reported** = the paper was checked and is silent.
**INFERRED** = our inference. **(fig.)** = a value read off a plotted figure (±~0.5 pt). **(repo)** = a fact taken
from the authors' released code, not the paper.*

*Re-checked by hand before writing:*
- OpenMobile's `AndroidWorld/random_walk_aw.py` iterates the AW registry ("All AW tasks, total 116"). For each task it
  calls `random.seed(...)`, `generate_random_params()` and `initialize_task(env)`, with default `task_random_seed`
  **222**.
- UI-Voyager §4.1, verbatim: "we follow MobileRL and employ training sets from AndroidWorld, consisting over 7000
  tasks".
- AndroidGen Tab. 6 and Tab. 8, every row.
- V-Droid Tab. 3 and Tab. 4, and its negative-pair rule.

**M4 applies to every number below.** Harnesses, base models, run counts and step budgets all differ. Compare deltas
only *within* one paper, never a number against another paper's number, and never against ours.

---

## 0. The answer in one table

The "AW apps in training?" column matters most for us. **No paper in this set reports a model trained without
AndroidWorld's apps that was then tested on AndroidWorld.**

| Recipe | How goals are made | Who acts | Success check | Scale (as reported) | AW effect (within-paper Δ) | AW apps / templates in training? | Released |
|---|---|---|---|---|---|---|---|
| **OpenMobile** 2604.15093 | Random walk → screen-function memory → **Gemini-3.1-Pro** writes tasks | Own early checkpoint + Gemini expert (error-intervention switching) | **Self-declared** completion (expert says done) | 2.8K traj, 34K steps, 20 apps | 7B 25.5→**51.7**; 8B 47.6→**64.7** (Tab. 1) | **Yes**: AW apps, and states from AW's own `initialize_task` for all 116 classes (seed 222, repo) | Code Apache-2.0; data on HF (no licence) |
| **UI-Voyager** 2603.24533 | **AW templates with re-sampled parameters** (>7,000 tasks) | Qwen3-VL (round 1), then own policy | **Programmatic** (adb state checks) | not reported | 4B ≈37→73.2 (3 RFT rounds) → **81.0** (GRSD) | **Yes, templates** | Code MIT; data not released |
| **MobileGen** 2601.22781 | Difficulty-controlled exploration → reverse synthesis | **Qwen3-VL-8B** explorer + supervisor (rollback) | None reported (relabelled, so success holds by construction) | 500 traj, ≈20 steps each | 8B 31.0→**45.7** (M3A harness, Tab. 1) | Likely AW's 20 apps (INFERRED) | Nothing released |
| **AutoPlay** 2509.25047 | MLLM explores, then guideline prompts → parameterised tasks | GPT-4o planner + UI-TARS-1.5-7B grounder | **GPT-4o judge** | 20K tasks → **≈8K traj** (≈40%) | 7B 19.5→**40.1** (P@1, 5 trials) | **Yes**: the 20 AW apps | Nothing released |
| **OS-Genesis** 2412.19723 | Rule-based walk → GPT-4o turns (s, a, s′) into low- then high-level goals | GPT-4o | GPT-4o reward model, 1–5; weights sampling, discards nothing | 1K traj, 6.4 steps | 4B 0.0→**15.18**; 7B 0.89→17.41 (Tab. 1) | Yes, AW apps (INFERRED: "20 of 833 AC apps seen") | Data MIT (HF card); repo has no LICENSE file |
| **AndroidGen** 2504.19298 | **GPT-4o writes ≈300 goals "drawing on the instructions in AndroidWorld"** | AndroidGen framework + GPT-4o | **GPT-4o StepCritic** (sub-goals) | >1K traj (×2.52 from prefix augmentation) | Llama-3-70B 10.8→25.2 (→35.3 with retrieval at inference) | **Yes**, AW-instruction-derived | Apache-2.0 |
| **GUI-explorer** 2505.16827 | MLLM goals anchored on Manifest activities | GPT-4o, DFS b=10 d=5 | none (transitions only) | 1,300 knowledge items, 46 apps | 47.4 (GPT-4o, **training-free**) | Explores the AW apps | MIT |
| **Explorer** 2502.11357 | Propose → refine → summarise (hindsight) | GPT-4o | GPT-4o verifier (81% agreement with humans) | 94K traj | **Web only, no AW** | — | Code MIT |
| **Learn-by-Interact** 2501.10893 | Docs → Claude-3.5 goals; **every sub-trajectory relabelled** | Claude-3.5-sonnet | LLM committee | ≈10K examples per benchmark | **No Android** ("Android" appears 0 times) | — | No code or data |
| **AgentTrek** 2412.09605 | Web tutorials mined from RedPajama | GPT-4o guided replay | GPT-4o evaluator (84% agreement) | 10,398 traj | **Web only** | — | No licence |
| **ZeroGUI** 2505.23762 | **GPT-4o** writes tasks from screenshot + benchmark exemplars | Own policy (online GRPO) | **Qwen2.5-VL-32B**, 4 votes, must be unanimous | 175 Android tasks used | **AndroidLab** 45.7→47.5; no AW | Yes: AndroidLab apps and its **test instructions** | Apache-2.0 (OSWorld code only) |
| **AndroidLab / AndroidInstruct** 2410.24024 | Seeds from AITW/Snips + manual + LLM expansion | **Crowd annotators on real phones**, plus LLM self-exploration | Second annotator; reward model after 500 traces | 10.5K traces / 94.3K steps collected; **726 / 6,208 used** | **AndroidLab** Qwen2-VL-7B 3.62→18.12 | Same 9 apps as its own test | MIT; 6,208 steps released |
| **GUI-Owl** 2508.15144 | Human-annotated per-app screen DAGs + slot values → LLM writes goals | GUI-Owl + Mobile-Agent-v3; humans when stuck | Step critic + trajectory critic, both must agree | not reported | Fig. 10 (fig.): online RS 35→39→48→51 | not disclosed | Weights MIT; data not released |
| **MAI-UI** 2512.22047 | Manuals + expert tasks + open data, expanded by MLLM | **Human annotators on an emulator** + multiple GUI agents | MLLM judge; **longest correct prefix kept** | not reported | 2B **45.1** SFT → 49.1 RL (Tab. 9) | not disclosed | 2B/8B weights Apache-2.0; data not released |
| **DigiRL** 2406.11896 | Human AitW instructions | Own policy (AutoUI 1.3B) | **Gemini-1.5-Pro** (2.8% error vs humans) | ≈1.5K traj; up to 64 emulators | AitW General 14.6→71.9; no AW | No (AitW) | Apache-2.0 |
| **DistRL** 2410.14803 | AitW + **some AW instructions** + curated | Own policy, asynchronous | Gemini-1.5-Pro (<2% discrepancy) | 32 emulators, 7.7 traj/min | AitW 73.2; **no AW eval** | AW instructions (count not reported) | Apache-2.0; data "later" |
| **Mobile3M** 2409.14818 | **None** (goal-free BFS crawl) | Scripted (Appium) | none | 3.1M screens, 20.1M actions, 49 apps | No online evaluation | No (Chinese apps) | CC BY-NC-SA 4.0 |
| **MobileViews** 2409.14337 | **None** (DroidBot crawl; GPT-4o only when stuck) | Scripted + GPT-4o + human | none | 1.21M screens, 30K apps | Grounding only | No | MIT (HF) |
| **V-Droid** 2503.15937 | AndroidControl/AitW goals + LLM-written goals from app-store text | **Humans pick from candidate list**; later the verifier, with humans correcting | Human (each trace driven to success) | 9K→110K pairs; >90 apps | AW **59.5** (with GPT-4 memory at inference) | AW is its "in-domain" test; apps likely seen (INFERRED) | Weights Apache-2.0; pairs not released |

---

## 1. Per-paper extraction

### 1.1 OpenMobile — arXiv 2604.15093 (v1, full text + App. A–E; repo read)
1. **Goals.** A decoupled three-stage pipeline (§3.1, App. A.1–A.3):
   - *Random walk*: 10 steps per session, each picking a random interactable a11y element to click or type.
   - *Environment memory*: pHash screen clustering at τ=0.95. **Gemini-3.1-Pro-Preview** annotates each unique
     screen's functionalities. An index is built with text-embedding-3-large.
   - *Memory-augmented synthesis*: Gemini writes 1–3 tasks per screen. Context is short-term memory (1 predecessor
     and ≤3 successor screens) plus long-term memory (30 related functionalities from the same app).
   - *Filter*: an LLM scores clarity and reasonableness 1–5 and drops anything <4. Embedding dedup at cos 0.8, and a
     per-app cap.

   Data-time models: Gemini-3.1-Pro-Preview (annotator, synthesiser, monitor, expert, CoT rewriter) and
   text-embedding-3-large. Inference-time models: fine-tuned Qwen2.5-VL-7B / Qwen3-VL-8B / Qwen2.5-VL-72B.
2. **Execution.** Runs on the AW emulator and its 20 apps (§3.3). (repo) Exploration and rollouts both start from the
   **state produced by AW's own `initialize_task` for every one of the 116 task classes**, with seed 222, instance 0.
   So "no test instructions" covers only the goal strings: apps, data and start states are AW's.
   - **Error-intervention switching** (§3.2, App. A.4):
     - The learner (an early fine-tuned checkpoint) acts.
     - A Gemini monitor sees recent actions and the last two screenshots and flags deviations.
     - The expert takes over for **≥3 steps**, at most **twice** per rollout.
   - **Verification:** the kept trajectories are those whose expert signalled `complete`/`answer`, i.e.
     **self-declared** (B.2; repo `process_trajs.py`). No programmatic checker is used. Yield: not reported.
3. **Per step.** Qwen3-VL action format. The expert rewrites the CoT (≈129 words per step).
   - **Targets are expert steps only**, but "preserving the full interaction history, including learner errors" (§4.1).
   - (repo) The input is the **current screenshot only**; history is `Step j: <conclusion>` text lines.
4. **Scale.** ≈2,800 tasks, 34K steps, 20 apps, mean 12.2 steps. Cost (emulator hours, API spend, parallel
   environments): not reported. SFT: lr 1e-5, 3 epochs, batch 32.
5. **Effect.** AW is reported as the mean of 3 runs ± half-range.
   - **Tab. 1:** 7B 25.5→51.7±1.7; 8B 47.6→64.7±3.2.
   - **Tab. 2b** (task synthesis, fixed 1.5K trajectories, 7B): OS-Genesis 34.1; coupled pipeline 45.3; OpenMobile
     48.3.
   - **Tab. 3** (rollout strategy, 1.5K trajectories, 7B learner):

     | Rollout | Error recoveries per trajectory | AW |
     |---|---|---|
     | Expert distillation | 0.42 | 44.8±1.7 |
     | **Self-evolution** (3 rounds, expert as judge) | 0.10 | **33.8±0.9** |
     | Random switch | 0.64 | 45.1±0.9 |
     | **Error-intervention** | 1.56 | **48.3±1.3** |

     With n=3 and no test, random switch vs distillation is within noise (INFERRED).
   - **Fig. 1 (fig.), data scaling:** 0→25.5, 0.5K→41.7, 1K→45.1, 2K→49.6, 2.8K→51.7.
   - **Tab. 4:** trajectory-level RL does not beat SFT on AW (64.7 → 64.1).
   - **Contamination (§5.2, Fig. 3).** 3.5% of instructions have cos >0.7 to an AW test instruction. Removing the
     most-similar 10% / 40% gives 50.5 / 42.2 (fig.); random 40% removal gives 48.0 (fig.).
   - **Tab. 5b** mixes base models and harnesses:
     - "AutoPlay 20K" counts tasks; AutoPlay trained on ≈8K trajectories.
     - MobileGen's 45.7 is under the M3A harness, where the base model scores 31.0 (OpenMobile's own harness gives it
       47.6).
     - Its OS-Genesis row (17.4) disagrees with OpenMobile's own OS-Genesis re-run (34.1).

     So this table is **not** a controlled comparison.
6. **Release.** Code github.com/njucckevin/OpenMobile-Code, **Apache-2.0**. Data HF `cckevinn/OpenMobile-Data`
   (LLaMA-Factory JSON + raw dumps with screenshots): **no licence stated**. Model `cckevinn/OpenMobile-8B`: no
   licence tag.

### 1.2 UI-Voyager — arXiv 2603.24533 (v1, full text; repo read)
1. **Goals.** A "seed task generator" perturbs parameters of the original templates (§3.2). The training tasks are
   **AW templates with new parameters**, ">7000 tasks", following MobileRL (§4.1, verbatim above). No external LLM
   writes tasks. Inference model: Qwen3-VL-4B fine-tuned.
2. **Execution.**
   - **Actor:** Qwen3-VL models in round 1, then the policy itself.
   - **Verifier:** "rule-based … checking application states using the Android Debug Bridge" (§3.1).
   - **RFT (fig.):** ≈37 → 63.5 → 69.0 → **73.2** (round 3) → 73.3. Text: "from 37% to 73%"; one round = 64.0%.
   - **GRSD (§3.3):**
     - Take the shortest successful rollout of the same task.
     - Find fork points by SSIM on cropped grayscale thumbnails ((repo) threshold 0.95, 480×270).
     - Training sample = the failed trajectory's context at step j + the successful action. Plain NLL.

   Yield (success/total rollouts): not reported.
3. **Per step.** (repo) Current screenshot only (`n_history_image: 0`), `history_len: 30` text conclusions. The output
   is `<thinking>` + tool call + `<conclusion>`. No a11y input.
4. **Scale.** Trajectories, GPU hours, emulators: not reported.
5. **Effect.** AW averaged over 64 seeds with random parameters (Tab. 2): 73.2 (RFT) → **81.0** (GRSD). GRPO/PPO from
   the same checkpoint plateau at ≈76 (Fig. 8). From the base model, GRPO/PPO need ≈175 steps to match one RFT round
   (Fig. 4). It is an **in-template** result: the test families are seen with other parameters.
6. **Release.** Code and model github.com/ui-voyager/UI-Voyager, **MIT**. Training data **not released**.

### 1.3 MobileGen — arXiv 2601.22781 (v1, full text)
1. **Goals.** Exploration controlled by sampled difficulty: depth (#steps), breadth (#apps), and instruction
   difficulty and ambiguity. A synthesiser then writes per-step thoughts and a trajectory-level instruction (reverse
   synthesis, §3.2.3).
   - The difficulty target comes from the student's Pass@3 on a 500-trajectory prior set: C* = C(1+αη), α=0.5 (Tab. 5).
   - Data-time model: **Qwen3-VL-8B-Instruct** as "the backbone for data generation" (§4.1). That it fills every role
     is INFERRED.
2. **Execution.**
   - **Input:** SoM screenshot + UI tree + supervisor-summarised history. The supervisor sets per-app step budgets,
     penalises cycles, and **rolls back** on anomalies (App. A.6).
   - **Device:** not stated (likely the 20 AW apps, INFERRED from the prompts and from "20 of the AC apps seen").
   - **Verification:** none for the 500 training trajectories, which are relabelled exploration. The prior set is
     filtered by a VLM judge (score >8).
3. **Per step.** (s̃_t = SoM + UI tree, a_t, s_{t+1}) + a thought. Training history format: not specified.
4. **Scale.** **500 trajectories, ≈20 steps** each. LoRA, 10 epochs, 8×4090. Cost: not reported.
5. **Effect.** Tab. 1, AW, M3A harness, all methods at 500 trajectories from the same generator:

   | Model | Zero-shot | OS-Genesis | GUI-ReWalk | MobileGen |
   |---|---|---|---|---|
   | Qwen3-VL-8B | 31.0 | 37.9 | 41.4 | **45.7** |
   | Qwen3-VL-4B | 25.0 | 31.9 | 33.6 | 40.5 |

   Removing rollback costs 4.4–12.2 (Fig. 3).
6. **Release.** None found.

### 1.4 AutoPlay — arXiv 2509.25047 (v1, full text)
1. **Goals.**
   - An MLLM explorer ("explore exhaustively", 3 turns per app; GPT-4o summarises memory).
   - A task generator then uses one exploration trace and one of **4 guideline prompts**: feature-use, composition,
     info-retrieval, repetition.
   - It writes 50 tasks per guideline × context, as **parameterised templates** with 5 values each (App. C.1).
   - Generator model: not named in the text read.
2. **Execution.**
   - **Executor:** GPT-4o planner + reflection + UI-TARS-1.5-7B grounding, on **the 20 AW apps** (§3.3).
   - **Verifier:** GPT-4o judge with no privileged state.
   - **Yield:** **≈20K tasks → ≈8K successful trajectories (≈40%)**. On the 5K ablation sets: executor success 46.0%;
     no-exploration 21.3% (Tab. 2).
3. **Per step.** Current screenshot + text action history (§3.3). AW action space.
4. **Scale.** ≈8K Android trajectories. RL: 32 H100, 192 environments (24 GPUs × 8), 368,640 environment samples.
5. **Effect.** AW, P@1 = mean of 5 trials with the AW checker:
   - 7B 19.5 → **40.1**; 72B 35.0 → 47.9; 3B 34.2 → 39.9 with RL.
   - Tab. 2 (5K tasks each): no exploration 28.8 vs AutoPlay 38.2; without guidelines 26.7.
   - Tab. 3: GPT-4o-verified 8K → 40.1; Qwen2.5-VL-72B-verified 12K → 38.6. A **weaker verifier with more data
     lost**.
6. **Release.** The code link is empty; nothing released.

### 1.5 OS-Genesis — arXiv 2412.19723 (v3, full text)
1. **Goals.** Rule-based traversal of interactive elements; GPT-4o fills text fields only. Each ⟨s_pre, a, s_post⟩
   (red box on the clicked element) goes to **GPT-4o**, which writes a low-level instruction and then a high-level
   goal (§3.1–3.2). Inference models: InternVL2-4B/8B, Qwen2-VL-7B.
2. **Execution.**
   - **Actor:** GPT-4o executes the high-level goals.
   - **Environment:** an emulator built on AW. That it is the **20 AW apps** is INFERRED from "of the 833
     AndroidControl apps … only 20 have been directly encountered".
   - **Verification:** no checker. A GPT-4o **trajectory reward model** scores 1–5 from the goal, the low-level steps
     and the **last 3 screenshots**. Nothing is discarded; sampling is weighted P ∝ R (Alg. 1). Spearman with humans
     0.813 (mobile).
   - Yield: not reported.
3. **Per step.** Pre/post screenshots, filtered a11y tree, low- and high-level instructions, action. Training input:
   screenshot + a11y + text "Action history". There is a single `<image>`, so no past screenshots (INFERRED).
4. **Scale.** **1K trajectories, 6.4 steps** on average. Cost: not reported.
5. **Effect.** Tab. 1, AW over **112 of 116 tasks**:

   | Model | Zero-shot | Task-driven | OS-Genesis |
   |---|---|---|---|
   | InternVL2-4B | 0.00 | 4.02 | **15.18** |
   | Qwen2-VL-7B | 0.89 | 6.25 | 17.41 |

   AW **saturates** with more data (Fig. 6; values figure-only). The trajectory reward model beats keep-complete-only
   (Fig. 5).
6. **Release.** github.com/OS-Copilot/OS-Genesis (README badge MIT, no LICENSE file). HF `OS-Copilot/OS-Genesis-mobile-data`
   (card: **MIT**).

### 1.6 AndroidGen — arXiv 2504.19298 (v1, full text)
1. **Goals.** "We utilize GPT-4o to generate about 300 task instructions **drawing on the instructions in
   AndroidWorld**" (§4.1). These are in-distribution to the AW templates. Inference models: LoRA-tuned GLM-4-9B and
   Llama-3-70B (text-only, XML input).
2. **Execution.**
   - **Actor:** the AndroidGen framework with GPT-4o: ExpSearch retrieval, ReflectPlan, and rule checks (AutoCheck).
   - **Verifier:** **StepCritic** (GPT-4o) splits the goal into sub-goals and finds the step that completed each one,
     using the action sequence and the final device state. Against the AW oracle it is 82.3% accurate per trajectory
     and 92.8% per sub-goal (Tab. 5).
   - **Hindsight prefix augmentation:** each completed prefix g₁…g_i of a failed run becomes a new task. This
     multiplies the data **×2.52** (Tab. 8).
   - **Raw success:** 58.2% (M3A 32%, human 80%).
3. **Per step.** Simplified XML (no screenshot) + a Python-call action + a short analysis. History: earlier turns
   carry only past actions and thoughts, with the observation elided (App. F).
4. **Scale and cost.** >1K trajectories. **$71.6 and 8.9 h per 1,000 trajectories with quality control** (human:
   $1,250 / 52.1 h) (Tab. 8).
5. **Effect.** Tab. 6, AW, Llama-3-70B: untrained 10.8 → all data unfiltered **16.2** → StepCritic-filtered **25.2**
   → oracle-checker-filtered **27.0** → with retrieval at inference 35.3.
   - Filtering is worth ≈+9.
   - An LLM critic gets within 1.8 points of the oracle checker.
6. **Release.** github.com/THUDM/AndroidGen, **Apache-2.0**. A training-trajectory release is not evident in the
   README.

### 1.7 GUI-explorer — arXiv 2505.16827 (v1, full text)
Training-free.
- An MLLM (GPT-4o, INFERRED) proposes goals anchored on `AndroidManifest.xml` activities and explores by DFS (b=10,
  d=5, ≤30 steps), restoring the parent state between siblings.
- Transitions whose screen does not change (pHash) are dropped.
- The output is >1,300 "knowledge items" (element patch → function text) over 46 apps, **including the AW apps**.
  They are retrieved at inference.
- AW 47.4 with GPT-4o (Tab. 3). Cost $0.064 and 66 s per step at inference.
- Not a training-data recipe; its anchor-driven exploration is reusable. MIT.

### 1.8 Explorer — arXiv 2502.11357 (v4, full text) — **web only**
- The caller's "web+mobile" description is wrong: Android appears only in related work.
- GPT-4o proposer → refiner (rewrites the goal after each action) → summariser (hindsight goal) → verifier (81%
  agreement with humans; precision ≈74%, our arithmetic from Tab. 10).
- 175K raw → **94K kept** (≈54%); $0.28 per kept trajectory; 50 h with 60 parallel processes.
- History is text past actions, with no past screenshots.
- Mind2Web-Live 7B: 14.5 → 19.3.
- Code MIT. A same-name paper, arXiv 2504.09352 ("Explorer: Robust Collection of Interactable GUI Elements"; abstract
  only), trains Android element detectors, not agents.

### 1.9 Learn-by-Interact — arXiv 2501.10893 (v1, full text) — **no Android**
- Claude-3.5-sonnet writes 3 goals per documentation page and executes them.
- **Backward construction:** every contiguous sub-trajectory is relabelled with an LLM-written goal, so the data
  grows quadratically with trajectory length.
- Filter: a unanimous Gemini-1.5-Pro + Claude committee keeps ≈25–60% (Tab. 1).
- WebArena, Codestral-22B: 4.7 → 24.2, vs 10.2 for distillation without backward construction (Tab. 3). Mixing short,
  medium and long sub-trajectories is best (Tab. 5).
- No code or data link.

### 1.10 AgentTrek — arXiv 2412.09605 (v2, full text) — **web only**
- 18.8M tutorials are filtered out of RedPajama (GPT-4o-mini + FastText).
- A GPT-4o agent replays them on live sites; a GPT-4o evaluator agrees with humans 84%.
- **Replay success 39.9%**; 10,398 trajectories (§4 also says "nearly 5,000" — the paper contradicts itself).
- **$0.551 per verified trajectory.** A tutorial-guided goal succeeds 52% vs 15.78% without the tutorial.
- No licence on code or data.

### 1.11 ZeroGUI — arXiv 2505.23762 (v1, full text)
1. **Goals.** **GPT-4o** writes 5 tasks per prompt from a screenshot + benchmark exemplars, including some infeasible
   tasks. 225 Android tasks were generated and 175 used.
2. **Execution and judging.**
   - The policy acts (online GRPO).
   - Reward comes from **Qwen2.5-VL-32B**, queried 4 times; it must be unanimous.
   - Against human labels (OSWorld only) the judge reaches **precision 61.5 / recall 51.1**. "False positives hurt
     more than false negatives" (Tab. 3b).
   - Stage 2 trains on the **benchmark's own test instructions**.
3. **Per step.** Screenshot-only; the base model's own token format.
4. **Scale.** G=64 rollouts per task. Compute: not reported.
5. **Effect.** **AndroidLab** UI-TARS-7B 45.7 → 47.5 (+1.8), of which generated-only is +0.7. No AW number.
6. **Release.** Apache-2.0 (OSWorld code only).

### 1.12 AndroidLab / AndroidInstruct — arXiv 2410.24024 (v2, full text)
1. **Goals.** Seeds from AITW/Snips + manual writing + expansion by an unnamed LLM, then reviewed.
2. **Execution.**
   - **Crowd annotators** under contract use an ADB + Accessibility tool on their own **real phones**. A second
     annotator cross-verifies.
   - LLM self-exploration was hand-checked for the first 500 traces, then checked by an unnamed reward model.
3. **Per step.** Screenshot, XML from a re-implemented Accessibility Service (uiautomator fails on animated pages), and
   the operation. Targets are direct actions (4.96 tokens on average), with no CoT.
4. **Scale.** **10.5K traces / 94.3K steps collected; only 726 traces / 6,208 steps (the 9 benchmark apps) used for
   training.**
5. **Effect.** AndroidLab (the same 9 apps as training): Qwen2-VL-7B SoM 3.62 → 18.12; Llama-3.1-8B XML 2.17 → 23.91.
6. **Release.** MIT. 6,208 XML / 6,053 SoM steps released; the full 94.3K is not.

### 1.13 GUI-Owl / Mobile-Agent-v3 — arXiv 2508.15144 (v2, full text)
1. **Goals.** Mobile: per-app **human-annotated screen DAGs** with hand-written slot values → path sampling → an LLM
   writes and refines goals (§2.2.1). Named data-time models (all listed as "e.g."): Qwen2.5-VL, Qwen2.5-Max,
   Qwen3-235B, Qwen-VL-Max.
2. **Execution.**
   - GUI-Owl and the Mobile-Agent-v3 roles roll out on Alibaba cloud phones.
   - Queries that keep failing get human or other-model reference trajectories, which are turned into guidance for a
     re-rollout.
   - **Step critic:** GOOD / NEUTRAL / HARMFUL. **Trajectory critic:** text and multimodal channels, which must both
     say correct.
   - Critic models and accuracies: not reported. Yield: not reported.
3. **Per step.** Reasoning + `conclusion` + action. **Only the conclusion enters history; the last 1–3 images are
   kept.**
4. **Scale.** No counts are reported anywhere.
5. **Effect.** AW: GUI-Owl-7B 66.4 (Tab. 7). Fig. 10 (early checkpoint, read off the figure):
   - Iterative online rejection sampling: 35 → 39 → 48 → 51.
   - Offline reasoning synthesis: 35 → 43 → 45 → 53.

   OSWorld, 1/2/3 history images: 26.6 / 28.5 / 29.1. Whether it trained on AW apps: not disclosed.
6. **Release.** Weights MIT; data, DAGs and critics not released.

### 1.14 MAI-UI — arXiv 2512.22047 (full text; no per-step metric reported, none derived)
1. **Goals.** App manuals + expert-written tasks + filtered open data. An MLLM expands them by changing parameters (L1)
   or swapping objects (L2). An LLM user-simulator answers `ask_user`.
2. **Execution.**
   - **Human annotators on an Android emulator**, each trajectory reviewed by a second annotator, plus "multiple GUI
     agents".
   - An unnamed **MLLM judge** keeps the **longest correct prefix** of failed trajectories.
   - Iterative rejection sampling (Eqs. 2–3).
   - RL environment: >35 apps, **up to 512 emulators on 10 servers**, hybrid rule/MLLM verifier with **83% agreement
     with humans**.
3. **Per step.** Instruction + "one or more screenshots" + action/observation history. The number of past images is
   not reported.
4. **Scale.** Not reported.
5. **Effect.** Tab. 9, AW: 2B **45.1 → 49.1**, 8B 64.7 → 70.7, 32B 69.8 → 73.3 (SFT → RL). Parallel environments
   32 → 512: 65.5 → 70.7 (Fig. 12b). Whether AW apps were in training: not disclosed.
6. **Release.** 2B/8B weights, Apache-2.0 per the README. No data.

### 1.15 DigiRL — arXiv 2406.11896 (v1, full text)
- Human AitW tasks: General 545 train / first 96 test; Web Shopping 438 / 96.
- Data-time models: rollouts from AutoUI (1.3B); evaluator **Gemini-1.5-Pro** on the current screenshot (2.8% error
  vs humans).
- State: **last two screenshots**. 1,552 / 1,296 offline trajectories.
- Online: 16 trajectories per iteration, **up to 64 emulators, 1.74 trajectories/min**.
- Step filter on advantage > 1/H, plus a curriculum on instruction value.
- AitW General 14.6 → **71.9**. No AW. Apache-2.0.

### 1.16 DistRL — arXiv 2410.14803 (v5, full text)
- 600 General tasks = AitW + **instructions selected from AndroidWorld** (count not reported) + curated tasks.
- Gemini-1.5-Pro reward from the **last screenshot + last two actions**: <2% discrepancy. Adding more images *hurt*
  the judge.
- **32 emulators, ≈7.7 trajectories/min.**
- AitW General 73.2. No AW evaluation. Apache-2.0; data "later".

### 1.17 Mobile3M — arXiv 2409.14818 (v2, full text)
- **Goal-free BFS** over every element of **49 Chinese apps**, Appium on >50 Cuttlefish Android-14 emulators.
- **Page dedup:** BM25 over XML retrieves the top-5 existing pages; a page is a duplicate if **<5 elements and <30% of
  pixels** differ, and becomes an edge instead of a node.
- Each node stores screenshot + XML + its shortest action trace from the app's home screen.
- Size: **3,098,786 screens, 20,138,332 actions**.
- Offline gains only; seen apps transfer poorly to 7 held-out apps (action IoU 35.85 vs 9.80).
- **CC BY-NC-SA 4.0.**

### 1.18 MobileViews — arXiv 2409.14337 (v3, full text)
- DroidBot crawl, ≤1,000 actions per app.
- GPT-4o steps in on login pages or when stuck; humans resolve the rest. Apps traversed without getting stuck: rules
  alone 18.6%, + GPT-4o 44.2%, + humans 100%.
- **1,213,866 screens with view hierarchies from 30,037 apps; 151,600 device-hours** on >200 virtualised ARM instances.
- **Quality over quantity:** keeping samples where the base model passes 1–7 of 8 tries matches 26K random samples.
- Grounding only. MIT (HF).

### 1.19 V-Droid — arXiv 2503.15937 (v5, full text) — the structurally closest system to ours
1. **Goals.** AndroidControl and AitW instructions + LLM-written goals from app-store descriptions, manually verified.
   AW is its "in-domain" test. Only AndroidLab and MobileAgentBench apps were excluded from training, so AW apps were
   likely included (INFERRED).
2. **Execution.**
   - **Round 1:** humans pick the correct action from the extracted candidate list (≈5 h for 9K pairs).
   - **Rounds 2–4:** the verifier labels steps, and those whose score entropy is above the median are sent to a human.
   - Every trace is driven to success.
3. **Pairs.** For each step on a correct trace, the labelled candidate is the positive and **every other extracted
   candidate is a negative** (N−1 pairs per step). Loss −log σ(τ⁺−τ⁻).
   - Self-correction pairs (reverse action such as back) are capped at ≈2.5%; more than that collapsed the model into
     always pressing back.
   - State is text a11y XML; ≈50.3 candidates per step.
4. **Scale.** 9K → 27K → 55K → **110K** pairs over 4 rounds; >90 apps; 90 h on 16×A100 for the last round.
5. **Effect.**
   - AW SR rises 15.0 → 37.5 → 47.4 → **59.5** as the pairs grow from 9K to 110K. The 27K/55K points are mapped by the
     order of the text.
   - **Tab. 3, same data:** pairwise verifier **47.4** vs SFT *selector* (outputs an index) **35.8** vs SFT generator
     27.4.
   - Self-correction pairs: 52.2 → 59.5.
   - **Tab. 4:** GPT-4-written working memory 59.5 vs rule-based 46.1 vs action history only 40.0. The headline
     number depends on an external LLM at inference.
6. **Release.** LoRA weights Apache-2.0; the repo has no licence; pairs not released.

---

## 2. What the evidence says (cross-paper; each line names its source)

1. **Every recipe with a strong AW number trains inside AW's apps.**
   - UI-Voyager uses the templates themselves. OpenMobile uses AW's task initialisation for all 116 classes. AutoPlay
     uses the 20 apps. AndroidGen uses rewrites of AW instructions. OS-Genesis likely uses the 20 apps (INFERRED).
   - None tests a model trained *without* AW apps on AW.
   - The closest evidence is OpenMobile Fig. 3: removing the 40% of its data most similar to AW costs ≈9.5 points vs
     ≈3.7 for random removal (fig.).
   - **Consequence for us:** any AW gain from AW-app data must be reported next to an AC-only / no-AW-app arm (R3
     risk 2).
2. **Who verifies matters more than how much.**
   - AndroidGen Tab. 6: no filter 16.2, LLM critic 25.2, checker 27.0.
   - AutoPlay Tab. 3: GPT-4o-verified 8K (40.1) beats Qwen-72B-verified 12K (38.6).
   - ZeroGUI: judge precision is only 61.5, and false positives hurt more.
   - OpenMobile self-evolution with an expert judge: 33.8 vs expert distillation 44.8.
   - UI-Voyager self-evolution with **a programmatic checker** works: 37 → 73 in 3 rounds.
   - **The programmatic AW checker we already have is the best verifier in this set.**
3. **Small, well-targeted data moves AW a lot.** OpenMobile goes 25.5 → 41.7 with 0.5K trajectories (fig.). MobileGen
   gains +14.7 with 500. OS-Genesis gains +15 with 1K. OS-Genesis saturates (Fig. 6). MobileViews and AndroidLab used
   only a small, filtered or in-domain fraction of what they collected.
4. **Recovery behaviour has to be put into the data on purpose.** OpenMobile's error-intervention data carries 1.56
   recoveries per trajectory vs 0.42 for expert-only, and scores 48.3 vs 44.8. The trick: the targets are the
   expert's steps, but the learner's errors are kept in the history. V-Droid's self-correction pairs add +7.3 but
   collapse above ≈2.5%. UI-Voyager's GRSD fork points sit on the same axis.
5. **Partial credit from failures is standard.** Sources: MAI-UI's longest correct prefix, AndroidGen's sub-goal
   prefixes (×2.52), and Learn-by-Interact's sub-trajectory relabelling (+14.0 on WebArena).
6. **History format is nearly universal.** The current screenshot plus text lines for past actions or "conclusions"
   (OpenMobile, UI-Voyager, AutoPlay, AndroidGen, OS-Genesis, Explorer). GUI-Owl keeps 1–3 images; DigiRL keeps 2. This
   matches our contract v3 (text history) and R3 Task 12 (2 past screenshots).
7. **For a choice head, the training objective matters as much as the data.** V-Droid Tab. 3: a pairwise
   positive-vs-every-other-candidate loss scores **47.4 vs 35.8** for an SFT index selector on identical data. Ours is
   an index selector today.
8. **Throughput and cost anchors.**
   - AndroidGen: $71.6 / 8.9 h per 1K trajectories.
   - AgentTrek: $0.55 per verified trajectory.
   - Explorer: $0.28 per kept trajectory.
   - DigiRL: 1.74 trajectories/min on 64 emulators. DistRL: 7.7/min on 32.
   - MAI-UI: 512 emulators, with 32 → 512 environments worth +5.2 in RL.
   - **We have 2 emulators.** RL-scale parallelism is out of reach; SFT/RFT-scale data (≤3K trajectories) is not
     (INFERRED).

---

## 3. Recommended pipeline for us

**Setting.** A ≈2B single-pass decision model (choices only, no text), one 24 GB GPU, two emulator hosts, AW @55ddaf5
with its checkers.

**Hard rules.**
- No external model at inference.
- The 116 evaluation instances (seed 30, `sha256("30_{name}_0")`) never enter data. Any instance whose parameters equal
  an evaluation instance is refused by test.

**Fits the existing R3 plan.** Task 6 rollouts, Task 7 termination, Task 9 data mix, Task 10 RFT.

### 3.0 Common core (both variants)

**C1. Exploration graph of the 20 AW apps** (Mobile3M / OpenMobile pattern; no model needed).
- Start states come from `initialize_task` on **non-eval seeds**, as OpenMobile does with seed 222, which yields
  realistic app data. If operator decision D1 forbids template initialisation, use a plain app launch instead.
- Run BFS or random walk over the element list that is actually served (`forest_to_ui_elements`).
- Dedup pages Mobile3M-style (<5 elements and <30% pixels differ) or by pHash ≥0.95 (OpenMobile).
- Drop no-change transitions (GUI-explorer).
- Store per transition: screenshot, served element list, contract-v3 action, next screenshot, effect, app, seed.
- Value: the screen and transition graph gives (a) start states and anchors for goals, (b) `open_app` / navigation
  rows for the 13 apps with zero training goals, and (c) the reverse-action map for recovery pairs.

**C2. Rows in the v3 schema** with these extra fields:
- `policy_source` (own / expert / human)
- `is_target` (false for learner-error steps kept as history — the OpenMobile rule)
- `instance_seed`, `template`
- `checker_success`, `premature_done_step`, `prefix_ok_until`

History = the same text lines used at serving. Current screenshot only, until Task 12 tests 2 past screenshots.

**C3. Verification = AW's own checker wherever a goal comes from a template** (the UI-Voyager rule). Failed
trajectories contribute their longest correct prefix only when a checker or critic can locate it (MAI-UI /
AndroidGen). Otherwise they contribute nothing except the premature-`status` mark (Task 7).

**C4. Objective.** Keep imitation on (state → chosen element / action). Add a **pairwise term over the candidate list**
on the same rows: target candidate vs every other candidate (V-Droid P³).
- This is the one measured objective change sized for a choice head: Tab. 3 shows +11.6 for an 8B model on AW.
- Use it as a pre-registered arm, not a default (M6: 3 seeds).
- Recovery rows (reverse action after a deliberately wrong step) are capped at ≈2.5% (V-Droid's collapse finding).

**C5. Contamination audit.** For every synthetic goal, compute similarity to the 116 eval goals. Report the share above
0.7 (OpenMobile's statistic) and a removal ablation. Every number is reported **with and without** AW-template data
(R3 risk 2).

**C6. Throughput budget** (INFERRED from our own record: the 116-task suite took ≈2.5 h, i.e. ≈78 s per episode
including setup).
- ≈1,100 episodes per emulator per day, ≈2,200 per day on both hosts, with a restart every ≈20 episodes (the known
  a11y-forwarder decay).
- At today's ≈15% success: ≈330 checker-verified successes per day.
- The 1–3K-trajectory regime where the papers saw their gains (OpenMobile 0.5–2.8K, MobileGen 0.5K, OS-Genesis 1K) is
  therefore **1–9 emulator-days**.
- The GPU alternates between serving the rollout policy and training; it never does both at once.

### 3.1 Variant N — no external model at data time either

The only actors are our own policy and scripted exploration. The only verifier is the AW checker. This is the
UI-Voyager / MobileRL family, adapted to a model that cannot write text.

1. **Goals (no model writes text).**
   - (a) If D1 = yes: AW templates on seeds disjoint from 30. This is the UI-Voyager source: ">7000 tasks" from
     re-sampled parameters, plus perturbation of parameters such as dates, counts and file names.
   - (b) If D1 = no, or as the held-out arm: **our own parameterised templates** over the 20 apps. Each is written by
     a person once, with a programmatic checker reading app state (adb/sqlite, the way AW checkers do). Goals are
     generated from the C1 graph — for example "open X and go to <screen title>", "toggle <setting>", "create <entity>
     with <fields>" — with the C1 trace as the reachability proof.
   - (c) **Deterministic hindsight relabelling** (the AndroidGen / Learn-by-Interact idea without an LLM): when a
     rollout changes app state in a way a checker can read (a note created, a contact added), emit the template goal
     that state satisfies, labelled with the trajectory prefix up to that step.
2. **Rollouts.** The policy samples from its own choice distribution (temperature) for k samples per instance, for
   UI-Voyager-style group rollouts.
   - Keep checker-verified successes, choosing the shortest per instance.
   - **Fork-point rows** (GRSD): where a failed rollout diverged from a successful sibling of the same instance, take
     the failed context plus the successful action. This is pure imitation, so it fits our head.
   - Failed rollouts give premature-`status` negatives (Task 7). They give no positive targets.
3. **Rounds.** RFT (Task 10). Retrain on mix + successes; stop when the gain on `aw_val_v3` is <1 point, or after 3
   rounds. For reference, UI-Voyager's first round was the largest jump (≈37 → 64). GUI-Owl went 35 → 39 → 48 → 51
   (fig.).
4. **Bootstrapping risk** (named, not hidden). At ≈15% success, the first round sees only the easy templates.
   - MobileGen's difficulty sampling (sample goals near the model's current pass@k boundary) and MobileViews' "1–7 of
     8" filter both apply here: prefer instances where some but not all of k rollouts succeed.
   - OpenMobile Tab. 3 is the warning: pure self-evolution scored 33.8 vs 44.8 for expert data. That was with an LLM
     judge, though, and UI-Voyager's checker-verified self-evolution did not stall.

   **Kill condition:** if round 1 adds <2 points on `aw_val_v3` (the seed noise), go to Variant E or stop.

### 3.2 Variant E — external model at data time only (inference stays single-model)

The OpenMobile recipe with checkers added. Models: an API model, or a local open VLM.
- MobileGen used **Qwen3-VL-8B** as its whole data generator. ZeroGUI judged with **Qwen2.5-VL-32B**.
- An 8B fits next to nothing else on the 24 GB card, so generation and training alternate (INFERRED).
- The model's licence must permit training on its outputs. Check before use.

1. **Goals** (OpenMobile §3.1 / Tab. 2b: decoupled 48.3 vs coupled 45.3 vs OS-Genesis 34.1 at 1.5K):
   - C1 graph → the model annotates each unique screen's functions → it writes 1–3 goals per screen from short-term
     context (neighbour screens) and long-term context (other functions of the same app).
   - Score and filter (≥4/5), embedding dedup at cos 0.8, per-app caps.
   - Reverse synthesis (OS-Genesis / MobileGen) is the cheaper fallback: one model call per transition.
2. **Rollouts with error intervention** (OpenMobile Tab. 3: 48.3 vs expert-only 44.8 at equal size):
   - Our policy acts. The model, as monitor, reviews the last two screenshots plus the recent actions.
   - On a deviation the model takes over for ≥3 steps, at most twice per rollout.
   - **Targets = expert steps only; learner errors stay in history.**
   - Expert actions must map onto our action space: the element index from the served list, and app names from the
     launchable list. **Unmappable expert steps end the rollout** and are counted.
3. **Verification.** The AW checker when the goal comes from a template. Otherwise an AndroidGen-style sub-goal
   critic (82.3% trajectory accuracy, within 1.8 points of the oracle checker in Tab. 6) with unanimous voting
   (ZeroGUI: precision first), plus the longest-correct-prefix rule. OpenMobile's self-declared completion is **not**
   adopted: it has no measured precision.
4. **Size.** 1–3K kept trajectories (OpenMobile's curve: 0.5K → 41.7, 2.8K → 51.7, fig.). Then the Variant N RFT
   rounds on top. The external model is no longer needed once the policy's own successes take over.
5. **Disclosure.** State the data-time model, its share of target steps, and the eval-similarity audit (C5). This
   variant still uses **no external model at inference**, so it is legal under the operator rule as stated today.

### 3.3 Pre-registration items both variants need (from M4/M6/M11/M15)
- Three seeds and a paired test on the 116. Report SR on 116 and 104 with the denominator named.
- Arms:
  - AC-only
  - AC + exploration (C1) only
  - AC + AW-template rollouts (if D1 = yes)
  - AC + Variant E data (if approved)
  - each with and without the pairwise term (C4)
- The rollout-seed list is checked against all 116 eval parameter sets over the **real** registry (M15), not on a
  hand-picked sample.

---

## Sources (all read 2026-11-11, arXiv HTML full text unless noted)
OpenMobile 2604.15093 v1 (+ github.com/njucckevin/OpenMobile-Code `AndroidWorld/random_walk_aw.py`, `process_trajs.py`,
`convert_traj.py`) · UI-Voyager 2603.24533 v1 (+ repo) · MobileGen 2601.22781 v1 · AutoPlay 2509.25047 v1 ·
OS-Genesis 2412.19723 v3 · AndroidGen 2504.19298 v1 · GUI-explorer 2505.16827 v1 · Explorer 2502.11357 v4
(2504.09352 abstract only) · Learn-by-Interact 2501.10893 v1 · AgentTrek 2412.09605 v2 · ZeroGUI 2505.23762 v1 ·
AndroidLab 2410.24024 v2 · GUI-Owl 2508.15144 v2 · MAI-UI 2512.22047 (and `references/raw/maiui2026-androidworld-table9.txt`) ·
DigiRL 2406.11896 v1 · DistRL 2410.14803 v5 · MobileVLM/Mobile3M 2409.14818 v2 · MobileViews 2409.14337 v3 ·
V-Droid 2503.15937 v5. Licences from the GitHub API and HF metadata of the linked repositories.
Not added to `references/ledger.md` in this pass (backfill needed per AGENTS §4 before any of these enters the paper).
