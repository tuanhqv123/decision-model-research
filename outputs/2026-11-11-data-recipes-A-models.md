# How the strong small/medium mobile GUI agents were trained — data recipes (set A)

Date: 2026-11-11. Scope: training data, observation/history format, action space, stages, AndroidWorld (AW) /
AndroidControl (AC) numbers, and data ablations, for the models the brief named.

**Method.** Every paper below was downloaded as arXiv HTML on 2026-11-11, converted to plain text (math
`alttext` kept so numbers inside LaTeX survive), and the numbers were checked against that text by `grep`, not
taken from abstracts or third-party summaries. For MAI-UI the public GitHub prompt/agent code was also read.
Section/table numbers refer to the arXiv HTML version listed under each heading.

**Conventions.** "not reported" = absent from the paper text we read. **INFERRED** = our reading, not a
statement in the source. Any number we computed ourselves is marked **(ours)**. No number in this file is
derived from a "per-step" inversion of task success (see AGENTS.md M13).

**Coverage.**

| Model asked for | Primary source found | Status |
|---|---|---|
| MAI-UI 2B/8B/32B | arXiv 2512.22047v1 + GitHub `Tongyi-MAI/MAI-UI` | covered |
| UI-Venus-1.5 2B/8B/30B-A3B | arXiv 2602.09082v1 | covered |
| UI-Venus(-Navi) 7B/72B (v1) | arXiv 2508.10833 | covered (no 2B Navi exists in v1; the 2B is UI-Venus-1.5) |
| GUI-Owl 7B/32B / Mobile-Agent-v3 | arXiv 2508.15144 | covered |
| UI-TARS (v1) / UI-TARS-1.5 / UI-TARS-2 | arXiv 2501.12326; 2509.02544; 1.5 = GitHub README only | v1 and 2 covered; **1.5 has no technical report** (README: "reasoning enabled by RL", 7B released, AW 64.2; training data not described) |
| AutoGLM-Mobile / AutoGLM-Phone-9B | arXiv 2411.00820 (AutoGLM); HF card `zai-org/AutoGLM-Phone-9B` | **training data not disclosed anywhere we found**; card says architecture = GLM-4.1V-9B-Thinking and links the MobileRL paper |
| Step-GUI 4B/8B | arXiv 2512.15431v2 | covered |
| MobileRL 7B/9B | arXiv 2509.18119 | covered (most fully disclosed recipe in this set) |
| V-Droid 8B | arXiv 2503.15937 | covered (closest architecture to ours) |
| UI-Voyager 4B | arXiv 2603.24533 | covered |
| Hammer-UI | — | **not found**: no paper, report or model card located; only an aggregator leaderboard row ("Hammer-UI-32B 75%") with no source. Skipped. |
| MobileUse-v2 | — | **not found**. MobileUse v1 (arXiv 2507.16853, NeurIPS 2025) is a **training-free** multi-agent framework on Qwen2.5-VL-72B-Instruct (temperature 0), AW 62.9 / AndroidLab 44.2 — no training recipe to extract. |

---

## 1. MAI-UI (Alibaba Tongyi) — 2B / 8B / 32B / 235B-A22B — arXiv 2512.22047v1

**Backbone.** Qwen3-VL (§2.1). Jointly trained on grounding, perception and navigation with SFT + RL.

**1. Training data**
- Grounding/perception (§2.2.1): open-source **JEDI** and **OS-Atlas**, plus screenshots from containerized OSes
  explored by an MLLM; elements localized via a11y tree or OmniParser V2. Instructions written from four
  perspectives (appearance, function, location, intent). Paper states ~23.3% of open-source grounding
  instructions have quality issues. Counts: not reported.
- Navigation tasks (§2.3.1): seeds from (1) app manuals, (2) expert-designed tasks, (3) "open-source datasets,
  filtered by task complexity and reachability" — which datasets is not named.
- Trajectories: human annotation on an Android emulator (second-annotator review), model-based synthesis, and
  rejection-sampled rollouts judged by an MLLM at trajectory and step level. Counts and mix ratios: not reported.
- ask_user / mcp_call trajectories: synthetic-user (LLM with hidden context) and MCP tool traces (§2.3.2). Counts:
  not reported. 32B/235B "augmented with more real-world trajectories" (§3.1), amount not reported.
- **AC / GUI-Odyssey** appear only as evaluation benchmarks; training use not stated. AITW, AMEX: not mentioned.
- **AndroidWorld relation — not disclosed.** The RL environment is a Docker image "inspired by an experimental
  feature in AndroidWorld" with "over 35 applications, encompassing system utilities and open-source software"
  (§2.3.3). Whether the AW apps/templates are among them is not stated. **The released mobile system prompt
  (`src/prompt.py`) hard-codes "Available Apps" = Camera, Chrome, Clock, Contacts, Dialer, Files, Settings, Markor,
  Tasks, Simple Draw Pro, Simple Gallery Pro, Simple SMS Messenger, Audio Recorder, Pro Expense, Broccoli APP,
  OSMand, VLC, Joplin, Retro Music, OpenTracks, Simple Calendar Pro** — i.e. the AndroidWorld app set.
  INFERRED: the AW app list is part of the deployed (and probably trained) prompt; overlap of RL tasks with AW
  is likely but unproven.

**2. Observation / history.** Screenshot only as model input (a11y used only for data construction). History
`h_t = (a_1, o_1, …, a_{t-1}, o_{t-1})` (§2.1.1); on-device memory keeps "historical screenshots" plus past thoughts
and actions (§2.4.1). Number of past screenshots: not in the paper; **the released agent defaults to
`history_n = 3` history images**. Images resized to half resolution in RL (§2.3.3); 720p ≈ 1080p in performance
with ~50.1% per-step speedup, 540p "substantially degrades" (§3.5, no number for 540p). Coordinates: code uses
`SCALE_FACTOR = 999` (0–999 relative grid, Qwen3-VL convention).

**3. Actions** (Table 1): click(x,y), long_press, type(text), swipe(direction[, coord]), drag, system_button
(back/home/menu/enter), wait, terminate(success|fail), answer(text), ask_user(text), mcp_call. Table 1 has no
open-app action, **but the released prompt adds `{"action":"open","text":app_name}`** and tells the model to prefer
it. Output: `<thinking>…</thinking><tool_call>{json}</tool_call>`.

**4. Stages** (§3.1): (i) SFT on perception+grounding; (ii) SFT on navigation with a small portion of grounding;
(iii) grounding RL (GRPO; format + point-in-box reward; "instruction-as-reasoning"); (iv) online navigation RL.
Self-evolving loop: cold-start SFT model, then alternate fine-tuning and rejection-sampled rollouts (round count
not reported). Online RL (§2.3.3): GRPO variant, group 16, no KL, token-level loss, ε_low 0.2 / ε_high 0.3; binary
task reward + penalty for repeated 3–5-action cycles; replay buffer keeps 8 most recent successes per task;
**up to 512 concurrent environments**; budget up to 50 steps; curriculum in four pass@K tiers (0–25/25–50/50–75/
75–100%), shifting toward hard tasks.

**5. Numbers.** AW (Table 6): 2B 49.1, 8B 70.7, 32B 73.3, 235B-A22B 76.7. **SFT vs RL (Table 9): 2B 45.1 → 49.1
(+4.0); 8B 64.7 → 70.7 (+6.0); 32B 69.8 → 73.3 (+3.5).** AC-High (Table 7): 2B 67.3, 8B 69.1, 32B 75.5. GUI-Odyssey:
72.6 / 80.1 / 83.4.

**6. Ablations (all on RL, 8B).** Plain GRPO +1.8 vs their GRPO +6.0 (Table 9); max_env_steps 15/30/50 → +1.7/+3.8/
+6.0; parallel envs 32 → 512: 65.5 → 70.7 (Fig. 12b). **No data-mix, dataset add/remove, or quantity ablation.**

---

## 2. UI-Venus-1.5 (Ant Group) — 2B / 8B / 30B-A3B — arXiv 2602.09082v1

**1. Data.** Mid-training corpus: **10B tokens from "over 30 diverse sources", naming Mind2Web, ShowUI, AITW
"and so on"** (§2.1.1); shares: semantic perception 20.8%, GUI-VQA 22.1%, grounding 24.8%, remainder hybrid
navigation-reasoning (share not given numerically). Teacher refinement (§2.1.2): Qwen3-VL-235B-A22B scores each
sample 0–10; ≥7 kept, 4–6 rewritten, 0–3 rebuilt or dropped; high-fidelity share 69.7% → 89.7%. Real-device
generation loop (§2.1.3): MLLM-generated tasks → executed on cloud devices → multi-annotator verification;
pipeline success 17.9% → >70%; **>30,000 verified trajectories**. Online-RL task pool: LLM tasks for predefined
apps/websites + MLLM tasks from offline screenshots, bucketed Easy ≤10 / Medium 10–20 / Hard >20 steps; pool size
not reported. AC, GUI-Odyssey, AMEX: not named as training data. AW apps/tasks in training: not stated.

**2. Observation.** Screenshot only ("only raw screen screenshots", §3.2.2). History is **text**: prompt
placeholder `{previous_actions}` (Appendix A; Chinese prompt: "previous actions and reasoning process").
Number of past screenshots, max history, max pixels: not reported. Coordinates normalized to [0,1000] (§3.1.1).

**3. Actions** (Table 8, 16 actions): Click, Drag, Scroll (mobile: start+end coords), Type, **Launch**, Wait,
Finished, **CallUser** (answer/ask), LongPress, PressBack, PressHome, PressEnter, PressRecent, Hover, DoubleClick,
Hotkey. Output: `<think>`, `<action>`, `<conclusion>` tags. Grounding refusal = `[-1,-1]`.

**4. Stages.** Mid-training → offline RL (GRPO; grounding reward = format + point-in-box; navigation reward =
format + action type + content F1 + hierarchical coordinate reward) → online RL (trajectory-level GRPO
advantages, adaptive KL with reference blending, annealed entropy; "hundreds to thousands of concurrent
devices"; group size not reported) → TIES model merge. SFT appears only as "prior initialization" — no details.

**5. Per-stage AW (Table 7):**

| Size | Mid-train | Offline-RL | Online-RL | Merge (final) |
|---|---|---|---|---|
| 2B | 39.0 | 45.3 | **59.8** | 55.6 (−4.2) |
| 8B | 57.0 | 63.5 | 72.7 | 73.7 |
| 30B-A3B | 67.1 | 68.0 | 75.5 | 77.6 |

Note the 2B: online RL gave +14.5 and the merge then **cost 4.2 points on AW**; the shipped 2B is the merged one.
AC: not reported. AndroidLab 36.2 / 55.1 / 52.9.

**6. Ablations.** Stage ablation above; Linear vs TIES merge (30B: Linear −2.3 AW, TIES +2.1). **No dataset or
quantity ablation.**

---

## 3. UI-Venus (v1) — UI-Venus-Navi-7B / 72B — arXiv 2508.10833

**1. Data — fully itemized.** Navigation (Table 2, samples / traces):

| GUI-Odyssey | AITZ | AC | AITW | AMEX | Navi* (in-house Chinese apps) | GD* (grounding) | VQA* | **Sampled training set** |
|---|---|---|---|---|---|---|---|---|
| 114k / 7k | 14k / 2k | **84k / 13k** | 371k / 5k | 39k / 3k | 20k / – | 107k | 42k | **350k** |

(Raw pool sums to ~791k samples **(ours)**; 350k selected after cleaning, i.e. AC was at most 84k of 350k — the
per-source share after sampling is not reported.) Grounding (Table 1): 627k open-source (Widget Captioning 34k,
UI RefExp 17k, SeeClick-Web 325k, ShowUI 110k, OmniAct 141k) → 107k after **manual** cleaning; "~40% of
open-source grounding data contain significant noise". Automatic cleaning with open models and rejection-sampled
hard-sample RFT were tried and **failed** (§3.2.2).
Pipeline (§3.3.2): filter short/inconsistent traces (MLLM summary vs task), unify scroll direction, **resample
by app/subtask for diversity**; **Trace Reconstruction** inserts a `CallUser(answer)` step before `Finished` in
information-retrieval traces (answers generated by an MLLM from the last screenshot); iterative trace generation
on "dozens" of cloud phones with rule + ORM + annotator filtering; failed actions fixed by annotators and added
with their valid prefix. AW usage: not stated.

**2. Observation.** Current screenshot only; history = previous n−1 **thought–action pairs** as text (§3.3.3,
Fig. 2). No past screenshots (INFERRED from "current UI screenshot" + text H_{n−1}).

**3. Actions** (Table 3): Click, Drag, Scroll(start,end,direction), Type, **Launch(app)**, Wait, Finished(content),
CallUser(content), LongPress, PressBack, PressHome, PressEnter, PressRecent. Pixel coordinates.

**4. Stages.** GRPO only on Qwen2.5-VL (no SFT stage described). Navigation reward = format + type + coordinate
(two distance thresholds, partial credit) + content. Hyper-params (Table 4): Navi-7B rollout 8, batch 256,
β_KL 1e-3, frozen ViT, 10 epochs; Navi-72B rollout 8, batch 512, 1 epoch. **Self-evolving history alignment**:
between epochs, historical thoughts are replaced by the current model's own thoughts (kept only if the rollout's
action exactly matches GT). **Sparse action enhancement**: oversample long-tail actions (e.g. LongPress).

**5. Numbers.** AW: 7B 49.1, 72B 65.9. AC (Table 11, type acc / step SR): 7B Low 97.1/92.4, High 86.5/**76.1**;
72B Low 96.7/92.9, High 85.9/**77.2**. GUI-Odyssey 7B 87.3/71.5, 72B 87.2/72.4.

**6. Ablations.** No numeric data-diversity ablation in the text we read.

---

## 4. GUI-Owl / Mobile-Agent-v3 (Alibaba) — 7B / 32B — arXiv 2508.15144

**1. Data.** Self-Evolving GUI Trajectory Production on Alibaba cloud phones/computers (§2.2): query generation
(mobile: **human-annotated DAG of app navigation flows** + slot-values + LLM instruction synthesis), rollouts by
GUI-Owl / Mobile-Agent-v3, correctness judging, and query-specific guidance for hard queries. Grounding: public
**UI-Vision** and **GUI-R1**, a11y-tree synthesis, SAM-segmented dense grounding on crawled PC screenshots.
Planning knowledge distilled from historical trajectories and Qwen3-235B; action-semantics data from pre/post
screenshots (predict the action, then describe it and its effect). **AC, AITW, GUI-Odyssey, AMEX are not named as
training data.** Counts and mix ratios: not reported. AW apps/tasks in training: not stated.

**2. Observation.** Screenshot; "we typically retain only the most recent **1 to 3 images**" (§2.1). Model emits
reasoning, then a **conclusion**; **only the conclusion is stored in history**. Fig. 9 (OSWorld): more history
images → steadily better (no numbers in text). Resolution: not reported (32k context of Qwen2.5-VL).

**3. Actions** (Table 9, mobile): key, click, long_press, swipe, type, answer, system_button, **open**, wait,
terminate. Pixel coordinates; Qwen function-calling format.

**4. Stages.** Continual pre-training on Qwen2.5-VL → iterative SFT on offline reasoning data (offline hint-guided
rejection sampling; distillation from Mobile-Agent-v3; iterative online rejection sampling with critic filter,
thought–action consistency check, **task re-weighting (down-sample high-success, up-sample low-success tasks)**,
positive/negative re-balancing) → online RL "TRPO": trajectory reward 1/0, format penalty −0.5, trajectory-level
normalized advantage on all steps, success-replay when a group fails entirely, long trajectories split into
single-step instances scaled by length. Parallel env count: not reported.

**5. Numbers.** AW (Table 7): GUI-Owl-7B **66.4**; Mobile-Agent-v3 (framework with GUI-Owl-32B) 73.3; GUI-Owl-32B
end-to-end AW: not in Table 7. Inside Mobile-Agent-E (Table 8): 7B 59.5, 32B 62.1. AC-High (Table 6): 7B 72.8,
32B 76.6. Per-stage AW: figure only.

**6. Ablations.** Fig. 10: adding hint-guided rejection sampling, multi-agent distillation, then general
reasoning SFT each raises AW from an early checkpoint (no numbers in text). RL filtering: DAPO-style online
filtering ~31.5 vs offline filtering ~29.1 on OSWorld.

---

## 5. UI-TARS family (ByteDance Seed)

**UI-TARS v1 (arXiv 2501.12326; 2B/7B/72B on Qwen2-VL).**
- Data: ~50B tokens total. Grounding (Table 2): open-source web 14.8M elements, **mobile 2.5M**, desktop 1.1M,
  from SeeClick, GUIAct, MultiUI, Rico-SCA, WidgetCaption, MUG, Rico Icon, CLAY, UIBERT, OmniACT, AutoGUI, OS-Atlas
  + own parsed screenshots. **Open-source traces: MM-Mind2Web, GUIAct, AITW, AITZ, AndroidControl, GUI-Odyssey,
  AMEX** → mobile **145k traces, avg 9.6 steps**; web 6.4k traces; own annotated traces avg 14.9 steps (count
  starred/undisclosed). ~6M GUI tutorials (avg 510 text tokens, 3.3 images) from MINT/OmniCorpus for reasoning.
- Thoughts added by ActRe (thought conditioned on GT action) and thought bootstrapping (sample, keep pairs whose
  action = GT); vanilla traces without thoughts are also kept in training.
- Observation: screenshot; input limited to the **last N observations** (N not given numerically in the text we
  read); full action+thought history kept as text. Terminal actions Finished(), CallUser().
- Stages: continual pre-training → annealing → **DPO** on online-bootstrapped error→correction pairs ("reflective
  tuning": SFT on corrected steps only; DPO on erroneous-vs-corrected).
- AW: UI-TARS-72B-DPO 46.6 (as quoted in GUI-Owl Table 7). §5 Fig. 8: on in-domain AC/GUI-Odyssey with N=1
  sample, System-2 (thought) is **slightly worse** than System-1.
**UI-TARS-1.5.** No report. README: reasoning via RL; 7B open-sourced; AW 64.2 (model size of that run not stated).
**UI-TARS-2 (arXiv 2509.02544; Seed-thinking-1.6 init, 532M ViT, MoE 23B active / 230B total).** Data flywheel:
CT on tutorials/videos/in-house data (cold start includes **all UI-TARS and UI-TARS-1.5 training data**) and SFT;
samples scored V(s)∈{0,1}, V=1 → SFT, V=0 → CT. In-situ think-aloud annotation and interactive (model-proposes,
human-overrides) annotation; no counts. Named open datasets: none. RL: PPO with decoupled/length-adaptive GAE and
value pretraining; RL domains GUI-Browsing / GUI-General (690 websites) / games; "several thousand" VM instances.
**AW 64.6 → 73.3 after RL, described as transfer to a domain "not the primary focus of training"** (§3.2).
AC: not reported. Observation format of the GUI policy: not explicitly stated (ORM sees full text history + last
five screenshots).

---

## 6. AutoGLM / AutoGLM-Phone-9B (Zhipu / Z.AI) — arXiv 2411.00820 + HF card

The AutoGLM report gives no dataset names or counts. Techniques named: pre-training on weakly supervised
decision data, behavior cloning, curriculum, **intermediate interface separating planner (`do(action, element_description)`) from grounder (coordinates)**, and self-evolving online curriculum RL (WebRL). Reported:
AndroidLab 36.2, 89.7% on Chinese-app tasks (human eval). AutoGLM-Phone-9B card: architecture identical to
GLM-4.1V-9B-Thinking; **no training data, input format, action space or benchmark numbers**. INFERRED: the
disclosed recipe from the same lab for a GLM-4.1V-9B mobile agent is MobileRL-9B (next section).

---

## 7. Step-GUI (StepFun) — 4B / 8B — arXiv 2512.15431v2

**1. Data** (Table 1, samples):

| Type | Mid-train | Cold-start |
|---|---|---|
| General multimodal | 1.9M | 284K (17%) |
| Grounding | 2.7M | 122K (7%) |
| Knowledge | 2.0M | **864K (52%)** |
| Action alignment | 170K | – |
| Trajectory | 4.0M | 404K (24%) |
| Environment-specific | 420K | – |
| Total | ~11.2M | ~1.67M |

Grounding from "open-source corpora" (unnamed). Cold-start knowledge data is **error-driven**: failures in
rollouts are diagnosed for the missing knowledge (UI semantics, app behavior, domain facts) and converted into VQA
samples. CSRS (Calibrated Step Reward System) turns model rollouts into training data via trajectory-level
verification + LLM knowledge extraction (claimed >90% annotation accuracy, 10–100× cheaper than step labels).
Named mobile datasets (AC/AITW/GUI-Odyssey): not reported. AW in training: not stated.

**2. Observation.** Screenshot + instruction + "historical interaction logs"; "purely through visual
understanding without relying on structured accessibility information". Past-screenshot count, resolution: not
reported. **3. Actions:** AWAKE, CLICK, COMPLETE, INFO, LONGPRESS, SLIDE, TYPE, WAIT (Fig. 5).

**4. Stages.** Mid-train → cold-start SFT → RLVR (GRPO with gradient preservation, **semi-online hints**: failed
rollout groups get ground-truth hints injected for a second pass, dynamic ε_high). Self-evolving loop over ≥6
rounds. Env count: not reported.

**5. Numbers.** AW (Table 4): **8B pass@1 67.7, pass@3 80.2; 4B pass@1 63.9, pass@3 75.8.** The headline 80.2 is
pass@3 (up to 3 attempts per task, justified as excluding infrastructure failures) — not comparable to pass@1
numbers. Per round (§5.3.1): round 2 44.83% → round 3 73.40% ("phase transition" once CSRS starts capturing
verified long-horizon successes). AC: not reported.

**6. Ablations.** No data ablation; RL-strategy entropy curves only.

---

## 8. MobileRL (Tsinghua / Z.AI) — Qwen2.5-VL-7B → MobileRL-7B; GLM-4.1V-9B-Base → MobileRL-9B — arXiv 2509.18119

**1. Data — the most explicit recipe in this set.**
SFT (Table 7, steps):

| Stage | AC (Low) | AC (High) | Human annotation (AndroidLab protocol) | Total |
|---|---|---|---|---|
| Reasoning-free SFT | 21.3k | 14.3k | 62.2k | 97.9k |
| Reasoning SFT | 7.2k | 4.1k | 12.2k | 23.6k (§3.1) |

AC = 35.6k of 97.9k reasoning-free steps = **36% (ours)**; the AC train split has 74,714 samples (App. D.5.1), so a
subset was used (selection rule not stated). Human demos supplemented by self-exploration rollouts keeping correct
action sequences (App. D.2). **RL tasks: 2,000 AndroidWorld tasks + 1,103 AndroidLab tasks** (App. D.1). AW tasks
are generated from **AW's own templates by re-randomizing substitutable components and initial device states**;
"we remove all tasks whose specific content and initial states coincide with those in the test set" (App. D.4.1)
— i.e. **same templates, different parameters; disclosed.** MobileRL-9B used "online RL purely on the
AndroidWorld environment" (App. E). AndroidLab tasks lack rule checks → VLM reward model (86% accuracy on 1,000
labelled traces). AW+AndroidLab mixing helped the Qwen backbone, AW-only was better for GLM (no numbers).

**2. Observation.** **Screenshot + compressed XML UI hierarchy** (off-screen and layout-only nodes removed,
attributes shortened, bounds kept; App. C). "The agent observes only the current state" (§2, MDP); no history
mechanism is described for AW (for AC-High eval they keep a history with corrected actions, App. D.5). Full
resolution, max pixels 5,000,000 / min 65,536 (Table 4); a 500k-pixel cap hurt GLM but not Qwen (Table 6).

**3. Actions** (Table 5, merged AndroidLab + AC space): Tap(element=bbox), Type, Swipe(direction, dist[, element]),
Long Press(element), **Launch(app)**, Back, Home, Wait, Finish(message). Targets are **XML-aligned bounding boxes**,
not free coordinates (Appendix notes this fails on apps with poor XML, e.g. TikTok).

**4. Stages.** Reasoning-free SFT (2 epochs, LR 1e-5→1e-6) → reasoning SFT (bootstrap: instruct model samples
reasoning–action pairs, keep those whose action = expert action; iterate) → **AdaGRPO**: SPA (reward rescaled by
shortest successful length), AdaPR (replay buffer 256, top κ=0.25), FCF (tasks failing 2 consecutive epochs get a
3-epoch cooldown, weight exp(−f), then removed), n=16 samples/task, max 50 turns, KL 0.001, LR 1e-6, batch 256.
Claimed >1,000 environments; App. E: stably ~256 parallel rollouts, >10 min per training step.

**5. Per-stage results (Fig. 3a):**

| Backbone | Base | +Reasoning-free SFT | +Reasoning SFT | +AdaGRPO |
|---|---|---|---|---|
| Qwen2.5-VL-7B (AW / AndroidLab) | 27.6 / 10.1 | **50.2** / 36.9 | 56.8 / 38.7 | **72.0** / 42.5 |
| GLM-4.1V-9B-Base (AW / AndroidLab) | 7.7 / 10.1 | **48.1** / 42.7 | 66.2 / 45.0 | **80.2** / 53.6 |

AC (Table 2; type / grounding / SR): 9B w/o RL Low 98.6/95.8/95.9, High 89.8/78.1/**77.5**; w/ RL High 87.0/74.5/73.9
— **AW-focused RL lowered AC-High by 3.6 SR (ours)**; 7B High 72.8 → 71.3. AC was not in the RL stage.

**6. Ablations.** AdaGRPO components (Fig. 3c, 3 runs): full 71.1; w/o AdaPR 63.6; w/o SPA 69.1; w/o FCF 64.8; w/o
AdaPR&SPA 58.5; w/o AdaGRPO 56.8. Removing reasoning-free SFT "consistently" lowers AW (numbers not legible in the
HTML). No AC-removal ablation.

---

## 9. V-Droid (verifier-driven, 8B) — arXiv 2503.15937

**1. Data.** Task instructions from **AndroidControl and AITW** (`Li et al. 2024a; Rawles et al. 2023`) plus
LLM-synthesized tasks from App Store descriptions (manually verified); executed on an emulator cluster; steps
labelled by **human–agent joint annotation** (§5). Data pairs per round: **9K (human only, ~4 min/task, ~5 h
total) → 27K → 55K → 110K**; >90 apps in 7 categories. Verifier entropy (median threshold) flags steps for human
review; annotation accuracy 0.51 → 0.76 → 0.71 (Table 1). AndroidLab and MobileAgentBench apps excluded from
training; **AndroidWorld is called the "in-the-domain test set"** (§6.1.1) — whether AW tasks/templates were
used is not stated. Self-correction pairs (reverse the last action after an error state) ≈ **2.5%** of the set;
"training with excessive self-correcting pairs would lead to collapsed performance by continuing output the
navigating back action" (§4.2).

**2. Observation.** **Text only**: streamlined XML from the accessibility service; no screenshots. Working memory
= action history + per-step reflection-style summary (generated by GPT-4 in the reported system, §6.4).

**3. Action space.** Candidate actions extracted from UI elements (click, long-press, scroll, type, clear) +
defaults (open app, wait, home, back, complete, answer); ~50.3 candidates per step. Each candidate is scored in a
batched **prefill-only** pass ("Yes" logit → MLP score), argmax executed. Open-app / type / answer need a second
LLM call for the argument (12.4% of actions; 25.5% on calendar tasks).

**4. Training.** Llama-3.1-8B-Instruct 4-bit + 1 MLP layer, Q-LoRA rank 16, LR 1e-4→1e-6, 20 epochs at iteration 4
(110K pairs, 90 h on 16×A100-40G). Loss: pairwise process preference `−log σ(τ(pos) − τ(neg))` (P³).

**5. Numbers.** AW **59.5**, AndroidLab 38.3, MobileAgentBench 49. Decision latency 0.7 s, 4.3 s/step.

**6. Ablations — the clearest data-quantity curve in this set.** AW SR **15.0 → 37.5 → 47.4 → 59.5** as pairs grow
9K → (27K) → (55K) → 110K (§6.3; the paper does not label each point explicitly, but four points for four
rounds). Self-correction pairs: 52.2 → 59.5. **Working memory: LLM summary 59.5 / rule-based 46.1 / action
history only 40.0** (Table 4). Training method (Table 3): selector-SFT 35.8, generator-SFT 27.4, untrained
Llama-8B judge 0, GPT-4 judge 34.5.

---

## 10. UI-Voyager (Tencent Hunyuan) — 4B — arXiv 2603.24533

**1. Data.** "we follow MobileRL and employ training sets from AndroidWorld, consisting over **7000 tasks**"
(§4.1), generated from AW's randomized initialization. A seed-task generator perturbs template parameters
(temporal constraints, quantities, file entities). First-iteration trajectories from several Qwen3-VL scales;
later iterations from the previous model; **no human annotation**. Overlap with the 116 test instances: not
discussed. Evaluation: average over **64 runs with randomized task parameters**. **Training and test are the same
116 AW templates — disclosed, but not framed as a contamination caveat.** (MobileRL said 2,000 tasks; the
"over 7000" figure is UI-Voyager's own.)

**2. Observation.** Screenshots + instruction; history window h over past actions/observations (h not given);
resolution not reported. **3. Actions:** AW's native space (click, long_press, swipe, open_app, input_text,
keyboard_enter, navigate_back, navigate_home, wait, status, answer).

**4. Stages.** Qwen3-VL-4B-Instruct → **Rejection Fine-Tuning** (4 rounds; rule-based verifier; new tasks each
round; round-3 checkpoint used) → **GRSD**: within a rollout group, find "fork points" (same screen by SSIM on
cropped grayscale thumbnails, mean-hash pre-filter 0.80) where a failed trajectory took a different action than
the shortest successful one; SFT the failed prompt (with its own history) onto the successful action. Pure SFT
objective, no policy gradient.

**5. Numbers.** RFT: Pass@1 **37% → 73%** over three iterations (round 3 = 73.2). From 73.2: GRPO/PPO plateau
~76; **GRSD 81.0**. Direct GRPO/PPO from the base model: marginal gains. AC: not reported.

**6. Ablations.** RL-vs-GRSD only; no data quantity/diversity ablation.

---

## (a) Which datasets each model used (training; Y = stated, — = not stated, ✗ = stated not used)

| Model | AC | AITW | AITZ | GUI-Odyssey | AMEX | Mind2Web | Open grounding sets | Own human-annotated mobile traj. | Model rollouts + verifier/judge | AW-template tasks | Own RL env / task pool |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MAI-UI | — (eval only) | — | — | — (eval only) | — | — | JEDI, OS-Atlas | Y | Y | — (AW apps in released prompt) | Y, 35+ apps, ≤512 envs |
| UI-Venus-1.5 | — | Y (mid-train) | — | — | — | Y | ShowUI + "30+ sources" | Y (verification) | Y (>30k) | — | Y, 100s–1000s devices |
| UI-Venus v1 | **Y 84k** | Y 371k | Y 14k | Y 114k | Y 39k | — | 5 sets → 107k | Y 20k (Chinese apps) | Y | — | cloud phones (data gen) |
| GUI-Owl / MA-v3 | — | — | — | — | — | — | UI-Vision, GUI-R1 | Y (DAG + guidance) | Y | — | Y |
| UI-TARS v1 | Y | Y | Y | Y | Y | Y (MM-Mind2Web) | 12 sets | Y | Y (bootstrapping) | — | Y (VMs) |
| UI-TARS-2 | via v1/1.5 data | via v1 | via v1 | via v1 | via v1 | via v1 | via v1 | Y (think-aloud) | Y (flywheel) | — (AW = transfer) | Y (thousands of VMs) |
| Step-GUI | — | — | — | — | — | — | unnamed open | — | Y (CSRS) | — | — |
| MobileRL | **Y 35.6k steps** | — | — | — | — | — | — | Y 62.2k steps | Y (self-exploration) | **Y 2,000** | Y (AW + AndroidLab 1,103) |
| V-Droid | Y (task instructions) | Y (task instructions) | — | — | — | — | — | Y (joint) | Y (agent annotation) | — (AW = "in-domain test") | emulator cluster |
| UI-Voyager | — | — | — | — | — | — | — | ✗ ("without manual annotation") | Y (RFT) | **Y >7,000** | AW only |

AutoGLM-Phone-9B, UI-TARS-1.5: data not disclosed. MobileUse: training-free.

## (b) The five most common ingredients

1. **Own-environment model rollouts filtered by a verifier or judge** (rejection sampling / "self-evolving" /
   flywheel) — every trained model in this set with a disclosed recipe (V-Droid's form is agent-annotated,
   human-corrected steps). Typical effect sizes reported: UI-Voyager RFT 37 → 73 (4B); Step-GUI round 2→3 44.83 → 73.40; UI-Venus-1.5
   pipeline yield 17.9% → >70%.
2. **Human-annotated trajectories on live emulators/devices**, increasingly *model-proposed, human-corrected*
   (MAI-UI, MobileRL 62.2k steps, UI-TARS/-2, UI-Venus 20k, V-Droid 9K→110K, GUI-Owl DAGs).
3. **Large GUI grounding corpora** (SeeClick/ShowUI/OS-Atlas/JEDI/UI-Vision/GUI-R1/OmniAct…), usually heavily
   cleaned (UI-Venus: ~40% noisy; MAI-UI: 23.3% bad instructions).
4. **Online RL with outcome reward on an own task pool** (GRPO variants, success replay, difficulty curricula,
   hundreds of parallel emulators) — MAI-UI, UI-Venus-1.5, MobileRL, GUI-Owl, Step-GUI, UI-TARS-2.
5. **Public offline mobile trajectories (AndroidControl, AITW, GUI-Odyssey, AITZ, AMEX)** — used by UI-TARS,
   UI-Venus v1, MobileRL, UI-Venus-1.5 (AITW), V-Droid (as task sources). Where shares are given, AC is a
   minority: MobileRL 36% of reasoning-free SFT steps (ours); UI-Venus v1 84k of a 791k raw pool (ours). The newer
   reports (MAI-UI, GUI-Owl, Step-GUI) do not name it as training data at all.

Runner-up: reasoning traces (thoughts/conclusions) — every generative model, but MobileRL shows reasoning-free SFT
alone already gives 48–50 AW, and UI-TARS found System-2 slightly worse than System-1 on in-domain AC at N=1.

**No paper in this set ablates removing AndroidControl, or reports AW as a function of AC volume.** Data-quantity
evidence is limited to V-Droid's 9K→110K curve and the self-evolving round curves.

## (c) Lessons for a ~2B single-pass choice model (goal + screenshot + text history + a11y element list; mostly AC today)

1. **The data that moves AndroidWorld is in-environment, step-labelled, and verified — not more AndroidControl.**
   The two most comparable recipes both reach ~48–60 AW *without RL*: MobileRL's reasoning-free SFT (36% AC + 64%
   AndroidLab-protocol human demos, screenshot + XML, no reasoning) gives 50.2 (7B) / 48.1 (9B); V-Droid's
   prefill-only candidate scorer reaches 59.5 on 110K pairs collected *on an emulator cluster*. Neither is
   AC-only. Priority: build an emulator-side label stream, not a bigger AC mix.
2. **V-Droid is the architectural twin and its curve is the most actionable number here.** Candidate actions from
   the a11y tree + one scoring pass + pairwise preference loss, AW 15.0 → 59.5 as labelled pairs went 9K → 110K.
   For us: train the element/action head with **pairwise (positive vs. sibling candidate) process labels** on
   the same screen, collected by our model and corrected by a human only where its own score entropy is high
   (their entropy flag reached 0.76 annotation accuracy). Cap self-correction (undo/back) pairs at a few percent
   (2.5% in V-Droid) or the model collapses to `navigate_back`.
3. **History representation is worth ~20 points in a choice model.** V-Droid: action history only 40.0 vs
   rule-based memory 46.1 vs LLM-summarized memory 59.5. GUI-Owl stores only a per-step *conclusion*; UI-Venus
   stores thought–action pairs and re-aligns them to the current model each epoch. Our history is bare action
   text. A zero-token option (INFERRED): record a deterministic **effect** per step (screen changed / unchanged,
   new activity, field now contains X) next to each action — this attacks our measured repeat-on-unchanged-screen
   and no-effect failures without any text generation.
4. **Past screenshots are optional when the element list is present.** MobileRL (screenshot + XML, current state
   only) is the best ≤9B pass@1 result with a disclosed recipe (80.2); screenshot-only models keep 1–3 past images
   (GUI-Owl; MAI-UI code default 3). Keep our single current screenshot; spend capacity on history text instead.
5. **Rejection fine-tuning works for any policy that can be rolled out, including a classifier.** UI-Voyager's
   RFT (37 → 73 for 4B, no humans, rule-verified AW-template tasks) and GRSD (73 → 81) are both *pure SFT* on
   (state, chosen action) pairs. GRSD's fork points give exactly the label a decision head needs: "on this
   screen, the successful sibling rollout chose element k". This is the cheapest bridge from offline SFT to
   online improvement and needs no policy-gradient infrastructure. Generation of the RFT task pool from AW
   templates is the norm (MobileRL 2,000; UI-Voyager >7,000).
6. **Disclose template overlap explicitly.** MobileRL and UI-Voyager train on AW's own 116 templates with new
   parameters and say so; MAI-UI ships the AW app list in its prompt and does not say whether RL used AW tasks;
   UI-Venus-1.5 / Step-GUI / GUI-Owl are silent. Any number we report must state which of three regimes it is:
   (i) no AW templates/apps seen, (ii) AW apps seen, (iii) AW templates seen with re-randomized parameters — and
   it may only be compared to papers in the same regime (AGENTS.md M4).
7. **Answer and termination must be in the training distribution.** All models have explicit
   terminate/status + answer (+ ask_user). UI-Venus's **Trace Reconstruction** — inserting an answer step before
   `Finished` on information-retrieval traces — is a cheap fix for AC's lack of an answer action (relevant to the
   25/116 AW tasks scored only via `answer`). For a zero-token model the answer must be a *read slot* over screen
   text (span/element choice), not generated text.
8. **Rebalance rare actions and hard tasks.** UI-Venus oversamples long-tail actions; GUI-Owl down-samples
   high-success tasks and up-samples low-success ones; MAI-UI/MobileRL curricula shift sampling toward
   0–25%-pass tasks. Our `type_text` is chosen in ~2% of AW decisions — the same long-tail symptom.
9. **Clean the labels before scaling them.** ~40% (UI-Venus) and 23.3% (MAI-UI) of open grounding data is noisy;
   UI-Venus-1.5's judge-score-rewrite loop moved high-fidelity share 69.7% → 89.7%; automatic cleaning with open
   models *failed* for UI-Venus v1 grounding (they reverted to manual). AC-Curated has its own published defect
   rates; treat any AC-only gain with that in mind.
10. **What RL buys at 2B is uncertain and expensive.** MAI-UI-2B: SFT 45.1 → RL 49.1 (+4.0) with up to 512
    environments; UI-Venus-1.5-2B: online RL +14.5 (45.3 → 59.8), then the merge cost 4.2. MobileRL shows RL can
    *lower* AC (9B AC-High 77.5 → 73.9). For a 2B on one GPU, items 1–5 (verified in-environment SFT/RFT data,
    pairwise labels, history effects) are better value than standing up online RL.
11. **Resolution is a cheap knob, measured.** MAI-UI: 720p ≈ 1080p (−50% step time), 540p degrades; MobileRL: a
    500k-pixel cap hurt GLM, not Qwen. Check our serving resolution against 720p before blaming grounding.

---

### Sources (all retrieved 2026-11-11)
- MAI-UI — arXiv 2512.22047v1 (https://arxiv.org/html/2512.22047v1); code https://github.com/Tongyi-MAI/MAI-UI (`MAI-UI/src/prompt.py`, `MAI-UI/src/mai_naivigation_agent.py`). Local verbatim capture of Table 9: `references/raw/maiui2026-androidworld-table9.txt`.
- UI-Venus-1.5 — arXiv 2602.09082v1. UI-Venus v1 — arXiv 2508.10833.
- GUI-Owl / Mobile-Agent-v3 — arXiv 2508.15144.
- UI-TARS — arXiv 2501.12326; UI-TARS-2 — arXiv 2509.02544; UI-TARS-1.5 — https://github.com/bytedance/UI-TARS README.
- AutoGLM — arXiv 2411.00820; AutoGLM-Phone-9B — https://huggingface.co/zai-org/AutoGLM-Phone-9B.
- Step-GUI — arXiv 2512.15431v2. MobileRL — arXiv 2509.18119. V-Droid — arXiv 2503.15937.
- UI-Voyager — arXiv 2603.24533. MobileUse — arXiv 2507.16853.
- Hammer-UI, MobileUse-v2: no primary source found (web searches 2026-11-11).

Ledger note: these arXiv IDs are not yet all in `references/ledger.md`; add entries before citing any figure
from this file in the paper (AGENTS.md §4 backfill rule).
