# Slice A — AndroidWorld leaderboards, aggregators and grey literature (2026-11-11)

Scope: any NON-GENERATIVE decision-model / single-pass / classifier-style agent result on AndroidWorld (116 tasks) that
exists **outside** the arXiv set this repo already knows (MAI-UI, MobileRL, UI-Venus-1.5/2, GUI-Libra, M3A, T3A, SeeAct,
MobileAgentV2, AutoDroid, JEPA-GUI, Jev-Mobile, V-Droid, UI-TARS, OpenMobile, LFM/d1).

All numbers below are labelled **VERIFIED** (read in the source's own text/table this session), **UNVERIFIED** (snippet or
secondary only) or **INFERRED** (my reasoning). No cross-harness SR comparison is made anywhere.

## 0. Verdict

| Claim | Verdict | One line |
|---|---|---|
| **T-A** (matched one-pass vs AR: competence at lower latency, lost beyond a measurable length) | **DUPLICATES** | The *qualitative* result and the *length-stratified* version are both already published on AndroidWorld — but neither source matches model family/size/data, and neither has a zero-planner one-pass policy. A narrow, defensible gap survives. |
| **T-B** (discrete option set + verified-failure masking removes period-1 repeats) | **CONFIRMS-NOVEL** | Nothing in this slice masks a *verified no-effect* action and re-takes the next-best from the model's own distribution. Two near-misses to cite (Jev-Mobile's candidate invalidation; `2610.13092`'s failed-recovery/premature-completion diagnosis). |
| **T-C** (offline confidence fails to flag no-effect actions; env-verified labels restore signal, AUROC ≥ 0.70) | **CONFIRMS-NOVEL** (weak) | No source measures that AUROC. But a non-generative pre/post-screenshot change detector on AndroidWorld **does** exist (VisCritic, P=87.1 / R=83.4 / F1=85.2) and must be cited as the mechanism someone already built. |
| **F10** (has the public leaderboard been checked for a standalone decision-model result?) | **YES — checked, and the answer is zero** | The official sheet holds 47 entries; **not one** is a decision model. Every row is either an LLM/VLM prompted once ("Model") or a multi-call scaffold ("AI agent"). No Jev, no d1, no typed/classifier entry. The one decision-model AndroidWorld result that exists (Jev-Mobile, 0.79) was **never submitted** to the leaderboard. |

Most dangerous source: **arXiv 2610.13092**, *When Faster Decisions Do Not Make Better Agents* — it already publishes the
AndroidWorld step-budget-stratified deficit that T-A proposes to measure.

## 1. Sources read

| id | Title | Venue + year | URL | Citations | What it is |
|---|---|---|---|---|---|
| S1 | AndroidWorld official leaderboard (sheet `AndroidWorld`) | Google Research, live; snapshot read 2026-11-11 | https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?gid=0 | n/a | 47 community-submitted rows + human baseline; self-reported, "no independent verification" |
| S2 | AndroidWorld repo (README, docs/, issues) | GitHub, 1012★ / 207 forks, pushed 2026-11-10 | https://github.com/google-research/android_world | n/a | Submission is by email/issue; no `leaderboard` file; `docs/` contains only `tasks_guide.md` |
| S3 | Papers with Code — AndroidWorld leaderboard | paperswithcode.co, live | https://paperswithcode.co/benchmark/androidworld | n/a | 43 ranked entries, all with paper links |
| S4 | BenchLM — AndroidWorld | BenchLM, updated 2026-11-10 | https://benchlm.ai/benchmarks/androidworld | n/a | 10-model display-only table |
| S5 | SOTA2 (formerly Wizwand) — AndroidWorld hub + buckets | SOTA2, live | https://www.sota2.com/research/dataset/androidworld | n/a | 24 result buckets; the only place PRM/critic/RL-policy buckets appear |
| S6 | Jev-Mobile: Jev as an Executor for Mobile GUI Agents | arXiv 2609.30186, Sep 2026 | https://arxiv.org/abs/2609.30186 | n/a | **Decision model as GUI executor**, AndroidWorld 0.79 |
| S7 | When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows | arXiv 2610.13092, Oct 2026 | https://arxiv.org/abs/2610.13092 | n/a | 5 agents, ALFWorld + AndroidWorld, 4,500 executions; **length-stratified deficit** |
| S8 | VisCritic: Visual State Comparison as Process Reward for GUI Agents | SOTA2 record, 2026 (no arXiv id on page) | https://www.sota2.com/research/paper/6a3b64d8206542251d7c993a | n/a | Siamese ViT + action-aware critic head; non-generative change detector |
| S9 | Open d1: Edge decision models for text, vision, and audio | Liquid AI blog, 2026-10-07 | https://www.liquid.ai/blog/d1-open | n/a | d1-3B / d1-omni-600M; **no AndroidWorld number** |
| S10 | Introducing d1 (with vision) | Liquid AI blog, 2026-10-05 | https://www.liquid.ai/blog/d1-decision-model | n/a | d1 Web Agent demo; **no AndroidWorld number** |
| S11 | AGI-0 product page | AGI, Inc., published 2026-09-25 | https://theagi.company/agi-0 | n/a | "small action model, 2B to 20B+"; claims 116/116 |
| S12 | Taqwright LIME — AndroidWorld runs | Taqwright, updated 2026-08-29 | https://www.taqwright.ai/benchmark | n/a | 97.4% (113/116), best-of-repeated-runs, video per task |
| S13 | Droidrun AndroidWorld benchmark | Droidrun, 2025-10-03 | https://www.droidrun.ai/benchmark | n/a | 91.4% (106/116), GPT-5 + a11y API |
| S14 | ai-agent-book, chapter7/android-world (Chinese) | bojieli/ai-agent-book, GitHub book companion | https://github.com/bojieli/ai-agent-book/blob/main/chapter7/android-world/README.md | n/a | Paired control/treatment AndroidWorld experiments, T3A + local Qwen |
| S15 | Jev-Mobile cost-table critique | Blolabel, 2026-09-27 | https://blolabel.ai/blog/jev-mobile-savings-measured-on-different-tasks | n/a | Third-party audit disputing the cost comparison |
| S16 | AwesomeCUA benchmark comparison | AwesomeCUA, updated 2026-11-10 | https://awesomecua.github.io/compare/benchmarks | n/a | Benchmark *catalogue* (93 entries), not a results leaderboard |
| S17 | AndroidLab (THUDM) | GitHub | https://github.com/THUDM/Android-Lab | n/a | Separate 36-task benchmark; its README table is AndroidLab, not AndroidWorld |

## 2. Findings

### S1 — Official leaderboard: 47 entries, zero decision models (VERIFIED)

Read via the sheet's CSV export. Header row 2: `Rank, Release Date, Result Source, Model Type, Open?, Model Size, Model,
Screen Representation, Success Rate (pass@1), Number of trials, Success Rate (pass@k), Trajectory submissions, Note`.
Row 1 is a disclaimer: *"Community-submitted results. No independent verification is performed. Scores are self-reported
and may be inaccurate."* The sheet defines the two categories at the bottom:

- **Model** = *"A relatively simple prompt involving one LLM / VLM call"*
- **AI agent** = *"A multi-agent architecture involving several LLM calls and a protocol to coordinate the various
  agents or an LLM wrapped into an advanced agent with memory, subgoal planning, etc."*

**There is no third category for a decision head, classifier or non-generative policy.** That is decisive for F10: the
leaderboard's own taxonomy cannot express our artifact.

Table A — full leaderboard roster (VERIFIED, read row by row from the CSV export).

| # | Name | Declared type | Size | SR (pass@1) | trials | Paper? | Link (from sheet) |
|---|---|---|---|---|---|---|---|
| 1 | FluizAI | AI agent (gpt-4o / gpt-5.6-sol) | – | 100 | 1 | no | https://android-world.fluiz.ai/ |
| 1 | Kirk Engine | AI agent (gpt-4o / gpt-5.4) | – | 100 | 1 | no | https://abhishekdh.github.io/kirk-engine/ |
| 1 | MobileRun | AI agent (GPT6 Astra) | – | 100 | 1 | no | trajectories only |
| 1 | AGI-0 | AI agent (AGI-AW) | – | 100 | 1 | no ("upon publication") | trajectories only |
| 1 | agent-qa by Vostride AI | AI agent (gpt-6-sol) | – | 100 | 1 | no | https://github.com/vostride/agent-qa |
| 6 | Artemis | AI agent (Gemini 3.7 Flash + Gemini Robotics ER 2) | – | 99.1 | 1 | no | https://github.com/google/artemis |
| 7 | MobileUseAgent | AI agent (Seed1.8-GUI) | – | 97.4 | 1 | no | – |
| 7 | Finalrun | AI agent (Gemini 3 flash / flash lite) | – | 97.4 | 1 | no | https://github.com/final-run/finalrun-agent |
| 9 | askui AndroidVisionAgent | AI agent (Claude 4.5/4.0 Sonnet) | – | 94.8 | 1 | no | – |
| 9 | AutoDevice | AI agent (gemini 3 pro + sonnet 4.5) | – | 94.8 | 1 | no | https://autodevice.io/benchmark/ |
| 11 | Midscene.js | AI agent (Gemini-3.5-Flash) | – | 93.1 (pass@2 95.69, pass@3 97.41) | 1 | no | https://midscenejs.com/android-world-benchmark-report |
| 12 | mobile-use | AI agent (Llama 4-scout + Gemini 2.5 pro + GPT-5 nano) | – | 91.4 | 1 | no | https://minitap.ai/benchmark |
| 13 | Agent-Visco | AI agent | – | 88.8 | 1 | no | aliyun attachment |
| 14 | Surfer 2 | AI agent (o3 + holo1.5-72b) | – | 87.1 (pass@3 93.1) | 1 | no | https://hcompai.github.io/android-world-traces/ |
| 15 | gbox.ai | AI agent (Sonnet 4.5 + Sonnet 4) | – | 86.2 | 1 | no | https://github.com/babelcloud/android_world_benchmark |
| 16 | AutoGLM-Mobile | Model (prompted VLM) | 9B | 80.2 | 1 | no | checkpoint only |
| 17 | LX-GUIAgent | AI agent | – | 79.3 | 1 | no | trajectories |
| 18 | AgentProg | AI agent (Gemini-2.5-Pro + UI-TARS-1.5) | – | 78.0 | 1 | **yes** 2512.10371 | github.com/MobileLLM/AgentProg |
| 19 | K²-Agent | AI agent (Qwen2.5-VL-72B + 7B) | 72B+7B | 76.7 | 1 | no | github.com/k2-agent/k2-agent |
| 19 | MAI-UI-235B-A22B | Model | 235B | 76.7 | – | yes (known) | – |
| 21 | MobileUse-v2 | AI agent (Hammer-UI-32B) | 32B | 75.0 | 1 | no | checkpoint |
| 22 | Mobile-Agent-v3 | AI agent (GUI-Owl-32B) | 32B | 73.3 | 1 | yes | github.com/X-PLUG/MobileAgent |
| 22 | MAI-UI-32B | Model | 32B | 73.3 | – | yes (known) | – |
| 24 | MAI-UI-8B | Model | 8B | 70.7 | – | yes (known) | – |
| 25 | Gemini 2.5 Computer Use | Model | – | 69.7 | 1 | no | – |
| 26 | JT-GUIAgent-V2 | AI agent | – | 67.2 | 1 | no | – |
| 27 | GUI-Owl-7B | Model | 7B | 66.4 | 1 | yes | github.com/X-PLUG/MobileAgent |
| 28 | UI-Venus-Navi-72B | Model | 72B | 65.9 | 1 | yes (known) | HF inclusionAI/UI-Venus-Navi-72B |
| 29 | MobileUse | AI agent (Qwen2.5-VL-72B) | 72B | 62.9 | 1 | no | – |
| 30 | Seed1.5-VL | Model | 20B | 62.1 | 1 | yes | – |
| 31 | JT-GUIAgent-V1 | AI agent | – | 60.0 | 1 | no | – |
| 32 | V-Droid (Llama8B) | AI agent | 8B | 59.5 | 1 | yes (known) | – |
| 33 | Agent S2 | AI agent | – | 54.3 | 1 | yes | – |
| 34 | UI-Venus-Navi-7B | Model | 7B | 49.1 | 1 | yes (known) | HF inclusionAI/UI-Venus-Navi-7B |
| 34 | MAI-UI-2B | Model | 2B | 49.1 | – | yes (known) | – |
| 36 | GUI-Explorer | AI agent (GPT-4o) | – | 47.4 | 1 | yes | – |
| 37 | AndroidGen | AI agent (GPT-4o) | – | 46.8 | 1 | yes | – |
| 38 | UI-TARS | Model | 72B | 46.6 | 1 | yes (known) | – |
| 39 | Aria-UI | Model (GPT-4o + Aria-UI) | – | 44.8 | 1 | yes | – |
| 40 | ScaleTrack-7B | Model | 8B | 44.0 | 1 | no | – |
| 40 | UGround | Model (GPT-4o + UGround) | – | 44.0 | 1 | yes | – |
| 42 | Mirage-1 | AI agent (GPT-4o) | – | 42.2 | 1 | no | – |
| 43 | Ponder & Press | AI agent (GPT-4o) | – | 34.5 | 1 | no ("not yet open-sourced") | – |
| 44 | AndroidWorld (M3A) | AI agent (GPT-4 Turbo) | – | 30.6 | 1 | yes | – |
| 45 | GUI-Critic-R1 | Model | 7B | 27.6 | 1 | yes | – |
| 45 | EcoAgent | AI agent (GPT-4o + OS-Atlas-Pro 4B + Qwen2-VL-2B) | – | 27.6 | 1 | no | – |
| 47 | InfiGUIAgent | Model | 2B | 9.0 | 1 | yes | – |
| – | OSCAR | AI agent (GPT-4o) | – | – (pass@4 61.6) | 1 | no | – |
| – | Human Performance | – | – | 80.0 | 3 | – | – |

**Null for this slice, stated positively:** no row is a decision model, classifier head, typed-decision service, or
non-generative policy. Every named model is an LLM/VLM; every "AI agent" row names LLMs in its `Model` cell or links to a
repo whose agent wraps one. No entry contains "Jev", "d1", "Liquid", "TypeSafe", "decision", "classifier", or "typed".

**Secondary observation (VERIFIED, matters for our eval design):** 43 of the 47 rows report `Number of trials = 1`; the
other four are the MAI-UI rows, which leave the cell blank. A single
trial per task with self-report and no independent verification is the norm on this leaderboard. Our planned protocol
(seed 30 instances × 3 seeds + paired sign test) is *stricter* than the median entry, which is a defensible claim in
itself and worth one sentence in the paper.

### S2 — Repo surfaces: no leaderboard file, no decision-model issue (VERIFIED)

- Root tree contains no `leaderboard*` file. `docs/` contains exactly one file, `tasks_guide.md`.
- Issues/PRs matching "leaderboard": 7 total. Relevant ones:
  - **#311** (2025-06-26, open, "How to Submit to Leaderboards") — asks the submission procedure; answered only by
    pointing at the spreadsheet. Confirms **there is no formal submission pipeline**.
  - **#356** (2026-01-24, open) — ByteDance asks to submit a **Doubao Visual Model** agent at "~95 points". Unverified,
    not on the sheet, no numbers published. Not usable.
  - **#193** (closed) — UGround author asks for a release-date correction. Structural, not a result.
- Discussions endpoint was rate-limited on the unauthenticated API; the repo's public surfaces (README website, paper,
  tasks, leaderboard) expose no results page of their own — the spreadsheet *is* the results page.
- README confirms the 2024-11-18 change: `max_steps` per task was set to **≈2× the human average completion time**, with
  per-task values in a separate sheet. **INFERRED:** any AndroidWorld SR quoted without saying which step-budget revision
  it used is not comparable to ours. Almost no grey source says.

### S3 — Papers with Code (paperswithcode.co): 43 entries, all generative (VERIFIED)

Read the full ranked table. Entries and SR: Qwen3.8-Max 85.3; Qwen3.8-Flash-Next 84.5; UI-Venus-2-27B 84.0;
PhoneBuddy-4B-Real+Mock 83.2; EAPO (8B) 82.05; Qwen3.8-27B 81.9; UI-Voyager 81.0; OpenMobile-2-27B 80.8;
Holo3.1-35B-A3B 79.3; Xiaomi-GUI-0 (Qwen3-VL-30B, agentic RL) 78.9; UI-Venus-1.5-30B-A3B 77.6; UI-KOBE (Qwen3.5-Plus) 77.6;
MAI-UI-235B-A22B 76.7; GLM-5V-Turbo 75.7; UI-TARS-2 73.3; Mobile-Agent-v3 w/ GUI-Owl-32B 73.3; Holo2-30B-A3B 71.6;
GUI-Owl-1.5-8B-Thinking 71.6; Qwen3.5-9B + PhoneWorld 71.6; Qwen3.5-35B-A3B 71.1; Qwen3.6-27B 70.3; Holo3-35B-A3B 67.2;
ForgeOwl-8B 67.2; GHD (Qwen3-VL-8B) 66.47; Qwen3-VL ft on OpenMobile 64.7; UI-TARS-1.5 64.2; Qwen3-VL-235B-A22B 63.7;
Latent2World (Gemini-3.6-Flash) 63.39; Seed1.5-VL 62.1; GLM-4.6V 57.0; OmegaUse 55.7; Agent S2 54.3;
Qwen3-VL-8B + MobileWorld semantic WM 54.3; LLaDA-UI 53.5; Gemini-2.5-Flash + Code2World 50.9; GPT-4o + Aria-UI 44.8;
SeeAct-V (GPT-4o + UGround-V1-7B) 44.0; UI-Copilot 39.1; GPT-4o + AGUVIS-7B 37.1; Qwen2.5-VL-72B 35.0;
Mimo-VL-7B + WildGUI 31.9; ScaleCUA-32B 30.6; InfiGUIAgent-2B 9.0.

**Null:** no decision model. The single most architecture-adjacent entry is **LLaDA-UI (53.5)** — block-wise *diffusion*
VLM, i.e. non-autoregressive but still token-generating. Worth one clause in related work: non-autoregressive generation
has been tried on AndroidWorld; non-generation has not (on this leaderboard).

Every PWC row has a paper. So PWC cannot supply a paperless result.

**Aggregator hygiene note (VERIFIED):** paperswithcode.com now redirects to a HuggingFace page; the working leaderboard is
paperswithcode.co. BenchLM's AndroidWorld page lists its "Paper" as GLM-5V-Turbo, which is a *model*, not the benchmark
paper — aggregator metadata here is unreliable and must not be cited.

### S4 — BenchLM: 10 entries, all LLM/VLM (VERIFIED)

Qwen3.8-Omni-Flash 87.1; Qwen3.8 Max 85.3; Qwen3.8-Flash-Next 84.5; Qwen3.8-27B 81.9; Qwen3.7 Plus 81.0;
Holo3.1-35B-A3B 79.3; Holo3.1-4B 71.0; Holo3.1-9B 71.0; Qwen3.6-27B 70.3; Claude Opus 4.6 62.0.
Page states AndroidWorld is "excluded from the weighted scoring formula" and is "display-only". **Null.**

### S5 — SOTA2: the only place non-generative *components* appear (VERIFIED)

The dataset hub lists 24 AndroidWorld result buckets. Three matter:

- **Online Reinforcement Learning on AndroidWorld** (9 rows): StainFlow 62.28 (verifier = Qwen3.5-27B) / 60.34
  (Qwen3.5-9B) / **54.31 (verifier = Environment)** / 48.28 (no training); OS-Themis 60.34 / 58.62; ADMIRE 57.76 / 56.90;
  GUI-Critic-R1 56.03. Average steps 18.05–21.73.
  The **Environment-verifier vs model-verifier vs no-training** triple is the closest published thing to our T-B/T-C
  contrast — but the *policy* under training is generative; only the *reward* is environment-based.
- **Action Success Prediction on AndroidWorld (smallest held-out split)** (3 rows): **VisCritic P=87.1 / R=83.4 / F1=85.2**
  (base Qwen2.5-VL, visual); BacktrackAgent P=75.6 / R=64.3 / F1=69.5 (text); GUI-PRA P=73.2 / R=67.8 / F1=70.4 (text).
- **Reward Modeling on AndroidWorld** (14 rows): top 92.5 precision.

VisCritic (S8) is the one genuinely **non-generative architecture**: *"a Siamese vision transformer to extract
change-aware representations, coupled with an Action-Aware Critic Head that jointly evaluates action success, task
progress, and error type"*, trained *"from existing trajectories without additional human labels"*. That is
"did this action change the screen", learned without human labels — precisely the signal T-B says a screen+action check
must supply. Its AndroidWorld *acting* score is 29.8 overall task success (S5 bucket "Mobile UI Control") and 87.1
precision on the split above.

**INFERRED:** none of these is a decision *policy*; they are critics/PRMs bolted onto generative agents. T-B's masking
rule is still unclaimed. But T-C can no longer claim to be the first to argue that pre/post visual change is the right
label source — VisCritic's title is literally "Visual State Comparison as Process Reward".

### S6 — Jev-Mobile (arXiv 2609.30186): the decision-model AndroidWorld precedent (VERIFIED, primary table)

Architecture: *"the VLM specifies local goals, the accessibility tree defines a structured executable action space, and
Jev, a fast typed decision model, repeatedly selects actions within this space."* Shared VLM across all three arms:
**Qwen3.8-Max**. SeeAct-V's grounder is disclosed as a **UI-TARS-1.5-7B** substitution for UGround.

Table 1 of the paper, read directly from the HTML (not the blog):

| Metric | Step-wise VLM | SeeAct-V | Jev-Mobile |
|---|---|---|---|
| Full-task success rate | 0.84 | 0.78 | **0.79** |
| Mean total time per success (s) | 197.21 | 162.63 | **132.67** |
| Mean executor time per success (s) | 94.30 | 12.37 | **5.16** |
| Executor cost, successful trajectories summed (USD) | 0.694774 | 0.030579 | 0.013091 |
| Mean model API cost per success (USD) | 0.273694 | 0.193207 | **0.072744** |

Step budget: **not stated**. Harness: "full AndroidWorld suite" with official initialization and terminal evaluators.

Reading this honestly: it is the same *shape* of result T-A proposes — decision model keeps success close to a grounded
VLM agent at a third of the wall time and a quarter of the API cost. **But it cannot be our T-A**, for three reasons the
paper itself states:
1. it has a **VLM planner** delegating local goals — our design has no planner and zero generated tokens;
2. it is a **system comparison**, not a matched one — the authors say there is *"no ablation that replaces Jev with a
   small VLM inside the same delegated workflow"*, so the decision-model contribution is unisolated;
3. **no length/budget stratification** — no curve of SR against task length, so "loses it beyond a measurable task
   length" is absent.
Also note the entry is **not on S1**, and the third-party audit S15 disputes the cost table (claims baseline cost rows
disagree 38-fold across different task subsets). Treat the −73.4% cost figure as **contested**.

### S7 — arXiv 2610.13092: the most dangerous source (VERIFIED, primary text)

Design: five agent families, two benchmarks, three runs each, **4,500 task executions**, paired task-cluster bootstrap
(20,000 replicates, n=116 AndroidWorld tasks). AndroidWorld configurations: **T3A**, **AutoDroid**, **Droidrun
FastAgent** — each baseline (B) vs Jev-integrated (J). Text-only only (Jev is text-only).

Verified numbers from the results section:

- **"On AndroidWorld, Jev trails T3A, AutoDroid, and FastAgent by 13.22, 2.30, and 18.10 pp, respectively, with both
  AutoDroid variants below 20% success."**
- **"On AndroidWorld, T3A and FastAgent lose 12.5 and 15.3 pp even in the shortest-budget stratum, with larger deficits
  in the longest-budget stratum."**
- Figure 2(b) is titled: *"Jev–baseline success-rate differences (pp) by AndroidWorld predefined step budget"*.
- The 95% task-cluster bootstrap intervals for the J–B success difference are **entirely negative** for T3A and FastAgent.
- Cost: T3A −17.1%, AutoDroid −68.8%, FastAgent −43.0%. **Time: T3A −10.6%, AutoDroid −5.7%, FastAgent +0.7%
  (i.e. it got slower).** So the task-level latency win is *not* uniform even in their own data.
- Failure coding (Table 3, 251 codable executions): failed recovery 73 (29.1%); ineffective search 62 (24.7%);
  target identification 42 (16.7%); prerequisite violation 27 (10.8%); premature completion 14 (5.6%). Among 57 Jev
  executions in persistent-loss pairs, **40 contain failed recovery** vs 3 of 57 baseline executions.

Why this is dangerous, stated precisely:
- It is the **length-stratified AndroidWorld deficit** that T-A proposes to measure, already published, with a
  step-budget axis and negative confidence intervals. If T-A is framed as "one pass keeps short-task competence and
  loses it past a length", this paper is the prior report of exactly that shape on this benchmark.
- Its diagnosis (failed recovery, premature completion, "verifying completion", "decision models trained for the
  consequences of sequential choices") is the same remedy space T-B and T-C occupy.

Why it does **not** kill T-A outright, stated precisely:
- Jev is a **pretrained third-party decision model dropped into frontier-LLM agents**; the baseline decision-maker is a
  frontier LLM of unknown size. It is **not** a matched family/size/data comparison, and there is no one-pass policy
  trained by imitation learning on AndroidControl.
- Task length is proxied by *predefined step budget*, not by the policy's own trajectory length. There is no
  latency-vs-length curve for the decision model itself.
- Its baselines also use a *different* input modality constraint (text-only) than the mobile SOTA, so its absolute SRs
  (~<20% for AutoDroid) sit far below the leaderboard's 70–85% band. **Do not put these SRs next to S1/S3 numbers.**

### S8 — VisCritic (SOTA2 record; no arXiv id visible on the page) (VERIFIED as page text, UNVERIFIED as a paper)

Quoted mechanism: Siamese ViT on (pre, post) screenshot pairs → change-aware representation → action-aware critic head
predicting action success, task progress, error type; training samples weakly supervised from existing trajectories with
no human labels. Related-benchmark rows on the record: WebArena 29.4 SR; AndroidWorld Mobile UI Control 29.8; AITW
offline action consistency 74.2; AndroidWorld action-success precision 87.1.

Labels: the architecture and the 87.1/83.4/85.2 split are **VERIFIED** as text on the SOTA2 page. That a peer-reviewed
paper exists is **UNVERIFIED** — the SOTA2 record carries a title, author ("Jiachen Qian"), year and abstract but no
arXiv id or venue. Do not cite it as a published paper until the venue is found; cite it as a record.

### S9/S10 — Liquid AI d1: the architectural precedent, with no AndroidWorld number (VERIFIED)

Both blog posts read in full, including the benchmark tables and the methodology note.

- `d1-3B` is trained from **LFM2.5-VL-3B** (weights averaged with LFM2.5-2.6B); `d1-omni-600M` from
  LFM2.5-Encoder-350M. Both *"don't produce tokens. Instead, they produce an answer in a single forward pass."*
- Benchmarks reported are text-only: SQuAD 2.0, Civil Comments, MASSIVE intent, PubMedQA, BoolQ, XNLI, PAWS-X — d1-3B
  mean **82.9**, d1-omni-600M **78.4** vs Decider 2B 77.1 / Decider 4B 81.1. Decision Index v0.2.1: d1-3B **48.57**.
  *"The Decision Index v0.3 includes a private vision split, which we do not report on in this release."*
- **The agentic claim is present but web-only and unquantified**: *"**Web Agent:** d1 operates a flight-search website
  from a one-sentence goal. At each step, it picks the next action among everything the page allows."* No success rate,
  no baseline, no latency-vs-steps figure, no AndroidWorld.
- Latency is reported per *request* only (8 ms RTX 4090, 16 ms Jetson Thor, 30 ms M5 Pro), including a 3.4K-token state
  (102 ms on 4090, 640 ms on M5 Pro) and a 384 px image (17 ms / 62 ms).

**Grep-level null, stated exactly:** the strings "AndroidWorld", "Android", "mobile", "GUI" do not appear in either d1
post. The vendor that makes our base model has published the one-pass decision *architecture* and a web agent demo, and
has **not** published any mobile/GUI benchmark.

This is the source that most changes our framing: **the architecture is not novel and cannot be claimed.** What is
unclaimed is the measurement — decision model as a mobile GUI agent, on AndroidWorld, against a matched generative
baseline, with a length axis.

### S11 — AGI-0: "small action model", 116/116, no paper (VERIFIED from page + S1 row)

Product page text: *"AGI-0. Our small model, built for action. AGI-0 sees the screen, reasons, and acts across any app.
Trained in our lab, from 2B to 20B+ parameters"*; capabilities: *"Faster — Small models respond in a fraction of the time
of frontier-scale APIs"*; *"100% task success on AndroidWorld, 116 of 116 tasks, listed on the official leaderboard.
Published Sep 25, 2026."* Leaderboard note (S1): *"[09/2026]: 97.4 → 100. Trained open source models using our novel
training technique on AndroidWorld tasks. Code and model will be open-sourced upon publication."*

Assessment: **generative** ("reasons", "breaks a request into steps and plans across apps"), so not our architecture. But
two things must be recorded:
1. The phrase **"action model" at 2B–20B** is already taken by a vendor claiming 100% on this benchmark. Our positioning
   must distinguish "decision/one-pass" from "action model", or we inherit their baggage.
2. That 100% is **trained on AndroidWorld tasks** and the leaderboard does not verify. **INFERRED:** it is not usable as
   a comparison point, and a 100/116 with `trials = 1` on a benchmark the model trained on should be cited, if at all,
   as an example of leaderboard contamination — not as a ceiling.

(Note: the blog URL quoted in search results, `theagi.company/blog/our-approach-to-androidworld`, returns HTTP 404;
the claims above are from the live `/agi-0` page and the S1 row, both VERIFIED.)

### S12 — Taqwright LIME 97.4%: a self-auditing grey entry (VERIFIED)

113/116, emulator `sdk_gphone64_arm64`, Android 13/API 33, updated 2026-08-29. Agent type: LLM-driven UI-test agent
(not a decision model). Critically, the page's own methodology note says:

> *"This is the best result per task across repeated runs, not a single clean pass of the suite… The figure on first
> attempt was materially lower. We do not publish a first-attempt rate."*

and it discloses that one task (`ClockStopWatchRunning`) is *"recorded on our own authority rather than the benchmark's"*
because the a11y tree does not repaint for a running stopwatch. **Not on S1.** This is the best example in the whole
slice of why grey numbers cannot be compared to ours: 97.4% here is a best-of-repeated-runs maximum with one manual
override, while our protocol is 3 seeds × 30 instances on a fixed harness.

### S13 — Droidrun 91.4% (VERIFIED)

*"91.4% success rate across 116 diverse tasks… using GPT-5 and direct Accessibility API access"*, 106/116, dated
03.10.2025, per-task logs and pages published, links back to the official sheet. Generative (GPT-5). Same nominal figure
as the sheet's `mobile-use` row (91.4) but a different system — **do not merge these two 91.4s.**

### S14 — Chinese-language grey literature: the paired design exists, the policy is generative (VERIFIED)

`bojieli/ai-agent-book`, `chapter7/android-world` — a Chinese book's companion repo with a real controlled-experiment
runner over AndroidWorld. Verified numbers, quoted from the README:

- **H1** (T3A baseline prompt vs + Wi-Fi guidance and final-state-confirmation guidance), 4 paired tasks: both arms
  **1/4** success, mean evaluator reward **0.50** each. Latency **233.47 s → 156.98 s**; input+output tokens
  **442,619 → 210,039**; **no paired success gain → not promoted.** This is a negative result with a latency/token win
  and no SR win — the *failure mode* of T-A, in Chinese, on this benchmark, as an experiment.
- **H5** (gRPC a11y forwarder vs `A11yMethod.UIAUTOMATOR`): **1/4 → 4/4**, but token ratio **2.498×** > the 1.5× gate →
  not promoted.
- **H5C** (raw UIAutomator vs compact UIAutomator, non-semantic containers filtered): both **4/4**; tokens
  **139,439.5 → 70,557.5 (0.506×)**; latency **101.20 s → 99.18 s (0.980×)**.
- Full campaign: **580/580 unique episodes** (116 tasks × 1–5 rounds), zero runtime errors, five Pixel 6 / API-33
  shards, 24/24 apps at identical versions; **26 strict successes** and 77 evaluator-satisfied final states among 580.
- The repo's promotion gate is explicit: ≥4 complete pairs, net positive success gain, zero paired regressions, and
  latency and tokens each ≤1.5× control. **`decision.deployment_approved` remains `false`.**

Policy: T3A + a locally-run Qwen model → **generative**. Not a decision model. But: this is a Chinese-language grey
source that already runs paired AndroidWorld experiments with a latency/token promotion gate and reports the null. Our
T-A experimental *design* is not novel as a design; the artifact under test is. (One line in the README gives
"总体成功率 ~88%"; I could not establish what that figure refers to, so it is **UNVERIFIED** and is not used above.)

### S15/S16/S17 — Negative surfaces (VERIFIED nulls)

- **S15 Blolabel** audits Jev-Mobile's cost table and argues the baseline rows are drawn from different task subsets
  ("disagree 38-fold"). Secondary source; do not cite its numbers, but it is a reason to quote Jev-Mobile's cost figure
  with a caveat.
- **S16 AwesomeCUA** is a *catalogue of benchmark protocols* (93 benchmarks), not a results leaderboard. No SR table.
  **Null** for our target.
- **S17 AndroidLab** is a separate benchmark (its README table is AndroidLab SR/Sub-SR/RRR/ROR). It reports no
  AndroidWorld numbers. **Null.**

## 3. What this changes for us

1. **T-A cannot be "first measurement of a decision model as an agent."** Jev-Mobile (S6) already put a typed decision
   model in an AndroidWorld executor and got 0.79 vs 0.84 at a third of the wall time. The claim must narrow to the
   matched, planner-free, size-controlled comparison plus the length axis — both of which S6 explicitly does not have
   (*"no ablation that replaces Jev with a small VLM"*; no budget stratification).

2. **T-A's length half is already published as a figure.** S7 Figure 2(b) is "Jev–baseline success-rate differences by
   AndroidWorld predefined step budget", with T3A −12.5 pp and FastAgent −15.3 pp *even in the shortest stratum* and
   larger deficits in the longest. Two consequences:
   - we must cite S7 and state the difference (its baselines are frontier LLMs; ours is a matched generative sibling of
     the same base);
   - S7's own finding complicates our hypothesis: the deficit is *already present on short tasks*, so a clean
     "short-OK / long-fails" story may not replicate. **Recommendation:** pre-register the length split, and be prepared
     to report a uniform deficit — which S7 has already reported and which would make T-A a replication, not a finding.

3. **The latency claim needs care.** S7 measured task-level time changes of −10.6% / −5.7% / **+0.7%**. S6 measured
   −32.7% but only on *successful* trajectories and only with a VLM planner absorbing much of the loop. Our "lower
   end-to-end latency" must be reported per-decision *and* per-task, with failures included, or it will look like a
   selective subset.

4. **Positioning against "action model".** AGI-0 (S11) already markets a 2B–20B "small action model" claiming 116/116.
   We must not use that phrase. "One-pass decision model" / "typed decision" is the vocabulary that is *architecturally*
   accurate and, per S1, absent from the leaderboard.

5. **The official leaderboard is a clean, checkable null.** F10 is answered with a 47-row enumeration. That is worth
   one sentence in the paper's evaluation section: *no decision-model entry appears among 47 leaderboard submissions;
   43 of 47 report a single trial.*

6. **Our protocol is stronger than the norm and we should say so** — 3 seeds × 30 instances + paired sign test vs
   `trials = 1` self-report everywhere, and vs Taqwright's explicit best-of-repeated-runs (S12). This is a cheap,
   defensible contribution to the evaluation section.

7. **T-B: keep, but cite the near-misses.** Jev-Mobile's "candidates are checked against a fresh observation; after one
   action the program rebuilds C_{t+1} and invalidates old IDs" is stale-ID invalidation, not no-effect masking. S7's
   failed-recovery/premature-completion taxonomy (73/251 = 29.1% failed recovery; 40 of 57 persistent-loss Jev runs) is
   the *diagnosis* T-B fixes. Neither masks a verified no-effect action and re-takes the next-best option.

8. **T-C: the "environment-verified labels" idea is partly taken.** VisCritic (S5/S8) builds a non-generative Siamese
   ViT change detector from trajectory weak supervision and reports P=87.1 on a held-out split. If T-C's novelty is "use
   pre/post change as the label", it is gone. If it is "offline-valid confidence fails *online under the policy's own
   actions*, and env-verified labels restore AUROC ≥0.70 on a task split", that specific measurement was not found —
   but VisCritic must be the first citation in that section.

9. **Do not build any comparison table that mixes S1, S3, S6, S7.** Their step budgets, harnesses, trial counts and
   modalities differ (S7 is text-only; S6 substitutes a grounder; S12 is best-of-runs). Rule 3 of this sweep applies to
   our own paper too.

## 4. Searches run (queries + sources searched + nulls)

**Surface 1 — official leaderboard and repo.**
- Fetched the sheet's CSV export (`export?format=csv&gid=0`) and the XLSX to enumerate tabs → 2 tabs:
  `AndroidWorld` and `MobileMiniWoB++`. The MiniWoB tab has 4 obfuscated value-rows and is out of scope.
- GitHub API: `contents/` (root), `contents/docs`, `search/issues?q=repo:google-research/android_world+leaderboard`
  (7 results, all read or triaged), repo stats (1012★, 207 forks, pushed 2026-11-10).
- GitHub GraphQL discussions query → **rate-limited (unauthenticated)**. **Null by unavailability, not by absence** —
  the leaderboard is a spreadsheet, and the repo surfaces no results page, so this does not change F10.
- Web: `google-research.github.io/android_world/` → served a feed document only, no results table.
- **Result: HIT for the surface, NULL for the target.** 47 entries, 0 decision models.

**Surface 2 — mirrors and aggregators.**
- `paperswithcode.com/dataset/androidworld` → 200 but now a HuggingFace papers page (PwC absorbed). 
- `paperswithcode.co/benchmark/androidworld` → 43-entry table, read in full. **NULL for target.**
- `benchlm.ai/benchmarks/androidworld` → 10 entries, read in full. **NULL.**
- `wizwand.com/dataset/androidworld` → redirected to `sota2.com/research/dataset/androidworld`; 24 buckets enumerated,
  3 read in full (online RL, action-success prediction, and the hub itself). **HIT for non-generative *components*,
  NULL for a non-generative policy.**
- `emergentmind.com/topics/androidworld` → topic summary page, no independent result table. Not used.
- HuggingFace API: `spaces?search=androidworld` (0 hits for `androidworld`, 1 unrelated for `android-world`),
  `models?search=androidworld` (3 hits: `zihuanjiang/AndroidWorld-Generalization`, `Ray2333/androidworld`,
  `gavinlaw/qwen3vl4b-androidworld-lora` — all generative or dataset artifacts, 0 likes). **NULL: there is no
  AndroidWorld leaderboard Space on the Hub.**
- `awesomecua.github.io/compare/benchmarks` → 93-benchmark protocol catalogue, no SR table. **NULL.**
- `github.com/THUDM/Android-Lab` README → AndroidLab's own metrics only. **NULL.**
- Non-search (noted, not queried): `multimodalart/jev-decision-index` HF Space is the Decision Index leaderboard, not an
  agent benchmark — it contains no AndroidWorld column.

**Surface 3 — GitHub repos with an AndroidWorld SR and no paper.**
- Query: `GitHub "AndroidWorld" success rate 116 tasks agent README no paper`.
- Repos/pages enumerated and typed: droidrun (91.4, GPT-5), midscene (93.1, Gemini-3.5-Flash), minitap/mobile-use
  (91.4, multi-LLM), finalrun (97.4, Gemini 3), gbox.ai (86.2, Sonnet 4.5), k2-agent (76.7, Qwen2.5-VL), askui (94.8,
  Claude), AutoDevice (94.8), Artemis (99.1, Gemini), Vostride agent-qa (100, gpt-6-sol), FluizAI (100), Kirk Engine
  (100), MobileRun (100), AGI-0 (100), Taqwright LIME (97.4). **All LLM/VLM-backed. NULL for classifier/decision-head.**
- One repo with a genuine paired-experiment harness (S14, Chinese) — generative policy. **NULL for target, HIT for
  methodology prior art.**

**Surface 4 — vendor / company grey literature.**
- Liquid AI `blog/d1-open` and `blog/d1-decision-model` read in full, including both benchmark tables and the
  methodology footnote. **NULL: no AndroidWorld, no mobile, no GUI.** The web-agent demo is qualitative only.
- AGI, Inc. `theagi.company/agi-0` (200) and `/glossary/androidworld` (200); `/blog/our-approach-to-androidworld` → 404.
  Agent type resolved to a generative VLM action model. **NULL for target.**
- Taqwright `taqwright.ai/benchmark` → 97.4% with a best-of-runs protocol. **NULL for target.**
- Droidrun `droidrun.ai/benchmark` → 91.4%, GPT-5. **NULL for target.**
- TypeSafe/Jev: Jev-Mobile is in the known set, so not re-surveyed as novel; the paper (S6) was nevertheless opened for
  its Table 1 because F10 requires knowing whether *any* decision-model result is on the leaderboard. It is not.
- Third-party: Blolabel's audit of Jev-Mobile's cost table (secondary; used only to caveat S6).
- X/Twitter and Reddit: **not searched as primary sources**, per the brief's rule that they count only if they point at
  a repo or table. The Liquid AI X post surfaced by search carried no AndroidWorld figure (UNVERIFIED — snippet only).

**Surface 5 — Chinese-language sources.**
- Query: `知乎 OR CSDN OR 公众号 AndroidWorld 成功率 116 个任务 决策模型 小模型 非生成`.
- Returned: Midscene's Chinese report (same 93.1 as its English page, LLM-driven), `bojieli/ai-agent-book` chapter 7
  (read in full — S14), and Chinese pages of Western repos. **HIT for a Chinese grey source with real AndroidWorld
  paired experiments (generative policy); NULL for a non-generative Chinese result.**
- ModelScope / BAAI model cards: no AndroidWorld-reporting non-generative card surfaced under these queries. **NULL**
  (declared as a query-level null, not an exhaustive ModelScope crawl).

**Surface 6 — preprint servers beyond arXiv.**
- OpenReview API v2 `notes/search` with `term=androidworld` → count 500, 25 inspected: the ICLR 2025 AndroidWorld paper,
  ICLR 2025/2026 official reviews and author comments (e.g. `ICLR.cc/2026/Conference/Submission1442` "General Response
  to Reviewers (Updated Results on AndroidWorld)"), and one ICLR 2027 submission **"Do Mobile Agents Need the Phone
  Screen?"**. No decision-model/non-generative submission.
- `term="decision model GUI agent"`, `term="non-generative GUI agent"` → the API returns `count: 10000` and no titles
  (search index degenerate for these phrases — **null by unusability**, recorded as such).
- `term="typed decision android"` → only **"Typed Decision Models on Tabular Data"** (ACL ARR 2026 October) — a typed
  decision model submission, but tabular, not GUI. **NULL for AndroidWorld.**
- ACL Anthology (needed for non-arXiv venues): surfaced *"Same Tasks, Different Apps"* (EMNLP 2026 Findings),
  *"Agent+P"* (AACL 2026), *"D-Artemis"* (ACL 2026 Findings), *"MobileWorld"* (ACL 2026 long). These are generative
  multi-agent/planning or benchmark-generalization papers; none reports a decision-model AndroidWorld result.
  **Not read in full — recorded as UNVERIFIED and not used for any number.**
- bioRxiv/medRxiv: irrelevant by construction, not queried.

**Explicit statement of the nulls this slice delivers.**
- Searched the official leaderboard (47 rows), paperswithcode.co (43), BenchLM (10), SOTA2 (24 buckets, 3 read), the
  AndroidWorld repo (docs, issues, PRs), HuggingFace Hub (spaces + models), AwesomeCUA (93 benchmarks), AndroidLab, and
  six vendor/repo pages; **found no entry anywhere that is a non-generative decision model acting on AndroidWorld.**
- Searched for the specific claim "decision model on AndroidWorld" and found **exactly two** things: Jev-Mobile
  (0.79 SR, VLM planner, arXiv) and the Jev-drop-in study (arXiv, length-stratified deficits). Neither is on the
  leaderboard and neither is a matched, planner-free, size-controlled comparison.
- The nearest non-generative *component* is VisCritic's Siamese-ViT change detector (P=87.1). **No source reports an
  AUROC for confidence-under-own-action-shift on a task split** — T-C's specific measurement remains unclaimed, but its
  label source is not.
