# Seam E — app choice and element choice (2026-11-11)

Scope: (1) target APP as a classification over launchable apps vs span-copying an app name out of the goal;
(2) target ELEMENT as an id over an accessibility-tree element list vs regressing a pixel box, including list
truncation and unseen apps. Low-visibility sources prioritised. Labels: **VERIFIED** = read in the source's own
text this session; **UNVERIFIED** = snippet only; **INFERRED** = my reasoning.

---

## 0. Verdict

| Claim / question | Verdict | One line |
|---|---|---|
| (a) Is app-name span copying a documented failure mode with a published rate? | **CONFIRMS-NOVEL (the measurement)** | No paper reports a span-copy failure rate. The nearest published number is *app-name generation* accuracy 85.6% (HL) / 88.0% (LL) at 1B on AndroidControl — the **best** of all action arguments, so the failure is real but bounded at ~1 in 7 and not the observed bottleneck. |
| (1) App choice as classification over a launchable-app list | **CONFIRMS-NOVEL (interface)** | No prior work passes a launchable-app list. Official M3A passes **no** app list and emits free-text `app_name` with the guideline "nothing will happen if the app is not installed" — the failure is designed-in, never counted. |
| (b) Does anyone publish list-size vs element-selection accuracy on Android screens? | **CONFIRMS-NOVEL** | No. Five near-misses give *pool-size statistics* and one gives *predictability of hit from pool size* (AUC 0.82–0.88), but no accuracy-vs-N curve, and no recall-of-the-gold-element curve for a capped shortlist. |
| (2) Element as id over an a11y list | **DUPLICATES (the design choice)** + **CONFIRMS-NOVEL (the measurement)** | AndroidControl already states indices and pixel centres work "equally well"; M3A/T3A/SeeAct/AndroidLab/AppVLM/Jev all enumerate elements. Nobody measures the cost of the cap. |
| (3) Candidate-list recall vs length on Android screens (our 93.8% @≤40 / 78.5% @R=450) | **CONFIRMS-NOVEL**, with a flag | No independent number, and no contradiction at equal population. But independent pool statistics (AndroidControl 222.2 elements/screen; 62 median candidates/step) make a 40-cap look more aggressive than 93.8% implies — our population definition has to be published with the number. |
| (c) Any ≤3B result separating seen-app from unseen-app **AndroidWorld SR**? | **CONFIRMS-NOVEL (formal split)** — but **DUPLICATES (the magnitude)** | No formal seen/unseen SR split at ≤3B. AppVLM (3B) gives the equivalent ablation with 3 seeds: AndroidControl-only 11.4% → +AndroidWorld rollouts 37.8% (82 tasks). Offline, AndroidControl's app-unseen step accuracy at PaLM-2S is published. |
| T-A (matched one-pass vs autoregressive) | **SILENT, but pre-empted on one axis** | AppVLM is a 3B, single-call, no-planner **generative** agent (it emits the action string, so it is the autoregressive arm, not a zero-token one) at 37.8% on AndroidWorld with 3 seeds. It is not the matched comparison and makes no latency claim, but it is the *control* arm we planned to run (Q1) — already run, at 3B. |
| T-B (option set + verified-failure masking) | **SILENT** | No source found that masks a screen+action verified to change nothing. Adjacent: AppVLM's data preprocessing **drops steps whose consecutive screenshots are identical** before fine-tuning (a data-side, not inference-side, version of the same idea). |
| T-C (confidence under own-action shift) | **SILENT** | Nothing in this slice. |

---

## 1. Sources read

| id | Title | Venue + year | URL | Citations | What it is |
|---|---|---|---|---|---|
| AW | AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents | ICLR 2025 | https://arxiv.org/html/2405.14573v3 ; code: https://raw.githubusercontent.com/google-research/android_world/main/android_world/agents/m3a.py | not shown | The harness we run; also the source of the M3A scaffold whose element-list interface we are compared against |
| AC | On the Effects of Data Scale on UI Control Agents (AndroidControl) | NeurIPS 2024 D&B | https://arxiv.org/pdf/2406.03679 | not shown | The dataset behind d1; publishes app-unseen / task-unseen / category-unseen splits and per-argument accuracies |
| AppVLM | AppVLM: A Lightweight Vision Language Model for Online App Control | arXiv 2502.06395, Feb 2025 (Huawei Noah's Ark) | https://arxiv.org/html/2502.06395 | not shown | 3B one-pass online AndroidWorld agent; per-source data ablation and app-unseen action accuracy |
| AAB | Same Tasks, Different Apps: Why Mobile GUI Agents Fail to Generalize? (AnyAppBench) | EMNLP 2026 Findings | https://aclanthology.org/2026.findings-emnlp.1909.pdf | not shown | App-identity-controlled live Android benchmark, 52 apps; measures the app-change penalty directly |
| ALab | AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents | ACL 2025 | https://aclanthology.org/2025.acl-long.107.pdf | not shown | 9-app live benchmark; XML (a11y) mode; app always named in the instruction |
| OSG | OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis | ACL 2025 | https://arxiv.org/html/2412.19723v3 | not shown | Synthetic in-domain rollouts; per-source ablation on AndroidWorld at 4B/7B |
| LC | Lexical Coupling in GUI Element Grounding: Sentence Embeddings Track Labels across Mobile and Web | arXiv 2608.21794 (Univ. Helsinki) | https://arxiv.org/pdf/2608.21794 | not shown | Measures same-screen candidate ranking on AndroidControl; gives the real candidate-pool size distribution |
| SIMP | Less is More: Empowering GUI Agent with Context-Aware Simplification (SimpAgent) | arXiv 2507.03730, Jul 2025 | https://arxiv.org/pdf/2507.03730 | not shown | Quantifies element-density harm (pixel masking, not list truncation) |
| JEV | Jev-Mobile: Jev as an Executor for Mobile GUI Agents | arXiv 2609.30186, Sep 2026 | https://arxiv.org/html/2609.30186 | not shown | Typed decision model choosing over program-generated candidate IDs — nearest architectural neighbour |
| MID | Why Act Without Asking? Evaluating Mobile GUI Agents under Ambiguous Instructions (MID-Bench) | EMNLP 2026 Main | https://aclanthology.org/2026.emnlp-main.1236.pdf | not shown | Shows the app name is normally a substring of the instruction, and that "which app?" is a live ambiguity class |
| AMEX | AMEX: Android Multi-annotation Expo Dataset for Mobile GUI Agents | ACL 2025 Findings | https://arxiv.org/pdf/2407.17490 | not shown | Claims a11y support is sparse on many apps; asserts, never measures |
| GH1 | MobileAgent issue #269 (Chinese) | GitHub, 2026-03-16 | https://github.com/X-PLUG/MobileAgent/issues/269 | open issue | "Agent cannot recognise/open system apps (e.g. Phone); after a failed click it tells you to install" |
| GH2 | mobile-use issue #27 (Chinese) | GitHub, 2026-03-16 | https://github.com/MadeAgents/mobile-use/issues/27 | open issue | "Agent cannot correctly identify and open the target app, falls into a repeat loop" |

---

## 2. Findings

### 2.0 The two seam tables (acceptance format)

**Seam 1 — target app.** "App list / truncation" = is a launchable-app list supplied, and how it is bounded.

| Source | Venue/year | Method | App list / truncation | Metric | Number | Harness | Table ref | Contradicts our number? |
|---|---|---|---|---|---|---|---|---|
| AW / M3A (code) | ICLR 2025 | free-text `app_name` in `open_app`, GPT-4 Turbo | **none** — no installed-app list is built or passed; `ui_elements` passed whole | AndroidWorld SR | 30.6 | AndroidWorld 116 tasks, seed 30, per-task step budget | §4.1.1, Table 3; `m3a.py:75-76,101-103` | No. Confirms the interface is unclaimed. |
| AC | NeurIPS 2024 D&B | PaLM-2S LoRA r64, a11y text list, arg prediction | n/a (app name generated, not classified) | arg accuracy given correct type | **85.6 (HL) / 88.0 (LL)** open_app; 93.3 / 98.5 type recall | AndroidControl test splits, offline step match | **Table 10**; Tables 8–9 | **Yes — bounds our failure.** App name is the *easiest* argument; a ~1 in 7 error, not a dominant one. |
| AC | NeurIPS 2024 D&B | same | n/a | step accuracy on app-unseen | 58.5 (HL, −12.3) / 78.5 (LL, −4.7) vs IDD 70.8 / 83.2 | AndroidControl app-unseen (64 apps) | **Table 5**, Table 3 | Partly: unseen apps cost little at low level, a lot at high level. |
| AAB | EMNLP 2026 Findings | 13 agents, smallest 4B; goal + verifier fixed, app varied | none | live Android SR, equal-app-weighted | **38.7 → 27.3 (−11.4 pp, CI −14.1…−9.1)** for 11 instruction-only agents | AnyAppBench, 520 pairs / 52 apps, 3 runs per cell | §4.2, Table 1 | **Sizes our 13-app risk** at the app level, but no ≤3B row. |
| ALab | ACL 2025 | 9 apps, LLM (XML) and LMM (SoM) modes | app **named in instruction** by construction | SR | GPT-4-1106 31.16 (XML) | AndroidLab 138 tasks | §3.2.1, §3.1 | No — sidesteps app grounding entirely. |
| AppVLM | arXiv 2502.06395, 2025 | PaliGemma-3B, one pass, SoM boxes | none | AW SR | **11.4 (AC only) → 37.8 (+AW rollouts)** | AndroidWorld **82/116** tasks, **3 seeds** | Table 2, Table 4 | **Yes — this is the app-coverage ablation we planned to run.** |
| OSG | ACL 2025 | InternVL2-4B / Qwen2-VL-7B, 1K trajectories | none | AW SR | 4.02 → 15.18 (4B); 6.25 → 17.41 (7B) | AndroidWorld **112/116** tasks, single run, no seed stated | Table 1 | Same shape as AppVLM at 4B; single-run so not comparable to a paired test. |
| MID | EMNLP 2026 Main | instruction rewriting pipeline | n/a | rule R3: mask "app-name-leak" | app name is a substring of the instruction for open_app steps | AndroidControl + GUI-Odyssey steps | §4.1, A.4, B | Explains *why* span copy works and why its failure is invisible in these datasets. |
| GH1 / GH2 | GitHub, 2026-03 | practitioner bug reports (Chinese) | n/a | qualitative | app-open failure → repeat loop | — | issues #269 / #27 | No number. Best evidence the failure class is recognised in the field. |

**Seam 2 — target element.**

| Source | Venue/year | Method | List size / truncation | Metric | Number | Harness | Table ref | Contradicts our number? |
|---|---|---|---|---|---|---|---|---|
| AW / M3A | ICLR 2025 | a11y leaf-node list, JSON action referencing mark or element | **no cap, no truncation rule, no ordering rule stated**; prompt shows `... More elements listed ...` | AW SR | 30.6 (GPT-4 Turbo, text-only a11y); 25.4 (SoM) | AW 116 tasks, seed 30 | §4.1.1, D.1 Listing 2, Table 3 | Not comparable; no list-length metric exists. |
| AW / SeeAct-choice | ICLR 2025 | a11y candidates after interactability filter | filter rule only; "majority of screens < **50** candidates" | AW SR | 15.5 | AW 116 tasks, seed 30 | §4.1.2, D.5, Table 3 | **Tension:** their filtered mode is <50, our cap is 40 — the margin is thin. |
| AC | NeurIPS 2024 D&B | a11y flat list, predicts **element centre** | **no cap**: 222.2 elements/screen avg (train), 185–221 per split | step accuracy | best 70.8 (HL) / 83.2 (LL) | AndroidControl, offline | Table 3, §4.1, **C.2** | §C.2 states indices and centres are **equally good** — kills id-vs-box novelty. |
| AC | NeurIPS 2024 D&B | as above | unlocatable targets **discarded** | — | **rate not reported** | AndroidControl | D.1(ii) | **Null for gold-in-list recall.** |
| LC | arXiv 2608.21794 | rank every visible element against the step instruction (BM25 / 5 encoders) | **uncapped**: median **62**/step, max **1,657**; clean-main n=32,646 | R@1; hit predictable from log n_cands | AUC **0.82–0.88**; oracle R@1 0.706 (+0.151 over BM25) | AndroidControl, 58,078 steps | §3.1, §3.6, §4 | **Tension:** median pool 62 > our cap 40, so 93.8% cannot hold on an uncapped visible-element pool. |
| SIMP | arXiv 2507.03730 | pixel masking of irrelevant regions (not list truncation) | AITW 10–100 elements/screen typical | step SR | +2.4 (AITW) / +2.3 (GUI-Odyssey) | AITW, GUI-Odyssey; 2B Qwen2-VL | §1, Table 5, Fig. 8 | Consistent: density hurts. No curve. |
| AppVLM | arXiv 2502.06395 | numbered SoM boxes for every a11y clickable | no cap stated | AC action accuracy | 73.9 IDD / 65.4 app-unseen (3B) | AndroidControl | Table 1, §3.2 | Beats the 1B text-list model on app-unseen → "small model needs the list" is not supported. |
| JEV | arXiv 2609.30186 | typed decision model over program-built a11y candidate IDs, fresh per step | **not stated** | AW SR / executor time | 0.79 SR; 5.16 s per success | AndroidWorld full suite; planner Qwen3.8-Max | §3.2, §6 Table 1, §8 | **Promise without number:** coverage is "measured" in §1, absent from results. |
| ALab | ACL 2025 | "compressed XML" tree, SoM mode for LMMs | no cap or rule stated | SR | GPT-4-1106 31.16 (XML) | AndroidLab 138 tasks | §3.1, B.1–B.4 | Null. |

### 2.1 AW — the M3A interface has no app list and no element cap

**VERIFIED**, https://arxiv.org/html/2405.14573v3 §4.1.1: M3A is given "a list of available action types, guidelines
for operating the phone, and a list of UI elements derived from the Android accessibility tree's **leaf nodes**"; the
text-only variant "consumes the screen represented using the accessibility tree and selects the relevant action in
JSON". Appendix D.1 Listing 2 renders the element prompt as `UIelement0: UIElement(text=..., bbox_pixels=...)` …
`... More elements listed ...` — **no cap, no truncation rule, no ordering statement**. No number of elements is
given in the prompt section.

**VERIFIED**, `m3a.py` (raw URL above): the action doc says
`'- Open an app (nothing will happen if the app is not installed): {"action_type": "open_app", "app_name": <name>}'`
(line 75–76) and the guideline `'- Use the open_app action whenever you want to open an app (nothing will happen if
the app is not installed)'` (line 101–103). No installed-app list is constructed or passed anywhere in the file;
`ui_elements` is passed whole (line 295, `ui_elements=ui_elements if ui_elements else 'Not available'`).

*Consequence (INFERRED):* the official baseline's app step is an open-vocabulary string emission with a silent,
unvalidated failure path — which is exactly the mechanism behind our F7 third failure. It is also why **no published
rate exists**: the harness never observes whether the app opened.

**VERIFIED**, §4.1.2 and Appendix D.5: for the SeeAct baseline they "utilize the accessibility tree to construct
candidate UI actions … after applying a filtering heuristic to remove non-interactable elements, **the majority of
screens contains less than 50 candidate elements**". This is the only statement in AW about candidate count, and it is
a mode, not a distribution, and carries no accuracy curve.

**UNVERIFIED (snippet only)**: the ICLR camera-ready table additionally reports `M3A-Simple, a11y tree, Gemma 2, 9.5`
on AndroidWorld (and GPT-4 Turbo 19.8 / Gemini 1.5 Pro 14.7) — i.e. a ~9B open model at 9.5% with the same a11y-list
interface. Seen only in a search snippet of
https://proceedings.iclr.cc/paper_files/paper/2025/file/01a83bc2f2732a58e6aa731e659e7101-Paper-Conference.pdf ;
model size not stated in the snippet. Do not quote without opening the PDF.

### 2.2 AC — the app argument is the *easiest* argument, and the app-unseen split is published

**VERIFIED**, https://arxiv.org/pdf/2406.03679 **Table 10** (argument accuracy when the action type is correct;
LoRA-tuned PaLM 2S, rank 64, all training data; parameter count not stated):

| argument | high-level instr. | low-level instr. |
|---|---|---|
| click | 76.2 | 85.6 |
| input_text | 75.9 | 85.7 |
| long_press | 60.0 | 66.7 |
| **open_app (app name)** | **85.6** | **88.0** |
| scroll (direction) | 90.6 | 91.0 |

The paper's own reading: "Detecting the name of the target app works well" (Appendix E.1). Action-*type* recall for
`open_app` in the confusion matrices (Tables 8, 9) is **93.3** (HL) / **98.5** (LL). This is the only published
app-name accuracy I found anywhere. It is *generation from a goal that names the app*, not span copying, so it bounds
the failure at ~12–14% at ~1B scale but does not measure our mechanism.

**VERIFIED**, Table 5 (step accuracy, LT-all-r64, PaLM-2S; square brackets are deltas from IDD):

| split | HL | LL |
|---|---|---|
| IDD | 70.8 | 83.2 |
| app-unseen | 58.5 (−12.3) | 78.5 (−4.7) |
| task-unseen | 59.6 (−11.2) | 77.3 (−5.9) |
| category-unseen | 57.4 (−13.4) | 76.8 (−6.4) |

With rank=4 (LT-all) the app-unseen deltas shrink to −6.8 (HL) / −2.1 (LL). **This is the single most useful number
for our Q9 fear**: at the *step* level, an unseen app costs little for low-level instructions and a lot for
high-level ones. Note the accidental collision: `78.5` here is an app-unseen low-level accuracy, unrelated to our
"78.5% at R=450" recall figure. Never place them side by side.

**VERIFIED**, §3.3 / Table 3: train 769 apps, 39 categories, **222.2 UI elements per screen**; IDD 296 apps / 221.5;
app-unseen 64 apps / 185.4; task-unseen 90 / 181.6; category-unseen 68 / 184.7.

**VERIFIED**, Appendix B.4: the 222 → 185 drop between IDD and the OOD splits is **a sampling artifact of how few apps
each OOD split contains**, not a property of unfamiliar apps: "If we remove these outliers by dropping the top 5% of
apps with the most elements per screen, the average number of UI elements across the remaining apps is 180.9." Any
argument of the form "unseen apps have shorter element lists, therefore our cap behaves differently" is contradicted
here.

**VERIFIED**, §4.1 / Appendix C.2: "We found LLMs work equally well with predicting element centers or element
indices, but as the former approach is compatible with click actions that are not restricted to specific UI elements,
our implementation always outputs the center." Direct prior statement that the id-vs-box choice is a wash for a
small model on AndroidControl — this is the citation that kills any novelty claim for the *interface*.

**VERIFIED**, Appendix D.1(ii): steps whose click/long_press has no enclosing accessibility node ("either a touch on
an empty area or a target UI element missing from the accessibility tree") are **discarded**; the discarded count is
never reported. Appendix D.3 relaxes matching to "coordinates inside the ground-truth bounding box".

**VERIFIED**, §3.1: annotators "always include the name of the target app in the task description, unless obvious
(e.g. Google first-party apps such as Cl…". So in AndroidControl the app name is a literal substring of the
instruction in the general case — span copying is a viable strategy *by dataset construction*, and the dataset never
tests its failure.

### 2.3 AppVLM — 3B, one pass, no planner, and the per-source ablation we need

**VERIFIED**, https://arxiv.org/html/2502.06395 §3.2 / §4.1: PaliGemma-3B-896; input = screenshot with numbered
bounding boxes for every clickable a11y element + goal + last five actions as text; **one model call per step and no
planner or reflection pass — but it still *generates* the action** (it is not a zero-token decision model); SFT on
AndroidControl, then rejection-sampled fine-tuning ("RFT") on AndroidWorld's own apps.

**VERIFIED**, Table 1 (AndroidControl action accuracy; 3B vs the PaLM-2S reference):

| agent | input | IDD | Task-Unseen | Cat-Unseen | App-Unseen |
|---|---|---|---|---|---|
| SeeAct (GPT-4o) | screen + tree | 31.5 | 30.7 | 30.6 | 30.9 |
| T3A (GPT-4o) | tree | 56.1 | 55.8 | 56.5 | 54.2 |
| M3A (GPT-4o) | screen + tree | 60.8 | 59.3 | 60.8 | 60.4 |
| Llama-3 8B | tree | 65.5 | 58.7 | 58.3 | 57.1 |
| LT-all-r64 (PaLM-2S) | tree | 70.8 | 59.6 | 57.4 | 58.5 |
| **AppVLM-base 3B** | screen + boxes | **73.9** | 65.9 | 65.1 | **65.4** |
| AppVLM 3B | screen + boxes | 69.0 | 62.7 | 61.9 | 62.2 |

AppVLM-base is +3.1 over the best reported PaLM-2S model in-domain and **+6.9 on app-unseen** (−8.5 vs −12.3). A 3B
model with pixel-anchored numbered boxes generalises to unseen apps *better* than the 1B text-list model — the
opposite of what a "small model needs the list" prior would predict.

**VERIFIED**, Table 2 (AndroidWorld, **82 of 116 tasks** — verification and Q&A tasks omitted; average of **3 seeds**):

| agent | size | avg infer. time (s) | overall SR |
|---|---|---|---|
| SeeAct | GPT-4o | 15.82 | 22.0 |
| T3A | GPT-4o | 4.29 | 41.9 |
| M3A | GPT-4o | 11.42 | 36.6 |
| Llama-3 (AndroidControl only) | 8B | 2.35 | 17.5 |
| InfiGUIAgent (from its paper) | 2B | — | 9.0 |
| **AppVLM-base (AndroidControl only)** | **3B** | 0.91 | **11.4** |
| **AppVLM (AndroidControl + AW rollouts)** | **3B** | 0.91 | **37.8** |

**VERIFIED**, Table 4 (per-source ablation, all at 3B):

| data sources | AndroidControl | AndroidWorld |
|---|---|---|
| AndroidControl + AndroidWorld rollouts (AppVLM) | 69.0 | **37.8** |
| AndroidWorld rollouts only, chained RFT (RFT_4) | 64.3 | 35.0 |
| **AndroidWorld rollouts only, no AndroidControl (AWO)** | 29.8 | **22.4** |

Table 3 gives the RFT trajectory: 17.9 → 23.2 → 30.5 → 35.0 across iterations 1–4.

*Reading (INFERRED):* at 3B, app coverage alone is worth +11.0 over nothing (22.4 vs 11.4) but AndroidControl +
coverage is worth +26.4 (37.8 vs 11.4) — the two sources are **complementary, not substitutes**. This is a published
3B version of our Q9 question, and the answer is that the offline corpus is not made redundant by in-domain rollout
data. It also means the "13 unseen apps" worry is already sized: AndroidControl-only 3B lands at 11.4%, which is
within noise of our own B0.

### 2.4 AAB — the app-identity penalty on live Android, but not at ≤3B

**VERIFIED**, https://aclanthology.org/2026.findings-emnlp.1909.pdf §4.2: 100 task templates, 520 task–application
pairs, 52 apps (11 AndroidWorld originals + 41 new same-category apps), goal and verifier held fixed, only the app
changes. **11 instruction-only agents: 38.7% on the original apps vs 27.3% on new apps, −11.4 pp, cell-level
bootstrap CI [−14.1, −9.1].** With the two hybrid systems, 13 agents: 43.6% → 31.4%.

Table 1 (equal-app-weighted, AW-inherited templates): UI Voyager-4B 53.9 → 28.6 (−25.3); Gemini-3.1-Pro 81.1 → 52.1
(−29.0); GPT-5.1 60.4 → 41.8 (−18.6); Qwen3-VL-8B 20.7 → 17.7 (−3.0); Qwen3-VL-30B-A3B 21.7 → 18.5 (−3.2).
**The smallest agent in the study is 4B** (UI Voyager-4B); there is no ≤3B row.

**VERIFIED**, §4.2: "The large per-app standard deviations (typically 20–37 points) show that this is uneven across
apps, not a uniform shift." §4.5: failure composition shifts with the target app (execution/tooling share 11.3% →
18.3% under the Gemini judge; grounding share 42.2% → 46.7% under Qwen). §6: "AnyAppBench evaluates fixed agents
rather than strict unseen-app generalization, because the training exposure of proprietary and GUI-trained models is
generally unknown."

*So:* a live-Android app-change penalty exists and is large per app (−11.4 pp equal-weighted), but no study does the
seen/unseen split at ≤3B, and AAB explicitly declines to call its result unseen-app generalisation.

### 2.5 ALab — the app is always named, so there is nothing to ground

**VERIFIED**, https://aclanthology.org/2025.acl-long.107.pdf §3.2.1: "To avoid confusion during testing, we specify
the app required to complete the task in natural language" — the app is given, not inferred. The XML mode gives "the
compressed XML" of the screen; no element cap or ordering is stated (§3.1, Appendix B.4). Training data: 10.5k traces
/ 94.3k steps, of which **726 traces / 6,208 steps** come from the nine benchmark apps (Appendix C). No seen/unseen
app split anywhere in the paper.

*Consequence:* AndroidLab cannot answer (c) and its XML interface cannot answer (b). It is a null for both, and its
"app is specified" design is the cleanest statement that the field currently sidesteps app grounding by fiat.

### 2.6 OSG — the in-domain-rollout ablation at 4B/7B on AndroidWorld

**VERIFIED**, https://arxiv.org/html/2412.19723v3 Table 1 (AndroidWorld SR; **112 of 116 tasks** used "due to app
unavailability"; single run, no seeds reported — Appendix A):

| base | zero-shot | task-driven | + self-instruction | OS-Genesis |
|---|---|---|---|---|
| InternVL2-4B | 0.00 | 4.02 | 7.14 | **15.18** |
| Qwen2-VL-7B | 0.89 | 6.25 | 9.82 | **17.41** |
| GPT-4o (M3A) | 23.70 | — | — | — |

Both the task-driven baseline and OS-Genesis train on **1K trajectories**, average mobile trajectory length **5.60
steps** (OS-Genesis) / 5.64 (task-driven) — i.e. ≈5.6k steps, versus Self-Instruction's 1.5K trajectories at 6.4
steps. The gain is therefore attributable to *where* the trajectories came from (the eval apps themselves), not to
volume. This is the closest published quantity to our Q9 plan and it is a clean per-source ablation — but the
comparison is 4B, single-run, and reports no seed, so it cannot be put in a table with our paired sign test.

### 2.7 LC — the only real distribution of Android candidate-pool size

**VERIFIED**, https://arxiv.org/pdf/2608.21794 §3.1: from the AndroidControl accessibility tree they extract "every
visible UI element and resolve the target at the action coordinate, yielding **58,078 grounded steps and 4.56M total
candidates (median 62 per step, max 1,657)**"; clean-main subset (click/long_press whose target has visible text or
content_description) **n = 32,646**. Mind2Web: "mean 531, median 410, max 4,301 candidates per step".

**VERIFIED**, §3.6/§4: "Encoder hit@1 outcomes are also predictable from lexical rank, label type, and
**candidate-pool size**, with AUCs of **0.82–0.88**" — i.e. pool size is a *known* confound in element selection, but
it enters as a regression covariate, never as a curve. Target recovery R@1 for BM25 on clean-main is reported, and
the oracle over five encoders + BM25 + Jaccard reaches R@1 = 0.706 (+0.151 over BM25) — these are ranking metrics
over the *uncapped* pool, not recall of a capped shortlist, and the paper states no rate at which the gold element is
absent from the tree.

*This is the strongest available independent statistic for our (3)*: the median AndroidControl screen has 62 candidate
elements, more than our 40-cap. That does not contradict our 93.8% (different population: AndroidWorld screens,
interactable-only, reading order, app list stripped), but it means the 93.8% is only credible if the filter is
published with it.

### 2.8 SIMP — element density hurts, measured by masking not by truncation

**VERIFIED**, https://arxiv.org/pdf/2507.03730 §1/§5: "the number of elements per screenshot can reach dozens or even
hundreds"; AITW distribution "typically falls within the range of 10 to 100" (Appendix C, Fig. 8). Masking irrelevant
regions during training improves step SR by **+2.4 (AITW)** and **+2.3 (GUI-Odyssey)**, with −27% inference FLOPs
(§1, Table 5). The mechanism is pixel masking of irrelevant *regions*, not dropping list entries, and no
accuracy-vs-N curve is given.

### 2.9 JEV — the nearest duplicate of "typed decision over candidate IDs"

**VERIFIED**, https://arxiv.org/html/2609.30186 §1: "the VLM specifies local goals, the accessibility tree defines a
structured executable action space, and Jev, a fast typed decision model, repeatedly selects actions within this
space"; §3.2: candidates are enumerated from the raw tree (visible, enabled, valid bounds), bound to local IDs, and
invalidated after each action; §5.1: planner = Qwen3.8-Max, grounder variant = UI-TARS-1.5-7B for SeeAct-V.

**VERIFIED**, §1: "We therefore measure candidate coverage and handoff separately from success" and "We distinguish
coverage (is an acceptable candidate available?), selection (does Jev choose it?), and handoff". **But §6 Table 1
reports only SR / time / cost**, and §7 discusses only those. **No coverage number is published anywhere in the
paper.** §8 Limitations: "a visually apparent target absent from the tree cannot be turned into an executable
candidate by the current controller."

**VERIFIED**, §6 Table 1: Jev-Mobile 0.79 full-suite SR, executor time 5.16 s/success vs 94.30 s for the step-wise
VLM and 12.37 s for UI-TARS; mean total time per success 132.67 s vs 197.21 s. §4: "to our knowledge, peer-reviewed
GUI-agent work has not systematically studied whether a typed decision model such as Jev can serve as the execution
model for GUI tasks."

*Consequence:* this is the paper that most directly pre-empts "typed decision model over an a11y candidate list" as a
contribution, and it makes an explicit novelty claim of exactly that kind. Two differentiators survive: (i) it has an
external planner at inference, so it is not a standalone decision model; (ii) it **promises candidate coverage and
never reports it** — the measurement is still open. Also note its own error admission: no list-size or truncation
statement anywhere, so our 40-cap question is untouched.

### 2.10 MID — the app name is normally *in* the instruction; "which app" is an ambiguity class

**VERIFIED**, https://aclanthology.org/2026.emnlp-main.1236.pdf §4.1: construction rule R3 drops any rewritten
instruction containing "the GT **open_app target name (app-name-leak)** as a substring" — i.e. for open_app steps the
app name normally appears verbatim in the instruction and has to be masked to create ambiguity. §4.2/A.4: the strict
user simulator's worked example is `"Which app?" → "Gmail"`, and the ambiguity taxonomy includes `target: the UI
element/app/file/item to act on is missing`. Appendix B: GUI manipulation actions (click, type, swipe,
system_button) absorb >98.5% of emissions; `call_user` pools to 4/3600 = 0.111%.

*Consequence for (a):* the field's own construction pipeline confirms the app name is a substring of the goal in the
general case, which is why span copying "works" — and why its failure is invisible in every dataset built this way.
It also confirms that when the app is *not* named, the correct behaviour is a clarification, not a guess.

### 2.11 AMEX — the a11y-sparsity claim is asserted, never measured

**VERIFIED**, https://arxiv.org/pdf/2407.17490 §2: "many apps don't support accessibility information or only provide
very little of that" — offered as the motivation for a pure-vision model, with no citation and no number. AMEX's own
element annotations are parsed from XML then human-filtered for misaligned boxes (§3.3). No coverage figure.

### 2.12 Grey literature — the failure our F7 describes is reported by users, unquantified

**VERIFIED**, https://github.com/X-PLUG/MobileAgent/issues/269 (open, 2026-03-16, Chinese): the agent cannot
recognise and open system apps such as Phone; after a failed click it tells the user to install the app.
**VERIFIED**, https://github.com/MadeAgents/mobile-use/issues/27 (open, 2026-03-16, Chinese): "Agent 无法正确识别并打开
目标应用，陷入重复循环" — cannot correctly identify and open the target app, falls into a repeat loop.

Both are exactly our mechanism (app choice failure → no effect → repeat) and neither carries a rate. They are the best
evidence that the failure class is *practitioner-recognised*, and the reason we must be the ones to count it.

---

## 3. What this changes for us

1. **Drop the id-vs-pixel framing entirely.** AndroidControl §C.2 already reports index ≈ centre as equivalent for a
   small model. Presenting element-as-id as a design decision invites the reviewer to say "settled in 2024". The
   only defensible novelty in seam (2) is **the cap and what it costs** — nobody has an accuracy-vs-list-length curve
   and nobody reports candidate-generator recall against a shortlist. Make that the claim, and cite AC §C.2 and LC
   §3.6 as the reason it needed measuring.

2. **App choice: claim the interface, not the failure.** No prior system passes a launchable-app list; M3A's
   open-vocabulary `app_name` with "nothing will happen if the app is not installed" is the status quo. But the
   failure itself is *bounded*, not unknown: 85.6/88.0 argument accuracy and 93.3/98.5 type recall at ~1B
   (AC Table 10). Frame our contribution as **making the failure observable and recoverable** (verified no-op →
   mask → next-best), never as "we discovered app grounding fails". Quote AC Table 10 in the same paragraph or a
   reviewer will.

3. **T-A's control arm has already been run, at 3B, with 3 seeds.** AppVLM: one model call per step, no planner,
   37.8% on 82 AndroidWorld tasks (11.4% from AndroidControl alone). It is generative, so it *is* the control arm,
   not a competing decision model. Our Q1 as written ("fine-tune LFM3-VL-2B to emit the action as text") would
   reproduce a published result unless we change the comparison. Either (i) match AppVLM's protocol and disclose
   82-vs-116 tasks, SoM-vs-list, and 3 seeds, or (ii) restate T-A against AppVLM directly and make the claim about
   the *decision-model* interface at equal data — in which case AppVLM's 11.4% AndroidControl-only is the number to
   beat at equal data, and its 37.8% is the number to beat at equal *total* data.

4. **Q9's premise is confirmed and already sized at 3B.** AppVLM Table 4: AC-only 11.4 → AC+rollouts 37.8; rollouts
   only 22.4. Both sources matter and the offline corpus is not redundant. OS-Genesis Table 1 gives the same shape at
   4B (4.02 → 15.18 at 1K trajectories, ~5.6k steps). We should cite both and stop treating the 13 unseen apps as an
   unquantified risk: the published penalty for the *step*-level app shift is −2.1 to −4.7 pp for low-level
   instructions (AC Table 5), which is smaller than our current diagnosis implies.

5. **Do not attribute element-count differences to app novelty.** AC Appendix B.4 shows the 222 → 185 drop between
   IDD and OOD splits is a sampling artifact (top 5% outlier apps removed → 180.9). If our analysis regresses
   success on list length and app novelty together, it will find a spurious app effect.

6. **The 93.8% needs its population written down.** Independent pool statistics: AndroidControl median 62 candidates
   per step, max 1,657 (LC §3.1); 222.2 elements per screen (AC Table 3); AITW 10–100 (SIMP Fig. 8); AndroidWorld
   SeeAct-filtered "majority < 50" (AW §4.1.2). A 40-cap cannot be 93.8% on a 62-median pool. Publish the exact
   filter (interactable-only? text-bearing? reading order? deduplicated?) with the number, or the number will be read
   as contradicted by LC.

7. **Cite Jev-Mobile and differentiate explicitly.** It is a typed decision model over program-generated a11y
   candidate IDs on AndroidWorld with a published novelty claim for exactly that. Our differentiators: no external
   planner at inference (it has Qwen3.8-Max), and the candidate-coverage/recall measurement it *promises in §1 and
   never reports*. Filling that gap is a defensible contribution; claiming the interface is not.

8. **One free method gift.** AppVLM §3.3.1 drops, before fine-tuning, any step where consecutive screenshots are
   identical. That is our T-B masking idea applied to the *data* rather than the *inference loop*. It strengthens the
   T-B premise (identical screens are recognised as data poison by an independent group) while leaving the
   inference-time masking claim unclaimed. Cite it as convergent motivation, and check whether the same filter
   explains part of any repeat-rate improvement we see.

---

## 4. Searches run (queries + sources searched + nulls)

Sources searched: arXiv (abs + HTML + PDF), ACL Anthology, NeurIPS/ICLR proceedings, Microsoft Research, Hugging
Face, GitHub, Chinese-language web (Tencent Cloud, FoneClaw, Huawei support), Google web.

| # | Query | Result |
|---|---|---|
| 1 | `AndroidWorld M3A agent accessibility tree element index open_app app selection arXiv 2405.14573` | AW + ICLR PDF; found M3A a11y/SoM table (snippet) |
| 2 | `mobile GUI agent "unseen apps" generalization small model 2B success rate separately reported` | **AAB (EMNLP 2026)**, LearnAct, UI-Venus-1.5 |
| 3 | `"number of candidate" elements element selection accuracy GUI agent accessibility tree list length accuracy degrades` | **LC**, UGround, UI-AGILE, Aria-UI, OS-Atlas. No list-length curve. |
| 4 | `recall of accessibility tree "gold element" candidate set Android screens GUI grounding why ground truth element missing` | **JEV**, Screen2AX (macOS a11y completeness), Lexical Coupling. **No source reports the Android gold-element-in-tree rate.** |
| 5 | `AndroidLab arXiv 2410.23218 training apps testing unseen apps six apps three apps generalization` | correct id is 2410.24024; ACL 2025 page + PDF |
| 6 | `mobile GUI agent "app selection" choose target app from instruction classification package name failure` | **MID-Bench**, GH1, anyappbench. No app-selection classifier found. |
| 7 | `AndroidControl accessibility tree element coverage percentage target element present annotation recall AMEX` | AMEX (assertion only), DigiData, GUI-Odyssey |
| 8 | `"app grounding" mobile agent benchmark "which app" intent to application mapping` | nothing using the term. **Null: "app grounding" is not an established term.** |
| 9 | `MobileGUI-VBench vivo capability dimensions "app" selection accuracy report` | BlueLM-GUI vendor benchmark (40 apps, 8 scenario classes); no app-selection metric surfaced |
| 10 | `GUI agent ablation "number of elements" candidate list length accuracy drop 2B model element selection index` | ShowUI (2B grounding), **SIMP**, UI-AGILE, BAMI. No length ablation. |
| 11 | `手机 智能体 打开应用 失败 应用名 无法识别 open_app 错误率 报告` (Chinese) | **GH1, GH2**; Tencent Cloud / FoneClaw practitioner posts; Huawei support page. No quantitative report. |
| 12 | `GUI-Actor coordinate-free grounding attention over candidate elements arXiv 2506 accuracy number of candidates` | GUI-Actor (NeurIPS 2025). Region candidates from pixels, not an a11y list — out of scope for the table. |
| 13 | `"AndroidWorld" "3B" model trained "AndroidControl" success rate unseen apps zero-shot Qwen2-VL 2B` | **AppVLM** (headline hit) |
| 14 | `OS-Genesis synthetic trajectory AndroidWorld ablation with and without synthetic data step counts 2B` | **OSG** ACL 2025 |
| 15 | `percentage target UI element missing from Android accessibility tree cannot be grounded a11y incompleteness measured` | Android platform a11y docs, a11y-toolkit (web). **Null for Android coverage numbers.** |

Nulls, stated plainly:

- **N1.** Searched AW, AC, AAB, ALab, OSG, LC, SIMP, JEV, MID, AMEX, AppVLM and 15 web queries above; found **no
  source that reports the rate at which span-copying the app name from the goal fails**. The nearest measurement is
  app-name *generation* accuracy (AC Table 10: 85.6/88.0).
- **N2.** Found **no source that reports element-selection accuracy as a function of candidate-list length on Android
  screens**, and **no source that reports the truncation rule together with the resulting recall**. AW, ALab, AC and
  JEV either cap nothing or state no rule; AAB and AppVLM give no list.
- **N3.** Found **no source that reports "gold element inside the candidate list" recall** for an Android shortlist.
  AC discards unlocatable steps without a count (D.1(ii)); LC drops unresolved targets without a count; JEV promises
  coverage and does not report it.
- **N4.** Found **no ≤3B study that separates seen-app from unseen-app success on AndroidWorld**. AAB's smallest
  agent is 4B and it explicitly disclaims strict unseen-app generalisation; ALab has no app split; AppVLM's split is
  AndroidControl's offline app-unseen, not an AndroidWorld SR split.
