# T2 — Published AndroidWorld numbers and their stated protocol (mechanical corpus)

Extraction only. Every cell below comes from a page fetched during this task; nothing is inferred,
derived, or converted. `NOT STATED` means the fetched text does not say. No per-step accuracy is
computed from any task-success figure and no formula is inverted anywhere in this file.

Sources fetched: `https://arxiv.org/abs/<id>` for all eight papers, and `https://arxiv.org/html/<id>`
where the abstract lacked the protocol detail (all eight HTML pages returned HTTP 200).

## Corpus table

| Paper (arXiv id) | Model + param count | AndroidWorld success % | Task count stated | Step budget stated | Seeds / runs stated | Harness or eval code stated | Backbone family | URL |
|---|---|---|---|---|---|---|---|---|
| MAI-UI (2512.22047) | MAI-UI-2B (2B) | 49.1 | 116 tasks across 20 mobile applications | NOT STATED (only training-side `max_env_steps` 15/30/50) | NOT STATED | NOT STATED for the eval harness; project repo `https://github.com/Tongyi-MAI/MAI-UI` printed in Fig. 1 | Qwen3-VL | https://arxiv.org/html/2512.22047 |
| MAI-UI (2512.22047) | MAI-UI-8B (8B) | 70.7 | 116 tasks across 20 mobile applications | NOT STATED | NOT STATED | NOT STATED for the eval harness | Qwen3-VL | https://arxiv.org/html/2512.22047 |
| MAI-UI (2512.22047) | MAI-UI-32B (32B) | 73.3 | 116 tasks across 20 mobile applications | NOT STATED | NOT STATED | NOT STATED for the eval harness | Qwen3-VL | https://arxiv.org/html/2512.22047 |
| MAI-UI (2512.22047) | MAI-UI-235B-A22B (235B) | 76.7 | 116 tasks across 20 mobile applications | NOT STATED | NOT STATED | NOT STATED for the eval harness | Qwen3-VL | https://arxiv.org/html/2512.22047 |
| MobileRL (2509.18119) | MobileRL-7B (7B; Qwen2.5-VL-7B-Instruct) | 72.0 | 116 tasks across 20 apps | NOT STATED for evaluation (the "100-step budget" in the text refers to RL training) | Table 1: NOT STATED. Fig. 3 ablation: "All test set results are averaged over three runs". Fig. 5: "eight test trials at temperature 1.0" for pass@k | AndroidWorld environment with adapted action space (App. D.4.1); framework open-sourced at `https://github.com/THUDM/MobileRL` | Qwen2.5-VL | https://arxiv.org/html/2509.18119 |
| MobileRL (2509.18119) | MobileRL-9B (9B; GLM-4.1V-9B-Base) | 80.2 | 116 tasks across 20 apps | NOT STATED for evaluation | same as above | same as above | GLM-4.1V | https://arxiv.org/html/2509.18119 |
| GUI-Libra (2602.22190) | GUI-Libra-3B (3B; Qwen2.5-VL-3B) | 25.2 | 116 tasks stated; "we report results on the remaining 115 tasks" (Task #82 `SimpleSmsReplyMostRecent` cannot be initialized) | 20 steps ("maximum horizon of 20 steps"; Table 5 caption: "in 20 steps") | "we report the average across four runs for all experiments, with the temperature set to 0.0 and top_p to 1.0" (stated in the WebArena-Lite-v2 setup paragraph) | "the official Docker environment"; "the benchmark's rule-based completion checker"; See-Act-V framework following UGround, step-wise reflection/summary module removed | Qwen2.5-VL | https://arxiv.org/html/2602.22190 |
| GUI-Libra (2602.22190) | GUI-Libra-4B (4B; Qwen3-VL-4B) | 42.6 | 116 stated / 115 evaluated | 20 steps | same | same | Qwen3-VL | https://arxiv.org/html/2602.22190 |
| GUI-Libra (2602.22190) | GUI-Libra-7B (7B; Qwen2.5-VL-7B) | 29.6 | 116 stated / 115 evaluated | 20 steps | same | same | Qwen2.5-VL | https://arxiv.org/html/2602.22190 |
| GUI-Libra (2602.22190) | GUI-Libra-8B (8B; Qwen3-VL-8B) | 42.6 | 116 stated / 115 evaluated | 20 steps | same | same | Qwen3-VL | https://arxiv.org/html/2602.22190 |
| AndroidWorld original (2405.14573) | M3A + GPT-4 Turbo, a11y tree | 30.6 | 116 programmatic tasks across 20 real-world Android apps | "Each task has a maximum allowed number of steps (detailed in Appendix F), typically set to twice the number of steps needed by human annotators" | "We set the seed to 30 and the temperature to 0"; three-seed robustness run reported separately (27.6, 26.3, 33.2; mean 29.0) | The benchmark itself; code and experiments at `http://github.com/google-research/android_world` | GPT-4 Turbo (closed) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | M3A + GPT-4 Turbo, SoM (screen + a11y tree) | 25.4 | 116 tasks across 20 apps | same | seed 30, temperature 0 | same | GPT-4 Turbo (closed) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | M3A + Gemini 1.5 Pro, a11y tree | 19.4 | 116 tasks across 20 apps | same | seed 30, temperature 0 | same | Gemini 1.5 Pro (closed) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | M3A + Gemini 1.5 Pro, SoM | 22.8 | 116 tasks across 20 apps | same | seed 30, temperature 0 | same | Gemini 1.5 Pro (closed) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | M3A + Gemma 2, a11y tree | 9.5 | 116 tasks across 20 apps | same | seed 30, temperature 0 | same | Gemma 2 27B (open) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | M3A-Simple + GPT-4 Turbo, a11y tree | 19.8 | 116 tasks across 20 apps | same | seed 30, temperature 0 | same | GPT-4 Turbo (closed) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | M3A-Simple + Gemini 1.5 Pro, a11y tree | 14.7 | 116 tasks across 20 apps | same | seed 30, temperature 0 | same | Gemini 1.5 Pro (closed) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | M3A-Simple + Gemma 2, a11y tree | 3.4 | 116 tasks across 20 apps | same | seed 30, temperature 0 | same | Gemma 2 27B (open) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | SeeAct (adapted web agent) + GPT-4 Turbo, SoM | 15.5 | 116 tasks across 20 apps | same | seed 30, temperature 0 | same | GPT-4 Turbo (closed) | https://arxiv.org/html/2405.14573 |
| AndroidWorld original (2405.14573) | Human | 80.0 | 116 tasks across 20 apps | same | NOT APPLICABLE (human baseline; the seed-30/temperature-0 setting applies to the agents) | same | n/a (human) | https://arxiv.org/html/2405.14573 |
| Mem-W (2605.09317) | Mem-W-4B / Mem-W-8B (4B, 8B) | NOT REPORTED — "AndroidWorld" does not occur in the fetched abstract or HTML full text | NOT APPLICABLE (benchmarks used: MMInA, Mind2Web, GUI-Odyssey, AndroidControl-v2) | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE | UI-TARS-1.5-7B, Qwen3-VL-4B, UI-Venus-1.5-8B (frozen backbones) | https://arxiv.org/html/2605.09317 |
| UI-Venus-1.5 (2602.09082) | UI-Venus-1.5-2B (2B) | 55.6 | 116 programmatic tasks across 20 real-world Android applications | NOT STATED | NOT STATED | Repo linked `https://github.com/inclusionAI/UI-Venus`; the AndroidWorld eval harness itself NOT STATED (page says "based on a live Android emulator") | Qwen3-VL | https://arxiv.org/html/2602.09082 |
| UI-Venus-1.5 (2602.09082) | UI-Venus-1.5-8B (8B) | 73.7 | 116 tasks across 20 apps | NOT STATED | NOT STATED | same as above | Qwen3-VL | https://arxiv.org/html/2602.09082 |
| UI-Venus-1.5 (2602.09082) | UI-Venus-1.5-30B-A3B (30B) | 77.6 | 116 tasks across 20 apps | NOT STATED | NOT STATED | same as above | Qwen3-VL (MoE) | https://arxiv.org/html/2602.09082 |
| Jev-Mobile (2609.30186) | Jev-Mobile (param count NOT STATED; VLM = Qwen3.8-Max, executor = Jev decision model) | 79% (Table 1 prints 0.79) | NOT STATED — "the full AndroidWorld task suite"; no task count printed | NOT STATED — "Wall time, actions, calls, waits, and retries share an episode budget" (no number) | NOT STATED | "using its task initialization and terminal evaluators (Rawles et al., 2025)" — i.e. AndroidWorld's own initializers/evaluators | Qwen3.8-Max (VLM) + Jev typed decision model; UI-TARS-1.5-7B as SeeAct-V grounder | https://arxiv.org/html/2609.30186 |
| GUI-JEPA (2610.36877) | GUI-JEPA (0.50B total; ≈132M trained) | NOT REPORTED — "AndroidWorld" does not occur in the fetched abstract or HTML full text; paper evaluates AndroidControl and ScreenSpot-Pro | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE | NOT APPLICABLE (`https://github.com/bcwang-sjtu/GUI-JEPA`) | ViT-L/16 JEPA encoder + CLIP text encoder | https://arxiv.org/html/2610.36877 |

## Papers that report more than one model size

### MAI-UI — 2512.22047 — https://arxiv.org/html/2512.22047
Main table (Table 6, printed values):

| Model | Paras. | Success Rate |
|---|---|---|
| MAI-UI-2B | 2B | 49.1 |
| MAI-UI-8B | 8B | 70.7 |
| MAI-UI-32B | 32B | 73.3 |
| MAI-UI-235B-A22B | 235B | 76.7 |

Same paper, Table 9 (SFT vs RL, printed values): MAI-UI-2B-SFT 45.1; MAI-UI-2B-RL 49.1 (+4.0);
MAI-UI-8B-SFT 64.7; MAI-UI-8B-RL 70.7 (+6.0); MAI-UI-32B-SFT 69.8; MAI-UI-32B-RL 73.3 (+3.5).
Table 9 RL ablation on 8B: MAI-UI-8B-SFT 64.7; GRPO (max_env_steps=50) 66.5 (+1.8);
Our method (max_env_steps=15) 66.4 (+1.7); Our method (max_env_steps=30) 68.5 (+3.8);
Our method (max_env_steps=50) 70.7 (+6.0).

### MobileRL — 2509.18119 — https://arxiv.org/html/2509.18119
| Model | #Params | AndroidWorld |
|---|---|---|
| MobileRL w/ Qwen2.5-VL-7B-Instruct | 7B | 72.0 |
| MobileRL w/ GLM-4.1V-9B-Base | 9B | 80.2 |

Stage-wise ablation (Table 3(a), Qwen2.5-VL-7B): Qwen2.5-VL-7B-Instruct 27.6; + Reasoning-Free SFT 50.2;
+ Reasoning SFT 56.8; + AdaGRPO (MobileRL-7B) 72.0. GLM-4.1V-9B-Base: 7.7; + Reasoning-Free SFT 48.1;
+ Reasoning SFT 66.2; + AdaGRPO (MobileRL-9B) 80.2. Ablation Table 3(c), AndroidWorld test: MobileRL 71.1;
w/o AdaPR 63.6; w/o SPA 69.1; w/o AdaPR&SPA 58.5; w/o FCF 64.8; w/o AdaGRPO 56.8.
Table 6: AdaGRPO (MobileRL-9B) 80.2 (original images) / 75.8 (compressed images).

### GUI-Libra — 2602.22190 — https://arxiv.org/html/2602.22190
Table 5(a), Native Models (printed values):

| Model | Acc. |
|---|---|
| Qwen2.5-VL-3B (Baseline) | 3.5 |
| GUI-Libra-3B (Ours) | 25.2 |
| Qwen2.5-VL-7B (Baseline) | 7.8 |
| GUI-Libra-7B (Ours) | 29.6 |
| Qwen3-VL-4B (Baseline) | 27.0 |
| GUI-Libra-4B (Ours) | 42.6 |
| Qwen3-VL-8B (Baseline) | 30.4 |
| GUI-Libra-8B (Ours) | 42.6 |

### AndroidWorld original — 2405.14573 — https://arxiv.org/html/2405.14573
Table 3, SR AndroidWorld (printed values): Human 80.0; SeeAct (SoM, GPT-4 Turbo) 15.5;
M3A-Simple (a11y, Gemma 2) 3.4; M3A-Simple (a11y, Gemini 1.5 Pro) 14.7; M3A-Simple (a11y, GPT-4 Turbo) 19.8;
M3A (a11y, Gemma 2) 9.5; M3A (a11y, Gemini 1.5 Pro) 19.4; M3A (SoM, Gemini 1.5 Pro) 22.8;
M3A (a11y, GPT-4 Turbo) 30.6; M3A (SoM, GPT-4 Turbo) 25.4.

### UI-Venus-1.5 — 2602.09082 — https://arxiv.org/html/2602.09082
Table 2, `Ours` (printed values): UI-Venus-1.5-2B, 2B, 55.6; UI-Venus-1.5-8B, 8B, 73.7;
UI-Venus-1.5-30B-A3B, 30B, 77.6. Table 7 ablation prints the final AndroidWorld column as
55.6 (−4.2), 73.7 (+1.0) and 77.6 (+2.1) for the 2B, 8B and 30B-A3B rows respectively; the surrounding
text states "the 2B model showing a remarkable +14.5% absolute gain" from Online-RL, and that the
30B-A3B model "achieves 71.0% and 75.5% accuracy on ScreenSpot-Pro and AndroidWorld before Model Merging"
(merged checkpoint prints 69.6 / 77.6).

### Papers with a single reported configuration
Jev-Mobile (2609.30186) prints one system result, 0.79, alongside SeeAct-V 0.78 and Step-wise VLM 0.84
(Table 1); no size variants. Mem-W (2605.09317) and GUI-JEPA (2610.36877) print no AndroidWorld number.

## Not verified

Everything below was searched for and not found in the fetched text. The URL is the page that was checked.

| Field | Status | URL checked |
|---|---|---|
| MAI-UI — AndroidWorld evaluation step budget | NOT STATED (only `max_env_steps` 15/30/50 for RL training ablated in Table 9) | https://arxiv.org/html/2512.22047 |
| MAI-UI — seeds / number of runs behind Table 6 | NOT STATED | https://arxiv.org/html/2512.22047 |
| MAI-UI — named evaluation harness or eval code for AndroidWorld | NOT STATED (only the project repo URL in Fig. 1) | https://arxiv.org/html/2512.22047 |
| MobileRL — AndroidWorld evaluation step budget | NOT STATED ("100-step budget" is an RL-training budget) | https://arxiv.org/html/2509.18119 |
| MobileRL — seeds behind Table 1 | NOT STATED (three-run averaging is stated for the Fig. 3 ablation only) | https://arxiv.org/html/2509.18119 |
| GUI-Libra — whether the "four runs" averaging applies to the AndroidWorld table | Not restated next to Table 5; the sentence sits in the WebArena-Lite-v2 setup paragraph and says "all experiments" | https://arxiv.org/html/2602.22190 |
| GUI-Libra — eval code / repo for AndroidWorld | NOT STATED as a URL in the fetched text (only "official Docker environment" and "rule-based completion checker") | https://arxiv.org/html/2602.22190 |
| AndroidWorld original — numeric per-task step budget | NOT STATED numerically; points to Appendix F ("typically twice the number of steps needed by human annotators") | https://arxiv.org/html/2405.14573 |
| Mem-W — any AndroidWorld number | NOT FOUND: 0 occurrences of "AndroidWorld" in abstract and HTML full text | https://arxiv.org/abs/2605.09317 ; https://arxiv.org/html/2605.09317 |
| Mem-W — task count / step budget / seeds for a mobile online suite | NOT APPLICABLE (no AndroidWorld evaluation) | https://arxiv.org/html/2605.09317 |
| UI-Venus-1.5 — AndroidWorld step budget | NOT STATED | https://arxiv.org/html/2602.09082 |
| UI-Venus-1.5 — seeds / runs | NOT STATED | https://arxiv.org/html/2602.09082 |
| UI-Venus-1.5 — named AndroidWorld eval harness | NOT STATED (repo and model links printed; "based on a live Android emulator") | https://arxiv.org/html/2602.09082 |
| Jev-Mobile — task count | NOT STATED ("full AndroidWorld task suite"; no count printed) | https://arxiv.org/html/2609.30186 |
| Jev-Mobile — numeric step budget | NOT STATED ("share an episode budget", no number) | https://arxiv.org/html/2609.30186 |
| Jev-Mobile — seeds / runs | NOT STATED | https://arxiv.org/html/2609.30186 |
| Jev-Mobile — Jev / system param count | NOT STATED | https://arxiv.org/html/2609.30186 |
| GUI-JEPA — any AndroidWorld number | NOT FOUND: 0 occurrences of "AndroidWorld" in abstract and HTML full text; benchmarks are AndroidControl and ScreenSpot-Pro | https://arxiv.org/abs/2610.36877 ; https://arxiv.org/html/2610.36877 |
| All papers — protocol details not printed in the abstract | Abstracts were fetched for all eight; for every paper above the protocol fields were taken from the HTML full text, so no row rests on abstract-only evidence. GUI-JEPA's and Mem-W's absence of an AndroidWorld number is confirmed on both the abstract and the full text | https://arxiv.org/abs/2512.22047 ; https://arxiv.org/abs/2509.18119 ; https://arxiv.org/abs/2602.22190 ; https://arxiv.org/abs/2405.14573 ; https://arxiv.org/abs/2605.09317 ; https://arxiv.org/abs/2602.09082 ; https://arxiv.org/abs/2609.30186 ; https://arxiv.org/abs/2610.36877 |

No fetch failed: all eight abstract pages and all eight HTML full-text pages returned HTTP 200.
