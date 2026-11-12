# Published AndroidControl (AC) and AndroidWorld (AW) metrics — primary sources only

Retrieved 2026-11-11. Every number below was read from the arXiv HTML/PDF table, the official HF model card, or the official
GitHub README named in the row. Table cells were checked against the parsed source HTML/PDF where possible (UI-TARS,
UItron, InfiGUI-R1, GUI-Owl, UI-Venus, OmegaUse, MAI-UI AW, GUI-SAGE).
Legend: `n/r` = the source reports no number for this cell; `nv` = not verified (source unreadable or internally
inconsistent). "SR" = step success rate unless stated otherwise. Low = high-level goal + per-step low-level instruction; High = goal only.

## 1. AndroidControl

| Model | Size | Low Type | Low Gr | Low SR | High Type | High Gr | High SR | Test set (as stated) | Source |
|---|---|---|---|---|---|---|---|---|---|
| UI-TARS-2B | 2B | 98.1 | 87.3 | 89.3 | 81.2 | 78.4 | 68.9 | "follow the settings and metrics outlined in OS-Atlas"; size not stated | arXiv 2501.12326 Tab. 8 |
| UI-TARS-7B | 7B | 98.0 | 89.3 | 90.8 | 83.7 | 80.5 | 72.5 | same | 2501.12326 Tab. 8 |
| UI-TARS-72B | 72B | 98.1 | 89.9 | 91.3 | 85.2 | 81.5 | 74.7 | same | 2501.12326 Tab. 8 |
| UI-TARS-1.5(-7B) | 7B | n/r | n/r | n/r | n/r | n/r | n/r | README has no AC row | github.com/bytedance/UI-TARS |
| MAI-UI-2B | 2B | n/r | n/r | n/r | n/r | n/r | 67.3 | not stated; metric not named ("AC-High") | arXiv 2512.22047 Tab. 7 |
| MAI-UI-8B | 8B | n/r | n/r | n/r | n/r | n/r | 69.1 | same | 2512.22047 Tab. 7 |
| MAI-UI-32B | 32B | n/r | n/r | n/r | n/r | n/r | 75.5 | same | 2512.22047 Tab. 7 |
| InfiGUI-R1-3B | 3B | 96.0 | 93.2 | 92.1 | 82.7 | 74.4 | 71.1 | "test split"; processed copy released on HF (`InfiX-ai/android_control_test`, size bucket 1K–10K, not counted) | arXiv 2504.14239 Tab. 3; README identical |
| GUI-SAGE-3B | 3B | 95.5 | n/r | 93.4 | 86.4 | n/r | 75.4 | not stated; "for the evaluation, we followed Gu et al." (= UI-Venus) | CVPR 2026 paper Tab. 2 (CVF open access PDF) |
| GUI-SAGE-7B | 7B | 96.0 | n/r | 93.7 | 87.0 | n/r | 76.8 | same | CVPR 2026 Tab. 2 |
| UI-Venus-Navi-7B | 7B | 97.1 | n/r | 92.4 | 86.5 | n/r | 76.1 | not stated | arXiv 2508.10833 Tab. 11 |
| UI-Venus-Navi-72B | 72B | 96.7 | n/r | 92.9 | 85.9 | n/r | 77.2 | not stated | 2508.10833 Tab. 11 |
| UI-Venus-1.5-2B | 2B | n/r | n/r | n/r | n/r | n/r | n/r | AC absent from the whole paper (0 mentions) | arXiv 2602.09082 |
| UItron-7B (paper) | 7B | 95.8 | 90.7 | 88.2 | 96.1 | 87.1 | 86.9 | 1,000 held-out episodes, 5,477 samples per split ("Following OS-Atlas") | arXiv 2508.21767 Tab. 7 (+Tab. 2) |
| UItron-7B (README rev.) | 7B | 96.5 | 90.6 | 90.1 | 91.4 | 77.2 | 79.0 | revised 2025-09-06 "as we have fixed the vulnerability in the evaluation" | github.com/UITron-hub/UItron |
| UItron-72B (paper) | 72B | 96.5 | 97.1 | 92.8 | 96.7 | 94.2 | 92.4 | as above | 2508.21767 Tab. 7 |
| UItron-72B (README rev.) | 72B | 96.4 | 93.8 | 92.0 | 93.9 | 85.3 | 86.1 | revised, as above | UItron README |
| GUI-Owl-7B | 7B | n/r | n/r | n/r | n/r | n/r | 72.8 | not stated; "Extract match scores with high-level instruction" | arXiv 2508.15144 Tab. 6 |
| GUI-Owl-32B | 32B | n/r | n/r | n/r | n/r | n/r | 76.6 | same | 2508.15144 Tab. 6 |
| AgentCPM-GUI-8B | 8B | TM 94.4 | n/r | EM 90.2 | TM 77.7 | n/r | EM 69.2 | not stated; eval data + code released | arXiv 2506.01391 Tab. 3; README 94.39/90.20/77.70/69.17 |
| OS-Atlas-4B (SFT) | 4B | 91.92 | 83.76 | 80.64 | 84.69 | 73.79 | 67.54 | 7,708 test samples per split (App. D Tab. 9) | arXiv 2410.23218 Tab. 5 |
| OS-Atlas-4B (zero-shot) | 4B | 64.58 | 71.19 | 40.62 | 49.01 | 49.51 | 22.77 | same | 2410.23218 Tab. 5 |
| OS-Atlas-7B (SFT) | 7B | 93.61 | 87.97 | 85.22 | 85.22 | 78.48 | 71.17 | same | 2410.23218 Tab. 5 |
| OS-Atlas-7B (zero-shot) | 7B | 73.00 | 73.37 | 50.94 | 57.44 | 54.90 | 29.83 | same | 2410.23218 Tab. 5 |
| Aguvis-7B | 7B | n/r | n/r | 80.5 | n/r | n/r | 61.5 | **500 randomly sampled step-actions** from the full test set; "step accuracy" | arXiv 2412.04454 Tab. 3 |
| Aguvis-72B | 72B | n/r | n/r | 84.4 | n/r | n/r | 66.4 | same | 2412.04454 Tab. 3 |
| OmegaUse | "30B-A3B" MoE | n/r | n/r | n/r | 87.6 | n/r | 79.1 | one AC setting, described as high-level; split not stated | arXiv 2601.20380 Tab. 6 |
| Qwen2.5-VL-3B | 3B | nv | nv | nv | nv | nv | nv | card rows inconsistent (see note 5) | HF Qwen/Qwen2.5-VL-3B-Instruct |
| Qwen2.5-VL-7B | 7B | n/r | n/r | EM 93.7 | n/r | n/r | EM 60.1 | not stated | HF Qwen/Qwen2.5-VL-7B-Instruct card |
| Qwen2.5-VL-72B | 72B | n/r | n/r | EM 93.7 | n/r | n/r | EM 67.36 | not stated | arXiv 2502.13923 Tab. 9 |
| GUI-R1-3B | 3B | 83.68 | 81.59 | 64.41 | 58.04 | 56.24 | 46.55 | "test splits", size not stated | arXiv 2504.10458 Tab. 2 (Low) / Tab. 3 (High) |
| GUI-R1-7B | 7B | 85.17 | 84.02 | 66.52 | 71.63 | 65.56 | 51.67 | same | 2504.10458 Tab. 2 / Tab. 3 |

**Baselines re-run or copied (per source):** AgentCPM-GUI is the only paper that says it re-ran baselines ("All baseline
results are from our own re-implementations", using the official AC eval code). MAI-UI Tab. 7: "Baseline results are mainly
sourced from scores reported in (Zhang et al., 2025b)"; its baseline values (UI-TARS-7B 74.4, OS-Atlas-7B 56.5, Aguvis-7B 54.2,
AgentCPM 69.2, OdysseyAgent 32.7) match AgentCPM-GUI Tab. 3 exactly. UI-Venus marks own reproductions with `*` and has none in
Tab. 10/11. GUI-SAGE: dashes are "missing values due to unavailable results in the original paper, unreleased model checkpoints
and inference code" (implies a mix of copied and re-run; not stated per row). All others: not stated.

**Public AC evaluation code:** AgentCPM-GUI — yes (`github.com/OpenBMB/AgentCPM-GUI/tree/main/eval`, `run_eval_agent.py
--eval_android_control`, plus prediction scripts for Qwen2.5-VL, OS-Atlas, OS-Genesis, Aguvis, OdysseyAgent). InfiGUI-R1 —
yes (`github.com/Reallm-Labs/InfiGUI-R1/tree/main/eval/android_control` + processed test set on HF). GUI-R1 — yes
(`github.com/ritzz-ai/GUI-R1`, `guir1/eval.sh`, `androidcontrol_{high,low}_test.parquet`). Aguvis — README lists the evaluation
toolkit as in progress. UI-TARS, MAI-UI, UI-Venus, UItron, GUI-Owl, OS-Atlas, OmegaUse, Qwen2.5-VL, GUI-SAGE — no AC eval code
found in the README/paper (nv, not proven absent).

## 2. AndroidWorld

| Model | Size | SR (%) | Runs / budget | Input | Source |
|---|---|---|---|---|---|
| UI-TARS-2B | 2B | n/r | — | — | 2501.12326 |
| UI-TARS-7B-SFT | 7B | 33.0 | not stated (15-step/3-run text applies to OSWorld only) | screenshot-only model (abstract) | 2501.12326 Tab. 9 |
| UI-TARS-72B-SFT | 72B | 46.6 | not stated | screenshot-only | 2501.12326 Tab. 9 |
| UI-TARS-1.5 | n/s | 64.2 | not stated | not stated | UI-TARS README |
| MAI-UI-2B / 8B / 32B / 235B-A22B | 2/8/32/235B | 49.1 / 70.7 / 73.3 / 76.7 | not stated in paper | README: leaderboard category "pure-vision, end-to-end" | 2512.22047 Tab. 6 |
| InfiGUI-R1-3B | 3B | n/r | — | — | 2504.14239 |
| GUI-SAGE-3B / 7B | 3B/7B | 19.8 / 23.3 (bases Qwen2.5-VL-3B 3.5, 7B 19.0) | not stated | not stated | CVPR 2026 Tab. 7 |
| UI-Venus-Navi-7B / 72B | 7B/72B | 49.1 / 65.9 | pass@1, "one-time evaluation"; budget not stated | screenshot only, no planner, no a11y | 2508.10833 Tab. 10 |
| UI-Venus-1.5-2B (8B, 30B-A3B) | 2B | 55.6 (73.7, 77.6) | not stated | not stated in Tab. 2 | 2602.09082 Tab. 2 |
| UItron-7B / 72B | — | n/r (AW listed in data Tab. 2, no result) | — | — | 2508.21767 |
| GUI-Owl-7B | 7B | 66.4 | not stated | not stated | 2508.15144 Tab. 7 |
| GUI-Owl-32B | 32B | 73.3 inside Mobile-Agent-v3 (multi-agent framework); 62.1 inside Mobile-Agent-E | not stated | not stated | 2508.15144 Tab. 7 / Tab. 8 |
| AgentCPM-GUI-8B | 8B | n/r | — | — | 2506.01391 |
| OS-Atlas-4B/7B | — | n/r | — | — | 2410.23218 |
| Aguvis-7B | 7B | 37.1 **as grounder with GPT-4o planner** | not stated | screenshot only | 2412.04454 Tab. 5 |
| Aguvis-72B | 72B | 26.1 (planner + grounder) | not stated | screenshot only | 2412.04454 Tab. 5 |
| OmegaUse | 30B-A3B | 55.7 | not stated | screenshot only, no planner, no a11y | 2601.20380 Tab. 7 |
| Qwen2.5-VL-72B | 72B | 35 | not stated | "without auxiliary marks" (baselines used SoM) | 2502.13923 Tab. 9 |
| Qwen2.5-VL-7B | 7B | 25.5 | not stated | not stated | HF card |
| Qwen2.5-VL-3B | 3B | nv (card prints 90.8) | — | — | HF card (see note 5) |
| GUI-R1-3B / 7B | — | n/r | — | — | 2504.10458 |

No source in this set states the number of AW runs or the per-task step budget, except UI-Venus ("pass@1", "one-time").

## 3. Why these numbers are not comparable

1. **Different AC test sets.** Aguvis: 500 random steps. UItron: 1,000 episodes / 5,477 samples ("following OS-Atlas").
   OS-Atlas: 7,708 samples per split. UI-TARS: "OS-Atlas settings" (size not given; its ablation uses a separate
   1,000-example random sample). InfiGUI: its own processed test set. Everyone else: unstated. None states the full
   official test split (1,543 episodes / 8,444 steps as counted in this repo, commit `66651c5c`).
2. **Same model, different numbers (AC-High).** UI-TARS-7B: 72.5 (own) vs 74.43 EM (AgentCPM re-run; reprinted by MAI-UI).
   OS-Atlas-7B: 71.17 (own SFT) / 29.83 (own zero-shot) / 56.53 (AgentCPM re-run). Aguvis-7B: 61.5 (own, 500 steps) vs
   54.18 (AgentCPM re-run). Qwen2.5-VL-7B: 60.1 (card) / 62.9 (AgentCPM re-run) / 47.06 (GUI-R1, zero-shot prompt).
   The spread for one checkpoint reaches ~15 points — larger than most gaps between the models in Table 1.
3. **Metric names and definitions differ.** SR (Type + Grounding), EM (AgentCPM, Qwen), "step accuracy" (Aguvis), "extract
   match" (GUI-Owl), unnamed (MAI-UI). GUI-SAGE, UI-Venus and OmegaUse report no grounding column. AgentCPM reverses the
   predicted scroll direction for OS-Atlas/OS-Genesis baselines on AC-Low; AC-Low itself gives the step instruction,
   so Low numbers are not comparable with High ones.
4. **Numbers change after publication.** UItron's README lowered AC-High SR from 86.9→79.0 (7B) and 92.4→86.1 (72B)
   after "fix[ing] the vulnerability in the evaluation"; the paper still prints the old values.
5. **Copy errors in derivative tables.** The Qwen2.5-VL-3B card prints AC-Low EM 22.2 and AndroidWorld SR 90.8, which
   cannot both be right (the 72B scores 35% on AW), so we mark it nv. GUI-SAGE Tab. 2 lists Qwen2.5-VL-3B Low SR 90.8 and
   7B Low SR 91.4: these equal the cards' AndroidWorld (3B) and MobileMiniWob++ (7B) rows, not their AC-Low rows (7B
   card: 93.7). GUI-SAGE also prints InfiGUI-R1-3B as 94.8/91.1/81.8/70.7 (paper: 96.0/92.1/82.7/71.1) and OS-Atlas-7B
   Low Type 98.6 (paper: 93.61).
6. **AW setups differ.** Some systems run end-to-end on screenshots (UI-Venus, OmegaUse, MAI-UI); others use a planner (Aguvis-7B + GPT-4o)
   or a multi-agent framework (Mobile-Agent-v3 73.3 vs GUI-Owl-7B alone 66.4). Some baselines use SoM or the a11y tree
   (UI-Venus Tab. 10 lists these per row). Runs and step budgets are almost never stated. RL-trained (MAI-UI, UI-Venus)
   and SFT checkpoints (UI-TARS "-SFT") are mixed in the same columns.

Rule for this repo (AGENTS.md M4): compare against a number only if it was produced with the same split, metric, and
scorer. For AC, the only published re-run baselines with public code are AgentCPM-GUI's and InfiGUI-R1's.
