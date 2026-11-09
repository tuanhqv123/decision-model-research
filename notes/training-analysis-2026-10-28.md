# Should we fine-tune d1-2B? What SOTA training actually teaches us

*2026-10-28. Written in response to the user's challenge: "Are you sure just running training is better? Study how the SOTA models were trained, what went wrong, how they won, and the complexity of our task — before doing anything."*

**Bottom line up front:** No — "just train" is NOT guaranteed to be better. Naive SFT on the raw data would likely reproduce our own majority-collapse (0.5828 = always-click, retracted). The literature shows training works only when **5 conditions** hold, 2 of which (completion labels, class balancing) we can build and verify cheaply BEFORE spending compute. Training is still the only path to SOTA (zero-shot d1-2B is measured and final: 15.83% vision / 28.39% a11y).

---

## 1. Facts we measured ourselves (trustworthy, scorer differential-validated 398/398)

| Item | Value |
|---|---|
| d1-2B zero-shot, vision (box) | **15.83%** step acc (63/398); grounding 27.9% (n=165); status **0/87** |
| d1-2B zero-shot, a11y choice-over-K | **28.39%** (113/398); grounding 50.0% (n=204); status **0/87** |
| Trivial baselines | always-status 21.86%, always-scroll-down 8.04%, always-click+center 4.02%, type-only majority 53.27% |
| Our prior "result" (retracted) | G2/G3 0.5828 == always-click majority → learned nothing |

**Data audit (verified this session, no compute):**
- `android_control_train.jsonl`: 12,226 trajectories; keys only `goal`/`step_instructions`/`actions`. **No success flag. No terminate/status action anywhere.** Last action of 500 sample trajectories: click 342, wait 81, scroll 68, input_text 6, navigate_back 3.
- Curated 8,435 (`android_control_high_bbox.json`): no status/terminate class at all.
- Original HF dataset `smolagents/android-control` (train 12,232 + test 3,051): schema is episode_id, goal, screenshots_b64, actions, step_instructions — **also no success flag.**
- **Key finding:** in the 398-step eval, **85/87 status rows are the LAST step of their episode** (0 episodes with >1 status row; status rows have `step_instruction="terminate"`, `goal_status="successful"`; 350 episodes total). → "Task complete" ≈ "this is the final screen of an episode", which **can be synthesized from training trajectories by position**: last step → complete=1, all intermediate steps → complete=0.
- Caveat: this assumes training episodes are completed (successful) demos. Verification gate = probe recall on the 87 eval status screens (in-distribution success screens).

---

## 2. How the SOTA actually wins (mechanism decomposition, not vibes)

AC-v2 leaderboard, 398 steps, High protocol (goal + screenshot, no history), Pass@1 [sota2androidcontrol]:

| System | Backbone | Score | Type | Grounding | What it actually did to win |
|---|---|---|---|---|---|
| Mem-W-8B | UI-Venus-1.5-8B + latent memory | **68.59** | 81.16 | 71.3 | FROZEN 8B backbone (already SFT+RL'd) + Q-Former memory compressor; Stage-1 self-distillation (action loss + KL), Stage-2 outcome-aware policy gradient, KL-anchored; memory bank 22.3k web + 5k mobile trajectories [memw2026] |
| GUI-Libra-8B | own | **64.3** | — | — | Action-aware SFT (token reweight α_action=2, α_grounding=4) → conservative GRPO (KL β=0.001 + SNGS for partial verifiability) [guilibra2026] |
| Mem-W-4B | Qwen3-VL-4B + memory | **63.07** | 74.62 | 70.85 | Same memory recipe on 4B [memw2026] |
| UI-Venus-1.5-8B | Qwen3-VL-8B fine-tune | **61.06** | 74.62 | 61.06 | 4-stage training; pure RFT (GRPO, no SFT stage); 4-term additive action-wise reward; sparse-action context enhancement (Cartesian-product augmentation) [uivenus15] |
| Qwen3-VL-2B | general 2B VLM | **45.48** | 64.07 | 59.64 | **Zero-shot, no GUI training** — the small-model bar |
| Magma-R1-3B/4B* | qwen2_5_vl base | (own AC-Curated eval: Type 84.2 / Grounding 84.8 on Hard) | | | Single-stage GRPO, **dense 2D-Gaussian grounding reward** (exp(−‖c−p‖²/2σ²)), **action-type proportional sampling**, **2.4k curated samples**, 60h H20 ≈ $60 → matches Qwen3-VL-235B [magma-r1-2025] |

\* Paper says 3B MobileVLM base; released checkpoints are 4B Qwen2.5-VL — unresolved mismatch, flagged.

**Closest precedent for "small model wins on this benchmark family" = Magma-R1.** Its winning ingredients, in order of stated importance:
1. **Data quality > quantity** — ~30% of AndroidControl has ambiguities/factual errors; "the benchmark was actively teaching incorrect behaviors"; training on the larger raw set *reduced* accuracy.
2. **Dense continuous grounding reward** instead of binary hit/miss ("no effective learning signal" for near-misses).
3. **Class balancing** via stratified sampling to a target action distribution (long-tail: click dominates).
4. GRPO on top — but on single-turn step data with immediate dense reward (no emulator, no multi-turn).

**MobileRL** (for contrast, why we CNOT copy it): 2 SFT stages + AdaGRPO *online* RL with hundreds of Dockerized Android emulators (>10 min/step, 100 RL steps > 25h, 4×1TB-RAM machines). The single biggest jump was still the **cheap offline SFT** (Qwen2.5-VL-7B: 27.6 → 50.2, +22.6). Online RL is not feasible for us (no emulator/adb) [mobilerl2025].

**GUI-Libra 3B ladder** (their AndroidWorld numbers): base 3.5 → SFT 5.2 → action-aware SFT 13.0 → ASFT+RL 25.2. SFT alone ≠ SOTA; RL on top adds the rest — but with KL control or it fails (see below).

**Mem-W key observation:** memory adds **+7.5 over its own frozen backbone** (68.59 vs 61.06). Current SOTA is won with 8B + memory, not with small models. A 2B single-pass model is structurally below that axis.

## 3. What they got wrong (failure modes to avoid — each is documented)

| Failure | Who hit it | Consequence | Our guard |
|---|---|---|---|
| CE loss over point coordinates | UI-Venus | over-penalizes valid off-center hits → grounding degrades | choice-over-K (a11y) for element selection; if box: point-in-box reward, not CE over coords |
| RL without KL anchor | GUI-Libra | reward goes up, eval goes down; entropy collapse | KL to reference policy, β≈0.001 |
| 20–30% label noise in AndroidControl | GUI-Libra (20%), Magma-R1 (30%) | "benchmark actively teaching incorrect behaviors"; bigger data = worse | curation pass BEFORE training (bbox↔a11y consistency, agreement filter); measure noise rate on sample |
| Class imbalance (click 60–70%, rare ≈0) | Magma-R1, UI-Venus | SFT alone cannot learn rare actions | stratified sampling to target distribution + synthetic completion rows |
| Long CoT | GUI-Libra | measured negative correlation with grounding | single-pass, zero tokens — aligned with our design by default |
| Grounding vs navigation reward conflict | UI-Venus | one reward term degrades the other | train completion head separately from action selection (two-stage) |
| Partial verifiability (r=0 ambiguous) | GUI-Libra | false negatives poison advantages | SNGS-style downscaling IF we ever do RL; not needed for SFT-only |
| Majority-prior collapse | **us (0.5828)** | model = prior, learns nothing | verification gate: majority + random baselines before any number is a result |

## 4. The real complexity of OUR task (398-step High-v2)

It is not "predict an action type". It is **three different sub-problems in one benchmark**:

| Sub-problem | Share of eval | Label availability | Mechanism needed |
|---|---|---|---|
| Element grounding (click/scroll target) | 212 click (53%) + 37 scroll | EXISTS but 20–30% noisy (bbox + a11y) | grounding-friendly objective |
| **Completion detection (status/terminate)** | **87 (22%)** | **DOES NOT EXIST** in train/curated/original dataset — verified | binary success head; labels must be **synthesized** (position: last step of episode → complete) |
| Discrete params (open_app/type_text/back/wait) | 68 (17%) | exist, rare | class balancing |

Consequences:
- Without completion labels, fine-tuning caps the model at ~78% of the leaderboard regardless of everything else (status stays 0/87 → max ≈ 311/398 if everything else were perfect, and realistic far below).
- The synthesized completion set is **on-distribution** for the eval's status class (final screens of successful episodes) — that's the 85/87 finding.
- The two-stage protocol (p_done noul → action choice) is the decision-model-native way to train the completion head **independently** of action selection (avoids UI-Venus's reward-conflict failure).

## 5. The honest answer: is "just train" the best next move?

**No.** Training is *necessary* (zero-shot is measured and far below every SOTA) but *not sufficient*. The evidence-based conditions for training to work, each sourced:

- **C1 — Curate first.** Remove noisy steps (20–30% per [guilibra2026][magma-r1-2025]); measure our own noise rate on a sample before training.
- **C2 — Balance classes.** Stratified sampling to a target distribution [magma-r1-2025]; augment rare actions (UI-Venus context enhancement) [uivenus15].
- **C3 — Synthesize completion labels** (position-based: last step → complete=1). Verify via probe recall on the 87 eval status screens.
- **C4 — Objective matches the primitive.** CE over choices is fine for action type (d1's native training). For grounding: point-in-box / dense signal, not CE over raw coords [uivenus15]. Two-stage: completion head separate from action head.
- **C5 — Conservative + calibrated.** Option shuffling + shortcut removal (d1's own recipe [d1-recipe-note]); per-type temperature calibration to keep the paper's thesis (calibrated decisions) intact; if RL is ever added: KL anchor [guilibra2026].

**If any of C1–C3 is missing → we re-create the 0.5828 collapse.** That is why the plan below verifies them cheaply before any full training run.

**Realistic expectations (stated plainly):**
- **Beat Qwen3-VL-2B 45.48 (small-model zero-shot bar): plausible.** Logic: Magma-R1 showed a small model + clean data + right reward beats much bigger zero-shot models on this benchmark family; we additionally fix the completion class, which a zero-shot general VLM does not optimize.
- **Beat Mem-W-4B 63.07: low probability.** That is a 4B model with trained latent memory + RL, on the 8B-recipe family. A 2B single-pass offline model is structurally below it; we do not claim it.
- If the learnability probe (Stage C) fails its gate: we stop, and the honest deliverable is a negative result — "a 2B single-pass decision model does not break AC-v2 offline, and here is the measured decomposition of why" — which still serves the paper (Q1: when/why single-pass miscalibrates/fails).

## 6. Revised plan (cheap-first, every stage has a pass/stop gate)

| Stage | What | Where / cost | Pass gate | Stop rule |
|---|---|---|---|---|
| **A. Data prep** (local, no compute) | (a) verify training episodes look like completed demos (sample 20 final screens vs their goals); (b) measure label noise (bbox↔a11y consistency on 500 steps); (c) build: cleaned + class-balanced train mix + synthetic completion set (last-step rows) | Mac, ~1–2h agent time | noise rate known; completion set built; no obvious train/eval leakage | if >50% of sampled train steps fail bbox↔a11y consistency → curation infeasible offline, stop & redesign |
| **B. Two-stage zero-shot, full 398** | p_done on all 398 (87 status vs 311 non-status) + full two-stage step acc | Kaggle T4, ~1h | p_done(status) mean ≫ p_done(non-status) → threshold rule gives immediate free points pre-fine-tune; interface validated on remote | p_done gap ~0 → completion not detectable from a single screen zero-shot; training the completion head becomes mandatory (not optional) |
| **C. Learnability probe** | LoRA SFT on ~2–3k cleaned+balanced+completion rows, 1–2 epochs, two-stage protocol; eval 398 with official scorer | Kaggle T4, ~2–4h | **step acc ≥ 32% (i.e., +10 over strongest trivial baseline 21.86%) AND status recall ≥ 30%** | below gate → do NOT full-train; write negative-result note; reconsider interface (maybe a11y-only, or different base) |
| **D. Full fine-tune** (only if C passes) | 8.4k cleaned + balanced + completion; d1-style recipe (option shuffling, long inputs, shortcut removal); per-type temperature calibration; full eval + ECE/Brier + act/ask/refuse risk gate | Kaggle T4, 1–2 runs | **≥ 45.48** (breaks the small-model bar) with ECE reported | — |

RL (GRPO-style on the choice distribution) is **not** in the plan: it's novel for a single-pass decision model, high-risk, and MobileRL's lesson is that cheap SFT gives the biggest jump first. Revisit only if D saturates below target.

## 7. Sources

- [sota2androidcontrol] sota2.com AC-v2 leaderboard (May 2026) — scores, Type/Grounding splits.
- [memw2026] Mem-W, arXiv:2605.09317 — frozen backbone + Q-Former compressor, self-distillation + outcome-aware PG, memory bank, M sweep 63.1→70.8.
- [guilibra2026] GUI-Libra, arXiv:2602.22190 + repo — ASFT α_a=2/α_g=4, KL β=0.001 + SNGS, ~20% label noise finding, long-CoT negative correlation.
- [uivenus15] UI-Venus-1.5-8B, arXiv:2602.09082 — pure RFT, 4-term action-wise reward, sparse-action context enhancement.
- [magma-r1-2025] Magma-R1 / AndroidControl-Curated, arXiv:2510.18488 — 2.4k curated, dense 2D-Gaussian grounding reward, proportional sampling, ~30% noise, $60.
- [mobilerl2025] MobileRL, arXiv:2509.18119 — 2 SFT stages + AdaGRPO; SFT = biggest jump (+22.6); online RL cost (hundreds of AVDs, >10 min/step).
- Local verifications this session: train-data audit (12,226 trajectories, no success flag), 85/87 status=last-step finding, smolagents/android-control schema check.
---

## 8. Step A results (2026-10-28, done)

### 8.1 Data model — CRITICAL (verified, do not re-derive)
- Trajectory with T actions has T+1 screenshots `screenshot_0..screenshot_T` (UGround layout, 0-indexed).
- `screenshot_i` = state where action i (0-indexed) is EXECUTED (pre-action state).
- **`screenshot_T` = FINAL state, after the last action, no action on it** ← this is the completion screen.
- Curated rows map `step_n.png` (1-indexed) → `screenshot_{n-1}.png` → curated covers only pre-action states; final state is NEVER in curated (that's why the "high bbox" filter dropped all completions).
- AC-v2 eval status rows confirm: status screenshot = state AFTER the last UI action (e.g. ep=4114: last action step=4 on screenshot_4, status img = screenshot_5).
- Visual verification (3 episodes, viewed): 11843 (Oxford Dict "evacuation" rendered), 14319 (HelloTalk DE selected), 945 (Lunar Phase detail page for Jan 1 2024: Waning Gibbous 79.81%, Apogee) — all final screens show the goal achieved.

### 8.2 Completion set (built, local)
- `benchmarks/data/curated/android_control_completion.json`: **1,242 rows** = final-state image + goal, label complete=1.
- Built by matching episode instruction → train trajectory (goal text + length constraint); 10 episodes with smolagents/UGround length mismatch excluded (0.8%); 284 episodes unmatched (goal text mismatch) left out.
- Zero overlap with curated images. n_steps=1: 97 rows (final state = state after the single action — correct completion screens, e.g. app opened).
- All images local in `benchmarks/data/acv2_shots/images/` (6.4GB full UGround set). **No Kaggle data upload needed** — kernel can extract 9,683 shots from the 6.59GB public zip (56s on T4, verified).

### 8.3 Label noise / ambiguity (measured on 398 eval, local)
- 212 click rows: gold point inside ≥1 a11y node: **100%** (scorer spatial check valid). Inside LABELED node: 79.7%. Inside clickable node: 63.7%. Inside ≥2 overlapping nodes: **74.5%** (tree-nesting ambiguity — "which element is the target" is structurally ambiguous for 3/4 of click steps; consistent with the 20–30% noise findings in [guilibra2026][magma-r1-2025]).
- Curated 8,435: 5,074 rows have bbox; 100% point-in-bbox (self-consistency, weak check — same label source); 3,361 rows no bbox (non-spatial actions).
- a11y tree field = `bbox_pixels` dict (x_min/x_max/y_min/y_max), NOT `bbox` (which is null). Our ax runner uses bbox_pixels correctly.

### 8.4 Training mix definition (for Stage C)
- Action head: 8,435 curated rows (complete=0), stratified sampling by action type (Magma-R1 style; click 5,065 dominant).
- Completion head: 1,242 positives + sampled negatives (intermediate steps of same episodes), target ratio ~1:2 to ~1:3 (eval ratio is 22:78).
- Two-stage: completion head trained as a separate noul objective; action head trained with CE over choices (d1-native).

### 8.5 Stage B results (2026-10-28) — two-stage zero-shot full 398, Kaggle T4
- Latest run **73/398 (18.34%)** @thr 0.5; earlier run 61/398 (15.33%) → T4 bf16 nondeterminism ±~3pts. See `benchmarks/results/2026-10-28-d12b2s-full-vision-twostage.md`.
- p_done separates status vs non-status (med 0.187 vs 0.073) but at thr 0.5 catches only 10/87 → "free points" hypothesis DEAD at zero-shot; best eval-fit thr 0.10 → 25.6% (overfit, not deployable).
- ECE(action-prob) 0.362 — action head overconfident on misses (mean conf 0.528 when wrong).
- **Decision: Stage C (train completion head on 1,242 rows) is now MANDATORY.** Kernel bug (conf=None → ECE crash) fixed; end-of-run /kaggle/working cleanup added.
