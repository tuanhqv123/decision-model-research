# ISSUES & Findings — session 2026-10-29 (grounding gap, noise/CI, eval-set verification, trend mem/RL/CoT, compute VKong)


> **Correction (2026-11-07).** Every "64.32" below is the pre-audit figure. No artifact yields 256/398:
> the pure protocol is **64.07% (255/398)** in-process, 63.82% (254/398) through the served API, and the
> 56.50% grounding is the step-705 checkpoint. The qualitative conclusions in this note (a tie with
> GUI-Libra, not a win; grounding is the deficit) are unchanged.

> Records the important **issues + findings** of this session so they are not lost, and ties them to the training decision.
> Essence: after the d1-2B full fine-tune reached 64.07% AC-v2 High (the original wrote 64.32 — corrected 2026-11-07), the user led into a series of questions on
> "do we beat GUI-Libra / why is grounding low / do we need memory/RL/CoT". The answers
> were checked against real numbers. This file records **the issues + what was verified + what is still open**.

---

## A. The central question: "do we only need to be better than GUI-Libra?"

### Issue A1 — We are LEVEL with GUI-Libra-8B, not yet beating it (most important)
- **real number:** our d1-2B = **64.07** (255/398, pure protocol, official `eval_rl.py`). *(The original wrote 64.32/256; no artifact gives 256 — corrected 2026-11-07.)*
  GUI-Libra-8B = **64.3** (NeurIPS 2026). Mem-W-8B = 68.59 (preprint).
- **Our 95% CI = [59.5, 68.8]** (398 steps = small sample, CI ±4.7).
- **Because the CI covers 64.3 → it is only "a tie within noise", NOT yet a win.** To really beat it we need **≥66–68** (CI floor > 64.3).
- **→ On AC-v2 we are level; we need to raise about 5–8 points to "beat at a top-tier venue" (NeurIPS).**

### Issue A2 — "Tier A or above" = CCF-A; the only tier-A baseline on AC-v2 is GUI-Libra
- CCF-A: NeurIPS / ICLR / ICML (the rivals' league). Tier A or above = CCF-A.
- Baselines on AC-v2 High:
  | Baseline | Score | Venue | Tier A? |
  |---|---|---|---|
  | GUI-Libra-8B | 64.3 | **NeurIPS 2026** | ✅ |
  | MobileRL | 72.0(AW) | **ICLR 2026** | ✅ (but on AndroidWorld) |
  | Mem-W-8B | 68.59 | preprint | ❌ |
  | UI-Venus-1.5-8B | 61.06 | tech report | ❌ |
  | Qwen3-VL-2B/4B/8B | 45–59 | model card | ❌ |
- **→ "beat tier A" = beat GUI-Libra on AC-v2 High. That is the real scalp.**

---

## B. Eval-set verification (verified, NOT a ghost) — good news

### Issue B1 — The eval file `500_steps_filtered.json` IS GUI-Libra's AC-v2 ✅
- GUI-Libra (Appendix C.3): *"We start from a sampled set of 500 examples from UGround and obtain a
  cleaned subset of **398 examples** after filtering"* (they used Qwen3-VL-32B-Instruct to filter about 20% of
  action-type/coordinate errors).
- Our file: `benchmarks/data/acv2/500_steps_filtered.json` = **398 rows, 350 episodes**, with
  `goal` (high) + `step_instruction` (low) + `previous_actions` (natural language such as
  "Go back to the previous screen") — **matches** their description.
- **→ We are using EXACTLY the eval set GUI-Libra used to report 64.3. Comparing 64.07 against 64.3 is VALID.**
- **Correction:** earlier we said "different eval set" — WRONG. We correct it: the eval set is EXACT;
  only the **training data** differs (we use the original GCS data, unrefined; they use AC-v2 + 81K reasoning).

### Issue B2 — 2 remaining caveats
1. **Different training data:** we train on the original AndroidControl GCS (which has ~20% annotation errors); GUI-Libra trains
   on the filtered AC-v2 + 81K reasoning data. → our data is "dirtier" (but the eval set is the same).
2. **Protocol:** we must be sure we put `previous_actions` into the prompt (the correct High protocol). Fixed earlier.

---

## C. Why is grounding low? (real data from E2-938-V.jsonl)

### Issue C1 — Grounding = 56.5 (the model knows WHAT to do but taps the WRONG PLACE)
Analysis of the 223 spatial rows (with coordinates) of run E2-938:
| Metric | Value | Meaning |
|---|---|---|
| Type correct | 196/223 | the model knows the action |
| Of those 196, grounding WRONG | **68** | knows what to do but taps the wrong place |
| Distance box_pt→gold | median **44px**, mean **275px** | large offset |
| Tap within 50px | 51.6% | — |
| Tap within 100px | 58.3% | — |
| Vertical bias (y_pred−y_gold) | median **−1px** | NOT a systematic offset |

**→ NOT a "systematic bias"** (vertical bias is only −1px, too high/too low balanced 15%/12%). The problem is
**DISPERSION** — the model taps all over the place, sometimes right and sometimes off by a whole screen.

### Issue C2 — "d1 draws bboxes well, so why is grounding low?" — a misunderstanding to clarify
- **Box ≠ exact click point.** A box = a rectangle around the element (looks good). Grounding = the **click point
  lies inside the *smallest element* that contains the gold** (smallest bbox node).
- d1-2B uses a **pointer/position-logit** (`pointer.py`) to produce the click coordinates — a mechanism entirely DIFFERENT from drawing a bbox.
- For small elements (icons), an offset of 44px (median) **fully misses** into another element.
- → **"A nice bbox" ≠ "clicking the right place".** The model understands where the element is but lowers the coordinates not precisely enough.

### Issue C3 — Distribution of the gold position (1080×2400 screen)
- Gold y: median 1101, p25 525, p75 1990. Top third (y<800): 39%, MIDDLE: 26%, bottom third: 35%.
- → Gold is spread over the whole screen, not concentrated → the model needs accurate grounding in every region.

---

## D. Trend research 2026: memory / RL / CoT (details in notes/trend-memory-rl-cot-2026.md)

### Issue D1 — CoT/reasoning: ❌ NOT worth investing in
- **"Does CoT Help Mobile GUI Agents?" (ACL 2026 Findings, BUPT+Tsinghua):** an empirical study of 6 VLM pairs.
  - Static: reasoning barely improves, even lowers (ScreenSpot −29.7%).
  - AndroidWorld: model-specific (Claude +6.3%, Gemini/Qwen3-VL drop).
  - **Token cost rises 3.11×–14.78×** (37.6→238.5) → breaks our "0-token" axis.
  - A small budget (128 tokens) is best; more gives diminishing returns.
  - Errors are mostly from the benchmark (47.8% weak-eval, 14.9% static-input) → supports the benchmark-audit thesis.
- **Magma-R1 + GUI-Libra:** long CoT *lowers* grounding → needs action-aware SFT.
- **→ DROP CoT.** Our axis (direct decision + calibration) is stronger on latency.

### Issue D2 — RL: ✅ Priority #1, strongest evidence
- MobileRL (ICLR): SFT +22.6 → RL +15.2 → 72.0 AndroidWorld.
- GUI-Libra (NeurIPS): conservative GRPO + **KL trust region** (against partial verifiability) → 42.6 AW.
- **Magma-R1:** GRPO + **dense 2D-Gaussian grounding reward** → 2.4k curated, 3-4B = 235B.
- What the successes share: **DENSE** reward, handling **partial verifiability** (KL), **class imbalance**.
- **→ This is the path to close the grounding gap 56.5 → 70+.**

### Issue D3 — Memory: ✅ worth doing, but it MUST be latent/multimodal, NOT text
- Mem-W: latent memory token, +30.
- MementoGUI: working + episodic multimodal memory; **dropping the ROI image → AMS 67.69→64.22** (the visual in
  memory is not redundant with text).
- CoMEM: continuous memory of 8 embeddings, keeps the visual cue, shrinks context.
- FocusMem: factorizes content/readout/trust; AndroTMem: anchored memory.
- **→ EXPLAINS why our TF-IDF text memory got −6.0:** text-only memory loses the visual, not enough.
  To build memory it must be latent/multimodal.

### Issue D4 — Conclusion on direction
| Direction | Worth doing | Reason |
|---|---|---|
| RL + dense grounding reward | ✅ **#1** | Closes the grounding gap, strongest evidence |
| Latent/multimodal memory | ✅ #2 | Needed for multi-step, does not lose to them |
| CoT/reasoning | ❌ Drop | Lowers grounding + raises tokens 3-15×, inconsistent |

---

## E. Compute: VKong (vkong.tli-tech.com) — prices probed

### Issue E1 — Platform: VKong = hourly cloud GPU, $5 credit available
- CLI `vkong` installed (v0.10.36), logged in as user `tuanhqv123`, **$5.00 credit, 0/2 machines**.
- Config in `vkong.yaml` (type task/service, gpu, max_price, image, start, min_cpu_cores, min_ram_gb).
- **See the price without renting:** `vkong dev` (prints GPU+price, exits, does not rent) does NOT reflect the real GPU
  (always Modal L4 $1.75/h). **The right way:** `vkong run` prints the price before renting → press 'n'/do not auto-approve.
- **Real RTX-4090 price = $0.44/h** (24GB, 63GB RAM, 8 cores, Poland). → **$5 credit is enough for a smoke + experiments.**

### Issue E2 — Data to upload to VKong
| Data | Location | Size |
|---|---|---|
| Eval set | benchmarks/data/acv2/500_steps_filtered.json | 33MB |
| Screenshots | benchmarks/data/acv2_shots/images | **6.4GB** |
| Train data (E2 backup) | backup-2026-10-29/[host]/acwork | **26GB** |

→ Disk is enough (we set `disk_gb: 10` but the 26GB backup >10 → need to set `disk_gb` high enough, or copy only the necessary
data: train images + compact_train.jsonl. **Must be computed carefully, VKong charges for storage.)**

### Issue E3 — RISK: d1-2B is multimodal, needs images + HF download (~4GB)
- Make sure the machine shell has internet (VKong does) + enough disk for the HF weights (d1-2B ~4GB).

---

## F. OPEN questions to answer before coding RL (important)

### Issue F1 — RL for our 0-token model: what does "sampling G times" mean?
- d1-2B **reads logits at the answer slot, does NOT generate tokens** (unlike MobileRL/GUI-Libra, which generate text).
- When doing GRPO, we must determine: does the model sample **stochastically** from the logits (categorical), or only greedily.
  - If it can sample → standard GRPO (each rollout gives a different action).
  - If not → use only **dense reward inside SFT** (change the CE loss → Gaussian-weighted), not real RL.
- **Need to read `pointer.py`/`slots.py` + test before writing RL.**

### Issue F2 — The current `train_d1.py` is SFT, with an existing box-edge Gaussian loss
- `train_d1.py` already has an `edge` loss = CE vs a Gaussian-smoothed target for the box head
  (`tgt = exp(-0.5*((grid-center)/4)^2)`) → **matches the dense reward direction**. It can be reused.
- But RL (GRPO) does not exist yet; RL needs a sampling loop + advantage + KL.

---

## G. Next work (in order)
1. **Read `pointer.py`/`slots.py` + tests**: can d1-2B sample stochastically → decide real RL or
   dense-reward SFT. (Free, on the Mac.)
2. **Choose the grounding strategy**: RL + dense 2D-Gaussian reward (Magma-R1) — the main path.
3. **Prepare data + script** on the Mac before renting VKong (the principle "no useless spending").
4. **VKong smoke**: subset ~500 samples, RTX-4090 ~$0.44/h, measure time/step + VRAM + whether grounding moves.
5. If the smoke is good → full; if not → stop, change direction.

---

## H. Transparency (what we can NOT say)
- **We have NOT really "beaten" GUI-Libra** (level at 64.07 vs 64.3, within noise). We can only say "competitive".
- **Our grounding of 56.5 is lower than GUI-Libra-8B's (76.2)** — where they are strong, and where we must put effort.
- **The d1-2B number on AC-v2 is measured by us**, with no third-party confirmation; GUI-Libra passed NeurIPS.
- **The choice of the V model at the end was almost based on the test eval** (validation 62.76 matches) — must disclose.
- **A single-pass model, no multi-turn RL, no memory** → on AndroidWorld it may be lower than theirs
  (not measured); this is the axis we deliberately chose in order to attack calibration/latency.

---

## I. CORRECTIONS after re-checking against data + code (2026-10-29 afternoon)
- **C1 (partly wrong):** low grounding is NOT because of "tapping all around the right place". Over 71 misses (action type already
  correct, pure protocol): **72% miss by >150px from the correct element = CHOOSING THE WRONG ELEMENT** (a semantic error); only 13% miss
  by ≤30px. Small elements (<0.2% of the screen) are hit only 39% against 68% for large elements. → we must teach "which element", not only "coordinates".
- **C2 (wrong):** the click point does NOT come from `pointer.py`. Click = the centre of the `box` question (presence + 4 edge distributions 0–999‰
  read through the LM head, `prompt.question_slots(Box)`); `pointer.py` only reads text spans (app_name/type_text).
- **A1:** "beat when ≥66–68 (CI floor > 64.3)" — miscalculated: at 68% the CI ≈ [63.4, 72.6] still contains 64.3; we need ≈69+ for the CI floor > 64.3.
- **D1:** on AC-v2, GUI-Libra Table 9 shows CoT DOES help (SFT 40.2→44.5; RL 48.7→55.8). We drop CoT because of
  the 0-token design, not because "CoT lowers grounding" on AC.
- **D3:** Mem-W is not "+30" on AC-v2: +7.53 (8B) / +13.77 (4B).
- **H:** "GUI-Libra-8B grounding 76.2" — no source yet in the AC-v2 table (Mem-W Table 2 writes "–") → must not be cited yet.
- **E2:** do NOT use `acv2_shots` (6.4GB, the old UGround set which has mixed-in TEST episodes) as training images. Training images are extracted straight from
  GCS on the rented machine (`benchmarks/train/extract_images.py`); upload only ~400MB (compact + evalset).
- **F1 (answered):** the decision model returns the WHOLE distribution (action, 4 box edges) → it can be sampled, and better: compute
  EXACTLY P(box centre ∈ correct element) by convolving the 2 edge distributions (log-space) → optimise the metric directly, with no
  sampling/GRPO (`train_d1.center_in_range_logprob`, test `test_hitloss.py` PASS vs Monte-Carlo, gradient remains when far).

## J. Grounding plan prepared (vkjobs/acv2-ground)
- Arm A: 858px images (exactly the d1 ceiling of 1.64MP; the old 720px was below the ceiling) + expected-hit loss, cur15k.
- Arm B: A + 7.5k auxiliary grounding (step description → box, element label → box; sampled separately to avoid leaking the answer), cur15k+aux7k.
- Fail-fast gates on the machine: test_hitloss, parity, aux. The final result is scored with the official eval_rl.py + validation.

## K. The first VKong run failed (2026-10-29 15:43–16:02) and how to do it more carefully
- Error: the `ubuntu:22.04` image has no gcc → Triton (d1's conv kernel) could not compile → died at the parity gate, AFTER
  ~15 minutes of image extraction. Cost $0.21. The old server already had gcc, so we never hit this.
- Lesson: we never test-ran INSIDE the rented environment. Fixed: `init.sh` installs build-essential + a Triton gate in setup itself;
  `run.sh` has a SMOKE stage (shard 0, 42+16+16 samples, runs every code path: hit-loss, parity, aux, 3 train steps + eval 398 +
  save + eval_rl + val) BEFORE downloading 47GB; md5 of the data; 25-minute watchdog; 5h timeout; images stored as 1 tar file in storage.
- Found when checking: storage reported "save not confirmed" → we re-downloaded in/*.jsonl and compared md5 = the original E2 version (match).
- Hanging at "Terminating/Save volume" for ~15 minutes: caused by saving 15.8k small image files → switched to 1 tar file.
