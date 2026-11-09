# Which benchmark can we actually measure? AndroidControl (offline) vs AndroidWorld/AndroidLab (online)

> Written 2026-10-27 to answer: "what benchmark did MobileRL use?" and "which is feasible with our
> infra?" This is the decision point for which metric we can honestly report.

---

## 1. The three benchmarks and their ground truth

| | **AndroidControl** (Li et al. 2024) | **AndroidWorld** (Rawles et al. 2024) | **AndroidLab** (Xu et al. 2024) |
|---|---|---|---|
| **Environment** | **OFFLINE / static** — screenshots + a11y tree + gold action | **ONLINE / interactive** — live Android emulator (app actually runs) | **ONLINE / interactive** — live emulator |
| **What the model does** | Look at 1 screenshot, predict the next action (type + coords) | Multi-step interaction with a real app to complete a task | Same as AndroidWorld |
| **Correctness** | Compare predicted action to gold action (+ grounding in the GT box) | **Task success** — reward = 0/1 at the end (did the whole task complete?) | Same |
| **Metric** | per-step accuracy, Type / Grounding / SR | **per-episode task success rate (SR)** | SR |
| **Needs emulator?** | **NO** | **YES** (hundreds of Docker AVDs) | **YES** |
| **MobileRL uses it for** | **SFT only** (learn action + grounding) — NOT RL | **RL train (2,000 tasks) + main eval** | **RL train (1,103 tasks) + eval** |

**Key MobileRL quotes (§3.1):**
> "During the RL stage, we employ interactive training sets from AndroidWorld and AndroidLab."
> (AndroidControl) "data ... was **not included during the RL stage**."
> "existing datasets with single-step expert labels ... are **insufficient** for training or evaluating
> policies on **agentic tasks** (i.e., planning and reasoning over multi-step action sequences)."

**Conclusion:** MobileRL measures **task success (SR) on ONLINE environments** (AndroidWorld/AndroidLab).
AndroidControl (offline) is only a *training warm-up*, never the headline number.

---

## 2. What WE can actually measure today (infra check)

| Asset | Have it? | Notes |
|---|---|---|
| AndroidControl offline data | ✅ | train 12,226 / test 3,370 / curated-high-bbox 8,435 / AC-v2 leaderboard 398 |
| Screenshots for that data | ✅ (partial) | `acv2_shots/` + `ac_test_imgs/` exist; curated bbox rows reference `android_control_images/` |
| **Official harness** `eval_official.py` | ✅ | `benchmarks/data/acv2/eval_official.py` (we already ported its logic into `run_acv2.py`) |
| Our runner | ✅ | `benchmarks/run_acv2.py` (offline AC-v2 protocol, 8-action native-agent, a11y grounding) |
| Kaggle GPU T4 | ✅ | `trnanh140999`, CLI push allocates T4×2 |
| **Android emulator / adb** | ❌ | `which adb` → none; no AVD, no Dockerized Android |
| AndroidWorld/AndroidLab interactive env | ❌ | not present; needs Docker + many AVDs (MobileRL: ~256 parallel, >10 min/step, 1 TB RAM ×4 machines) |

**So the headline:** we can run **OFFLINE AndroidControl** end-to-end today (data + harness + GPU), but we
**cannot run ONLINE AndroidWorld/AndroidLab** (no emulator / no adb / no scale-out infra).

---

## 3. What Magma-R1 actually reports (and why its "SR" is offline)

Magma-R1 (AndroidControl-Curated, arXiv 2510.18488) is the **closest existing precedent to our goal**
(small model, grounded next-action, ~$60). It is **offline**:

- Output = `{"bbox_2d":[x1,y1,x2,y2], "action_type":...}` for the **next single action** (prompt.py).
- Rewarded per-step with a **dense 2D-Gaussian grounding reward** `exp(-‖c_pred-p_gt‖²/(2σ²))`.
- Metrics on **AndroidControl-Curated-Hard**: Type 84.2% / Grounding 84.8% / SR 75.3%.
- **IMPORTANT nuance:** its "SR" (75.3%) is the **success rate over the curated offline evaluation
  sequences** — i.e., "does the predicted action sequence match the gold sequence / intended completion",
  NOT a live-interaction task-success (there is no emulator, no live app state). This is a *static*
  analogue of SR, not MobileRL's *online* SR.

> **Read:** Magma-R1 is a *grounding/type* benchmark on clean offline data. It does NOT need AndroidWorld
> and does NOT report MobileRL-style live task success.

---

## 4. Decision for our project

Because we have **no emulator**, the honest, feasible path today is **OFFLINE AndroidControl** —
which is exactly what Magma-R1 does. This means:

1. **We can report** (honestly, offline): `Type` accuracy, `Grounding` accuracy, per-step `SR` on a
   curated/cleaned AndroidControl-Curated split, plus ECE/Brier/risk-coverage for calibration —
   compared to majority/random baselines.
2. **We CANNOT honestly report** MobileRL-style **online task-success SR** (needs AndroidWorld/AndroidLab
   + emulator infra). If we claimed "SR = X on AndroidWorld" it would be unverifiable and wrong.
3. **The framing that works:** our thesis is a **single-pass, calibrated, low-latency decision model**
   for grounded next-action selection. The honest metric is **calibration + grounding + latency** on an
   offline AndroidControl benchmark — and we should be explicit that (a) it's offline, (b) it's compared
   to majority/random + the published offline numbers (Magma-R1, UI-Tars, etc.), (c) it does NOT compete
   with MobileRL/UI-Venus online task-success.

### What we'd need if we ever wanted ONLINE AndroidWorld success rate
- adb + at least one Android AVD (Docker), an emulator harness, and a verifiable reward for task success.
- Realistically this is a much bigger infra lift (MobileRL's setup) than our current project scope.

---

## 5. Recommended next step (feasible, honest, aligns with Magma-R1)

Build an **offline grounded next-action choice** dataset from our curated AndroidControl data:
- Per step, build **K grounded candidate actions** (each = action_type + point_2d/bbox), drawn from the
  a11y tree / candidate builder.
- Gold = the candidate whose bbox contains the ground-truth coordinate (and whose type matches).
- Train a small model to score K candidates in one pass (d1-style readout / choice-over-K), reward with a
  **dense grounding signal** (2D-Gaussian on point distance) — Magma-R1's mechanism.
- Evaluate honestly vs majority/random baseline + ECE + grounding accuracy + latency.

This needs **no emulator**, uses **data + harness we already have**, and produces a real, defensible
offline result.

---

## Coverage / honesty
- MobileRL benchmark usage read from arXiv 2509.18119 §2.1, §3.1, Appendix D (verbatim).
- Magma-R1 metric definitions read from arXiv 2510.18488 (§2.2.1 reward, §3.4 metrics) + released repo.
- Infra check done live in this session (adb absent, no emulator, Kaggle CLI + T4 present, acv2 harness present).
- "SR" of Magma-R1 is offline (static sequence success), not live task success — flagged explicitly as a
  nuance so we never conflate it with MobileRL's online SR.
