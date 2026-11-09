# What the SOTA actually does — and the four places we differ

*Read 2026-11-08 from the primary sources: MAI-UI technical report (arXiv:2512.22047), MobileRL
(arXiv:2509.18119), GUI-Odyssey (ICCV 2025, arXiv:2406.08451), AndroidWorld (arXiv:2405.14573).
Nothing here is inferred.*

## The leaderboard we are actually on (AndroidWorld, 2026-11-08)

| Model | Params | AndroidWorld |
|---|---|---|
| Qwen3.8-Omni-Flash | — | 87.1% |
| Qwen3.8 Max | 2.4T | 85.3% |
| UI-Venus-2-27B | 27B | 84.0% |
| PhoneBuddy-4B | 4B | 83.2% |
| Holo3.1-35B-A3B | 35B | 79.3% |
| **MAI-UI-235B-A22B** | 235B | **76.7%** |
| MAI-UI-32B | 32B | 73.3% |
| Holo3.1-4B | 4B | 71.0% |
| MAI-UI-8B | 8B | 70.7% |
| **MAI-UI-2B** | **2B** | **49.1%** |
| M3A (AndroidWorld's own agent) | — | 30.6% |
| Ferret-UI-Lite-3B | 3B | 28.0% |

**The bar for a 2B model is 49.1%.** That is MAI-UI-2B, the strongest published 2B result. Anyone who
reports a 2B-class number below that has not matched the state of the art; anyone above it has beaten it.
My earlier "15–35%" was invented and is retracted.

Note the 76.7% figure belongs to the **235B** model. Quoting it as a 2B result would be the same class of
error as the retracted 64.32.

## MAI-UI's recipe, in its own words

**Backbone.** Qwen3-VL (2B / 8B / 32B / 235B-A22B).

**Action space (Table 1, 12 actions):** `click`, `long_press`, `type`, `swipe`, `drag`, `system_button`
(back/home/menu/enter), `wait`, `terminate`, `answer`, `ask_user`, `mcp_call`.

Compared with our 8 (`click, long_press, scroll, type_text, open_app, navigate_back, wait, status`):
they have `swipe`+`drag` instead of `scroll`, `system_button` instead of `open_app`+`navigate_back`, and
**`answer`, `ask_user`, `mcp_call` which we do not have at all**. `ask_user` is the whole abstention axis
they build the paper's story on.

**Stage 1 — SFT with a self-evolving data pipeline.** Three components:
1. *Navigation task generation* from three sources: **application manuals**, **expert-designed tasks**,
   and open-source datasets filtered for complexity and reachability.
2. *Trajectory synthesis*: seed task expansion (L1 changes parameters, L2 replaces core objects) then two
   parallel pipelines — **human annotation on a real emulator** and **model-based rollouts by several GUI
   agents**. Because many tasks have multiple valid paths, both sources are kept.
3. *Iterative rejection sampling*: train → deploy the model → roll out new trajectories → filter with an
   **MLLM-as-a-judge** → mix with fresh synthesis data → retrain. Loop. Failed trajectories are **reused
   as their longest correct prefix**, so a partial success is still training data.

**Stage 2 — online RL.** Two alternating phases on a **containerized AVD**:
* Rollout: policy interacts with the live GUI, collects full trajectories.
* Train: trajectory-level reward (binary task success from a rule-based verifier, or an MLLM judge that
  agrees with humans 83% of the time).
* **GRPO**, group size **16**, no KL, DAPO-style asymmetric clipping `ε_low=0.2, ε_high=0.3`,
  **action-repetition penalty** (single repeats and 3–5-action cycles), **experience replay** of the last
  8 successful trajectories per task so a group with no success still yields a signal.
* **Scale: 512 concurrent environment instances** on 10 ECS machines; trajectories up to **50 interactive
  steps** and millions of tokens, trained with Megatron TP+PP+CP hybrid parallelism.
* **Automatic curriculum** by difficulty stratum from the current policy's pass@K: frontier (0–25%
  success), exploration (25–50%), near-mastery (50–75%), exploitation (75–100%).

**Measured ablations they report:** scaling parallel environments 32 → 512 is worth **+5.2 points**;
raising the step budget 15 → 50 is worth **+4.3 points**.

## Where we differ — four gaps, in cost order

| # | Gap | Their setting | Ours | Cost to close |
|---|---|---|---|---|
| 1 | **Multi-step trajectory data** | 8,334 cross-app episodes (GUI-Odyssey) + expert tasks + rollouts | **15,000 single-step AC decisions, no trajectories** | dataset available (GUI-Odyssey, Apache/CC) |
| 2 | **Online RL** | GRPO, 512 envs, 50-step horizon | **none** | needs 500+ AVD instances — not feasible here |
| 3 | **Action space** | 12 incl. `answer`, `ask_user`, `mcp_call` | 8, no `answer` | `answer` = 2/116 tasks; `ask_user` is the abstention axis |
| 4 | **Judge + rejection sampling loop** | MLLM judge, iterative | **none** | needs an MLLM API or a local judge |

**Gap 1 is the one that matters most and the one we can actually close.** MAI-UI reports 83.4% on
GUI-Odyssey; GUI-Odyssey is 8,334 multi-step cross-app episodes over 212 apps, publicly available
(`OpenGVLab/GUI-Odyssey`, `OpenGVLab/GUI-Odyssey` on HF). We have been training on single-step decisions
and then evaluating on multi-step tasks — which is exactly the mismatch that makes a step-accurate model
fail an episode.

**Gap 2 cannot be closed on one 4090.** 512 concurrent emulators is a datacentre. Any honest plan has to
say so up front rather than imply we can match their number.

## Consequences for this programme

1. **The 49.1% bar is the target**, not "beat AC-v2's 64%". AC-v2 is retired per the operator's decision.
2. **Whatever we report must carry the model class.** A 1.92B single-pass SFT model with no RL is not
   competing with a 235B RL model; it is competing with MAI-UI-2B (49.1%) and Ferret-UI-Lite-3B (28.0%).
3. **The `app_intent` fix already in flight is a small patch, not a recipe.** It closes one read-slot
   failure; it does not close Gap 1 or Gap 2. Reporting it as the headline would be dishonest.
4. **The `ask_user` axis is where a new contribution is possible**: MAI-UI reports agent–user interaction
   is worth **+32.1 absolute** on their interaction subset (37.5% vs 5.4%). Our calibrated gate is a
   different mechanism for the same axis, and no GUI paper has published an outcome-level abstention
   result. That is the surviving opening — but it needs multi-step data (Gap 1) to be measurable.

## Sources

* MAI-UI: arXiv:2512.22047v1 (Tongyi Lab, Alibaba) — §2.1.2 Table 1 (action space), §2.2–2.3 (data
  pipeline, SFT, online RL), Fig. 5–6.
* MobileRL: arXiv:2509.18119 (THUDM) — Docker images `xuyifan0731/mobilerl-androidworld-eval`,
  `...-androidlab-eval`; android_world@`cac18549`, android_env@`1c4e8a92` plus their patches;
  "variations of 1–2% are not uncommon… averaged over three independent runs".
* GUI-Odyssey: arXiv:2406.08451, ICCV 2025 — 8,334 episodes, 6 devices, 212 apps, 1.4K app combos.
* AndroidWorld: arXiv:2405.14573 — 116 tasks, 20 apps; M3A 30.6%.
* Leaderboards read 2026-11-08: llm-stats.com, benchlm.ai, paperswithcode.co (all report Qwen3.8 Max 85.3%
  at rank 1, consistent with each other).

---

## Correction (same day, after reading MAI-UI Table 9)

The 49.1% above is the **post-RL** 2B number. Table 9 separates it from SFT, and the separation changes the
conclusion:

| model | SFT (e2e) | RL (e2e) | Δ | SFT (per-step) | RL (per-step) |
|---|---|---|---|---|---|
| MAI-UI-2B | **45.1%** | 49.1% | +4.0 | **92.3%** | 93.1% |
| MAI-UI-8B | 64.7% | 70.7% | +6.0 | 95.7% | 96.6% |
| MAI-UI-32B | 69.8% | 73.3% | +3.5 | 96.5% | 96.9% |

**Online RL is worth +0.8 per-step points at 2B.** Their SFT checkpoint is already at 92.3%. So my earlier
framing — "the gap is online RL on 512 emulators" — was wrong in the direction that matters: the gap is the
**supervision**, and RL is a refinement on top of it.

Our 64.07% per-step is **28 points below MAI-UI-2B's SFT checkpoint**, at the same scale and the same
SFT-only setting. That is the number this programme has to explain, and the explanation is the six missing
pieces listed above, of which trajectory data is the largest.

**And it changes the comparison class.** We are not competing with 76.7% (235B, post-RL). We are not even
competing with 49.1% (2B, post-RL). We are competing with **45.1% (2B, SFT only)** — and we do not reach it.
