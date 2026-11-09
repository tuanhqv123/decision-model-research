# Plan 2026-10-29 — d1-2B decision-model agent vs AC-v2 SOTA

## Goal (fixed)
Beat a published AC-v2 398-step leaderboard point with a **single-pass decision model** (d1-2B, 0 generated
tokens). Primary bar: **Mem-W-8B 68.59**; secondary: GUI-Libra-8B 64.3 / Mem-W-4B 63.07. Report two tracks:
- **V**  = screenshot + goal + history (the exact leaderboard protocol), box readout for the target.
- **VA** = V + a11y candidate list (choice over K) — a separate, a11y-assisted track, never mixed with V.

## What research changed (2026-10-29, verified in code)
1. **The leaderboard protocol includes history.** GUI-Libra `guilibra_plan.py` puts `row['previous_actions']`
   (earlier step instructions; len == step on 398/398 rows) into the prompt as "Interaction History". Our old
   "goal-only" premise was wrong and handicapped every earlier run. State now carries it.
2. **Leaderboard scorer = `eval_rl.py`**; diffed against our `score_row` port: equivalent, except it reads no
   point for non-spatial predictions → eval_d1 passes pt=None for non-spatial actions (conservative).
3. **Data leakage in the old curated/completion sets**: they contain AndroidControl TEST episodes (ep 27 …).
   Rebuilt train data from `gs://gresearch/android_control` (13,603 train eps → 88,288 decisions incl. 13.6k
   synthesized `status` rows from final screens; 350 eval episodes asserted absent).
4. Literature (GUI-Actor, V2P, LiMAC, AC-Curated): element-box targets with sharp soft labels; full FT >> head-only;
   completion is the largest lever (22% of eval); no published single-pass agent result on AC-High exists.

## Method
- Forward = the checkpoint's own serving path (`_sequences` → `set` attention → slot/pointer readouts) with grads.
  Gates passed: **parity** (train log-probs == `system_one_batch`, 24/24, both tracks), **grad-ckpt**
  (rel grad diff 6e-4; backward runs inside the attention context), **ckpt reload** (save → load → eval).
- Losses: action CE (8 options incl. terminate); box edges CE vs Gaussian(σ=4/1000) on the gold element's edges;
  target_el CE (VA); scroll CE; pointer start/end CE for app_name/type_text spans.
- App-name memory: TF-IDF retrieval of app names from TRAIN goals (leave-own-episode-out) — disclosed, ablatable.
- Full FT, bf16 + torchao AdamW8bit (bf16 stochastic rounding), lr 1e-5 cosine, grad-ckpt.
- **3-GPU local SGD (DiLoCo, outer lr 1):** 3 replicas on disjoint data shards, weights averaged every 300 steps
  (`sync_avg.sh` hub on Model_Training_Server). ~1 epoch in ~4h instead of ~11h on one 4090.

## Pass conditions
- P1: V-track step acc > 45.48 (Qwen3-VL-2B, same size class).
- P2: V-track > 63.07 / 64.3; P3: > 68.59 (SOTA). VA-track reported separately.
- Always shown with trivial baselines (always-status 21.86%) and ECE/coverage-risk (our mechanism axis).
