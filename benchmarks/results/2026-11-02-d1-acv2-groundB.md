# 2026-11-02 — d1-2B grounding run groundB (858px + hit loss + aux grounding) → AC-v2

> This file fills the gap where the groundB run result (64.82%) existed only in a session transcript.
> Protocol = GUI-Libra `eval_rl.py` (pure, relative_coord=0), AC-v2 398 steps, Pass@1.
> Comparisons: E2 (720px) is the previous run; differences are image resolution + loss changes only.

## Run config (groundB)
- Base: `LiquidAI/d1-2B` (1.92B, LFM3-VL-2B + 164M ViT + 25M projector), pinned rev cfe5760f.
- Data: cur15k (15,000 curated AC decisions, same md5 as E2) + 7,500 aux grounding samples (no extra images).
- Images: **858px** (fixed, d1's native training resolution) vs E2's 720px.
- Loss additions vs E2: (a) **hit** loss — optimize probability the click lands inside the correct element;
  (b) aux grounding supervision ("describe step → element", "label → element", UGround/Aguvis style).
- Freeze: tied embedding/LM head; lr 1e-5; 1,406 steps on one RTX 4090 (~4.8h w/ eval).

## Result (official `eval_rl.py`)

| Metric | E2 (720px) | groundB (858px) |
|---|---|---|
| **Step accuracy** | 64.07% (255/398) | **64.82% (258/398)** |
| Type | 84.17% | 82.16% |
| **Grounding** | 56.05% (125/223) | **60.09% (134/223)** |

## Interpretation (honest)
- Grounding improved (+8 correct clicks; wrong-place misses 71 → 58) and overall step acc moved 64.07 → 64.82 (255 → 258, +3), but **action-Type accuracy dropped** (84.17 → 82.16, −8 type rows) — the grounding gain is only partially offset by the type loss. The headline moves +3 rows (+0.75 pts), not +0.
- Three different SFT recipes (720px / 858px+hit / 858px+aux) all converge to ~64.3–64.8. **Ceiling of the
  SFT framing on AC-v2.** Remaining gap is **element selection (semantic) + attention dispersion**, not
  pixel precision — see `notes/grounding-precision-2026.md` for the evidence-backed path (V2P / ConClick,
  training-time, NOT RL).
- **Caveat:** step accuracy = ties GUI-Libra-8B (64.3) within CI [59.5, 68.8]; NOT a SOTA break
  (Mem-W-8B 68.59 ~4.5 pts ahead). Final checkpoint was **not** selected on test.

## Artifacts
- Checkpoint: `backup-2026-10-29/ckpt-groundB/ckpt` (model.safetensors 3,836,417,920 bytes, matches E2 size).
- Training ran on VKong RTX-4090 (~$0.38/h); images (858px) cached in storage `acv2-a` (data858.tar, 2.5GB).
- Loss-change code + gates (hit loss parity, sample grounding) live in `benchmarks/train/`.

<!-- Numbers corrected 2026-11-07 after the internal audit: 64.32/256 -> 64.07/255; 56.50 -> 56.05 pure. -->
