# N1 — objective ablation: CE vs proper score vs joint (2026-11-02)

**Why this is the paper spine (from `outputs/gap-analysis-calibrated-single-pass-2026.md`):**
"single-pass zero-token GUI decision model" is taken (`Jev-Mobile` 2609.30186, `GUI-JEPA` 2610.36877).
What is **not** taken is the mechanism question: *which training objective makes the single-pass
confidence an informative signal (not merely average-calibrated), and does it hold a risk budget under
shift?* N1 is exactly that one-variable comparison on identical data/model/splits.

## Design (one variable: the readout objective)
| Arm | Flag | What changes |
|---|---|---|
| A (baseline) | `--objective ce` | released recipe; unchanged behaviour when the flag is absent |
| B (proper score) | `--objective brier` | option readouts minimise Brier instead of CE |
| C (joint) | `--objective joint --brier-lambda 0.5` | CE + 0.5·Brier (CAAG-style) |

Everything else identical: data, base checkpoint, steps, LR, seed grid, `--freeze-embed`, `--track V`.
Box/pointer/read heads keep their existing losses in all arms (only the **categorical option** readouts
change), so we isolate the decision confidence, not grounding.

## Fixed configuration
- Data: `compact_cur15k.jsonl` (15k curated AC-v2 decisions, same file for every arm).
- Base: `LiquidAI/d1-2B` @ `cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41` (pinned in `train_d1.py`).
- `--track V --lr 1e-5 --freeze-embed --grad-ckpt --micro 1 --accum 32`.
- Steps: **400** (not the 938 of the full run) — a deliberate budget compromise; state it in the paper.
- Seeds: **0,1,2** (3 seeds minimum for a top-venue ablation; drop to 2 only if compute forces it and say so).

## Metrics (report ALL, per seed and pooled with CIs)
1. **Accuracy** (step accuracy on the 398-step AC-v2 eval) + Wilson CI.
2. **ECE** across a bin grid (10/15/20 bins) **and** Smooth ECE — ECE alone is binning-sensitive.
3. **Brier score + Murphy decomposition** (reliability / resolution / uncertainty).
4. **AUROC(confidence, correctness)** — the informativeness signal, the actual claim.
5. **Selective risk**: risk at coverage 50/70/90%, plus the τ that meets a 20% / 10% risk budget.
6. Majority-class baseline (`click`) and uniform-random next to every accuracy number (VERIFICATION GATE).

## Expected (pre-registered, from 2610.01006 / 2610.19811)
- CE: AUROC ~0.62 (our measured 0.626/0.631 offline), confidence rises on no-signal items.
- Proper-scoring / joint: AUROC ≥ 0.75 on OOD, and confidence does **not** rise on no-signal items.
- If that separation does not appear, report the null honestly — a null on N1 collapses the mechanism
  claim, which is better known before writing than after.

## Commands (run on the server, one arm per process; never two on the same GPU)
```bash
cd ~/d1-api            # venv + train_d1.py + ac_common.py already there
export PATH=$HOME/.local/bin:$PATH
for seed in 0 1 2; do
  for obj in ce brier joint; do
    python train_d1.py --data acdata --train-set cur15k --out runs/n1-$obj-s$seed \
      --track V --lr 1e-5 --freeze-embed --grad-ckpt --micro 1 --accum 32 \
      --steps 400 --seed $seed --objective $obj --brier-lambda 0.5
  done
done
```
(NB: `--data` points at the directory holding `compact_<train-set>.jsonl`; upload the file first.)

## Analysis
`benchmarks/train/analyze_eval.py` already re-scores a jsonl; the AUROC/ECE/coverage-risk table comes
from `benchmarks/calib_metrics.py`. Add the Brier/Murphy decomposition if not present, otherwise report
Brier + AUROC and say the decomposition is omitted.

## Compute reality
6-9 runs × ~40 min (400 steps) ≈ **4-6 h** on one 4090. If the emulator loop is also running, sequence
them: emulator smoke first (needs the d1 API served), then N1 in the free GPU window.
Second GPU option: `GPU_HOST_B` (15 GB free; use `--mem-frac 0.55 --micro 1`). Prod shares that card.
