# N1 pilot — does the training objective change calibration? (2026-11-02)

**Answer at this budget: no. CE, Brier and joint are statistically indistinguishable on accuracy and
discrimination, and all three are strongly overconfident.** Reported as a null, with the baseline shown.

## Design (one variable: the readout objective)
- Data: curated AC-v2 decisions rebuilt for this run, **12,750 train / 2,250 holdout** (85/15, seed 0).
  The holdout is in-distribution; it is **not** the AC-v2 leaderboard test set.
- Base: `LiquidAI/d1-2B` @ `cfe5760f…`, V-track, `--freeze-embed --grad-ckpt --micro 1 --accum 16`,
  lr 1e-5, **250 optimizer steps** (≈4,000 samples seen), seed 0, `--no-memory`.
- Arms: `--objective ce` (released recipe), `brier` (proper score on the option readouts),
  `joint` (CE + 0.5·Brier). Box/pointer/read heads unchanged across arms.
- Scored in the serving path (`model.system_one_batch`, zero generated tokens) with the scorer port
  that was differential-tested 398/398 against the official evaluator.

## Result (2,250 holdout decisions each)
| Arm | Step acc | Type acc | Grounding | ECE (10 bins) | Brier | AUROC(conf,correct) | Mean conf | Acc when conf ≥ 0.9 |
|---|---|---|---|---|---|---|---|---|
| **CE** | **0.5658** | 0.8084 | 0.5425 | **0.2370** | 0.3387 | **0.6570** | 0.8027 | 0.698 (n=1000) |
| **Brier** | 0.5653 | 0.8076 | 0.5508 | 0.2425 | **0.3291** | 0.6468 | 0.8079 | 0.680 (n=952) |
| **Joint** | **0.5707** | 0.8084 | 0.5557 | 0.2423 | 0.3305 | 0.6532 | 0.8130 | 0.704 (n=997) |
| Majority class (`click`) | **0.5218** | — | — | — | — | — | — | — |

## Reading
1. **Accuracy is flat across objectives** (0.565–0.571); the spread is smaller than any plausible
   interval at n = 2,250 (±~2 points). The objective is not an accuracy lever at this budget.
2. **Brier improves slightly** (0.339 → 0.329–0.331) while **ECE gets slightly worse**
   (0.237 → 0.242) and **AUROC slightly worse** (0.657 → 0.647–0.653). A proper score improves the
   *average* of the probability while leaving — or slightly worsening — the *ranking* of correct vs
   incorrect decisions. This is the same dissociation the gap analysis warned about: average calibration
   and informativeness are different properties.
3. **Everything is overconfident by a wide margin.** Mean confidence ≈ 0.80 against accuracy ≈ 0.57 —
   a ~0.23 gap, essentially the ECE. At conf ≥ 0.9 the model is right only ~70% of the time. Any
   act/ask/refuse gate built on this readout would be badly over-trusting.
4. **Accuracy is only +4.4 points over always-predicting-`click`.** At this budget the head is close to
   a prior-plus-grounding solution, not a strong policy.

## Why this is not yet the paper's N1
- **One seed.** The gap analysis requires ≥3 seeds and paired CIs; single-seed differences of this size
  are noise.
- **250 steps**, deliberately reduced to fit one evening on a shared GPU. The released model used 938.
  A budget-matched conclusion needs the full schedule.
- **In-distribution holdout.** The claim under test ("an informative confidence under shift") needs the
  hard/OOD split, which is the next step.
- **Joint λ = 0.5 chosen arbitrarily**, not tuned; a different λ could move the Brier arm.

## What the result does support
- The measurement pipeline (train → serve → score → ECE/Brier/AUROC/majority baseline) is correct and
  reproducible, and it produces the trivial baseline alongside every number.
- The objective comparison is **flat enough that a full-budget, 3-seed run is the only way to resolve
  it** — which is exactly the experiment the paper needs, and it is now one command.

## Artifacts (on `Model_Training_Server`, backed up locally before cleanup)
`runs/n1m-{ce,brier,joint}` checkpoints · `results/n1-n1m-{ce,brier,joint}.json` ·
`logs/n1m.log`, `logs/n1eval.log`, `logs/evjoint.log`.

## Reproduce
```bash
cd ~/d1-api
for obj in ce brier joint; do
  venv/bin/python train_d1.py --data acdata --train-set ftr --out runs/n1m-$obj --track V \
    --lr 1e-5 --freeze-embed --grad-ckpt --micro 1 --accum 16 --steps 250 --seed 0 \
    --objective $obj --brier-lambda 0.5 --no-memory --mem-frac 0.5 --save-every 250
  venv/bin/python n1_analyze.py --ckpt runs/n1m-$obj/ckpt --data acdata \
    --valfile compact_fva.jsonl --tag n1m-$obj
done
```
