# A better way to replace self-play (2026-11-11)

All our own measurements are in `benchmarks/results/2026-11-11-v3/postmortem/`. The figures from papers were read by an agent
from the arXiv pages on 2026-11-11. They are **their numbers, on their benchmarks**, and must be checked against the PDF before they go into the paper.

## 1. Why the last self-play round was the weaker method (measured)
- Each task was run **once**, always taking the highest-probability option. Result: 9/54 tasks succeeded, of which **8 were tasks v3
  could already solve**, and only **1 was new** (OsmAndMarker) (`selfplay_value.out`). Retraining on that mostly re-learns what is already known.
- The model is deterministic: the same screen gives the same decision. Running it again the same way gives the same result, so nothing is explored.

## 2. Two measurements show the right direction
| Measurement | Number | File |
|---|---|---|
| AC-High, the correct answer lies in the model's top-k | action type: top1 81.0 → top2 **93.5** → top3 97.8% · click element: top1 60.4 → top3 **73.9** → top5 77.1% (ceiling 83.3%) | `topk_actest.out` |
| Draw one sample by probability (T = 1) | 51.2%, **worse** than taking rank 1 (60.4%) | `topk_actest.out` |
| Does the confidence predict that a click changes the screen (AndroidWorld) | by action type AUROC 0.698 · by element 0.553 (close to guessing) | `conf_noeffect.out` |

What this means:
- Most of the model's errors sit in the **2nd–3rd choice**. Trying the top choices may find new successful tasks.
- Free sampling hurts, so sampling must be **limited to the top-k**.
- Confidence about the element predicts almost nothing, so we should **not** branch only at the steps where the model is less confident.

## 3. Evidence from papers (no external model, only the AndroidWorld checker)
| Paper | Numbers they report | How it is close to ours |
|---|---|---|
| AppVLM 2502.06395 (3B) | AW: AndroidControl-only training 11.4 → retraining over several rounds on runs confirmed by the checker (RFT) 17.9 / 23.2 / 30.5 / 35.0 → **37.8** (Tab. 2–4) | Closest: starts from AC only, ~11% (ours 14%), no external model, each task is run **several times** |
| UI-Voyager 2603.24533 (4B) | RFT, 3 rounds: 37 → 73. GRSD (find the branching step between a failed run and a successful run of the same task, then learn the correct step there) **81.0** versus GRPO/PPO 76 | The branching step is found by screenshot matching (SSIM), no LLM needed; learning uses likelihood only, which fits our selection heads |
| Generalization in Online RL 2603.07432 (7B) | +26.1% on unseen instances, but only **+15.7%** on unseen templates and **+8.3%** on unseen apps | **Main risk:** the gain mostly sits inside the templates that were learned |
| MobileRL 2509.18119 | 7B: SFT 56.8 → RL 72.0. Removing positive replay loses 7.5 points | Needs up to 256 parallel runs; we have 2 emulators |

Correction to the record: in `AGENTS.md`, "StepReflect 2608.05587, 882 transitions, 41.4%" is wrong. The paper is named
**Q-Reflector**. It uses 880 transitions, 41.82% negative, labels made by GPT-5.2, and it does not satisfy the "no external model" constraint.

## 4. Proposed method: repeated runs, top-k exploration, learning at the branching step
1. **Bounded exploration:** each practice task is run k times. At each step, sample only within the top-3 of each selection head (taken
   from the probabilities the server already returned, no change to the model). The first run keeps the rank-1 choice as the reference.
2. **Labels come only from the AndroidWorld checker and from whether the screen changes:**
   - successful run: relearn the steps taken (imitate);
   - **branching step** between a failed run and a successful run of the same task: learn the correct step on exactly that screen (GRSD style);
   - correct `status` (the checker says it is done) and wrong `status` (reported done when not done);
   - step with no screen change: learn "do not choose this".
3. **Retrain** (v4 = v3 data + new data), then repeat the loop.
4. **Guard against template contamination:**
   - set aside one third of the AndroidWorld templates and never learn from them;
   - never learn on seed 30;
   - always report two numbers: learned templates and unlearned templates.

## 5. Decision gate before spending effort on retraining (no retraining needed)
**Measure pass@k:** about 30 tasks that v3 currently fails, practice seeds (not 30), k = 4 top-3 exploration runs per task. Run on 2
emulators, estimated at about 3 hours.
- **Go ahead** if the number of tasks with at least 1 success rises by **≥ 5 tasks** over taking rank 1. Then the RFT round has new data
  to learn from.
- **Stop this direction** if the rise is < 5 tasks. Then try changing the input first: remove the cut at 40, check the Progress line, and learn "type text
  after clicking the input field".
- **Prediction made in advance (so that it can be wrong):** tasks with a budget > 20 steps stay at 0, because per-step accuracy compounds with the number of steps.

## 6. Cheap work to run in parallel (no emulator needed)
- Check the `Progress` line: use exactly the self-play input that was logged, change only the step count, and measure how much the `status` probability changes.
- Remove the cut at 40: rerun the 656 AC-High rows that have more than 40 candidates.
