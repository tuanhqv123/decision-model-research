# exec2: why AC-Low is still low, and the next direction (analysis of 15/11)

Every number below is measured from saved prediction files and checkpoints. Scripts and output: `benchmarks/results/2026-11-15-exec2/diag/`.
The measurements on intermediate checkpoints are **for diagnosis only**. They do not change the final checkpoint (`ckpt_7000`) and do not change the "not met" conclusion of `PRE-REG-2026-11-14-exec2.md`.

## 1. Summary

- **exec2's action skill is better than DMA-2B's.**
  - If declaring done is forbidden, exec2's AC-Low is 90.50, against 89.34 for DMA-2B.
  - Paired per step: exec2 is right on 227 steps where DMA-2B is wrong; conversely DMA-2B is right on 129 steps where exec2 is wrong. McNemar p = 2.3e-7.
- **The whole failure of condition 2 sits in the "done" decision.** There are 449 wrong done declarations. Of these, 292 fall on a step where the model would have been right had it not declared done.
- **80% of the wrong done declarations are premature by exactly one step** (359/449): the model declares done at the last step of an instruction, before doing that step.
- **Not a measurement error:**
  - With the same instruction and the same screen, AC-Low and the execution-role test give the same prediction (2,166/2,168 and 605/606 steps).
  - The training data has only 5 directly contradictory label pairs.
- **Training longer is unlikely to fix it.** On validation (not used for training), the number of wrong done declarations stays flat from step 4,000 to 7,000. Over the same range, the fit on the training data keeps improving.
- **No harness rule tried so far brings exec2 past 88.3:**
  - threshold τ: at most 87.08;
  - block done at the first step of an instruction: 87.81 (this rule was set after the test had been seen, for diagnosis only).

## 2. Breakdown of AC-Low errors (8,444 steps, InfiGUI scorer)

| | DMA-2B | exec1 | exec2 | exec2, done forbidden |
|---|---|---|---|---|
| Correct | 7,544 (89.34) | 7,097 (84.05) | 7,328 (86.78) | 7,642 (90.50) |
| Wrong done | 0 | 720 | 449 | — |
| Wrong action type | 284 | 126 | 152 | 266 |
| Right type, wrong element or value | 616 | 501 | 515 | 536 |

Correct by gold action (exec2 against DMA-2B):
- wait: 371/567 against 465;
- scroll: 1,076/1,211 against 1,096;
- click: 4,375/5,074 against 4,448;
- navigate_back: 304/343 against 335.

When done is forbidden, exec2 beats DMA-2B on every action type: wait 500, scroll 1,118, click 4,488, navigate_back 333.

Reaching 88.3 needs 7,457 correct steps, that is 129 more. For that, the number of steps lost to declaring done must fall from 292 to ≤ 185.

## 3. Where the wrong done declarations are

**How position is defined.** The position of a step is computed with the instruction-splitting rule of `build_exec_data.units`: compare the instruction after removing the trailing period, and break at `local_done`. By this rule there are 1,547 "instruction continuation" steps and 6,897 "instruction start" steps, as pre-registered. The figure 1,544 recorded on 15/11 came from `exec_report.py`, which compares raw strings.

| Group (exec2) | Count |
|---|---|
| Same instruction as the previous step, and this is the last step of the instruction | 222 |
| New instruction, model declares done before doing anything | 137 |
| In the middle of a long instruction, not yet the last step | 90 |

**Group 1 (222): after a click the screen is still loading.** Example: "Click on the gmail option" is a click, then a wait. After the click, the model treats the instruction as finished. The AC annotator keeps the old instruction for the wait step.
- 152/449 wrong done declarations fall on a step where the gold is wait.
- Rate of wrong done by previous step:
  - wait after a click that changed the screen: 96/350 (27%);
  - wait after a wait: 16/21 (76%);
  - scroll after a scroll: 41/598 (7%).

**Group 2 (137): a new instruction that is nearly the same as the previous one.** Example: "Click on the Time section" then "Click on the Time unit section", with p = 0.99. The rate of wrong done at the first step of an instruction rises with the word overlap with the previous instruction (Jaccard, common words removed):

| Word overlap | 0 | < 0.25 | 0.25–0.5 | ≥ 0.5 |
|---|---|---|---|---|
| Wrong done | 64/3,479 (1.8%) | 26/1,028 (2.5%) | 30/602 (5.0%) | 37/243 (15.2%) |

**exec2 against exec1, by gold action at an instruction continuation step:**
- scroll: 157 → 46;
- click: 101 → 55;
- navigate_back: 51 → 28;
- **wait: 209 → 151.** Up-weighting has almost no effect on the wait group.

At the instruction start step, click falls from 169 to 149.

**The model's confidence when it declares done wrongly:** p(terminate) ≥ 0.95 in 86 cases, 0.8–0.95 in 146 cases, 0.5–0.8 in 192 cases, < 0.5 in 25 cases.

## 4. Not a measurement error, not contradictory labels

- **Same input gives the same prediction.** At the steps where the execution-role test supplies exactly the AC-Low instruction, exec2 gives the same prediction on 2,166/2,168 instruction start steps and 605/606 continuation steps. The mismatches occur when p ≈ 0.5, caused by the trailing period of the sentence.
- **Why the execution-role test has fewer wrong done declarations.** There the sub-goal often consists of several instructions joined with ", then", so the model knows work remains.
- **Training data:**
  - A "done" screen and an action screen with the same instruction coincide in only 5 rows.
  - After a "done" row, the next action is almost never wait (50/14,963).
  - There are 2,380 wait rows that continue an instruction right after a click.
  - So the information "still loading means not done" is present in the data. The model has to recognise a loading screen by itself.

## 5. During training: intermediate checkpoints (for diagnosis only)

Two measurement sets:
- **val-low:** validation in the AC-Low style (690 action rows, not used for training).
- **train-150:** 150 training episodes that have instruction-continuation steps, scored with `score_exec.py`. exec2 saw each row once, and the continuation rows 3 times.

| ckpt | val-low correct | val-low wrong done | of which at continuation steps (/114) | train-150 correct steps | train-150 wrong done at last step (/482) | train-150 "done" recall | train-150 wrong done at continuation steps (/286) |
|---|---|---|---|---|---|---|---|
| exec2 1,000 | 500 | 68 | 33 | 73.3% | 58 | 81.5% | 47 |
| exec2 2,000 | 569 | 14 | 11 | 82.1% | 23 | 73.7% | 20 |
| exec2 3,000 | 567 | 25 | 20 | 83.0% | 32 | 79.9% | 34 |
| exec2 4,000 | 568 | 33 | 21 | 83.5% | 33 | 89.8% | 33 |
| exec2 5,000 | 568 | 34 | 21 | 86.2% | 26 | 89.8% | 24 |
| exec2 6,000 | 575 | 31 | 18 | 87.5% | 20 | 89.2% | 19 |
| exec2 7,000 | 575 | 30 | 18 | 87.5% | 18 | 90.3% | 18 |
| exec1 6,000 | 571 | 38 | 29 | 82.8% | 50 | 95.9% | 59 |

How to read the table:
- **Recall and wrong done trade off.** At step 2,000 the model has few wrong done declarations (14) but recall is only 73.7%. By step 4,000, recall rises to about 90% and wrong done rises to about 30.
- **On data it was not trained on, the model stands still from step 4,000.** Wrong done on val-low goes 33 → 34 → 31 → 30. On train-150, wrong done at the last step keeps falling 33 → 18. More training fits the data already seen more closely, but does not generalise better.
- **Loss on the training stream** (`train.log`; each sample is met for the first time, so it is close to unseen data): action loss 0.32 → 0.24 in the second half, flat over the last 10% when lr goes to 0.
- **Up-weighting works on data not trained on:** val-low at continuation steps falls from 29 (exec1) to 18 (exec2), but not enough.

## 6. Conclusion on the cause

What remains is not a pipeline error, not wrong labels, and not training that was too short. It is an ambiguous boundary between "the instruction is literally done" and the AC labelling convention:
- AC keeps the old instruction for the wait step, and for repeats such as scroll or back.
- AC writes consecutive instructions that are nearly identical in wording.

The model sees only one 720 px image and the element list. It has to guess "loading" from a single frame, and on data not trained on, that ability has saturated since about step 4,000.

## 7. Next direction (ordered by value over cost)

1. **Do not train more in the old way.**
   - More epochs: val-low is already flat, so no help is expected.
   - Up-weighting: a stopping rule already exists (prediction 1 was wrong).
2. **Give the model information to tell "loading" apart.** This is the largest error group: 152 cases where gold is wait, 222 in group 1.
   - The real harness on AndroidWorld reads state with `wait_to_stabilize`, so it rarely gives the model a loading screen.
   - There are two ways to measure this directly, from cheap to costly:
     - (a) Add a "screen just changed or already stable" signal to the history. The history currently records only whether the action changed the screen.
     - (b) Let the model see the previous image as well. This changes the input contract, so it costs more.
   - Both need a new pre-registration.
3. **Revisit how condition 2 is measured, but only for the next model, written before training.**
   - Condition 2 compares a model that must know how to declare done against DMA-2B, a model that never declares done on AC-Low. And it measures on a set that has no "done" step at all.
   - A cleaner split:
     - action skill = AC-Low with done forbidden (exec2 90.50 against DMA-2B 89.34, p = 2.3e-7);
     - timing of done = the execution-role test, which already has condition 1.
   - **This change does not apply retroactively to exec2:** exec2 stays recorded as not met. Changing the criterion after seeing the result is moving the goalposts, so this is the operator's decision, not mine.
4. **Measure on the real target.** The final goal is the LLM + decision model harness on AndroidWorld (section 8 of the exec1 registration).
   - If direction 3 is accepted, exec2 is a candidate for the harness comparison run.
   - If not, a new model is needed that meets all three conditions under direction 2.

Group 2 (a new instruction nearly identical to the previous one, 137 cases) is smaller. It depends on how the LLM writes sub-goals in the harness: the LLM rarely produces two nearly identical sub-goals in a row. Not a priority.
