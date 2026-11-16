# Final summary (2026-11-12): what to fix, what we learned, what we apply

Every number below is computed from a file and comes with its denominator. The numbers in section 1 were recomputed by an independent agent (read-only, without knowing the conclusion beforehand). The detailed analysis is in `2026-11-12-ACHIGH-vs-AW-FINAL.md`.

## 1. Measured state

- **AndroidWorld** (v4 + harness V12, seed 30): **25/115 = 21.74%**. It consists of 24 full-score tasks and 2 half-score tasks; OsmAndMarker had a setup error and was removed from the denominator.
- **Failed tasks:**
  - 49 tasks reported "done" when not done, plus 2 half-score tasks;
  - 40 tasks ran out of step budget, of which 2 tasks had the device **already in the right state** but the model did not stop.
- **AC-High** (same v4 model): 61.7% per step. But only **344/1,543 = 22.3%** of tasks are right at every step, even though the model is given the human's history at every step. For tasks with 6 or more steps, this figure drops to 59/628 = 9.4%.

## 2. What we learned

| # | Lesson | Evidence |
|---|---|---|
| L1 | AC-High is a **per-step score, with the human's history**, not a task-completion score. Counted per whole task it equals AndroidWorld (22.3% against 21.7%). | `eval_d1.py:42,48-49,220-225`; the scorer does not check the goal (`vendor/infigui_eval.py:83-122`) |
| L2 | The most missing skill is **stopping at the right time**. No stop step is the correct answer on the AC test, yet the model still demands to stop at **1,171/8,444 steps**, spread over 546 tasks. Of these, 460 fall exactly on the last real operation step; by the gold answer, 566 should have been a swipe and 420 a click. | `perstep-v4_s0_ckpt_2482_actest_v4.jsonl` |
| L3 | **v5's deduplication step destroys exactly the data that teaches stopping.** The pair "last operation + `status`" was complete in 4,191 tasks before curation, but only 1,918 remain after curation. | counted on `compact_awlike_full_v5` / `compact_cur_v5` |
| L4 | First error on AC-High: **wrong action type 684**, wrong target or value 515 (of which click on the wrong target 435), early stop 154. | same file |
| L5 | **AMEX** reports only per-step scores and did not run online. The file has BACK at only 0.35%, contrary to the paper's claim. AMEX's operation chains transfer poorly to other data (56.2 against 76.3). | file + paper |
| L6 | **AndroidControl-Curated / Magma-R1** (2510.18488): | HF `batwBMW/AndroidControl_Curated`; GitHub `src/eval/` |
| | • Only the **cleaned test set** is released: 8,435 rows, 1,540 tasks; there is no train set. | |
| | • The paper says the 2,400 train samples were *chosen from AndroidControl-Curated itself*. | |
| | • The history in the file is a **per-step description written by a human**. | |
| | • In the `task-improved` file, the history **contains the current step's instruction itself** in 8,377/8,377 rows, and the eval script feeds the history into the prompt as is (`evaluate_actions_androidControl_vllm.py:110,125`, `prompt.py:5`). | |
| | • Their "SR 75–80%" figure is an AC-style per-step score, not task completion on AndroidWorld. | |
| L7 | Worth learning from the two papers: **quality over quantity**; **weight by action type**; **screen-understanding data** helps grounding. | |

## 3. What is applied in the final training ("final")

| Problem | How final fixes it | Status |
|---|---|---|
| Early stopping (L2) | **New:** keep every pair "last operation + `status`" (4,191/4,191), with the test `test_stop_pairs_survive_dedupe_and_click_cap` | **new, fixes L3** |
| Early stopping when a task has several parts | Join 2 tasks into a chain (491 chains): at the screen where part A is done, the answer is `navigate_home`, not `status` | present since v5 |
| Repeating when a tap changes nothing | The `nc_*` rows ("just tapped and nothing changed → do something different"): 3,627 rows | present since v4 |
| Not knowing how to go back | 3,468 `recover_back` rows, each with one control row (`recover_twin`) | present since v5, **not yet measured separately** |
| Elements with no label, on/off switches | Contract v5 shows `[id: …]` and on/off | present since v5 |
| Duplicated and imbalanced data (L7) | Each (screen, action, target) keeps at most 2 rows; click ≤ 45% | present since v5 |
| Numeric answers | Harness V13: `answer_format` (+1 task when re-scored offline) | harness |

**Final data:** `compact_cur_v6` has 38,195 rows, plus 1,736 `awgen_v4` rows.
- **Action ratios:** click 45.3% · status 16.3% · back 11.7% · type 7.1% · open_app 6.0% · scroll 5.2% · wait 4.5%.
- **Train:** initialised from v4 ckpt_2482, **2,496 steps × 16 = exactly 1 epoch** on 39,931 rows, lr 5e-6, checkpoint saved every 250 steps.
  - The first version ran 1,000 steps, meaning it learned only 40% of the rows; it was restarted at step 60.
  - v4 also trained exactly 1 epoch. Unsloth recommends 2 epochs, or 3–4 if data is scarce.

## 4. What final does not fix, and why

- **Apps never seen** (62/95 failed tasks are in 7 apps that are not in AndroidControl): fixing this requires data from the AndroidWorld apps themselves, which means learning the test in advance. That is forbidden (no self-play, no app exploration). AMEX does not have these apps either.
- **Wrong labels in AndroidControl train:**
  - the cleaned set only has the test part;
  - filtering with our own models would also delete the hard samples;
  - using an external LLM to fix labels violates the constraint.
- **AMEX-style screen-understanding data:** AMEX has no class/flag/id so its format differs from v5, and it does not have the AndroidWorld apps.

## 5. Check gates (pre-registered; if they fail, the final version is v4 + V13)

1. **Early warning at ckpt_250** (run the validation eval on another GPU).
   - Stop the run if P(back | gold ≠ back) exceeds v4's level by more than 0.02.
   - Stop the run if the number of times the model reports `status` when the answer is not status exceeds v4's level.
2. **Gate 1** (final checkpoint, against v4, same file and same scorer). All four must be met:
   - AC-High: number of tasks right in full ≥ 344/1,543, and McNemar by task does not lose;
   - the number of `terminate` drops clearly from 1,171/8,444;
   - validation: the number of tasks right at every step does not drop;
   - P(back | gold ≠ back) ≤ v4's + 0.02.
3. **Gate 2:** run 13 AndroidWorld tasks aimed at the errors.
   - early stop: 6 tasks;
   - no stop or out of steps: 3 tasks;
   - keep the tasks that are already right: 4 tasks.
4. **Final number:** run all 116 tasks with harness V13, on 6 emulators, paired with v4 + V12 (21.74%).

## 6. Comparison with the Unsloth guide "train your own decision model" (read 2026-11-12)

This guide turns an ordinary LLM (Qwen3.5 / Gemma 4 / Llama 3.2) into a decision model:
- attach a Clef-style head that scores each choice after one read, without generating text;
- train with LoRA r16, lr 2e-4, head lr 1e-4, 2 epochs;
- data in the form `state` / `questions` / `gold`;
- **calibrate the probabilities on held-out data** after training.

The guide says nothing about images or vision.

| Point | Unsloth | Ours | Conclusion |
|---|---|---|---|
| Data format | state + questions + gold, one read | same | correct |
| Starting model | ordinary LLM, new head | d1-2B is already a decision model, head included | different starting point, not an error |
| LoRA or full | LoRA (so the new head learns from scratch) | full fine-tune, lr 5e-6 | both valid; a full-FT lr smaller than a LoRA lr is normal |
| Number of epochs | 1–2, 3–4 if data is scarce | 0.4 epoch | **fixed to 1 epoch** |
| Calibration after training | `calibrate()` on holdout | not done (v4's action ECE on validation is 0.19) | only affects confidence, does not change the chosen action (argmax) because the done gate is off. Will calibrate on validation after gate 1 |

## 7. Does the model use the image? Try replacing the image with a grey image (2026-11-12)

Same v4, same 827 validation rows. The second run replaces every screenshot with a grey image of the same size. Files: `benchmarks/results/2026-11-12-imgablate/`.

| | Real image | Grey image |
|---|---|---|
| Right per step | **519/827** | **337/827** (−22 points) |
| Tasks right at every step | 14/137 | 6/137 |
| click | 234/420 | 121/420 |
| type_text | 37/53 | 3/53 |
| scroll | 40/86 | 7/86 |
| navigate_back | 18/29 | 7/29 |
| wait | 22/56 | 39/56 (when it sees nothing, the model chooses to wait) |
| open_app | 45/45 | 45/45 |
| Action-type selection accuracy | 80.0% | 50.8% |
| Grounding | 65.5% | 59.9% |

On 216 rows, the model is right with the image and wrong with the grey image. The reverse happens in only 34 rows.

**Conclusion:** the model **uses the image a lot**, mainly to decide *what to do*: sees a keyboard so it types, sees a long list so it swipes, sees that the screen is done. The choice of *which element* depends less on the image (65.5% → 59.9%), because the element list is already available as text.

**What can be done later** (not part of this training run): the image is at 720 px width, while d1 supports up to 858 px. To use 858 px, the AndroidControl image store must be rebuilt from the original, because the stored images are only 720 px.

## 8. AndroidGen (arXiv 2504.19298; HF `zai-org/androidgen-llama-3-70b`), read 2026-11-12

**What AndroidGen is:** an LLM agent that calls the model several times per step and reads text only (a11y tree, no image), writing out code-style commands, with chain-of-thought. It has four modules:
- **ExpSearch:** takes the 1 most similar successful trajectory as an example;
- **ReflectPlan:** a separate LLM call to build and update the list of sub-goals;
- **AutoCheck:** a rule-based check, no LLM;
- **StepCritic:** GPT-4o filters trajectories.

**Results on AndroidWorld:**

| Configuration | SR |
|---|---|
| GPT-4o | 46.8 |
| Llama-3-70B fine-tune | 35.3 (easy 55.8 / medium 12.5 / hard 5.9) |
| GLM-4-9B fine-tune | 29.2 |

The paper does not state the number of tasks, seeds or step budget.

**Ablation:**
- ReflectPlan +11.7; ExpSearch +12.6; AutoCheck +1.8.
- Filtered data against unfiltered: 25.2 against 16.2.

**Why it cannot be compared with ours:**
- **The training data was collected inside AndroidWorld itself:** about 300 tasks written by GPT-4o based on the AndroidWorld tasks, run on the AndroidWorld emulator.
- **ExpSearch adds the eval runs themselves to the retrieval store** (`androidgen.py`; `allow_same=True`).

By our rules, both of these are learning the test in advance.

**Ideas that fit our constraints:**
- **"Goal-prefix data augmentation"** (Algorithm 1): the same screen is "done" for a short goal but "keep going" for a long goal.
  - The final version already has exactly this pair, thanks to two things:
    - **chain-joining rows:** at the done screen of task A, the goal "A then B" has the answer `navigate_home`;
    - **task A's own stop pair:** at the same screen, the goal "A" has the answer `status`, and it was kept in full, 4,191/4,191.
- **Rule-based checks (AutoCheck):** mostly already in the harness (the gate that selects from the list, the check that the text went into the field). The gain they measured is small (+1.8).

**Cannot be applied:**
- ReflectPlan: needs a planner that generates text.
- ExpSearch as they do it: uses a retrieval store that also holds test data.

## 9. Trace analysis of all 91 failed tasks (v4 + V12) and the final decision

Four agents read the traces and screenshots independently, one group each. The per-task table is in `benchmarks/results/2026-11-12-failanalysis/*.tsv`. Every task has its first wrong step recorded, what the model did, what it should have done, and the evidence.

**Environment error (not a model error).** The emulator **aw-emu3** ran all 6 Markor tasks while the app was on the intro screen. Fixed with `hostscripts/app_setup.py`: only when the intro screen is seen does it run the official `MarkorApp.setup`. From now on `run_arm_v2.sh` refuses a container that is still stuck on that screen.

VLC on **every** emulator shows "GO TO MEDIALIBRARY PREFERENCES" and the list stays at "Loading." forever. Because it is the same on every machine, it is not a single machine's error; it still has to be investigated.

**Early check at ckpt_250 of final run 1** (v5 data + stop pairs): wrong go-back 44/798 (v4: 16/798), P(back | gold ≠ back) 0.077 (v4: 0.025). This exceeds the pre-registered rule (+0.02), so the run was stopped. This is the second time that go-back data did harm.

| Group (failed) | Main cause (number of tasks) | Fix in final (run 2) | Blocked by the constraint |
|---|---|---|---|
| Information questions (21) | Could answer but did not answer: 4 · format: 1 · went to the wrong place: 4 · date/time/colour reasoning: 4 · counting across several screens: 2 · answer mechanism cannot be represented: 6 | +350 answer tasks; V13 `answer_format` (+1) | Date/time, colour, addition/unit conversion, multi-item answers |
| Easy (27) | Needs long-press: 5 · Verify tasks already right: 3 · early stop: 4 · wrong element chosen: 4 · target off-screen: 3 · typing always clears the field: 2 · clipboard filtered: 2 · environment/app knowledge: 4 | Long-press ×4 · row "if already right then report done" (154) · stop pairs · same-row text | Hints on the keyboard (AndroidControl has no label), brightness is not in the a11y tree, relative dates, append typing |
| Medium/hard (34) | Wrong way of operating the app: 7 · values that must be computed: 6 · finished item 1 then stopped: 4 · Markor environment error: 3 · form field hidden by the keyboard: 3 · duplicates: 3 · note content: 2 · VLC: 2 · odd ones: 4 | Long-press · same-row text · task chains + stop pairs · environment fix | Date/time computation, duplicate detection across several screens, content longer than 40 characters |
| Image (9) | Wandered off: 2 (SimpleDraw, SaveCopy) · Browser: 2 · text only in the image: 4 · multiplication: 1 | Long-press → "Copy to" | The d1 questions `text`/`number`/`date` are a pointer that can only read the text part of the input (`pointer.py`), so it cannot read text inside the image |

**Total:** about 36/91 tasks are blocked by the constraint (one model, no planner, AndroidControl data only). Final run 2 targets the rest.

**Not included, with reasons:**
- **A separate "is it done yet" question:** measured on 2026-11-10 and found to coincide with the action question.
- **Repeat mask:** measured, 2 wins 2 losses, and early stops went up.
- **Hints on the keyboard:** ids in AndroidControl are hidden so there is no data to learn from.
- **Hide the keyboard after typing:** is a harness change, will be tried on the form tasks without retraining.
