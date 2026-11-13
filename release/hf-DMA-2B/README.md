---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: LiquidAI/d1-2B
base_model_relation: finetune
language:
  - en
tags:
  - decision-model
  - single-pass
  - calibration
  - gui-agent
  - mobile-agent
  - android
  - androidworld
  - androidcontrol
  - custom_code
model-index:
  - name: DMA-2B
    results:
      - task:
          type: gui-action-selection
          name: AndroidControl-High (step success rate, InfiGUI-R1 scorer)
        dataset:
          type: androidcontrol
          name: AndroidControl test, high-level goals (8,444 steps)
        metrics:
          - type: step_success_rate
            name: SR
            value: 61.7
      - task:
          type: gui-action-selection
          name: AndroidControl-Low (step success rate, InfiGUI-R1 scorer)
        dataset:
          type: androidcontrol
          name: AndroidControl test, low-level instructions (8,444 steps)
        metrics:
          - type: step_success_rate
            name: SR
            value: 89.3
      - task:
          type: gui-agent
          name: AndroidWorld (task success rate, seed 30, pass@1, our harness)
        dataset:
          type: androidworld
          name: AndroidWorld (116 tasks)
        metrics:
          - type: task_success_rate
            name: TSR
            value: 21.6
---

# DMA-2B: a single-pass decision model for Android GUI agents

**DMA-2B** (Decision Model for Android, 2B) picks the next action of an Android GUI agent in **one forward pass,
with zero generated tokens**. It is a full fine-tune of [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B), a
1.92B vision-language *decision model*: instead of writing an action as text, it answers typed questions (which
action, which on-screen element, which app, which scroll direction, which span to type) and returns a probability
for every option.

What this gives an agent:

- **No parsing.** Every answer is one of the listed options, so every decoded action is a valid AndroidWorld
  `JSONAction`.
- **Speed.** 135 ms per decision at the median over 1,335 AndroidWorld steps (181 ms at the 95th percentile, one
  shared 24 GB GPU, served over HTTP). The emulator step takes 1,624 ms, so the device dominates the loop.
- **A probability for each decision.** On AndroidControl-High the action probability ranks the model's own errors
  with AUROC 0.758. Acting only on the more confident half of the steps cuts the error rate from 38.3% to 19.0%.

> This is a research checkpoint from one training seed. All numbers below are measured by us from files in the
> project, except the rows marked *reported*, which are copied from the cited papers.

**Try it:** [DMA-2B demo Space](https://huggingface.co/spaces/tuanhqv123/DMA-2B-demo): step through three
AndroidWorld tasks and see the probability of every action, element, app and scroll direction, replayed from
our evaluation run.

## Results

### What the numbers mean

**AndroidControl** (Google, NeurIPS 2024) is a set of 1,543 recorded human episodes on real Android apps
(8,444 steps in the test split). At each step the model sees the screen and the history of what the human did, and
must predict the human's next action. It is scored step by step:

| Metric | Question it answers | Example of a miss |
|---|---|---|
| **Type** | Is the *kind* of action right (click, scroll, type, open app, back, wait, ...)? | predicts scroll when the human clicked |
| **Grounding** | Among steps with the right kind of action, is the click on the right element? | clicks "Settings" when the human clicked "Search" |
| **SR** (step success rate) | Is the whole step right: kind of action and all its arguments (element, text, direction, app)? | right element but scroll direction wrong |

- **AC-Low**: the model also gets the human's short instruction for the step ("click on the search bar"). This
  tests whether it can *carry out* an instruction.
- **AC-High**: the model gets only the overall goal ("find the hello note and share it"). This tests whether it can
  *decide* the next step, which is much harder.
- **Episodes all right**: episodes where every step is right. Each step still sees the human's correct history,
  so this is not the same as completing the task alone.

**AndroidWorld** (Google, ICLR 2025) runs the agent live on an Android emulator: 116 tasks in 20 apps. Nobody
helps it; after the agent stops, a checker reads the device state (a contact exists, wifi is on, the answer is
right). **TSR** (task success rate) is the share of tasks completed. This is the closest to real use, and the
hardest: a task fails if any step goes wrong and the agent cannot recover.

### Our checkpoints, same evaluation

All rows use the same 8,444 AndroidControl test steps and the InfiGUI-R1 scorer. **Bold** = best of our
checkpoints in the column.

**AndroidControl-Low** (step instruction given)

| Checkpoint | Training | Type | Grounding | SR |
|---|---|---|---|---|
| d1-2B (base, not fine-tuned) | none | 80.0 | 76.9 | 59.8 |
| v3 (2026-11-11) | 29,190 rows, element list capped at 40, no step instructions | – | – | – |
| **DMA-2B (v4, this repo)** | 39,713 rows, all elements, `answer` action, step instructions on half the rows | **96.6** | **90.1** | **89.3** |
| v6 (2026-11-13) | v4 + episode stop rows kept, 2,313 more steps | 96.4 | 89.4 | 88.8 |

**AndroidControl-High** (goal only)

| Checkpoint | Type | Grounding | SR | Episodes all right |
|---|---|---|---|---|
| d1-2B (base, not fine-tuned) | 63.6 | 63.1 | 40.2 | 145 / 1,543 |
| v3 (2026-11-11) | **81.0** | 77.4 | **67.1** | **374 / 1,543** |
| **DMA-2B (v4, this repo)** | 74.3 | **78.4** | 61.7 | 344 / 1,543 |
| v6 (2026-11-13) | 73.3 | **78.4** | 61.3 | 323 / 1,543 |

**AndroidWorld** (116 live tasks, seed 30, pass@1, our harness)

| Checkpoint | Harness A (first version) | Harness D (final version) |
|---|---|---|
| v3 (2026-11-11) | 15.0% (17 / 113 scored) | not run |
| **DMA-2B (v4, this repo)** | 13.5% (15.5 / 115 scored) | **21.6% (25 / 116)** |
| v6 (2026-11-13) | not run | not run |

The harness is our code around the model (how the screen is read, how actions are executed, how answers reach
the checker). Fixing it raised DMA-2B from 13.5% to 21.6% with the same weights (answers forwarded to the
checker, waiting until the element list matches the screenshot, a working SIM card, first-run screens cleared,
numeric answer format).

Reading the table:

- **Fine-tuning gains +29.5 SR on AC-Low and +21.5 on AC-High** over the base model.
- **v3 is better than DMA-2B on AC-High step accuracy (67.1 vs 61.7).** No AC-High test step is a correct
  `status` (task done), yet DMA-2B predicts it on 1,171 steps against v3's 539. At least 430 of the 827 steps that
  v3 gets right and DMA-2B gets wrong are such a `status`, 300 of them on scroll steps. On AndroidWorld, under the same
  harness, the two are tied (5 tasks won, 6 lost; sign test p = 1.0). We release DMA-2B because it adds the
  `answer` action (needed by 25 of the 116 AndroidWorld tasks), reads every element instead of the first 40, and is
  the checkpoint evaluated under the final harness.
- **v6 does not improve on DMA-2B** (AC-Low paired p = 0.007, worse; AC-High p = 0.19). It is not released.
- AC-Low is not reported for v3: it was not trained with step instructions, so the AC-Low input is new to it.

### Compared with published models on AndroidControl

Rows marked *reported* are copied from InfiGUI-R1 (Table 3), GUI-R1 (Tables 2–3) and MobileRL (Table 2); we did
not reproduce them. **Bold** = best in the column, *italic* = our model. The comparison is indicative only:
the other models read pixels only and generate the action as text, while DMA-2B also reads the accessibility element
list and chooses among its elements. Prompts, app lists and history formats also differ between papers.

| Model | Size | AC-Low Type | AC-Low Grounding | AC-Low SR | AC-High Type | AC-High Grounding | AC-High SR |
|---|---|---|---|---|---|---|---|
| GPT-4o *(reported, InfiGUI-R1)* | – | 74.3 | 0.0 | 19.4 | 66.3 | 0.0 | 20.8 |
| Qwen2.5-VL-3B, zero-shot *(reported, GUI-R1)* | 3B | 62.0 | 74.1 | 59.3 | 47.8 | 46.5 | 38.9 |
| GUI-R1-3B *(reported, GUI-R1)* | 3B | 83.7 | 81.6 | 64.4 | 58.0 | 56.2 | 46.6 |
| OS-Atlas-4B *(reported, InfiGUI-R1)* | 4B | 91.9 | 83.8 | 80.6 | 84.7 | 73.8 | 67.5 |
| UI-TARS-2B *(reported, InfiGUI-R1)* | 2B | 98.1 | 87.3 | 89.3 | 81.2 | 78.4 | 68.9 |
| InfiGUI-R1-3B *(reported, InfiGUI-R1)* | 3B | 96.0 | 93.2 | 92.1 | 82.7 | 74.4 | 71.1 |
| UI-TARS-7B *(reported, MobileRL)* | 7B | 98.0 | 89.3 | 90.8 | 83.7 | 80.5 | 72.5 |
| MobileRL-9B *(reported, MobileRL)* | 9B | 97.9 | 94.6 | 94.3 | **87.0** | 74.5 | 73.9 |
| UI-Genie-72B *(reported, MobileRL)* | 72B | **98.3** | **95.4** | **94.8** | 84.9 | **86.3** | **77.0** |
| d1-2B (base) | 2B | 80.0 | 76.9 | 59.8 | 63.6 | 63.1 | 40.2 |
| ***DMA-2B (ours)*** | 2B | *96.6* | *90.1* | *89.3* | *74.3* | *78.4* | *61.7* |

At 2–4B, DMA-2B matches UI-TARS-2B on AC-Low SR (89.3 vs 89.3) and its AC-High grounding (78.4 vs 78.4), and is
clearly above GUI-R1-3B and zero-shot Qwen2.5-VL-3B. It is behind UI-TARS-2B, OS-Atlas-4B and InfiGUI-R1-3B on
AC-High SR (61.7 vs 67.5–71.1), mainly on action type (74.3 vs 81–85).

### AndroidWorld

116 tasks, official task seeds (seed 30), pass@1, one run, our harness (accessibility element list + screenshot,
one decision per step, no planner, no external model). Score 25.0 of 116 = **21.6%**: 24 full successes and two
half-credit tasks. The other rows are reported by their authors with their own harnesses, for context only.

| System | Size | Model calls per step | Generates text | AndroidWorld TSR (%) |
|---|---|---|---|---|
| ***DMA-2B (ours)*** | 2B | 1 | no | *21.6* |
| M3A, GPT-4 Turbo *(reported)* | – | ≥2 | yes | 30.6 |
| UI-TARS *(reported)* | – | 1 | yes | 46.6 |
| MAI-UI-2B *(reported)* | 2B | 1 | yes | 49.1 |
| UI-Venus-1.5-2B *(reported)* | 2B | 1 | yes | 55.6 |
| GUI-Owl-7B *(reported)* | 7B | 1 | yes | 66.4 |
| MAI-UI-235B-A22B *(reported)* | 235B | 1 | yes | 76.7 |
| MobileRL-9B *(reported)* | 9B | 1 | yes | **80.2** |

DMA-2B is far from the AndroidWorld state of the art. In the run with 115 scored tasks (harness C, same score of
25), its 91 tasks without full success split into 51 where it declared the task done too early (49 scored 0, 2
scored half) and 40 that ran out of steps. The models above 45% train on
multi-step trajectories, and several train on AndroidWorld apps or on generated AndroidWorld task variants, often
with online RL. DMA-2B was trained only on AndroidControl human demonstrations plus 1,736 goal-to-app rows from
AndroidWorld's task generator (no AndroidWorld screen, trajectory or success signal).

### Confidence

AndroidControl-High, 8,444 steps, confidence c = p(chosen action). Risk at 50% coverage is the error rate on the
half of the steps with the highest confidence. **Bold** = better.

| | d1-2B (base) | DMA-2B |
|---|---|---|
| Step accuracy | 40.2% | **61.7%** |
| Mean confidence | 62.7% | 81.1% |
| ECE (15 bins) | 0.226 | **0.194** |
| AUROC (confidence vs. correct) | 0.721 | **0.758** |
| Risk at 50% coverage | 44.2% | **19.0%** |
| AURC | 0.420 | **0.196** |

Both models are over-confident, so treat the probability as a ranking signal (for an act / ask threshold chosen on
held-out data), not as a calibrated estimate. On AndroidWorld, p(done) separates correct from wrong terminations
with AUROC 0.746 (75 terminations).

## How to use

```bash
pip install "transformers>=5" torch pillow
git clone https://huggingface.co/tuanhqv123/DMA-2B && cd DMA-2B
python example.py --inputs-only   # builds the state and questions, no GPU
python example.py                 # one forward pass on a recorded AndroidWorld step
```

`example.py` rebuilds the inputs of a real step of the task "Turn wifi on." with `ac_common.py`, the same code used
for training, evaluation and serving, runs the model and decodes the answer into an AndroidWorld action. The
recorded decision was `click` (p = 0.988) on element [6] "Network & internet" (p = 0.776).

The core of it:

```python
import json, torch
from transformers import AutoModel
import ac_common as v
from example import load_image, decode

model = AutoModel.from_pretrained(".", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
model.engine.calibration = None          # raw readouts, as trained and evaluated

apps = [a["name"] for a in json.load(open("aw_apps.json"))]   # the 33 apps AndroidWorld can launch
# goal: str; history: list of previous-action lines; elements: AndroidWorld ui_elements as dicts
# (text, content_description, class, bbox_pixels, is_clickable, ..., package_name); W, H: screenshot size
state, kept, keep_idx = v.element_inputs_v3(goal, history, elements, W, H, "all_reading_noime")
questions = v.build_questions("VA", kept, contract="v4", apps=apps)
with torch.inference_mode():
    ans = model.system_one_batch([(state, questions, [load_image("screen.png")])])[0]["answers"]
action = decode(ans, history, kept, keep_idx, elements, apps)   # {"action", "probability", "json_action", ...}
```

After executing the action, append one history line built with `v.history_entry_with_label(...)` and
`v.effect_changed_v4(before, after)` (whether the screen changed), as in training.

### Inputs and outputs

- **Inputs, per step:** one screenshot resized to 720 px wide; a text state with the goal, progress, current app,
  the last 12 actions (each with the element acted on and whether the screen changed) and every on-screen app
  element in reading order (system UI, off-screen and keyboard elements removed), shown as
  `[id] label | type | state | position`; and the list of launchable apps.
- **Questions:** `action` (click, type, long press, scroll, wait, back, home, open app, done, answer), `target_el`
  (an element id), `scroll_dir`, `app_name` (from the app list), `type_text` (a span copied from the input) and
  `answer_value` (the chosen element's text, a number 0–20, or True/False).
- **Decoding:** a chosen element is tapped at the centre of its bounding box. On the first step (empty history) only
  `open_app` or `status` is allowed, since every AndroidWorld task starts on the home screen.

## Training

| | |
|---|---|
| Base | `LiquidAI/d1-2B`, revision `cfe5760` |
| Data | 37,977 decisions from the AndroidControl training split (rebuilt from the original release, test episodes excluded, task kinds that also occur in AndroidWorld) + 1,736 goal-to-app rows from AndroidWorld's task generator (97 of the 116 evaluated templates; goals only) |
| Added rows | actions with no effect followed by a different action; episodes that end with an answer; AndroidControl step instructions on a fixed half of the rows |
| Objective | cross-entropy on every question the gold action defines; unlikelihood on rows that mark an action to avoid |
| Optimisation | full fine-tune except the tied embedding / output head; 2,482 steps × 16 rows (one epoch); AdamW8bit (torchao), bf16 with stochastic rounding, lr 1e-5, 30 warm-up steps, cosine decay, weight decay 0, gradient clip 1.0; seed 0 |
| Hardware | one RTX 5090 |

## Limitations

- **Stops too early.** Most AndroidWorld failures are premature "done". The probability of `status` helps rank these
  (AUROC 0.746) but does not remove them.
- **Teacher-forced imitation only.** Trained on human demonstrations with the correct history; it never saw its own
  mistakes and was not trained on task success.
- **Needs the accessibility element list.** It cannot act from pixels only, or tap a target that has no element.
- **Types only text that appears in its input** (in practice, the goal).
- **One seed.** Seed-to-seed spread is not measured.
- **No safety training.** Do not let it control real phones, accounts, payments or messages without a human in the
  loop.

## License

A derivative of LiquidAI/d1-2B, distributed under the **LFM Open License v1.0** ([LICENSE](./LICENSE); the license
text governs, this is a summary): redistribution must include the license and keep the attribution notices; changed
files are listed in [NOTICE.md](./NOTICE.md); commercial use is not licensed for an entity with annual revenue of
USD 10 million or more; provided as is. AndroidControl and AndroidWorld keep their own terms and are not
redistributed here.

## Citation

```bibtex
@article{liquidAI2026opend1,
  author  = {Liquid AI},
  title   = {Open d1: Edge decision models for text, vision, and audio},
  journal = {Liquid AI Blog},
  year    = {2026},
  note    = {https://www.liquid.ai/blog/open-d1}
}
@inproceedings{li2024androidcontrol,
  title     = {On the Effects of Data Scale on {UI} Control Agents},
  author    = {Li, Wei and Bishop, William and Li, Alice and Rawles, Chris and Campbell-Ajala, Folawiyo and Tyamagundlu, Divya and Riva, Oriana},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2024}
}
@inproceedings{rawles2025androidworld,
  title     = {{AndroidWorld}: A Dynamic Benchmarking Environment for Autonomous Agents},
  author    = {Rawles, Christopher and Clinckemaillie, Sarah and Chang, Yifan and Waltz, Jonathan and Lau, Gabrielle and Fair, Marybeth and Li, Alice and Bishop, William and Li, Wei and Campbell-Ajala, Folawiyo and Toyama, Daniel and Berry, Robert and Tyamagundlu, Divya and Lillicrap, Timothy and Riva, Oriana},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```
