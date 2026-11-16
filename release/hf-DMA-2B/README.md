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
  - androidcontrol
  - custom_code
model-index:
  - name: DMA-2B
    results:
      - task:
          type: gui-action-selection
          name: AndroidControl-Low (step success rate, InfiGUI-R1 scorer)
        dataset:
          type: androidcontrol
          name: AndroidControl test, low-level instructions (8,444 steps)
        metrics:
          - type: step_success_rate
            name: SR
            value: 90.5
      - task:
          type: gui-action-selection
          name: AndroidControl-High (step success rate, InfiGUI-R1 scorer)
        dataset:
          type: androidcontrol
          name: AndroidControl test, high-level goals (8,444 steps)
        metrics:
          - type: step_success_rate
            name: SR
            value: 74.2
---

# DMA-2B: a single-pass decision model for Android GUI control

**DMA-2B** (Decision Model for Android, 2B) picks the next action on an Android screen in **one forward pass, with
zero generated tokens**. It is a full fine-tune of [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B), a 1.92B
vision-language *decision model*: instead of writing an action as text, it answers typed questions (which action,
which on-screen element, which app, which scroll direction, which span to type) and returns a probability for every
option.

- **No parsing.** Every answer is one of the listed options, so every decoded action is valid.
- **One forward pass per decision.** 123 ms at the median (144 ms at the 95th percentile) on one otherwise idle
  RTX 4090, served over HTTP (500 AndroidControl-High steps, batch 1); 255 ms when another job keeps the GPU busy.
  No tokens are generated.
- **A probability for each decision.** On AndroidControl-High, acting only when the action probability is above a
  threshold chosen on held-out episodes keeps the error rate at 19% while acting on 80% of the steps.

> Research model, one training seed. Rows marked *reported* are copied from other papers; everything else is
> measured by us.

## Results on AndroidControl

[AndroidControl](https://arxiv.org/abs/2406.03679) is a set of recorded human episodes on real Android apps; the test
split has 1,543 episodes and 8,444 steps. At each step the model sees the screen and the human's history and must
predict the human's next action. **AC-Low** also gives the step instruction ("click on the search bar"); **AC-High**
gives only the goal ("find the hello note and share it"). **SR** (step success rate) counts a step as right only if
the kind of action and all its arguments are right.

Scored with the InfiGUI-R1 evaluation code. DMA-2B is scored in AndroidControl's own action space, which has no
"done" action; the base model as originally evaluated.

| Model | Size | Generates text | AC-Low SR | AC-High SR |
|---|---|---|---|---|
| Qwen2.5-VL-7B, zero-shot *(reported)* | 7B | yes | 62.5 | 47.1 |
| UI-TARS-2B *(reported)* | 2B | yes | 89.3 | 68.9 |
| MAI-UI-2B *(reported)* | 2B | yes | – | 67.3 |
| InfiGUI-R1-3B *(reported)* | 3B | yes | 92.1 | 71.1 |
| GUI-SAGE-3B *(reported)* | 3B | yes | 93.4 | 75.4 |
| AgentCPM-GUI-8B *(reported)* | 8B | yes | 90.2 | 69.2 |
| OS-Atlas-7B *(reported)* | 7B | yes | 85.2 | 71.2 |
| UI-TARS-7B *(reported)* | 7B | yes | 90.8 | 72.5 |
| GUI-Owl-7B *(reported)* | 7B | yes | – | 72.8 |
| MobileRL-9B *(reported)* | 9B | yes | 94.3 | 73.9 |
| UI-Genie-7B *(reported)* | 7B | yes | 94.3 | 74.5 |
| UI-Venus-Navi-7B *(reported)* | 7B | yes | 92.4 | 76.1 |
| GUI-SAGE-7B *(reported)* | 7B | yes | 93.7 | 76.8 |
| UI-Genie-72B *(reported)* | 72B | yes | 94.8 | 77.0 |
| d1-2B (base, not fine-tuned) | 2B | no | 59.8 | 40.2 |
| **DMA-2B** | **2B** | **no** | **90.5** | **74.2** |

Sources: InfiGUI-R1 (arXiv 2504.14239), GUI-R1 (2504.10458), MobileRL (2509.18119), MAI-UI (2512.22047),
GUI-SAGE (CVPR 2026), AgentCPM-GUI (2506.01391), OS-Atlas (2410.23218), GUI-Owl (2508.15144), UI-Venus (2508.10833).
Prompts, action spaces and scorers differ between papers, so the comparison is approximate.

On AC-High, DMA-2B is the best reported 2B model and is above several 7B–9B generative agents; four 3B–7B models
are higher. On AC-Low it is level with UI-TARS-7B.

| | Type | Grounding | SR | Episodes fully correct |
|---|---|---|---|---|
| AC-Low, base / DMA-2B | 80.0 / 96.8 | 76.9 / 91.2 | 59.8 / 90.5 | 18.8 / 63.8 |
| AC-High, base / DMA-2B | 63.6 / 87.0 | 63.1 / 80.4 | 40.2 / 74.2 | 9.4 / 30.6 |

### AndroidControl-Curated

[AndroidControl-Curated](https://arxiv.org/abs/2510.18488) re-scores the same test steps with box-based grounding and
corrects the high-level split (527 rewritten tasks, 573 steps with alternative valid actions, 86 fixed labels). Scored
with the authors' rules; DMA-2B was re-run only on the 527 rewritten tasks.

| Model | Easy SR | Hard (Box) SR | Hard SR |
|---|---|---|---|
| GUI-R1-7B *(reported)* | 72.7 | 53.2 | 57.5 |
| InfiGUI-R1-3B *(reported)* | 87.2 | 67.6 | 70.7 |
| Qwen3-VL-30B *(reported)* | 70.5 | – | 70.0 |
| Qwen3-VL-235B *(reported)* | 74.5 | 71.7 | 76.5 |
| Magma-R1 *(reported)* | 88.0 | 69.1 | 75.3 |
| **DMA-2B** | **89.7** | **73.6** | **77.0** |

The reported models read the screenshot and a written history of the previous step instructions; DMA-2B also reads the
accessibility element list and its own previous actions.

### Confidence (AC-High, c = p(chosen action))

| | d1-2B (base) | DMA-2B |
|---|---|---|
| Step accuracy | 40.2% | 74.2% |
| AUROC (confidence vs. correct) | 0.721 | 0.731 |
| Error rate on the 50% most confident steps | 44.2% | 13.6% |
| ECE (15 bins) | 0.226 | 0.145 |
| Gate for a 20% target risk: share of steps acted on / error rate | no stable threshold | 80% / 19.0% |

The model is over-confident (mean confidence 88.7% for 74.2% accuracy), so use the probability as a ranking signal
for an act / ask threshold chosen on held-out data, not as a calibrated estimate.

## How to use

```bash
pip install "transformers>=5" torch pillow
git clone https://huggingface.co/tuanhqv123/DMA-2B && cd DMA-2B
python example.py --inputs-only   # builds the state and questions, no GPU
python example.py                 # one forward pass on an AndroidControl test step
```

`example.py` rebuilds the inputs of AndroidControl test episode 27, step 1 (goal "Open the Cx file Explorer and
rename the Flowers folder to Flora.", instruction "Click on the square icon next to the Flowers folder") with
`ac_common.py`, the code used for training and evaluation, runs the model, and decodes the answer. Expected output:
`click` (p = 0.999) on element [13] "Toggle selection" (p = 0.987), the correct action.

The core of it:

```python
import torch
from transformers import AutoModel
import ac_common as v
from example import load_image, decode

model = AutoModel.from_pretrained(".", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
model.engine.calibration = None          # raw readouts, as trained and evaluated

# goal: str; instruction: str or None; history: previous-action lines; elements: accessibility elements as dicts
# (text, content_description, class_name, bbox_pixels, is_clickable, ..., package_name); W, H: screen size in pixels;
# apps: names of the installed apps
state, kept, keep_idx = v.element_inputs_v3(goal, history, elements, W, H, "all_reading_noime", instruction)
questions = v.build_questions("VA", kept, contract="v4", apps=apps)
with torch.inference_mode():
    ans = model.system_one_batch([(state, questions, [load_image("screen.jpg")])])[0]["answers"]
action = decode(ans, kept, keep_idx, elements, apps)   # {"action", "probability", "element", "json_action"}
```

After executing the action, append one history line built with `v.history_entry_with_label(...)` and
`v.effect_changed_v4(before, after)` (whether the screen changed), as in training.

### Inputs and outputs

- **Inputs, per step:** one screenshot resized to 720 px wide; a text state with the goal, the optional step
  instruction, progress, current app, the previous actions (each with the element acted on and whether the screen
  changed) and every on-screen app element in reading order, shown as `[id] label | type | state | position`; and the
  list of installed apps.
- **Questions:** `action` (click, type, long press, scroll, wait, back, home, open app, done, answer), `target_el`
  (an element id), `scroll_dir`, `app_name` (from the app list), `type_text` (a span copied from the input) and
  `answer_value`.
- **"Done":** the model learned to answer "done" when a step instruction has been carried out. AndroidControl has no
  such action, so `decode` skips it unless `allow_done=True`.

## Training

| | |
|---|---|
| Base | `LiquidAI/d1-2B`, revision `cfe5760` |
| Data | the full AndroidControl training split (13,594 episodes, 88,308 steps), rebuilt from the original release with every test episode excluded. 34% of the episodes are shown with the goal only. In the others, step instructions are grouped into local goals of 1–4 instructions, each step shows its local goal, and the screen after each local goal is added with the answer "done". Steps that continue an instruction from the previous step have three times the weight. 116,139 rows in total. |
| Objective | cross-entropy on every question the gold action defines |
| Optimisation | full fine-tune except the tied embedding / output head; one epoch (7,259 steps × 16 rows; the released weights are the last saved checkpoint, step 7,000, learning rate 3e-8); lr 1e-5, 30 warm-up steps, cosine decay; seed 0 |
| Hardware | one RTX 4090 |

## Limitations

- **Teacher-forced imitation only.** Trained on human demonstrations with the correct history; it never saw its own
  mistakes. AndroidControl scores single steps, so these numbers do not measure task completion: on AC-High it gets
  every step of an episode right in 30.6% of the episodes.
- **"Done" one step early.** With "done" allowed, it ends an instruction one step early on 449 of 8,444 AC-Low steps
  (AC-Low SR 86.8%), mostly when the screen is still loading after a click.
- **Needs the accessibility element list.** It cannot act from pixels only, or tap a target that has no element.
- **Types only text that appears in its input** (the goal or the instruction).
- **One seed.** Seed-to-seed spread is not measured.
