# LLM + decision model harness: feasibility, effectiveness, and numbers from other studies (2026-11-13)

Question: if we pair an LLM (planning) with our own decision model (execution), is it doable, is it
worthwhile, and what have other groups measured. Every number below comes from the original paper (links at the end) or from our own result files;
where something is our own inference, it is marked clearly.

## 1. Numbers from the systems that did it

| System | Architecture | Result | Time / cost | Authors' assessment |
|---|---|---|---|---|
| **Jev-Mobile** (arXiv 2609.30186) | A VLM (Qwen3.8-Max) produces a **local goal**, decides done / blocked, and supplies the text to type. Jev (decision model) reads the accessibility tree as text, chooses the ID of a program-generated candidate, and returns DONE or BLOCKED for the local goal. No images used. | AndroidWorld: **79%**, versus 84% when the VLM decides every step (−5 points) and 78% for SeeAct-V | On successful runs: time **−32.7%** (197 s → 133 s), API cost **−73.4%** ($0.274 → $0.073), executor time −94.5% | Separating "sparse planning" from "dense execution" reduces the number of VLM calls while staying adaptive. Limits: a target not in the tree cannot be clicked; only ASCII can be typed; long history needs compression. The authors claim this only for the design of this executor. |
| **"When Faster Decisions Do Not Make Better Agents"** (arXiv 2610.13092), 4,500 runs | Put Jev **in the place of the action-selection step** of an existing agent; input is text only | AndroidWorld: T3A **57.5% → 44.3%** (−13.2 points; 9 wins, 55 losses); FastAgent **50.0% → 31.9%** (−18.1); AutoDroid 18.7% → 16.4% (not statistically significant) | Decision time falls 74–82% (T3A 7.24 s → 1.34 s), but **whole-task time is only −10.6% (T3A) and +0.7% (FastAgent)**. T3A's cost per success **rises 7.7%**. FastAgent makes 44.1% more decisions. | The saving at the decision step is offset by extra reasoning and more steps. **Failure recovery and early done declarations** break the task (for example the Wi‑Fi task: declared done before it was reached). Evaluate by "successful tasks", recovery ability, and resources per successful task. |
| **V-Droid** (arXiv 2503.15937) | A program generates candidates from the accessibility tree; **a fine-tuned Llama-3.1-8B acts as the scorer** (one Yes/No token per candidate). An LLM fills the app, text, answer (12.4% of actions). GPT-4 keeps the working memory. | AndroidWorld **59.5%**, AndroidLab 38.3% | **0.7 s per decision**, 4.3 s per step. The LLM-held memory takes 3.0 s. | "Verifying is easier than generating". Choosing within an enumerable action set avoids format errors. Without the LLM memory SR falls to 46.1% (rule-based memory) or 40.0% (history only). An unfine-tuned verifier gets 0%. |
| **UGround / SeeAct-V** (arXiv 2410.05243) | GPT-4o plans, UGround (a grounding model) locates on the image | AndroidWorld: GPT-4o + UGround-V1-7B **44.0%**, versus M3A (GPT-4 choosing on the tree) 30.6%. AndroidControl 500 steps: High 48–50, Low 62–66 | – | **"Planning errors are the main cause of failure on all benchmarks."** The grounding module depends on an external planner. |
| **AndroidControl** (arXiv 2406.03679) | Fine-tuned PaLM-2S, reads only the accessibility tree | With step-by-step instructions (LL): 80.7; goal only (HL): 65.5 (in-domain). Out of domain: about 78 and 58 | – | The LL–HL gap is the "planning" part. Out of domain HL falls much more than LL. |
| **DMA-2B (ours)** | 2B decision model, reads image + element list | AC-Low 89.3; AC-High 61.7. AndroidWorld 21.6% running alone (91 unsuccessful tasks: 51 declared done early, 40 ran out of steps) | 135 ms per decision; one emulator step takes 1,624 ms. UI-TARS-2B measured on an RTX 5090: 1.7 s per step, about 90 tokens. | – |

## 2. What this evidence says

1. **Splitting the roles correctly preserves success.** Jev-Mobile loses only 5 points against a VLM deciding every step, but cuts
   1/3 of the time and 3/4 of the cost. The condition for doing this: the VLM keeps every decision that needs thought (splitting the goal, when to be
   done, the text to type, handling being blocked). The decision model only handles the local goal.
2. **Directly replacing the LLM with a decision model in the same place loses.** It loses 13–18 points. The errors are exactly early done declarations and poor
   recovery. These are also the two main errors of DMA-2B on AndroidWorld (51 of 91 tasks).
3. **A saving at the decision step does not automatically become a saving for the whole task.** Whole-task time is determined by the emulator and the remaining
   LLM calls (T3A: −10.6%; ours: model 0.135 s versus emulator 1.6 s; V-Droid: LLM memory takes
   3.0 of 4.3 s). The real benefit comes from **calling the LLM fewer times**, not from the model running faster.
4. **The planner decides the success ceiling** (UGround: planning errors dominate). The quality of the LLM matters more than the quality
   of the executor, once the executor is good enough.
5. **Choosing from a list is more stable than generating text** (V-Droid, Jev-Mobile, and our AC-Low 89.3). The executor
   must be trained for exactly that job. An untrained verifier gets 0% (V-Droid).

## 3. Feasibility with our system

| Component | Already have | Still missing |
|---|---|---|
| Executor that chooses an action + element from a list, with a probability | DMA-2B: AC-Low 89.3; AUROC (action probability × element) 0.812 on AC-Low | The threshold must be chosen on a validation set (the figure "80% of wrong steps ≤ 5%" is currently chosen directly on test, so it is optimistic) |
| AndroidWorld harness (reads the screen, executes, scores by the official protocol) | Yes (version D) | The place to attach the LLM: call the LLM, pass the local goal, receive DONE / BLOCKED |
| Multi-step local goal, and the DONE / BLOCKED signal from the executor | **Not yet.** AC-Low data only have instructions for single steps | Data must be created: merge consecutive instructions in an AC episode into one local goal, label "goal done" at the last step. This is augmentation from AndroidControl, no emulator used. |
| Text to type, answers, when to be done | – | Give to the LLM, as Jev-Mobile and V-Droid do |

**Key point:** if the LLM must write an instruction for **every step** (exactly the AC-Low style) then the LLM is called at every step, and
the benefit remains only that the LLM does not need to see the image or return coordinates. To get the benefit of Jev-Mobile (−32.7% time, −73.4%
cost) the executor must handle **multi-step goals** and report DONE / BLOCKED itself.

## 4. Expected effectiveness (our inference, not measured)

- **Success rate:** limited by the LLM planner. By Jev-Mobile, the target should be "not more than 5 points lower" than
  the same LLM deciding every step. We should not expect to exceed the LLM running alone.
- **Time and cost:** fall roughly in proportion to the number of LLM calls removed. The model's speed (135 ms) is not the deciding
  factor, because the emulator takes about 1.6 s per step.
- **Risks:** (1) an element is not in the accessibility tree; (2) the executor reports DONE wrongly for a local goal (the same
  kind of error as early done declarations); (3) trading many steps for few LLM calls (FastAgent: 44% more decisions); (4) sending screen content
  to an external LLM is a privacy issue.

## 5. Proposed measurement for fairness (pre-register before running)

Compare on the 116 AndroidWorld tasks, seed 30, **the same LLM** in three configurations:
(a) the LLM decides every step, reading the element list;
(b) the LLM produces a local goal, DMA-2B executes;
(c) DMA-2B running alone (21.6%).

Report: success rate, LLM calls per task, time per task, cost per successful task, and the number of tasks declared
done early. "Pass" condition: (b) is not more than 5 points below (a), and time per task falls by at least 25%.

## Sources

- Jev-Mobile: https://arxiv.org/abs/2609.30186 (HTML: architecture, time/cost table, limits section)
- When Faster Decisions Do Not Make Better Agents: https://arxiv.org/abs/2610.13092 (Tables 2, 5, 6, 8, 10)
- V-Droid: https://arxiv.org/abs/2503.15937
- UGround: https://arxiv.org/abs/2410.05243 (AndroidWorld, AndroidControl tables; conclusion section)
- AndroidControl: https://arxiv.org/abs/2406.03679 (Tables 4, 5, 7, 8; appendix D.1, D.3)
- Our numbers: `benchmarks/results/2026-11-12-v4/actest/` (AC-High/Low), `benchmarks/results/2026-11-12-v14targeted/`
  (AndroidWorld), `benchmarks/results/2026-11-12-calibration/`; UI-TARS-2B time measured on GPU_HOST_C, 40 AC-High steps, batch 1, bf16
