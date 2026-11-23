# LLM planner + exec2 executor: trial on 20 AndroidWorld tasks (16/11)

**Setup:**
- LLM: Qwen3.5-27B through an OpenAI-compatible endpoint provided by the operator, thinking not enabled. The LLM writes the sub-goal, answers questions and decides when the whole task is done.
- Executor model: exec2 `ckpt_7000`, which takes the sub-goal as the per-step instruction.
- Harness V12 (same as `aw116_v4g`), plus `awenv/planner.py` version `c8eaaf66` plus the fix for the "reports done without having done anything" loop.
- Tasks: the `aw20_q10` list, seed 30, official protocol.

**Reference:** DMA-2B run alone (`aw116_v4g`) on exactly these 20 tasks, same seed and same harness.

| | LLM + exec2 | DMA-2B alone |
|---|---|---|
| Success under the official rule (device correct **and** agent reports done) | **7/20** | 7/20 |
| Device correct but not reported done (steps ran out) | 5/20 | 0/20 |
| LLM calls / steps | 349 / 352 | — |

Task by task: LLM + exec2 wins 2 tasks (MarkorMoveNote, SaveCopyOfReceiptTaskEval) and loses 2 tasks (SystemBrightnessMaxVerify, RecipeDeleteDuplicateRecipes). With n = 20 the two configurations cannot be told apart.

**Failures read from the traces:**
1. **The LLM gives commands that exec2 cannot perform.** Examples: dragging the brightness slider, swiping open the notification shade. exec2 has no drag action, so it keeps scrolling.
2. **Stuck detection is slow.** The `ui_changed` flag, based on element structure, still reports a change, while the history line the model sees says "no change". So being stuck is detected only after hitting the 6-action limit.
3. **The LLM does not report done in time.** In 5 tasks the device was already correct, but the LLM kept checking until the steps ran out.
4. **Almost 1 LLM call per step.** The sub-goal is almost always a single step, so there is no "fewer LLM calls" benefit as in Jev-Mobile.

**Fixed in code, not yet measured:**
- the prompt states clearly what exec2 can and cannot do;
- merge 1–3 steps joined by ", then";
- report "finish" as soon as the device is correct;
- treat the history line "→ no change" as an unchanged screen.

**Note:** these fixes were derived from these same 20 tasks. The number for the fixed version must be measured on the 116-task set, and must not be reported on these 20 tasks as an independent result.
