# Giving the decision model tools: a design for the "blocked by constraint" errors (2026-11-12)

## Idea

The model is already good at one thing: reading the goal, **pointing at** the right element, and **copying** the right span of text found in the input. What it lacks is arithmetic, memory across several screens, and information that is not in the element tree.

So we let a **deterministic tool** (ordinary code, not a model) do the calculation and the remembering, then **print the result as lines of text in the state**. The model only needs to point at or copy the right line. This is still one reading pass and generates no tokens. The constraint "no external model" still holds, because a tool is not a model. The only exception is OCR, see the table below.

| Blocked error | Tool (deterministic, runs in the harness) | Line added to the state | What the model must do | Which data teach it (taken from AndroidControl) | Tasks it could unblock |
|---|---|---|---|---|---|
| Date computation ("next Thursday", "tomorrow", "in 2 weeks") | Read the date and time on the device, precompute a date table | `Dates: today Sun Oct 15 2023 · tomorrow Mon Oct 16 · this Thu Oct 19 · next week Oct 22–28 · in two weeks Sun Oct 29` | Copy the right date with a `type_text` question, or choose the calendar cell whose label matches | Rewrite AndroidControl tasks that type a date in relative form, then add a Dates line containing the answer | 4–7 (Calendar add/delete events, date questions) |
| Time + duration ("+1 hour", "at 8:00 pm" → "20:00") | Convert every time in the goal into equivalent forms and added offsets | `Times: 8:00pm = 20:00 · 13:00 +30m = 13:30 · +1h = 14:00` | Copy | Same, rewritten from AndroidControl data | 1–3 |
| Summing, unit conversion, counting across several screens | **Memory:** a new action `note` stores the text of the chosen element. The tool sums and converts units | `Notes: 2:15:00 · 0:45:00 → total 3:00:00 = 180 min` | Press `note` on each item to record, then `answer` by copying the total line | Synthesised from AndroidControl tasks that have lists (record n items then answer the total) | 2–4 |
| Finding duplicates across several screens | (a) Collect the text of each row of the list and mark identical rows; (b) for the hard version: `note` memory stores the text of the detail screen, then the tool compares the notes | (a) `… · same as row 3` / `… · seen before`; (b) `Notes: 2 = 5 (identical)` | (a) Choose the marked row and delete it; (b) open each variant, record a note, delete the copy the tool flags as duplicate | Synthesised from AndroidControl lists | 1–2. The easy version +1 (ExpenseDeleteDuplicates). The hard version 0–1, because the same-name distractors differ only in a hidden field and need about 18 steps out of a budget of 24 (upstream code `recipe.py`) |
| Colour or icon carrying meaning (priority) | Take the dominant colour of a small element from the screenshot | Add `color: red` to the element label | Link "high priority" with red | There is no label in AndroidControl, so it must rely on d1's existing knowledge. **Uncertain** | 0–2 |
| Clipboard hint | Read the device clipboard (with AndroidWorld's own `get_clipboard` function) | `Clipboard: I think, therefore I am.` | Copy with `type_text` | Add a Clipboard line to the typing rows of AndroidControl when the goal mentions the clipboard (synthetic form) | 2 |
| Text that exists only in the image (receipts, expenses) | **OCR** | `Text in image: 2023-03-23, Monitor Stand, $33.22` | Copy | **OCR is a model.** Using it violates the constraint "no external model"; **the operator must decide** | 4 if allowed |

**Total:** optimistically about 10–15 tasks. Realistically about 5–8 tasks without OCR.

## Conditions to do it correctly

- **Within the protocol.** Any agent may use this information:
  - the date and time are available on the status bar;
  - clipboard: the agent can paste;
  - colour: it is pixels the agent can still see;
  - calculation: done on the agent side.

  The paper will state clearly that this is a "decision model with tools".
- **Train and serve must match.** The tool lines must be built by the same function on both sides (`ac_common`), like every other part of the state. Each tool needs a new contract (v7) and retraining.
- **Check early as usual.** A new line can cause unexpected harm, as the "state already correct" line did (wrong done declarations 108/690).

## Proposed order

1. Let the current final version finish and be measured. It is the baseline.
2. Meanwhile code the tool layer: harness, contract v7, synthetic data, with tests. Order of work:
   1. Dates/Times and Clipboard: cheapest, unblock the most tasks;
   2. `note` memory with summing;
   3. duplicate marking;
   4. colour.
3. Train v7 once, continuing from the final checkpoint, then pair-compare with final.

## Cross-check with TypeSafe's own guidance (Jev), read 2026-11-12

Source: the community list awesome-jev-use-cases, which quotes TypeSafe pages. The saved copy is at `references/raw/awesome-jev-use-cases-README-2026-11-12.md`; ledger `typesafe-jev-limits-2026`.

**Limits TypeSafe itself records for Jev 1.13:**
- "It is not a calculator. Keep math in code, which includes counting and date comparison, and use Jev to extract the parts." This matches the design above: code calculates, the model only points or copies.
- Two cookbooks follow exactly this pattern:
  - **Date extraction:** the model extracts the parts of the date, code resolves the real date and checks it.
  - **Pre-parsed value extraction:** a regex finds the candidates beforehand (email, number, money), the model chooses, code normalises.
- "Accuracy falls when the state carries content unrelated to the question." So a tool line may be added only when the goal needs it (for example the goal has words that denote a relative date, or mentions the clipboard). Do not add it at every step.

**Consequence for the design: a variant that could be more compact for dates and times.**
- Instead of printing the whole date table, ask the model one more question of type *choice* in the same reading pass (the "speculative fan-out" style). For example "target date relative to today": {today, tomorrow, Monday … Sunday this week / next week, after N days/weeks}.
- Code computes the real date then types it into the field (function-calling style: the model chooses typed parameters, code executes).
- This approach needs one more question in the contract and synthetic data to teach it.
- We will choose between "print the table" and "choice question" when v7 starts.

**On OCR:** `awlevin/typesafe-computer-use` uses OCR to read the screen for Jev. This shows that systems using a decision model often attach OCR as a tool, but it does not change our constraint "no external model". The decision still belongs to the operator.

## What AndroidWorld says about tools, and how others do it (original sources read 2026-11-12)

Sources: upstream code `android_world@55ddaf5`, paper v5, the leaderboard (CSV copy saved 2026-11-12 at `references/raw/androidworld-leaderboard-2026-11-12.csv`) and each agent's paper.

**AndroidWorld's rules**
- **No rule forbids tools.** The README only describes an agent that reads screenshots and UI elements through AndroidEnv, chooses an action and executes it. No sentence forbids it.
- **The reference agent M3A also uses something beyond the model:** at every step it writes an LLM summary as memory ("can be used as memory … shared between different apps"), and the prompt has hints written by hand.
- **The paper itself names the weaknesses:** agents fail at tasks that need memory, copying content between apps, and multiplying numbers. Information-retrieval tasks are scored with count / sum / identity operations.
- **The device clock is fixed at 15/10/2023, 15:34 UTC.** So a date table computed from the device clock is deterministic.

**Action set (`json_action.py`)**
- Contains: click, double_tap, scroll, swipe, input_text, navigate_home, navigate_back, keyboard_enter, open_app, status, wait, long_press, answer.
- There is no clipboard or date action.
- The clipboard can only be read through the helper function `adb_utils.get_clipboard_contents`, and this function brings the Clipper app onto the screen before reading.

**Leaderboard: self-reported, not verified**
- **Model Type:**
  - **Model** = "one LLM/VLM call";
  - **AI agent** = "several LLM calls … or an LLM wrapped into an advanced agent with memory, subgoal planning, etc.".
- **Screen Representation:**
  - Screenshot: 32;
  - Screenshot + A11y tree: 12;
  - A11y tree: 4;
  - blank: 1.
- There is no separate column for tools; tools are declared in the Note column. 44/49 rows have Number of trials = 1.

**Other agents**
- **Memory / notes are very common:** M3A, Mobile-Agent-v3 (Notetaker), Agent S2 (`save_to_knowledge`), V-Droid (step summaries).
- **OCR is rare.** Only Agent S2 describes using Tesseract, and it is unclear whether it is switched on when running AndroidWorld. Mobile-Agent-v2/E use OCR but have no AndroidWorld numbers.
- **No source mentions a calculator tool or a date-time tool.**
- **MAI-UI reports numbers "without … external tools".** V-Droid and MobileRL train on AndroidWorld's own tasks.

**Conclusion for us**

1. The deterministic tool layer (date table, notes with summing, duplicate-row marking, pixel colour) **fits the norms** of AndroidWorld. When publishing, each tool must be declared. Classing it as **"AI agent"** is the most honest (because it has memory). If classed as "Model", the Note column must list all the tools.
2. **Drop the clipboard-reading tool.** Reasons:
   - it is a privileged adb channel, not information shown on the screen;
   - the reading function changes the screen (brings Clipper up);
   - it can turn a "paste" task into a "type" task.

   The clipboard tasks should be solved with on-screen actions (long-press the field → Paste, or the suggestion chip on the keyboard). If it is still used, the result with and without the tool must be reported on the 3 clipboard tasks.
3. **OCR:** rare on the leaderboard and against the constraint "no external model". Keep it unused, unless the operator changes the constraint.
4. **When reporting our AndroidWorld numbers**, declare in full:
   - the number of runs (1);
   - the version of the step budget;
   - no AndroidWorld data used in training;
   - the harness and the tools that come with it.
