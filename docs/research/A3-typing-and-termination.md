# A3 — Why typing fails, and why the readout is not the problem

*Written 2026-11-09. The mechanism section is measured from our own 128 episode files on disk; the literature
section is cited. Both are labelled. This is the issue-I4 answer.*

## VERDICT

**The failure is mechanism M3 — no state/progress model — and M1 (supervision imbalance) is the cause, not M2.**

The decisive measurement: **the action slot works.** All 19 typing episodes emit a `type_text` with a
**non-empty string and a valid point, `action_ok=True`, `invalid_action=False` — 19/19.** The field is found, the
string is right, the harness accepts it. What fails is everything after: **15 to 22 further clicks**, on tasks
that needed 2 to 5 more actions.

Company the model already had the answer and could not remember giving it.

This **refutes mechanism M2 (readout-slot collapse)**, which was the most attractive hypothesis because the
`app_name` slot did collapse before. It does not collapse here. That leaves M1 and M3, and they are the same
defect seen from two sides: the model never learned the *shape* of a multi-step typing task, because 3.0% of its
labels are `type_text` and 17.6% of its rows have no interaction history at all.

---

## 1. The measured trace (our data, not inference)

Source: `benchmarks/results/2026-11-08-aw116/episodes/episodes/*.jsonl`, 125 episodes with a result row.
`action` = the model's raw emission (`action_raw.action`), matching `evidence-82tasks.txt`.

| | n | successes | median steps |
|---|---|---|---|
| emitted `type_text` ≥1× | **19** | **1** | **25** (budget) |
| never emitted it | 106 | 15 | 7 |

That reproduces the recorded 19/1 and 85/14 split (the 85/14 in the evidence counts the 104 finished; the 106
here includes two more with a result row).

### The action slot is intact on every typing row

```
MarkorChangeNoteContent    i= 3  text='3StvyHSIadeFKoh6uCcC'    point=[543.8, 907.2]  ok=True  inv=False
SimpleSmsReply             i= 3  text='To be or not to be.'     point=[555.7, 354.0]  ok=True  inv=False
ContactsAddContact         i= 4  text='Emily Mohammed'         point=[535.1, 933.6]  ok=True  inv=False
ContactsNewContactDraft    i= 9  text='965-688-5552'            point=[534.1, 387.6]  ok=True  inv=False
SimpleCalendarAddOneEvent  i= 4  text='Review session for ...'  point=[540.0, 387.6]  ok=True  inv=False
```

**19/19 non-empty text. 19/19 valid point. 19/19 executed without rejection.** There is nothing to fix in the
string slot or the coordinate slot. (This is in sharp contrast to `open_app`, where `app_name` was empty on
24/24 before the `app_intent` readout existed — a genuine slot defect, since repaired.)

### What happens after the typing — the actual failure

```
MarkorChangeNoteContent       n=25  type_text@[3]   then 21 clicks
ExpenseAddSingle              n=25  type_text@[3,7] then 17 clicks
ContactsNewContactDraft       n=25  type_text@[9]   then 15 clicks
RecipeAddSingleRecipe         n=25  type_text@[3,4] then 20 clicks
MarkorCreateNoteFromClipboard n=25  type_text@[2]   then 22 clicks (incl. 1 scroll)
ContactsAddContact            n=25  type_text@[4]   then 19 clicks + 1 scroll
```

Every one of these burns the full 25-step budget. The task was not "find the field"; it was "find the field,
type, then press save/OK/send".

### Two post-typing behaviour patterns, both wrong

Classifying each typing episode by whether the post-typing clicks repeat a point (within 40 px) or move around:

| pattern | n | meaning |
|---|---|---|
| **LOOP** — repeats the same point | **7** | stuck re-tapping one element (e.g. `ContactsNewContactDraft`: 15 clicks inside x 534–548, y 664–881 — a 14×217 px strip) |
| **SEARCH** — moves widely | **5** | hunting the screen (e.g. `MarkorChangeNoteContent`: 21 clicks spanning x 255–909, y 208–1170 — the whole screen) |

**Neither pattern is "aiming at the wrong button". Both are "I do not know what I already did or what is
next".** A grounding fix cannot address this; a loss-function fix cannot address this. Only state can.

---

## 2. Mechanism by mechanism

### M1 — Supervision imbalance: **CONFIRMED as the cause**

| evidence | value |
|---|---|
| `type_text` share of training labels | **450 / 15,000 = 3.0%** |
| `type_text` share of emitted actions in AndroidWorld | 24 / 1,393 = 1.7% |
| share of episodes that *need* typing | **19 / 125 = 15.2%**, and of those, **18/19 fail** |
| `status` share of labels | 3,300 / 15,000 = **22.0%** |
| rows with **empty history** | **2,635 / 15,000 = 17.6%** |

The corpus teaches "declare done" 7× more often than it teaches "type". It has seen `status` 3,300 times and
`type_text` 450 times, and the model's behaviour mirrors that prior: it declares done too early (31 premature
stops) and types too rarely.

**GUI-Odyssey corrects exactly this**, which is the strongest argument for the data:

| | our cur15k | GUI-Odyssey |
|---|---|---|
| episodes containing text entry | 15.2% (AndroidWorld need) | **86.7%** (6,707 / 7,735) |
| `TEXT` steps | 3.0% of labels | **10.44%** of steps (12,405) |
| field localised | yes (`target` bbox on 450/450) | yes, **via the preceding `CLICK` on 99.27%** |

Mixing at a ratio that brings `type_text` from 3% toward ~10% is the intervention with the clearest measured
motivation. **Do not re-weight by loss** — that is already refuted (see the negatives below).

### M2 — Readout-slot collapse: **REFUTED for `type_text`**

19/19 rows carry a non-empty string and a valid point. The pre-fix `app_name` collapse (24/24 empty) was real
but is a different slot, and it is already repaired (1/8 → 8/8 in the served path). **Drop this mechanism.**

What survives from M2 is a narrower, unverified concern: the model types into the field it *clicked*, and the
harness then keeps focus. If a task needs a **second** field, focus has moved and the model has no readout for
"which field is focused". `ExpenseAddSingle` types at steps 3 and 7 — two fields — and still fails at 25 steps.
That is an untested sub-case, not the main failure.

### M3 — No progress/state model: **CONFIRMED, and it is the same defect as M1**

Direct evidence:

1. **21 clicks after a successful type** is not an aiming error. It is the absence of any representation of
   "the form is filled; press save".
2. **LOOP vs SEARCH splits 7/5** — one model, two opposite failure modes, both consistent with "no memory".
3. **17.6% of training rows have empty history**, and only 1,367 of 9,005 episodes have contiguous step indices,
   so the model has rarely seen a *sequence* it can condition on.
4. **A threshold on `p(status)` cannot fix it** — measured monotone-harmful at every τ (2026-11-07). No scalar
   confidence can manufacture a progress signal.
5. **Down-weighting `status` 22% → 5.3% traded −27 premature terminations for −26 correct ones**
   (val +1.09, McNemar p=0.44; test −3.27, p=0.15). A **perfect** progress oracle was worth only **+1.26
   points** on that split. So the signal is not recoverable from class balance; it needs to be *supervised*.

**M1 and M3 are one mechanism.** Rare typing + no history ⇒ the model cannot learn task shape ⇒ it cannot know
when it is mid-task. Fixing the data mix (M1) is the cheapest way to attack M3; adding an explicit completion
readout is the alternative, and it has already been shown to be worth at most +1.26 points by itself.

---

## 3. Negatives that must not be re-proposed

| proposed fix | measured outcome | verdict |
|---|---|---|
| Loss-reweight `status` down | −27 premature, −26 correct; val +1.09 p=0.44, test −3.27 p=0.15 | **fails both clauses** |
| Threshold on `p(status)` | monotone-harmful at every τ | **refuted** |
| Switch premature `status` to best expressible action | best rule = never switch (val 62.76, test 63.82) | **refuted** |
| Ensemble / routing two checkpoints | +1.01 pts, McNemar p=0.56, and the router's *type* decision is the bottleneck | **not reachable** |
| More data at the same recipe (24k), continued training (+8k) | no gain | **refuted** |
| Candidate-choice interface (VA track) | val −3.75 p=0.019, test −7.29; the box-only path also dropped | **refuted** |

The pattern across all six: **post-processing and re-weighting a 64%-per-step model does not work.** The
defects are in what the model was taught, not in how its output is thresholded. Any new proposal must be a
*supervision* change.

---

## 4. Remedies ranked by expected end-to-end delta per GPU-hour

Ranking is by the strength of the *measured* supporting evidence, not by enthusiasm. All are SFT-only unless
marked.

| # | Remedy | Evidence | Est. cost | Closable on one 4090 |
|---|---|---|---|---|
| **1** | **Mix GUI-Odyssey trajectories so `type_text` goes 3% → ~10%**, keeping cur15k so nothing is forgotten | 86.7% of GUI-Odyssey episodes contain text entry; A2's Ferret-UI-Lite ladder says the *first* 5K trajectories are worth only +0.7 but 0→17K is +5.6 e2e | 1 run, 938 steps, ~1.6 h | **yes** |
| **2** | **Render a real progress/state line** from the trajectory (GUI-Odyssey's new release ships `low_level_instruction` + `context` per step) | our 17.6% empty-history rows; +1.26 oracle bound | 1 run | **yes** |
| **3** | **Longer CoT / low-level instruction supervision** | Ferret-UI Lite: **+3.8 e2e** for long CoT — the largest single jump in that ladder | 1 run + data prep | **yes** |
| 4 | Explicit `task_complete` readout, trained (not thresholded) | oracle bound +1.26; needs its own labels | 1 run | yes |
| 5 | Rejection sampling with an MLLM judge | MAI-UI reports it as core; no independent e2e ablation found | needs judge + many rollouts | **partly** |
| 6 | Online RL (GRPO) | MAI-UI Table 9: +4.0 e2e at 2B **on top of** a 45.1% SFT base | 512 AVDs | **no** |

**Order matters.** Remedy 1 addresses the measured cause and is the only one whose ingredient count we have
verified end-to-end. Remedy 3 is the single largest measured per-ingredient effect in the ≤4B literature and
should be tried in the same run if the budget allows. Remedy 6 is not available and quoting its effect as if it
were would repeat mistake M2.

**Important bound from A2 + `scripts/power_aw116.py`:** with 116 AndroidWorld tasks the minimum detectable
effect at 80% power is **~10–13 points** (paired, ψ=0.15–0.25); a +4-point claim needs **~1,224** paired tasks.
Remedies 1–3 are each plausibly worth +2 to +6 points, which is **below** what this benchmark can resolve. That
is not a reason not to run them; it is a reason to **pre-register the primary metric as a lower-variance one**
(per-step accuracy, or the coverage/risk curve) and treat task success as a secondary, descriptive number.

---

## 5. Android text-entry mechanics (what the harness must guarantee)

Grounded in how `android_env` / AndroidWorld execute actions, and consistent with what our traces show:

1. **Click-then-type is the correct primitive.** Our model emits `type_text` with a point; the adapter clicks
   the field then sets the text. All 19 typing rows executed with `action_ok=True`, so this path demonstrably
   works in our harness. Any change here would break a working component.
2. **The a11y `editable` flag is the field oracle.** Our audit joined `type_text` targets to the raw a11y nodes:
   **435/450 (96.7%) of `type_text` targets match a node that is editable**; the 15 that do not are a known,
   enumerated flag (`type_text_target_not_editable`). This is a *label* issue on 3.3% of typing rows, not a
   harness issue.
3. **Focus does not survive a field change.** After a second click, text lands in the new field. Two-field tasks
   (`ExpenseAddSingle` types at steps 3 and 7) need the model to re-emit the point each time — it does — but
   still fails, so focus is not the binding issue.
4. **The IME / `ADBKeyboard` path is only needed if the soft keyboard must be bypassed.** Setting text directly
   through `android_env` avoids IME state entirely; our `action_ok=True` rate indicates this is already what
   happens. Do not add IME switching without evidence that it is needed.
5. **What to check before the next run (not run here):** that `navigate_back` dismissal of the keyboard does not
   itself consume a step, and that the step budget counts an executed `type_text` as one step (it appears to).

---

## 6. What I could not verify

- The **second-field / focus** hypothesis is untested: I did not instrument the a11y focused-node state per step.
  `ExpenseAddSingle` is the one trace that suggests it, and n=1.
- **No published ablation** was found that isolates "trajectory data" from "more data" for text entry
  specifically. The mixing ratio in remedy 1 is therefore chosen to move the *label share* to a defensible
  target (GUI-Odyssey's own 10.44%), not from a measured optimum.
- The literature search for this note was cut short twice by subagent timeouts; the cited numbers are from
  A2's completed search (10 papers, arXiv HTML/PDF) plus our own artifacts. **No claim in §1 or §4 depends on a
  source I did not read**; §2's mechanism ranking is our data.
- Whether `type_text` succeeds when the model is *forced* to type and then given a save hint — that is the
  cheapest decisive experiment for the M3 diagnosis and it has **not** been run. It is one run, and it is the
  one I would do next.
