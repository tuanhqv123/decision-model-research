# First 116-task AndroidWorld run: the two failure modes, named

*2026-11-08, run `aw116_appint`, checkpoint `appint_full@938`, protocol `aw-main-v1`, step budget 25,
seed 30, no app-name fallback. Partial results — the suite is still running. Every number below comes
from `/tmp/awruns/aw116_appint/episodes/*.jsonl` on the emulator, not from an estimate.*

## What is measured so far

| task | success | timeout | declared done | steps | action mix |
|---|---|---|---|---|---|
| CameraTakePhoto | **True** | – | True | 3 | open_app 1, click 1, status 1 |
| AudioRecorderRecordAudio | False | **True** | False | 25 | open_app 1, click 24 |
| ClockStopWatchRunning | False | **True** | False | 25 | open_app 1, click 24 |
| ClockTimerEntry | False | **True** | False | 25 | open_app 1, click 24 |
| ContactsAddContact | False | **True** | False | 25 | open_app 1, click 22, type_text 1, scroll 1 |
| ContactsNewContactDraft | False | **True** | False | 25 | open_app 1, click 22, scroll 1, type_text 1 |
| ExpenseAddMultiple | False | **True** | False | 25 | open_app 1, click 24 |
| ExpenseAddMultipleFromMarkor | False | **True** | False | 25 | open_app 1, click 15 |
| BrowserDraw | False | – | True | 5 | open_app 2, click 2, status 1 |
| BrowserMaze | False | – | True | 4 | open_app 1, click 1, wait 1, status 1 |
| CameraTakeVideo | False | – | True | 3 | open_app 1, click 1, status 1 |
| ClockStopWatchPausedVerify | False | – | True | 4 | open_app 1, click 2, status 1 |
| BrowserMultiply | False | – | True | 18 | open_app 2, click 11, wait 4, status 1 |
| ExpenseAddMultipleFromGallery | False | – | True | 10 | open_app 1, click 7, wait 1, status 1 |

**SR so far: 1/14 finished episodes.** Wilson 95% ≈ 1.3%–35.1%. One success is `CameraTakePhoto`,
which needs a single `open_app` + `click` + `status` — three actions, no text entry, no navigation.

## The two failure modes are opposite, and both are the same defect

**Mode A — premature `status` (6 episodes).** The model declares the task done after **3–5 actions**
on tasks that need 8–15. `BrowserMaze` declares done after *two* actions. It is not confused about
the screen; it is confident the task is finished. This is exactly the bucket the 2026-11-07
down-weighting arm tried to fix from the loss side, and it failed there too (val +1.09, test −3.27):
**a single class weight cannot separate "not finished" from "finished"**.

**Mode B — never declares, clicks 24 times (8 episodes).** On clock, contacts and expense tasks the
model opens the right app (`open_app clock`, confidence 0.996 — the `app_intent` fix works) and then
emits an unbroken stream of `click` with high confidence (0.92–0.99) while `gate_score` decays to
**0.0**:

```
i=0 open_app clock   conf=0.996  gate=0.996
i=1 click (539,2200) conf=0.994  gate=0.619
i=2 click (268,1367) conf=0.976  gate=0.364
i=3 click (272,1368) conf=0.979  gate=0.140
i=5 click (546,1322) conf=0.924  gate=0.000
i=7 click (549,1256) conf=0.929  gate=0.000
```

**The gate already knows.** `gate_score` hits 0.0 by step 5 and stays there. The action head keeps
clicking at 0.92 confidence. This is the strongest single result in the run so far: **the confidence
signal separates "I am doing the right thing" from "I am clicking at random" even though the action
head does not act on it.**

## The action mix says the interface is under-used

Across 193 emitted actions: `click` 180, `open_app` 16, `status` 7, `wait` 6, **`type_text` 2,
`scroll` 2**. AndroidWorld needs `input_text` on most contacts/notes/expense tasks; the model emits it
twice in 193 actions. ContactsAddContact emitted `type_text` once and then never again — it had a text
field to fill and did not return to it.

So the dominant error is not grounding (positions look plausible) and not the read slot (app names
resolve). It is **the model has no model of task progress**: no signal that says "step 3 of 9", "I have
not typed the name yet", "I am finished". That is what `p^n` compounding looks like from the inside.

## What this means for the number

At 1/14, the Wilson upper bound is 35%. The suite is not finished, so **no headline SR exists yet.**
But the shape is already clear and it matches the arithmetic in
`notes/2026-11-08-compound-accuracy-reframe.md`: 64% per-step accuracy predicts a few percent per task,
and we are seeing ~7% with one trivial success.

**Do not report 1/14 as our AndroidWorld score.** Report it only with the completeness line, and only
after the suite finishes, with `wait`/`random` controls beside it.

## The concrete lever this run identifies

Not more data, not more seeds, not a better grounder. **A progress readout.** The model needs a
learned signal for "how far through the task am I" that both (a) suppresses `status` when the task is
not done and (b) re-surfaces `type_text` when a field is still empty. `gate_score` is already partway
there and is being ignored by the policy — that is a training target, not a new architecture.

## `gate_score` decays on failures and does not decay on successes

Measured over the first 42 finished episodes (732 emitted actions). For each episode, take the mean
`gate_score` of the first three actions and of the last three, and subtract.

| | n | mean Δ | median Δ | fraction decaying |
|---|---|---|---|---|
| successful episodes | 4 | **−0.091** | 0.000 | **25%** |
| failed episodes | 38 | **−0.267** | −0.288 | **84%** |

Pooled: `gate_score` p10 = 0.001, p50 = 0.346, p90 = 0.828, and it is **exactly 0.0 on 9.0% of all
actions**, while the action head's own `confidence` stays at p10 = 0.512, p50 = 0.803, p90 = 0.978.

`gate_score` also tracks whether the screen moved at all: `screen_changed` is 88.6% of actions with
`gate>0.0` but 79.9% with `gate>0.5` — so it is only weakly a "will this work" predictor, and that is
**not** its main use here. Its main use is the trend: it reports *drift*, and the drift is 3.4× more
common on failures.

**Caveat, stated now so it cannot be dropped later: n = 4 successes.** The direction is consistent
with the qualitative transcripts (the ClockTimerEntry trace above is literally this pattern), but 4
episodes cannot certify anything. This is a hypothesis with a mechanism, not a result.

## Where this leaves the two buckets

* **Premature `status`** (6 of 14 earlier, 18 status actions over 42 episodes): the model terminates
  at step 3–5 on tasks needing 8–15. `gate_score` is *high* on these — the model is confident and
  wrong, which is the worst combination for a selective gate because confidence will not filter them.
* **Never terminates** (24 of 42): `gate_score` decays to 0 by step 5 and the policy ignores it.

So the same gate score is **uninformative in one bucket and accurate in the other**, and the model has
no way to tell which bucket it is in. That is the concrete thing to build next — and it is exactly a
`task_complete` readout, the arm the 2026-11-07 pre-registration already named
(`benchmarks/results/2026-11-07-status-downweight-arm.md`).

---

## Update at 59 finished episodes: the successes are all one shape

**SR = 6/59 = 10.2%.** The three buckets, disjoint and exhaustive:

| bucket | n | median steps |
|---|---|---|
| **success** | 6 | 3–11 |
| **timeout, never terminates** | 31 | 25 |
| **premature `status`** | 22 | 7 |

**Every one of the six successes is a click-only single-app task:**

| task | steps | action mix |
|---|---|---|
| CameraTakePhoto | 3 | open_app 1, click 1, status 1 |
| OpenAppTaskEval | 3 | open_app 1, click 1, status 1 |
| RecipeDeleteSingleRecipe | 6 | open_app 1, click 4, status 1 |
| ExpenseDeleteSingle | 10 | open_app 1, click 8, status 1 |
| ExpenseDeleteMultiple | 11 | open_app 1, click 9, status 1 |
| RecipeDeleteSingleWithRecipeWithNoise | 10 | open_app 1, click 7, **type_text 1**, status 1 |

Four of six are literally *open the app, click the thing, say done*. Two are delete flows.

**Zero successes on any task that requires filling a text field and navigating.** The model emits
`type_text` in only **14 of 59 episodes (24%)**, and on those 14 it succeeds **1/14 (7.1%)** — the single
success is the one above that types once. Median clicks on failures is **22**, and **51% of failures emit
20 or more clicks** — it is grinding, not searching.

## What that means, said plainly

The model's competence is **one click after opening the right app**. That is not a weak version of a GUI
agent; it is a different, much smaller capability. Everything that makes AndroidWorld a benchmark —
multi-field forms, navigation, verification, recovery — is untouched. And the 6 successes are exactly the
tasks whose gold trajectory is within one click of `open_app`, which is also exactly what AC-v2 (95% click
action type) rewards.

So the honest sentence is: **AC-v2's 64% and AndroidWorld's 10% are not in tension — they are the same
model measured on the capability it has versus the capability the benchmark requires.** 95% of AC-v2 rows
are `click`; the model is good at `click` and at nothing else, and AndroidWorld is mostly not `click`-only.

## The three things this rules out

1. **Not a grounding failure.** Positions are plausible; the app opens correctly (35/59 first actions are
   `open_app` with the right app, confidence ≥0.99).
2. **Not the read slot.** `app_name` resolves on every goal that names or implies an app.
3. **Not the harness.** 128 tests pass, the action adapter round-trips, `ASR_exec` is 98%+, invalid rate 0.

What is missing is **composition**: stringing more than ~1 correct action together, and re-entering a
field it has already visited. Both are properties of the training distribution (single-step, no
trajectory, 8 actions) rather than of the weights.

---

## FINAL STATE (2026-11-09) — what the run produced, and what it could not

**104 of 116 tasks finished. 15 successes.** Three denominators, all stated: **18.8% (15/80) over scorable
episodes** — the only one that answers "how good where we could judge", 95% Wilson **11.7–28.7%** — 16.3%
(15/92) over genuinely attempted, and 14.4% (15/104) counting everything (Wilson 8.9–22.4%). Zero invalid
actions in 1,394 emitted.

| bucket | n | median steps |
|---|---|---|
| success | **15** | 3–11 |
| non-terminating timeout | **34** | 25 |
| premature `status` | **31** | 7 |
| `unscored` — ran, but the checker raised (`checker_unavailable`) | **12** | 4–25 |
| device never became usable (`setup_error`) | **12** | 0 |

The unscored row is the point of the next section: those 12 episodes are neither a model outcome nor a device
failure, and before the fix they were being reported as 6 timeouts and 6 premature terminations.

### Two defects, not one — and the second was scoring as a model failure

An auditor found a defect in this write-up that matters more than any number in it. The 104 finished episodes
split into **three** groups, not two:

| group | n | what it is |
|---|---|---|
| scorable | **80** | ran and were judged — **SR 18.8% (15/80)** |
| `unscored` / `checker_unavailable` | **12** | ran 4–25 steps, but `task.is_successful(env)` raised |
| `setup_error` | **12** | device never became usable (`env.reset()` could not get a tree) |

The 12 unscored episodes raised `AttributeError: 'AndroidWorldEnv' object has no attribute 'get_state'`,
because `awenv_bridge.AndroidWorldEnv` did not forward `get_state` to the wrapped environment while
AndroidWorld's own checkers call it. The runner recorded `success = False`, and since `timeout` and
`premature status` were derived from what the *agent* did, those episodes were reported as **6 timeouts and 6
premature terminations** — model failures no checker ever assessed. The bucket overlaps the setup failures on
only 11 rows and is otherwise disjoint, so this was a **second, independent defect**, and the earlier sentence
in this note that called it "one device defect, counted once" was wrong.

Both are fixed at the source: the wrapper now forwards `get_state` (and anything else it does not define), the
runner records `unscored` as its own bucket and excludes it from `timeout`, and `scripts/refresh_aw116.py`
reports all three denominators. The rate that answers "how good is it where we could judge" is **18.8%
(15/80)**; it appears in the paper, GOAL and HANDOFF.

**Why this is the more important finding than the SR.** A harness that silently converts an unscorable
episode into a model failure produces a number that looks conservative and is simply wrong — and the run had
already been quoted at 14.4% before anyone noticed 12 episodes had never been judged.

### The 12 setup failures are an infrastructure limit, and it is measured

They are not model failures and they are not a harness bug. The emulator's **accessibility forwarder degrades
after several hours of continuous use**: `dumpsys accessibility` still reports the service as *enabled and
bound*, `/health` still returns `success`, and `adb devices` still shows `device` — but `get_a11y_tree` raises,
so `env.reset()` cannot complete. Late in the session `/screenshot` began returning HTTP `000` (connection
timeout) as well.

Two restarts briefly restored it (18 elements on a fresh container; the retry that followed succeeded on
`SystemWifiTurnOffVerify`), and then it degraded again within ~20 episodes. **A 116-task suite needs roughly
2.5 hours of continuous emulator uptime and this container does not reliably provide it.** That is the honest
statement: the number is *incomplete*, not suppressed, and the incompleteness has a cause that is written down.

**Operationally, this is the thing a future run must fix:** either (a) restart the emulator every ~20 episodes
by chunking the suite into resumable segments, or (b) run on a host where the forwarder does not degrade. The
runner is already resumable and already skips finished episodes, and `scripts/refresh_aw116.py` recomputes
everything from whatever is on disk — so (a) is a scheduling change, not a code change.

### What did not change across the whole session

`0 / 1,394` invalid actions; 11 environment crashes, all of them setup failures; and — after the unscored
bucket became its own category — **34 non-terminating timeouts and 31 premature `status` out of the 80
scorable episodes**, plus 12 unscored and 12 setup failures. Every success remains a task whose gold
trajectory is one or two actions beyond `open_app`; `type_text` appears in 19 episodes and succeeds once; no
success involves a text field plus navigation. The conclusion is stable at every snapshot from 42 to 104
episodes.
