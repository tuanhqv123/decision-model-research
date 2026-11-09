# 2026-11-08 — The "missing 6 actions" was wrong; the real defect is one line of state

*Everything below is measured on the live emulator + live server, not inferred.*

## What I claimed, and what was true

| Claim | Verdict |
|---|---|
| "The model emits only 8 of AW's 14 actions, so we cannot run the benchmark" | **Wrong in effect.** 5 of the 6 "missing" actions are not model decisions at all. |
| "We need AMEX (93 GB) / AITW (700 GB) to get those labels" | **Wrong.** No dataset is needed. |
| "The read slot is broken / the model never saw 18 of 24 AW apps" | **Wrong.** The read slot is a **pointer copy** and emits app names it has never seen. |

## Evidence 1 — the AW task suite does not need 5 of the 6 actions

Grepped all 87 files under `android_world/task_evals/`, and every one of the 116 task classes:

| action | tasks that need it |
|---|---|
| `double_tap` | **0** |
| `swipe` | **0** |
| `keyboard_enter` | **0** |
| `navigate_home` | **0** |
| `infeasible` / `TASK_IMPOSSIBLE` | **0** |
| `answer` | **2** (`BrowserMultiply`, `RetroPlaylistDuration`) |

And the reasons are in the code, not in an opinion:

* `actuation.py:107-108` — an `input_text` action **presses Enter itself**:
  `adb_utils.type_text(...)` then `adb_utils.press_enter_button(env)`. The model never needs to emit Enter.
* `interface.py:233` — `reset(go_home=True)` presses Home at episode start, so Home is the harness's job.
* `actuation.py:169` — `elif action.action_type == 'swipe':  # Inverse of scroll.` Same gesture, reversed
  direction. `scroll` covers it.

So the model's 8 options cover **114/116 tasks**. The honest coverage number is 114/116, not 8/14.

## Evidence 2 — the failure was reproduced, and it is `app_name`

The 2026-11-06 eight-task run reports `SR 1/8` in both conditions. Reading the episode JSONLs, every step of
`ContactsAddContact` and `SystemWifiTurnOn` looks like this:

```
action_raw      : {"action": "open_app", "app_name": "", "confidence": 0.92}
action_executed : {"action_type": "wait"}
invalid_action  : true
reason          : "app_name_missing"
```

The action was **right** (`open_app`); the app name was **empty**, so the adapter refused to invent one and
the episode never left its first screen. Both failing tasks fail this way for all 11 steps.

## Evidence 3 — the read slot works; the state had nothing to copy

Asked the live server (E2@938) for goals that **name** the app:

| goal | emitted |
|---|---|
| "Create a new note in **Markor** …" | `open_app app_name="Markor"` (0.805) |
| "…the **Joplin** app…" | `open_app app_name="Joplin"` (0.787) |
| "…the **Broccoli** app…" | `open_app app_name="Broccoli"` (0.963) |

Markor, Joplin and Broccoli exist in **no AndroidControl training label** (only 6 of AW's 24 apps do:
`clock, contacts, dialer, files, settings, tasks`). The read slot is a span copy, so it transcribes whatever
string is in the state — it does not need to have memorised the app.

Asked the same server for goals that **imply** an app without naming it:

| goal | emitted before the fix |
|---|---|
| "Turn wifi on." | `click` (0.79) |
| "Create a new contact for Jack Ahmed." | `click` (0.67) |
| "Run the stopwatch." | `click` (0.49) |
| "Send a text saying hello." | `click` (0.49) |
| "Add an expense of 12 dollars." | `click` (0.54) |

Measured on the 116-task suite: **69 tasks name their app in the goal, 22 do not.** The defect is those 22.

## The fix — one line of state, identical at train and serve

`gold_answers()` used to drop the `app_name` supervision whenever the app was not literally in the state
(`if g["app_name"] is None: del g["app_name"]`), so the model was never told what to read. It then copied the
nearest goal noun ("wifi", "contact", "photo"), which is not an app.

The fix is not to retrain the read slot. It is to give it something to copy:

```
Goal: Turn wifi on
Previous actions: none
Application for this task: settings        <- added, one line, both paths
```

* `ac_common.APP_INTENT` — a table from intent words to AW's canonical app names.
* `ac_common.app_intent_for(goal)` — resolves the app, preferring one the goal already names.
* `build_state(..., app_intent=...)` — renders the line.
* `train_d1.py` and `serving/d1_api.py` both call it, so the string cannot drift between paths.

## Measured result of the fix

Same server, same checkpoint family, only the state line added:

| goal | before | after |
|---|---|---|
| Turn wifi on. | `click` | **`open_app` → settings** |
| Create a new contact for Jack Ahmed. | `click` | **`open_app` → contacts** |
| Take one photo. | `click` | **`open_app` → camera** |
| Run the stopwatch. | `click` | **`open_app` → clock** |
| Send a text saying hello. | `click` | **`open_app` → simple sms messenger** |
| Add an expense of 12 dollars. | `click` | **`open_app` → pro expense** |
| Create a note in Markor. | `open_app` → Markor | `open_app` → Markor (unchanged) |
| Play music. | `click` | **`open_app` → retro music** |

**1/8 → 8/8**, through the served HTTP path.

## The trap I fell into, recorded so it is not repeated

Training **only** on the 1,380 synthetic rows destroys the model:

| run | data | steps | AC-v2 step acc |
|---|---|---|---|
| E2 baseline | cur15k | 938 | **64.07%** |
| `appint_smoke` | app_intent only | 30 | **24.37%** ← catastrophic forgetting |
| `hint_only` | app_intent only | 60 | **26.88%** ← same |
| `appint_mix60` | cur15k + app_intent | 60 | 59.55% |
| `appint_v2` | cur15k + app_intent | 120 | **63.07%** (grounding 68.23%, up from 56.05) |

A single-class dataset teaches "always `open_app`" and the model forgets everything else. The mixer
(`--train-set cur15k,app_intent`) is mandatory. This is the same lesson as the earlier `aux7k` ablation,
learned again the hard way.

## What is still open

* The full 938-step mixed run is in flight; whether it lands at or above 64.07% is not yet known.
* 2/116 tasks (`BrowserMultiply`, `RetroPlaylistDuration`) need `answer`, which is not in the option set.
  They will fail, and that is 1.7% stated up front rather than discovered later.
* `app_intent_for` is a keyword table. It is right on the 8 probes above and it is the same table the paper
  must disclose as **scaffolding**, not as a learned capability.
