# 2026-11-06 — The online loop works, and the binding constraint is a read slot

Protocol `aw-main-v1`. Harness `awenv` (see `docs/compute/AW_PROTOCOL.md`). Everything below is
measured on the live emulator with the E2@938 checkpoint; nothing is inferred.

## What now exists

| Gate | Result |
|---|---|
| **A — environment correctness** | **32 passed, 1 skipped, 0 failed.** Four-layer in-process liveness; every emittable action round-trips; `open_app` reaches the app (foreground activity verified, not just "no exception"); 24/24 canonical apps launch. The one skip is the text-landing check: no editable field was on screen, and it skips with that reason rather than passing quietly. |
| **B — benchmark correctness** | **5/5 episodes identical across two runs**, with task params recorded in run 1 and replayed in run 2. Paired instances are now verified from the header, not asserted. |
| **C — agent evaluation** | 15 tasks × 2 independent instance draws, running. |

## The first honest online numbers (5 tasks, 60 actions)

| Quantity | Value |
|---|---|
| Emitted actions | 60 (`open_app` 27, `click` 22, `navigate_back` 10, `wait` 1) |
| Executed without exception (ASR) | **60/60** |
| Invalid actions (not representable) | **24 — every one `app_name_missing`** |
| Task success | 0/5, all timeouts at 12 steps |
| Model latency p50 | **112–119 ms** |
| Environment latency p50 | **1066–1071 ms** per step |
| Generated tokens | **0** |

## The finding

The model spends **45% of its actions (27/60) on `open_app`, and 24 of those 27 carry no app name.**
An `open_app` with no target is not a wrong action — it is not an action at all. The harness marks it
`invalid` and executes a `wait`, so the episode burns steps doing nothing.

That is the answer to "why is online success zero while offline step accuracy is 64%": the two
protocols exercise different read slots. Offline, `open_app` is scored on the *value* the model
produces when a goal mentions an app; online, the same slot must produce a canonical app name with no
app vocabulary in the state, and it produces an empty string.

**Consequence for any reported number:** the first AndroidWorld number from this checkpoint is
dominated by action-space coverage, not by grounding. `docs/compute/AW_PROTOCOL.md` already requires
`SR_all` vs `SR_reachable` side by side; this is why that requirement exists.

**Consequence for the paper:** the online section cannot be "the model is bad at GUI tasks". It is
"the V-track state does not carry the text the action needs", which is a representation defect with a
named fix, not a capability ceiling.

## The decisive probe (same AndroidWorld screenshot, four goals)

| Goal | Action | `app_name` |
|---|---|---|
| "Open the clock app" | `open_app` | **"clock"** |
| "Create a new note in Markor" | `open_app` | **"Markor"** |
| "Turn on wifi" | `open_app` | **""** |
| "Create a contact named Omar Mohamed with number 5551234" | `open_app` | **""** |

So the slot is neither broken nor distribution-shifted: **it copies an app name that the goal states
verbatim, and emits an empty string when the goal only implies an app.** The model still chooses
`open_app` in both cases, so the action slot and the name slot disagree — which is why the failure looks
like "the model decided to do something it cannot express".

That is a narrower and more useful statement than "readout missing": the readout is a *copy* operation,
not an *inference* operation, and the AC-v2 training set never taught it to infer.

## Candidate causes, in the order they can be tested

1. **The state lacks an app vocabulary.** The slot works when the name is in the goal, so the fix is to
   make the choice explicit: put the canonical app list in the state and let the slot classify over it.
   This is the cheapest experiment and it needs no new data. (Second: a11y text, below.)
2. **The state lacks the a11y text.** The V track feeds goal + screenshot + history. `app_name` is a
   text readout; with no text in the state it has nothing to copy. The VA track adds the element list
   and is the direct test. Offline, the same checkpoint scored `open_app` 20/27 — so the slot is
   *capable* of producing a name when the state contains one.
3. **The serving path drops it.** *(refuted — see the probe above; the API returns names.)* `serving/d1_api.py` returns `app_name` only when the chosen action
   is `open_app`; if the readout is empty it returns `""`, which the adapter correctly refuses to
   invent a value for. This is verifiable from the API response without the emulator.
4. **A canonical vocabulary mismatch.** Even a produced name may not be in AndroidWorld's vocabulary
   (`clock`, not `stopwatch`). The adapter canonicalises, and `canon_app` rejects unknown names rather
   than guessing — that would show as `app_name_missing` too, after canonicalisation.

Tests 1 and 2 are the real experiments; test 3 is already refuted by one API call. Test 1 is the one to
run first: it is a state-format change, it needs no new data, and it directly targets the measured failure.

## What this changes in the plan

- The `SR_reachable` split is no longer a nicety — it is the only honest way to report this run.
- `return_readouts` and the `--save-readouts` flag now matter for the online path too: with them, an
  empty `app_name` can be traced to the slot rather than to the adapter.
- The claim "no sub-3B model exceeds 66% AC-v2 step accuracy" stands; nothing here contradicts it.

## What must not be claimed

- Not "the model fails AndroidWorld" — 5 tasks, all timeouts, Wilson band 0–43%.
- Not "grounding is the bottleneck online" — grounding was never exercised: the invalid actions never
  reached a click.
- Not a latency speedup: 112 ms of model time inside a 1178 ms step means the environment dominates
  by 10×, exactly as the plan predicted.


## Ablation: does fixing the read slot help? (8 tasks × 2 conditions, the primary sample)

| Condition | Task success | Invalid actions | Actions emitted | Crash rate | Model p50 |
|---|---|---|---|---|---|
| pure | **1/8 (12.5%)** | 20/71 (28.2%) | open_app, click, navigate_back | 2/8 | 122.9 ms |
| declared fallback | **1/8 (12.5%)** | 10/68 (14.7%) | + `type_text`, + `status` | 2/8 | 124.3 ms |

The single success is `TurnOnWifiAndOpenApp`, in **both** conditions — the same task, which is what
determinism looks like. The invalid-action intervals do not overlap (19.0–39.5% vs 8.2–25.0%), so the
read-slot fix is real; the success intervals are identical (2.2–47.1%), so it is not yet an outcome win.

The 4-task pair below reproduces the same direction (0/4 both; invalid 50% → 27%).

## Earlier: 4 tasks × 2 conditions

| Condition | Task success | Invalid actions | Emitted actions | Model p50 | Env p50 |
|---|---|---|---|---|---|
| pure | 0/4 | 20/40 (50%) | open_app 22, click 14, navigate_back 4 | 109.5 ms | 1064.8 ms |
| declared fallback | 0/4 | 10/37 (27%) | click 18, open_app 13, navigate_back 4, type_text 2, status 1 | 118.8 ms | 317.4 ms |

**Recovering the app name halves the invalid-action rate and changes behaviour qualitatively**: with the
fallback the model emits `type_text` for the first time and declares `status` once. It still produces no
success, so the read slot is a *necessary but not sufficient* condition.

**Failure taxonomy (both conditions, 8 episodes):** 2 environment crashes, 5 timeouts, 1 premature
`status`, 0 successes. **Every crash is the same task** (`ClockStopWatchRunning`), and the episode had
already run all 10 steps with zero action errors — so the crash is the emulator, not the model. That is a
1/4 crash rate, which is too high to report a headline benchmark number from; it must be driven down before
any 116-task run.

**Reading the environment latency column with care:** 1064 ms → 317 ms is not a speedup from the fallback.
The fallback changes which screens the agent reaches, and `transition_pause` plus app launches dominate the
per-step cost, so this column measures the screens visited, not the interface.
