# The 25% "environment crash" rate was a checker-classification bug

*2026-11-07. Evidence: `backup-2026-11-06/gate_runs/aw_runs_20261106/` (pure8, fallback8), local screenshots.*

## Finding

The 2026-11-06 online run reported **2/8 environment crashes (25%) in both conditions**, and that number was
treated as the binding infrastructure problem. It is not a device-crash rate. Both episodes are the two
**Clock** tasks, and in both the environment was alive.

## Evidence

| Check | Result |
|---|---|
| Crashed episodes | `ClockStopWatchRunning`, `ClockTimerEntry` — in both conditions |
| Last recorded step | `error: null`, `action_ok: true`, `env_crash_candidate` absent |
| Step budget | 10/10 steps executed; no step raised |
| Result record | `env_crash: true`, `checker_unavailable: true`, `emulator_restarts: 0` |
| Device after the crash | the **next** episode in the same suite ran 10 steps and finished (pure8: crash 20:40:51Z → `SystemWifiTurnOn` ends 20:43:08Z; crash 20:45:30Z → `MarkorAddNoteHeader` ends 20:46:17Z) |
| Harness stderr | `WARNING:absl:Could not get a11y tree, retrying.` repeatedly |
| Last screenshot | an **actively ticking** stopwatch (1:18.80, 7 laps) and a timer set to **20 h 00 m 85 s** |

The only place `env_crash` can be set after the step loop is the success check:

```python
try:
    success = bool(task.is_successful(env))
except Exception:
    success, env_crash = False, True      # ← the bug
```

AndroidWorld's Clock checkers read the running stopwatch/timer through the accessibility tree. A ticking
screen changes continuously, uiautomator's dump times out, the accessor retries and then raises. The runner
caught that and labelled it an environment crash — without ever asking whether the environment was alive.

## Fix

`benchmarks/aw/awenv/runner.py`: a raising checker now records `checker_error`, and `env_crash` is set only
if the liveness probe also fails:

```python
except Exception as exc:
    checker_error = "%s: %s" % (type(exc).__name__, exc)
    env_crash = not _alive(env)
```

`checker_unavailable` is still true (we could not score), but `env_crash` is now an environment claim, not a
checker claim. Two tests added (`test_a_raising_checker_on_a_live_device_is_not_an_environment_crash`,
`..._on_a_dead_device_is_still_a_crash`); suite: **110 passed, 35 skipped**.

## What changes in the paper

- The 25% crash rate is withdrawn. The corrected statement is **0 device deaths in 16 episodes**; 2/8 per
  condition hit a **task-checker exception** on a live device.
- The real Clock failure is behavioural and more interesting: the model drove the Clock into a valid but
  wrong terminal state (a 20 h timer; a running stopwatch), then ran out of steps. That is a task-progress
  failure, not an emulator failure.
- The 4-task pair's "2 environment crashes" is the same bug counted once per condition (one Clock task).

## What this does not fix

The a11y dump still fails on rapidly animating Clock screens. Until the checker can read that state (or the
harness scores it from a state snapshot), Clock tasks will remain unscorable — which is different from
"crashed" and must be reported as `checker_unavailable`, with its own denominator. `SR_all` and
`SR_reachable` must both carry this count.
